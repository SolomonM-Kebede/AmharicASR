"""Evaluate a full fine-tuned Whisper model on the Leyu test set.

One script covers both splits and both inference precisions:
  --split joint   Joint test set (speaker- and text-independent)
  --split si      speaker-independent test set (needs the SI cache rebuilt)
  --dtype bf16    bfloat16 inference, used for the C4 and C4b bf16 results
  --dtype fp32    float32 inference, used for the float32 comparison

Settings: beam 5, no repetition penalty, 448-token generation limit. A lower
--max-new-tokens reproduces the earlier capped decoding. Capped outputs are named
with the cap, so they cannot replace an uncapped file.

Writes results/persample/<name>_persample.jsonl (read by run_bootstrap.py) and
results/persample/<name>.json. Existing files are kept unless --overwrite is given.
Run from the workspace root after sourcing env.amh.sh.
run ex:C4
python -u scripts/eval/eval_full_ft.py --split joint --dtype bf16 --name fullft_rerun

"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import io
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import jiwer
import librosa
import torch
from datasets import Audio, load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED_TEST = 4588
REPORTED_LIMIT = cfg.MAX_LABEL_LENGTH  # 448
DEFAULT_MODEL = cfg.DATA_DIR / "phase_fullft_joint_indep" / "models" / "fullft_joint_indep_final"  # C4
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL,
                    help="merged full fine-tuned model directory (default: C4)")
    ap.add_argument("--split", choices=["joint", "si"], required=True)
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--max-new-tokens", type=int, default=REPORTED_LIMIT)
    ap.add_argument("--name", help="output prefix (default: fullft_<split>_<dtype>)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if not 0 < args.max_new_tokens <= REPORTED_LIMIT:
        ap.error(f"--max-new-tokens must be between 1 and {REPORTED_LIMIT}")
    return args


def load_test(split):
    cache = cfg.require(cfg.LEYU_JOINT_CACHE if split == "joint" else cfg.LEYU_SI_CACHE)
    test = load_from_disk(str(cache))["test"]
    if len(test) != EXPECTED_TEST:
        raise SystemExit(f"{split} test split has {len(test)} utterances, expected {EXPECTED_TEST}")
    text_col = "norm_text" if "norm_text" in test.column_names else "text"
    print(f"{split} test split: {len(test):,} utterances (text column: {text_col})")
    return test, text_col


def prepare(batch, processor, text_col):
    audio = batch["audio"]
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    wav, _ = librosa.load(src, sr=16_000, mono=True)
    batch["input_features"] = processor.feature_extractor(wav, sampling_rate=16_000).input_features[0]
    text = normalize_amharic(batch[text_col] or "")
    batch["labels"] = processor.tokenizer(text, truncation=True, max_length=REPORTED_LIMIT).input_ids
    return batch


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all().item():
            ids = ids[:, 1:]
        audio["labels"] = ids
        return audio


class PredictOnlyTrainer(Seq2SeqTrainer):
    """Batched generation only; the loss is never used."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for key in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                    "attention_mask", "dialect", "orig_idx"):
            inputs.pop(key, None)
        inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def corpus_wer_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0, 0.0
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def main():
    args = parse_args()
    model_dir = cfg.require(args.model_dir)

    stem = args.name or f"fullft_{args.split}_{args.dtype}"
    if args.max_new_tokens != REPORTED_LIMIT:
        stem += f"_cap{args.max_new_tokens}"
    persample_path = cfg.PERSAMPLE / f"{stem}_persample.jsonl"
    summary_path = cfg.PERSAMPLE / f"{stem}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite or choose --name")

    test, text_col = load_test(args.split)
    test = test.add_column("orig_idx", list(range(len(test))))
    test = test.cast_column("audio", Audio(decode=False))

    processor = WhisperProcessor.from_pretrained(str(model_dir), language="amharic", task="transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=DTYPES[args.dtype], attn_implementation="sdpa").to("cuda").eval()
    print(f"model: {model_dir}\nweights in memory: {model.dtype}  inference: {args.dtype}")
    gc = model.generation_config
    gc.forced_decoder_ids = forced_ids
    gc.num_beams = 5
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.use_cache = True
    gc.max_new_tokens = None
    gc.max_length = args.max_new_tokens

    test = test.map(partial(prepare, processor=processor, text_col=text_col),
                    remove_columns=[c for c in test.column_names if c not in ("dialect", "orig_idx")])
    dialects = test["dialect"]

    trainer = PredictOnlyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch" / stem),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            generation_max_length=args.max_new_tokens,
            bf16=True, fp16=False,
            report_to=["none"],
            remove_unused_columns=False,
        ),
        data_collator=Collator(processor, decoder_start),
    )

    pad = processor.tokenizer.pad_token_id
    summary, all_refs, all_hyps = {}, [], []
    with open(persample_path, "w", encoding="utf-8") as out:
        for dialect in DIALECTS:
            idx = [i for i, d in enumerate(dialects) if d == dialect]
            if not idx:
                continue
            subset = test.select(idx)
            pred = trainer.predict(subset)
            pred_ids = pred.predictions.copy()
            pred_ids[pred_ids == -100] = pad
            label_ids = pred.label_ids.copy()
            label_ids[label_ids == -100] = pad
            hyps = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
            refs = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

            for orig, r, h in zip(subset["orig_idx"], refs, hyps):
                out.write(json.dumps({"orig_idx": int(orig), "dialect": dialect,
                                      "reference": r, "prediction": h}, ensure_ascii=False) + "\n")

            wer, cer = corpus_wer_cer(refs, hyps)
            summary[dialect] = {"n": len(subset), "wer": round(wer, 2), "cer": round(cer, 2)}
            print(f"{dialect:12s} n={len(subset):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")
            all_refs += refs
            all_hyps += hyps

    if len(all_refs) != EXPECTED_TEST:
        raise SystemExit(f"scored {len(all_refs)} utterances, expected {EXPECTED_TEST}")
    wer, cer = corpus_wer_cer(all_refs, all_hyps)
    longest = max(len(processor.tokenizer(h, add_special_tokens=False).input_ids) for h in all_hyps)
    summary["OVERALL"] = {"n": len(all_refs), "wer": round(wer, 2), "cer": round(cer, 2),
                          "split": args.split, "dtype": args.dtype,
                          "max_new_tokens": args.max_new_tokens, "longest_prediction_tokens": longest,
                          "model_dir": str(model_dir)}
    print(f"{'OVERALL':12s} n={len(all_refs):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%  longest={longest} tokens")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()