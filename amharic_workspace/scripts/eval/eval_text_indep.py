"""Test-set evaluation of the text-independent LoRA run (B1).

The text-independent split guarantees that no test text occurs in training.
It does not hold out speakers: all 273 test speakers also appear in training,
so B1 is not a speaker-independent result.

Scores the 4,894-utterance test set of phase2_leyu/dataset_cache_text_independent.
Writes results/persample/<name>_persample.jsonl and <name>.json. Settings:
bfloat16 inference, beam 5, 448-token limit, no repetition penalty.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import io
import json
import argparse
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
EXPECTED_TEST = 4894
NUM_BEAMS = 5
DEFAULT_MODEL = cfg.DATA_DIR / "phase2_leyu" / "models" / "text_indep_lora_run" / "final_merged"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--name", default="b1", help="prefix for the output files")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def prepare(batch, processor):
    audio = batch["audio"]
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    wav, _ = librosa.load(src, sr=16_000, mono=True)
    batch["input_features"] = processor.feature_extractor(wav, sampling_rate=16_000).input_features[0]
    batch["labels"] = processor.tokenizer(
        normalize_amharic(batch["norm_text"]), truncation=True, max_length=cfg.MAX_LABEL_LENGTH).input_ids
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
        inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
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
    persample_path = cfg.PERSAMPLE / f"{args.name}_persample.jsonl"
    summary_path = cfg.PERSAMPLE / f"{args.name}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite, or choose another --name")

    test = load_from_disk(str(cfg.require(cfg.TI_CACHE)))["test"]
    if len(test) != EXPECTED_TEST:
        raise SystemExit(f"text-independent test split has {len(test)} utterances, expected {EXPECTED_TEST}")
    if "norm_text" not in test.column_names:
        raise SystemExit(f"test split has no norm_text column: {test.column_names}")
    test = test.add_column("orig_idx", list(range(len(test))))
    test = test.cast_column("audio", Audio(decode=False))

    processor = WhisperProcessor.from_pretrained(str(model_dir), language="amharic", task="transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    gc = model.generation_config
    gc.forced_decoder_ids = forced_ids
    gc.num_beams = NUM_BEAMS
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.max_length = cfg.MAX_LABEL_LENGTH
    gc.max_new_tokens = None
    gc.use_cache = True

    test = test.map(partial(prepare, processor=processor),
                remove_columns=[c for c in test.column_names if c not in ("dialect", "orig_idx")])
    dialects = test["dialect"]

    trainer = PredictOnlyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch" / args.name),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            generation_max_length=cfg.MAX_LABEL_LENGTH,
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
    summary["OVERALL"] = {"n": len(all_refs), "wer": round(wer, 2), "cer": round(cer, 2),
                          "model_dir": str(model_dir), "beams": NUM_BEAMS}
    print(f"{'OVERALL':12s} n={len(all_refs):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()