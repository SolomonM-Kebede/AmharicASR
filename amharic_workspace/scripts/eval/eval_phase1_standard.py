"""Phase 1 on its own held-out Standard-Amharic test set (1,087 utterances).

This produces the 25.91% WER (7.51% CER) reported for Phase 1 on Standard Amharic.
The test set is ALFFA-only: the earlier provenance check found no Common Voice
sentence in it. The cache has only input_features and labels, so references are
decoded from the labels, not read from a text column.

Settings: float32 inference throughout (the checkpoint stores float16 weights,
which are upcast on load; bf16=False, so no autocast), beam 5, 448-token limit,
no repetition penalty.

Writes results/persample/<name>_persample.jsonl and <name>.json. Existing files
are kept unless --overwrite is given. Run from the workspace root after sourcing
env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import jiwer
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic

EXPECTED_TEST = 1087
NUM_BEAMS = 5
REQUIRED_COLUMNS = {"input_features", "labels"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="phase1_standard", help="output prefix")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


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
                    "attention_mask", "orig_idx"):
            inputs.pop(key, None)
        inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def corpus_wer_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def main():
    args = parse_args()
    model_dir = cfg.require(cfg.PHASE1_MERGED)
    persample_path = cfg.PERSAMPLE / f"{args.name}_persample.jsonl"
    summary_path = cfg.PERSAMPLE / f"{args.name}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite or choose --name")

    test = load_from_disk(str(cfg.require(cfg.PHASE1_STD_TEST)))["test"]
    if len(test) != EXPECTED_TEST:
        raise SystemExit(f"test split has {len(test)} utterances, expected {EXPECTED_TEST}")
    missing = REQUIRED_COLUMNS - set(test.column_names)
    if missing:
        raise SystemExit(f"test split lacks columns {sorted(missing)}")
    test = test.add_column("orig_idx", list(range(len(test))))

    processor = WhisperProcessor.from_pretrained(str(model_dir), language="amharic", task="transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, attn_implementation="sdpa").to("cuda").eval()
    print(f"model: {model_dir}\nweights in memory: {model.dtype}  inference: float32, no autocast")
    gc = model.generation_config
    gc.forced_decoder_ids = forced_ids
    gc.num_beams = NUM_BEAMS
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.use_cache = True
    gc.max_new_tokens = None
    gc.max_length = cfg.MAX_LABEL_LENGTH

    trainer = PredictOnlyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch" / args.name),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            generation_max_length=cfg.MAX_LABEL_LENGTH,
            bf16=False, fp16=False,
            report_to=["none"],
            remove_unused_columns=False,
        ),
        data_collator=Collator(processor, decoder_start),
    )

    pred = trainer.predict(test)
    pad = processor.tokenizer.pad_token_id
    pred_ids = pred.predictions.copy()
    pred_ids[pred_ids == -100] = pad
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = pad
    hyps = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    refs = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    with open(persample_path, "w", encoding="utf-8") as out:
        for i, (r, h) in enumerate(zip(refs, hyps)):
            out.write(json.dumps({"orig_idx": i, "dialect": "standard",
                                  "reference": r, "prediction": h}, ensure_ascii=False) + "\n")

    wer, cer = corpus_wer_cer(refs, hyps)
    longest = max(len(processor.tokenizer(h, add_special_tokens=False).input_ids) for h in hyps)
    summary = {"n": len(refs), "wer": round(wer, 2), "cer": round(cer, 2),
               "empty_references_skipped": sum(1 for r in refs if not r),
               "inference": "float32, no autocast", "beams": NUM_BEAMS,
               "max_length": cfg.MAX_LABEL_LENGTH, "longest_prediction_tokens": longest,
               "model_dir": str(model_dir), "test_cache": str(cfg.PHASE1_STD_TEST)}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"n={len(refs)}  WER={wer:.2f}%  CER={cer:.2f}%  longest={longest} tokens")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()