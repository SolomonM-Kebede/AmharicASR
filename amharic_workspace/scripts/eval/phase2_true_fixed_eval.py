"""Phase 2 (Leyu-only) evaluation of the merged LoRA model on the SI test split.

This is the record of the run that produced the 16.01% WER reported for A2.
Settings, as they were in that run:
  - Model: phase2_leyu/models/phase2_final_merged (LoRA merged).
  - Weights: float32, as loaded (torch_dtype=float32).
  - Trainer: bf16=True, so predictions were computed under bf16 autocast.
    The label for this result is "float32 weights, bf16 autocast".
  - Decoding: beam 5, max_length 448. No generation_max_length is set on the
    trainer, so the model's generation_config governs. An earlier version set
    225 on the trainer, which overrode it; that version is not reported.
  - Test set: 4,588 utterances from the SI test split.

The SI training cache is not retained, so this script cannot be rerun from the
repository. Its saved summary and log are in results/summaries/.

Writes results/persample/<name>_persample.jsonl and <name>.json. Existing files
are kept unless --overwrite is given. Run from the workspace root after sourcing
env.amh.sh.
"""
from common import config as cfg  # sets cache and offline flags; keep first

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED_TEST = 4588
NUM_BEAMS = 5
DEFAULT_MODEL = cfg.DATA_DIR / "phase2_leyu" / "models" / "phase2_final_merged"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--name", default="a2_float32", help="output prefix")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
    """Batched generation only. Casts inputs to the model's dtype, as in the original run."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raise RuntimeError("loss is not used in this evaluation")

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for key in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                    "attention_mask", "dialect", "orig_idx"):
            inputs.pop(key, None)
        inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def main():
    args = parse_args()
    model_dir = cfg.require(args.model_dir)
    persample_path = cfg.PERSAMPLE / f"{args.name}_persample.jsonl"
    summary_path = cfg.PERSAMPLE / f"{args.name}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite or choose --name")

    test = load_from_disk(str(cfg.require(cfg.LEYU_SI_CACHE)))["test"]
    if len(test) != EXPECTED_TEST:
        raise SystemExit(f"SI test split has {len(test)} utterances, expected {EXPECTED_TEST}")
    test = test.add_column("orig_idx", list(range(len(test))))

    processor = WhisperProcessor.from_pretrained(str(model_dir), language="amharic", task="transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    # Weights are float32 as loaded. bf16=True below makes the trainer run the
    # forward pass under bf16 autocast, so the reported label must say so.
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32).to("cuda").eval()
    gc = model.generation_config
    gc.forced_decoder_ids = forced_ids
    gc.max_length = cfg.MAX_LABEL_LENGTH
    gc.max_new_tokens = None
    gc.num_beams = NUM_BEAMS
    gc.use_cache = True

    trainer = PredictOnlyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch" / args.name),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            # generation_max_length deliberately not set: the model's max_length governs.
            bf16=True, fp16=False,
            report_to=["none"],
            remove_unused_columns=False,
        ),
        data_collator=Collator(processor, decoder_start),
    )

    wer_metric, cer_metric = evaluate.load("wer"), evaluate.load("cer")
    dialects = test["dialect"]
    summary, all_refs, all_hyps = {}, [], []
    pad = processor.tokenizer.pad_token_id

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

            wer = wer_metric.compute(predictions=hyps, references=refs) * 100
            cer = cer_metric.compute(predictions=hyps, references=refs) * 100
            summary[dialect] = {"n": len(subset), "wer": round(wer, 2), "cer": round(cer, 2)}
            print(f"{dialect:12s} n={len(subset):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")
            all_refs += refs
            all_hyps += hyps
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if len(all_refs) != EXPECTED_TEST:
        raise SystemExit(f"scored {len(all_refs)} utterances, expected {EXPECTED_TEST}")
    wer = wer_metric.compute(predictions=all_hyps, references=all_refs) * 100
    cer = cer_metric.compute(predictions=all_hyps, references=all_refs) * 100
    summary["OVERALL"] = {
        "n": len(all_refs), "wer": round(wer, 2), "cer": round(cer, 2),
        "weights": "float32 as loaded", "inference": "bf16 autocast (bf16=True)",
        "beams": NUM_BEAMS, "max_length": cfg.MAX_LABEL_LENGTH,
        "model_dir": str(model_dir), "test_split": "SI test split",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{'OVERALL':12s} n={len(all_refs):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()