"""Test-set evaluation of a full fine-tuned Whisper model on the Joint split.

Writes one per-utterance file (orig_idx, dialect, reference, prediction) and one
summary file. The per-utterance file is what run_bootstrap.py reads.

Defaults reproduce C4: full fine-tuning, FP16 stored weights, bfloat16 inference,
beam 5. Output goes to results/persample/fullft_joint_indep_TEST_persample.jsonl.
For another model, pass --model-dir and a --name that describes that model.
Existing outputs are not replaced unless --overwrite is given.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
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

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED_TEST = 4588
DEFAULT_MODEL = cfg.DATA_DIR / "phase_fullft_joint_indep" / "models" / "fullft_joint_indep_final"
DEFAULT_NAME = "fullft_joint_indep_TEST"


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
    """Batched generation only; the loss is never used."""

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for key in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                    "attention_mask", "dialect", "orig_idx"):
            inputs.pop(key, None)
        inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


def corpus_wer_cer(refs, hyps):
    """Corpus WER and CER over utterances with a non-empty reference."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0, 0.0
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--name", default=DEFAULT_NAME, help="prefix for the output files")
    ap.add_argument("--beams", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = cfg.require(args.model_dir)
    persample_path = cfg.PERSAMPLE / f"{args.name}_persample.jsonl"
    summary_path = cfg.RESULTS / f"{args.name}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it, or choose another --name")

    processor = WhisperProcessor.from_pretrained(str(model_dir), language="amharic", task="transcribe")
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    gc = model.generation_config
    gc.forced_decoder_ids = forced_ids
    gc.max_length = cfg.MAX_LABEL_LENGTH
    gc.max_new_tokens = None
    gc.num_beams = args.beams
    gc.use_cache = True

    test = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"]
    if len(test) != EXPECTED_TEST:
        raise SystemExit(f"test split has {len(test)} utterances, expected {EXPECTED_TEST}")
    test = test.add_column("orig_idx", list(range(len(test))))
    all_dialects = test["dialect"]

    trainer = PredictOnlyTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch" / args.name),
            per_device_eval_batch_size=args.batch_size,
            predict_with_generate=True,
            bf16=True, fp16=False,
            dataloader_num_workers=4,
            report_to=["none"],
            remove_unused_columns=False,
        ),
        data_collator=Collator(processor, decoder_start),
    )

    summary, all_refs, all_hyps = {}, [], []
    with open(persample_path, "w", encoding="utf-8") as out:
        for dialect in DIALECTS:
            idx = [i for i, d in enumerate(all_dialects) if d == dialect]
            if not idx:
                continue
            subset = test.select(idx)
            pred = trainer.predict(subset)
            pred_ids = pred.predictions.copy()
            pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
            label_ids = pred.label_ids.copy()
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            hyps = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
            refs = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

            for orig, r, h in zip(subset["orig_idx"], refs, hyps):
                out.write(json.dumps({"orig_idx": int(orig), "dialect": dialect,
                                      "reference": r, "prediction": h}, ensure_ascii=False) + "\n")

            wer, cer = corpus_wer_cer(refs, hyps)
            runaway = sum(1 for r, h in zip(refs, hyps) if r and len(h) > 2 * len(r))
            summary[dialect] = {"n": len(subset), "wer": round(wer, 2), "cer": round(cer, 2),
                                "hyp_over_2x_ref_chars": runaway}
            print(f"{dialect:12s} n={len(subset):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%  runaway={runaway}")
            all_refs += refs
            all_hyps += hyps

    wer, cer = corpus_wer_cer(all_refs, all_hyps)
    summary["OVERALL"] = {"n": len(all_refs), "wer": round(wer, 2), "cer": round(cer, 2),
                          "model_dir": str(model_dir), "beams": args.beams}
    print(f"{'OVERALL':12s} n={len(all_refs):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()