"""Does merging the LoRA adapter change decoded output? (Phase 2b model)

Three arms on the same test subsample with the same decoding settings:
  A  merged checkpoint, bf16 weights (the form used for reported evaluations)
  B  unmerged adapter on a float32 base
  C  unmerged adapter on a bf16 base

Comparisons:
  A - C  effect of the merge, both with bf16 base weights
  B - C  effect of base precision, both unmerged

Generation settings are written to the base model's config, because generate()
on a PeftModel reads the base model's config, not the wrapper's.

The generation cap is MAX_NEW_TOKENS, default 448 (the decoder limit). Earlier
runs used 225, which truncated long utterances; set MAX_NEW_TOKENS=225 only to
reproduce those numbers.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found; aborting.")

# Paths (from config)
PHASE1_MERGED = cfg.require(cfg.PHASE1_MERGED)
PHASE2B_DIR = cfg.DATA_DIR / "phase2b_leyu_standard"
ADAPTER_DIR = cfg.require(PHASE2B_DIR / "models" / "phase2b_lora_adapter")
MERGED_DIR = cfg.require(PHASE2B_DIR / "models" / "phase2b_final_merged")
TEST_CACHE = cfg.require(cfg.LEYU_SI_CACHE)
RESULTS_DIR = PHASE2B_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Settings
DIAG_SUBSET_SIZE = int(os.environ.get("DIAG_SUBSET_SIZE", 300))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", cfg.MAX_LABEL_LENGTH))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))

processor = WhisperProcessor.from_pretrained(str(PHASE1_MERGED), language="amharic", task="transcribe")
DECODER_START = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
FORCED_IDS = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

test = load_from_disk(str(TEST_CACHE))["test"]
subset = test.shuffle(seed=cfg.SEED).select(range(min(DIAG_SUBSET_SIZE, len(test))))
print(f"diagnostic subsample: {len(subset):,} of {len(test):,} test utterances")

WER = evaluate.load("wer")
CER = evaluate.load("cer")


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


collator = Collator(processor=processor, decoder_start_token_id=DECODER_START)


def decode(ids):
    ids = ids.copy()
    ids[ids == -100] = processor.tokenizer.pad_token_id
    return [normalize_amharic(s) for s in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]


class PredictOnlyTrainer(Seq2SeqTrainer):
    """Used only for batched generation on the test subset."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        features = inputs.pop("input_features")
        inputs.clear()
        out = model(input_features=features, labels=labels)
        return (out.loss, out) if return_outputs else out.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for key in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                    "attention_mask", "dialect", "path", "duration_sec"):
            inputs.pop(key, None)
        inputs["input_features"] = inputs["input_features"].to(next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir=str(RESULTS_DIR / "_scratch"),
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=MAX_NEW_TOKENS,
    bf16=True,
    fp16=False,
    report_to=["none"],
    remove_unused_columns=False,
)


def set_decoding(model):
    """Write the decoding settings to the object generate() reads.

    For a PeftModel, generate() delegates to the base model, so assigning
    model.generation_config on the wrapper has no effect.
    """
    target = model.get_base_model() if isinstance(model, PeftModel) else model
    gc = target.generation_config
    gc.forced_decoder_ids = FORCED_IDS
    gc.max_new_tokens = MAX_NEW_TOKENS
    gc.num_beams = 5
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.use_cache = True


def evaluate_arm(model, label):
    set_decoding(model)
    trainer = PredictOnlyTrainer(model=model, args=eval_args, data_collator=collator)
    out = trainer.predict(subset)
    hyp = decode(out.predictions)
    ref = decode(out.label_ids)
    wer = WER.compute(predictions=hyp, references=ref) * 100
    cer = CER.compute(predictions=hyp, references=ref) * 100
    print(f"[{label}] WER={wer:.2f}%  CER={cer:.2f}%")
    for i in range(min(2, len(hyp))):
        print(f"    ref: {ref[i][:120]}")
        print(f"    hyp: {hyp[i][:120]}")
    return {"wer": round(wer, 2), "cer": round(cer, 2)}


def load_merged():
    return WhisperForConditionalGeneration.from_pretrained(
        str(MERGED_DIR), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()


def load_unmerged(dtype):
    base = WhisperForConditionalGeneration.from_pretrained(
        str(PHASE1_MERGED), torch_dtype=dtype, attn_implementation="sdpa")
    return PeftModel.from_pretrained(base, str(ADAPTER_DIR)).to("cuda").eval()


ARMS = {
    "A_merged_bf16": load_merged,
    "B_adapter_fp32_base": lambda: load_unmerged(torch.float32),
    "C_adapter_bf16_base": lambda: load_unmerged(torch.bfloat16),
}

results = {}
for label, load in ARMS.items():
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    model = load()
    results[label] = evaluate_arm(model, label)
    del model
    torch.cuda.empty_cache()

a, b, c = (results[k]["wer"] for k in ARMS)
print(f"\nmerge effect, A - C (both bf16 base): {a - c:+.2f} points")
print(f"base precision, B - C (unmerged):     {b - c:+.2f} points")

out_path = RESULTS_DIR / f"merge_precision_diagnostic_max{MAX_NEW_TOKENS}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"subset": len(subset), "max_new_tokens": MAX_NEW_TOKENS, "results": results},
              f, indent=2, ensure_ascii=False)
print(f"saved {out_path}")