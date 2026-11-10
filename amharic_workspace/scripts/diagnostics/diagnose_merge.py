"""
Amharic ASR — Diagnose: is the merge step degrading the model?
==================================================================
In-training eval showed ~16% WER (live PEFT-wrapped model, evaluated
directly, no merge). Final eval on the MERGED model shows ~55-60% WER,
barely different from the Leyu-only baseline. That's a huge, unexplained
gap given both use the same clean decode config (beam=5, no penalty).

Leading hypothesis: the training script loads the base model with
torch_dtype=torch.bfloat16 before merge_and_unload() — merging LoRA
weights (base + lora_A @ lora_B * scaling) in bf16 (~8 bits mantissa)
instead of fp32 (23 bits) could introduce real precision loss for a
rank-64 adapter trained over 15,750 steps. Your original successful
Leyu-only run merged in default fp32.

This script evaluates THREE versions on a quick test subsample, no
retraining needed:
  A. The saved MERGED model, loaded in bf16 (matches what final eval used)
  B. The saved LoRA ADAPTER applied UNMERGED on top of an fp32 base model
     (matches what in-training eval effectively evaluated)
  C. The saved LoRA ADAPTER applied unmerged on top of a bf16 base model
     (isolates: is it the merge math itself, or just bf16 base weights?)

If B/C are close to the in-training ~16-20% ballpark and A stays near
55-60%, that confirms the merge precision is the culprit.
"""

import os

CUSTOM_CACHE = os.environ.get("STT_CACHE_DIR", os.path.expanduser("~/AmharicASR/clean_cache"))
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig,
)
from peft import PeftModel

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

BASE_DIR      = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
PHASE2B_DIR   = os.path.join(BASE_DIR, "phase2b_leyu_standard")
ADAPTER_DIR   = os.path.join(PHASE2B_DIR, "models", "phase2b_lora_adapter")
MERGED_DIR    = os.path.join(PHASE2B_DIR, "models", "phase2b_final_merged")
LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")

DIAG_SUBSET_SIZE = int(os.environ.get("DIAG_SUBSET_SIZE", 300))
EVAL_BATCH_SIZE  = int(os.environ.get("EVAL_BATCH_SIZE", 8))
SEED = 42

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


print(f"\nLoading processor from {PHASE1_MERGED} ...")
processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

print(f"\nLoading Leyu test split from {LEYU_CACHE} ...")
cached = load_from_disk(LEYU_CACHE)
test_dataset = cached["test"]
diag_test = test_dataset.shuffle(seed=SEED).select(range(min(DIAG_SUBSET_SIZE, len(test_dataset))))
print(f"  diagnostic subsample: {len(diag_test):,} examples")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch["labels"] = lids
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=DECODER_START_TOKEN_ID)


def compute_metrics(pred):
    import evaluate as _eval
    _wer, _cer = _eval.load("wer"), _eval.load("cer")
    pred_ids = pred.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    return {"wer": round(_wer.compute(predictions=pred_str, references=label_str) * 100, 2),
            "cer": round(_cer.compute(predictions=pred_str, references=label_str) * 100, 2)}


class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_diag_merge",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    bf16=True, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)

results = {}

def run_eval(model, label):
    model.generation_config = GenerationConfig(
        forced_decoder_ids=forced_ids, max_new_tokens=225, num_beams=5,
        use_cache=True, decoder_start_token_id=model.config.decoder_start_token_id,
    )
    trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator, compute_metrics=compute_metrics)
    result = trainer.predict(diag_test)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100
    print(f"\n[{label}] WER={wer:.2f}%  CER={cer:.2f}%")
    print("  Sample predictions:")
    for i in range(min(2, len(pred_str))):
        print(f"    ref: {label_str[i][:120]}")
        print(f"    hyp: {pred_str[i][:120]}")
    results[label] = {"wer": round(wer, 2), "cer": round(cer, 2)}
    del trainer, model
    torch.cuda.empty_cache()


# ── A: the saved MERGED model, bf16 (what final eval actually used) ────────
print(f"\n{'='*70}\nA: MERGED model (bf16) — what final eval used\n{'='*70}")
model_a = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
run_eval(model_a, "A_merged_bf16")

# ── B: LoRA adapter UNMERGED on top of fp32 base (matches in-training eval) ─
print(f"\n{'='*70}\nB: PEFT adapter unmerged, fp32 base — matches in-training eval setup\n{'='*70}")
base_fp32 = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED)
model_b = PeftModel.from_pretrained(base_fp32, ADAPTER_DIR).to("cuda").eval()
run_eval(model_b, "B_adapter_unmerged_fp32base")

# ── C: LoRA adapter UNMERGED on top of bf16 base — isolates merge vs base dtype ─
print(f"\n{'='*70}\nC: PEFT adapter unmerged, bf16 base — isolates merge-math vs base-dtype\n{'='*70}")
base_bf16 = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model_c = PeftModel.from_pretrained(base_bf16, ADAPTER_DIR).to("cuda").eval()
run_eval(model_c, "C_adapter_unmerged_bf16base")

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
for label, r in results.items():
    print(f"  {label:35s}  WER={r['wer']:6.2f}%  CER={r['cer']:6.2f}%")

out_path = os.path.join(PHASE2B_DIR, "results", "merge_precision_diagnostic.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")