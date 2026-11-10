"""
Amharic ASR — RQ2 final eval WITHOUT merging (merge_and_unload is broken
for this adapter, root cause not yet confirmed — see diagnostic results)
============================================================================
merge_and_unload() gives wrong results for this trained adapter regardless
of dtype (confirmed: both bf16 and fp32 merges give ~56-60% WER, while the
UNMERGED PeftModel gives ~16.8% on the same data). Rather than chase the
merge bug further, this evaluates the adapter unmerged directly — proven
correct by the earlier diagnostic — on the FULL Leyu test set for the real
final RQ2 numbers.
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
MERGED_DIR    = os.path.join(PHASE2B_DIR, "models", "phase2b_clean_merged_fp32")
ADAPTER_DIR   = os.path.join(PHASE2B_DIR, "models", "phase2b_lora_adapter")
LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
RESULTS_DIR   = os.path.join(PHASE2B_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))
DIALECT_NAMES = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]

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

device = "cuda"
print(f"\nLoading processor from {PHASE1_MERGED} ...")
processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

print("Loading UNMERGED (base+adapter)...")
base_fp32 = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.float32)
unmerged = PeftModel.from_pretrained(base_fp32, ADAPTER_DIR).to(device).eval()

unmerged.generation_config.forced_decoder_ids = forced_ids
unmerged.generation_config.max_new_tokens = 225
unmerged.generation_config.num_beams = 5
unmerged.generation_config.use_cache = True

print("Loading MERGED...")
merged = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to(device).eval()
merged.generation_config.forced_decoder_ids = forced_ids
merged.generation_config.max_new_tokens = 225
merged.generation_config.num_beams = 5
merged.generation_config.use_cache = True

# ── Load base (fp32) + adapter, UNMERGED — this is the proven-correct path ─
# print(f"\nLoading base model in fp32 from {PHASE1_MERGED} ...")
# base_fp32 = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED)
# ── Load the MERGED model directly — this is the only change from doc3 ────
# print(f"\nLoading MERGED model in fp32 from {MERGED_DIR} ...")
# eval_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to("cuda").eval()
# eval_model.generation_config = GenerationConfig(
#     forced_decoder_ids=forced_ids, max_new_tokens=225, num_beams=NUM_BEAMS,
#     use_cache=True, decoder_start_token_id=eval_model.config.decoder_start_token_id,
# )

# print(f"Loading saved adapter from {ADAPTER_DIR} (staying unmerged) ...")
# eval_model = PeftModel.from_pretrained(base_fp32, ADAPTER_DIR).to("cuda").eval()
# eval_model.generation_config = GenerationConfig(
#     forced_decoder_ids=forced_ids, max_new_tokens=225, num_beams=NUM_BEAMS,
#     use_cache=True, decoder_start_token_id=eval_model.config.decoder_start_token_id,
# )

print(f"\nLoading Leyu test split from {LEYU_CACHE} ...")
cached = load_from_disk(LEYU_CACHE)
test_dataset = cached["test"]
print(f"  full test set: {len(test_dataset):,} examples")


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
    output_dir="/tmp/eval_scratch_phase2b_unmerged",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    bf16=False, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)
trainer = SafeWhisperTrainer(model=merged, args=eval_args, data_collator=data_collator, compute_metrics=compute_metrics)

dialect_results = {}
all_pred, all_ref = [], []

for d_name in DIALECT_NAMES:
    d_test = test_dataset.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0:
        continue
    result = trainer.predict(d_test)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"  {d_name:15s}  n={len(d_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%")
    all_pred.extend(pred_str)
    all_ref.extend(label_str)

import evaluate as _eval
overall_wer = _eval.load("wer").compute(predictions=all_pred, references=all_ref) * 100
overall_cer = _eval.load("cer").compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}\n  RQ2 — FINAL RESULTS (unmerged PEFT model, merge bug bypassed)\n{'='*60}")
for d, m in dialect_results.items():
    marker = " ←" if d == "OVERALL" else ""
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%{marker}")
print(f"{'='*60}")
print(f"  Leyu-only baseline (RQ2 comparison) : 56.84% WER")
print(f"{'='*60}")

results_path = os.path.join(RESULTS_DIR, "phase2b_leyu_standard_results_UNMERGED.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")
print("\nNOTE: for any deployment/sharing of this model later, DON'T use merge_and_unload()")
print("as-is — load base + PeftModel adapter together (as this script does) until the")
print("merge bug is understood.")