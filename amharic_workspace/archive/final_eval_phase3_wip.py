"""
Amharic ASR — Phase 2 FINAL evaluation, corrected decode config
=================================================================
Diagnostic (diagnose_decode_config.py) showed repetition_penalty=1.2 +
no_repeat_ngram_size=7 causes major degradation on its own (greedy WER
57.40% -> 69.73% on the same subsample, with visibly corrupted output) —
not a beam-search interaction. This re-runs the FULL held-out test set,
per dialect, with beam=5 and NO repetition penalty / no-repeat-ngram —
this is the number to report as your Phase 2 result.
"""

import os

CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

BASE_DIR   = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
# Point directly to your brand new Phase 3 model!
PHASE3_MODEL  = os.path.join(BASE_DIR, "phase3_full_finetune/models/phase3_full_finetuned_final")
# 2. Point the cache to the exact same Leyu Phase 2 test set for a fair comparison
DATASET_VERSION = os.environ.get("DATASET_VERSION", "2026-08-expanded-spkindep")
LEYU_CACHE      = os.path.join(BASE_DIR, f"phase2_leyu/dataset_cache_{DATASET_VERSION}")

# 3. Save the results in the Phase 2b folder so we don't overwrite Phase 2 results
RESULTS_DIR     = os.path.join(BASE_DIR, "phase2b_leyu_standard/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 1))
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))

DIALECT_CONFIGS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    #("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    #("leyu-amharic/leyu-amharic-wello-dialect",  "wello"),
    #("leyu-amharic/leyu-amharic-shewa-dialect",  "shewa"),
    #("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]

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

print(f"\nLoading FULLY FINE-TUNED Phase 3 model from {PHASE3_MODEL} ...")
model = WhisperForConditionalGeneration.from_pretrained(
    PHASE3_MODEL, 
    torch_dtype=torch.float32
).to("cuda").eval()
# from peft import PeftModel

# print(f"\nLoading BASE model in strict FP32 from {PHASE1_MERGED} ...")
# # Force FP32 so the base weights don't get truncated!
# base_model = WhisperForConditionalGeneration.from_pretrained(
#     PHASE1_MERGED, 
#     torch_dtype=torch.float32
# )

# print(f"Loading LoRA adapter from {ADAPTER_DIR} ...")
# peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# print("Merging weights in-memory (FP32, safe_merge=True) ...")
# # Let Hugging Face do the math in 32-bit floating point
# merged_model = peft_model.merge_and_unload(safe_merge=True)

# # Move the finalized, cleanly merged model to the GPU
# model = merged_model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)

## CORRECTED: We must update the existing config IN-PLACE so we don't accidentally
# delete Whisper's built-in suppress_tokens rules which prevent timestamp hallucinations.
model.generation_config.forced_decoder_ids     = forced_ids
model.generation_config.max_new_tokens         = 225
model.generation_config.num_beams              = NUM_BEAMS
model.generation_config.use_cache              = True
model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id

# Explicitly ensure these are disabled (independent of beam search)
model.generation_config.repetition_penalty     = 1.0
model.generation_config.no_repeat_ngram_size   = 0

print(f"\nLoading cached test split from {LEYU_CACHE} ...")
cached = load_from_disk(LEYU_CACHE)
test_dataset = cached["test"]
print(f"  full test set: {len(test_dataset):,} examples")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]}              for f in features]

        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch["labels"] = lids
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=DECODER_START_TOKEN_ID,
)


def compute_metrics(pred):
    import evaluate as _eval
    _wer = _eval.load("wer")
    _cer = _eval.load("cer")

    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    wer = _wer.compute(predictions=pred_str, references=label_str) * 100
    cer = _cer.compute(predictions=pred_str, references=label_str) * 100
    return {"wer": round(wer, 2), "cer": round(cer, 2)}


class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels         = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()
        outputs = model(input_features=input_features, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        inputs.pop("decoder_input_ids",     None)
        inputs.pop("decoder_inputs_embeds", None)
        inputs.pop("input_ids",             None)
        inputs.pop("attention_mask",        None)
        inputs.pop("dialect",               None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch_final_fixed",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    generation_max_length      = 225,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

trainer = SafeWhisperTrainer(
    model           = model,
    args            = eval_args,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

dialect_results = {}

for _, dialect_name in DIALECT_CONFIGS:
    dialect_test = test_dataset.filter(lambda x: x["dialect"] == dialect_name)
    if len(dialect_test) == 0:
        print(f"  {dialect_name}: no test samples found, skipping")
        continue

    result = trainer.predict(dialect_test)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(result.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

    dialect_results[dialect_name] = {
        "n_samples": len(dialect_test),
        "wer":       round(wer, 2),
        "cer":       round(cer, 2),
    }
    print(f"  {dialect_name:15s}  n={len(dialect_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%")

print(f"\n{'='*60}")
print(f"  PHASE 2 FINAL RESULTS — CORRECTED DECODE CONFIG (beam={NUM_BEAMS}, no penalty)")
print(f"{'='*60}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect, metrics in dialect_results.items():
    marker = " ←" if dialect == "OVERALL" else ""
    print(f"  {dialect:<18} {metrics['n_samples']:>7}  "
          f"{metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%{marker}")
print(f"{'='*60}")
print(f"  Previous (buggy decode config) OVERALL WER : 64.76%")
print(f"  Phase 1 (standard Amharic) best WER        : 27.69%")
print(f"  Previous Phase 2 (smaller corpus) WER       : 64.57%")
print(f"  Google STT baseline WER                     : 35.00%")
print(f"{'='*60}")

results_path = os.path.join(RESULTS_DIR, f"phase2_dialect_results_{DATASET_VERSION}_fixed_decode.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")