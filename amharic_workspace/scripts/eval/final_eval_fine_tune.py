"""
Standalone Phase 3 Final Evaluation
Forces strict offline mode to prevent Hugging Face network deadlocks.
"""
import os

# 1. FORCE STRICT OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

# 2. CACHE DIRECTORIES
CUSTOM_CACHE = os.environ.get("STT_CACHE_DIR", os.path.expanduser("~/AmharicASR/clean_cache"))
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE

import json
import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

print(f"GPU Available: {torch.cuda.is_available()}", flush=True)

# 3. PATHS
BASE_DIR = "~/AmharicASR/amharic_workspace"
PHASE3_MODEL = os.path.join(BASE_DIR, "phase3_full_finetune/models/phase3_full_finetuned_final")
LEYU_ONLY_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
RESULTS_DIR = os.path.join(BASE_DIR, "phase3_full_finetune/results")

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

print(f"Loading processor and model from {PHASE3_MODEL}...", flush=True)
processor = WhisperProcessor.from_pretrained(PHASE3_MODEL)
model = WhisperForConditionalGeneration.from_pretrained(PHASE3_MODEL).to("cuda").eval()

# Beam search configuration
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_new_tokens = 225
model.generation_config.num_beams = 5
model.generation_config.use_cache = True
model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
model.generation_config.repetition_penalty = 1.0
model.generation_config.no_repeat_ngram_size = 0

print(f"Loading Test Dataset from {LEYU_ONLY_CACHE}...", flush=True)
test_dataset = load_from_disk(LEYU_ONLY_CACHE)["test"]

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# NO COMPUTE METRICS HERE - Prevents double evaluation hang
eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch",
    per_device_eval_batch_size=1, # Bypasses the padding bug
    predict_with_generate=True,
    bf16=True,
    report_to=["none"],
    remove_unused_columns=False,
)

from transformers import default_data_collator
# Custom collator for generation
def data_collator(features):
    input_features = [{"input_features": f["input_features"]} for f in features]
    label_features = [{"input_ids": f["labels"]} for f in features]
    batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
    labels = processor.tokenizer.pad(label_features, return_tensors="pt")
    lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
    batch["labels"] = lids
    return batch

trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator)

DIALECT_NAMES = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
dialect_results = {}
all_pred, all_ref = [], []

import evaluate as _eval
wer_metric = _eval.load("wer")
cer_metric = _eval.load("cer")

print("\nStarting Evaluation...", flush=True)

for d_name in DIALECT_NAMES:
    d_test = test_dataset.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0: continue
    
    print(f"\nRunning generation for {d_name} (n={len(d_test)})...", flush=True)
    result = trainer.predict(d_test)
    
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
    cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100
    
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"--> {d_name.upper()} RESULTS: WER={wer:.2f}% | CER={cer:.2f}%", flush=True)
    
    all_pred.extend(pred_str)
    all_ref.extend(label_str)

overall_wer = wer_metric.compute(predictions=all_pred, references=all_ref) * 100
overall_cer = cer_metric.compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(flush=True)
print(f"  PHASE 3 FINAL RESULTS (FULL FINE-TUNING)", flush=True)
print(flush=True)
for d, m in dialect_results.items():
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%", flush=True)
print(flush=True)

with open(os.path.join(RESULTS_DIR, "phase3_final_offline.json"), "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)