from data.normalize import normalize_amharic
"""
Amharic ASR — Phase 3 (Full Fine-Tune) TRUE Evaluation
========================================================
Bypasses the `generation_max_length=225` bug present in the original training script.
Evaluates the saved Whisper Large-v3 FFT model using the correct 448 max_length.
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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from common import config as cfg  # must load before transformers/datasets
import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# Paths
BASE_DIR   = "~/AmharicASR/amharic_workspace"
MODEL_DIR  = os.path.join(BASE_DIR, "phase3_full_finetune/models/phase3_full_finetuned_final")
LEYU_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
RESULTS_DIR= os.path.join(BASE_DIR, "phase3_fft/results")

os.makedirs(RESULTS_DIR, exist_ok=True)
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8)) 
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))
DIALECT_NAMES   = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]

RESULTS_PATH = os.path.join(RESULTS_DIR, "phase3_fft_TRUE_fixed.json")

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

print(f"\nLoading processor/model from {MODEL_DIR} ...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float32)
model = model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

# TRUE FIX: In-place mutation, allowing up to 448 tokens
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = NUM_BEAMS
model.generation_config.use_cache = True

print(f"\nLoading cached test split from {LEYU_CACHE} ...")
cached = load_from_disk(LEYU_CACHE)
test_dataset = cached["test"]

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
    processor=processor, decoder_start_token_id=DECODER_START_TOKEN_ID
)

class SafeWhisperTrainer(Seq2SeqTrainer):
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
    output_dir                 = "/tmp/eval_scratch_phase3",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    bf16                       = False, # Kept FP32 for stability
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator)

dialect_results = {}
all_pred, all_ref = [], []

for d_name in DIALECT_NAMES:
    d_test = test_dataset.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0:
        continue
    print(f"\nRunning {d_name} (n={len(d_test)}) ...")
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

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(dialect_results, f, ensure_ascii=False, indent=2)

overall_wer = _eval.load("wer").compute(predictions=all_pred, references=all_ref) * 100
overall_cer = _eval.load("cer").compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}")
print(f"  PHASE 3 (FULL FINE-TUNE) — TRUE FINAL RESULTS")
print(f"{'='*60}")
for d, m in dialect_results.items():
    marker = " ←" if d == "OVERALL" else ""
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%{marker}")
print(f"{'='*60}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")