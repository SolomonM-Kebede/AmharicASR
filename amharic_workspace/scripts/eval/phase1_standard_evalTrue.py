from data.normalize import normalize_amharic
"""
Amharic ASR — Phase 1 (standard dialect) TRUE evaluation
============================================================
In-place generation_config mutation, max_length=448, forced Amharic language
explicitly re-applied (Phase 1's saved checkpoint has an unforced/auto-detect
language slot — confirmed earlier in this investigation), no
generation_max_length override in TrainingArguments.
"""

import os
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from common import config as cfg  # must load before transformers/datasets
import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)

BASE_DIR   = "~/AmharicASR/amharic_workspace"
MODEL_DIR  = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
TEST_CACHE = os.path.join(BASE_DIR, "phase1_curriculum/dataset_cache_rebuilt")
RESULTS_DIR = os.path.join(BASE_DIR, "phase1_curriculum/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", 5))

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text):
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

print(f"Loading processor/model from {MODEL_DIR} ...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).to("cuda").eval()
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = NUM_BEAMS
model.generation_config.use_cache = True

print(f"Loading rebuilt test split from {TEST_CACHE} ...")
test_dataset = load_from_disk(TEST_CACHE)["test"]
print(f"  test: {len(test_dataset):,} examples")

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

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_phase1_true",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    bf16=False, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)
trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator)

result = trainer.predict(test_dataset)
label_ids = result.label_ids.copy(); label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
pred_ids  = result.predictions.copy(); pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

import evaluate as _eval
wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

print(f"\nPHASE 1 (standard dialect) — TRUE FINAL RESULT")
print(f"  n={len(pred_str)}  WER={wer:.2f}%  CER={cer:.2f}%")

results_path = os.path.join(RESULTS_DIR, "phase1_TRUE_fixed.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump({"n_samples": len(pred_str), "wer": round(wer, 2), "cer": round(cer, 2)}, f, indent=2)
print(f"Saved to {results_path}")