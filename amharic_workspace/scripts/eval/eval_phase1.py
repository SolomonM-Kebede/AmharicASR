from data.normalize import normalize_amharic
"""
Amharic ASR — Phase 1 (ALFFA + CV) Evaluation 
=================================================================
Evaluates the Phase 1 merged Whisper-large-v3 checkpoint using the
exact per-dialect iterative loop architecture.
(beam=5, no penalty, aligned decode config).
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from common import config as cfg  # must load before transformers/datasets
import torch
import librosa
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

CUSTOM_CACHE = "~/AmharicASR/clean_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
for env_var in [
    "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE", 
    "HF_MODULES_CACHE", "HF_METRICS_CACHE", "HF_EVALUATE_CACHE"
]:
    os.environ[env_var] = CUSTOM_CACHE

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# Phase 1 Specific Paths
BASE_DIR      = "~/AmharicASR/amharic_workspace"
MANIFEST_PATH = os.path.join(BASE_DIR, "test_set_audio_export/manifest.jsonl")
RESULTS_DIR   = os.path.join(BASE_DIR, "phase2_leyu/results")
MODEL_DIR     = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")

if not Path(MODEL_DIR).exists():
    raise FileNotFoundError(f"Merged model not found at {MODEL_DIR}")

os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]

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

model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
model = model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)

# True Fix applied inline
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = NUM_BEAMS
model.generation_config.use_cache = True

print(f"\nLoading manifest from {MANIFEST_PATH} ...")
entries = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))
print(f"  full test set: {len(entries):,} examples")

test_dataset = Dataset.from_list(entries)

def prepare_dataset(batch):
    audio_arr, _ = librosa.load(batch["path"], sr=16_000, mono=True)
    batch["input_features"] = processor.feature_extractor(
        audio_arr, sampling_rate=16_000, return_tensors="pt"
    ).input_features[0]
    
    # FIXED: max_length=448 to match Whisper's absolute maximum positional context length
    batch["labels"] = processor.tokenizer(
        normalize_amharic(batch["reference"]), truncation=True, max_length=448
    ).input_ids
    return batch

print("Mapping audio files to log-Mel features...", flush=True)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=["reference"])


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


class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        for key in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", 
                    "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(key, None)
        
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch_phase1_fixed",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

# No compute_metrics passed here; we will decode manually in the loop.
trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator)

import evaluate as _eval
wer_metric = _eval.load("wer")
cer_metric = _eval.load("cer")

dialect_results = {}
all_pred, all_ref = [], []
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase1_dialect_results_beam5.json")

print("\nRunning dialect-by-dialect evaluation...")
for dialect_name in DIALECTS:
    # Safely match dialect names ignoring casing
    dialect_test = test_dataset.filter(lambda x: str(x.get("dialect", "")).strip().lower() == dialect_name.lower())
    
    if len(dialect_test) == 0:
        print(f"  {dialect_name:15s}  n=   0  [Skipped]")
        continue
    
    print(f"Running {dialect_name} (n={len(dialect_test)})...")
    result = trainer.predict(dialect_test)
    
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
    cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100

    dialect_results[dialect_name] = {"n_samples": len(dialect_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"  {dialect_name:15s}  n={len(dialect_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%")

    all_pred.extend(pred_str)
    all_ref.extend(label_str)

    # Save incrementally so you don't lose progress
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(dialect_results, f, ensure_ascii=False, indent=2)

# --- 2. OVERALL Inference Pass (Calculated instantly, no extra GPU time) ---
print("\nCalculating OVERALL evaluation...")
overall_wer = wer_metric.compute(predictions=all_pred, references=all_ref) * 100
overall_cer = cer_metric.compute(predictions=all_pred, references=all_ref) * 100

dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

# --- 3. Final Print & Save ---
print(f"\n{'='*60}")
print(f"  PHASE 1 FINAL RESULTS (beam={NUM_BEAMS}, max_length=448)")
print(f"{'='*60}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect, metrics in dialect_results.items():
    marker = " ←" if dialect == "OVERALL" else ""
    print(f"  {dialect:<18} {metrics['n_samples']:>7}  "
          f"{metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%{marker}")
print(f"{'='*60}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nPhase 1 results successfully saved to {RESULTS_PATH}")