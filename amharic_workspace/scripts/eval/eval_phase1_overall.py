from data.normalize import normalize_amharic
"""
Amharic ASR — Phase 1 OVERALL-only completion
============================================================
Use this if the full eval_phase1.py run got killed by the
walltime limit after finishing all 5 per-dialect passes but before the
redundant full-set OVERALL pass. This does ONLY the missing overall pass
(beam=5, no penalty) and merges it into the existing per-dialect results
JSON — it does NOT redo gojjam/gonder/wello/shewa/addis ababa.
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
    GenerationConfig,
)

CUSTOM_CACHE = "~/AmharicASR/clean_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

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

EXISTING_RESULTS_PATH = os.path.join(RESULTS_DIR, "phase1_dialect_results_beam5.json")

# The completed Phase 1 dialect metrics extracted from the timed-out log
KNOWN_DIALECT_RESULTS = {
    "gojjam":       {"n_samples": 654,  "wer": 110.30, "cer": 35.16},
    "gonder":       {"n_samples": 1237, "wer": 109.95, "cer": 34.92},
    "wello":        {"n_samples": 836,  "wer": 103.04, "cer": 35.33},
    "shewa":        {"n_samples": 1101, "wer": 93.73,  "cer": 26.41},
    "addis ababa":  {"n_samples": 760,  "wer": 99.94,  "cer": 25.77},
}

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

# H200 Optimizations retained (bfloat16 + native SDPA flash attention)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
model = model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)
# Corrected Config: max_length used to stop warnings, no penalties
model.generation_config = GenerationConfig(
    forced_decoder_ids     = forced_ids,
    max_length             = 225,
    num_beams              = NUM_BEAMS,
    use_cache              = True,
    decoder_start_token_id = model.config.decoder_start_token_id,
)

print(f"\nLoading manifest from {MANIFEST_PATH} ...")
entries = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))

test_dataset = Dataset.from_list(entries)
print(f"  full test set: {len(test_dataset):,} examples")

def prepare_dataset(batch):
    audio_arr, _ = librosa.load(batch["path"], sr=16_000, mono=True)
    batch["input_features"] = processor.feature_extractor(
        audio_arr, sampling_rate=16_000, return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(
        normalize_amharic(batch["reference"]), truncation=True, max_length=225
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
        # Safely catch extra manifest columns to prevent crashes
        for key in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(key, None)
            
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch_phase1_overall",
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

print("\nRunning the OVERALL pass only (full test set, beam=5, no penalty) ...")
overall_result = trainer.predict(test_dataset)
label_ids = overall_result.label_ids.copy()
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

pred_str  = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(overall_result.predictions, skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

import evaluate as _eval
overall_wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
overall_cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

print(f"\nOVERALL  n={len(test_dataset):,}  WER={overall_wer:.2f}%  CER={overall_cer:.2f}%")

if Path(EXISTING_RESULTS_PATH).exists():
    with open(EXISTING_RESULTS_PATH, "r", encoding="utf-8") as f:
        dialect_results = json.load(f)
    print(f"\nLoaded existing per-dialect results from {EXISTING_RESULTS_PATH}")
else:
    print(f"\n{EXISTING_RESULTS_PATH} not found on disk (expected, since the killed "
          f"job never reached its save step) — using the known per-dialect numbers "
          f"from the log instead.")
    dialect_results = dict(KNOWN_DIALECT_RESULTS)

dialect_results["OVERALL"] = {
    "n_samples": len(test_dataset),
    "wer":       round(overall_wer, 2),
    "cer":       round(overall_cer, 2),
}

print(f"\n{'='*60}")
print(f"  PHASE 1 FINAL RESULTS — (beam={NUM_BEAMS}, no penalty)")
print(f"{'='*60}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect, metrics in dialect_results.items():
    marker = " ←" if dialect == "OVERALL" else ""
    print(f"  {dialect:<18} {metrics['n_samples']:>7}  "
          f"{metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%{marker}")
print(f"{'='*60}")

with open(EXISTING_RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nUpdated results (with OVERALL) saved to {EXISTING_RESULTS_PATH}")