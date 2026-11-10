"""
Amharic ASR — Joint Composition Ablation: Final Evaluation
Evaluates either arm (raw or phase1) on the shared joint-independent test set,
beam=5, forced decoding, max_length=448.
Generates per-sample JSONL for paired bootstrap statistical testing.
"""

import os
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import librosa
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

EVAL_MODEL_SOURCE = os.environ.get("EVAL_MODEL_SOURCE", "raw")
assert EVAL_MODEL_SOURCE in ("raw", "phase1"), "EVAL_MODEL_SOURCE must be 'raw' or 'phase1'"

BASE_DIR     = "~/AmharicASR/amharic_workspace"
MERGED_DIR   = os.path.join(BASE_DIR, f"joint_composition_ablation/models/{EVAL_MODEL_SOURCE}_joint_lora_run/final_merged")
DATASET_DIR  = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")
RESULTS_PATH = os.path.join(BASE_DIR, f"joint_composition_ablation/results_{EVAL_MODEL_SOURCE}.json")
PERSAMPLE_PATH = os.path.join(BASE_DIR, f"joint_composition_ablation/results_{EVAL_MODEL_SOURCE}_persample.jsonl")

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'
def normalize_amharic(text):
    if text is None: return ""
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

# ---------------------------------------------------------
# 1. DATASET FIRST
# ---------------------------------------------------------
print(f"EVAL_MODEL_SOURCE = {EVAL_MODEL_SOURCE}")
print(f"Loading processor from {MERGED_DIR}...")
processor = WhisperProcessor.from_pretrained(MERGED_DIR)

print("Loading TEST dataset...")
raw_ds = load_from_disk(DATASET_DIR)
raw_test = raw_ds["test"]
# ADDITION 1: Add orig_idx to track exactly which utterance is which for bootstrap
raw_test = raw_test.add_column("orig_idx", list(range(len(raw_test))))
raw_test = raw_test.cast_column("audio", Audio(decode=False))

def prepare_dataset(batch):
    audio = batch["audio"]
    if audio.get("bytes") is not None:
        arr, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=16_000, mono=True)
    elif audio.get("path") is not None:
        arr, _ = librosa.load(audio["path"], sr=16_000, mono=True)
    else:
        arr = audio["array"]
    batch["input_features"] = processor.feature_extractor(arr, sampling_rate=16_000).input_features[0]
    raw_text = batch.get("norm_text") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(normalize_amharic(str(raw_text)), max_length=448, truncation=True).input_ids
    return batch

cols_to_remove = [c for c in raw_test.column_names if c not in ["dialect", "orig_idx"]]
print(f"Extracting log-Mel features for {len(raw_test)} test samples...")
processed_test = raw_test.map(prepare_dataset, remove_columns=cols_to_remove, num_proc=4)

# ADDITION 2: Pre-compute dialect indices to bypass filter() disk I/O bottleneck
all_dialects = processed_test["dialect"]

# ---------------------------------------------------------
# 2. MODEL
# ---------------------------------------------------------
print(f"Loading model from {MERGED_DIR}...")
model = WhisperForConditionalGeneration.from_pretrained(
    MERGED_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda")

forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.num_beams = 5
model.generation_config.repetition_penalty = 1.0
model.generation_config.no_repeat_ngram_size = 0
model.config.use_cache = True

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch["labels"] = lids
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for key in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "orig_idx"]:
            inputs.pop(key, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

eval_args = Seq2SeqTrainingArguments(
    output_dir=f"/tmp/eval_scratch_{EVAL_MODEL_SOURCE}",
    per_device_eval_batch_size=16, # Optimized batch size
    dataloader_num_workers=4,      # Optimized workers
    predict_with_generate=True,
    bf16=True,
    report_to=["none"],
    remove_unused_columns=False,
)
eval_trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator)

print("\nRunning per-dialect evaluation on the full test set...")
dialect_results = {}
unique_dialects = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
all_pred, all_ref = [], []

import evaluate as _eval
wer_metric, cer_metric = _eval.load("wer"), _eval.load("cer")

# ADDITION 3: Open the JSONL file to record the predictions
persample_file = open(PERSAMPLE_PATH, "w", encoding="utf-8")

for dialect in unique_dialects:
    # OPTIMIZED selection
    d_indices = [i for i, d in enumerate(all_dialects) if d == dialect]
    if not d_indices:
        continue
    dialect_subset = processed_test.select(d_indices)
    
    print(f"  Evaluating {dialect} (n={len(dialect_subset)})...")
    result = eval_trainer.predict(dialect_subset)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(result.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    
    # ADDITION 4: Write out each utterance exactly like the FFT script did
    for idx, r, p in zip(dialect_subset["orig_idx"], label_str, pred_str):
        persample_file.write(json.dumps({
            "orig_idx": int(idx),
            "dialect": dialect,
            "reference": r,
            "prediction": p
        }, ensure_ascii=False) + "\n")
    persample_file.flush()
    
    # Filter empty references to avoid evaluation crashes
    valid_pairs = [(r, p) for r, p in zip(label_str, pred_str) if r.strip()]
    valid_refs = [r for r, _ in valid_pairs]
    valid_preds = [p for _, p in valid_pairs]

    wer = wer_metric.compute(predictions=valid_preds, references=valid_refs) * 100
    cer = cer_metric.compute(predictions=valid_preds, references=valid_refs) * 100
    dialect_results[dialect] = {"n_samples": len(dialect_subset), "wer": round(wer, 2), "cer": round(cer, 2)}
    all_pred.extend(valid_preds); all_ref.extend(valid_refs)
    print(f"    n={len(dialect_subset):4d}  WER={wer:.2f}%  CER={cer:.2f}%")

persample_file.close()

overall_wer = wer_metric.compute(predictions=all_pred, references=all_ref) * 100
overall_cer = cer_metric.compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}")
print(f"  JOINT COMPOSITION ABLATION [{EVAL_MODEL_SOURCE.upper()}] — FINAL RESULTS (beam=5)")
print(f"{'='*60}")
for d in unique_dialects:
    if d in dialect_results:
        m = dialect_results[d]
        print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%")
m = dialect_results["OVERALL"]
print(f"  {'OVERALL':<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}% ←")
print(f"{'='*60}")

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {RESULTS_PATH}")
print(f"Per-sample results saved for bootstrap to {PERSAMPLE_PATH}")