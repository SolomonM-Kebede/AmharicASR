"""
Amharic ASR — RQ2: Joint Leyu + ALFFA + Common Voice Training
================================================================================
Controlled comparison against the Leyu-only Phase 2 result (56.84% OVERALL WER).
Mixes in standard speech (100% of curated ALFFA, topped off with CV) up to the
defined STANDARD_RATIO to match the baseline training volume.
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # add at the very top

import io
import json
import pandas as pd
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import (
    load_dataset, concatenate_datasets, Audio, load_from_disk, Dataset as HFDataset,
)
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments,
    Seq2SeqTrainer, GenerationConfig, EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model

print(f"torch  : {torch.__version__}")
if torch.cuda.device_count() == 0:
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ─── Hyperparameters (Held constant from Phase 2) ───────────────────────────
LORA_R           = int(os.environ.get("LORA_R", 64))
LORA_ALPHA       = int(os.environ.get("LORA_ALPHA", 128))
LORA_DROPOUT     = float(os.environ.get("LORA_DROPOUT", 0.09))
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", 1e-4))
EPOCHS           = int(os.environ.get("EPOCHS", 10))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 8))
EVAL_BATCH_SIZE  = int(os.environ.get("EVAL_BATCH_SIZE", 8))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", 4))
EVAL_STEPS       = int(os.environ.get("EVAL_STEPS", 200))
MAX_TEXT_CHARS   = 350
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 800))
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "true").lower() == "true"

# ─── Dataset Ratios & Paths ─────────────────────────────────────────────────
STANDARD_RATIO = float(os.environ.get("STANDARD_RATIO", 1.0))
MIX_SEED       = 42

BASE_DIR      = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
LOCAL_CV_DIR  = os.path.join(BASE_DIR, "cv-corpus-25.0-2026-03-09/am")

PHASE2B_DIR   = os.path.join(BASE_DIR, "phase2b_leyu_standard")
OUTPUT_DIR    = os.path.join(PHASE2B_DIR, "models")
RESULTS_DIR   = os.path.join(PHASE2B_DIR, "results")
STANDARD_CACHE = os.path.join(PHASE2B_DIR, f"standard_speech_cache_ratio{STANDARD_RATIO}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Normalization ──────────────────────────────────────────────────────────
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

# ─── 1. Load Leyu (Unchanged) ───────────────────────────────────────────────
print(f"\nLoading existing Leyu splits from {LEYU_CACHE} ...")
leyu_cached = load_from_disk(LEYU_CACHE)
leyu_train  = leyu_cached["train"]
leyu_val    = leyu_cached["validation"]
leyu_test   = leyu_cached["test"]

# ─── 2. Prepare Standard Speech (ALFFA + CV) ────────────────────────────────
if Path(STANDARD_CACHE).exists():
    print(f"\nStandard Cache found — loading from {STANDARD_CACHE}")
    standard_train = load_from_disk(STANDARD_CACHE)
else:
    print("\nBuilding Standard Speech Cache (ALFFA + Local Common Voice)...")
    
    # Load ALFFA
    alffa_raw = load_dataset("hadamard-2/alffa-amharic", split="train", verification_mode="no_checks")
    alffa_raw = alffa_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    if "transcript" in alffa_raw.column_names:
        alffa_raw = alffa_raw.rename_column("transcript", "text")
    
    # Load Local CV and STRICTLY FILTER missing files
    cv_tsv = os.path.join(LOCAL_CV_DIR, "train.tsv")
    cv_df = pd.read_csv(cv_tsv, sep="\t")[["path", "sentence"]]
    cv_df = cv_df.rename(columns={"sentence": "text"})
    
    def resolve_cv_path(p):
        clip_path = os.path.join(LOCAL_CV_DIR, "clips", p)
        if os.path.exists(clip_path): return clip_path
        direct_path = os.path.join(LOCAL_CV_DIR, p)
        if os.path.exists(direct_path): return direct_path
        return None

    cv_df["audio"] = cv_df["path"].apply(resolve_cv_path)
    initial_cv_len = len(cv_df)
    cv_df = cv_df.dropna(subset=["audio"])  # Drop missing files to prevent silence injection
    print(f"  CV: Kept {len(cv_df):,} of {initial_cv_len:,} files (dropped {initial_cv_len - len(cv_df):,} missing)")

    cv_df = cv_df.drop(columns=["path"])
    cv_raw = HFDataset.from_dict(cv_df.to_dict("list"))
    cv_raw = cv_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    
    # ─── Corrected Sampling: Keep curated ALFFA, fill remainder with CV ─────
    target_standard_n = int(len(leyu_train) * STANDARD_RATIO)
    
    alffa_take = min(len(alffa_raw), target_standard_n)
    alffa_raw = alffa_raw.shuffle(seed=MIX_SEED).select(range(alffa_take))
    
    cv_take = min(len(cv_raw), max(0, target_standard_n - alffa_take))
    cv_raw = cv_raw.shuffle(seed=MIX_SEED).select(range(cv_take))
    
    print(f"  Target standard size : {target_standard_n:,}")
    print(f"  Taking from ALFFA    : {alffa_take:,}")
    print(f"  Taking from CV       : {cv_take:,}")
    
    raw_standard = concatenate_datasets([alffa_raw, cv_raw])
    
    def prepare_standard(batch):
        import librosa
        audio_data = batch["audio"]
        try:
            if audio_data.get("bytes") is not None:
                audio_arr, _ = librosa.load(io.BytesIO(audio_data["bytes"]), sr=16_000, mono=True)
            elif audio_data.get("path") is not None:
                audio_arr, _ = librosa.load(audio_data["path"], sr=16_000, mono=True)
            else:
                audio_arr = np.array(audio_data["array"], dtype=np.float32)
        except Exception:
            audio_arr = np.zeros(16_000, dtype=np.float32)

        batch["input_features"] = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        ).input_features[0]
        
        batch["labels"] = processor.tokenizer(
            normalize_amharic(str(batch["text"])), max_length=448, truncation=True
        ).input_ids
        batch["dialect"] = "standard"
        return batch

    print("  Mapping Standard Audio (running single-threaded to prevent deadlocks)...")
    standard_train = raw_standard.map(
        prepare_standard,
        remove_columns=["audio", "text"],
    )
    standard_train = standard_train.filter(lambda x: len(processor.tokenizer.decode(x["labels"])) <= MAX_TEXT_CHARS)
    standard_train.save_to_disk(STANDARD_CACHE)
    print(f"  Saved standard cache to {STANDARD_CACHE}")

# ─── 3. Final Combine ───────────────────────────────────────────────────────
train_dataset = concatenate_datasets([leyu_train, standard_train]).shuffle(seed=42)

print(f"\nFinal Train Set: {len(train_dataset):,} (Leyu: {len(leyu_train):,} + Standard: {len(standard_train):,})")

if 0 < EVAL_SUBSET_SIZE < len(leyu_val):
    train_time_eval_dataset = leyu_val.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
    print(f"Validation Set : {EVAL_SUBSET_SIZE:,} subsampled for in-training eval (from {len(leyu_val):,} total)")
else:
    train_time_eval_dataset = leyu_val
    print(f"Validation Set : {len(leyu_val):,} total")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features):
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
    return {"wer": round(_wer.compute(predictions=pred_str, references=label_str)*100, 2),
            "cer": round(_cer.compute(predictions=pred_str, references=label_str)*100, 2)}

class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()
        outputs = model(input_features=input_features, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# ─── 4. Training ────────────────────────────────────────────────────────────
print(f"\nLoading base model from {PHASE1_MERGED} ...")
model = WhisperForConditionalGeneration.from_pretrained(
    PHASE1_MERGED,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj",
                    "encoder_attn.q_proj", "encoder_attn.v_proj",
                    "encoder_attn.k_proj", "encoder_attn.out_proj"],
    bias="none", inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config = GenerationConfig(
    forced_decoder_ids=forced_ids, max_new_tokens=225, num_beams=1,
    use_cache=True, decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=200,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    lr_scheduler_type="cosine",
    bf16=True, fp16=False,
    gradient_checkpointing=True,
    eval_strategy="steps", eval_steps=EVAL_STEPS, save_steps=EVAL_STEPS,
    save_total_limit=3, load_best_model_at_end=True,
    metric_for_best_model="wer", greater_is_better=False,
    predict_with_generate=True, # generation_max_length removed to stop the warning
    logging_steps=50, report_to=["none"],
    dataloader_num_workers=4, remove_unused_columns=False,
)

trainer = SafeWhisperTrainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=train_time_eval_dataset,
    data_collator=data_collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print("\nStarting Phase 2b Joint Training...")
resume_arg = RESUME_FROM_CHECKPOINT
if resume_arg and get_last_checkpoint(OUTPUT_DIR) is None:
    print(f"RESUME_FROM_CHECKPOINT=true but no checkpoint found in {OUTPUT_DIR} — "
          f"this looks like a first run, starting fresh instead.")
    resume_arg = False

print("\nStarting Phase 2b Joint Training...")
trainer.train(resume_from_checkpoint=resume_arg)

# ─── 5. Save & Merge ────────────────────────────────────────────────────────
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "phase2b_lora_adapter")
MERGED_DIR  = os.path.join(OUTPUT_DIR, "phase2b_final_merged")
trainer.model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
merged = trainer.model.cpu().merge_and_unload()
merged.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)

# ─── 6. H200-Optimized Final Evaluation ─────────────────────────────────────
print("\nRunning FINAL evaluation on Leyu-only test set...")
eval_model = WhisperForConditionalGeneration.from_pretrained(
    MERGED_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
).to("cuda").eval()

eval_model.generation_config = GenerationConfig(
    forced_decoder_ids=forced_ids, max_new_tokens=225, num_beams=5,
    use_cache=True, decoder_start_token_id=eval_model.config.decoder_start_token_id,
)

eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_phase2b", per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True, bf16=True, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)
eval_trainer = SafeWhisperTrainer(model=eval_model, args=eval_args, data_collator=data_collator, compute_metrics=compute_metrics)

DIALECT_NAMES = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
dialect_results = {}
all_pred, all_ref = [], []

for d_name in DIALECT_NAMES:
    d_test = leyu_test.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0: continue
    result = eval_trainer.predict(d_test)
    
    # 1. Safely filter predictions
    p_ids = result.predictions
    if isinstance(p_ids, tuple):  # Catch edge cases where predictions return a tuple
        p_ids = p_ids[0]
    p_ids = p_ids.copy()
    p_ids[p_ids == -100] = processor.tokenizer.pad_token_id
    
    # 2. Safely filter labels
    l_ids = result.label_ids.copy()
    l_ids[l_ids == -100] = processor.tokenizer.pad_token_id
    
    # 3. Decode
    p_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(p_ids, skip_special_tokens=True)]
    l_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(l_ids, skip_special_tokens=True)]
    
    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=p_str, references=l_str) * 100
    cer = _eval.load("cer").compute(predictions=p_str, references=l_str) * 100
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    all_pred.extend(p_str)
    all_ref.extend(l_str)
    print(f"  {d_name:15s}  n={len(d_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%")

overall_wer = _eval.load("wer").compute(predictions=all_pred, references=all_ref) * 100
overall_cer = _eval.load("cer").compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}\n  RQ2 — LEYU+ALFFA+CV JOINT TRAINING — FINAL RESULTS\n{'='*60}")
for d, m in dialect_results.items():
    marker = " ←" if d == "OVERALL" else ""
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%{marker}")
print(f"{'='*60}")
print(f"  Leyu-only baseline (RQ2 comparison) : 56.84% WER")

results_path = os.path.join(RESULTS_DIR, f"phase2b_leyu_standard_results_ratio{STANDARD_RATIO}.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {results_path}")