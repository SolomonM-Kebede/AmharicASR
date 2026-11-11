#!/usr/bin/env python3
"""
train_whisper_amharic_largev3.py
================================
Full fine-tune of Whisper large-v3 for Amharic ASR.

Datasets
  - 4 Leyu dialect datasets  (Gojjam, Gonder, Wello, Shewa)
  - ALFFA Amharic
  - Mozilla Common Voice Amharic  ← new

GPU target  : single H200 NVL (80 GB) or L40S (48 GB)
Multi-GPU   : set --nproc_per_node N and launch with torchrun

Key upgrades over the medium / notebook version
  - Whisper large-v3  (1.5 B params)
  - Flash Attention 2 for 40-50 % memory saving on attn
  - bf16 + TF32 matmuls
  - LoRA (r=64, alpha=128) — keeps VRAM inside H200 budget;
    flip USE_LORA=False for a true full fine-tune if you have headroom
  - Gradient checkpointing
  - Cosine LR schedule with warmup
  - Early stopping (patience = 5 eval rounds)
  - Mozilla CV Amharic loading from local directory (TSV + clips/)
"""

# ── 0. Env vars (must be set before any torch/HF import) ────────────────────
import os

os.environ["FORCE_SOUNDFILE"]    = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# TF32 — free 10 % speed on H200/A100 with negligible precision loss
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

# ── 1. Configuration ─────────────────────────────────────────────────────────
# ─── Datasets ────────────────────────────────────────────────────────────────
LEYU_DATASETS = [
    "leyu-amharic/leyu-amharic-gojjam-dialect",
    "leyu-amharic/leyu-amharic-gonder-dialect",
    "leyu-amharic/leyu-amharic-wello-dialect",
    "leyu-amharic/leyu-amharic-shewa-dialect",
    "gheero-Leyu/leyu-amharic-addis-ababa-dialect",
]
ALFFA_DATASET  = "hadamard-2/alffa-amharic"

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_NAME = "openai/whisper-large-v3"

# ─── LoRA ────────────────────────────────────────────────────────────────────
USE_LORA    = False     # set False for full fine-tune (needs > 60 GB VRAM)
LORA_R      = 64
LORA_ALPHA  = 128
LORA_DROPOUT= 0.05
# Whisper large-v3 attention projection names
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",   # encoder + decoder self-attn
    "fc1", "fc2",                                 # FFN layers
]

# ─── Hyperparameters ─────────────────────────────────────────────────────────
EPOCHS         = 10
BATCH_SIZE     = 4       # per GPU; effective batch = BATCH_SIZE * GRAD_ACCUM
GRAD_ACCUM     = 8        # effective = 32 — same as notebook
LEARNING_RATE  = 1e-4     # larger LR works better with LoRA
WARMUP_STEPS   = 500
EVAL_STEPS     = 300
SAVE_STEPS     = 300
MAX_TEXT_CHARS = 350
FREEZE_ENCODER = False    # large-v3 encoder is powerful; keep it trainable
LABEL_MAX_LEN  = 448

# ─── Data splits ─────────────────────────────────────────────────────────────
VAL_RATIO  = 0.10
TEST_RATIO = 0.10
SEED       = 42

# ─── I/O ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.expanduser("~")
OUTPUT_DIR = os.path.join(BASE_DIR, "whisper-amharic-largev3")
LOG_DIR    = os.path.join(BASE_DIR, "logs-largev3")

# Use /tmp if home partition is small (common on university clusters)
CACHE_DIR  = os.environ.get("HF_CACHE", os.path.join(BASE_DIR, ".cache", "huggingface"))

# ─── Hub ─────────────────────────────────────────────────────────────────────
PUSH_TO_HUB  = False
HUB_MODEL_ID = "your-username/whisper-amharic-large-v3"

# ─── Local Mozilla Common Voice ───────────────────────────────────────────────
# Point LOCAL_CV_DIR to the folder where you extracted the CV corpus, e.g.:
#   export LOCAL_CV_DIR=/data/cv-corpus-am
# Expected layout:
#   $LOCAL_CV_DIR/train.tsv  (and/or dev.tsv / test.tsv / validated.tsv)
#   $LOCAL_CV_DIR/clips/<filename.mp3>
LOCAL_CV_DIR = os.environ.get("LOCAL_CV_DIR", os.path.join(BASE_DIR, "cv-corpus-am"))

# ── 2. Imports ───────────────────────────────────────────────────────────────
import sys
import random
import gc
import io
import json
import logging
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── 3. GPU check ─────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. Aborting.")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
log.info(f"GPU  : {gpu_name}")
log.info(f"VRAM : {vram_gb:.1f} GB")

if vram_gb < 40 and not USE_LORA:
    log.warning("< 40 GB VRAM detected and USE_LORA=False — forcing USE_LORA=True")
    USE_LORA = True

# ── 4. Create dirs ───────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# Normalization
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
    text = text.translate(str.maketrans("", "", PUNCT))
    return " ".join(text.split()).strip()

# ── 6. Load datasets ─────────────────────────────────────────────────────────
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
import soundfile as sf
import librosa

log.info("Loading Leyu dialect datasets...")
all_parts: List[Any] = []

for repo_id in LEYU_DATASETS:
    log.info(f"  {repo_id}")
    ds = load_dataset(repo_id, split="train", cache_dir=CACHE_DIR)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    n_spk = len(set(ds["speaker_id"]))
    log.info(f"    → {len(ds):,} examples | {n_spk} speakers")
    all_parts.append(ds)

log.info("Loading ALFFA Amharic...")
alffa = load_dataset(ALFFA_DATASET, split="train", cache_dir=CACHE_DIR)
alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
alffa = alffa.rename_column("transcript", "text")
alffa = alffa.map(lambda x: {"dialect": "standard", "gender": "unknown"})
# Keep only columns matching Leyu schema
keep_cols = {"audio", "text", "dialect", "speaker_id", "gender"}
drop_cols  = [c for c in alffa.column_names if c not in keep_cols]
if drop_cols:
    alffa = alffa.remove_columns(drop_cols)
# Add speaker_id if missing
if "speaker_id" not in alffa.column_names:
    alffa = alffa.map(lambda x, idx: {"speaker_id": f"alffa_{idx}"}, with_indices=True)
log.info(f"  → {len(alffa):,} ALFFA examples")
all_parts.append(alffa)

# ── Mozilla Common Voice Amharic (local) ─────────────────────────────────────
log.info("Loading Mozilla Common Voice Amharic (local)...")
mcv_loaded = False

# We merge all available TSV splits so no labelled data is wasted.
# The speaker-independent split later ensures no speaker leaks across train/val/test.
CV_TSV_SPLITS = ["train.tsv", "dev.tsv", "test.tsv",
                 "validated.tsv", "other.tsv"]   # load whichever exist

cv_frames = []
for tsv_name in CV_TSV_SPLITS:
    tsv_path = os.path.join(LOCAL_CV_DIR, tsv_name)
    if not os.path.exists(tsv_path):
        continue
    df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "path", "sentence", "gender"],
                     dtype=str)
    df["_split"] = tsv_name.replace(".tsv", "")
    cv_frames.append(df)
    log.info(f"  {tsv_name}: {len(df):,} rows")

if not cv_frames:
    log.warning(
        f"No Common Voice TSV files found in {LOCAL_CV_DIR}. "
        "Set LOCAL_CV_DIR to the extracted CV corpus folder and re-run."
    )
else:
    cv_df = pd.concat(cv_frames, ignore_index=True)
    cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
    cv_df["dialect"] = "cv_amharic"
    cv_df["gender"]  = cv_df["gender"].fillna("unknown")

    # Drop rows with empty transcripts
    cv_df = cv_df[cv_df["text"].notna() & (cv_df["text"].str.strip() != "")]

    # Resolve audio paths: prefer clips/ sub-folder, fall back to bare path
    sample_path = cv_df["path"].iloc[0]
    clips_path  = os.path.join(LOCAL_CV_DIR, "clips", sample_path)
    bare_path   = os.path.join(LOCAL_CV_DIR, sample_path)
    if os.path.exists(clips_path):
        cv_df["audio_path"] = cv_df["path"].apply(
            lambda p: os.path.join(LOCAL_CV_DIR, "clips", p))
    elif os.path.exists(bare_path):
        cv_df["audio_path"] = cv_df["path"].apply(
            lambda p: os.path.join(LOCAL_CV_DIR, p))
    else:
        raise FileNotFoundError(
            f"Cannot resolve CV audio path.\n"
            f"Tried:\n  {clips_path}\n  {bare_path}\n"
            f"Check that LOCAL_CV_DIR={LOCAL_CV_DIR} is correct."
        )

    # Drop rows whose audio file is missing on disk (corrupt/incomplete download)
    before_drop = len(cv_df)
    cv_df = cv_df[cv_df["audio_path"].apply(os.path.exists)]
    dropped = before_drop - len(cv_df)
    if dropped:
        log.warning(f"  Dropped {dropped:,} CV rows with missing audio files.")

    # Build a HuggingFace Dataset from the dataframe.
    # Audio is stored as file paths and decoded on-the-fly in prepare_batch
    # (same soundfile decode path used for Leyu/ALFFA).
    from datasets import Dataset as HFDataset

    def _cv_row_to_hf(row):
        """Return a dict with the unified schema expected by prepare_batch."""
        return {
            "audio"     : {"path": row["audio_path"], "bytes": None},
            "text"      : row["text"],
            "dialect"   : row["dialect"],
            "speaker_id": row["speaker_id"],
            "gender"    : row["gender"],
        }

    cv_records = [_cv_row_to_hf(r) for _, r in cv_df.iterrows()]
    mcv_hf = HFDataset.from_list(cv_records)

    log.info(f"  → {len(mcv_hf):,} Common Voice examples (after path validation)")
    log.info(f"  → {cv_df['speaker_id'].nunique():,} unique speakers")
    # ── PREVENT GRADIENT STARVATION: Balance Standard vs. Dialect Data ──
    # 1. Count total Leyu dialect samples (all datasets in all_parts except the last one, ALFFA)
    leyu_count = sum(len(d) for d in all_parts[:-1])
    alffa_count = len(alffa)

    # 2. Set your balance knob (1.0 = perfect 50/50 balance between Standard and Dialects)
    # Lowering to 0.8 means the final dataset will be 55% dialects / 45% standard Amharic.
    STANDARD_TO_DIALECT_RATIO = 1.0

    max_standard_allowed = int(leyu_count * STANDARD_TO_DIALECT_RATIO)
    mcv_quota = max(0, max_standard_allowed - alffa_count)

    if len(mcv_hf) > mcv_quota:
        log.info(f"  Downsampling Common Voice from {len(mcv_hf):,} → {mcv_quota:,} to match dialect volume.")
        mcv_hf = mcv_hf.shuffle(seed=SEED).select(range(mcv_quota))
    else:
        log.info(f"  Keeping all {len(mcv_hf):,} Common Voice examples (under dialect quota).")
    # ────────────────────────────────────────────────────────────────────

    log.info(f"  → {len(mcv_hf):,} Common Voice examples kept")
    log.info(f"  → {cv_df['speaker_id'].nunique():,} unique speakers")
    all_parts.append(mcv_hf)
    mcv_loaded = True

combined = concatenate_datasets(all_parts)
log.info(f"Combined dataset: {len(combined):,} examples")
log.info(f"Dialects        : {set(combined['dialect'])}")

# ── 7. Speaker-independent split ─────────────────────────────────────────────
log.info("Building speaker-independent splits...")
speakers: Dict[str, List[int]] = defaultdict(list)
for i, row in enumerate(combined):
    speakers[row["speaker_id"]].append(i)

spk_ids = list(speakers.keys())
random.Random(SEED).shuffle(spk_ids)
n = len(spk_ids)

n_test = max(1, int(n * TEST_RATIO))
n_val  = max(1, int(n * VAL_RATIO))

test_spk  = set(spk_ids[:n_test])
val_spk   = set(spk_ids[n_test:n_test + n_val])
train_spk = set(spk_ids[n_test + n_val:])

train_idx = [i for s, idxs in speakers.items() if s in train_spk for i in idxs]
val_idx   = [i for s, idxs in speakers.items() if s in val_spk   for i in idxs]
test_idx  = [i for s, idxs in speakers.items() if s in test_spk  for i in idxs]

dataset = DatasetDict({
    "train"     : combined.select(train_idx),
    "validation": combined.select(val_idx),
    "test"      : combined.select(test_idx),
})

for split, ds in dataset.items():
    spks     = len(set(ds["speaker_id"]))
    dialects = set(ds["dialect"])
    log.info(f"  {split:12s}: {len(ds):5,} examples | {spks} speakers | dialects: {dialects}")

# ── 8. Text length filter ─────────────────────────────────────────────────────
before  = {k: len(v) for k, v in dataset.items()}
dataset = dataset.filter(lambda x: len(x["text"]) <= MAX_TEXT_CHARS)
after   = {k: len(v) for k, v in dataset.items()}

for split in dataset:
    dropped = before[split] - after[split]
    pct     = dropped / before[split] * 100
    log.info(f"  {split:12s}: {after[split]:,} kept ({dropped:,} dropped, {pct:.0f}%)")

# ── 9. Processor & feature extraction ────────────────────────────────────────
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

log.info(f"Loading processor for {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language="amharic",
    task="transcribe",
    cache_dir=CACHE_DIR,
)

def prepare_batch(batch):
    audio_data = batch["audio"]

    if isinstance(audio_data, dict):
        # Case 1: raw bytes in parquet (Leyu / ALFFA)
        if audio_data.get("bytes") is not None:
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16_000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)
        # Case 2: local file path (Common Voice clips on disk)
        elif audio_data.get("path") is not None:
            audio_array, sr = sf.read(audio_data["path"])
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            if sr != 16_000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)
        # Case 3: already decoded array (fallback)
        else:
            audio_array = np.array(audio_data["array"], dtype=np.float32)
    else:
        audio_array = np.array(audio_data, dtype=np.float32)

    audio_array = audio_array.astype(np.float32)

    batch["input_features"] = processor.feature_extractor(
        audio_array,
        sampling_rate=16_000,
        return_tensors="pt",
    ).input_features[0]

    batch["labels"] = processor.tokenizer(
        normalize_amharic(batch["text"]),
        max_length=LABEL_MAX_LEN,
        truncation=True,
    ).input_ids

    return batch

log.info("Extracting log-Mel features (this takes a while)...")
dataset = dataset.map(
    prepare_batch,
    remove_columns=["text", "audio", "dialect", "speaker_id", "gender"],
    num_proc=4,
    desc="Feature extraction",
)
log.info("Features ready.")
for split in dataset:
    log.info(f"  {split:12s}: {len(dataset[split]):,} examples")

# ── 10. Model ─────────────────────────────────────────────────────────────────
from transformers import WhisperForConditionalGeneration

log.info(f"Loading {MODEL_NAME}...")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
    # Flash Attention 2 — requires transformers >= 4.36 and flash-attn installed
    # If flash-attn is not installed, remove the line below; it will fall back to SDPA
    attn_implementation="flash_attention_2",
)

model.config.use_cache = False
model.generation_config.language = "amharic"
model.generation_config.task     = "transcribe"
model.generation_config.forced_decoder_ids = None

if FREEZE_ENCODER:
    log.info("Freezing encoder.")
    model.freeze_encoder()

# ── 11. LoRA ──────────────────────────────────────────────────────────────────
if USE_LORA:
    from peft import LoraConfig, get_peft_model, TaskType

    log.info(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        # For Whisper we use SEQ_2_SEQ_LM — the model *does* accept input_features
        # through its own forward; PEFT just wraps the attention/FFN layers
        task_type=TaskType.SEQ_2_SEQ_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    log.info("Full fine-tune (no LoRA).")

model.gradient_checkpointing_enable()

total     = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
log.info(f"Total params    : {total:.0f}M")
log.info(f"Trainable params: {trainable:.0f}M")

# ── 12. Data collator & metrics ───────────────────────────────────────────────
import evaluate as hf_evaluate

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]}              for f in features]

        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features,         return_tensors="pt")

        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]

        batch["labels"] = lids
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

wer_metric = hf_evaluate.load("wer")
cer_metric = hf_evaluate.load("cer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
    }

# ── 13. VRAM snapshot ────────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()
free, total_vram = torch.cuda.mem_get_info()
log.info(f"Free VRAM before training: {free/1e9:.2f} GB / {total_vram/1e9:.2f} GB")

# ── 14. Training ─────────────────────────────────────────────────────────────
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    # Epochs & batch
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Optimiser
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    lr_scheduler_type="cosine",        # cosine decay — better than linear for long runs
    weight_decay=0.01,

    # Precision
    bf16=True,
    fp16=False,
    tf32=True,
    gradient_checkpointing=True,

    # Eval & save
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,

    # Generation
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=5,

    # Logging
    logging_steps=25,
    logging_dir=LOG_DIR,
    report_to=["tensorboard"],

    # Data loading
    dataloader_num_workers=4,
    dataloader_pin_memory=True,

    # Hub
    push_to_hub=PUSH_TO_HUB,
    hub_model_id=HUB_MODEL_ID if PUSH_TO_HUB else None,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)

log.info("=" * 60)
log.info("Training started")
log.info(f"  Model           : {MODEL_NAME}")
log.info(f"  LoRA            : {USE_LORA}")
log.info(f"  Train examples  : {len(dataset['train']):,}")
log.info(f"  Val   examples  : {len(dataset['validation']):,}")
log.info(f"  Effective batch : {BATCH_SIZE * GRAD_ACCUM}")
log.info(f"  MCV loaded      : {mcv_loaded}")
log.info("=" * 60)

train_result = trainer.train()

trainer.save_model()
processor.save_pretrained(OUTPUT_DIR)
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()

log.info(f"Model saved to: {OUTPUT_DIR}")

# ── 15. Test-set evaluation ───────────────────────────────────────────────────
log.info("Evaluating on test set...")
results   = trainer.predict(dataset["test"])
pred_ids  = results.predictions
label_ids = results.label_ids
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

pred_str  = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

test_wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
test_cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100

log.info("=" * 50)
log.info(f"  Test WER : {test_wer:.2f}%")
log.info(f"  Test CER : {test_cer:.2f}%")
log.info(f"  Baseline (medium, no MCV): 43.9% WER")
log.info(f"  Google STT baseline      : ~35%  WER")
log.info("=" * 50)

# Save results JSON
results_path = Path(OUTPUT_DIR) / "test_results.json"
with open(results_path, "w") as f:
    json.dump({"test_wer": test_wer, "test_cer": test_cer}, f, indent=2)
log.info(f"Results saved to {results_path}")

# ── 16. Training curves plot ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    log.info("Generating training curves plot...")
    log_path = Path(OUTPUT_DIR) / "trainer_state.json"
    with open(log_path) as f:
        trainer_state = json.load(f)

    logs       = trainer_state["log_history"]
    train_logs = [l for l in logs if "loss" in l and "eval_loss" not in l]
    eval_logs  = [l for l in logs if "eval_loss" in l]

    train_steps = [l["step"]      for l in train_logs]
    train_loss  = [l["loss"]      for l in train_logs]
    eval_steps  = [l["step"]      for l in eval_logs]
    eval_loss   = [l["eval_loss"] for l in eval_logs]
    eval_wer    = [l["eval_wer"]  for l in eval_logs]
    eval_cer    = [l["eval_cer"]  for l in eval_logs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Whisper large-v3 — Amharic ASR (Leyu + ALFFA + MCV)", fontsize=13, fontweight="bold")

    ax1.plot(train_steps, train_loss, color="#3266ad", lw=2, label="train loss")
    ax1.plot(eval_steps,  eval_loss,  color="#e07b39", lw=2, label="val loss", marker="o", ms=4)
    ax1.axvline(eval_steps[np.argmin(eval_loss)], color="gray", ls="--", alpha=0.6, label="best val loss")
    ax1.set(title="Loss curves", xlabel="Step", ylabel="Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(bottom=0)

    ax2.plot(eval_steps, eval_wer, color="#9b59b6", lw=2, label="WER %",  marker="o", ms=4)
    ax2.plot(eval_steps, eval_cer, color="#27ae60", lw=2, label="CER %",  marker="s", ms=4)
    ax2.axhline(35.0, color="#e24b4a", ls="--", lw=1.5, label="Google STT (~35%)")
    ax2.axhline(43.9, color="#888888", ls=":",  lw=1.5, label="Medium baseline (43.9%)")
    best_idx = int(np.argmin(eval_wer))
    ax2.annotate(
        f"best: {eval_wer[best_idx]:.1f}%",
        xy=(eval_steps[best_idx], eval_wer[best_idx]),
        xytext=(eval_steps[best_idx] + max(eval_steps) * 0.05, eval_wer[best_idx] + 5),
        arrowprops=dict(arrowstyle="->", color="#9b59b6"),
        fontsize=9, color="#9b59b6",
    )
    ax2.set(title="WER & CER", xlabel="Step", ylabel="Error Rate (%)")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = Path(OUTPUT_DIR) / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved to {plot_path}")
except Exception as e:
    log.warning(f"Could not generate plot: {e}")

# ── 17. Export model zip ──────────────────────────────────────────────────────
import zipfile, shutil

zip_path = Path(OUTPUT_DIR).parent / "whisper-amharic-largev3-model.zip"
log.info(f"Zipping model to {zip_path} ...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fp in Path(OUTPUT_DIR).rglob("*"):
        if fp.is_file() and "checkpoint" not in str(fp):
            zf.write(fp, fp.relative_to(OUTPUT_DIR))

size_mb = zip_path.stat().st_size / 1e6
log.info(f"Done. Archive size: {size_mb:.0f} MB")
log.info("All done.")