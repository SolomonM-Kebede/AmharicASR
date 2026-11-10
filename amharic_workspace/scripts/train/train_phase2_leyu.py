"""
Amharic ASR — Phase 2 Leyu Dialect Adaptation with resume capability from the last checkpoint
Dataset : Leyu Amharic dialect(Gojjam, Gonder, Wello, Shewa + Addis Ababa), expanded corpus
Model   : Merged Phase 1 model + fresh LoRA adapter
"""

import os

# cache override to prevent duplication
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

import os
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import (
    load_dataset, concatenate_datasets,
    DatasetDict, Audio, load_from_disk,
)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType


# Sanity check
print(f"torch  : {torch.__version__}")
N_GPUS = torch.cuda.device_count()
if N_GPUS == 0:
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")


# Hyperparameters (env-overridable for Ray HPO trials)
LORA_R          = int(os.environ.get("LORA_R", 64))
LORA_ALPHA      = int(os.environ.get("LORA_ALPHA", 128))
LORA_DROPOUT    = float(os.environ.get("LORA_DROPOUT", 0.09))
LEARNING_RATE   = float(os.environ.get("LEARNING_RATE", 1e-4))
WEIGHT_DECAY    = float(os.environ.get("WEIGHT_DECAY", 0.01))
WARMUP_STEPS    = int(os.environ.get("WARMUP_STEPS", 200))
EPOCHS          = int(os.environ.get("EPOCHS", 10))
TRAIN_BATCH_SIZE= int(os.environ.get("TRAIN_BATCH_SIZE", 16))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
GRAD_ACCUM      = int(os.environ.get("GRAD_ACCUM", 2))
EVAL_STEPS      = int(os.environ.get("EVAL_STEPS", 200))
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 5))
LABEL_SMOOTHING = float(os.environ.get("LABEL_SMOOTHING", 0.1))
APPLY_SPEC_AUGMENT = os.environ.get("APPLY_SPEC_AUGMENT", "false").lower() == "true"

# Split ratios 
TEST_SIZE               = float(os.environ.get("TEST_SIZE", 0.10))
VAL_SIZE_OF_REMAINING   = float(os.environ.get("VAL_SIZE_OF_REMAINING", 0.1111))
SPLIT_SEED              = int(os.environ.get("SPLIT_SEED", 42))

# cache match
DATASET_VERSION      = os.environ.get("DATASET_VERSION", "2026-08-expanded-spkindep")
FORCE_REFRESH_CACHE   = os.environ.get("FORCE_REFRESH_CACHE", "false").lower() == "true"

# fast training and eval by subsexmapling 
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 800))

# Resume control 
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "true").lower() == "true"


# Paths 
BASE_DIR   = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(
    BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged"
)
if not Path(PHASE1_MERGED).exists():
    raise FileNotFoundError(
        f"Phase 1 merged model not found at {PHASE1_MERGED}\n"
        "Run Phase 1 training first."
    )

PHASE2_DIR  = os.path.join(BASE_DIR, "phase2_leyu")
OUTPUT_DIR  = os.path.join(PHASE2_DIR, "models")   
LEYU_CACHE  = os.path.join(PHASE2_DIR, f"dataset_cache_{DATASET_VERSION}")
RESULTS_DIR = os.path.join(PHASE2_DIR, "results")

for d in [OUTPUT_DIR, LEYU_CACHE, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

if RESUME_FROM_CHECKPOINT:
    ckpt_path = Path(OUTPUT_DIR) / "checkpoint-1000"
    if not ckpt_path.exists():
        print(f"WARNING: RESUME_FROM_CHECKPOINT=true but {ckpt_path} not found. "
              f"Trainer will look for the latest checkpoint under {OUTPUT_DIR} instead.")


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
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


# Processor
print(f"\nLoading processor from {PHASE1_MERGED} ...")
processor = WhisperProcessor.from_pretrained(
    PHASE1_MERGED, language="amharic", task="transcribe"
)
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")


# Dataset — speaker-independent splits (loaded from cache; must match checkpoint-1000's run)
DIALECT_CONFIGS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect",  "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect",  "shewa"),
    ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]

cache_ready = (
    not FORCE_REFRESH_CACHE
    and Path(LEYU_CACHE).exists()
    and any(Path(LEYU_CACHE).iterdir())
)


def speaker_independent_dialect_split(ds_raw, dialect_name,
                                       test_size=TEST_SIZE,
                                       val_size_of_remaining=VAL_SIZE_OF_REMAINING,
                                       seed=SPLIT_SEED):
    from sklearn.model_selection import train_test_split as sk_split

    meta = ds_raw.remove_columns(
        [c for c in ds_raw.column_names if c not in ("speaker_id", "gender")]
    ).to_pandas()

    spk_meta = meta.drop_duplicates(subset="speaker_id").reset_index(drop=True)
    spk_meta["gender"] = spk_meta["gender"].fillna("unknown")

    def _safe_split(ids, strat, size):
        try:
            return sk_split(ids, test_size=size, random_state=seed, stratify=strat)
        except ValueError:
            print(f"    [{dialect_name}] gender-stratified split not feasible — using plain speaker split.")
            return sk_split(ids, test_size=size, random_state=seed)

    all_spk    = spk_meta["speaker_id"].tolist()
    all_gender = spk_meta["gender"].tolist()

    trainval_spk, test_spk = _safe_split(all_spk, all_gender, test_size)

    gender_by_spk = spk_meta.set_index("speaker_id")["gender"]
    trainval_gender = gender_by_spk.loc[trainval_spk].tolist()
    train_spk, val_spk = _safe_split(trainval_spk, trainval_gender, val_size_of_remaining)

    train_spk_set, val_spk_set, test_spk_set = set(train_spk), set(val_spk), set(test_spk)

    print(f"    [{dialect_name}] speakers: train={len(train_spk_set)} "
          f"val={len(val_spk_set)} test={len(test_spk_set)}")

    train_ds = ds_raw.filter(lambda x: x["speaker_id"] in train_spk_set)
    val_ds   = ds_raw.filter(lambda x: x["speaker_id"] in val_spk_set)
    test_ds  = ds_raw.filter(lambda x: x["speaker_id"] in test_spk_set)

    return train_ds, val_ds, test_ds


if cache_ready:
    print(f"\nCache found — loading from {LEYU_CACHE}")
    cached = load_from_disk(LEYU_CACHE)
    train_dataset = cached["train"]
    val_dataset   = cached["validation"]
    test_dataset  = cached["test"]
    print(f"  train={len(train_dataset):,}  val={len(val_dataset):,}  test={len(test_dataset):,}")
else:
    print(f"\nNo cache found at {LEYU_CACHE} (or FORCE_REFRESH_CACHE=true) — "
          f"rebuilding from scratch. If this is meant to resume checkpoint-1000, "
          f"double-check DATASET_VERSION matches the original run before continuing, "
          f"since the split is reproducible from the same seed but this costs real time.")
    print("\nDownloading Leyu dialect datasets ...")

    train_splits = []
    val_splits   = []
    test_splits  = []

    for repo_id, dialect_name in DIALECT_CONFIGS:
        print(f"  Fetching {dialect_name} ({repo_id}) ...")
        ds_raw = load_dataset(repo_id, split="train", verification_mode="no_checks")
        ds_raw = ds_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        n_spk = len(set(ds_raw["speaker_id"]))
        print(f"    → {len(ds_raw):,} examples | {n_spk} unique speakers")
        ds_raw = ds_raw.map(lambda x, d=dialect_name: {"dialect": d})

        dialect_train, dialect_val, dialect_test = speaker_independent_dialect_split(
            ds_raw, dialect_name
        )
        train_splits.append(dialect_train)
        val_splits.append(dialect_val)
        test_splits.append(dialect_test)

    raw_train = concatenate_datasets(train_splits)
    raw_val   = concatenate_datasets(val_splits)
    raw_test  = concatenate_datasets(test_splits)

    print(f"\n  Raw sizes — train={len(raw_train):,}  "
          f"val={len(raw_val):,}  test={len(raw_test):,}")

    train_spk_all = set(raw_train["speaker_id"])
    val_spk_all   = set(raw_val["speaker_id"])
    test_spk_all  = set(raw_test["speaker_id"])
    assert not (train_spk_all & val_spk_all),  "Speaker leakage between train/val"
    assert not (train_spk_all & test_spk_all), "Speaker leakage between train/test"
    assert not (val_spk_all & test_spk_all),   "Speaker leakage between val/test"
    print("  ✓ Speaker independence verified — zero overlapping speaker_id across splits.")

    def prepare_dataset(batch):
        import librosa, io as _io

        audio_data = batch["audio"]
        try:
            if isinstance(audio_data, dict):
                if audio_data.get("bytes") is not None:
                    audio_arr, _ = librosa.load(
                        _io.BytesIO(audio_data["bytes"]), sr=16_000, mono=True
                    )
                elif audio_data.get("path") is not None:
                    audio_arr, _ = librosa.load(
                        audio_data["path"], sr=16_000, mono=True
                    )
                else:
                    audio_arr = np.array(audio_data["array"], dtype=np.float32)
            else:
                audio_arr = np.array(audio_data, dtype=np.float32)
        except Exception as e:
            print(f"[prepare_dataset] audio load failed: {e} — substituting silence")
            audio_arr = np.zeros(16_000, dtype=np.float32)

        batch["input_features"] = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        ).input_features[0]

        transcript = batch.get("text") or batch.get("sentence") or ""
        batch["labels"] = processor.tokenizer(
            normalize_amharic(str(transcript)),
            max_length=448,
            truncation=True,
        ).input_ids

        return batch

    keep_cols = {"input_features", "labels", "dialect"}

    def map_split(raw, name):
        mapped = raw.map(
            prepare_dataset,
            remove_columns=[c for c in raw.column_names if c not in keep_cols],
            num_proc=4,
        )
        print(f"  {name}: {len(mapped):,} samples processed")
        return mapped

    print("\nExtracting log-Mel features ...")
    train_dataset = map_split(raw_train, "train")
    val_dataset   = map_split(raw_val,   "validation")
    test_dataset  = map_split(raw_test,  "test")

    print("\nPer-dialect training example counts:")
    for _, dialect_name in DIALECT_CONFIGS:
        n_d = sum(1 for d in train_dataset["dialect"] if d == dialect_name)
        print(f"  {dialect_name:15s}: {n_d:,}")

    DatasetDict({
        "train":      train_dataset,
        "validation": val_dataset,
        "test":       test_dataset,
    }).save_to_disk(LEYU_CACHE)
    print(f"  Saved to {LEYU_CACHE}")


# fast training eval by early stopping
if 0 < EVAL_SUBSET_SIZE < len(val_dataset):
    train_time_eval_dataset = val_dataset.shuffle(seed=SPLIT_SEED).select(range(EVAL_SUBSET_SIZE))
    print(f"\nUsing a {EVAL_SUBSET_SIZE}-example subsample of validation "
          f"(full val set has {len(val_dataset):,}) for in-training eval_steps checkpoints.")
else:
    train_time_eval_dataset = val_dataset
    print(f"\nEVAL_SUBSET_SIZE disabled or >= full val set — using all "
          f"{len(val_dataset):,} validation examples for in-training eval.")


# Data collator
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


# Metrics
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


# Trainer(PEFT+Whisper)
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


# Model(Phase 1 merged base + fresh LoRA adapter)
print(f"\nLoading Phase 1 merged model from {PHASE1_MERGED} ...")
model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED)
model.config.use_cache          = True
model.config.apply_spec_augment = APPLY_SPEC_AUGMENT

lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = [
        "q_proj", "v_proj", "k_proj", "out_proj",
        "encoder_attn.q_proj", "encoder_attn.v_proj",
        "encoder_attn.k_proj", "encoder_attn.out_proj",
    ],
    bias           = "none",
    inference_mode = False,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.0f}M "
      f"({100*trainable/total:.2f}%)")

forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)
model.generation_config = GenerationConfig(
    forced_decoder_ids     = forced_ids,
    max_new_tokens         = 225,
    repetition_penalty     = 1.2,
    no_repeat_ngram_size   = 7,
    num_beams              = 1,
    use_cache              = True,
    decoder_start_token_id = model.config.decoder_start_token_id,
)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = TRAIN_BATCH_SIZE,
    per_device_eval_batch_size  = EVAL_BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate                = LEARNING_RATE,
    warmup_steps                 = WARMUP_STEPS,
    weight_decay                 = WEIGHT_DECAY,
    label_smoothing_factor       = LABEL_SMOOTHING,
    lr_scheduler_type             = "cosine",

    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = False,

    eval_strategy               = "steps",
    eval_steps                  = EVAL_STEPS,
    save_steps                  = EVAL_STEPS,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,

    predict_with_generate       = True,
    generation_max_length       = 225,

    logging_steps               = 50,
    report_to                   = ["none"],
    dataloader_num_workers      = 4,
    remove_unused_columns       = False,
    ddp_find_unused_parameters  = False,
)

trainer = SafeWhisperTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = train_time_eval_dataset,   
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)


# Train resuming log
print("\nResuming Phase 2: Leyu Dialect Adaptation ...")
print(f"  Dataset cache tag : {DATASET_VERSION}")
print(f"  LoRA              : r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  Learning rate     : {LEARNING_RATE}")
print(f"  use_cache         : {model.config.use_cache} (fixed from False)")
print(f"  In-training eval  : {len(train_time_eval_dataset):,} examples "
      f"(full val set: {len(val_dataset):,})")
print(f"  Resume checkpoint : {RESUME_FROM_CHECKPOINT}")
trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)


# Save adapter + merged model
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "phase2_lora_adapter")
MERGED_DIR  = os.path.join(OUTPUT_DIR, "phase2_final_merged")
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MERGED_DIR,  exist_ok=True)

print("\nSaving LoRA adapter ...")
trainer.model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)

print("Merging LoRA weights into base model ...")
merged = trainer.model.cpu().merge_and_unload()
merged.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)
print(f"  Merged model saved to {MERGED_DIR}")


# Per-dialect evaluation on the FULL held-out test set (unaffected by EVAL_SUBSET_SIZE)
print("\nRunning per-dialect evaluation on the full test set ...")

eval_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR)
eval_model = eval_model.to("cuda").eval()
eval_model.generation_config = GenerationConfig(
    forced_decoder_ids     = forced_ids,
    max_new_tokens         = 225,
    repetition_penalty     = 1.2,
    no_repeat_ngram_size   = 7,
    num_beams              = 5,
    use_cache              = True,
    decoder_start_token_id = eval_model.config.decoder_start_token_id,
)

eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    generation_max_length      = 225,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

eval_trainer = SafeWhisperTrainer(
    model           = eval_model,
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

    result = eval_trainer.predict(dialect_test)
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

overall_test = test_dataset
eval_trainer_overall = SafeWhisperTrainer(
    model           = eval_model,
    args            = eval_args,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)
overall_result = eval_trainer_overall.predict(overall_test)
label_ids = overall_result.label_ids.copy()
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

pred_str  = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(overall_result.predictions, skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in
             processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

import evaluate as _eval
overall_wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
overall_cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

dialect_results["OVERALL"] = {
    "n_samples": len(overall_test),
    "wer":       round(overall_wer, 2),
    "cer":       round(overall_cer, 2),
}

print(f"\n{'='*55}")
print(f"  PHASE 2 (RESUMED) — FINAL RESULTS (beam search, n_beams=5)")
print(f"{'='*55}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect, metrics in dialect_results.items():
    marker = " ←" if dialect == "OVERALL" else ""
    print(f"  {dialect:<18} {metrics['n_samples']:>7}  "
          f"{metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%{marker}")
print(f"{'='*55}")
print(f"  Phase 1 (standard Amharic) best WER  : 27.69%")
print(f"  Previous Phase 2 (smaller corpus) WER: 64.57%")
print(f"  This run at checkpoint-1000 (0.82ep) : 62.97%")
print(f"  Google STT baseline WER              : 35.00%")
print(f"{'='*55}")

results_path = os.path.join(RESULTS_DIR, f"phase2_dialect_results_{DATASET_VERSION}_resumed.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*55}")
print(f"  PHASE 2 (RESUMED) COMPLETE")
print(f"  LoRA adapter  →  {ADAPTER_DIR}")
print(f"  Merged model  →  {MERGED_DIR}")
print(f"  Results       →  {results_path}")
print(f"{'='*55}")