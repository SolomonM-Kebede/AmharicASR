"""
Amharic ASR — RQ3 completion: FULL FINE-TUNE (no LoRA) on the joint
speaker+text-independent split, matched to the best LoRA arm
(Raw+ALFFA+CV, 32.06% WER) for a clean LoRA-vs-full-fine-tune comparison.
"""

import os

CUSTOM_CACHE = "/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import io
from normalize import normalize_amharic

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import (
    load_dataset, concatenate_datasets, DatasetDict, Dataset, Audio, load_from_disk,
)
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

BASE_DIR = "/AmharicASR/amharic_workspace"
MODEL_NAME = "openai/whisper-large-v3"
LOCAL_CV_DIR = os.path.join(BASE_DIR, "cv-corpus-25.0-2026-03-09/am")
JOINT_INDEP_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")

PHASE_DIR   = os.path.join(BASE_DIR, "phase_fullft_joint_indep")
OUTPUT_DIR  = os.path.join(PHASE_DIR, "models")
DATA_CACHE  = os.path.join(PHASE_DIR, "dataset_cache")
RESULTS_DIR = os.path.join(PHASE_DIR, "results")
for d in [OUTPUT_DIR, DATA_CACHE, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_TEXT_CHARS   = 350
MAX_LABEL_LENGTH = 448
SEED             = 42

LEARNING_RATE   = float(os.environ.get("LEARNING_RATE", 5e-6))
OPTIM = os.environ.get("OPTIM", "adamw_bnb_8bit")
WEIGHT_DECAY    = 0.01
WARMUP_STEPS    = 500
EPOCHS          = int(os.environ.get("EPOCHS", 10))
# Default to batch 2 + accum 16 to safely fit full 1.5B AdamW states into 30GB VRAM
TRAIN_BATCH_SIZE= int(os.environ.get("TRAIN_BATCH_SIZE", 2))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 4))
GRAD_ACCUM      = int(os.environ.get("GRAD_ACCUM", 16))   # Effective batch = 32
EVAL_STEPS      = 200
EARLY_STOPPING_PATIENCE = 5
LABEL_SMOOTHING = 0.1
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT", "true").lower() == "true"

# AMHARIC_CHAR_MAP = {
#     "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
#     "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
#     "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
#     "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
#     "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
# }
# PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

# def normalize_amharic(text: str) -> str:
#     for src, tgt in AMHARIC_CHAR_MAP.items():
#         text = text.replace(src, tgt)
#     for p in PUNCT:
#         text = text.replace(p, " ")
#     return " ".join(text.split()).strip()


print(f"\nLoading processor from {MODEL_NAME} ...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="amharic", task="transcribe", cache_dir=CUSTOM_CACHE)
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

def prepare_batch(batch):
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

    batch["input_features"] = processor.feature_extractor(audio_arr, sampling_rate=16_000).input_features[0]
    
    # Handle either norm_text or text seamlessly
    raw_str = batch.get("norm_text") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(normalize_amharic(str(raw_str)), max_length=MAX_LABEL_LENGTH, truncation=True).input_ids
    return batch

cache_ready = Path(DATA_CACHE).exists() and any(Path(DATA_CACHE).iterdir())

if cache_ready:
    print(f"\nCache found — loading from {DATA_CACHE}")
    dataset = load_from_disk(DATA_CACHE)
    print(f"  train={len(dataset['train']):,}  val={len(dataset['validation']):,}  test={len(dataset['test']):,}")
else:
    print(f"\nBuilding composition dataset — joint-indep Leyu + quota-balanced ALFFA+CV ...")

    print("Loading joint-independent Leyu splits ...")
    leyu = load_from_disk(JOINT_INDEP_CACHE)
    leyu = leyu.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    leyu_train, leyu_val, leyu_test = leyu["train"], leyu["validation"], leyu["test"]
    leyu_quota = len(leyu_train)
    print(f"  Leyu train={leyu_quota:,}  val={len(leyu_val):,}  test={len(leyu_test):,}")

    print("Loading ALFFA ...")
    alffa = load_dataset("hadamard-2/alffa-amharic", split="train", cache_dir=CUSTOM_CACHE)
    alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    alffa = alffa.rename_column("transcript", "text")
    alffa = alffa.map(lambda x, i: {"dialect": "standard", "gender": "unknown", "speaker_id": f"alffa_{i}"}, with_indices=True)
    alffa = alffa.remove_columns([c for c in alffa.column_names if c not in ["audio", "text", "dialect", "gender", "speaker_id"]])
    alffa = alffa.filter(lambda x: x["text"] and len(x["text"]) <= MAX_TEXT_CHARS)

    print("Loading Common Voice ...")
    cv_tsv = os.path.join(LOCAL_CV_DIR, "train.tsv")
    cv_df = pd.read_csv(cv_tsv, sep="\t")[["client_id", "path", "sentence", "gender"]]
    cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
    cv_df["dialect"] = "standard"
    cv_df["gender"] = cv_df["gender"].fillna("unknown")
    cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(LOCAL_CV_DIR, "clips", p))
    cv_df = cv_df.drop(columns=["path"])
    cv_dataset = Dataset.from_dict(cv_df.to_dict("list"))
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    cv_dataset = cv_dataset.filter(lambda x: x["text"] and len(x["text"]) <= MAX_TEXT_CHARS)

    standard_combined = concatenate_datasets([alffa, cv_dataset])
    print(f"  ALFFA+CV combined (post-filter): {len(standard_combined):,}, Leyu quota target: {leyu_quota:,}")

    if len(standard_combined) > leyu_quota:
        rng = random.Random(SEED)
        idx = list(range(len(standard_combined)))
        rng.shuffle(idx)
        standard_combined = standard_combined.select(idx[:leyu_quota])
    print(f"  Standard-speech selected: {len(standard_combined):,}")

    keep_cols = {"input_features", "labels", "dialect"}
    
    print("Extracting features for Standard portion (num_proc=4) ...")
    standard_processed = standard_combined.map(
        prepare_batch, 
        remove_columns=[c for c in standard_combined.column_names if c not in keep_cols], 
        num_proc=4
    )

    print("Extracting features for Leyu portions (num_proc=4) ...")
    leyu_train_processed = leyu_train.map(
        prepare_batch, 
        remove_columns=[c for c in leyu_train.column_names if c not in keep_cols], 
        num_proc=4
    )
    leyu_val_processed = leyu_val.map(
        prepare_batch, 
        remove_columns=[c for c in leyu_val.column_names if c not in keep_cols], 
        num_proc=4
    )
    leyu_test_processed = leyu_test.map(
        prepare_batch, 
        remove_columns=[c for c in leyu_test.column_names if c not in keep_cols], 
        num_proc=4
    )

    train_dataset = concatenate_datasets([leyu_train_processed, standard_processed])
    val_dataset   = leyu_val_processed
    test_dataset  = leyu_test_processed

    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})
    dataset.save_to_disk(DATA_CACHE)
    print(f"  Final composition saved to {DATA_CACHE} — train={len(train_dataset):,} (Leyu={len(leyu_train_processed):,} + Standard={len(standard_processed):,})  val={len(val_dataset):,}  test={len(test_dataset):,}")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=DECODER_START_TOKEN_ID)

def compute_metrics(pred):
    import evaluate as _eval
    _wer, _cer = _eval.load("wer"), _eval.load("cer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    return {"wer": round(_wer.compute(predictions=pred_str, references=label_str) * 100, 2),
            "cer": round(_cer.compute(predictions=pred_str, references=label_str) * 100, 2)}

class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()
        outputs = model(input_features=input_features, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

print(f"\nLoading {MODEL_NAME} for FULL fine-tuning (no LoRA) ...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir=CUSTOM_CACHE)
model.config.use_cache = False          # required with gradient checkpointing
model.config.apply_spec_augment = False
model.gradient_checkpointing_enable()

model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = 1
model.generation_config.use_cache = True

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {n_trainable/1e6:.1f}M / {n_params/1e6:.1f}M (100% — full fine-tune)")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    optim=OPTIM,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    label_smoothing_factor=LABEL_SMOOTHING,
    lr_scheduler_type="cosine",
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    logging_steps=25,
    report_to=["none"],
    dataloader_num_workers=4,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
)

trainer = SafeWhisperTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

resume_arg = None
if RESUME_FROM_CHECKPOINT:
    last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_ckpt is not None:
        resume_arg = last_ckpt
        print(f"Found existing checkpoint — resuming from {resume_arg}")
    else:
        print("RESUME_FROM_CHECKPOINT=true but no checkpoint exists yet — starting fresh.")

print(f"\nStarting full fine-tune — LR={LEARNING_RATE}, effective batch={TRAIN_BATCH_SIZE*GRAD_ACCUM}")
trainer.train(resume_from_checkpoint=resume_arg)

FINAL_DIR = os.path.join(OUTPUT_DIR, "fullft_joint_indep_final")
os.makedirs(FINAL_DIR, exist_ok=True)
trainer.save_model(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)
print(f"\nDone. Model saved to {FINAL_DIR}")
Path(FINAL_DIR, ".done").touch()
print("Training complete! Run beam=5 evaluation on the final model against test split.")