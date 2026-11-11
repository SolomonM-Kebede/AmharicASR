"""
Amharic ASR — Single-Phase Joint Training (Leyu + ALFFA + CV) from raw Whisper large-v3
Comparison arm: isolates the effect of the Phase 1 curriculum step by holding
data composition, split methodology, and LoRA config identical to Phase 2b,
varying only the starting checkpoint (raw Whisper vs. Phase1_merged).
"""

# Config sets the Hugging Face cache and offline flags. It must load before
# transformers and datasets, so it comes first among the project imports.
from common import config as cfg  # must load before transformers/datasets

import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
from datasets import (
    Audio, Dataset as HFDataset, concatenate_datasets, load_dataset, load_from_disk,
)
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic  # shared normalization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paths (from config; override with AMHARIC_* variables in env.amh.sh)
BASE_DIR      = str(cfg.REPO_ROOT)
LEYU_CACHE    = str(cfg.require(cfg.LEYU_JOINT_CACHE))
LOCAL_CV_DIR  = str(cfg.require(cfg.CV_DIR))
OUTPUT_DIR    = os.path.join(BASE_DIR, "single_phase_raw_whisper", "models", "joint_lora_run")
MODEL_NAME    = cfg.WHISPER_MODEL
ALFFA_DATASET = "hadamard-2/alffa-amharic"

STANDARD_TO_DIALECT_RATIO = 1.0
SEED = cfg.SEED
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 800))

#  LOAD & COMBINE DATA — before any CUDA context exists
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
print(f"  forced_ids = {forced_ids}")

print(f"Loading canonical Leyu split from {LEYU_CACHE}...")
leyu = load_from_disk(LEYU_CACHE)
leyu_train, leyu_val, leyu_test = leyu["train"], leyu["validation"], leyu["test"]
print(f"  Leyu: train={len(leyu_train):,}  val={len(leyu_val):,}  test={len(leyu_test):,}")
leyu_count = len(leyu_train)

print("Loading ALFFA Amharic...")
alffa = load_dataset(ALFFA_DATASET, split="train")
alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
alffa = alffa.rename_column("transcript", "text")
alffa = alffa.map(lambda x: {"dialect": "standard"})
keep_cols = {"audio", "text", "dialect", "speaker_id"}
alffa = alffa.remove_columns([c for c in alffa.column_names if c not in keep_cols])
if "speaker_id" not in alffa.column_names:
    alffa = alffa.map(lambda x, idx: {"speaker_id": f"alffa_{idx}"}, with_indices=True)
print(f"  ALFFA: {len(alffa):,} examples")

print("Loading Mozilla Common Voice Amharic (local)...")
cv_frames = []
for tsv_name in ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]:
    tsv_path = os.path.join(LOCAL_CV_DIR, tsv_name)
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "path", "sentence"], dtype=str)
        cv_frames.append(df)
        print(f"  {tsv_name}: {len(df):,} rows")

if cv_frames:
    cv_df = pd.concat(cv_frames, ignore_index=True)
    cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
    cv_df["dialect"] = "standard"
    cv_df = cv_df[cv_df["text"].notna() & (cv_df["text"].str.strip() != "")]
    clips_dir = os.path.join(LOCAL_CV_DIR, "clips")
    cv_df["audio_path"] = cv_df["path"].apply(lambda p: os.path.join(clips_dir, p))
    before = len(cv_df)
    cv_df = cv_df[cv_df["audio_path"].apply(os.path.exists)]
    if before - len(cv_df):
        print(f"  Dropped {before - len(cv_df):,} CV rows with missing audio")

    mcv_hf = HFDataset.from_list([
        {"audio": {"path": r["audio_path"], "bytes": None}, "text": r["text"],
         "dialect": r["dialect"], "speaker_id": r["speaker_id"]}
        for _, r in cv_df.iterrows()
    ])
    mcv_hf = mcv_hf.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    print(f"  CV: {len(mcv_hf):,} examples, {cv_df['speaker_id'].nunique():,} speakers")
    
    alffa_count = len(alffa)
    max_standard_allowed = int(leyu_count * STANDARD_TO_DIALECT_RATIO)
    mcv_quota = max(0, max_standard_allowed - alffa_count)
    if len(mcv_hf) > mcv_quota:
        print(f"  Downsampling CV {len(mcv_hf):,} -> {mcv_quota:,} to match dialect volume")
        mcv_hf = mcv_hf.shuffle(seed=SEED).select(range(mcv_quota))
    else:
        print(f"  Keeping all {len(mcv_hf):,} CV examples (under quota of {mcv_quota:,})")
    standard_parts = [alffa, mcv_hf]
else:
    print(f"  No CV TSVs found in {LOCAL_CV_DIR} — proceeding with ALFFA only")
    standard_parts = [alffa]

standard_combined = concatenate_datasets(standard_parts)
print(f"\nTotal standard (ALFFA+CV): {len(standard_combined):,} examples")
print(f"Leyu train                : {leyu_count:,} examples")

# Combine: train = Leyu-train + standard; val/test = Leyu ONLY 
train_dataset = concatenate_datasets([leyu_train, standard_combined])
print(f"Combined train: {len(train_dataset):,}  (val/test remain Leyu-only, unchanged)")

def prepare_dataset(batch):
    audio = batch["audio"]
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            arr, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=16_000, mono=True)
        elif audio.get("path") is not None:
            arr, _ = librosa.load(audio["path"], sr=16_000, mono=True)
        else:
            arr = np.array(audio["array"], dtype=np.float32)
    else:
        arr = np.array(audio, dtype=np.float32)
    batch["input_features"] = processor.feature_extractor(arr, sampling_rate=16_000).input_features[0]
    raw_text = batch.get("norm_text") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(normalize_amharic(str(raw_text)), max_length=448, truncation=True).input_ids
    return batch

keep = {"input_features", "labels", "dialect"}
def map_split(ds, name):
    mapped = ds.map(prepare_dataset, remove_columns=[c for c in ds.column_names if c not in keep], num_proc=4)
    print(f"  {name}: {len(mapped):,} processed")
    return mapped

print("\nExtracting log-Mel features...")
train_dataset = map_split(train_dataset, "train")
val_dataset   = map_split(leyu_val, "validation")
test_dataset  = map_split(leyu_test, "test")

if 0 < EVAL_SUBSET_SIZE < len(val_dataset):
    train_time_eval_dataset = val_dataset.shuffle(seed=SEED).select(range(EVAL_SUBSET_SIZE))
    print(f"  Using {EVAL_SUBSET_SIZE}-example subsample for in-training eval")
else:
    train_time_eval_dataset = val_dataset


# LOAD MODEL — only now
print(f"\nLoading {MODEL_NAME} into GPU...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
model.config.use_cache = True

lora_config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.09,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",
        "encoder_attn.q_proj", "encoder_attn.v_proj",
        "encoder_attn.k_proj", "encoder_attn.out_proj",
    ],
    bias="none", inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None

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

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        for key in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect"]:
            inputs.pop(key, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=200,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    lr_scheduler_type="cosine",
    bf16=True, fp16=False, gradient_checkpointing=False,
    eval_strategy="steps", eval_steps=200, save_steps=200, save_total_limit=3,
    load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
    predict_with_generate=True, generation_max_length=448,
    logging_steps=50, report_to=["none"],
    dataloader_num_workers=4, remove_unused_columns=False, ddp_find_unused_parameters=False,
)

trainer = SafeWhisperTrainer(
    model=model, args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_time_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

ckpt_exists = os.path.isdir(OUTPUT_DIR) and any(f.startswith("checkpoint-") for f in os.listdir(OUTPUT_DIR))
print(f"\nStarting single-phase joint training from raw {MODEL_NAME} "
      f"(train={len(train_dataset):,}, resume={ckpt_exists})...")
trainer.train(resume_from_checkpoint=ckpt_exists)

print("Saving final model...")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora_adapter")
MERGED_DIR = os.path.join(OUTPUT_DIR, "final_merged")
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
trainer.model.save_pretrained(ADAPTER_DIR)
processor.save_pretrained(ADAPTER_DIR)
merged = trainer.model.cpu().merge_and_unload()
merged.save_pretrained(MERGED_DIR)
processor.save_pretrained(MERGED_DIR)
print(f"Merged model saved to {MERGED_DIR}")
print("Training Complete!")
open(os.path.join(MERGED_DIR, ".done"), "w").close()