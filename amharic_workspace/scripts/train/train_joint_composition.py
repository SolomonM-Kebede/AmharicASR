"""
Amharic ASR — Joint Composition Ablation (Leyu joint-independent + ALFFA + CV)
Controlled pair: MODEL_SOURCE=raw uses openai/whisper-large-v3,
                 MODEL_SOURCE=phase1 uses Phase1_merged.
Everything else — split, data composition, LoRA config — held identical.
"""

import os
CUSTOM_CACHE = "/home/stud_homes/s8345520/AmharicASR/hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

import io
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import librosa
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset as HFDataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Model source switch — the ONLY intended difference between the two arms 
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "raw")
assert MODEL_SOURCE in ("raw", "phase1"), "MODEL_SOURCE must be 'raw' or 'phase1'"

BASE_DIR      = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
MODEL_NAME    = "openai/whisper-large-v3" if MODEL_SOURCE == "raw" else PHASE1_MERGED

LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")
ALFFA_DATASET = "hadamard-2/alffa-amharic"
LOCAL_CV_DIR  = os.environ.get("LOCAL_CV_DIR", os.path.join(BASE_DIR, "cv-corpus-25.0-2026-03-09/am"))
OUTPUT_DIR    = os.path.join(BASE_DIR, f"joint_composition_ablation/models/{MODEL_SOURCE}_joint_lora_run")

STANDARD_TO_DIALECT_RATIO = 1.0
SEED = 42
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 800))

# Normalization 
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


# 1. LOAD & COMBINE DATA — before any CUDA context exists
print(f"MODEL_SOURCE = {MODEL_SOURCE}  ->  {MODEL_NAME}")
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
print(f"  forced_ids = {forced_ids}")

print(f"Loading canonical joint-independent Leyu split from {LEYU_CACHE}...")
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
alffa_count = len(alffa)
print(f"  ALFFA: {alffa_count:,} examples")

print("Loading Mozilla Common Voice Amharic (local)...")
cv_frames = []
for tsv_name in ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]:
    tsv_path = os.path.join(LOCAL_CV_DIR, tsv_name)
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", usecols=["client_id", "path", "sentence"], dtype=str)
        cv_frames.append(df)
        print(f"  {tsv_name}: {len(df):,} rows")

mcv_hf = None
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

# trim proportionally together, prevent ALFFA to claim first 
max_standard_allowed = int(leyu_count * STANDARD_TO_DIALECT_RATIO)
cv_count = len(mcv_hf) if mcv_hf is not None else 0
total_standard_available = alffa_count + cv_count

if total_standard_available > max_standard_allowed:
    keep_fraction = max_standard_allowed / total_standard_available
    n_alffa_keep = int(alffa_count * keep_fraction)
    print(f"  Standard data ({total_standard_available:,}) exceeds quota ({max_standard_allowed:,}) "
          f"-- trimming both ALFFA and CV to {keep_fraction:.1%}")
    alffa = alffa.shuffle(seed=SEED).select(range(n_alffa_keep))
    if mcv_hf is not None:
        n_cv_keep = int(cv_count * keep_fraction)
        mcv_hf = mcv_hf.shuffle(seed=SEED).select(range(n_cv_keep))
else:
    print(f"  Standard data ({total_standard_available:,}) within quota ({max_standard_allowed:,}) -- keeping all")

standard_parts = [alffa] + ([mcv_hf] if mcv_hf is not None else [])
standard_combined = concatenate_datasets(standard_parts)
print(f"\nTotal standard (ALFFA+CV) after quota: {len(standard_combined):,} examples")
print(f"Leyu train                            : {leyu_count:,} examples")

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

MAX_TEXT_CHARS = 350   # matches Phase 2b's filter on standard-speech transcripts

print("\nExtracting log-Mel features...")
standard_processed = map_split(standard_combined, "standard (ALFFA+CV)")
n_before = len(standard_processed)
standard_processed = standard_processed.filter(
    lambda x: len(processor.tokenizer.decode(x["labels"])) <= MAX_TEXT_CHARS
)
print(f"  standard (ALFFA+CV): {n_before:,} -> {len(standard_processed):,} "
      f"after MAX_TEXT_CHARS={MAX_TEXT_CHARS} filter (matches Phase 2b)")

leyu_train_processed = map_split(leyu_train, "leyu train")   # Phase 2b does NOT length-filter Leyu
train_dataset = concatenate_datasets([leyu_train_processed, standard_processed])
print(f"Combined train: {len(train_dataset):,}")

val_dataset  = map_split(leyu_val, "validation")
test_dataset = map_split(leyu_test, "test")

if 0 < EVAL_SUBSET_SIZE < len(val_dataset):
    train_time_eval_dataset = val_dataset.shuffle(seed=SEED).select(range(EVAL_SUBSET_SIZE))
    print(f"  Using {EVAL_SUBSET_SIZE}-example subsample for in-training eval")
else:
    train_time_eval_dataset = val_dataset


# 2. LOAD MODEL 
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
print(f"\nStarting joint-composition training [{MODEL_SOURCE}] "
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
open(os.path.join(MERGED_DIR, ".done"), "w").close()
print("Training Complete!")