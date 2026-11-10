"""
Amharic ASR — Joint (Speaker + Text) Independent Ablation, Leyu-only
Dataset : dataset_cache_joint_independent (0 speaker overlap, ~99.5%+ text independence)
Model   : Phase 1 merged base + fresh LoRA adapter
"""

import os
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import io
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

BASE_DIR      = "~/AmharicASR/amharic_workspace"
DATASET_DIR   = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")
OUTPUT_DIR    = os.path.join(BASE_DIR, "phase2_leyu/models/joint_indep_lora_run")
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 800))

# ---------------------------------------------------------
# 1. PROCESS DATASET FIRST — before any CUDA context exists (fork safety)
# ---------------------------------------------------------
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
print(f"  forced_ids = {forced_ids}")  # sanity: should show (1, 50334), (2, 50360), (3, 50364)

print(f"Loading joint-independent dataset from {DATASET_DIR}...")
raw_ds = load_from_disk(DATASET_DIR)
raw_ds = raw_ds.cast_column("audio", Audio(decode=False))

def prepare_dataset(batch):
    audio = batch["audio"]
    if audio.get("bytes") is not None:
        arr, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=16_000, mono=True)
    elif audio.get("path") is not None:
        arr, _ = librosa.load(audio["path"], sr=16_000, mono=True)
    else:
        arr = audio["array"]
    batch["input_features"] = processor.feature_extractor(arr, sampling_rate=16_000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["norm_text"], max_length=448, truncation=True).input_ids
    return batch

cols_to_remove = [c for c in raw_ds["train"].column_names if c != "dialect"]
print("Extracting log-Mel features...")
processed_ds = raw_ds.map(prepare_dataset, remove_columns=cols_to_remove, num_proc=4)
print(f"  train={len(processed_ds['train']):,}  val={len(processed_ds['validation']):,}  test={len(processed_ds['test']):,}")

if 0 < EVAL_SUBSET_SIZE < len(processed_ds["validation"]):
    train_time_eval_dataset = processed_ds["validation"].shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
    print(f"  Using {EVAL_SUBSET_SIZE}-example subsample for in-training eval "
          f"(full val: {len(processed_ds['validation']):,})")
else:
    train_time_eval_dataset = processed_ds["validation"]

# ---------------------------------------------------------
# 2. LOAD MODEL — only now, after multiprocessing is done
# ---------------------------------------------------------
print("Loading Phase 1 merged model into GPU...")
model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.bfloat16)
model.config.use_cache = True   # matches Phase 2's fix; do NOT revert to False

config = LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.09,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",
        "encoder_attn.q_proj", "encoder_attn.v_proj",
        "encoder_attn.k_proj", "encoder_attn.out_proj",
    ],
    bias="none", inference_mode=False,
)
model = get_peft_model(model, config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# --- Explicitly force language + task (the gap found in the previous run) ---
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None

# --- Normalization (kept only for compute_metrics decoding, not label prep) ---
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
    train_dataset=processed_ds["train"],
    eval_dataset=train_time_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

ckpt_exists = os.path.isdir(OUTPUT_DIR) and any(f.startswith("checkpoint-") for f in os.listdir(OUTPUT_DIR))
print(f"\nStarting LoRA fine-tuning on joint speaker+text independent split "
      f"(train={len(processed_ds['train']):,}, resume={ckpt_exists})...")
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