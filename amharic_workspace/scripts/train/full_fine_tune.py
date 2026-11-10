"""
Amharic ASR — RQ3: Full fine-tuning vs LoRA
================================================================================
Controlled comparison against the Leyu-only LoRA baseline. Same warm-start 
(Phase 1 merged model), same Leyu-only speaker-independent splits, same 
hyperparameters where applicable — the ONLY thing that changes is: every 
parameter in the model is trainable, not just a LoRA adapter.

Memory implications, addressed proactively:
  - gradient_checkpointing enabled from the start.
  - Small per-device batch, compensated with gradient accumulation to keep
    the same effective batch of 32 as your LoRA runs.
  - Standard adamw_torch optimizer used over 8-bit to ensure stable precision
    on Whisper's tied embedding layers during full parameter updates.
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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

print(f"torch  : {torch.__version__}")
if torch.cuda.device_count() == 0:
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

# ─── Hyperparameters — same as LoRA runs where applicable ───────────────────
LEARNING_RATE    = float(os.environ.get("LEARNING_RATE", 5e-6))  # 5e-6 for Full FT stability
EPOCHS           = int(os.environ.get("EPOCHS", 10))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 4))
EVAL_BATCH_SIZE  = int(os.environ.get("EVAL_BATCH_SIZE", 4))
GRAD_ACCUM       = int(os.environ.get("GRAD_ACCUM", 8))   # 4*8=32
EVAL_STEPS       = int(os.environ.get("EVAL_STEPS", 200))
EVAL_SUBSET_SIZE = int(os.environ.get("EVAL_SUBSET_SIZE", 400)) # Dropped to 400 to save eval time
EARLY_STOPPING_PATIENCE = int(os.environ.get("EARLY_STOPPING_PATIENCE", 8))
RESUME_FROM_CHECKPOINT  = os.environ.get("RESUME_FROM_CHECKPOINT", "true").lower() == "true"
OPTIM = os.environ.get("OPTIM", "adamw_bnb_8bit")

BASE_DIR      = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")

# ── The Two Caches for Apples-to-Apples Path B ──
JOINT_CACHE     = os.path.join(BASE_DIR, "phase2b_leyu_standard/standard_speech_cache_ratio1.0")
LEYU_ONLY_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")

PHASE3_DIR  = os.path.join(BASE_DIR, "phase3_full_finetune")
OUTPUT_DIR  = os.path.join(PHASE3_DIR, "models")
RESULTS_DIR = os.path.join(PHASE3_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

#  IN-MEMORY CONCATENATION
print(f"\nLoading STANDARD training cache from {JOINT_CACHE} ...")
standard_dataset = load_from_disk(JOINT_CACHE)

print(f"Loading LEYU dialect cache from {LEYU_ONLY_CACHE} ...")
leyu_cached = load_from_disk(LEYU_ONLY_CACHE)

print("Concatenating Standard and Leyu training splits in-memory...")
# Combine the two datasets to create the true Phase 2b/Phase 3 Joint Training set
train_dataset = concatenate_datasets([standard_dataset, leyu_cached["train"]])

# SHUFFLE the dataset so the model sees a mix of standard and dialect in every batch!
train_dataset = train_dataset.shuffle(seed=42)

# Use Leyu validation and test sets to track dialect performance
val_dataset  = leyu_cached["validation"]
test_dataset = leyu_cached["test"]

print(f"  train={len(train_dataset):,} (Joint)  val={len(val_dataset):,} (Leyu)  test={len(test_dataset):,} (Leyu)")

if 0 < EVAL_SUBSET_SIZE < len(val_dataset):
    train_time_eval_dataset = val_dataset.shuffle(seed=42).select(range(EVAL_SUBSET_SIZE))
    print(f"Using {EVAL_SUBSET_SIZE:,}-example subsample for in-training eval (from {len(val_dataset):,} total)")
else:
    train_time_eval_dataset = val_dataset

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

print(f"\nLoading base model from {PHASE1_MERGED} for FULL fine-tuning (no LoRA) ...")
model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params    : {total_params/1e6:.0f}M")
print(f"Trainable params: {trainable_params/1e6:.0f}M ({100*trainable_params/total_params:.1f}%) — ALL parameters, no LoRA")

forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config.forced_decoder_ids     = forced_ids
model.generation_config.max_new_tokens         = 225
model.generation_config.num_beams              = 1  # Greedy search for fast train-time eval
model.generation_config.use_cache              = True 
model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
model.generation_config.repetition_penalty     = 1.0
model.generation_config.no_repeat_ngram_size   = 0

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE, # Batched eval during training (may still show padding noise, but loss tells the truth)
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=200,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    lr_scheduler_type="cosine",
    bf16=True, fp16=False,
    optim=OPTIM,
    max_grad_norm=1.0,
    eval_strategy="steps", eval_steps=EVAL_STEPS, save_steps=EVAL_STEPS,
    save_total_limit=3, load_best_model_at_end=True,
    metric_for_best_model="wer", greater_is_better=False,
    predict_with_generate=True,
    logging_steps=50, report_to=["none"],
    dataloader_num_workers=4, remove_unused_columns=False,
)

trainer = SafeWhisperTrainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=train_time_eval_dataset,
    data_collator=data_collator, compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
)

print("\nStarting Phase 3: Full fine-tuning (RQ3) ...")
resume_arg = RESUME_FROM_CHECKPOINT
if resume_arg and get_last_checkpoint(OUTPUT_DIR) is None:
    print(f"RESUME_FROM_CHECKPOINT=true but no checkpoint found in {OUTPUT_DIR} — first run, starting fresh.")
    resume_arg = False

print(f"  Learning rate     : {LEARNING_RATE}")
print(f"  Optimizer         : {OPTIM}")
print(f"  Effective batch   : {TRAIN_BATCH_SIZE * GRAD_ACCUM}")
print(f"  Early stop patience: {EARLY_STOPPING_PATIENCE}")
trainer.train(resume_from_checkpoint=resume_arg)

FINAL_DIR = os.path.join(OUTPUT_DIR, "phase3_full_finetuned_final")
trainer.save_model(FINAL_DIR)
processor.save_pretrained(FINAL_DIR)
print(f"\nModel saved to {FINAL_DIR} (no merge needed — full fine-tune, single set of weights)")

# ── Final eval — same test set, same winning decode config as LoRA runs ────
print("\nRunning per-dialect evaluation on the full test set (beam=5, no penalty) ...")
eval_model = WhisperForConditionalGeneration.from_pretrained(FINAL_DIR).to("cuda").eval()
eval_model.generation_config.forced_decoder_ids     = forced_ids
eval_model.generation_config.max_new_tokens         = 225
eval_model.generation_config.num_beams              = 5  # Beam search for the final eval
eval_model.generation_config.use_cache              = True
eval_model.generation_config.decoder_start_token_id = eval_model.config.decoder_start_token_id
eval_model.generation_config.repetition_penalty     = 1.0
eval_model.generation_config.no_repeat_ngram_size   = 0

eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_phase3",
    per_device_eval_batch_size=1,  # STRICTLY 1 to avoid batch-padding hallucination on final testing!
    predict_with_generate=True, bf16=True, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)
eval_trainer = SafeWhisperTrainer(model=eval_model, args=eval_args, data_collator=data_collator, compute_metrics=compute_metrics)

DIALECT_NAMES = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
dialect_results = {}
all_pred, all_ref = [], []

for d_name in DIALECT_NAMES:
    d_test = test_dataset.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0:
        continue
    result = eval_trainer.predict(d_test)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"  {d_name:15s}  n={len(d_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%")
    all_pred.extend(pred_str)
    all_ref.extend(label_str)

import evaluate as _eval
overall_wer = _eval.load("wer").compute(predictions=all_pred, references=all_ref) * 100
overall_cer = _eval.load("cer").compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}\n  RQ3 — FULL FINE-TUNE — FINAL RESULTS\n{'='*60}")
for d, m in dialect_results.items():
    marker = " ←" if d == "OVERALL" else ""
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%{marker}")
print(f"{'='*60}")
print(f"  Phase 1 (Standard Amharic only)           : 27.69% WER")
print(f"  Phase 2 (LoRA baseline on Joint dataset)  : 15.89% WER (approx)")
print(f"{'='*60}")

results_path = os.path.join(RESULTS_DIR, "phase3_full_finetune_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")