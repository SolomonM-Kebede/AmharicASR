"""
Amharic ASR — Phase 2 Leyu Dialect Adaptation
Dataset : Leyu Amharic (Gojjam, Gonder, Wello, Shewa + Addis Ababa)
Model   : Merged Phase 1 model + fresh LoRA adapter

"""

import os
import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
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


# Paths
BASE_DIR   = "/tmp/amharic_workspace"
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
LEYU_CACHE  = os.path.join(PHASE2_DIR, "dataset_cache")
RESULTS_DIR = os.path.join(PHASE2_DIR, "results")

for d in [OUTPUT_DIR, LEYU_CACHE, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


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


# Dataset — with cache & correct splits
DIALECT_CONFIGS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect",  "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect",  "shewa"),
    ("gheero-Leyu/leyu-amharic-addis-ababa-dialect", "addis ababa")
]

cache_ready = Path(LEYU_CACHE).exists() and any(Path(LEYU_CACHE).iterdir())

if cache_ready:
    print(f"\nCache found — loading from {LEYU_CACHE}")
    cached = load_from_disk(LEYU_CACHE)
    train_dataset = cached["train"]
    val_dataset   = cached["validation"]
    test_dataset  = cached["test"]
    print(f"  train={len(train_dataset):,}  val={len(val_dataset):,}  test={len(test_dataset):,}")
else:
    print("\nDownloading Leyu dialect datasets ...")

    train_splits = []
    val_splits   = []
    test_splits  = []

    for repo_id, dialect_name in DIALECT_CONFIGS:
        print(f"  Fetching {dialect_name} ...")
        
        # Always load 'train' since it is the only split existing on HF Hub
        ds_raw = load_dataset(repo_id, split="train")

        # decode=False — defer audio decoding to prepare_dataset worker
        ds_raw = ds_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        
        # Tag each sample with its dialect using default arguments to prevent late-binding bugs
        ds_raw = ds_raw.map(lambda x, d=dialect_name: {"dialect": d})

        # Step 1: Extract a clean, held-out 10% for the final thesis evaluation
        split_test = ds_raw.train_test_split(test_size=0.10, seed=42)
        dialect_test = split_test["test"]
        tmp_train_val = split_test["train"]

        # Step 2: Extract 10% of the total (which is ~11.11% of the remaining 90%) for training validation
        split_val = tmp_train_val.train_test_split(test_size=0.1111, seed=42)
        dialect_train = split_val["train"]
        dialect_val   = split_val["test"]

        # Append to structural arrays
        train_splits.append(dialect_train)
        val_splits.append(dialect_val)
        test_splits.append(dialect_test)

    # Concatenate across all 5 distinct dialect blocks
    raw_train = concatenate_datasets(train_splits)
    raw_val   = concatenate_datasets(val_splits)
    raw_test  = concatenate_datasets(test_splits)

    print(f"\n  Raw sizes — train={len(raw_train):,}  "
          f"val={len(raw_val):,}  test={len(raw_test):,}")

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

        # Leyu datasets use "text" column; guard against any variant
        transcript = batch.get("text") or batch.get("sentence") or ""
        batch["labels"] = processor.tokenizer(
            normalize_amharic(str(transcript)),
            max_length=448,
            truncation=True,
        ).input_ids

        return batch

    # Columns to keep after map (dialect needed for per-dialect eval)
    keep_cols = {"input_features", "labels", "dialect"}

    def map_split(raw, name):
        drop = [c for c in raw.column_names if c not in keep_cols | {"text", "audio"}]
        mapped = raw.map(
            prepare_dataset,
            remove_columns=[c for c in raw.column_names if c not in keep_cols],
            num_proc=4,
        )
        # Drop audio/text after mapping
        drop_after = [c for c in mapped.column_names if c not in keep_cols]
        if drop_after:
            mapped = mapped.remove_columns(drop_after)
        print(f"  {name}: {len(mapped):,} samples processed")
        return mapped

    print("\nExtracting log-Mel features ...")
    train_dataset = map_split(raw_train, "train")
    val_dataset   = map_split(raw_val,   "validation")
    test_dataset  = map_split(raw_test,  "test")

    DatasetDict({
        "train":      train_dataset,
        "validation": val_dataset,
        "test":       test_dataset,
    }).save_to_disk(LEYU_CACHE)
    print(f"  Saved to {LEYU_CACHE}")


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
    """
    compute_loss  — explicit call prevents stray kwargs crashing PEFT forward
    prediction_step — pops input_ids/attention_mask before generate()
    """

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
        inputs.pop("dialect",               None)  # not a model input
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# Model(Phase 1 merged base + fresh LoRA adapter)
print(f"\nLoading Phase 1 merged model from {PHASE1_MERGED} ...")
model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED)
model.config.use_cache          = False
model.config.apply_spec_augment = False

lora_config = LoraConfig(
    r              = 64,
    lora_alpha     = 128,
    lora_dropout   = 0.09,                     # best dropout from Phase 1 HPO
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

# Restore forced decoder IDs(preventing language drift)
forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)
model.generation_config = GenerationConfig(
    forced_decoder_ids     = forced_ids,
    max_new_tokens         = 225,
    repetition_penalty     = 1.2,
    no_repeat_ngram_size   = 7,
    num_beams              = 1,        # greedy 
    use_cache              = True,
    decoder_start_token_id = model.config.decoder_start_token_id,
)


# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 10,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,          # effective batch = 32
    learning_rate               = 1e-4,       # conservative(preserve Phase 1 knowledge)
    warmup_steps                = 200,
    weight_decay                = 0.01,
    label_smoothing_factor      = 0.1,
    lr_scheduler_type           = "cosine",

    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = False,

    eval_strategy               = "steps",
    eval_steps                  = 200,
    save_steps                  = 200,
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
    eval_dataset    = val_dataset,             # validation split(test held out)
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
)


# Train
print("\nStarting Phase 2: Leyu Dialect Adaptation ...")
trainer.train()


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


# Per-dialect evaluation on held-out test set
print("\nRunning per-dialect evaluation on test set ...")

eval_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR)
eval_model = eval_model.to("cuda").eval()
eval_model.generation_config = GenerationConfig(
    forced_decoder_ids     = forced_ids,
    max_new_tokens         = 225,
    repetition_penalty     = 1.2,
    no_repeat_ngram_size   = 7,
    num_beams              = 5,        # beam search for final eval
    use_cache              = True,
    decoder_start_token_id = eval_model.config.decoder_start_token_id,
)

eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch",
    per_device_eval_batch_size = 8,
    predict_with_generate      = True,
    generation_max_length      = 225,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

# Initialize Trainer ONCE to prevent memory leaks
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

    # Use the single eval_trainer instance
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

# Overall test WER
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

# Print summary table
print(f"\n{'='*55}")
print(f"  PHASE 2 FINAL RESULTS (beam search, n_beams=5)")
print(f"{'='*55}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect, metrics in dialect_results.items():
    marker = " ←" if dialect == "OVERALL" else ""
    print(f"  {dialect:<18} {metrics['n_samples']:>7}  "
          f"{metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%{marker}")
print(f"{'='*55}")
print(f"  Phase 1 (standard dialect) best WER : 27.69%")
print(f"  Google STT baseline WER             : 35.00%")
print(f"{'='*55}")

# Save results JSON for thesis
results_path = os.path.join(RESULTS_DIR, "phase2_dialect_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {results_path}")

print(f"\n{'='*55}")
print(f"  PHASE 2 COMPLETE")
print(f"  LoRA adapter  →  {ADAPTER_DIR}")
print(f"  Merged model  →  {MERGED_DIR}")
print(f"  Results       →  {results_path}")
print(f"  Next step     →  run eval_kenlm.py on {MERGED_DIR}")
print(f"{'='*55}")
