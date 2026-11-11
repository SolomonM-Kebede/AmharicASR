
"""
Amharic ASR — Phase 1 HPO Training
Dataset : ALFFA + Common Voice (standard dialect)
Model   : openai/whisper-large-v3 + LoRA
Hardware: 1× H200 (150 GB) — sequential HPO trials
"""

import os
import random
import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import matplotlib

# Config sets the Hugging Face cache and offline flags. It must load before transformers
from common import config as cfg  

matplotlib.use("Agg")

from datasets import (
    load_dataset, concatenate_datasets,
    DatasetDict, Dataset, Audio, load_from_disk,
)
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from data.normalize import normalize_amharic    
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


# Sanity check
print(f"torch  : {torch.__version__}")
N_GPUS = torch.cuda.device_count()
if N_GPUS == 0:
    raise RuntimeError("No GPU found; aborting.")
print(f"GPUs   : {N_GPUS}  ({torch.cuda.get_device_name(0)})")
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM   : {VRAM_GB:.0f} GB")


# Paths and settings come from config (override with AMHARIC_* in env.amh.sh)
PHASE1_DIR   = cfg.DATA_DIR / "phase1_curriculum"
OUTPUT_DIR   = PHASE1_DIR / "models"
LOG_DIR      = PHASE1_DIR / "logs"
DATA_CACHE   = PHASE1_DIR / "dataset_cache"

for d in (OUTPUT_DIR, LOG_DIR, DATA_CACHE):
    d.mkdir(parents=True, exist_ok=True)

MODEL_NAME   = cfg.WHISPER_MODEL
LOCAL_CV_DIR = cfg.require(cfg.CV_DIR)

MAX_TEXT_CHARS   = 350
MAX_LABEL_LENGTH = cfg.MAX_LABEL_LENGTH
SEED             = cfg.SEED

HPO_N_TRIALS   = 6
HPO_MAX_EPOCHS = 3
HPO_EVAL_STEPS = 200
HPO_BATCH_SIZE = 16
HPO_GRAD_ACCUM = 2

FULL_MAX_EPOCHS = 10
FULL_EVAL_STEPS = 300

LORA_R              = 64
LORA_ALPHA          = 128
LORA_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "out_proj",
    "encoder_attn.q_proj", "encoder_attn.v_proj",
    "encoder_attn.k_proj", "encoder_attn.out_proj",
]

# Dataset preparation
print(f"\nLoading processor from {MODEL_NAME} ...")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language="amharic", task="transcribe"
)
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
print(f"Decoder start token id: {DECODER_START_TOKEN_ID}")

cache_ready = Path(DATA_CACHE).exists() and any(Path(DATA_CACHE).iterdir())

if cache_ready:
    print(f"\nCache found — loading from {DATA_CACHE}")
    dataset = load_from_disk(DATA_CACHE)
    print(f"  train={len(dataset['train']):,}  val={len(dataset['validation']):,}  test={len(dataset['test']):,}")
else:
    print("\nBuilding dataset from scratch ...")

    # ALFFA
    print("  Loading ALFFA ...")
    alffa = load_dataset("hadamard-2/alffa-amharic", split="train")
    alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    alffa = alffa.rename_column("transcript", "text")
    alffa = alffa.map(
        lambda x, i: {"dialect": "standard", "gender": "unknown", "speaker_id": f"alffa_{i}"},
        with_indices=True,
    )
    alffa = alffa.remove_columns([c for c in alffa.column_names
                                  if c not in ["audio", "text", "dialect", "gender", "speaker_id"]])

    # Common Voice 
    print("  Loading Common Voice ...")
    cv_tsv = os.path.join(LOCAL_CV_DIR, "train.tsv")
    if not os.path.exists(cv_tsv):
        raise FileNotFoundError(f"Common Voice TSV not found at {cv_tsv}")

    cv_df = pd.read_csv(cv_tsv, sep="\t")[["client_id", "path", "sentence", "gender"]]
    cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
    cv_df["dialect"] = "standard"
    cv_df["gender"]  = cv_df["gender"].fillna("unknown")

    sample_bare = os.path.join(LOCAL_CV_DIR, "clips", cv_df["path"].iloc[0])
    sample_direct = os.path.join(LOCAL_CV_DIR, cv_df["path"].iloc[0])
    if os.path.exists(sample_bare):
        cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(LOCAL_CV_DIR, "clips", p))
    elif os.path.exists(sample_direct):
        cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(LOCAL_CV_DIR, p))
    else:
        raise FileNotFoundError(f"Cannot resolve CV audio path.")
    
    cv_df = cv_df.drop(columns=["path"])
    cv_dataset = Dataset.from_dict(cv_df.to_dict("list"))
    cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    # Combine & filter
    combined = concatenate_datasets([alffa, cv_dataset])
    combined = combined.filter(lambda x: x["text"] and len(x["text"]) <= MAX_TEXT_CHARS)

    # Speaker-disjoint split
    speakers = defaultdict(list)
    for i, row in enumerate(combined):
        speakers[row["speaker_id"]].append(i)

    spk_ids = list(speakers.keys())
    random.Random(SEED).shuffle(spk_ids)
    n = len(spk_ids)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.10))

    test_spk  = set(spk_ids[:n_test])
    val_spk   = set(spk_ids[n_test:n_test + n_val])
    train_spk = set(spk_ids[n_test + n_val:])

    train_idx = [i for s, ii in speakers.items() if s in train_spk for i in ii]
    val_idx   = [i for s, ii in speakers.items() if s in val_spk   for i in ii]
    test_idx  = [i for s, ii in speakers.items() if s in test_spk  for i in ii]

    raw = DatasetDict({
        "train":      combined.select(train_idx),
        "validation": combined.select(val_idx),
        "test":       combined.select(test_idx),
    })

    # Feature extraction 
    def prepare_batch(batch):
        import librosa, io as _io

        audio_data = batch["audio"]
        try:
            if audio_data.get("bytes") is not None:
                audio_arr, _ = librosa.load(_io.BytesIO(audio_data["bytes"]), sr=16_000, mono=True)
            elif audio_data.get("path") is not None:
                audio_arr, _ = librosa.load(audio_data["path"], sr=16_000, mono=True)
            else:
                audio_arr = np.array(audio_data["array"], dtype=np.float32)
        except Exception as e:
            audio_arr = np.zeros(16_000, dtype=np.float32)

        batch["input_features"] = processor.feature_extractor(
            audio_arr, sampling_rate=16_000, return_tensors="pt"
        ).input_features[0]

        batch["labels"] = processor.tokenizer(
            normalize_amharic(batch["text"]),
            max_length=MAX_LABEL_LENGTH,
            truncation=True,
        ).input_ids

        return batch

    print("  Extracting log-Mel features (num_proc=4) ...")
    dataset = raw.map(
        prepare_batch,
        remove_columns=["text", "audio", "dialect", "speaker_id", "gender"],
        num_proc=4,
    )
    dataset.save_to_disk(DATA_CACHE)


# Data collator & Metrics
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
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

def compute_metrics(pred):
    # Imported strictly locally to prevent Ray worker pickling crashes
    import evaluate as _eval
    _wer = _eval.load("wer")
    _cer = _eval.load("cer")

    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    wer = _wer.compute(predictions=pred_str, references=label_str) * 100
    cer = _cer.compute(predictions=pred_str, references=label_str) * 100
    return {"wer": round(wer, 2), "cer": round(cer, 2)}


# Trainer (Overrides predict_step to stop Whisper collision)
class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels         = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()

        outputs = model(input_features=input_features, labels=labels)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        inputs.pop("decoder_input_ids",      None)
        inputs.pop("decoder_inputs_embeds",  None)
        inputs.pop("input_ids",              None)   
        inputs.pop("attention_mask",         None)   

        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )

        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


# HPO Initialization & Ray Logic
def model_init(trial=None):
    """Cleanly extracts lora_dropout from the HF trial dictionary."""
    current_dropout = 0.05 
    
    if trial is not None and isinstance(trial, dict):
        current_dropout = trial.get("lora_dropout", 0.05)

    print(f"  [model_init] dropout={current_dropout:.4f}")

    base = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    base.config.use_cache          = False
    base.config.apply_spec_augment = False

    lora_cfg = LoraConfig(
        # TaskType removed to stop sequence-generation input_ids clashing
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = current_dropout,
        target_modules = LORA_TARGET_MODULES,
        bias           = "none",
        inference_mode = False,
    )
    model = get_peft_model(base, lora_cfg)
    model.enable_input_require_grads()

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
    return model

def ray_hp_space(trial):
    """Ray Tune native API for Hugging Face hyperparameter injection."""
    return {
        "learning_rate": tune.choice([5e-5, 1e-4, 2e-4, 3e-4]),
        "lora_dropout":  tune.uniform(0.0, 0.15),
    }


# HPO Execution
hpo_training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = HPO_MAX_EPOCHS,
    per_device_train_batch_size = HPO_BATCH_SIZE,
    per_device_eval_batch_size  = HPO_BATCH_SIZE // 2,
    gradient_accumulation_steps = HPO_GRAD_ACCUM,
    learning_rate               = 1e-4,             
    warmup_steps                = 200,
    weight_decay                = 0.01,
    label_smoothing_factor      = 0.1,
    lr_scheduler_type           = "cosine",
    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = False,            
    eval_strategy               = "steps",
    eval_steps                  = HPO_EVAL_STEPS,
    save_steps                  = HPO_EVAL_STEPS,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    predict_with_generate       = True,
    generation_max_length       = 225,              
    logging_steps               = 50,
    report_to                   = ["none"],
    dataloader_num_workers      = 0,                
    remove_unused_columns       = False,
    ddp_find_unused_parameters  = False,
)

hpo_trainer = SafeWhisperTrainer(
    args            = hpo_training_args,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    model_init      = model_init,          
)

# Ray auto-detect the GPUs to prevent allocation conflicts
ray.init(ignore_reinit_error=True)

# max_t set higher as Ray ticks per `eval_step`, not per epoch
scheduler = ASHAScheduler(
    metric           = "eval_wer",
    mode             = "min",
    max_t            = 50,              
    grace_period     = 2,               
    reduction_factor = 2,
)

print(f"\n{'='*60}")
print(f"  Launching HPO: {HPO_N_TRIALS} trials on 1× H200")
print(f"{'='*60}\n")

best_run = hpo_trainer.hyperparameter_search(
    direction              = "minimize",
    backend                = "ray",
    hp_space               = ray_hp_space,
    n_trials               = HPO_N_TRIALS,
    resources_per_trial    = {"cpu": 8, "gpu": 1.0},   
    scheduler              = scheduler,
)

print(f"\n{'='*60}")
print(f"  HPO COMPLETE")
print(f"  Best learning_rate : {best_run.hyperparameters['learning_rate']}")
print(f"  Best lora_dropout  : {best_run.hyperparameters['lora_dropout']:.4f}")
print(f"  Best val WER       : {best_run.objective:.2f}%")
print(f"{'='*60}\n")

best_hp_path = os.path.join(PHASE1_DIR, "best_hyperparameters.json")
with open(best_hp_path, "w") as f:
    json.dump(best_run.hyperparameters, f, indent=2)


# Full training 
print("\nStarting full training with best hyperparameters ...")

BEST_OUTPUT = os.path.join(OUTPUT_DIR, "phase1_best_model")
os.makedirs(BEST_OUTPUT, exist_ok=True)

full_training_args = Seq2SeqTrainingArguments(
    output_dir                  = BEST_OUTPUT,
    num_train_epochs            = FULL_MAX_EPOCHS,
    per_device_train_batch_size = HPO_BATCH_SIZE,
    per_device_eval_batch_size  = HPO_BATCH_SIZE // 2,
    gradient_accumulation_steps = HPO_GRAD_ACCUM,
    learning_rate               = best_run.hyperparameters["learning_rate"],
    #lora_dropout                = best_run.hyperparameters["lora_dropout"],
    warmup_steps                = 500,
    weight_decay                = 0.01,
    label_smoothing_factor      = 0.1,
    lr_scheduler_type           = "cosine",
    bf16                        = True,
    fp16                        = False,
    gradient_checkpointing      = False,
    eval_strategy               = "steps",
    eval_steps                  = FULL_EVAL_STEPS,
    save_steps                  = FULL_EVAL_STEPS,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    predict_with_generate       = True,
    generation_max_length       = 225,
    logging_steps               = 25,
    report_to                   = ["none"],
    dataloader_num_workers      = 4,        
    remove_unused_columns       = False,
    ddp_find_unused_parameters  = False,
)

# Cleanly pass the trial dict to initialize the final model
best_trial_dict = {"lora_dropout": best_run.hyperparameters["lora_dropout"]}

full_trainer = SafeWhisperTrainer(
    model           = model_init(trial=best_trial_dict),           
    args            = full_training_args,
    train_dataset   = dataset["train"],
    eval_dataset    = dataset["validation"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
)

full_trainer.train()


# Save LoRA adapter + merged model
print("\nSaving LoRA adapter ...")
full_trainer.model.save_pretrained(BEST_OUTPUT)
processor.save_pretrained(BEST_OUTPUT)

print("Merging LoRA weights into base model ...")
merged_dir = BEST_OUTPUT + "_merged"
os.makedirs(merged_dir, exist_ok=True)

merged_model = full_trainer.model.cpu().merge_and_unload()
merged_model.save_pretrained(merged_dir)
processor.save_pretrained(merged_dir)

print(f"\n{'='*60}")
print(f"  Phase 1 complete.")
print(f"  LoRA adapter : {BEST_OUTPUT}")
print(f"  Merged model : {merged_dir}")
print(f"{'='*60}")