# Early full fine-tuning script (superseded, not used for reported results).
import os
import io
import random
import json
import zipfile
import shutil
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
import evaluate
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    GenerationConfig
)
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

print(f"Imports complete. torch: {torch.__version__}")
N_GPUS = torch.cuda.device_count()
print(f"GPUs available: {N_GPUS} | {torch.cuda.get_device_name(0) if N_GPUS > 0 else 'CPU'}")


# GLOBAL CONFIGURATION & STORAGE BYPASS
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN', '')
os.environ["HF_TOKEN"] = HF_TOKEN  

BASE_DIR = '/tmp/amharic_workspace'
OUTPUT_DIR = os.path.join(BASE_DIR, "whisper-amharic-full-ft")
LOG_DIR = os.path.join(BASE_DIR, "logs-full-ft")
DATA_CACHE = os.path.join(BASE_DIR, "amharic_dataset_cache_full")
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_DIR, "hf_datasets")
os.environ["TENSORBOARD_LOGGING_DIR"] = LOG_DIR

for d in [OUTPUT_DIR, LOG_DIR, DATA_CACHE]:
    os.makedirs(d, exist_ok=True)

MODEL_NAME = "openai/whisper-large-v3"

EPOCHS = 10
BATCH_SIZE = 16
GRAD_ACCUM = 2
LEARNING_RATE = 1e-5  
WARMUP_STEPS = 500
EVAL_STEPS = 200
SAVE_STEPS = 200
MAX_TEXT_CHARS = 350
MAX_LABEL_LENGTH = 448
SEED = 42

LEYU_DATASETS = [
    "leyu-amharic/leyu-amharic-gojjam-dialect",
    "leyu-amharic/leyu-amharic-gonder-dialect",
    "leyu-amharic/leyu-amharic-wello-dialect",
    "leyu-amharic/leyu-amharic-shewa-dialect",
    'gheero-Leyu/leyu-amharic-addis-ababa-dialect'
]
ALFFA_DATASET = "hadamard-2/alffa-amharic"


# NORMALIZATION
AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ":"ሰ", "ሡ":"ሱ", "ሢ":"ሲ", "ሣ":"ሳ", "ሤ":"ሴ", "ሥ":"ስ", "ሦ":"ሶ",
    "ፀ": "ጸ", "ፁ":"ጹ", "ፂ":"ጺ", "ፃ":"ጻ", "ፄ":"ጼ", "ፅ":"ጽ", "ፆ":"ጾ"
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, '') 
    return " ".join(text.split()).strip()


# DATASET PREPARATION 
print(f"Loading {MODEL_NAME} processor...")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="amharic", task="transcribe")

if Path(DATA_CACHE).exists():
    print(f"Found existing cached dataset at {DATA_CACHE}. Loading instantly...")
    from datasets import load_from_disk
    raw_dataset = load_from_disk(DATA_CACHE)
    dataset = raw_dataset
else:
    print("Loading Leyu dialect datasets from HuggingFace...")
    all_dialects = []
    for repo_id in LEYU_DATASETS:
        print(f"  Loading {repo_id}...")
        ds = load_dataset(repo_id, split='train')
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        all_dialects.append(ds)

    print("Loading ALFFA Amharic...")
    alffa = load_dataset(ALFFA_DATASET, split='train')
    alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    alffa = alffa.rename_column('transcript', 'text')
    alffa = alffa.map(lambda x: {'dialect': 'standard', 'gender': 'unknown'})
    alffa = alffa.remove_columns(["utterance_id", "split"])

    combined = concatenate_datasets([*all_dialects, alffa])
    combined = combined.filter(lambda x: len(x['text']) <= MAX_TEXT_CHARS)

    speakers = defaultdict(list)
    for i, row in enumerate(combined):
        speakers[row["speaker_id"]].append(i)

    spk_ids = list(speakers.keys())
    random.Random(SEED).shuffle(spk_ids)
    n = len(spk_ids)

    TEST_RATIO = 0.10
    VAL_RATIO  = 0.10

    n_test = max(1, int(n * TEST_RATIO))
    n_val  = max(1, int(n * VAL_RATIO))

    test_spk  = set(spk_ids[:n_test])
    val_spk   = set(spk_ids[n_test:n_test + n_val])
    train_spk = set(spk_ids[n_test + n_val:])

    train_idx = [i for s, idxs in speakers.items() if s in train_spk for i in idxs]
    val_idx   = [i for s, idxs in speakers.items() if s in val_spk   for i in idxs]
    test_idx  = [i for s, idxs in speakers.items() if s in test_spk  for i in idxs]

    raw_dataset = DatasetDict({
        "train":      combined.select(train_idx),
        "validation": combined.select(val_idx),
        "test":       combined.select(test_idx),
    })

    print(f"\n Data Splits Cleaned: Train={len(raw_dataset['train']):,} | Val={len(raw_dataset['validation']):,} | Test={len(raw_dataset['test']):,}")

    def prepare_batch(batch):
        audio_data = batch["audio"]
        if isinstance(audio_data, dict) and "bytes" in audio_data:
            audio_array, sr = sf.read(io.BytesIO(audio_data["bytes"]))
            if audio_array.ndim > 1: 
                audio_array = audio_array.mean(axis=1)
            if sr != 16_000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16_000)
        else:
            audio_array = audio_data["array"]

        batch["input_features"] = processor.feature_extractor(
            audio_array, sampling_rate=16_000, return_tensors="pt"
        ).input_features[0]

        batch["labels"] = processor.tokenizer(
            normalize_amharic(batch["text"]), max_length=MAX_LABEL_LENGTH, truncation=True
        ).input_ids

        return batch

    print(" Extracting log-Mel features...")
    dataset = raw_dataset.map(
        prepare_batch,
        remove_columns=["text", "audio", "dialect", "speaker_id", "gender"],
        num_proc=2
    )
    dataset.save_to_disk(DATA_CACHE)


# COLLATOR & METRICS
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
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

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    return {
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str) * 100, 2),
    }


# MODEL & TRAINING
print("Loading model for Full Fine-Tuning...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# THE LARGE-V3 SURVIVAL PROTOCOL 
model.config.use_cache = False
model.config.apply_spec_augment = False

# Freeze the massive 32-layer Encoder. 
# Medium could survive unthawed, but Large-v3 shatters in bf16 mixed precision.
model.freeze_encoder()

# Evaluation guardrails
forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config = GenerationConfig(
    forced_decoder_ids=forced_decoder_ids,
    max_new_tokens=225,
    repetition_penalty=1.2,
    no_repeat_ngram_size=5,
    use_cache=True,
    decoder_start_token_id=model.config.decoder_start_token_id
)

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=max(2, BATCH_SIZE // 2),
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    
    bf16=True,
    fp16=False,
    gradient_checkpointing=False, 
    
    # 2. Optimizer Stabilizers for Massive Models 
    adam_beta1=0.9,
    adam_beta2=0.98,       # standard is 0.999 (too fragile for 1.5B parameters)
    adam_epsilon=1e-6,     # standard is 1e-8 (causes div-by-zero in bf16)
    max_grad_norm=1.0,     # strictly clip exploding gradients

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    logging_steps=25,
    report_to=["none"],
    dataloader_num_workers=4,
)

# CASTING FIX FOR EVALUATION 
class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        if "input_features" in inputs:
            # Force the inputs to match the model's exact mixed-precision dtype (bf16)
            target_dtype = next(model.parameters()).dtype
            inputs["input_features"] = inputs["input_features"].to(dtype=target_dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

trainer = SafeWhisperTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

print(f"\n🚀 Launching Full Fine-Tune on {MODEL_NAME}...")
train_result = trainer.train()

trainer.save_model()
processor.save_pretrained(OUTPUT_DIR)
trainer.save_state()


# MANUAL EVALUATION ON HELD-OUT TEST SET
print("Evaluating on perfectly held-out test set...")
results = trainer.predict(dataset["test"])

pred_ids  = results.predictions
label_ids = results.label_ids
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

test_wer = wer_metric.compute(predictions=pred_str, references=label_str) * 100
test_cer = cer_metric.compute(predictions=pred_str, references=label_str) * 100

print(f"\n{'='*50}")
print(f"  Final Test WER : {test_wer:.2f}%")
print(f"  Final Test CER : {test_cer:.2f}%")
print(f"  Google STT baseline : ~35% WER")
print(f"{'='*50}")

# PLOTTING TRAINING CURVES
print("\n Plotting training curves...")
log_path = Path(OUTPUT_DIR) / "trainer_state.json"
if log_path.exists():
    with open(log_path) as f:
        trainer_state = json.load(f)

    logs = trainer_state["log_history"]
    train_logs = [l for l in logs if "loss" in l and "eval_loss" not in l]
    eval_logs  = [l for l in logs if "eval_loss" in l]

    train_steps = [l["step"]      for l in train_logs]
    train_loss  = [l["loss"]      for l in train_logs]
    eval_steps  = [l["step"]      for l in eval_logs]
    eval_loss   = [l["eval_loss"] for l in eval_logs]
    eval_wer    = [l["eval_wer"]  for l in eval_logs if "eval_wer" in l]
    eval_cer    = [l["eval_cer"]  for l in eval_logs if "eval_cer" in l]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Whisper Amharic Full FT ({MODEL_NAME})", fontsize=14, fontweight="bold")

    ax1.plot(train_steps, train_loss, color="#3266ad", linewidth=2, label="training loss")
    ax1.plot(eval_steps,  eval_loss,  color="#e07b39", linewidth=2, label="validation loss", marker="o", markersize=4)
    if eval_loss:
        ax1.axvline(x=eval_steps[np.argmin(eval_loss)], color="gray", linestyle="--", alpha=0.6, label="best val loss")
    ax1.set_title("Loss curves")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    if eval_wer and eval_cer:
        ax2.plot(eval_steps, eval_wer, color="#9b59b6", linewidth=2, label="WER %", marker="o", markersize=4)
        ax2.plot(eval_steps, eval_cer, color="#27ae60", linewidth=2, label="CER %", marker="s", markersize=4)
        ax2.axhline(y=35, color="#e24b4a", linestyle="--", linewidth=1.5, label="Google STT (~35%)")
        ax2.axhline(y=55, color="#888888", linestyle=":",  linewidth=1.5, label="Whisper no tuning (~55%)")
        best_wer_idx = np.argmin(eval_wer)
        ax2.annotate(f"best: {eval_wer[best_wer_idx]:.1f}%",
                     xy=(eval_steps[best_wer_idx], eval_wer[best_wer_idx]),
                     xytext=(eval_steps[best_wer_idx] + max(100, int(eval_steps[-1]*0.1)), eval_wer[best_wer_idx] + 5),
                     arrowprops=dict(arrowstyle="->", color="#9b59b6"),
                     fontsize=9, color="#9b59b6")
        
    ax2.set_title("WER and CER")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Error Rate (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = Path(OUTPUT_DIR) / "training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f" Saved plot to {plot_path}")
else:
    print(f" Could not find {log_path} to plot curves.")


# EXPORTING FINAL MODEL
zip_path = os.path.join(BASE_DIR, "whisper-amharic-model.zip")

if Path(zip_path).exists():
    if Path(zip_path).is_dir():
        shutil.rmtree(zip_path)
    else:
        os.remove(zip_path)

print("\n Zipping model for extraction (excluding massive checkpoints)...")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in Path(OUTPUT_DIR).rglob("*"):
        if f.is_file() and "checkpoint" not in str(f):
            zf.write(f, f.relative_to(OUTPUT_DIR))

size_mb = Path(zip_path).stat().st_size / 1e6
print(f" Zipped: {size_mb:.0f} MB")
print(f"   Saved to: {zip_path}")

print("\n Zip Contents:")
with zipfile.ZipFile(zip_path, "r") as zf:
    for name in zf.namelist():
        size = zf.getinfo(name).file_size / 1e6
        print(f"   {name:50s} {size:.1f} MB")