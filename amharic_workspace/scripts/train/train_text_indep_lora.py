"""
Test independent training script 
"""


# Config sets the Hugging Face cache and offline flags. It must load before transformers and datasets, so it comes first among the project imports.
from common import config as cfg 

import io
from dataclasses import dataclass
from typing import Any, Dict, List

import librosa
import torch
from datasets import Audio, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    EarlyStoppingCallback,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic  # shared normalization

# Paths (from config; override with AMHARIC_* variables in env.amh.sh)
DATASET_DIR   = cfg.require(cfg.DATA_DIR / "phase2_leyu" / "dataset_cache_text_independent")
OUTPUT_DIR    = cfg.DATA_DIR / "phase2_leyu" / "models" / "text_indep_lora_run"
PHASE1_MERGED = cfg.require(cfg.PHASE1_MERGED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading processor...")
processor = WhisperProcessor.from_pretrained(str(PHASE1_MERGED), language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

# PROCESS DATASET FIRST (Safe to use num_proc=4 here)

print(f"Loading text-independent dataset from {DATASET_DIR}...")
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
print("Extracting log-Mel features (this will take a few minutes)...")
processed_ds = raw_ds.map(
    prepare_dataset, 
    remove_columns=cols_to_remove,
    num_proc=4 
)


# LOAD THE MODEL (Wakes up the GPU)
print("Loading Phase 1 merged model into GPU...")
model = WhisperForConditionalGeneration.from_pretrained(
    PHASE1_MERGED,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
model.config.use_cache = False  

# --- Apply LoRA (Exact Phase 2 Config) ---
config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.09,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",
        "encoder_attn.q_proj", "encoder_attn.v_proj",
        "encoder_attn.k_proj", "encoder_attn.out_proj",
    ],
    bias="none",
    inference_mode=False,
)
model = get_peft_model(model, config)
model.enable_input_require_grads()
model.print_trainable_parameters()

#  Collator & Metrics 
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

    wer = _wer.compute(predictions=pred_str, references=label_str) * 100
    cer = _cer.compute(predictions=pred_str, references=label_str) * 100
    return {"wer": round(wer, 2), "cer": round(cer, 2)}

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

# Training Arguments (Exact Phase 2 Config + True Eval Fix) 
training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 10,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,
    learning_rate               = 1e-4,
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
    generation_max_length       = 448, 

    logging_steps               = 50,
    report_to                   = ["none"],
    dataloader_num_workers      = 4,
    remove_unused_columns       = False,
    ddp_find_unused_parameters  = False,
)

trainer = SafeWhisperTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = processed_ds["train"],
    eval_dataset    = processed_ds["validation"],   
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
)

print("Starting LoRA Fine-Tuning on Text-Independent Split...")
trainer.train(resume_from_checkpoint=True)

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

print("Training Complete!")