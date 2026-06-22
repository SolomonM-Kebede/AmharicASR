import os
import io
import json
import random
import pickle
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import evaluate
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio, load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

import ray
from ray import tune
from ray.tune import Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.train import RunConfig, CheckpointConfig, ScalingConfig
from ray.train.torch import TorchTrainer

print('Imports complete')
print(f'torch : {torch.__version__}')
print(f'ray   : {ray.__version__}')
N_GPUS     = torch.cuda.device_count()
GPU_NAMES  = [torch.cuda.get_device_name(i) for i in range(N_GPUS)]
print(f"GPUs available: {N_GPUS}")
for i, name in enumerate(GPU_NAMES):
    vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} — {vram:.0f} GB")

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
RUN_HPO = False  

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ":"ሰ", "ሡ":"ሱ", "ሢ":"ሲ", "ሣ":"ሳ", "ሤ":"ሴ", "ሥ":"ስ", "ሦ":"ሶ",
    "ፀ": "ጸ", "ፁ":"ጹ", "ፂ":"ጺ", "ፃ":"ጻ", "ፄ":"ጼ", "ፅ":"ጽ", "ፆ":"ጾ"
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

# ← FIXED: This is now the ONLY normalizer used across the entire script
def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, ' ')
    return " ".join(text.split()).strip()

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN', '')
BASE_DIR = os.getenv('WORKSPACE_DIR', '/tmp/amharic_workspace')
os.environ["HF_HOME"] = os.path.join(BASE_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(BASE_DIR, "hf_datasets")

STORAGE_PATH = os.path.join(BASE_DIR, 'ray_results')
DATA_CACHE   = os.path.join(BASE_DIR, 'amharic_dataset_cache')
for d in [STORAGE_PATH, DATA_CACHE]: os.makedirs(d, exist_ok=True)

LEYU_DATASETS = [
    'leyu-amharic/leyu-amharic-gojjam-dialect',
    'leyu-amharic/leyu-amharic-gonder-dialect',
    'leyu-amharic/leyu-amharic-wello-dialect',
    'leyu-amharic/leyu-amharic-shewa-dialect',
    'gheero-Leyu/leyu-amharic-addis-ababa-dialect',
]
ALFFA_DATASET        = 'hadamard-2/alffa-amharic'
COMMON_VOICE_DATASET = 'fsicoli/common_voice_22_0'
USE_COMMON_VOICE     = True
AMHARIC_CHAR_MAP_CONFIG = AMHARIC_CHAR_MAP

MODEL_NAME       = 'openai/whisper-large-v3'
MAX_TEXT_CHARS   = 350
MAX_LABEL_LENGTH = 448
LORA_TARGET_MODULES = ['q_proj', 'v_proj']

RUN_MODE = 'lora_32'
RUN_CONFIGS = {
    'lora_32': {
        'use_lora': True, 'lora_r': 16, 'lora_alpha': 32, 'lora_dropout': 0.1,
        'batch_size': 16, 'grad_accum': 1, 'learning_rate': 3e-4, 'grad_ckpt': False,
    },
}
cfg = RUN_CONFIGS[RUN_MODE]
USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT = cfg['use_lora'], cfg['lora_r'], cfg['lora_alpha'], cfg['lora_dropout']
BATCH_SIZE, GRAD_ACCUM, LEARNING_RATE = cfg['batch_size'], cfg['grad_accum'], cfg['learning_rate']

OUTPUT_DIR = os.path.join(BASE_DIR, f'whisper-amharic-{RUN_MODE}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HPO_NUM_TRIALS, HPO_GPUS_PER_TRIAL, HPO_MAX_EPOCHS, HPO_EVAL_STEPS = 16, 1, 3, 100
FULL_NUM_WORKERS, FULL_GPUS_PER_WORKER, FULL_MAX_EPOCHS = N_GPUS, 1, 10
FULL_EVAL_STEPS = 200 
WEIGHT_DECAY, LABEL_SMOOTHING, WARMUP_STEPS, TEST_RATIO, SEED = 0.05, 0.1, 500, 0.10, 42

print(f'\n{"="*52}\n  Amharic Whisper — {RUN_MODE.upper()} run\n{"="*52}')

# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
login(token=HF_TOKEN)
CACHE_FILE = os.path.join(DATA_CACHE, 'amharic_dataset_cache_processed')

def load_and_prepare_dataset():
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='amharic', task='transcribe')
    all_datasets = []
    
    for repo_id in LEYU_DATASETS:
        ds = load_dataset(repo_id, split='train').cast_column('audio', Audio(sampling_rate=16_000, decode=False))
        all_datasets.append(ds)

    alffa = load_dataset(ALFFA_DATASET, split='train').cast_column('audio', Audio(sampling_rate=16_000, decode=False))
    alffa = alffa.rename_column('transcript', 'text').map(lambda x: {'dialect': 'standard', 'gender': 'unknown'})
    alffa = alffa.remove_columns([c for c in alffa.column_names if c not in ['audio', 'text', 'dialect', 'gender', 'speaker_id']])
    all_datasets.append(alffa)

    if USE_COMMON_VOICE:
        try:
            cv = load_dataset('fsicoli/common_voice_22_0', 'am', split='train+validation', trust_remote_code=True)
            cv = cv.cast_column('audio', Audio(sampling_rate=16_000, decode=False)).rename_column('sentence', 'text')
            cv = cv.map(lambda x: {'dialect': 'standard', 'gender': x.get('gender', 'unknown'), 'speaker_id': x.get('client_id', 'unknown')})
            cv = cv.remove_columns([c for c in cv.column_names if c not in ['audio', 'text', 'dialect', 'gender', 'speaker_id']])
            all_datasets.append(cv)
        except Exception as e:
            print(f'⚠️ Skipping Common Voice: {e}')

    combined = concatenate_datasets(all_datasets).filter(lambda x: len(x['text']) <= MAX_TEXT_CHARS)
    
    speakers = defaultdict(list)
    for i, row in enumerate(combined): speakers[row['speaker_id']].append(i)
        
    spk_ids = list(speakers.keys())
    random.Random(SEED).shuffle(spk_ids)
    
    n = len(spk_ids)
    n_test, n_val = max(1, int(n * 0.10)), max(1, int(n * 0.10))
    
    test_spk, val_spk, train_spk = set(spk_ids[:n_test]), set(spk_ids[n_test:n_test + n_val]), set(spk_ids[n_test + n_val:])
    train_idx = [i for s, idxs in speakers.items() if s in train_spk for i in idxs]
    val_idx   = [i for s, idxs in speakers.items() if s in val_spk   for i in idxs]
    test_idx  = [i for s, idxs in speakers.items() if s in test_spk  for i in idxs]

    raw = DatasetDict({'train': combined.select(train_idx), 'validation': combined.select(val_idx), 'test': combined.select(test_idx)})

    def prepare_batch(batch):
        audio_data = batch['audio']
        if isinstance(audio_data, dict) and 'bytes' in audio_data:
            audio_array, _ = sf.read(io.BytesIO(audio_data['bytes']))
            if audio_array.ndim > 1: audio_array = audio_array.mean(axis=1)
            audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array(audio_data['array'], dtype=np.float32)
            
        batch['input_features'] = processor.feature_extractor(audio_array, sampling_rate=16_000, return_tensors='pt').input_features[0]
        # Text is normalized properly with spaces here
        batch['labels'] = processor.tokenizer(normalize_amharic(batch['text']), max_length=MAX_LABEL_LENGTH, truncation=True).input_ids
        return batch

    dataset = raw.map(prepare_batch, remove_columns=['text', 'audio', 'dialect', 'speaker_id', 'gender'], num_proc=1)
    dataset.save_to_disk(CACHE_FILE)
    return dataset, processor

if Path(CACHE_FILE).exists():
    print(f'Found existing cache, loading from: {CACHE_FILE}')
    dataset   = load_from_disk(CACHE_FILE)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language='amharic', task='transcribe')
else:
    dataset, processor = load_and_prepare_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# COLLATOR & METRICS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': f['input_features']} for f in features]
        label_features = [{'input_ids': f['labels']} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')
        labels = self.processor.tokenizer.pad(label_features, return_tensors='pt')
        lids = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch['labels'] = lids
        return batch

wer_metric = evaluate.load('wer')
cer_metric = evaluate.load('cer')

# ══════════════════════════════════════════════════════════════════════════════
# CORE TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def train_whisper(config: dict):
    class RayTuneReportCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics:
                tune.report({  
                    'eval_loss': metrics.get('eval_loss', float('nan')),
                    'eval_wer' : metrics.get('eval_wer',  float('nan')),
                    'eval_cer' : metrics.get('eval_cer',  float('nan')),
                })

    dataset = load_from_disk(config['cache_file'])
    def strip_columns(ds):
        to_remove = [c for c in ds.column_names if c not in {'input_features', 'labels'}]
        return ds.remove_columns(to_remove) if to_remove else ds
    dataset = DatasetDict({split: strip_columns(ds) for split, ds in dataset.items()})

    processor = WhisperProcessor.from_pretrained(config['model_name'], language='amharic', task='transcribe')
    
    @dataclass
    class _Collator:
        processor: object
        decoder_start_token_id: int
        def __call__(self, features):
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
            lids = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
            return {"input_features": batch["input_features"], "labels": lids}
            
    model = WhisperForConditionalGeneration.from_pretrained(config['model_name'])
    dec_start = processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>')

    # ← CRITICAL FIX: Guardrails applied correctly here
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

    model.config.decoder_start_token_id = dec_start
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.apply_spec_augment = True
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    model.config.use_cache = False
    
    model.generation_config.language = 'amharic'
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.decoder_start_token_id = dec_start
    model.generation_config.use_cache = False

    use_lora = config.get('use_lora', False)
    if use_lora:
        lora_config = LoraConfig(
            r=config.get('lora_r', 32), lora_alpha=config.get('lora_alpha', 64),
            target_modules=config.get('lora_target_modules', ["q_proj", "v_proj"]),
            lora_dropout=config.get('lora_dropout', 0.1), bias='none', inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads() 
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.base_model.model.config.dropout = config.get('lora_dropout', 0.1)
    else:
        model.config.use_cache = False
        model.config.dropout = 0.1
        model.gradient_checkpointing_enable()

    collator = _Collator(processor=processor, decoder_start_token_id=dec_start)

    def compute_metrics(pred):
        if not config.get('predict_with_generate', True): return {'wer': float('nan'), 'cer': float('nan')}
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # ← CRITICAL FIX: Now using the properly spaced global normalizer
        p = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
        l = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
        return {
            'wer': round(evaluate.load('wer').compute(predictions=p, references=l) * 100, 2),
            'cer': round(evaluate.load('cer').compute(predictions=p, references=l) * 100, 2),
        }
        
    class SafeWhisperTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            input_features = inputs.pop("input_features")
            outputs = model(input_features=input_features, labels=labels)
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            inputs.pop("decoder_input_ids", None)
            inputs.pop("decoder_inputs_embeds", None)
            inputs.pop("input_ids", None)
            if "input_features" in inputs:
                target_dtype = next(model.parameters()).dtype
                inputs["input_features"] = inputs["input_features"].to(dtype=target_dtype)
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    safe_eval_batch = max(2, config['per_device_batch_size'] // 2)
    predict_gen = config.get('predict_with_generate', True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.get('output_dir', '/tmp/whisper_trial'),
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['per_device_batch_size'],
        per_device_eval_batch_size=safe_eval_batch,
        gradient_accumulation_steps=config.get('grad_accum', 1),
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'],
        lr