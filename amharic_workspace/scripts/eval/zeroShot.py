from data.normalize import normalize_amharic
"""
Amharic ASR — Zero-shot Whisper large-v3 baseline + test-set audio export
============================================================================
Two things, both needed for RQ1 (fine-tuning vs. zero-shot vs. Google STT):

1. Reconstructs the EXACT same test-set speakers used throughout this
   project (same seed=42, same speaker-independent split logic) but keeps
   the raw audio this time — your cached dataset_cache_* only has
   pre-extracted features, not audio, since prepare_dataset() strips it.

2. Runs Whisper large-v3 ZERO-SHOT (no fine-tuning at all — the plain
   openai/whisper-large-v3 checkpoint) on that test set, beam=5 no
   penalty (matching your fine-tuned model's winning decode config, for
   a fair "how much did fine-tuning help" comparison).

3. Exports every test utterance as a 16kHz mono WAV file + a manifest.jsonl
   (path, reference text, dialect, duration) — this feeds the SEPARATE
   google_stt_eval.py script, which can run on any machine with internet
   access, not necessarily this cluster.

Only reproduces the TEST speakers (not train/val) — cheaper and all this
script needs.
"""

import os

CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from common import config as cfg  # must load before transformers/datasets
import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

BASE_DIR   = "~/AmharicASR/amharic_workspace"
PHASE2_DIR = os.path.join(BASE_DIR, "phase2_leyu")
RESULTS_DIR = os.path.join(PHASE2_DIR, "results")
AUDIO_EXPORT_DIR = os.path.join(BASE_DIR, "test_set_audio_export")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_EXPORT_DIR, exist_ok=True)

TEST_SIZE       = float(os.environ.get("TEST_SIZE", 0.10))
SPLIT_SEED      = int(os.environ.get("SPLIT_SEED", 42))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))
ZEROSHOT_MODEL  = os.environ.get("ZEROSHOT_MODEL", "openai/whisper-large-v3")

DIALECT_CONFIGS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect",  "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect",  "shewa"),
    ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]

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


def get_test_speakers(ds_raw, dialect_name, test_size=TEST_SIZE, seed=SPLIT_SEED):
    """Reproduces ONLY the test-speaker assignment from the original
    speaker_independent_dialect_split — same seed, same stratified split,
    same first step — so this is the identical test set, without also
    computing the train/val split we don't need here."""
    from sklearn.model_selection import train_test_split as sk_split

    meta = ds_raw.remove_columns(
        [c for c in ds_raw.column_names if c not in ("speaker_id", "gender")]
    ).to_pandas()
    spk_meta = meta.drop_duplicates(subset="speaker_id").reset_index(drop=True)
    spk_meta["gender"] = spk_meta["gender"].fillna("unknown")

    all_spk    = spk_meta["speaker_id"].tolist()
    all_gender = spk_meta["gender"].tolist()

    try:
        _, test_spk = sk_split(all_spk, test_size=test_size, random_state=seed, stratify=all_gender)
    except ValueError:
        print(f"    [{dialect_name}] gender-stratified split not feasible — plain speaker split.")
        _, test_spk = sk_split(all_spk, test_size=test_size, random_state=seed)

    test_spk_set = set(test_spk)
    print(f"    [{dialect_name}] test speakers: {len(test_spk_set)}")
    return ds_raw.filter(lambda x: x["speaker_id"] in test_spk_set)


print("\nReconstructing test-set speakers (with raw audio) per dialect ...")
per_dialect_raw = {}
for repo_id, dialect_name in DIALECT_CONFIGS:
    print(f"  Fetching {dialect_name} ({repo_id}) ...")
    ds_raw = load_dataset(repo_id, split="train", verification_mode="no_checks")
    ds_raw = ds_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    ds_raw = ds_raw.map(lambda x, d=dialect_name: {"dialect": d})
    dialect_test = get_test_speakers(ds_raw, dialect_name)
    print(f"    → {len(dialect_test):,} test examples")
    per_dialect_raw[dialect_name] = dialect_test


#  Zero-shot Whisper large-v3 evaluation 
print(f"\nLoading zero-shot model {ZEROSHOT_MODEL} (NOT fine-tuned) ...")
processor = WhisperProcessor.from_pretrained(ZEROSHOT_MODEL, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

model = WhisperForConditionalGeneration.from_pretrained(ZEROSHOT_MODEL)
model = model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = NUM_BEAMS
model.generation_config.use_cache = True


def prepare_dataset(batch):
    import librosa
    audio_data = batch["audio"]
    if audio_data.get("bytes") is not None:
        audio_arr, _ = librosa.load(io.BytesIO(audio_data["bytes"]), sr=16_000, mono=True)
    elif audio_data.get("path") is not None:
        audio_arr, _ = librosa.load(audio_data["path"], sr=16_000, mono=True)
    else:
        audio_arr = np.array(audio_data["array"], dtype=np.float32)

    batch["input_features"] = processor.feature_extractor(
        audio_arr, sampling_rate=16_000, return_tensors="pt"
    ).input_features[0]
    batch["labels"] = processor.tokenizer(
        normalize_amharic(str(batch.get("text") or batch.get("sentence") or "")),
        max_length=448, truncation=True,
    ).input_ids
    batch["_audio_array"] = audio_arr  # kept for the WAV export below
    return batch


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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=DECODER_START_TOKEN_ID)


def compute_metrics(pred):
    import evaluate as _eval
    _wer, _cer = _eval.load("wer"), _eval.load("cer")
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]
    return {
        "wer": round(_wer.compute(predictions=pred_str, references=label_str) * 100, 2),
        "cer": round(_cer.compute(predictions=pred_str, references=label_str) * 100, 2),
    }


class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs.pop("decoder_input_ids", None)
        inputs.pop("decoder_inputs_embeds", None)
        inputs.pop("input_ids", None)
        inputs.pop("attention_mask", None)
        inputs.pop("dialect", None)
        inputs.pop("_audio_array", None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch_zeroshot",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    #generation_max_length      = 225,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

trainer = SafeWhisperTrainer(model=model, args=eval_args, data_collator=data_collator, compute_metrics=compute_metrics)

dialect_results = {}
manifest_entries = []
all_pred_str, all_label_str = [], []

for _, dialect_name in DIALECT_CONFIGS:
    raw = per_dialect_raw[dialect_name]
    if len(raw) == 0:
        continue

    print(f"  Decoding audio for {dialect_name} ({len(raw):,} examples) ...", flush=True)
    dialect_audio_dir = os.path.join(AUDIO_EXPORT_DIR, dialect_name.replace(" ", "_"))
    os.makedirs(dialect_audio_dir, exist_ok=True)

    PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 25))
    input_features_list, labels_list, dialect_list = [], [], []
    t_start = time.time()

    for i, row in enumerate(raw):
        processed = prepare_dataset(row)
        input_features_list.append(processed["input_features"])
        labels_list.append(processed["labels"])
        dialect_list.append(dialect_name)

        wav_path = os.path.join(dialect_audio_dir, f"{dialect_name.replace(' ', '_')}_{i:05d}.wav")
        sf.write(wav_path, np.array(processed["_audio_array"], dtype=np.float32), 16_000)
        ref_text = normalize_amharic(str(row.get("text") or row.get("sentence") or ""))
        duration_sec = len(processed["_audio_array"]) / 16_000.0
        manifest_entries.append({
            "path": wav_path, "reference": ref_text, "dialect": dialect_name,
            "duration_sec": round(duration_sec, 2),
        })

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(raw):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta_sec = (len(raw) - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"    [{dialect_name}] {i+1}/{len(raw)}  "
                  f"({rate:.2f} ex/s, {elapsed:.0f}s elapsed, ~{eta_sec:.0f}s left)", flush=True)

    from datasets import Dataset as HFDataset
    eval_ds = HFDataset.from_dict({
        "input_features": input_features_list,
        "labels": labels_list,
        "dialect": dialect_list,
    })
    result = trainer.predict(eval_ds)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(result.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100
    dialect_results[dialect_name] = {"n_samples": len(eval_ds), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"  {dialect_name:15s}  n={len(eval_ds):4d}  WER={wer:.2f}%  CER={cer:.2f}%")
    all_pred_str.extend(pred_str)
    all_label_str.extend(label_str)

import evaluate as _eval
overall_wer = _eval.load("wer").compute(predictions=all_pred_str, references=all_label_str) * 100
overall_cer = _eval.load("cer").compute(predictions=all_pred_str, references=all_label_str) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred_str), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}

print(f"\n{'='*60}\n  ZERO-SHOT {ZEROSHOT_MODEL} — RESULTS\n{'='*60}")
for dialect, m in dialect_results.items():
    print(f"  {dialect:<28} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%")

results_path = os.path.join(RESULTS_DIR, "zeroshot_whisper_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nZero-shot results saved to {results_path}")

manifest_path = os.path.join(AUDIO_EXPORT_DIR, "manifest.jsonl")
with open(manifest_path, "w", encoding="utf-8") as f:
    for entry in manifest_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
print(f"Exported {len(manifest_entries):,} WAV files + manifest to {AUDIO_EXPORT_DIR}")
print(f"Manifest: {manifest_path}")
print("\nNext: run google_stt_eval.py against this manifest (can be on a different machine with internet access).")