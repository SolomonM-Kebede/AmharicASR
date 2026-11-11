from data.normalize import normalize_amharic
import os
import json
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk
from transformers import WhisperTokenizer

BASE_DIR = "~/AmharicASR/amharic_workspace"
CACHE_PATH = f"{BASE_DIR}/phase2_leyu/dataset_cache_2026-08-expanded-spkindep"
LOCAL_MODEL_PATH = f"{BASE_DIR}/phase1_curriculum/models/phase1_best_model_merged"

print(f"Loading tokenizer locally from: {LOCAL_MODEL_PATH}")
# Bypasses the internet and the SLURM permission error entirely
tokenizer = WhisperTokenizer.from_pretrained(LOCAL_MODEL_PATH)

print(f"Loading cached dataset from {CACHE_PATH}...")
ds = load_from_disk(CACHE_PATH)

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

print("\nDecoding cached test set (this takes a few seconds)...")
test_texts = set()
for row in ds["test"]:
    ids = [t for t in row["labels"] if t != -100]
    test_texts.add(normalize_amharic(tokenizer.decode(ids, skip_special_tokens=True)))

print(f"Cached Test Set unique texts: {len(test_texts)}")

print("\nChecking Zero-Shot exported manifest...")
MANIFEST_PATH = f"{BASE_DIR}/test_set_audio_export/manifest.jsonl"
manifest_texts = set()
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        manifest_texts.add(normalize_amharic(json.loads(line)["reference"]))

print(f"Exported Manifest unique texts : {len(manifest_texts)}")
print(f"Cache Test ∩ Manifest Test     : {len(test_texts & manifest_texts)}")