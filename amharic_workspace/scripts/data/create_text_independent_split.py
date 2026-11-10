import os
import random
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
from sklearn.model_selection import train_test_split

# --- Environment & Config ---
BASE_DIR = "./amharic_workspace"
CACHE_DIR = "./AmharicASR/hf_cache"
OUTPUT_DIR = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_text_independent")

DIALECTS = [
    "leyu-amharic/leyu-amharic-gojjam-dialect",
    "leyu-amharic/leyu-amharic-gonder-dialect",
    "leyu-amharic/leyu-amharic-wello-dialect",
    "leyu-amharic/leyu-amharic-shewa-dialect",
    "leyu-amharic/leyu-amharic-addis-ababa-dialect",
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

print("Loading raw datasets...")
all_parts = []
for repo in DIALECTS:
    dialect_name = repo.split("-")[-2] if "addis" not in repo else "addis ababa"
    ds = load_dataset(repo, split="train", cache_dir=CACHE_DIR)
    ds = ds.map(lambda x: {"dialect": dialect_name, "norm_text": normalize_amharic(str(x.get("text") or x.get("sentence") or ""))} )
    all_parts.append(ds)

combined = concatenate_datasets(all_parts)
print(f"Total Combined Samples: {len(combined)}")

# --- 1. Group by Normalized Text ---
print("Grouping by unique texts...")
text_to_indices = defaultdict(list)
# Grab the entire text column at once (bypasses audio decoding completely)
norm_texts = combined["norm_text"] 
for i, text in enumerate(norm_texts):
    text_to_indices[text].append(i)

unique_texts = list(text_to_indices.keys())
print(f"Total Unique Texts: {len(unique_texts)}")

# --- 2. Split the TEXTS (Not the speakers) ---
# 80% Train, 10% Val, 10% Test
train_texts, temp_texts = train_test_split(unique_texts, test_size=0.20, random_state=42)
val_texts, test_texts = train_test_split(temp_texts, test_size=0.50, random_state=42)

# Convert to sets for O(1) lookup
train_texts = set(train_texts)
val_texts = set(val_texts)
test_texts = set(test_texts)

# --- 3. Map audio indices back to their text split ---
train_idx, val_idx, test_idx = [], [], []
for text, indices in text_to_indices.items():
    if text in train_texts:
        train_idx.extend(indices)
    elif text in val_texts:
        val_idx.extend(indices)
    else:
        test_idx.extend(indices)

text_indep_dataset = DatasetDict({
    "train": combined.select(train_idx),
    "validation": combined.select(val_idx),
    "test": combined.select(test_idx),
})

print("\n--- Split Results ---")
for split in text_indep_dataset:
    num_samples = len(text_indep_dataset[split])
    num_speakers = len(set(text_indep_dataset[split]["speaker_id"]))
    print(f"{split:10s}: {num_samples:5d} samples, {num_speakers:4d} distinct speakers")

# If you want to use this for training, you would run your `prepare_batch` feature extractor here.
# For now, we will just save the raw dataset structure to disk.
text_indep_dataset.save_to_disk(OUTPUT_DIR)
print(f"\nText-independent dataset saved to {OUTPUT_DIR}")