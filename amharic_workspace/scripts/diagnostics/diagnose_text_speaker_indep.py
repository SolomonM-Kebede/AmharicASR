from data.normalize import normalize_amharic
"""
Amharic ASR — Joint (Speaker + Text) Independent Split Feasibility Check
Computes, per dialect and overall, how much Leyu training data survives
after enforcing BOTH speaker independence and text independence between
train and val/test. No audio decoding, no GPU — diagnostic pass only.
"""

import os

CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE

from common import config as cfg  # must load before transformers/datasets
from datasets import load_dataset, concatenate_datasets, DatasetDict, Audio
from sklearn.model_selection import train_test_split as sk_split

# --- Config ---
BASE_DIR  = "~/AmharicASR/amharic_workspace"
OUT_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")
TEST_SIZE, VAL_SIZE_OF_REMAINING, SPLIT_SEED = 0.10, 0.1111, 42

DIALECT_CONFIGS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect",  "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect",  "shewa"),
    ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]

# --- Normalization (same map used throughout the project) ---
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


def speaker_independent_dialect_split(ds_raw, dialect_name,
                                       test_size=TEST_SIZE,
                                       val_size_of_remaining=VAL_SIZE_OF_REMAINING,
                                       seed=SPLIT_SEED):
    meta = ds_raw.remove_columns(
        [c for c in ds_raw.column_names if c not in ("speaker_id", "gender")]
    ).to_pandas()
    spk_meta = meta.drop_duplicates(subset="speaker_id").reset_index(drop=True)
    spk_meta["gender"] = spk_meta["gender"].fillna("unknown")

    def _safe_split(ids, strat, size):
        try:
            return sk_split(ids, test_size=size, random_state=seed, stratify=strat)
        except ValueError:
            print(f"    [{dialect_name}] gender-stratified split not feasible — plain speaker split.")
            return sk_split(ids, test_size=size, random_state=seed)

    all_spk, all_gender = spk_meta["speaker_id"].tolist(), spk_meta["gender"].tolist()
    trainval_spk, test_spk = _safe_split(all_spk, all_gender, test_size)
    gender_by_spk = spk_meta.set_index("speaker_id")["gender"]
    trainval_gender = gender_by_spk.loc[trainval_spk].tolist()
    train_spk, val_spk = _safe_split(trainval_spk, trainval_gender, val_size_of_remaining)

    train_spk_set, val_spk_set, test_spk_set = set(train_spk), set(val_spk), set(test_spk)
    print(f"    [{dialect_name}] speakers: train={len(train_spk_set)} "
          f"val={len(val_spk_set)} test={len(test_spk_set)}")

    train_ds = ds_raw.filter(lambda x: x["speaker_id"] in train_spk_set)
    val_ds   = ds_raw.filter(lambda x: x["speaker_id"] in val_spk_set)
    test_ds  = ds_raw.filter(lambda x: x["speaker_id"] in test_spk_set)
    return train_ds, val_ds, test_ds


def apply_text_independence_filter(train_ds, val_ds, test_ds, dialect_name):
    heldout_texts = set(val_ds["norm_text"]) | set(test_ds["norm_text"])
    n_before = len(train_ds)
    train_ds = train_ds.filter(lambda x: x["norm_text"] not in heldout_texts)
    n_after = len(train_ds)
    pct_removed = 100 * (n_before - n_after) / n_before if n_before else 0.0
    print(f"    [{dialect_name}] train: {n_before:,} -> {n_after:,} "
          f"({pct_removed:.1f}% removed for text overlap with held-out speakers)")
    return train_ds, n_before, n_after


# --- Run across all five dialects ---
train_splits, val_splits, test_splits, summary_rows = [], [], [], []
print("Checking joint speaker+text independence feasibility per dialect...\n")

for repo_id, dialect_name in DIALECT_CONFIGS:
    print(f"--- {dialect_name} ({repo_id}) ---")
    ds_raw = load_dataset(repo_id, split="train", verification_mode="no_checks")
    ds_raw = ds_raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    ds_raw = ds_raw.map(
    lambda x: {
        "dialect": dialect_name,
        "norm_text": normalize_amharic(x.get("text") or ""),
        "speaker_id": f"{dialect_name}::{x['speaker_id']}",
    },
    num_proc=4,
    )
    print(f"    {len(ds_raw):,} examples | {len(set(ds_raw['speaker_id']))} unique speakers")

    dialect_train, dialect_val, dialect_test = speaker_independent_dialect_split(ds_raw, dialect_name)
    n_before = len(dialect_train)

    train_splits.append(dialect_train)   # unfiltered — global filter happens after concatenation
    val_splits.append(dialect_val)
    test_splits.append(dialect_test)
    summary_rows.append({"dialect": dialect_name, "train_before": n_before,
                        "val": len(dialect_val), "test": len(dialect_test)}
                        )
    print()

    raw_train_unfiltered = concatenate_datasets(train_splits)
    raw_val  = concatenate_datasets(val_splits)
    raw_test = concatenate_datasets(test_splits)

    global_heldout_texts = set(raw_val["norm_text"]) | set(raw_test["norm_text"])
    raw_train = raw_train_unfiltered.filter(lambda x: x["norm_text"] not in global_heldout_texts)

for row in summary_rows:
    d = row["dialect"]
    row["train_after"] = sum(1 for x in raw_train if x["dialect"] == d)
    row["pct_removed"] = round(100 * (row["train_before"] - row["train_after"]) / row["train_before"], 1)

# Sanity checks — both axes
train_spk, val_spk, test_spk = set(raw_train["speaker_id"]), set(raw_val["speaker_id"]), set(raw_test["speaker_id"])

# --- Diagnostic: are colliding IDs shared across genuinely different dialects? ---
overlap = train_spk & val_spk
dialect_by_id = {}
for x in raw_train:
    if x["speaker_id"] in overlap:
        dialect_by_id.setdefault(x["speaker_id"], set()).add(x["dialect"])
for x in raw_val:
    if x["speaker_id"] in overlap:
        dialect_by_id.setdefault(x["speaker_id"], set()).add(x["dialect"])

cross_dialect = {k: v for k, v in dialect_by_id.items() if len(v) > 1}
print(f"{len(overlap)} overlapping IDs total, {len(cross_dialect)} span more than one dialect")
if cross_dialect:
    # peek at a few to eyeball them
    for spk_id, dialects in list(cross_dialect.items())[:5]:
        print(f"    {spk_id}: {dialects}")

assert not (train_spk & val_spk),  "Speaker leakage between train/val"
assert not (train_spk & test_spk), "Speaker leakage between train/test"
assert not (val_spk & test_spk),   "Speaker leakage between val/test"

train_txt, val_txt, test_txt = set(raw_train["norm_text"]), set(raw_val["norm_text"]), set(raw_test["norm_text"])
text_leak_val, text_leak_test = train_txt & val_txt, train_txt & test_txt


print("=" * 60)
print("  JOINT SPEAKER + TEXT INDEPENDENCE — FEASIBILITY SUMMARY")
print("=" * 60)
print(f"  {'Dialect':<15}{'Before':>9}{'After':>9}{'Removed':>10}{'Val':>7}{'Test':>7}")
print(f"  {'-'*56}")
for row in summary_rows:
    print(f"  {row['dialect']:<15}{row['train_before']:>9,}{row['train_after']:>9,}"
          f"{row['pct_removed']:>9.1f}%{row['val']:>7,}{row['test']:>7,}")
print(f"  {'-'*56}")
total_before = sum(r["train_before"] for r in summary_rows)
total_after  = sum(r["train_after"]  for r in summary_rows)
total_pct    = 100 * (total_before - total_after) / total_before if total_before else 0.0
print(f"  {'TOTAL':<15}{total_before:>9,}{total_after:>9,}{total_pct:>9.1f}%"
      f"{len(raw_val):>7,}{len(raw_test):>7,}")
print("=" * 60)
print(f"  Speaker independence verified: 0 overlapping speakers across all splits")
print(f"  Remaining text leakage — train∩val: {len(text_leak_val)}  train∩test: {len(text_leak_test)}")
print("=" * 60)

DatasetDict({"train": raw_train, "validation": raw_val, "test": raw_test}).save_to_disk(OUT_CACHE)
print(f"\nJoint-independent raw split saved to {OUT_CACHE}")
print("(audio still undecoded — feature extraction happens in the training script, matching the Phase 2 pattern)")