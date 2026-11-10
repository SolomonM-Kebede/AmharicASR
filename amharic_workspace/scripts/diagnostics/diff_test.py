"""
Check text overlap between ALFFA+CV (training-only data) and Leyu val/test
(held-out). Run before committing GPU time to the joint training run.
"""
import os
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE

import pandas as pd
from datasets import load_dataset, load_from_disk, Dataset as HFDataset, Audio

BASE_DIR = "~/AmharicASR/amharic_workspace"

# Point this at whichever Leyu cache you decide to actually train against
LEYU_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_joint_independent")
ALFFA_DATASET = "hadamard-2/alffa-amharic"
LOCAL_CV_DIR = os.environ.get("LOCAL_CV_DIR", os.path.join(BASE_DIR, "cv-corpus-25.0-2026-03-09/am"))

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

print(f"Loading Leyu held-out sets from {LEYU_CACHE}...")
leyu = load_from_disk(LEYU_CACHE)
heldout_texts = set(leyu["validation"]["norm_text"]) | set(leyu["test"]["norm_text"])
print(f"  {len(heldout_texts):,} unique held-out texts")

print("Loading ALFFA...")
alffa = load_dataset(ALFFA_DATASET, split="train")
alffa_texts = {normalize_amharic(t) for t in alffa["transcript"]}
print(f"  {len(alffa_texts):,} unique ALFFA texts")

alffa_overlap = alffa_texts & heldout_texts
print(f"  ALFFA <-> Leyu held-out text overlap: {len(alffa_overlap)} "
      f"({100*len(alffa_overlap)/len(alffa_texts):.2f}% of ALFFA's unique texts)")

cv_texts = set()
for tsv_name in ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]:
    tsv_path = os.path.join(LOCAL_CV_DIR, tsv_name)
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep="\t", usecols=["sentence"], dtype=str)
        cv_texts |= {normalize_amharic(t) for t in df["sentence"].dropna()}
print(f"  {len(cv_texts):,} unique CV texts")

cv_overlap = cv_texts & heldout_texts
print(f"  CV <-> Leyu held-out text overlap: {len(cv_overlap)} "
      f"({100*len(cv_overlap)/len(cv_texts):.2f}% of CV's unique texts)" if cv_texts else "  (no CV texts found)")

print("\n" + "="*60)
print(f"  TOTAL distinct texts leaking from ALFFA+CV into Leyu held-out: "
      f"{len(alffa_overlap | cv_overlap)}")
print("="*60)