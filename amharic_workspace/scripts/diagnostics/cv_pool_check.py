import os
import pandas as pd
from datasets import Dataset, Audio

# ---- Settings (logged values from the training run) ----
CV_DIR = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/cv-corpus-25.0-2026-03-09/am"
SEED = 42
LEYU_TRAIN = 10_296    # quota: Leyu training count (log)
ALFFA_COUNT = 10_875   # ALFFA examples before quota (log)
SPLIT_FILES = ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]

# ---- 1. Build the Common Voice pool, as the training script does ----
frames = []
for name in SPLIT_FILES:
    df = pd.read_csv(os.path.join(CV_DIR, name), sep="\t", quoting=3)
    print(f"  {name}: {len(df):,} rows")
    frames.append(df)

cv_df = pd.concat(frames, ignore_index=True)
cv_df = cv_df[["client_id", "path", "sentence", "gender"]]
cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
cv_df["gender"] = cv_df["gender"].fillna("unknown")
cv_df["dialect"] = "standard"

# Resolve audio location: clips/ subfolder or the release root
probe = cv_df["path"].iloc[0]
audio_dir = os.path.join(CV_DIR, "clips") if os.path.exists(os.path.join(CV_DIR, "clips", probe)) else CV_DIR
cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(audio_dir, p))

# Drop rows with missing audio, as the training script does
exists = cv_df["audio"].apply(os.path.exists)
print(f"rows with missing audio: {(~exists).sum():,}")
cv_df = cv_df[exists].reset_index(drop=True)

cv_ds = Dataset.from_dict({
    "audio": cv_df["audio"].tolist(),
    "speaker_id": cv_df["speaker_id"].tolist(),
    "text": cv_df["text"].tolist(),
})
cv_ds = cv_ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

pool_paths = [a["path"] for a in cv_ds["audio"]]

# ---- 2. Reproduce the quota sampling ----
total_standard = ALFFA_COUNT + len(cv_ds)
keep_fraction = LEYU_TRAIN / total_standard
n_keep = int(len(cv_ds) * keep_fraction)
sample = cv_ds.shuffle(seed=SEED).select(range(n_keep))
sample_paths = [a["path"] for a in sample["audio"]]

# ---- 3. Report ----
distinct_sample = len(set(sample_paths))
print("\n== Common Voice sample check ==")
print(f"pool rows:           {len(cv_ds):,}     (log: 2,692)")
print(f"pool distinct clips: {len(set(pool_paths)):,}     (expected 1,622)")
print(f"keep fraction:       {keep_fraction:.4f}  (log: 75.9%)")
print(f"sampled rows:        {len(sample_paths):,}     (log: 2,042)")
print(f"distinct in sample:  {distinct_sample:,}")
print(f"repeated rows:       {len(sample_paths) - distinct_sample:,}")