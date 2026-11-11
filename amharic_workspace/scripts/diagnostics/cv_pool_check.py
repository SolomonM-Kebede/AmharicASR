"""Reproduces the Common Voice part of the C2/C3 training sample.

Builds the five-file Common Voice pool used by train_joint_composition.py,
applies the same quota and Hugging Face shuffle, and reports the number of
repeated rows in the sample. Expected values are the ones in Section 4.2:
2,692 pool rows, 1,622 distinct clips, 2,042 sampled rows, 614 repeats.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets
from pathlib import Path
import pandas as pd
from datasets import Audio, Dataset

SEED = cfg.SEED
LEYU_TRAIN = 10_296    # quota: Leyu training count (training log)
ALFFA_COUNT = 10_875   # ALFFA examples before quota (training log)
SPLIT_FILES = ["train.tsv", "dev.tsv", "test.tsv", "validated.tsv", "other.tsv"]


def build_pool():
    """Concatenate the five files exactly as the training script does, then keep
    rows whose audio exists."""
    frames = []
    for name in SPLIT_FILES:
        df = pd.read_csv(cfg.CV_DIR / name, sep="\t", quoting=3)
        print(f"  {name}: {len(df):,} rows")
        frames.append(df)

    pool = pd.concat(frames, ignore_index=True)
    pool = pool[["client_id", "path", "sentence", "gender"]]
    pool = pool.rename(columns={"client_id": "speaker_id", "sentence": "text"})

    clips = cfg.CV_DIR / "clips"
    audio_dir = clips if (clips / pool["path"].iloc[0]).exists() else cfg.CV_DIR
    pool["audio"] = pool["path"].map(lambda p: str(audio_dir / p))

    present = pool["audio"].map(lambda p: Path(p).exists())
    print(f"rows with missing audio: {(~present).sum():,}")
    return pool[present].reset_index(drop=True)


def main():
    pool = build_pool()
    ds = Dataset.from_dict({"audio": pool["audio"].tolist(),
                            "clip": pool["path"].tolist()})
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    keep_fraction = LEYU_TRAIN / (ALFFA_COUNT + len(ds))
    n_keep = int(len(ds) * keep_fraction)
    sample = ds.shuffle(seed=SEED).select(range(n_keep))

    sampled = sample["clip"]
    distinct = len(set(sampled))
    print("\n== Common Voice sample check ==")
    print(f"pool rows:           {len(ds):,}     (expected 2,692)")
    print(f"pool distinct clips: {len(set(ds['clip'])):,}     (expected 1,622)")
    print(f"keep fraction:       {keep_fraction:.4f}  (logged 0.759)")
    print(f"sampled rows:        {len(sampled):,}     (expected 2,042)")
    print(f"distinct in sample:  {distinct:,}")
    print(f"repeated rows:       {len(sampled) - distinct:,}     (expected 614)")


if __name__ == "__main__":
    main()