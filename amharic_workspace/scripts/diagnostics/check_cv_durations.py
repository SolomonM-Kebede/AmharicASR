"""Duration check for the Common Voice training split used in this thesis.

Reports the clip count, speaker count, and audio hours for train.tsv, which
the Corpora table lists as 553 clips and 0.93 hours, and for the whole release.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # keeps path handling and env setup in one place

import pandas as pd

TRAIN_TSV = cfg.require(cfg.CV_DIR / "train.tsv")
DURATIONS_TSV = cfg.require(cfg.CV_DIR / "clip_durations.tsv")
MS_PER_HOUR = 3_600_000


def hours(duration_ms):
    return duration_ms.sum() / MS_PER_HOUR


train = pd.read_csv(TRAIN_TSV, sep="\t", quoting=3, usecols=["path", "client_id"])
durations = (
    pd.read_csv(DURATIONS_TSV, sep="\t")
    .rename(columns={"clip": "path", "duration[ms]": "duration_ms"})
)

print(f"clip_durations entries: {len(durations):,}"
      f" | duplicated clip names: {durations.path.duplicated().sum():,}")

merged = train.merge(durations, on="path", how="left")
missing = int(merged.duration_ms.isna().sum())

print(f"train.tsv clips: {len(train):,}  speakers: {train.client_id.nunique()}")
print(f"with duration: {len(merged) - missing:,}  missing: {missing:,}")
print(f"train audio: {hours(merged.duration_ms.dropna()):.2f} h (Corpora table: 0.93 h)")
print(f"mean clip length: {merged.duration_ms.mean() / 1000:.1f} s")
print(f"clip_durations.tsv (all clips): {len(durations):,} clips, {hours(durations.duration_ms):.2f} h")