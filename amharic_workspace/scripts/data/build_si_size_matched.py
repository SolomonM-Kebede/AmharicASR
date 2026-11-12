"""Builds the D1 control: a speaker-independent (SI) training sample, size-matched per
dialect to the Joint training set (C1).

The pool is every utterance whose speaker is not in the Joint validation or test sets
(the SI training pool). For each dialect, D1 draws as many utterances as the Joint
training set has for that dialect, with random_state=SEED. Unlike the Joint split, D1
does not remove training texts that occur in validation or test, so its texts overlap
those sets by design. It is the control for the text-independence effect.

Writes:
  cfg.D1_CACHE                          train (the sample), validation and test (Joint)
  splits/manifests/d1_sample_manifest.csv  the sampled utterances, in sample order

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import os
import shutil
from collections import Counter

import pandas as pd
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from data.normalize import normalize_amharic

SEED = cfg.SEED
OUT_CACHE = cfg.D1_CACHE
MANIFEST = cfg.MANIFESTS / "d1_sample_manifest.csv"
COUNTS = cfg.SPLITS / "joint_split_counts.json"

# Same order as build_joint_split.py, so the pool has the same row order.
DIALECTS = [
    ("gojjam", "leyu-amharic/leyu-amharic-gojjam-dialect"),
    ("gonder", "leyu-amharic/leyu-amharic-gonder-dialect"),
    ("wello", "leyu-amharic/leyu-amharic-wello-dialect"),
    ("shewa", "leyu-amharic/leyu-amharic-shewa-dialect"),
    ("addis ababa", "leyu-amharic/leyu-amharic-addis-ababa-dialect"),
]


def namespaced(dialect, speaker):
    """Speaker ID in the Joint cache form, '<dialect>::<id>'."""
    s = str(speaker)
    return s if "::" in s else f"{dialect}::{s}"


def load_dialect(dialect, repo):
    ds = load_dataset(repo, split="train", verification_mode="no_checks")
    text_col = "text" if "text" in ds.column_names else "sentence"
    texts = [str(t or "") for t in ds[text_col]]
    speakers = [namespaced(dialect, s) for s in ds["speaker_id"]]
    gender = list(ds["gender"]) if "gender" in ds.column_names else ["unknown"] * len(ds)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    ds = ds.remove_columns([c for c in ds.column_names if c != "audio"])
    return (ds.add_column("text", texts)
              .add_column("norm_text", [normalize_amharic(t) for t in texts])
              .add_column("dialect", [dialect] * len(ds))
              .add_column("speaker_id", speakers)
              .add_column("gender", gender))


def main():
    if OUT_CACHE.exists() and os.environ.get("OVERWRITE") != "1":
        raise SystemExit(f"{OUT_CACHE} exists. Set OVERWRITE=1 to rebuild it.")

    joint = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    held = set(joint["validation"]["speaker_id"]) | set(joint["test"]["speaker_id"])
    target = Counter(joint["train"]["dialect"])
    expected = json.loads(COUNTS.read_text(encoding="utf-8"))
    expected_pool = sum(c["si_train"] for c in expected.values())
    print(f"held-out speakers (validation + test): {len(held)}")
    print(f"target per dialect (Joint training set): {dict(target)}  total {sum(target.values()):,}")

    combined = concatenate_datasets([load_dialect(d, r) for d, r in DIALECTS])
    meta = pd.DataFrame({"i": range(len(combined)),
                         "spk": combined["speaker_id"],
                         "dialect": combined["dialect"],
                         "norm": combined["norm_text"]})
    pool = meta[~meta.spk.isin(held)]
    if len(pool) != expected_pool:
        raise SystemExit(f"SI pool has {len(pool):,} utterances, joint_split_counts.json "
                         f"says {expected_pool:,}; speaker IDs or the dataset differ")
    if set(pool.dialect) != set(target):
        raise SystemExit("dialect labels in the pool differ from the Joint training set")
    print(f"SI pool: {len(pool):,} utterances, {pool.spk.nunique()} speakers")

    parts = []
    for dialect, _ in DIALECTS:
        need = target[dialect]
        have = (pool.dialect == dialect).sum()
        if need > have:
            raise SystemExit(f"{dialect}: need {need}, pool has {have}")
        parts.append(pool[pool.dialect == dialect].sample(n=need, random_state=SEED))
    sample = pd.concat(parts)
    if sample.i.duplicated().any() or len(sample) != sum(target.values()):
        raise SystemExit("sample is not a set of distinct utterances of the expected size")

    sampled_texts = set(sample.norm)