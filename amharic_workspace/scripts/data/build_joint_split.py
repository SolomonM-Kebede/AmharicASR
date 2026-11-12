"""Builds the Joint split: speaker-independent within each dialect, then
text-independent across dialects.

Writes phase2_leyu/dataset_cache_joint_independent, which every Joint-split run
reads, and splits/joint_split_counts.json, which holds the per-dialect counts
in Table 2. The cache is not overwritten unless OVERWRITE=1 is set.

Leyu is read from the Hugging Face cache (offline mode, set in config).
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import os
from collections import Counter

from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

from data.normalize import normalize_amharic

OUT_CACHE = cfg.LEYU_JOINT_CACHE
COUNTS_OUT = cfg.SPLITS / "joint_split_counts.json"

DIALECTS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect", "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect", "shewa"),
    ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]
TEST_SIZE = 0.10
# Share of the remaining 90% for validation, so 10% of all speakers.
VAL_SIZE_OF_REMAINING = 0.1111
SEED = cfg.SEED


def split_ids(ids, genders, size, label):
    """Stratify by gender when possible; otherwise split plainly and say so."""
    try:
        return train_test_split(ids, test_size=size, random_state=SEED, stratify=genders)
    except ValueError:
        print(f"  [{label}] gender stratification not feasible; plain split")
        return train_test_split(ids, test_size=size, random_state=SEED)


def speaker_split(ds, label):
    """Train, validation, and test speaker-ID sets for one dialect."""
    spk = ds.select_columns(["speaker_id", "gender"]).to_pandas()
    spk = spk.drop_duplicates("speaker_id").reset_index(drop=True)
    spk["gender"] = spk["gender"].fillna("unknown")
    trainval, test = split_ids(spk.speaker_id.tolist(), spk.gender.tolist(), TEST_SIZE, label)
    gender = spk.set_index("speaker_id").gender
    train, val = split_ids(trainval, gender.loc[trainval].tolist(), VAL_SIZE_OF_REMAINING, label)
    return set(train), set(val), set(test)


def build_dialect(repo_id, dialect):
    ds = load_dataset(repo_id, split="train", verification_mode="no_checks")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    ds = ds.map(
        lambda x: {
            "dialect": dialect,
            "norm_text": normalize_amharic(x.get("text") or ""),
            "speaker_id": f"{dialect}::{x['speaker_id']}",
        },
        num_proc=4,
    )
    tr, va, te = speaker_split(ds, dialect)
    return (ds.filter(lambda x: x["speaker_id"] in tr),
            ds.filter(lambda x: x["speaker_id"] in va),
            ds.filter(lambda x: x["speaker_id"] in te))


def main():
    if OUT_CACHE.exists() and os.environ.get("OVERWRITE") != "1":
        raise SystemExit(f"{OUT_CACHE} exists. Set OVERWRITE=1 to rebuild it.")

    train_parts, val_parts, test_parts, counts = [], [], [], {}
    for repo_id, dialect in DIALECTS:
        print(f"--- {dialect}")
        tr, va, te = build_dialect(repo_id, dialect)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        counts[dialect] = {"si_train": len(tr), "validation": len(va), "test": len(te)}

    si_train = concatenate_datasets(train_parts)
    validation = concatenate_datasets(val_parts)
    test = concatenate_datasets(test_parts)

    # Text independence is global: a text held out in any dialect is removed from training in every dialect.
    held_out = set(validation["norm_text"]) | set(test["norm_text"])
    joint_train = si_train.filter(lambda x: x["norm_text"] not in held_out)

    joint_counts = Counter(joint_train["dialect"])
    for dialect in counts:
        counts[dialect]["joint_train"] = joint_counts.get(dialect, 0)

    # Checks, computed rather than asserted in text
    speakers = {"train": set(joint_train["speaker_id"]),
                "validation": set(validation["speaker_id"]),
                "test": set(test["speaker_id"])}
    overlaps = {"train&validation": len(speakers["train"] & speakers["validation"]),
                "train&test": len(speakers["train"] & speakers["test"]),
                "validation&test": len(speakers["validation"] & speakers["test"])}
    if any(overlaps.values()):
        raise SystemExit(f"speaker overlap between splits: {overlaps}")
    if set(joint_train["norm_text"]) & held_out:
        raise SystemExit("text leakage: a training text occurs in validation or test")

    print(f"\n{'dialect':12s}{'SI train':>10s}{'Joint':>8s}{'removed':>9s}{'val':>7s}{'test':>7s}")
    for dialect, c in counts.items():
        removed = 100 * (1 - c["joint_train"] / c["si_train"])
        print(f"{dialect:12s}{c['si_train']:>10,}{c['joint_train']:>8,}"
              f"{removed:>8.1f}%{c['validation']:>7,}{c['test']:>7,}")
    totals = {k: sum(c[k] for c in counts.values()) for k in ("si_train", "joint_train", "validation", "test")}
    print(f"{'total':12s}{totals['si_train']:>10,}{totals['joint_train']:>8,}"
          f"{'':>9s}{totals['validation']:>7,}{totals['test']:>7,}")

    shared = len(set(validation["norm_text"]) & set(test["norm_text"]))
    print(f"\nspeaker overlap: {overlaps}")
    print(f"distinct normalized prompts shared by validation and test: {shared:,}")

    COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_OUT.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    DatasetDict({"train": joint_train, "validation": validation, "test": test}).save_to_disk(str(OUT_CACHE))
    print(f"saved {OUT_CACHE}\ncounts {COUNTS_OUT}")


if __name__ == "__main__":
    main()