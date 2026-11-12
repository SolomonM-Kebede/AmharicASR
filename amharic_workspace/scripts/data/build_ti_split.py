"""Builds the text-independent (TI) split used by B1 from the partition recorded in
splits/manifests/ti_split_manifest.csv.

The manifest is the record of the split B1 was trained on, so this script does not
re-derive it. It loads the five Leyu train splits, checks row by row that the data and
the manifest agree (speaker, then normalized text), and attaches the manifest's
partition labels. Output: cfg.TI_CACHE (train, validation, test) and
splits/ti_split_counts.json.

What the split guarantees: no test text occurs in training.
What it does not guarantee: speaker independence. Test speakers also appear in training
(273 of 273), so B1 is not a speaker-independent result.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import os
import shutil
from collections import Counter

import pandas as pd
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset

from data.normalize import normalize_amharic

DIALECTS = [
    ("gojjam", "leyu-amharic/leyu-amharic-gojjam-dialect"),
    ("gonder", "leyu-amharic/leyu-amharic-gonder-dialect"),
    ("wello", "leyu-amharic/leyu-amharic-wello-dialect"),
    ("shewa", "leyu-amharic/leyu-amharic-shewa-dialect"),
    ("addis ababa", "leyu-amharic/leyu-amharic-addis-ababa-dialect"),
]
PARTITIONS = ("train", "validation", "test")
EXPECTED = {"train": 39681, "validation": 4903, "test": 4894}
COUNTS_OUT = cfg.SPLITS / "ti_split_counts.json"
NUM_PROC = 4


def speaker_key(speaker):
    """Manifest form of a speaker ID: '<dialect>::<id>' becomes '<dialect>_<id>'."""
    return str(speaker).replace("::", "_").replace(" ", "_")


def load_leyu():
    parts = []
    for dialect, repo in DIALECTS:
        ds = load_dataset(repo, split="train", verification_mode="no_checks")
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        ds = ds.map(
            lambda x, d=dialect: {
                "dialect": d,
                "norm_text": normalize_amharic(x.get("text") or ""),
                "speaker_id": f"{d}::{x['speaker_id']}",
            },
            num_proc=NUM_PROC,
        )
        parts.append(ds)
    return concatenate_datasets(parts)


def check_alignment(ds, man):
    """Row i of the Leyu data must be manifest row i, on speaker and normalized text."""
    if len(ds) != len(man):
        raise SystemExit(f"Leyu has {len(ds)} utterances, manifest has {len(man)}")
    speakers = ds["speaker_id"]
    texts = ds["norm_text"]
    bad = [i for i in range(len(ds))
           if speaker_key(speakers[i]) != speaker_key(man.speaker.iloc[i])
           or texts[i].strip() != str(man.text.iloc[i]).strip()]
    if bad:
        raise SystemExit(f"{len(bad)} rows differ from the manifest; first at index {bad[0]}")


def main():
    final = cfg.TI_CACHE
    if final.exists() and os.environ.get("OVERWRITE") != "1":
        raise SystemExit(f"{final} exists. Set OVERWRITE=1 to rebuild it (back it up first).")

    man = pd.read_csv(cfg.require(cfg.MANIFESTS / "ti_split_manifest.csv"))
    counts = man.partition.value_counts().to_dict()
    if set(counts) != set(PARTITIONS) or counts != EXPECTED:
        raise SystemExit(f"manifest partitions {counts} differ from expected {EXPECTED}")

    ds = load_leyu()
    check_alignment(ds, man)
    splits = {name: ds.select(man.index[man.partition == name].tolist()) for name in PARTITIONS}

    train_texts = set(splits["train"]["norm_text"])
    shared = sum(t in train_texts for t in splits["test"]["norm_text"])
    if shared:
        raise SystemExit(f"{shared} test utterances have text that also occurs in train")

    train_spk = set(splits["train"]["speaker_id"])
    test_spk = set(splits["test"]["speaker_id"])
    report = {
        "partitions": {
            name: {"utterances": len(splits[name]),
                   "per_dialect": dict(Counter(splits[name]["dialect"]))}
            for name in PARTITIONS
        },
        "test_speakers": len(test_spk),
        "test_speakers_in_train": len(test_spk & train_spk),
        "test_utterances_with_text_in_train": shared,
        "guarantee": "text-independent; speakers are not held out",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # Write the cache to a temporary folder, then rename, so an interrupted write
    # cannot leave a half-written cache at the final path.
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    DatasetDict(splits).save_to_disk(str(tmp))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)

    COUNTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    COUNTS_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {final}\ncounts {COUNTS_OUT}")


if __name__ == "__main__":
    main()