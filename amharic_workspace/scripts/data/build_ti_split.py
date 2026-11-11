"""Builds the text-independent split used by B1 from the partition recorded in
splits/manifests/ti_split_manifest.csv.

The manifest is the record of the split B1 was trained on, so this script does
not re-derive it. It attaches the manifest's partition to the five Leyu train
splits, after checking that both follow the same order. The result is written to
cfg.TI_CACHE with train, validation, and test splits.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import os
import shutil

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
EXPECTED = {"train": 39681, "validation": 4903, "test": 4894}


def key(speaker):
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
            num_proc=4,
        )
        parts.append(ds)
    return concatenate_datasets(parts)


def check_alignment(ds, man):
    """Row i of the Leyu data must be manifest row i, on speaker and normalized text."""
    if len(ds) != len(man):
        raise SystemExit(f"Leyu has {len(ds)} utterances, manifest has {len(man)}")
    speakers = [key(s) for s in ds["speaker_id"]]
    texts = ds["norm_text"]
    bad = [i for i in range(len(ds))
           if speakers[i] != key(man.speaker.iloc[i])
           or texts[i].strip() != str(man.text.iloc[i]).strip()]
    if bad:
        raise SystemExit(f"{len(bad)} rows differ from the manifest; first at index {bad[0]}")


def main():
    final = cfg.TI_CACHE
    if final.exists() and os.environ.get("OVERWRITE") != "1":
        raise SystemExit(f"{final} exists. Set OVERWRITE=1 to rebuild it.")

    man = pd.read_csv(cfg.require(cfg.MANIFESTS / "ti_split_manifest.csv"))
    counts = man.partition.value_counts().to_dict()
    if counts != EXPECTED:
        raise SystemExit(f"manifest partitions {counts} differ from expected {EXPECTED}")

    ds = load_leyu()
    check_alignment(ds, man)

    splits = {name: ds.select(man.index[man.partition == name].tolist()) for name in EXPECTED}

    train_texts = set(splits["train"]["norm_text"])
    shared = sum(t in train_texts for t in splits["test"]["norm_text"])
    if shared:
        raise SystemExit(f"{shared} test utterances have text that also occurs in train")
    train_spk = set(splits["train"]["speaker_id"])
    test_spk = set(splits["test"]["speaker_id"])
    print({k: len(v) for k, v in splits.items()})
    print(f"test speakers also in train: {len(test_spk & train_spk)} of {len(test_spk)}")

    # Write to a temporary folder, then rename, so an interrupted write
    # cannot leave a half-written cache at the final path.
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    DatasetDict(splits).save_to_disk(str(tmp))
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    print(f"saved {final}")


if __name__ == "__main__":
    main()