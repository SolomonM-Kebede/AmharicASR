"""Derives the text-independent split (B1) from the Leyu metadata.

Procedure: group utterances by normalized text. Split the unique texts 80/10/10 with
train_test_split(random_state=42), in first-appearance order, and give each utterance
the partition of its text. The partition depends on the dialect order in which the
utterances are concatenated, because that order fixes the first-appearance order of
the texts. The order used for the thesis is DIALECT_ORDER below; a permutation search
found it.

Input: leyu_meta_raw.parquet (columns speaker, text), local and not tracked.
Output: a manifest with speaker, dialect, text, partition. It is written to --out,
which defaults to a scratch name, so the committed manifest is never overwritten by
accident. The result is compared with the committed manifest row by row.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # keep first

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DIALECT_ORDER = ("gojjam", "gonder", "wello", "shewa", "addis ababa")
EXPECTED = {"train": 39681, "validation": 4903, "test": 4894}
COMMITTED = cfg.MANIFESTS / "ti_split_manifest.csv"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=cfg.DATA_DIR / "leyu_meta_raw.parquet")
    ap.add_argument("--out", type=Path, default=cfg.MANIFESTS / "ti_split_manifest.derived.csv")
    ap.add_argument("--overwrite", action="store_true", help="replace --out if it exists")
    return ap.parse_args()


def derive(meta):
    m = meta.copy()
    m["dialect"] = m.speaker.str.rsplit("_", n=1).str[0]
    df = pd.concat([m[m.dialect == d] for d in DIALECT_ORDER], ignore_index=True)
    if len(df) != len(m):
        raise SystemExit(f"{len(m) - len(df)} rows have a dialect outside {DIALECT_ORDER}")

    idx = defaultdict(list)
    for i, t in enumerate(df.text):
        idx[t].append(i)
    texts = list(idx)                                   # first-appearance order
    train, rest = train_test_split(texts, test_size=0.20, random_state=cfg.SEED)
    val, test = train_test_split(rest, test_size=0.50, random_state=cfg.SEED)

    part = {}
    for name, group in (("train", train), ("validation", val), ("test", test)):
        for t in group:
            for i in idx[t]:
                part[i] = name
    df["partition"] = pd.Series(part)
    return df


def check(df):
    counts = df.partition.value_counts().to_dict()
    if counts != EXPECTED:
        raise SystemExit(f"partitions {counts} differ from expected {EXPECTED}")
    T = {p: set(g.text) for p, g in df.groupby("partition")}
    shared = {"train&validation": len(T["train"] & T["validation"]),
              "train&test": len(T["train"] & T["test"]),
              "validation&test": len(T["validation"] & T["test"])}
    if any(shared.values()):
        raise SystemExit(f"text shared between partitions: {shared}")
    sp = df.groupby("partition").speaker.agg(set)
    print(f"partitions: {counts}")
    print(f"text shared between partitions: {shared}")
    print(f"test speakers (dialect-namespaced): {len(sp['test'])}, also in train: {len(sp['test'] & sp['train'])}")


def compare_with_committed(df):
    if not COMMITTED.exists():
        print("no committed manifest to compare with")
        return
    old = pd.read_csv(COMMITTED)
    same = (len(old) == len(df)
            and (old.speaker.values == df.speaker.values).all()
            and (old.text.values == df.text.values).all()
            and (old.partition.values == df.partition.values).all())
    print(f"matches committed manifest row by row: {bool(same)}")


def main():
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} exists; pass --overwrite or choose --out")
    df = derive(pd.read_parquet(cfg.require(args.meta)))
    check(df)
    compare_with_committed(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[["speaker", "dialect", "text", "partition"]].to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()