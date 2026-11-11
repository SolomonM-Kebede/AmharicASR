"""Text overlap between the Standard-Amharic sources and the Leyu held-out text.

Compares normalized ALFFA transcripts and every Common Voice sentence in the
release with the normalized validation and test texts of the Joint split. The
thesis states zero overlap for the Standard-Amharic data; this script measures it.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import pandas as pd
from datasets import load_dataset, load_from_disk

from data.normalize import normalize_amharic

ALFFA_DATASET = "hadamard-2/alffa-amharic"
CV_FILES = ["train", "dev", "test", "validated", "other"]


def heldout_texts():
    """Normalized validation and test texts of the Joint split."""
    ds = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    for split in ("validation", "test"):
        sample = ds[split].select_columns(["text", "norm_text"]).to_pandas()
        agree = (sample["text"].map(normalize_amharic) == sample["norm_text"].str.strip()).mean()
        if agree < 1.0:
            raise SystemExit(f"normalizer disagrees with cache on {split}: {agree:.4f}")
    return set(ds["validation"]["norm_text"]) | set(ds["test"]["norm_text"])


def alffa_texts():
    alffa = load_dataset(ALFFA_DATASET, split="train")
    return {normalize_amharic(t) for t in alffa["transcript"] if t}


def cv_texts():
    texts, rows = set(), {}
    for name in CV_FILES:
        path = cfg.CV_DIR / f"{name}.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", quoting=3, usecols=["sentence"], dtype=str)
        rows[name] = len(df)
        texts |= {normalize_amharic(t) for t in df["sentence"].dropna()}
    return texts, rows


def main():
    held = heldout_texts()
    print(f"held-out normalized texts (validation + test): {len(held):,}")

    alffa = alffa_texts()
    cv, cv_rows = cv_texts()
    print(f"ALFFA unique normalized texts: {len(alffa):,}")
    print(f"CV rows per file: {cv_rows}")
    print(f"CV unique normalized texts: {len(cv):,}")

    alffa_hit = alffa & held
    cv_hit = cv & held
    print(f"ALFFA texts in held-out: {len(alffa_hit)} ({100 * len(alffa_hit) / len(alffa):.2f}%)")
    print(f"CV texts in held-out:    {len(cv_hit)} ({100 * len(cv_hit) / len(cv):.2f}%)")
    print(f"distinct Standard texts in held-out: {len(alffa_hit | cv_hit)}")


if __name__ == "__main__":
    main()