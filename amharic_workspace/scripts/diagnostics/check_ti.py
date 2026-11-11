"""Common Voice pool, text-mark sensitivity, and saved training arguments.

1. Counts the clips in each Common Voice TSV that the C2/C3 pool uses.
2. Measures how the Ethiopic word-separator marks and invisible characters
   change WER for the per-sample predictions of C2 and C4.
3. Prints the training arguments saved with the C2 and C4 runs.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import re

import pandas as pd
import torch
from datasets import load_from_disk

from common.bootstrap_utils import word_errors

CV_FILES = ["train", "dev", "test", "validated", "other"]
MARKS = "\u1361\u1367\u1368"                  # Ethiopic word separator and two related marks
INVISIBLE = "\u200b\ufe0f\u20e3\ufffd"        # zero-width space, variation selectors, replacement character

PERSAMPLE = {
    "C2 (LoRA)": cfg.PERSAMPLE / "results_raw_persample.jsonl",
    "C4 (full FT, FP16 weights)": cfg.PERSAMPLE / "fullft_joint_indep_TEST_persample.jsonl",
}
TRAINING_DIRS = {
    "C2": cfg.DATA_DIR / "joint_composition_ablation" / "models" / "raw_joint_lora_run",
    "C4": cfg.DATA_DIR / "phase_fullft_joint_indep" / "models",
}


def read_jsonl(path):
    with open(cfg.require(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def in_letter_set(ch):
    """Ethiopic syllables, space, and ASCII alphanumerics."""
    return 0x1200 <= ord(ch) <= 0x135A or ch == " " or (ch.isascii() and ch.isalnum())


def strict_clean(text):
    """Remove invisible characters, and map everything outside the letter set and
    Ethiopic punctuation and digits to a space."""
    text = re.sub(f"[{INVISIBLE}]", "", text)
    text = "".join(
        ch if (in_letter_set(ch) or 0x1369 <= ord(ch) <= 0x137C) else " "
        for ch in text
    )
    return " ".join(text.split())


def strip_marks(text):
    return " ".join(re.sub(f"[{MARKS}]", " ", text).split())


def corpus_wer(refs, hyps):
    """Corpus WER with the same word counting as the bootstrap. Empty references are skipped."""
    errors = words = 0
    for ref, hyp in zip(refs, hyps):
        if not ref.strip():
            continue
        e, n = word_errors(ref, hyp)
        errors += e
        words += n
    return 100.0 * errors / words


def check_common_voice_pool():
    print("1. Common Voice pool (thesis: 2,692 rows, 1,622 distinct clips, 46 speakers)")
    cfg.require(cfg.CV_DIR / "train.tsv")
    frames = {}
    for name in CV_FILES:
        path = cfg.CV_DIR / f"{name}.tsv"
        if path.exists():
            frames[name] = pd.read_csv(path, sep="\t", quoting=3, dtype=str,
                                       usecols=["client_id", "path", "sentence"])
    for name, df in frames.items():
        print(f"   {name + '.tsv':15s} {len(df):>6,} rows")

    pool = pd.concat(frames.values(), ignore_index=True)
    shared = int((pool.path.value_counts() > 1).sum())
    print(f"   rows: {len(pool):,}")
    print(f"   distinct clips: {pool.path.nunique():,}")
    print(f"   distinct speakers: {pool.client_id.nunique():,}")
    print(f"   clips listed in more than one file: {shared:,} (thesis: 1,070)")


def check_text_marks():
    print("\n2. WER with Ethiopic marks and invisible characters removed")
    ds = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    for split, data in ds.items():
        texts = data.select_columns(["norm_text"])["norm_text"]
        unclean = sum(any(not in_letter_set(ch) for ch in t) for t in texts) / len(texts)
        zero_width = sum("\u200b" in t for t in texts) / len(texts)
        print(f"   {split:10s} utterances with a non-letter character: {100 * unclean:5.1f}%"
              f" | with U+200B: {100 * zero_width:5.1f}%")

    results = {}
    for label, path in PERSAMPLE.items():
        rows = read_jsonl(path)
        refs = [r["reference"] for r in rows]
        hyps = [r["prediction"] for r in rows]
        as_stored = corpus_wer(refs, hyps)
        no_marks = corpus_wer([strip_marks(r) for r in refs], [strip_marks(h) for h in hyps])
        strict = corpus_wer([strict_clean(r) for r in refs], [strict_clean(h) for h in hyps])
        results[label] = (as_stored, strict)
        print(f"   {label}: as stored {as_stored:.2f} | marks removed {no_marks:.2f}"
              f" | strict {strict:.2f}")

    (c4_raw, c4_strict), (c2_raw, c2_strict) = (
        results["C4 (full FT, FP16 weights)"], results["C2 (LoRA)"])
    print(f"   C4 minus C2, as stored: {c4_raw - c2_raw:+.2f}"
          f" | strict: {c4_strict - c2_strict:+.2f}")


def print_training_args():
    print("\n3. Saved training arguments")
    for label, folder in TRAINING_DIRS.items():
        found = sorted(folder.rglob("training_args.bin"))
        if not found:
            print(f"   {label}: no training_args.bin under {folder}")
            continue
        args = torch.load(found[0], weights_only=False)
        print(f"   {label}: {found[0]}")
        print(f"     optim={args.optim} lr={args.learning_rate} warmup={args.warmup_steps}"
              f" batch={args.per_device_train_batch_size}"
              f" accum={args.gradient_accumulation_steps} bf16={args.bf16}")


def main():
    check_common_voice_pool()
    check_text_marks()
    print_training_args()


if __name__ == "__main__":
    main()