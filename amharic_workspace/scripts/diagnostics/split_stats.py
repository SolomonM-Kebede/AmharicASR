"""Statistics of the Joint split: partition sizes, speakers, distinct texts,
share of utterances over 30 s and over 448 tokens, median duration, per-dialect
counts, and overlap between partitions.

Writes splits/split_stats.json and splits/split_stats_partitions.csv and prints
the LaTeX rows for Chapter 4. Reads audio headers only, so it takes a few minutes.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import csv
import io
import json
from collections import Counter

import numpy as np
import soundfile as sf
from datasets import Audio, load_from_disk
from transformers import WhisperTokenizer

PARTITIONS = (("train", "Training"), ("validation", "Validation"), ("test", "Test"))
PAIRS = (("train", "validation"), ("train", "test"), ("validation", "test"))
MAX_SECONDS = 30
MAX_TOKENS = cfg.MAX_LABEL_LENGTH
# Must match the truncation rule used for the references. Set to False if the
# references were truncated on text tokens alone.
COUNT_SPECIAL_TOKENS = True


def duration_seconds(audio):
    """Duration from the file header, without decoding the audio."""
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    info = sf.info(source)
    return info.frames / info.samplerate


def load_partitions(ds, tok):
    info = {}
    for name, _ in PARTITIONS:
        data = ds[name].cast_column("audio", Audio(decode=False))
        durations = np.array([duration_seconds(a) for a in data["audio"]])
        tokens = np.array([len(tok(t, add_special_tokens=COUNT_SPECIAL_TOKENS).input_ids)
                           for t in data["norm_text"]])
        info[name] = {
            "n": len(data),
            "speakers": set(data["speaker_id"]),   # already dialect-prefixed in the cache
            "texts": set(data["norm_text"]),
            "text_list": list(data["norm_text"]),
            "durations": durations,
            "tokens": tokens,
            "dialect": Counter(data["dialect"]),
        }
    return info


def summarize(v):
    return {
        "utterances": v["n"],
        "speakers": len(v["speakers"]),
        "distinct_texts": len(v["texts"]),
        "pct_over_30s": round(100 * float((v["durations"] > MAX_SECONDS).mean()), 1),
        "pct_over_448_tokens": round(100 * float((v["tokens"] > MAX_TOKENS).mean()), 1),
        "median_duration_s": round(float(np.median(v["durations"])), 1),
        "per_dialect": dict(v["dialect"]),
    }


def overlaps(info):
    out = {}
    for a, b in PAIRS:
        out[f"{a}_{b}"] = {
            "shared_speakers": len(info[a]["speakers"] & info[b]["speakers"]),
            "shared_distinct_texts": len(info[a]["texts"] & info[b]["texts"]),
            f"pct_of_{b}_utterances_with_text_in_{a}": round(
                100 * float(np.mean([t in info[a]["texts"] for t in info[b]["text_list"]])), 1),
        }
    return out


def write_outputs(partitions, pairs):
    out_dir = cfg.SPLITS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_stats.json").write_text(
        json.dumps({"split": "joint_independent", "partitions": partitions, "overlap": pairs},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    with open(out_dir / "split_stats_partitions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["partition", "utterances", "speakers", "distinct_texts",
                    "pct_over_30s", "pct_over_448_tokens", "median_duration_s"])
        for name, v in partitions.items():
            w.writerow([name, v["utterances"], v["speakers"], v["distinct_texts"],
                        v["pct_over_30s"], v["pct_over_448_tokens"], v["median_duration_s"]])
    print(f"saved {out_dir / 'split_stats.json'} and {out_dir / 'split_stats_partitions.csv'}")


def main():
    tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL)
    ds = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    info = load_partitions(ds, tok)

    partitions = {name: summarize(info[name]) for name, _ in PARTITIONS}
    pairs = overlaps(info)

    print("\n=== partitions (Joint split) ===")
    for name, label in PARTITIONS:
        p = partitions[name]
        print(f"{label:11s} utterances {p['utterances']:6,} | speakers {p['speakers']:4d} | "
              f"distinct texts {p['distinct_texts']:5,} | >30 s {p['pct_over_30s']:5.1f}% | "
              f">448 tokens {p['pct_over_448_tokens']:5.1f}% | median {p['median_duration_s']:5.1f} s")
        print(f"            per dialect: {p['per_dialect']}")

    print("\n=== overlap ===")
    for a, b in PAIRS:
        o = pairs[f"{a}_{b}"]
        print(f"{a} vs {b}: shared speakers {o['shared_speakers']} | "
              f"shared distinct texts {o['shared_distinct_texts']} | "
              f"share of {b} utterances with text in {a}: "
              f"{o[f'pct_of_{b}_utterances_with_text_in_{a}']}%")

    write_outputs(partitions, pairs)

    print("\n=== LaTeX rows: Partition & Utterances & Speakers & >30 s & >448 tokens ===")
    for name, label in PARTITIONS:
        p = partitions[name]
        print(f"{label} & {p['utterances']:,} & {p['speakers']} & "
              f"{p['pct_over_30s']:.1f} & {p['pct_over_448_tokens']:.1f} \\\\")


if __name__ == "__main__":
    main()