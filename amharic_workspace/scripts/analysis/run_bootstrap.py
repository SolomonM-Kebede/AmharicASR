"""Paired bootstrap for the corpus-level WER difference between two systems.

delta = WER(X) - WER(Y), in percentage points.
A positive delta means Y has the lower WER.

Intervals are 95% percentile intervals from B resamples with a fixed seed.
For the full test set the speaker-level interval is the primary one. The
688-recording subset is reported at utterance level only, because it has
13 speakers. A difference is "detected" when its interval excludes zero.

Defaults compare C2 (LoRA, bf16) with C4b (full fine-tuning, bf16).
Run from the workspace root after sourcing env.amh.sh.

Examples:
    python scripts/analysis/run_bootstrap.py
    python scripts/analysis/run_bootstrap.py --subset
    python scripts/analysis/run_bootstrap.py --x a2_persample.jsonl --y d1_persample.jsonl
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from common.bootstrap_utils import bootstrap_delta, cache_ok, load, word_errors

B = 10_000
SEED = 42
SUBSET_SIZE = 688


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default="results_raw_persample.jsonl",
                    help="first system, a file in results/persample/ (default: C2, LoRA, bf16)")
    ap.add_argument("--y", default="c4b_bfloat16_persample.jsonl",
                    help="second system (default: C4b, full fine-tuning, bf16)")
    ap.add_argument("--subset", action="store_true",
                    help="restrict to the 688 complete-recording subset (utterance level only)")
    ap.add_argument("--json", type=Path,
                    help="also write the results to this JSON file")
    return ap.parse_args()


def read_subset():
    path = cfg.require(cfg.SUBSET_688_IDX)
    idx = {int(x) for x in path.read_text().split()}
    if len(idx) != SUBSET_SIZE:
        raise SystemExit(f"{path} lists {len(idx)} indices, expected {SUBSET_SIZE}")
    return idx


def pair(X, Y, test, subset):
    """Collect matched utterances in sorted index order.

    The order matters: resampling draws positions in this order, so changing it
    changes the intervals.
    """
    data = {"ex": [], "ey": [], "n": [], "spk": []}
    stats = {"matched": 0, "ref_mismatch": 0, "empty_ref": 0, "cache_mismatch": 0}
    examples = []

    for i in sorted(set(X) & set(Y)):
        ref = X[i]["reference"]
        if ref != Y[i]["reference"]:
            stats["ref_mismatch"] += 1
            continue
        if not ref.strip():
            stats["empty_ref"] += 1
            continue
        if subset is not None and i not in subset:
            continue

        row = test[i]
        if not cache_ok(row, ref):
            stats["cache_mismatch"] += 1
            if len(examples) < 3:
                examples.append((i, ref[-40:], row["norm_text"][:80]))
            continue

        e_x, n = word_errors(ref, X[i]["prediction"])
        e_y, _ = word_errors(ref, Y[i]["prediction"])
        data["ex"].append(e_x)
        data["ey"].append(e_y)
        data["n"].append(n)
        data["spk"].append(row["speaker_id"])
        stats["matched"] += 1

    return data, stats, examples


def main():
    args = parse_args()
    x_path = cfg.require(cfg.PERSAMPLE / args.x)
    y_path = cfg.require(cfg.PERSAMPLE / args.y)
    test = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"]
    subset = read_subset() if args.subset else None

    data, stats, examples = pair(load(x_path), load(y_path), test, subset)

    label = f"{x_path.name} vs {y_path.name}" + ("  [688 subset]" if args.subset else "")
    print(label)
    print("pairing:", stats)
    for i, tail, head in examples:
        print(f"  cache mismatch idx={i}\n    ref tail : {tail!r}\n    norm head: {head!r}")
    if stats["ref_mismatch"] or stats["cache_mismatch"]:
        raise SystemExit("Pairing invalid. Do not use these results.")
    if args.subset and stats["matched"] != SUBSET_SIZE:
        raise SystemExit(f"matched {stats['matched']} subset utterances, expected {SUBSET_SIZE}")

    ex, ey, n = (np.array(data[k], dtype=float) for k in ("ex", "ey", "n"))
    point = 100.0 * (ex.sum() - ey.sum()) / n.sum()
    print(f"\nutterances = {len(n)}")
    print(f"WER X      = {100 * ex.sum() / n.sum():.2f}%")
    print(f"WER Y      = {100 * ey.sum() / n.sum():.2f}%")
    print(f"delta = WER_X - WER_Y = {point:.2f} points (positive: Y has lower WER)")

    levels = {"utterance": (ex, ey, n)}
    if not args.subset:
        speakers = sorted(set(data["spk"]))
        index = {s: k for k, s in enumerate(speakers)}
        sid = np.array([index[s] for s in data["spk"]])
        S = len(speakers)
        sx = np.bincount(sid, weights=ex, minlength=S)
        sy = np.bincount(sid, weights=ey, minlength=S)
        sn = np.bincount(sid, weights=n, minlength=S)
        print(f"speakers   = {S}")
        levels["speaker"] = (sx, sy, sn)

    # Utterance level is drawn first, then speaker level, from one generator.
    # Keep this order so the intervals reproduce.
    rng = np.random.default_rng(SEED)
    intervals = {}
    for name, (a, b, c) in levels.items():
        d = bootstrap_delta(a, b, c, rng, B)
        lo, hi = np.percentile(d, [2.5, 97.5])
        detected = bool(lo > 0 or hi < 0)
        intervals[name] = {"low": float(lo), "high": float(hi), "detected": detected}
        print(f"{name:10s} 95% CI [{lo:.2f}, {hi:.2f}]  detected={detected}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "x": x_path.name,
            "y": y_path.name,
            "subset": args.subset,
            "utterances": len(n),
            "wer_x": float(100 * ex.sum() / n.sum()),
            "wer_y": float(100 * ey.sum() / n.sum()),
            "delta_x_minus_y": float(point),
            "intervals": intervals,
            "bootstrap_draws": B,
            "seed": SEED,
        }
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()