"""Paired bootstrap for the corpus-level WER difference between two systems.

delta = WER(X) - WER(Y), in percentage points.
A positive delta means Y has the lower WER.

Scopes (mutually exclusive):
  default                 all 4,588 test utterances; utterance and speaker intervals
  --subset                the 688 complete-recording subset; utterance interval only
  --outside-subset        the 3,900 utterances not in the subset; utterance interval only
  --exclude-val-overlap   the 3,992 utterances whose text does not occur in the frozen
                          800-utterance validation sample (splits/validation_800_idx.txt);
                          utterance interval only

Intervals are 95% percentile intervals from B resamples with a fixed seed. A
difference is "detected" when its interval excludes zero. The speaker-level
interval is the primary one for the full set.

Defaults compare C2 (LoRA, bf16) with C4b (full fine-tuning, bf16).
Run from the workspace root after sourcing env.amh.sh.

Examples:
    python scripts/analysis/run_bootstrap.py
    python scripts/analysis/run_bootstrap.py --subset
    python scripts/analysis/run_bootstrap.py --outside-subset --json results/summaries/outside_subset.json
    python scripts/analysis/run_bootstrap.py --exclude-val-overlap --json results/summaries/val_overlap_excluded.json
    python scripts/analysis/run_bootstrap.py --x a2_persample.jsonl --y d1_persample.jsonl
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from common.bootstrap_utils import B, SEED, bootstrap_delta, cache_ok, load, pair, word_errors

SUBSET_SIZE = 688
VAL_OVERLAP_EXPECTED = 596
EXPECTED = {
    "full": 4588,
    "subset688": 688,
    "outside_subset688": 3900,
    "val_excluded": 3992,
}


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default="results_raw_persample.jsonl",
                    help="first system, a file in results/persample/ (default: C2, LoRA, bf16)")
    ap.add_argument("--y", default="c4b_bfloat16_persample.jsonl",
                    help="second system (default: C4b, full fine-tuning, bf16)")
    scope = ap.add_mutually_exclusive_group()
    scope.add_argument("--subset", action="store_true",
                       help="688 complete-recording subset (utterance level only)")
    scope.add_argument("--outside-subset", action="store_true",
                       help="3,900 utterances outside the subset (utterance level only)")
    scope.add_argument("--exclude-val-overlap", action="store_true",
                       help="drop test utterances whose text occurs in the frozen "
                            "800-utterance validation sample (utterance level only)")
    ap.add_argument("--json", type=Path, help="also write the results to this JSON file")
    return ap.parse_args()


def scope_of(args):
    if args.subset:
        return "subset688"
    if args.outside_subset:
        return "outside_subset688"
    if args.exclude_val_overlap:
        return "val_excluded"
    return "full"


def read_subset():
    path = cfg.require(cfg.SUBSET_688_IDX)
    idx = {int(x) for x in path.read_text().split()}
    if len(idx) != SUBSET_SIZE:
        raise SystemExit(f"{path} lists {len(idx)} indices, expected {SUBSET_SIZE}")
    return idx


def val_overlap_exclusions(cache, test):
    """Test positions whose normalized text occurs in the frozen validation sample."""
    path = cfg.require(cfg.SPLITS / "validation_800_idx.txt")
    pos = [int(x) for x in path.read_text().split()]
    if len(pos) != 800:
        raise SystemExit(f"{path} lists {len(pos)} indices, expected 800")
    val_texts = {t.strip() for t in cache["validation"].select(pos)["norm_text"]}
    excluded = {i for i, t in enumerate(test["norm_text"]) if t.strip() in val_texts}
    if len(excluded) != VAL_OVERLAP_EXPECTED:
        raise SystemExit(f"{len(excluded)} test utterances overlap the validation sample, "
                         f"expected {VAL_OVERLAP_EXPECTED}")
    return excluded

def main():
    args = parse_args()
    scope = scope_of(args)
    x_path = cfg.require(cfg.PERSAMPLE / args.x)
    y_path = cfg.require(cfg.PERSAMPLE / args.y)

    cache = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    test = cache["test"]

    subset = mode = exclude = None
    if scope == "subset688":
        subset, mode = read_subset(), "in"
    elif scope == "outside_subset688":
        subset, mode = read_subset(), "out"
    elif scope == "val_excluded":
        exclude = val_overlap_exclusions(cache, test)

    data, stats, examples = pair(load(x_path), load(y_path), test, subset, mode, exclude)

    print(f"{x_path.name} vs {y_path.name}  [{scope}]")
    print("pairing:", stats)
    for i, tail, head in examples:
        print(f"  cache mismatch idx={i}\n    ref tail : {tail!r}\n    norm head: {head!r}")
    if stats["ref_mismatch"] or stats["cache_mismatch"]:
        raise SystemExit("Pairing invalid. Do not use these results.")
    if stats["matched"] != EXPECTED[scope]:
        raise SystemExit(f"matched {stats['matched']} utterances for scope {scope}, "
                         f"expected {EXPECTED[scope]}")

    ex, ey, n = (np.array(data[k], dtype=float) for k in ("ex", "ey", "n"))
    point = 100.0 * (ex.sum() - ey.sum()) / n.sum()
    print(f"\nutterances = {len(n)}")
    print(f"WER X      = {100 * ex.sum() / n.sum():.2f}%")
    print(f"WER Y      = {100 * ey.sum() / n.sum():.2f}%")
    print(f"delta = WER_X - WER_Y = {point:.2f} points (positive: Y has lower WER)")

    levels = {"utterance": (ex, ey, n)}
    if scope == "full":
        speakers = sorted(set(data["spk"]))
        index = {s: k for k, s in enumerate(speakers)}
        sid = np.array([index[s] for s in data["spk"]])
        S = len(speakers)
        print(f"speakers   = {S}")
        levels["speaker"] = (
            np.bincount(sid, weights=ex, minlength=S),
            np.bincount(sid, weights=ey, minlength=S),
            np.bincount(sid, weights=n, minlength=S),
        )

    # Utterance level is drawn first, then speaker level, from one generator.
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
            "scope": scope,
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