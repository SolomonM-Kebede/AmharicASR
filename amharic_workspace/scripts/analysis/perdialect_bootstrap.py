"""Per-dialect paired comparison of two systems on the Joint test set.

Uses the pairing, counting and resampling of run_bootstrap.py (both from
common/bootstrap_utils.py). Each dialect is resampled with its own generator,
seeded with SEED, so a dialect's interval does not depend on the other dialects
or on their order. Utterance-level intervals only.

Printed delta = WER_X - WER_Y (positive: Y has the lower WER).
Defaults compare C2 (LoRA, bf16) with C4b (full fine-tuning, bf16).
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from common.bootstrap_utils import B, SEED, bootstrap_delta, load, pair

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED = {"gojjam": 654, "gonder": 1237, "wello": 836, "shewa": 1101, "addis ababa": 760}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", default="results_raw_persample.jsonl", help="first system (default: C2)")
    ap.add_argument("--y", default="c4b_bfloat16_persample.jsonl", help="second system (default: C4b)")
    ap.add_argument("--json", type=Path, help="also write the results to this JSON file")
    return ap.parse_args()


def main():
    args = parse_args()
    x_path = cfg.require(cfg.PERSAMPLE / args.x)
    y_path = cfg.require(cfg.PERSAMPLE / args.y)
    test = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"]
    dialect_of = test["dialect"]
    X, Y = load(x_path), load(y_path)

    print(f"{x_path.name} vs {y_path.name}  [per dialect]")
    print(f"{'dialect':12s} {'utt':>5s} {'spk':>4s} {'WER_X':>7s} {'WER_Y':>7s} {'delta':>7s}  "
          f"utterance 95% CI          detected")
    results = {}
    for d in DIALECTS:
        members = {i for i, x in enumerate(dialect_of) if x == d}
        data, stats, _ = pair(X, Y, test, members, "in")
        if stats["ref_mismatch"] or stats["cache_mismatch"]:
            raise SystemExit(f"{d}: pairing invalid {stats}")
        if stats["matched"] != EXPECTED[d]:
            raise SystemExit(f"{d}: matched {stats['matched']}, expected {EXPECTED[d]}")

        ex, ey, n = (np.array(data[k], dtype=float) for k in ("ex", "ey", "n"))
        wx = 100 * ex.sum() / n.sum()
        wy = 100 * ey.sum() / n.sum()
        rng = np.random.default_rng(SEED)
        lo, hi = np.percentile(bootstrap_delta(ex, ey, n, rng, B), [2.5, 97.5])
        detected = bool(lo > 0 or hi < 0)
        spk = len(set(data["spk"]))
        results[d] = {"utterances": len(n), "speakers": spk, "wer_x": float(wx), "wer_y": float(wy),
                      "delta_x_minus_y": float(wx - wy), "low": float(lo), "high": float(hi),
                      "detected": detected}
        print(f"{d:12s} {len(n):5d} {spk:4d} {wx:7.2f} {wy:7.2f} {wx - wy:+7.2f}  "
              f"[{lo:+.2f}, {hi:+.2f}]   {detected}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"x": x_path.name, "y": y_path.name, "scope": "per_dialect",
                                         "dialects": results, "bootstrap_draws": B, "seed": SEED},
                                        indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {args.json}")


if __name__ == "__main__":
    main()