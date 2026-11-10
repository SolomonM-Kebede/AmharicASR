"""
Per-dialect paired comparison, C2 (LoRA) vs C4b (full FT), bf16 evaluation.
Printed delta = WER_X - WER_Y (positive: C4b better).
"""
import numpy as np
from datasets import load_from_disk
from common.bootstrap_utils import load, word_errors, cache_ok, bootstrap_delta

X_JSONL = "results/persample/results_raw_persample.jsonl"   # C2, LoRA
Y_JSONL = "results/persample/c4b_bfloat16_persample.jsonl"               # C4b, full FT
TEST_CACHE = "phase2_leyu/dataset_cache_joint_independent"
B, SEED = 10_000, 42
DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]

test = load_from_disk(TEST_CACHE)["test"]
X, Y = load(X_JSONL), load(Y_JSONL)

rows = {d: {"ex": [], "ey": [], "n": [], "spk": []} for d in DIALECTS}
seen = set()
for i in sorted(set(X) & set(Y)):
    ref = X[i]["reference"]
    if ref != Y[i]["reference"] or not ref.strip():
        continue
    row = test[i]
    if not cache_ok(row, ref):
        raise SystemExit(f"cache mismatch at idx {i}; pairing invalid")
    d = row["dialect"].lower()
    if d not in rows:
        raise SystemExit(f"unexpected dialect label {d!r} at idx {i}")
    e_x, N = word_errors(ref, X[i]["prediction"])
    e_y, _ = word_errors(ref, Y[i]["prediction"])
    r = rows[d]
    r["ex"].append(e_x); r["ey"].append(e_y); r["n"].append(N); r["spk"].append(row["speaker_id"])
    seen.add(i)

print(f"paired utterances: {len(seen)} of {len(X)}")
rng = np.random.default_rng(SEED)

print(f"\n{'dialect':12s} {'utt':>5s} {'spk':>4s} {'WER_C2':>8s} {'WER_C4b':>8s} {'delta':>7s}  utterance 95% CI")
for d in DIALECTS:
    r = rows[d]
    ex, ey, n = np.array(r["ex"], float), np.array(r["ey"], float), np.array(r["n"], float)
    if len(n) == 0:
        print(f"{d:12s} no utterances")
        continue
    wx = 100 * ex.sum() / n.sum()
    wy = 100 * ey.sum() / n.sum()
    d_point = wx - wy
    dist = bootstrap_delta(ex, ey, n, rng, B)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    print(f"{d:12s} {len(n):5d} {len(set(r['spk'])):4d} {wx:8.2f} {wy:8.2f} {d_point:7.2f}  [{lo:.2f}, {hi:.2f}]")