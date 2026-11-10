"""
Outside-subset comparison: test utterances NOT in the 688-recording subset.
C2 (LoRA, bf16) vs C4b (full FT, bf16). Printed delta = WER_X - WER_Y (positive: C4b better).
"""
import numpy as np
from datasets import load_from_disk
from run_bootstrap import load, word_errors, cache_ok, bootstrap_delta

X_JSONL = "joint_composition_ablation/results_raw_persample.jsonl"   # C2, LoRA
Y_JSONL = "eval_controls/c4b_bfloat16_persample.jsonl"               # C4b, full FT
TEST_CACHE = "phase2_leyu/dataset_cache_joint_independent"
SUBSET_FILE = "subset_688_idx.txt"
B, SEED = 10_000, 42

SUBSET = set(int(x) for x in open(SUBSET_FILE).read().split())
test = load_from_disk(TEST_CACHE)["test"]
X, Y = load(X_JSONL), load(Y_JSONL)

ex, ey, n, spk = [], [], [], []
stats = {"outside": 0, "ref_mismatch": 0, "empty_ref": 0}
for i in sorted(set(X) & set(Y)):
    if i in SUBSET:
        continue
    ref = X[i]["reference"]
    if ref != Y[i]["reference"]:
        stats["ref_mismatch"] += 1
        continue
    if not ref.strip():
        stats["empty_ref"] += 1
        continue
    row = test[i]
    if not cache_ok(row, ref):
        raise SystemExit(f"cache mismatch at idx {i}; pairing invalid")
    e_x, N = word_errors(ref, X[i]["prediction"])
    e_y, _ = word_errors(ref, Y[i]["prediction"])
    ex.append(e_x); ey.append(e_y); n.append(N); spk.append(row["speaker_id"])
    stats["outside"] += 1

print("pairing:", stats)
if stats["ref_mismatch"]:
    raise SystemExit("reference mismatch; pairing invalid")

ex, ey, n = np.array(ex, float), np.array(ey, float), np.array(n, float)
speakers = sorted(set(spk))
idx = {s: k for k, s in enumerate(speakers)}
sid = np.array([idx[s] for s in spk])
S = len(speakers)
sx = np.bincount(sid, weights=ex, minlength=S)
sy = np.bincount(sid, weights=ey, minlength=S)
sn = np.bincount(sid, weights=n, minlength=S)

print(f"\nutterances outside subset = {len(n)}  speakers = {S}")
print(f"WER C2  = {100*ex.sum()/n.sum():.2f}%")
print(f"WER C4b = {100*ey.sum()/n.sum():.2f}%")
print(f"delta = WER_X - WER_Y = {100*(ex.sum()-ey.sum())/n.sum():.2f} points")

rng = np.random.default_rng(SEED)
d = bootstrap_delta(ex, ey, n, rng, B)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"utterance 95% CI [{lo:.2f}, {hi:.2f}]  detected={lo > 0 or hi < 0}")
d = bootstrap_delta(sx, sy, sn, rng, B)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"speaker   95% CI [{lo:.2f}, {hi:.2f}]  detected={lo > 0 or hi < 0}")