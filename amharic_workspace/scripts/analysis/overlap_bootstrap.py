"""
Validation-test overlap check and bootstrap on the non-overlapping test utterances.
X = C2 (LoRA, bf16), Y = C4b (full FT, bf16). Printed delta = WER_X - WER_Y (positive: Y better).
"""
import numpy as np
from datasets import load_from_disk
from run_bootstrap import load, word_errors, cache_ok, bootstrap_delta

X_JSONL = "joint_composition_ablation/results_raw_persample.jsonl"   # C2, LoRA
Y_JSONL = "eval_controls/c4b_bfloat16_persample.jsonl"               # C4b, full FT
TEST_CACHE = "phase2_leyu/dataset_cache_joint_independent"
B, SEED = 10_000, 42

cache = load_from_disk(TEST_CACHE)
val, test = cache["validation"], cache["test"]

# 1. Overlap, normalized text, against both validation sets
sub = val.shuffle(seed=SEED).select(range(800))
sub_texts = {t.strip() for t in sub["norm_text"]}
full_texts = {t.strip() for t in val["norm_text"]}
test_texts = [t.strip() for t in test["norm_text"]]

ov800 = {i for i, t in enumerate(test_texts) if t in sub_texts}
ovfull = {i for i, t in enumerate(test_texts) if t in full_texts}
N = len(test_texts)
print(f"sharing text with 800-subset:      {len(ov800):5d} ({100*len(ov800)/N:.1f}%)")
print(f"sharing text with full validation: {len(ovfull):5d} ({100*len(ovfull)/N:.1f}%)")

# 2. Bootstrap on test utterances outside the 800-subset overlap
X, Y = load(X_JSONL), load(Y_JSONL)
keep = set(X) & set(Y) - ov800

ex, ey, n, spk = [], [], [], []
for i in sorted(keep):
    ref = X[i]["reference"]
    if ref != Y[i]["reference"] or not ref.strip():
        continue
    if not cache_ok(test[i], ref):
        raise SystemExit(f"cache mismatch at idx {i}; pairing invalid")
    e_x, Nw = word_errors(ref, X[i]["prediction"])
    e_y, _ = word_errors(ref, Y[i]["prediction"])
    ex.append(e_x); ey.append(e_y); n.append(Nw); spk.append(test[i]["speaker_id"])

ex, ey, n = np.array(ex, float), np.array(ey, float), np.array(n, float)
speakers = sorted(set(spk))
idx = {s: k for k, s in enumerate(speakers)}
sid = np.array([idx[s] for s in spk])
S = len(speakers)
sx = np.bincount(sid, weights=ex, minlength=S)
sy = np.bincount(sid, weights=ey, minlength=S)
sn = np.bincount(sid, weights=n, minlength=S)

print(f"\nremaining utterances={len(n)}  speakers={S}")
print(f"WER X (C2) = {100*ex.sum()/n.sum():.2f}%")
print(f"WER Y (C4b) = {100*ey.sum()/n.sum():.2f}%")
print(f"delta = WER_X - WER_Y = {100*(ex.sum()-ey.sum())/n.sum():.2f} points")

rng = np.random.default_rng(SEED)
d = bootstrap_delta(ex, ey, n, rng, B)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"utterance 95% CI [{lo:.2f}, {hi:.2f}]  detected={lo > 0 or hi < 0}")
d = bootstrap_delta(sx, sy, sn, rng, B)
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"speaker   95% CI [{lo:.2f}, {hi:.2f}]  detected={lo > 0 or hi < 0}")