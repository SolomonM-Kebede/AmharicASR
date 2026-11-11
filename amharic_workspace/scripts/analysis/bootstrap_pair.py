import sys, json, numpy as np, jiwer
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk

# WER(b) - WER(a): negative means b is better
name_a, path_a, name_b, path_b = sys.argv[1:5]        
B = 10_000
T = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"].select_columns(["speaker_id", "dialect"])
spk_all, dia_all = T["speaker_id"], T["dialect"]

def load(path):
    R = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); R[r["orig_idx"]] = r
    return R

def counts(R, idx):
    e = np.zeros(len(idx)); n = np.zeros(len(idx))
    for k, i in enumerate(idx):
        ref, hyp = R[i]["reference"], R[i]["prediction"]
        if not ref.strip():
            e[k], n[k] = len(hyp.split()), 0; continue
        if not hyp.strip():
            e[k], n[k] = len(ref.split()), len(ref.split()); continue
        o = jiwer.process_words(ref, hyp)
        e[k] = o.substitutions + o.deletions + o.insertions
        n[k] = o.hits + o.substitutions + o.deletions
    return e, n

A, Bm = load(path_a), load(path_b)
idx = sorted(set(A) & set(Bm)); assert len(idx) == 4588, len(idx)
assert all(A[i]["dialect"] == dia_all[i] for i in idx), "orig_idx does not index the Joint test set"
eA, nA = counts(A, idx); eB, nB = counts(Bm, idx)
assert (nA == nB).all(), "reference word counts differ"
print("reference string mismatches:", sum(A[i]["reference"].strip() != Bm[i]["reference"].strip() for i in idx))

wa, wb = eA.sum() / nA.sum() * 100, eB.sum() / nB.sum() * 100
print(f"WER {name_a}: {wa:.2f} | WER {name_b}: {wb:.2f} | diff ({name_b} - {name_a}): {wb - wa:+.2f}")

dia = np.array([dia_all[i] for i in idx])
for d in sorted(set(dia)):
    m = dia == d
    print(f"  {d:12s} {name_a} {eA[m].sum()/nA[m].sum()*100:5.2f}  {name_b} {eB[m].sum()/nB[m].sum()*100:5.2f}  diff {(eB[m].sum()-eA[m].sum())/nA[m].sum()*100:+.2f}")

spk = np.array([spk_all[i] for i in idx]); S = np.unique(spk)
agg = np.array([[eA[spk == s].sum(), eB[spk == s].sum(), nA[spk == s].sum()] for s in S])
rng = np.random.default_rng(0)
d_spk, d_utt, n = np.empty(B), np.empty(B), len(idx)
for b in range(B):
    t = agg[rng.integers(0, len(S), len(S))].sum(0); d_spk[b] = (t[1] - t[0]) / t[2] * 100
    k = rng.integers(0, n, n);                         d_utt[b] = (eB[k].sum() - eA[k].sum()) / nA[k].sum() * 100
pv = lambda d: max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / B)
for lbl, d in (("speaker-level", d_spk), ("utterance-level", d_utt)):
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"{lbl}: 95% interval [{lo:+.2f}, {hi:+.2f}] | two-sided p {'<' if pv(d) == 1 / B else '='} {pv(d):.4f}")
print(f"{name_b} better in {(d_spk < 0).mean() * 100:.1f}% of speaker-level resamples")