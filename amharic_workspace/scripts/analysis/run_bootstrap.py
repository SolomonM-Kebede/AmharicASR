"""
Paired bootstrap for the corpus-level WER difference (C2 LoRA vs C4b full FT).

delta = WER(X) - WER(Y), in percentage points.
Positive delta: Y (full fine-tuning) has lower WER.
Reports utterance-level and speaker-level 95% percentile intervals.
A difference is 'detected' when its interval excludes zero.
"""
import json
import numpy as np
import jiwer
from datasets import load_from_disk

X_JSONL = "results/persample/a2_persample.jsonl"  # C4, FP16 weights
Y_JSONL = "results/persample/c1_persample.jsonl"   
TEST_CACHE = "phase2_leyu/dataset_cache_joint_independent"
B = 10_000
SEED = 42
MAX_CACHE_FAILURES = 0   # strict: any failure stops the run
SUBSET_FILE = "splits/subset_688_idx.txt"
SUBSET = set(int(x) for x in open(SUBSET_FILE).read().split())
USE_SUBSET = False

def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["orig_idx"]] = o
    return out


def word_errors(ref, hyp):
    """Return (word edit errors, reference word count) for one utterance."""
    ref_words, hyp_words = ref.split(), hyp.split()
    n = len(ref_words)
    if not hyp_words:
        return n, n                      # empty hypothesis: every reference word deleted
    o = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    return o.substitutions + o.deletions + o.insertions, n


def cache_ok(row, ref):
    """The stored reference must be a prefix of the cached normalized text.
    A trailing broken character (U+FFFD) from byte-level truncation is ignored."""
    norm = row["norm_text"].strip()
    r = ref.strip().rstrip("\ufffd")
    return norm.startswith(r)


def bootstrap_delta(sx, sy, sn, rng, B):
    G = len(sn)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        out[b] = 100.0 * (sx[pick].sum() - sy[pick].sum()) / sn[pick].sum()
    return out


def main():
    X, Y = load(X_JSONL), load(Y_JSONL)
    test = load_from_disk(TEST_CACHE)["test"]

    ex, ey, n, spk = [], [], [], []
    stats = {"matched": 0, "ref_mismatch": 0, "empty_ref": 0, "cache_mismatch": 0}
    cache_examples = []

    for i in sorted(set(X) & set(Y)):
        ref = X[i]["reference"]
        if ref != Y[i]["reference"]:
            stats["ref_mismatch"] += 1
            continue
        if not ref.strip():
            stats["empty_ref"] += 1
            continue
        if USE_SUBSET and i not in SUBSET:
            continue

        row = test[i]
        if not cache_ok(row, ref):
            stats["cache_mismatch"] += 1
            if len(cache_examples) < 3:
                cache_examples.append((i, ref[-40:], row["norm_text"][:80]))
            continue

        e_x, N = word_errors(ref, X[i]["prediction"])
        e_y, _ = word_errors(ref, Y[i]["prediction"])
        ex.append(e_x); ey.append(e_y); n.append(N); spk.append(row["speaker_id"])
        stats["matched"] += 1

    print("pairing:", stats)
    for i, tail, head in cache_examples:
        print(f"  cache mismatch idx={i}\n    ref tail : {tail!r}\n    norm head: {head!r}")
    if stats["ref_mismatch"] or stats["cache_mismatch"] > MAX_CACHE_FAILURES:
        raise SystemExit("Pairing invalid; see the counts above before using any result.")
    if USE_SUBSET and stats["matched"] != 688:
        raise SystemExit(f"Subset size {stats['matched']} != 688; check subset_688_idx.txt.")

    ex, ey, n = np.array(ex, float), np.array(ey, float), np.array(n, float)
    speakers = sorted(set(spk))
    index = {s: k for k, s in enumerate(speakers)}
    sid = np.array([index[s] for s in spk])
    S = len(speakers)
    sx = np.bincount(sid, weights=ex, minlength=S)
    sy = np.bincount(sid, weights=ey, minlength=S)
    sn = np.bincount(sid, weights=n, minlength=S)

    print(f"\nutterances={len(n)}  speakers={S}")
    print(f"WER X (LoRA)     = {100*ex.sum()/n.sum():.2f}%")
    print(f"WER Y (full FT)  = {100*ey.sum()/n.sum():.2f}%")
    point = 100.0 * (ex.sum() - ey.sum()) / n.sum()
    print(f"delta = WER_X - WER_Y = {point:.2f} points (positive: full FT better)")

    rng = np.random.default_rng(SEED)
    levels = [("utterance", ex, ey, n)]
    if not USE_SUBSET:
        levels.append(("speaker", sx, sy, sn))
    for label, a, b, c in [("utterance", ex, ey, n), ("speaker", sx, sy, sn)]:
        d = bootstrap_delta(a, b, c, rng, B)
        lo, hi = np.percentile(d, [2.5, 97.5])
        detected = lo > 0 or hi < 0
        print(f"{label:10s} 95% CI [{lo:.2f}, {hi:.2f}]  detected={detected}")


if __name__ == "__main__":
    main()