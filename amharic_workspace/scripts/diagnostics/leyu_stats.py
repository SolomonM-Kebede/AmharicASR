#!/usr/bin/env python3
"""Paired bootstrap of WER(B) - WER(A) on per-sample predictions."""
import argparse, json
import numpy as np
import jiwer

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize(text):
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[str(r["orig_idx"])] = {"reference": r["reference"], "hypothesis": r["prediction"]}
    return out

def word_counts(ref, hyp):
    ref, hyp = normalize(ref), normalize(hyp or "")
    n = len(ref.split())
    if n == 0:
        return 0, 0
    if not hyp.strip():
        return n, n
    o = jiwer.process_words(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, n

def paired_bootstrap(keys, A, B, speaker_of, reps=10000, seed=0):
    ea, na, eb, nb = [], [], [], []
    for k in keys:
        e, n = word_counts(A[k]["reference"], A[k]["hypothesis"]); ea.append(e); na.append(n)
        e, n = word_counts(B[k]["reference"], B[k]["hypothesis"]); eb.append(e); nb.append(n)
    ea, na, eb, nb = (np.array(x, dtype=float) for x in (ea, na, eb, nb))
    rng = np.random.default_rng(seed)
    spk = np.array([speaker_of[k] for k in keys])
    groups = [np.where(spk == s)[0] for s in np.unique(spk)]
    n_utt = len(keys)
    d_spk, d_utt = np.empty(reps), np.empty(reps)
    for b in range(reps):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d_spk[b] = 100 * (eb[idx].sum() / nb[idx].sum() - ea[idx].sum() / na[idx].sum())
        idx = rng.integers(0, n_utt, n_utt)
        d_utt[b] = 100 * (eb[idx].sum() / nb[idx].sum() - ea[idx].sum() / na[idx].sum())
    return {
        "n_utt": n_utt,
        "n_speakers": len(groups),
        "wer_A": 100 * ea.sum() / na.sum(),
        "wer_B": 100 * eb.sum() / nb.sum(),
        "diff_B_minus_A": 100 * (eb.sum() / nb.sum() - ea.sum() / na.sum()),
        "speaker_95": np.percentile(d_spk, [2.5, 97.5]).tolist(),
        "utterance_95": np.percentile(d_utt, [2.5, 97.5]).tolist(),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_A"); ap.add_argument("pred_B")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--speakers", required=True)
    args = ap.parse_args()
    A, B = load(args.pred_A), load(args.pred_B)
    subset = [l.strip() for l in open(args.subset, encoding="utf-8") if l.strip()]
    speaker_of = json.load(open(args.speakers, encoding="utf-8"))
    keys = [k for k in subset if k in A and k in B]
    print(f"matched utterances: {len(keys)} of {len(subset)} listed")
    print(json.dumps(paired_bootstrap(keys, A, B, speaker_of), indent=2))

if __name__ == "__main__":
    main()