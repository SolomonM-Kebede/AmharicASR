"""Shared helpers for the paired WER bootstrap scripts."""
import json

import jiwer
import numpy as np


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
        return n, n
    o = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    return o.substitutions + o.deletions + o.insertions, n


def cache_ok(row, ref):
    """Stored reference must be a prefix of the cached normalized text."""
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
