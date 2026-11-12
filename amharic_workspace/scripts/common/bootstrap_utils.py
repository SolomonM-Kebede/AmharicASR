"""Shared helpers for the paired WER bootstrap scripts.

Sign convention: every delta is WER_X - WER_Y in percentage points, so a positive
delta means Y has the lower WER. The thesis tables use WER(C4b) - WER(C2), so
their signs are the reverse of the printed C2 - C4b values.
"""
import json

import jiwer
import numpy as np

B = 10_000   # bootstrap draws
SEED = 42    # fixed seed for every interval in the thesis


def load(path):
    """Per-sample JSONL file, keyed by orig_idx."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["orig_idx"]] = o
    return out


def word_errors(ref, hyp):
    """Return (word edit errors, reference word count) for one utterance.

    An empty hypothesis counts every reference word as deleted.
    """
    ref_words, hyp_words = ref.split(), hyp.split()
    n = len(ref_words)
    if not hyp_words:
        return n, n
    o = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    return o.substitutions + o.deletions + o.insertions, n


def cache_ok(row, ref):
    """The stored reference must be a prefix of the cached normalized text.

    A trailing partial character (U+FFFD) from byte-level truncation is ignored.
    """
    norm = row["norm_text"].strip()
    r = ref.strip().rstrip("\ufffd")
    return norm.startswith(r)


def pair(X, Y, test, subset=None, mode=None, exclude=None):
    """Collect matched utterances in sorted index order.

    subset is None for no subset filter. Otherwise mode "in" keeps only members of
    subset and mode "out" keeps only non-members. exclude is a set of test positions
    to drop. The order matters: resampling draws positions in this order, so
    changing it changes the intervals.
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
        if subset is not None and (i in subset) != (mode == "in"):
            continue
        if exclude is not None and i in exclude:
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


def bootstrap_delta(sx, sy, sn, rng, B):
    """Resample groups with replacement; return WER_X - WER_Y for each draw.

    For utterance-level intervals, pass per-utterance arrays. For speaker-level
    intervals, pass per-speaker totals. Draws come from one generator, so the
    order of calls must match the order used for the thesis figures.
    """
    G = len(sn)
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, G, G)
        out[b] = 100.0 * (sx[pick].sum() - sy[pick].sum()) / sn[pick].sum()
    return out