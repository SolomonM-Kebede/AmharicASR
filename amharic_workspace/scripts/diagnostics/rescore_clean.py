"""Re-score full-set WER after removing the replacement character from references.
Uses saved per-sample predictions only; no model or GPU is needed.
"""
import json
from leyu_stats import word_counts, paired_bootstrap
from leyu_stats import normalize



        
MANIFEST = "test_set_audio_export/manifest.jsonl"
EVAL = "eval_controls"
FILES = {
    "C2 (merged, bf16)": f"{EVAL}/c2fp32merge_bfloat16_persample.jsonl",
    "C4b (bf16)":        f"{EVAL}/c4b_bfloat16_persample.jsonl",
    "A2":                f"{EVAL}/a2_persample.jsonl",
    "C1":                f"{EVAL}/c1_persample.jsonl",
    "D1":                f"{EVAL}/d1_persample.jsonl",
}
BAD = "\ufffd"
DIALECTS = ["addis ababa", "gojjam", "gonder", "shewa", "wello"]



def clean(ref):
    """Cut at the replacement character and drop the partial word it sits in."""
    if BAD not in ref:
        return ref.strip()
    head = ref.split(BAD)[0]
    if not head.strip():
        return ""
    if head[-1].isspace():
        return head.strip()
    return head.rsplit(" ", 1)[0].strip() if " " in head else ""


def read(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append({"idx": r["orig_idx"], "dialect": r["dialect"],
                        "reference": r["reference"], "hypothesis": r["prediction"]})
    return out


def wer(rows):
    e = n = 0
    for r in rows:
        a, b = word_counts(r["reference"], r["hypothesis"])
        e += a
        n += b
    return 100 * e / n


def main():
    manifest = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    cleaned = {}
    for name, path in FILES.items():
        raw = read(path)
        clean_rows = [dict(r, reference=clean(r["reference"])) for r in raw]
        changed = sum(c["reference"] != r["reference"].strip() for c, r in zip(clean_rows, raw))
        empty = sum(not c["reference"] for c in clean_rows)
        not_prefix = sum(
            normalize(manifest[c["idx"]]["reference"]).split()[:len(normalize(c["reference"]).split())]
            != normalize(c["reference"]).split()
            for c in clean_rows)
        print(f"{name}: overall {wer(raw):.2f} -> {wer(clean_rows):.2f} | "
              f"changed {changed} | empty {empty} | not a prefix of manifest {not_prefix}")
        per = {d: round(wer([c for c in clean_rows if c["dialect"] == d]), 2) for d in DIALECTS}
        print("   by dialect:", per)
        cleaned[name] = {str(c["idx"]): c for c in clean_rows}

    A = cleaned["C2 (merged, bf16)"]
    B = cleaned["C4b (bf16)"]
    keys = sorted(A, key=int)
    ident = {k: k for k in keys}
    res = paired_bootstrap(keys, A, B, ident)
    print(f"C2 vs C4b, full set, cleaned: diff {res['diff_B_minus_A']:.2f}, "
          f"utterance_95 {[round(x, 2) for x in res['utterance_95']]}")
    shown = 0
    for c in clean_rows:
        m = manifest[c["idx"]]["reference"].split()
        r = c["reference"].split()
        if m[:len(r)] != r and shown < 3:
            k = next((i for i, (a, b) in enumerate(zip(m, r)) if a != b), min(len(m), len(r)))
            print("idx", c["idx"], "| first differing word at", k)
            print("  manifest:", m[max(0, k - 2):k + 3])
            print("  cleaned: ", r[max(0, k - 2):k + 3])
            shown += 1

if __name__ == "__main__":
    main()