"""Check that the per-sample predictions agree with the test manifest.

Compares the reference stored in each per-utterance record with the reference
in the test manifest, at the same index. Run from the workspace root after
sourcing env.amh.sh.
"""
import json

from common import config as cfg

MANIFEST = cfg.ROOT / "test_set_audio_export" / "manifest.jsonl"
PERSAMPLE = cfg.PERSAMPLE / "c4b_bfloat16_persample.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    rows = read_jsonl(cfg.require(MANIFEST))
    pers = read_jsonl(cfg.require(PERSAMPLE))
    by_idx = {p["orig_idx"]: p for p in pers}

    mismatches = [
        p["orig_idx"] for p in pers
        if rows[p["orig_idx"]]["reference"] != p["reference"]
    ]
    print(f"manifest rows: {len(rows):,}  per-sample lines: {len(pers):,}")
    print(f"reference mismatches: {len(mismatches):,}")
    if mismatches:
        i = mismatches[0]
        print(f"first mismatch orig_idx: {i}")
        print(f"  manifest: {rows[i]['reference']!r}")
        print(f"  persample: {by_idx[i]['reference']!r}")


if __name__ == "__main__":
    main()