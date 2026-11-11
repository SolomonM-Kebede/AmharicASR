"""Checks that Google Speech-to-Text output covers exactly the test recordings
sent to its synchronous API: those of at most GOOGLE_MAX_SYNC_SECONDS (55 s).

Expected for the thesis: 4,588 manifest rows, 4,198 at or below 55 s, 4,198
scored, no missing recordings and no extra ones.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json

LIMIT = cfg.GOOGLE_MAX_SYNC_SECONDS
MANIFEST = cfg.require(cfg.MANIFESTS / "test_manifest.jsonl")
GOOGLE_RAW = cfg.require(cfg.GOOGLE / "google_stt_raw_predictions.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    manifest = read_jsonl(MANIFEST)
    scored = {r["path"] for r in read_jsonl(GOOGLE_RAW)}
    sent = {m["path"] for m in manifest if m["duration_sec"] <= LIMIT}
    not_sent = {m["path"] for m in manifest if m["duration_sec"] > LIMIT}

    print(f"manifest rows:            {len(manifest):,}")
    print(f"at most {LIMIT} s (sent):   {len(sent):,}")
    print(f"over {LIMIT} s (not sent):  {len(not_sent):,}")
    print(f"Google rows:              {len(scored):,}")
    print(f"sent but not scored:      {len(sent - scored):,}")
    print(f"scored but not sent:      {len(scored - sent):,}")

    if sent - scored or scored - sent:
        raise SystemExit("Google output does not match the sending rule; see the counts above.")
    print("coverage matches the sending rule")


if __name__ == "__main__":
    main()