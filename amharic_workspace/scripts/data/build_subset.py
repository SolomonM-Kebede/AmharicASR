"""Builds the 688-recording subset used for every subset result.

A test recording is in the subset when its duration is at most 30 s and its full label
sequence fits the 448-token decoder limit. The label sequence counts the decoder prefix
(language and task tokens) and the end-of-text token, as in the label cache. This is
the rule that gives 688; counting text tokens alone gives 690.

Reads splits/manifests/test_manifest.jsonl and writes splits/subset_688_idx.txt, one
manifest position per line. Every subset result reads that file, so an existing file is
never replaced unless its content differs and --overwrite is given. If the content is
identical, nothing is written.

Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json

from transformers import WhisperTokenizer

MANIFEST = cfg.MANIFESTS / "test_manifest.jsonl"
OUT = cfg.SUBSET_688_IDX
WINDOW_SECONDS = 30
TOKEN_LIMIT = cfg.MAX_LABEL_LENGTH
TEST_SIZE = 4588
SUBSET_SIZE = 688


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overwrite", action="store_true", help="replace the file if its content differs")
    return ap.parse_args()


def main():
    args = parse_args()
    with open(cfg.require(MANIFEST), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    print(f"manifest rows: {len(rows):,}")
    if len(rows) != TEST_SIZE:
        raise SystemExit(f"manifest has {len(rows)} rows, expected {TEST_SIZE}")

    tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL, language="amharic", task="transcribe")
    subset = [i for i, r in enumerate(rows)
              if r["duration_sec"] <= WINDOW_SECONDS
              and len(tok(r["reference"]).input_ids) <= TOKEN_LIMIT]
    print(f"subset size: {len(subset)}")
    if len(subset) != SUBSET_SIZE:
        raise SystemExit(f"subset has {len(subset)} recordings, expected {SUBSET_SIZE}; "
                         f"the token rule or the manifest differs. Nothing written.")

    text = "\n".join(map(str, subset)) + "\n"
    if OUT.exists():
        if OUT.read_text(encoding="utf-8") == text:
            print(f"{OUT} already matches; nothing written")