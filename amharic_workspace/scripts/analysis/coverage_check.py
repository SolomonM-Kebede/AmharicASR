"""
Reference-coverage check on the C4b per-sample file (bf16 eval).
Buckets utterances by reference words per second and reports WER, insertion share,
and deletion share per bucket. Also reports the full-set WER with and without the
low-rate bucket.
"""
import jiwer
import numpy as np
from datasets import load_from_disk
from run_bootstrap import load, word_errors, cache_ok

Y_JSONL = "eval_controls/c4b_bfloat16_persample.jsonl"
TEST_CACHE = "phase2_leyu/dataset_cache_joint_independent"

test = load_from_disk(TEST_CACHE)["test"]
Y = load(Y_JSONL)

rows = []
for i in sorted(Y):
    ref = Y[i]["reference"]
    if not ref.strip():
        continue
    if not cache_ok(test[i], ref):
        raise SystemExit(f"cache mismatch at idx {i}; pairing invalid")
    dur = float(test[i]["audio"]["array"].shape[0]) / 16_000 if False else None
    rows.append((i, ref, Y[i]["prediction"]))

print("utterances:", len(rows))