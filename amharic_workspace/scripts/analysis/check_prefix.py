

import json
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk
from data.normalize import normalize_amharic


X = {}
with open("results/persample/results_raw_persample.jsonl", encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        X[o["orig_idx"]] = o

test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]

counts = {"exact_prefix": 0, "prefix_after_strip": 0, "other": 0}
shown = 0
for i, o in X.items():
    ref = o["reference"]
    norm = test[i]["norm_text"]
    if norm.startswith(ref):
        counts["exact_prefix"] += 1
    elif norm.strip().startswith(ref.strip()):
        counts["prefix_after_strip"] += 1
    else:
        counts["other"] += 1
        if shown < 5:
            shown += 1
            k = next((j for j, (a, b) in enumerate(zip(ref, norm)) if a != b),
                     min(len(ref), len(norm)))
            print(f"idx={i}  len_ref={len(ref)}  len_norm={len(norm)}  first_diff={k}")
            print("  ref :", repr(ref[max(0, k-25):k+25]))
            print("  norm:", repr(norm[max(0, k-25):k+25]))

print(counts)