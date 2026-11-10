# import json

# MANIFEST = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl"  # your manifest path
# PERSAMPLE = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/eval_controls/c4b_bfloat16_persample.jsonl"

# rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
# pers = [json.loads(l) for l in open(PERSAMPLE, encoding="utf-8")]
# subset_idx = {int(l) for l in open("subset_688_idx.txt", encoding="utf-8") if l.strip()}

# def full(p): return rows[p["orig_idx"]]["reference"].strip()
# def short(p): return p["reference"].strip()

# all_prefix = sum(full(p).startswith(short(p)) for p in pers)
# all_exact = sum(full(p) == short(p) for p in pers)
# sub = [p for p in pers if p["orig_idx"] in subset_idx]
# sub_exact = sum(full(p) == short(p) for p in sub)
# sub_prefix = sum(full(p).startswith(short(p)) for p in sub)

# print("all: exact", all_exact, "prefix", all_prefix, "of", len(pers))
# print("subset: exact", sub_exact, "prefix", sub_prefix, "of", len(sub))

import json
from datasets import load_from_disk
from normalize import normalize_amharic   # replace normalize_module


X = {}
with open("joint_composition_ablation/results_raw_persample.jsonl", encoding="utf-8") as f:
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