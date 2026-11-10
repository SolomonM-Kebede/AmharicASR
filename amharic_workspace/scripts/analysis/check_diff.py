# import json

# MANIFEST = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl"
# PERSAMPLE = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/eval_controls/c4b_bfloat16_persample.jsonl"

# rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
# pers = [json.loads(l) for l in open(PERSAMPLE, encoding="utf-8")]

# shown = 0
# for p in pers:
#     full = rows[p["orig_idx"]]["reference"].strip()
#     short = p["reference"].strip()
#     if full.startswith(short):
#         continue
#     k = next((i for i, (a, b) in enumerate(zip(full, short)) if a != b), min(len(full), len(short)))
#     print("orig_idx", p["orig_idx"], "| first difference at char", k)
#     print("  manifest:", repr(full[max(0, k-30):k+30]))
#     print("  persample:", repr(short[max(0, k-30):k+30]))
#     print("  lengths: manifest", len(full), "persample", len(short))
#     shown += 1
#     if shown == 4:
#         break

import json, jiwer

X = {o["orig_idx"]: o for o in map(json.loads, open("joint_composition_ablation/results_raw_persample.jsonl", encoding="utf-8"))}
SUB = [int(x) for x in open("subset_688_idx.txt").read().split()]

refs = [X[i]["reference"] for i in SUB]
hyps = [X[i]["prediction"] for i in SUB]
print("subset size:", len(refs))
print("corpus WER (jiwer):", round(100 * jiwer.wer(refs, hyps), 2))