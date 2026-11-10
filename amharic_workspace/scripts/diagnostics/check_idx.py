import json

MANIFEST = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl"  # replace with your manifest path
PERSAMPLE = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/eval_controls/c4b_bfloat16_persample.jsonl"

rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
pers = [json.loads(l) for l in open(PERSAMPLE, encoding="utf-8")]

mismatches = [p["orig_idx"] for p in pers if rows[p["orig_idx"]]["reference"] != p["reference"]]
print("rows:", len(rows), "persample lines:", len(pers))
print("reference mismatches:", len(mismatches))
if mismatches:
    i = mismatches[0]
    print("first mismatch orig_idx:", i)
    print("manifest:", rows[i]["reference"])
    print("persample:", [p["reference"] for p in pers if p["orig_idx"] == i][0])