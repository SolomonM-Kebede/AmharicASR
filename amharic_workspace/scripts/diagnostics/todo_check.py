import json
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk
from leyu_stats import load, normalize, paired_bootstrap

E = "eval_controls"
A = load(f"{E}/c2fp32merge_bfloat16_persample.jsonl")  # C2, merged, bf16
B = load(f"{E}/c4b_bfloat16_persample.jsonl")          # C4b, bf16
rows = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]
val = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["validation"]

v800 = set(val.shuffle(seed=42).select(range(800))["norm_text"])
hits = {str(i) for i, r in enumerate(rows) if normalize(r["reference"]) in v800}
print("overlap with the 800 early-stopping subset:", len(hits))

keep = [str(i) for i in range(len(rows)) if str(i) not in hits]
ident = {str(i): str(i) for i in range(len(rows))}
res = paired_bootstrap(keep, A, B, ident)
print(f"n={res['n_utt']} C2 {res['wer_A']:.2f} C4b {res['wer_B']:.2f} "
      f"diff {res['diff_B_minus_A']:.2f} utterance_95 {res['utterance_95']}")