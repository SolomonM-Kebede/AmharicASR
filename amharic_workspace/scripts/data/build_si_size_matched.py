
import os, collections
import pandas as pd
from datasets import load_dataset, load_from_disk, concatenate_datasets, DatasetDict, Audio
from normalize import normalize_amharic

BASE = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace"
HF_CACHE = "/home/stud_homes/s8345520/AmharicASR/hf_cache"
JOINT = f"{BASE}/phase2_leyu/dataset_cache_joint_independent"
OUT = f"{BASE}/phase2_leyu/dataset_cache_si_size_matched"
SEED = 42
REPOS = {"gojjam": "leyu-amharic/leyu-amharic-gojjam-dialect",
         "gonder": "leyu-amharic/leyu-amharic-gonder-dialect",
         "wello": "leyu-amharic/leyu-amharic-wello-dialect",
         "shewa": "leyu-amharic/leyu-amharic-shewa-dialect",
         "addis ababa": "leyu-amharic/leyu-amharic-addis-ababa-dialect"}

joint = load_from_disk(JOINT)
held = set(joint["validation"]["speaker_id"]) | set(joint["test"]["speaker_id"])
target = collections.Counter(joint["train"]["dialect"])          # C1's per-dialect training counts
print("held-out speakers:", len(held), "(expect 64) | target:", dict(target), "sum", sum(target.values()))
print("example held-out IDs:", sorted(held)[:3])

def ns(name, sid):                       # "<dialect>::<id>", as in the Joint cache
    sid = str(sid)
    return sid if "::" in sid else f"{name}::{sid}"

parts = []
for name, repo in REPOS.items():
    ds = load_dataset(repo, split="train", cache_dir=HF_CACHE)
    tcol = "text" if "text" in ds.column_names else "sentence"
    texts = [str(t or "") for t in ds[tcol]]
    spk = [ns(name, s) for s in ds["speaker_id"]]
    gender = list(ds["gender"]) if "gender" in ds.column_names else ["unknown"] * len(ds)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    ds = ds.remove_columns([c for c in ds.column_names if c != "audio"])
    ds = (ds.add_column("text", texts).add_column("norm_text", [normalize_amharic(t) for t in texts])
            .add_column("dialect", [name] * len(ds)).add_column("speaker_id", spk)
            .add_column("gender", gender))
    print(name, len(ds), "| example ID:", spk[0])
    parts.append(ds)
combined = concatenate_datasets(parts)

meta = pd.DataFrame({"i": range(len(combined)), "spk": list(combined["speaker_id"]),
                     "dialect": list(combined["dialect"]), "norm": list(combined["norm_text"])})
pool = meta[~meta.spk.isin(held)]
print("SI pool:", len(pool), "(expect 38,948) | speakers:", pool.spk.nunique(), "(expect 231)")
assert len(pool) == 38948 and pool.spk.nunique() == 231, "speaker-ID convention differs: send me the printed IDs"
assert set(target) == set(pool.dialect)

sample = pd.concat([pool[pool.dialect == d].sample(n=target[d], random_state=SEED) for d in target])
S = set(sample.norm)
te, va = joint["test"]["norm_text"], joint["validation"]["norm_text"]
print("D1 sample:", len(sample), "utts |", sample.spk.nunique(), "speakers | distinct texts", len(S))
print("share of test utts with text in sample: %.3f (SI full pool: 0.964, Joint: 0.0)" % (sum(t in S for t in te) / len(te)))
print("share of validation utts with text in sample: %.3f" % (sum(t in S for t in va) / len(va)))

os.makedirs("manifests", exist_ok=True)
sample[["spk", "dialect", "norm"]].to_csv("splits/manifests/d1_sample_manifest.csv", index=False)
DatasetDict({"train": combined.select(sorted(sample.i.tolist())),
             "validation": joint["validation"], "test": joint["test"]}).save_to_disk(OUT)
print("saved", OUT)