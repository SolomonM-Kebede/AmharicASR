import itertools, json, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

m = pd.read_parquet("leyu_meta_raw.parquet")        # text == norm_text (verified earlier)
m["dialect"] = m.speaker.str.rsplit("_", n=1).str[0]

def ti_split(df):
    idx = defaultdict(list)
    for i, t in enumerate(df.text):                  # same logic as the script
        idx[t].append(i)
    texts = list(idx)                                # first-appearance order
    tr, tmp = train_test_split(texts, test_size=0.20, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, random_state=42)
    n = lambda S: sum(len(idx[t]) for t in S)
    te_rows = [i for t in te for i in idx[t]]
    return (n(tr), n(va), n(te)), df.iloc[te_rows].speaker.nunique(), set(te)

# 'combined' was concatenated from per-dialect datasets, so try the dialect orders
hits = []
for order in itertools.permutations(m.dialect.unique()):
    df = pd.concat([m[m.dialect == d] for d in order], ignore_index=True)
    sizes, nspk, te_texts = ti_split(df)
    if sizes[2] == 4894:
        hits.append((order, sizes, nspk, te_texts))
        print(order, sizes, "test speakers:", nspk)
print(len(hits), "dialect orders reproduce the 4,894 test size")
for order, sizes, nspk, te_texts in hits[:1]:
    df = pd.concat([m[m.dialect == d] for d in order], ignore_index=True)
    print("train/val/test:", sizes, "| test speakers:", nspk)
    print(df[df.text.isin(te_texts)].dialect.value_counts().to_dict())
    

import json, numpy as np
from common import config as cfg  # must load before transformers/datasets
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
files = {   # add every per-sample file behind a reported number (A2, A3, A4, B1, C1, C2, C3, C4)
  "C4": "results/persample/fullft_joint_indep_TEST_persample.jsonl",
  "C-ablation": "results/persample/results_raw_persample.jsonl",
}
for name, p in files.items():
    L = [len(tok(json.loads(l)["prediction"]).input_ids) for l in open(p)]
    print(name, "max", max(L), "| p50/p90/p99", np.percentile(L, [50, 90, 99]).round(),
          "| >=215 tokens:", sum(x >= 215 for x in L), "of", len(L))
    
for order, sizes, nspk, te_texts in hits:
    df = pd.concat([m[m.dialect == d] for d in order], ignore_index=True)
    print(order, sizes, nspk, df[df.text.isin(te_texts)].dialect.value_counts().to_dict())
    


import os, pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

m = pd.read_parquet("leyu_meta_raw.parquet")
m["dialect"] = m.speaker.str.rsplit("_", n=1).str[0]
order = ("gojjam", "gonder", "wello", "shewa", "addis ababa")
df = pd.concat([m[m.dialect == d] for d in order], ignore_index=True)

idx = defaultdict(list)
for i, t in enumerate(df.text):
    idx[t].append(i)
texts = list(idx)
tr, tmp = train_test_split(texts, test_size=0.20, random_state=42)
va, te = train_test_split(tmp, test_size=0.50, random_state=42)
part = {i: name for name, S in (("train", tr), ("validation", va), ("test", te))
        for t in S for i in idx[t]}
df["partition"] = pd.Series(part)
print(df.partition.value_counts().to_dict())          # expect 39681 / 4903 / 4894

sp = df.groupby("partition").speaker.agg(set)
print("test speakers (namespaced):", len(sp["test"]), "| also in train:", len(sp["test"] & sp["train"]))
num = lambda S: {s.rsplit("_", 1)[1] for s in S}
print("test speakers (numeric id only):", len(num(sp["test"])), "| also in train:",
      len(num(sp["test"]) & num(sp["train"])), "(chapter said 265)")
T = {p: set(g.text) for p, g in df.groupby("partition")}
print("text overlap tr/va, tr/te, va/te:", len(T["train"] & T["validation"]),
      len(T["train"] & T["test"]), len(T["validation"] & T["test"]))

os.makedirs("manifests", exist_ok=True)
df[["speaker", "dialect", "text", "partition"]].to_csv("splits/manifests/ti_split_manifest.csv", index=False)

import json, numpy as np
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
files = {...}   # fill in from the find output
for name, p in files.items():
    L = [len(tok(json.loads(l)["prediction"]).input_ids) for l in open(p)]
    print(f"{name:12s} max {max(L):4d} | share at cap (>=440): {np.mean(np.array(L) >= 440):.2f} | n={len(L)}")

import os, pandas as pd
from datasets import load_from_disk
os.makedirs("manifests", exist_ok=True)

jt = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
joint = pd.concat([jt[p].select_columns(["speaker_id", "dialect", "norm_text"]).to_pandas()
                     .assign(partition=p) for p in ("train", "validation", "test")])
joint.to_csv("splits/manifests/joint_split_manifest.csv", index=False)

key = lambda s: s.str.replace("::", "_").str.replace(" ", "_")
m = pd.read_parquet("leyu_meta_raw.parquet")
m["key"] = key(m.speaker); m["dialect"] = m.speaker.str.rsplit("_", n=1).str[0]
m["partition"] = "train"
for p in ("validation", "test"):
    m.loc[m.key.isin(set(key(joint[joint.partition == p].speaker_id))), "partition"] = p
m[["speaker", "dialect", "text", "partition"]].to_csv("splits/manifests/si_split_manifest.csv", index=False)
print(m.partition.value_counts().to_dict())      # expect train 38948, validation 5942, test 4588

