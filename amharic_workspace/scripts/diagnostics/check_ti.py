# from datasets import load_from_disk
# ti = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
# print(ti)                                  # check the column names before selecting
# cols = ["speaker_id", "dialect", "norm_text"]          # adapt to the printed names
# d = {k: v.select_columns(cols).to_pandas() for k, v in ti.items()}
# tr_, va_, te_ = d["train"], d["validation"], d["test"]

# print("utts:", {k: len(v) for k, v in d.items()})
# print("speakers:", {k: v.speaker_id.nunique() for k, v in d.items()})
# print("test speakers also in train:", len(set(te_.speaker_id) & set(tr_.speaker_id)))
# T = {k: set(v.norm_text) for k, v in d.items()}
# print("text overlap train&val, train&test, val&test:",
#       len(T["train"] & T["validation"]), len(T["train"] & T["test"]), len(T["validation"] & T["test"]))
# print("distinct texts:", {k: len(t) for k, t in T.items()})
# print(te_.dialect.value_counts(normalize=True).round(3).to_dict())
# print(tr_.dialect.value_counts(normalize=True).round(3).to_dict())
# import ast, glob, hashlib, collections, pandas as pd
# from datasets import load_from_disk

# def load_normalizer(path):
#     tree = ast.parse(open(path, encoding="utf8").read())
#     keep = [n for n in tree.body if
#             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("AMHARIC_CHAR_MAP", "PUNCT") for t in n.targets))
#             or (isinstance(n, ast.FunctionDef) and n.name == "normalize_amharic")
#             or (isinstance(n, ast.Import) and all(a.name in ("re", "unicodedata", "string") for a in n.names))]
#     if not any(isinstance(n, ast.FunctionDef) for n in keep):
#         return None
#     ns = {}
#     exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
#     return ns["normalize_amharic"]

# ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
# raw = [t for sp in ds.values() for t in sp.select_columns(["text"])["text"] if isinstance(t, str)]
# cv = pd.read_csv("cv-corpus-25.0-2026-03-09/am/validated.tsv", sep="\t", quoting=3,
#                  usecols=["sentence"]).sentence.dropna().tolist()
# extra = ["ሠላም\u1361ሐገር", "ዐ\u1367 \u1368 x:y", "A\xa0B   C", "ፀሐይ ዓለም ሥራ ኅሊና", "ሗ ኋ"]
# corpus = raw + cv + extra

# groups, outs = collections.defaultdict(list), {}
# for f in sorted(glob.glob("*.py")):
#     try:
#         fn = load_normalizer(f)
#         if fn is None:
#             continue
#         o = [fn(t) for t in corpus]
#     except Exception as e:
#         print("skip", f, type(e).__name__); continue
#     h = hashlib.md5("\x00".join(o).encode()).hexdigest()[:8]
#     groups[h].append(f); outs[h] = o

# major = max(groups, key=lambda h: len(groups[h]))
# print({h: len(v) for h, v in groups.items()}, "| strings:", len(corpus))
# for h, files in groups.items():
#     print(h, f"{len(files)} files (majority)" if h == major else files)
#     if h != major:
#         d = [i for i in range(len(corpus)) if outs[h][i] != outs[major][i]]
#         print("   differing strings:", len(d))
#         for i in d[:3]:
#             print("   ", repr(corpus[i][:50]), "->", repr(outs[major][i][:50]), "|", repr(outs[h][i][:50]))
import os, pandas as pd
d = "cv-corpus-25.0-2026-03-09/am/"
fr = {n: pd.read_csv(d + n + ".tsv", sep="\t", usecols=["client_id", "path", "sentence"], dtype=str)
      for n in ("train", "dev", "test", "validated", "other") if os.path.exists(d + n + ".tsv")}
print({k: len(v) for k, v in fr.items()}, "| sum:", sum(len(v) for v in fr.values()), "(log: 2,692)")
cv = pd.concat(fr.values())
print("unique clips:", cv.path.nunique(), "| unique sentences:", cv.sentence.nunique(),
      "| speakers:", cv.client_id.nunique(), "| clips in several files:", int((cv.path.value_counts() > 1).sum()))

import re, json, jiwer, numpy as np
from datasets import load_from_disk
ok = lambda c: 0x1200 <= ord(c) <= 0x135A or c == " " or (c.isascii() and c.isalnum())
ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
for p, d in ds.items():
    T = d.select_columns(["norm_text"])["norm_text"]
    print(p, "utts with a non-clean char: %.1f%%" % (100 * np.mean([any(not ok(c) for c in s) for s in T])),
          "| with U+200B: %.1f%%" % (100 * np.mean(["\u200b" in s for s in T])))

def strict(s):
    s = re.sub("[\u200b\ufe0f\u20e3\ufffd]", "", s)
    s = "".join(c if (0x1200 <= ord(c) <= 0x135A or 0x1369 <= ord(c) <= 0x137C or c == " " or (c.isascii() and c.isalnum())) else " " for c in s)
    return " ".join(s.split())

res = {}
for name, P in {"C4": "results/persample/fullft_joint_indep_TEST_persample.jsonl",
                "C2": "results/persample/results_raw_persample.jsonl"}.items():
    R = [json.loads(l) for l in open(P)]
    a = jiwer.wer([r["reference"] for r in R], [r["prediction"] for r in R]) * 100
    b = jiwer.wer([strict(r["reference"]) for r in R], [strict(r["prediction"]) for r in R]) * 100
    res[name] = (a, b); print(name, round(a, 2), "->", round(b, 2), "| diff", round(b - a, 2))
print("C4 - C2 gap before/after:", round(res["C4"][0] - res["C2"][0], 2), round(res["C4"][1] - res["C2"][1], 2))

import torch, glob
for pat in ("phase_fullft_joint_indep/models/*/training_args.bin",
            "single_phase_raw_whisper/models/joint_lora_run/*/training_args.bin"):
    for p in sorted(glob.glob(pat))[:1]:
        a = torch.load(p, weights_only=False)
        print(p.split("/")[0], a.optim, a.learning_rate, a.warmup_steps,
              a.per_device_train_batch_size, a.gradient_accumulation_steps, a.bf16)