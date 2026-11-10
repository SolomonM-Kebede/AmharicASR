# import torch, glob
# for run in ("phase2_leyu/models/joint_indep_lora_run", "joint_composition_ablation/models/raw_joint_lora_run",
#             "joint_composition_ablation/models/phase1_joint_lora_run", "single_phase_raw_whisper/models/joint_lora_run"):
#     ck = sorted(glob.glob(run + "/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))
#     try:
#         a = torch.load((ck[-1] if ck else run) + "/training_args.bin", weights_only=False)
#         print(run.split("/")[-1], a.learning_rate, a.warmup_steps, a.per_device_train_batch_size,
#               a.gradient_accumulation_steps, a.num_train_epochs, a.optim)
#     except Exception as e:
#         print(run.split("/")[-1], "no training_args.bin:", type(e).__name__)
# import pandas as pd
# from datasets import load_from_disk
# d1 = pd.read_csv("splits/manifests/d1_sample_manifest.csv")
# te = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"].select_columns(["norm_text"])["norm_text"]
# S = set(d1.norm.dropna())
# print("test utts with text in the D1 sample: %.3f" % (sum(t in S for t in te) / len(te)), "| distinct texts:", len(S))
# import glob, json, os, re, torch
# runs = {}
# for p in glob.glob("**/training_args.bin", recursive=True):
#     if "ray_results" in p: continue
#     d = os.path.dirname(p); m = re.search(r"checkpoint-(\d+)$", d)
#     run, step = (os.path.dirname(d), int(m.group(1))) if m else (d, 10**9)
#     if run not in runs or step > runs[run][0]: runs[run] = (step, p)
# for run, (_, p) in sorted(runs.items()):
#     a = torch.load(p, weights_only=False); d = os.path.dirname(p)
#     ts = os.path.join(d, "trainer_state.json"); s = json.load(open(ts)) if os.path.exists(ts) else {}
#     print(run, "|", a.optim, a.learning_rate, "warm", a.warmup_steps, f"{a.per_device_train_batch_size}x{a.gradient_accumulation_steps}",
#           "ep", a.num_train_epochs, "bf16", a.bf16, "gc", a.gradient_checkpointing, "best", a.load_best_model_at_end,
#           "| stopped", round(s.get("epoch", 0), 2), "best_metric", s.get("best_metric"))
# import pandas as pd
# from datasets import load_from_disk
# from normalize import normalize_amharic
# ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
# held = set(ds["validation"]["norm_text"]) | set(ds["test"]["norm_text"])
# d = "cv-corpus-25.0-2026-03-09/am/"
# cv = pd.concat([pd.read_csv(d + n + ".tsv", sep="\t", usecols=["sentence"], dtype=str)
#                 for n in ("train", "dev", "test", "validated", "other")]).sentence.dropna().map(normalize_amharic)
# print("CV rows in Leyu val/test:", int(cv.isin(held).sum()), "of", len(cv), "| distinct:", cv[cv.isin(held)].nunique())
# import random
# n = 10876; perm = list(range(n)); random.Random(42).shuffle(perm)
# pos = perm.index(n - 1); print(pos, "test" if pos < 1087 else "validation" if pos < 2174 else "train")
# import json, numpy as np
# from datasets import load_from_disk
# from transformers import WhisperTokenizer
# def stats(texts):
#     w = [x for t in texts for x in t.split()]
#     return round(float(np.mean([len(x) for x in w])), 2), round(100 * float(np.mean([len(x) == 1 for x in w])), 1)
# R = [json.loads(l) for l in open("results/persample/p1_persample.jsonl")]
# print("mean word length, % one-character words | Leyu refs:", stats([r["reference"] for r in R]),
#       "| Phase 1 hyps:", stats([r["prediction"] for r in R]))
# print("words per utterance | refs / hyps:", round(np.mean([len(r["reference"].split()) for r in R]), 1),
#       round(np.mean([len(r["prediction"].split()) for r in R]), 1))
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# std = load_from_disk("phase2b_leyu_standard/standard_speech_cache_ratio1.0")
# S = [tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip() for x in std.select_columns(["labels"])["labels"]]
# print("ALFFA+CV labels:", stats(S))
# import json, numpy as np, jiwer
# from datasets import load_from_disk

# T = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"].select_columns(["speaker_id", "dialect"])
# spk_all, dia_all = T["speaker_id"], T["dialect"]

# def load(path):
#     R = {}
#     for line in open(path):
#         r = json.loads(line); R[r["orig_idx"]] = r
#     assert len(R) == 4588, len(R)
#     assert all(R[i]["dialect"] == dia_all[i] for i in R), "orig_idx does not index the Joint test set"
#     return R

# def counts(R, idx):
#     e = np.zeros(len(idx)); n = np.zeros(len(idx))
#     for k, i in enumerate(idx):
#         ref, hyp = R[i]["reference"], R[i]["prediction"]
#         if not ref.strip():  e[k], n[k] = len(hyp.split()), 0; continue
#         if not hyp.strip():  e[k], n[k] = len(ref.split()), len(ref.split()); continue
#         o = jiwer.process_words(ref, hyp)
#         e[k] = o.substitutions + o.deletions + o.insertions
#         n[k] = o.hits + o.substitutions + o.deletions
#     return e, n

# def paired(pa, pb, B=10_000, seed=0):
#     """WER(b) - WER(a); positive means a is better."""
#     A, Bm = load(pa), load(pb); idx = sorted(A)
#     eA, nA = counts(A, idx); eB, nB = counts(Bm, idx)
#     assert (nA == nB).all(), "reference word counts differ"
#     spk = np.array([spk_all[i] for i in idx]); S = np.unique(spk)
#     agg = np.array([[eA[spk == s].sum(), eB[spk == s].sum(), nA[spk == s].sum()] for s in S])
#     rng = np.random.default_rng(seed); d = np.empty(B); d2 = np.empty(B); n = len(idx)
#     for b in range(B):
#         t = agg[rng.integers(0, len(S), len(S))].sum(0); d[b] = (t[1] - t[0]) / t[2] * 100
#         k = rng.integers(0, n, n);                       d2[b] = (eB[k].sum() - eA[k].sum()) / nA[k].sum() * 100
#     pt = (eB.sum() - eA.sum()) / nA.sum() * 100
#     return round(pt, 2), "speaker CI", tuple(np.round(np.percentile(d, [2.5, 97.5]), 2)), \
#            "utterance CI", tuple(np.round(np.percentile(d2, [2.5, 97.5]), 2)), "b worse in", round((d > 0).mean(), 3)

# F = {"A2": "results/persample/a2_persample.jsonl", "D1": "results/persample/d1_persample.jsonl", "C1": "results/persample/c1_persample.jsonl",
#      "C2": "results/persample/results_raw_persample.jsonl",
#      "C4": "results/persample/fullft_joint_indep_TEST_persample.jsonl"}
# for a, b in (("A2", "D1"), ("D1", "C1"), ("A2", "C1"), ("C2", "C4")):
#     print(f"WER({b}) - WER({a}):", paired(F[a], F[b]))

# from transformers import WhisperForConditionalGeneration
# m = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
# print(next(m.parameters()).dtype)


# import os
# import glob
# from itertools import combinations
# import pandas as pd

# CV_DIR = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/cv-corpus-25.0-2026-03-09/am"

# names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CV_DIR, "*.tsv")))

# # Step 1: inventory. Keep only files that have a clip-level 'path' column.
# tables = {}
# print("== Step 1: TSV inventory ==")
# for n in names:
#     df = pd.read_csv(os.path.join(CV_DIR, n), sep="\t", quoting=3)
#     if "path" not in df.columns:
#         print(f"{n:32s} SKIPPED (no 'path' column; columns start with {list(df.columns)[:3]})")
#         continue
#     tables[n] = df
#     print(f"{n:32s} rows={len(df):>8,}  duplicate paths within file={df['path'].duplicated().sum():>6,}")

# # Step 2: cross-file overlap by clip path
# print("\n== Step 2: overlap between clip-level files ==")
# for a, b in [("train.tsv", "dev.tsv"), ("train.tsv", "test.tsv"), ("dev.tsv", "test.tsv")]:
#     if a in tables and b in tables:
#         shared = len(set(tables[a]["path"]) & set(tables[b]["path"]))
#         print(f"{a} & {b}: {shared:,}")

# # Step 3: five-file combinations matching the thesis counts
# TARGET_ROWS = 2692      # total rows in the five source files
# TARGET_DISTINCT = 1622  # distinct clips; 2692 - 1622 = 1070 repeats
# print("\n== Step 3: five-file combinations with 2,692 rows and 1,622 distinct clips ==")
# matches = []
# for combo in combinations(tables.keys(), 5):
#     if sum(len(tables[n]) for n in combo) != TARGET_ROWS:
#         continue
#     distinct = pd.concat([tables[n]["path"] for n in combo]).nunique()
#     if distinct == TARGET_DISTINCT:
#         matches.append(combo)

# for m in matches:
#     print(m)
# print(f"matching combinations: {len(matches)}")
# import os
# import pandas as pd

# CV_DIR = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/cv-corpus-25.0-2026-03-09/am"
# SPLITS = ["train.tsv", "dev.tsv", "test.tsv", "other.tsv"]

# def load(n):
#     return pd.read_csv(os.path.join(CV_DIR, n), sep="\t", quoting=3)

# four = pd.concat([load(n) for n in SPLITS], ignore_index=True)
# val = load("validated.tsv")

# four_paths = set(four["path"])
# val_paths = set(val["path"])

# print(f"four files: rows={len(four):,} distinct paths={len(four_paths):,}")
# print(f"four files: distinct speakers={four['client_id'].nunique():,}")
# print(f"validated paths also in the four files: {len(val_paths & four_paths):,} of {len(val_paths):,}")
# print(f"validated paths in none of the four files: {len(val_paths - four_paths):,}")

# import os
# import pandas as pd

# CV_DIR = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/cv-corpus-25.0-2026-03-09/am"
# load = lambda n: set(pd.read_csv(os.path.join(CV_DIR, n), sep="\t", quoting=3)["path"])

# tr, dv, te = load("train.tsv"), load("dev.tsv"), load("test.tsv")
# val, oth = load("validated.tsv"), load("other.tsv")

# print("validated == train ∪ dev ∪ test:", val == (tr | dv | te))
# print("other disjoint from validated:", len(oth & val) == 0)

# import jiwer
# from normalize import normalize_amharic   # the same function the app should use

# ref = "አናቴ ብቻውን የመንዝን በግ ነበር የሚመስለው።ሌላው ቀርቶ ጠጉሬ ለገዛ ጣቴ እንኳን አይበገርም ነበር። ያ በወስፌ ጠጉሩን የሚያከው ልጅ? ይሉኝ ነበር ወይዛዝርቱ የኔ ስም በተነሳ ቁጥር"
# hyp = "አናቴ ብቻውን የመንዝን በግ ነበር የሚመስለው ሌላው ቀርቶ ጠጉሬ ለገዛ ጣት የእንኳን አይበገርም ነበር ያ በወስፌ ጠጉሩን የሚያከውልይ ይልይ ነበር ወይዛ ሀዘርቱ የኔስን በተነሳ ቁጥር"
# r, h = normalize_amharic(ref), normalize_amharic(hyp)
# o = jiwer.process_words(r, h)
# E = o.substitutions + o.deletions + o.insertions
# print("reference words:", len(r.split()), "| errors:", E, "| WER:", round(100 * E / len(r.split()), 2))

# from transformers import WhisperTokenizer
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="amharic", task="transcribe")
# ids = tok("ሰላም").input_ids
# print(len(ids), tok.convert_ids_to_tokens(ids))

# import json, jiwer
# from transformers import WhisperTokenizer
# from normalize import normalize_amharic   # the same function used in the training scripts

# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="amharic", task="transcribe")

# man = {}
# for line in open("test_set_audio_export/manifest.jsonl", encoding="utf-8"):
#     m = json.loads(line); man[m["path"]] = m

# G = {}
# for line in open("results/google/google_stt_raw_predictions.jsonl", encoding="utf-8"):
#     g = json.loads(line); G[g["path"]] = g

# sel = [p for p, m in man.items()
#        if m["duration_sec"] <= 30
#        and len(tok(normalize_amharic(m["reference"])).input_ids) <= 448]
# print("selected:", len(sel), "| with Google output:", sum(p in G for p in sel))

# E = N = 0
# for p in sel:
#     if p not in G:
#         continue
#     ref = normalize_amharic(man[p]["reference"])
#     hyp = normalize_amharic(G[p]["hypothesis"])
#     if not ref:
#         continue
#     o = jiwer.process_words(ref, hyp) if hyp else None
#     if o is None:
#         E += len(ref.split()); N += len(ref.split()); continue
#     E += o.substitutions + o.deletions + o.insertions
#     N += len(ref.split())
# print("Google WER on subset:", round(100 * E / N, 2))

# import json
# from transformers import WhisperTokenizer
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="amharic", task="transcribe")
# from datasets import load_from_disk
# test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
# man = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]
# G = {json.loads(l)["path"]: json.loads(l) for l in open("results/google/google_stt_raw_predictions.jsonl", encoding="utf-8")}
# sel = [i for i, m in enumerate(man)
#        if m["duration_sec"] <= 30 and len(tok(test[i]["norm_text"]).input_ids) <= 448]
# for i in sel:
#     if man[i]["path"] not in G:
#         print("no Google output:", i, man[i]["path"], man[i]["duration_sec"])

# import json
# G = json.load(open("phase2_leyu/results/google_stt_results.json"))
# failed = [k for k, v in G.items() if isinstance(v, dict) and v.get("status") == "failed"]
# print(failed)
import json

man = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]
scored = {json.loads(l)["path"]
          for l in open("results/google/google_stt_raw_predictions.jsonl", encoding="utf-8")}

over60 = [m for m in man if m["duration_sec"] > 60]
under60 = [m for m in man if m["duration_sec"] <= 60]
failed = [m["path"] for m in under60 if m["path"] not in scored]

print("manifest:", len(man), "| >60 s:", len(over60), "| <=60 s:", len(under60), "| scored:", len(scored))
print("failed (<=60 s, no output):", len(failed))
for p in failed:
    print(p)