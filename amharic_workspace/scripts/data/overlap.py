from data.normalize import normalize_amharic
import io
import numpy as np
import soundfile as sf
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk, Audio
from transformers import WhisperTokenizer

LEYU_CACHE = "phase2_leyu/dataset_cache_joint_independent"
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
ds = load_from_disk(LEYU_CACHE)

# for split in ds:
#     d = ds[split].cast_column("audio", Audio(decode=False))  # keep raw bytes
#     durs, toks = [], []
#     for row in d:
#         a = row["audio"]
#         src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
#         info = sf.info(src)                      # reads the header only
#         durs.append(info.frames / info.samplerate)
#         toks.append(len(tok(row["text"]).input_ids))
#     durs, toks = np.array(durs), np.array(toks)
#     print(f"{split}: n={len(durs)}  max dur={durs.max():.1f}s  "
#           f"p99={np.percentile(durs,99):.1f}s  >30s: {(durs>30).sum()}  "
#           f"max tokens={toks.max()}  >448 tokens: {(toks>448).sum()}")


# print(ds["test"].column_names)                 # is there a 'labels' / 'input_features' column?
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

# lab = [len(x) for x in ds["test"]["labels"]]   # adjust to the real column name
# print("max label len:", max(lab), " n at exactly max:", sum(l == max(lab) for l in lab))

# # are the stored labels a prefix of the full text?
# for i in range(5):
#     full = ds["test"][i]["text"]
#     cut = tok.decode(ds["test"][i]["labels"], skip_special_tokens=True)
#     print(len(full), len(cut), full.startswith(cut))
    
# import io, json, numpy as np, soundfile as sf, jiwer
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer

# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# test = load_from_disk(LEYU_CACHE)["test"]
# d = test.cast_column("audio", Audio(decode=False))

# dur = []
# for r in d:
#     a = r["audio"]
#     info = sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
#     dur.append(info.frames / info.samplerate)
# ntok = [len(tok(t).input_ids) for t in test["norm_text"]]
# keep = np.array([(x <= 30 and n <= 448) for x, n in zip(dur, ntok)])
# print("subset size:", keep.sum())

# rows = [json.loads(l) for l in open("results/persample/fullft_joint_indep_TEST_persample.jsonl")]
# # assumes the JSONL is in test order; otherwise join on the utterance id
# refs = [r["reference"] for r, k in zip(rows, keep) if k]
# hyps = [r["prediction"] for r, k in zip(rows, keep) if k]
# print("subset WER:", jiwer.wer(refs, hyps))
# print(sum(r["reference"] == t for r, t in zip(rows, test["norm_text"])), "/", len(rows))
# ntok_arr = np.array(ntok)
# match = np.array([r["reference"] == t for r, t in zip(rows, test["norm_text"])])
# print("match rate, <=448 tokens:", match[ntok_arr <= 448].mean())   # expect ~1.0
# print("match rate, >448 tokens: ", match[ntok_arr > 448].mean())    # expect ~0.0
# miss = [(r["reference"].rstrip("\ufffd"), t) for r, t, m in zip(rows, test["norm_text"], match) if not m]
# print("non-matching refs that are prefixes:", np.mean([t.startswith(r) for r, t in miss]))

# import glob, json, numpy as np, jiwer

# def per_utt(refs, hyps):
#     e, n = [], []
#     for r, h in zip(refs, hyps):
#         o = jiwer.process_words(r, h)
#         e.append(o.substitutions + o.deletions + o.insertions)
#         n.append(o.substitutions + o.deletions + o.hits)
#     return np.array(e), np.array(n)

# def boot_ci(e, n, B=10000, seed=0):
#     rng = np.random.default_rng(seed)
#     idx = rng.integers(0, len(e), (B, len(e)))
#     w = e[idx].sum(1) / n[idx].sum(1)
#     return np.percentile(w, [2.5, 97.5])

# print("subset dialects:", dict(zip(*np.unique(np.array(test["dialect"])[keep], return_counts=True))))

# for f in sorted(glob.glob("*persample*.jsonl")):
#     rows = [json.loads(l) for l in open(f)]
#     if len(rows) != len(keep):
#         print(f, "-> skipped, n =", len(rows)); continue
#     short = np.array(ntok) <= 448
#     m = np.mean([r["reference"] == t for r, t, s in zip(rows, test["norm_text"], short) if s])
#     if m < 0.99:
#         print(f, "-> WARNING row order or references differ, match rate", round(m, 3)); continue
#     refs = [r["reference"] for r in rows]; hyps = [r["prediction"] for r in rows]
#     e, n = per_utt(refs, hyps)
#     lo, hi = boot_ci(e[keep], n[keep])
#     print(f"{f}: full {e.sum()/n.sum():.4f} | subset({keep.sum()}) {e[keep].sum()/n[keep].sum():.4f} [{lo:.4f}, {hi:.4f}]")

# import networkx as nx
# from collections import Counter
# import pyarrow.parquet as pq
# from huggingface_hub import HfFileSystem

# # reuse the exact normalization (char map + punctuation + function) from your training script
# src = open("train_joint_composition.py").read()
# a = src.index("AMHARIC_CHAR_MAP")
# b = src.index("# ----", src.index("def normalize_amharic"))
# exec(src[a:b])
# assert callable(normalize_amharic)

# DIALECT_CONFIGS = [
#     ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
#     ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
#     ("leyu-amharic/leyu-amharic-wello-dialect", "wello"),
#     ("leyu-amharic/leyu-amharic-shewa-dialect", "shewa"),
#     ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
# ]

# fs = HfFileSystem()
# spk, txt = [], []
# for repo_id, dialect in DIALECT_CONFIGS:
#     files = sorted(fs.glob(f"datasets/{repo_id}/**/*.parquet"))
#     print(f"{dialect}: {len(files)} parquet files")
#     n0 = len(spk)
#     for i, f in enumerate(files):
#         with fs.open(f, "rb") as fh:
#             pf = pq.ParquetFile(fh)
#             if i == 0:
#                 print("  columns:", pf.schema_arrow.names)   # if 'text' or 'speaker_id' differ, adjust below
#             t = pf.read(columns=["speaker_id", "text"]).to_pydict()
#         for s, x in zip(t["speaker_id"], t["text"]):
#             spk.append(f"{dialect}_{s}")                       # namespaced by dialect
#             txt.append(normalize_amharic(str(x)))
#     print(f"  utterances: {len(spk) - n0}")

# G = nx.Graph()
# for s, t in zip(spk, txt):
#     G.add_edge(("S", s), ("T", t))
# comp_of = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
# sizes = sorted(Counter(comp_of[("S", s)] for s in spk).values(), reverse=True)
# N = len(spk)
# print("\nutterances:", N, " components:", len(sizes))
# print(f"largest component: {sizes[0]} utterances = {100 * sizes[0] / N:.1f}%")
# print("top 5 shares (%):", [round(100 * x / N, 1) for x in sizes[:5]])
# print("components with one utterance:", sum(1 for x in sizes if x == 1))
# from collections import defaultdict
# comp_dialects, comp_spk = defaultdict(Counter), defaultdict(set)
# for s in spk:
#     c = comp_of[("S", s)]
#     comp_dialects[c][s.split("_")[0]] += 1
#     comp_spk[c].add(s)
# for c in sorted(comp_dialects, key=lambda c: -sum(comp_dialects[c].values())):
#     n = sum(comp_dialects[c].values())
#     print(f"{100*n/N:5.1f}%  utts={n:6d}  speakers={len(comp_spk[c]):4d}  {dict(comp_dialects[c])}")

# import pandas as pd   # keep the metadata so you never have to re-download it
# pd.DataFrame({"speaker": spk, "text": txt}).to_parquet("leyu_meta_raw.parquet")
import json, numpy as np, jiwer

# FILES = {
#     "C2_LoRA_raw_joint": "results/persample/results_raw_persample.jsonl",
#     "C4_FFT_raw_joint":  "results/persample/fullft_joint_indep_TEST_persample.jsonl",
# }
# v, t = set(ds["validation"]["norm_text"]), set(ds["test"]["norm_text"])
# print("val/test shared texts:", len(v & t))
# vt = set(ds["validation"]["norm_text"])
# shared = sum(t in vt for t in ds["test"]["norm_text"])
# print(shared, len(ds["test"]), shared / len(ds["test"]))
# vsub = ds["validation"].select(range(800))      # use the exact construction from the script
# vt800 = set(vsub["norm_text"])
# shared = sum(t in vt800 for t in ds["test"]["norm_text"])
# print(shared, len(ds["test"]), shared / len(ds["test"]))
# test = ds["test"]
# vt8008 = set(ds["validation"].select(range(800))["norm_text"])   # same construction as the training script
# clean = np.array([t not in vt8008 for t in test["norm_text"]])
# print(clean.sum())    # should print 4,423

# def per_utt(refs, hyps):
#     e, n = [], []
#     for r, h in zip(refs, hyps):
#         o = jiwer.process_words(r, h)
#         e.append(o.substitutions + o.deletions + o.insertions)
#         n.append(o.substitutions + o.deletions + o.hits)
#     return np.array(e), np.array(n)

# E, N = {}, {}
# for name, path in FILES.items():
#     rows = [json.loads(l) for l in open(path)]
#     print(name, len(rows), list(rows[0].keys()))
#     assert len(rows) == len(test), "row count differs from the test set"
#     ok = np.mean([t.startswith(r["reference"].rstrip("\ufffd")) for r, t in zip(rows, test["norm_text"])])
#     print("  prefix rate (should be ~1):", round(ok, 4))
#     E[name], N[name] = per_utt([r["reference"] for r in rows], [r["prediction"] for r in rows])
#     e, n = E[name], N[name]
#     print(f"  full {e.sum()/n.sum():.4f} | no-val-text ({clean.sum()}) {e[clean].sum()/n[clean].sum():.4f}")

# a, b = "C4_FFT_raw_joint", "C2_LoRA_raw_joint"
# assert np.array_equal(N[a], N[b]), "reference word counts differ between the two files"

# def boot(mask, B=10000, seed=0):
#     idx = np.where(mask)[0]
#     rng = np.random.default_rng(seed)
#     d = np.empty(B)
#     for i in range(B):
#         s = rng.choice(idx, len(idx))
#         d[i] = E[a][s].sum() / N[a][s].sum() - E[b][s].sum() / N[b][s].sum()   # FFT minus LoRA
#     return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()

# for label, mask in [("full", np.ones(len(test), bool)), ("no-val-text", clean)]:
#     m, ci, p = boot(mask)
#     print(f"{label}: delta {m:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  LoRA better in {100*p:.2f}% of resamples")

# spk = np.array([f"{d}_{s}" for d, s in zip(test["dialect"], test["speaker_id"])])
# groups = {s: np.where(spk == s)[0] for s in np.unique(spk)}
# keys = list(groups)
# rng = np.random.default_rng(0)
# d = np.empty(10000)
# for i in range(10000):
#     idx = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
#     d[i] = E[a][idx].sum()/N[a][idx].sum() - E[b][idx].sum()/N[b][idx].sum()
# print(d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean())
# dia = np.array([r["dialect"] for r in rows])
# for dd in np.unique(dia):
#     m = dia == dd
#     print(dd, m.sum(), f"LoRA {E[b][m].sum()/N[b][m].sum():.4f}  FFT {E[a][m].sum()/N[a][m].sum():.4f}")

# src = open("train_joint_composition.py").read()
# exec(src[src.index("AMHARIC_CHAR_MAP"):src.index("def normalize_amharic")])
# print(len(AMHARIC_CHAR_MAP))
# for k, v in AMHARIC_CHAR_MAP.items():
#     print(k, "->", v)

# import json, os
# from collections import Counter
# man_all = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl")]
# man_m   = [json.loads(l) for l in open("test_set_audio_export_MATCHED/manifest.jsonl")]
# g       = [json.loads(l) for l in open("results/google/google_stt_raw_predictions.jsonl")]

# print("all:", len(man_all), list(man_all[0].keys()))
# print("matched:", len(man_m), Counter(m["match_type"] for m in man_m), Counter(m["dialect"] for m in man_m))
# print("matched sample:", man_m[0])
# print("all sample:    ", man_all[0])

# gp = {r["path"] for r in g}
# gb = {os.path.basename(r["path"]) for r in g}
# print("matched paths found in google file:", sum(m["path"] in gp for m in man_m))
# print("matched basenames found in google file:", sum(os.path.basename(m["path"]) in gb for m in man_m))

# import json, numpy as np, jiwer
# from collections import Counter
# from transformers import WhisperTokenizer

# # reuse normalize_amharic exactly as before (from train_joint_composition.py)
# src = open("train_joint_composition.py").read()
# a = src.index("AMHARIC_CHAR_MAP"); b = src.index("# ----", src.index("def normalize_amharic"))
# exec(src[a:b])

# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# man = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl")]
# g = {r["path"]: r for r in map(json.loads, open("results/google/google_stt_raw_predictions.jsonl"))}

# rows = []
# for m in man:
#     ref = normalize_amharic(m["reference"])
#     rows.append(dict(d=m["dialect"], dur=m["duration_sec"], ntok=len(tok(ref).input_ids), ref=ref, g=g.get(m["path"])))

# print("max reference tokens in export:", max(r["ntok"] for r in rows))   # ~1,240 means the references are full-length
# short = [r for r in rows if r["dur"] <= 30 and r["ntok"] <= 448]
# print("short clips:", len(short), Counter(r["d"] for r in short))          # expect 690: gojjam 186, gonder 275, wello 229

# def gwer(sel):
#     sel = [r for r in sel if r["g"]]
#     return len(sel), jiwer.wer([r["ref"] for r in sel], [normalize_amharic(r["g"]["hypothesis"]) for r in sel])

# print("Google on all scored utterances:", gwer(rows))      # sanity check, expect about 0.69
# print("Google on the short clips:", gwer(short))
# import re
# files = ["train_joint_composition.py", "final_eval_overall.py", "phase2_true_fixed_eval.py",
#          "evaluate_joint_composition.py", "zeroShot.py", "google_stt.py"]
# probe = "ሐኀዐሠፀ ሑ።፣፤፥፦ !?,. Hello  ሠላም\tዓለም 123"
# res = {}
# for f in files:
#     s = open(f).read()
#     a = min(m.start() for m in re.finditer(r"^(AMHARIC_CHAR_MAP|PUNCT)\s*=", s, re.M))
#     b = s.index("def normalize_amharic")
#     lines = s[b:].split("\n")
#     n = 1
#     while n < len(lines) and (lines[n].startswith((" ", "\t")) or lines[n] == ""):
#         n += 1
#     ns = {}
#     exec(s[a:b] + "\n".join(lines[:n]), ns)
#     res[f] = (len(ns["AMHARIC_CHAR_MAP"]), ns["normalize_amharic"](probe))
#     print(f, res[f])
# print("all identical:", len(set(res.values())) == 1)
# import io, json, numpy as np, jiwer, soundfile as sf
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer

# CACHE = "phase2_leyu/dataset_cache_joint_independent"
# test = load_from_disk(CACHE)["test"]
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

# # 1. which test utterances are short clips (<= 30 s and <= 448 tokens)
# d = test.cast_column("audio", Audio(decode=False))
# dur = []
# for r in d:
#     au = r["audio"]
#     info = sf.info(io.BytesIO(au["bytes"]) if au.get("bytes") else au["path"])
#     dur.append(info.frames / info.samplerate)
# ntok = [len(tok(t).input_ids) for t in test["norm_text"]]
# keep = np.array([x <= 30 and n <= 448 for x, n in zip(dur, ntok)])
# print("short clips:", keep.sum())      # must print 690

# # 2. errors and reference words per utterance, for each system
# def per_utt(path):
#     e, n = [], []
#     for line in open(path):
#         r = json.loads(line)
#         o = jiwer.process_words(r["reference"], r["prediction"])
#         e.append(o.substitutions + o.deletions + o.insertions)
#         n.append(o.substitutions + o.deletions + o.hits)
#     return np.array(e), np.array(n)

# E_c2, N_c2 = per_utt("results/persample/results_raw_persample.jsonl")
# E_c4, N_c4 = per_utt("results/persample/fullft_joint_indep_TEST_persample.jsonl")

# # 3. WER on all utterances and on the short clips
# for name, E, N in [("C2 LoRA", E_c2, N_c2), ("C4 Full FT", E_c4, N_c4)]:
#     print(f"{name}: all {E.sum()/N.sum():.4f} | short clips {E[keep].sum()/N[keep].sum():.4f}")
    
# def boot(mask, E1, N1, E2, N2, B=10000, seed=0):
#     idx = np.where(mask)[0]
#     rng = np.random.default_rng(seed)
#     d = np.empty(B)
#     for i in range(B):
#         s = rng.choice(idx, len(idx))
#         d[i] = E1[s].sum() / N1[s].sum() - E2[s].sum() / N2[s].sum()
#     return d.mean(), np.percentile(d, [2.5, 97.5]), (d > 0).mean()

# print(boot(keep, E_c4, N_c4, E_c2, N_c2))   # Full FT minus LoRA on the 690 short clips

# import json
# from collections import Counter
# from datasets import load_from_disk
# src = open("train_joint_composition.py").read()
# exec(src[src.index("AMHARIC_CHAR_MAP"):src.index("# ----", src.index("def normalize_amharic"))])
# CACHE = "phase2_leyu/dataset_cache_joint_independent"
# test = load_from_disk(CACHE)["test"].select_columns(["dialect", "norm_text"])
# man = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl")]
# a = Counter(zip(test["dialect"], test["norm_text"]))
# b = Counter((m["dialect"], normalize_amharic(m["reference"])) for m in man)
# print(a == b, sum((a - b).values()), sum((b - a).values()))

# import pandas as pd

# def load(path):  # adapt to your manifest format
#     return pd.read_json(path, lines=True)  # or pd.read_csv

# si = {k: load(f"splits/manifests/si_{k}.jsonl") for k in ("train", "val", "test")}
# jt = {k: load(f"splits/manifests/joint_{k}.jsonl") for k in ("train", "val", "test")}
# ti = {k: load(f"splits/manifests/ti_{k}.jsonl") for k in ("train", "val", "test")}
# # columns assumed: speaker, dialect, gender, text_norm, duration_s

# def overlap(train, test, col):
#     return test[col].isin(set(train[col])).mean()

# def describe(name, d):
#     tr, va, te = d["train"], d["val"], d["test"]
#     print(f"== {name}: train {len(tr)}, val {len(va)}, test {len(te)}")
#     print(" speakers  tr/va/te:", tr.speaker.nunique(), va.speaker.nunique(), te.speaker.nunique())
#     print(" speaker overlap train-test:", len(set(tr.speaker) & set(te.speaker)))
#     print(" text overlap  train-test (test utts):", round(overlap(tr, te, "text_norm"), 4))
#     print(" text overlap  val-test   (test utts):", round(overlap(va, te, "text_norm"), 4))
#     for n, df in (("train", tr), ("val", va), ("test", te)):
#         print(f" {n}: >30s {(df.duration_s > 30).mean():.3f}, median {df.duration_s.median():.1f}s")

# for name, d in (("SI", si), ("TI", ti), ("Joint", jt)):
#     describe(name, d)

# # per-dialect counts before/after the filter (Table dialect_counts)
# tab = pd.DataFrame({
#     "train_speakers": jt["train"].groupby("dialect").speaker.nunique(),
#     "SI_train": si["train"].groupby("dialect").size(),
#     "Joint_train": jt["train"].groupby("dialect").size(),
# })
# tab["removed_pct"] = (1 - tab.Joint_train / tab.SI_train) * 100
# print(tab.round(1))

# # corpus totals (Table corpora)
# allleyu = pd.concat([*si.values()])
# print("Leyu: utts", len(allleyu), "speakers", allleyu.speaker.nunique(),
#       "hours", round(allleyu.duration_s.sum() / 3600, 1))

# # does the Joint guarantee cover ALFFA/CV in the mixed runs?
# std = load("splits/manifests/standard_train_in_mixed_joint.jsonl")  # adapt
# held = set(jt["val"].text_norm) | set(jt["test"].text_norm)
# print("ALFFA/CV train utts whose text is in Leyu val/test:", std.text_norm.isin(held).sum())
import json, numpy as np
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
for P in ("results/persample/fullft_joint_indep_TEST_persample.jsonl",
          "results/persample/results_raw_persample.jsonl"):
    R = [json.loads(l) for l in open(P)]
    lr = np.array([len(tok(r["reference"]).input_ids) for r in R])
    lh = np.array([len(tok(r["prediction"]).input_ids) for r in R])
    rc, hc = lr >= 440, lh >= 440
    print(P.split("/")[-1])
    print("  reference length max / p99:", lr.max(), int(np.percentile(lr, 99)))   # ~447 means truncated labels
    print("  both at cap:", int((rc & hc).sum()), "| ref at cap only:", int((rc & ~hc).sum()),
          "| hyp at cap only:", int((~rc & hc).sum()), "| neither:", int((~rc & ~hc).sum()))