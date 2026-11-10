"""Statistics of the Joint split (the only split cache still on disk).
Prints: partition sizes, speakers, speaker/text overlap, share of clips > 30 s and > 448 tokens,
per-dialect counts, and LaTeX table rows.  Takes a few minutes (reads audio headers only).
Run from the workspace folder:   python split_stats.py
"""
import io
from collections import Counter
import numpy as np
import soundfile as sf
from datasets import load_from_disk, Audio
from transformers import WhisperTokenizer

CACHE = "phase2_leyu/dataset_cache_joint_independent"
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
ds = load_from_disk(CACHE)

info = {}
for name in ("train", "validation", "test"):
    d = ds[name].cast_column("audio", Audio(decode=False))          # keep raw bytes, do not decode
    spk = [f"{a}_{b}" for a, b in zip(d["dialect"], d["speaker_id"])]
    txt = list(d["norm_text"])
    dur = []
    for r in d:
        au = r["audio"]
        i = sf.info(io.BytesIO(au["bytes"]) if au.get("bytes") else au["path"])
        dur.append(i.frames / i.samplerate)
    ntok = [len(tok(t).input_ids) for t in txt]
    info[name] = dict(n=len(txt), spk=set(spk), txt=set(txt), txt_list=txt,
                      dur=np.array(dur), ntok=np.array(ntok), dialect=Counter(d["dialect"]))

print("\n=== partition statistics (Joint split) ===")
for name, v in info.items():
    n = v["n"]
    print(f"{name:11s} utterances {n:6d} | speakers {len(v['spk']):4d} | distinct texts {len(v['txt']):5d} | "
          f">30 s {100*(v['dur']>30).mean():5.1f}% | >448 tokens {100*(v['ntok']>448).mean():5.1f}% | "
          f"median duration {np.median(v['dur']):5.1f} s")
    print("            per dialect:", dict(v["dialect"]))

print("\n=== overlap ===")
pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
for a, b in pairs:
    spk_ov = len(info[a]["spk"] & info[b]["spk"])
    txt_ov = len(info[a]["txt"] & info[b]["txt"])
    share_b = np.mean([t in info[a]["txt"] for t in info[b]["txt_list"]])
    print(f"{a:10s} vs {b:10s}: shared speakers {spk_ov:3d} | shared distinct texts {txt_ov:5d} | "
          f"share of {b} utterances with a text also in {a}: {100*share_b:5.1f}%")


# ---------- save in machine-readable form ----------
import csv, json
out = {"split": "joint_independent", "partitions": {}, "overlap": {}}
for name, v in info.items():
    out["partitions"][name] = {
        "utterances": v["n"], "speakers": len(v["spk"]), "distinct_texts": len(v["txt"]),
        "pct_over_30s": round(100 * float((v["dur"] > 30).mean()), 1),
        "pct_over_448_tokens": round(100 * float((v["ntok"] > 448).mean()), 1),
        "median_duration_s": round(float(np.median(v["dur"])), 1),
        "per_dialect": dict(v["dialect"]),
    }
for a, b in pairs:
    out["overlap"][f"{a}_{b}"] = {
        "shared_speakers": len(info[a]["spk"] & info[b]["spk"]),
        "shared_distinct_texts": len(info[a]["txt"] & info[b]["txt"]),
        f"pct_of_{b}_utterances_with_text_in_{a}": round(100 * float(np.mean([t in info[a]["txt"] for t in info[b]["txt_list"]])), 1),
    }
json.dump(out, open("split_stats.json", "w"), indent=2, ensure_ascii=False)
with open("split_stats_partitions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["partition", "utterances", "speakers", "distinct_texts", "pct_over_30s", "pct_over_448_tokens", "median_duration_s"])
    for k, v in out["partitions"].items():
        w.writerow([k, v["utterances"], v["speakers"], v["distinct_texts"], v["pct_over_30s"], v["pct_over_448_tokens"], v["median_duration_s"]])
print("saved split_stats.json and split_stats_partitions.csv")

print("\n=== LaTeX rows: Partition | Utterances | Speakers | >30 s (%) | >448 tokens (%) ===")
for name, label in (("train", "Training"), ("validation", "Validation"), ("test", "Test")):
    v = info[name]
    print(f"{label} & {v['n']:,} & {len(v['spk'])} & {100*(v['dur']>30).mean():.1f} & {100*(v['ntok']>448).mean():.1f} \\\\")