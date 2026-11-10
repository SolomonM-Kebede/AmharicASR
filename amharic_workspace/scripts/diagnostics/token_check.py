# import io, numpy as np, soundfile as sf
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer
# ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"].cast_column("audio", Audio(decode=False))
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# dur, ntok = [], []
# for ex in ds:
#     a = ex["audio"]
#     dur.append(sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]).duration)
#     ntok.append(len(tok(ex["norm_text"], add_special_tokens=False).input_ids))
# d, n = np.array(dur), np.array(ntok)
# short = (d <= 30) & (n <= 448)
# rate = np.median(n[short] / d[short])
# print(short.sum(), "utterances fit both limits (expect 690)")
# print("median tokens per second:", round(rate, 1), "| 448 tokens ~", round(448 / rate, 1), "s of speech")
# print("share of test utterances where 30 s of speech would exceed 448 tokens:", round(np.mean(n / d * 30 > 448), 3))
# print("share of test audio inside the 30 s window:", round(np.minimum(d, 30).sum() / d.sum(), 3))
# ref_end = d * np.minimum(1, 448 / n)
# print("evaluated reference prefix ends inside the 30 s window:", round(np.mean(ref_end <= 30), 3))
# print("reference prefix extends beyond the window (misaligned):", round(np.mean(ref_end > 30), 3))
# print("share of test audio covered by the evaluated prefix:", round(np.minimum(ref_end, 30).sum() / d.sum(), 3))
# r = n / d
# print("median tokens/s | utterances <=30 s:", round(np.median(r[d <= 30]), 1), "| >30 s:", round(np.median(r[d > 30]), 1))
# print("tokens/s percentiles (10, 50, 90) for >30 s:", np.round(np.percentile(r[d > 30], [10, 50, 90]), 1))

# import io, json, numpy as np, jiwer, soundfile as sf
# from datasets import load_from_disk, Audio

# ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"].cast_column("audio", Audio(decode=False))
# dur = np.array([sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]).duration for a in ds["audio"]])

# P = {}
# for line in open("eval_controls/c4b_bfloat16_persample.jsonl", encoding="utf-8"):
#     r = json.loads(line); P[r["orig_idx"]] = r

# groups = {"short (<=30 s)": [], "long (>30 s)": []}
# skipped = 0
# for i, d in enumerate(dur):
#     r = P[i]
#     if not r["reference"].strip() or not r["prediction"].strip():
#         skipped += 1; continue
#     o = jiwer.process_words(r["reference"], r["prediction"])
#     key = "short (<=30 s)" if d <= 30 else "long (>30 s)"
#     groups[key].append([o.hits, o.substitutions, o.deletions, o.insertions,
#                         len(r["reference"].split()), len(r["prediction"].split()), d])

# print("skipped (empty reference or prediction):", skipped)
# for k, v in groups.items():
#     H, S, D, I, Nref, Nhyp, T = np.array(v, dtype=float).sum(0)
#     E = S + D + I
#     print(f"{k}: utts {len(v)} | WER {100*E/Nref:.1f} | insertions {100*I/E:.1f}% of errors | "
#           f"deletions {100*D/E:.1f}% | hyp/ref words {Nhyp/Nref:.2f} | "
#           f"ref words/s {Nref/T:.2f} | hyp words/s {Nhyp/T:.2f}")

# import io, json, numpy as np, jiwer, soundfile as sf
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer

# test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
# dur = np.array([sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]).duration
#                 for a in test.cast_column("audio", Audio(decode=False))["audio"]])
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# ntok = np.array([len(tok(t).input_ids) for t in test["norm_text"]])   # same tokenization as the labels

# P = {}
# for line in open("eval_controls/c4b_bfloat16_persample.jsonl", encoding="utf-8"):
#     r = json.loads(line); P[r["orig_idx"]] = r

# G = {
#     "A audio<=30 s, reference complete":  (dur <= 30) & (ntok <= 448),
#     "B audio>30 s, reference complete":   (dur > 30) & (ntok <= 448),
#     "C audio>30 s, reference truncated":  (dur > 30) & (ntok > 448),
#     "D audio<=30 s, reference truncated": (dur <= 30) & (ntok > 448),
# }
# for name, m in G.items():
#     S = D = I = Nr = Nh = 0
#     for i in np.where(m)[0]:
#         o = jiwer.process_words(P[i]["reference"], P[i]["prediction"])
#         S += o.substitutions; D += o.deletions; I += o.insertions
#         Nr += len(P[i]["reference"].split()); Nh += len(P[i]["prediction"].split())
#     E = S + D + I
#     heard = np.mean(np.minimum(30, dur[m]) / dur[m])     # share of audio inside the window
#     print(f"{name}: n={m.sum()} | WER {100*E/Nr:.1f} | del {100*D/E:.1f}% ins {100*I/E:.1f}% | "
#           f"hyp/ref {Nh/Nr:.2f} | audio heard {100*heard:.0f}%")

# import io, json, numpy as np, jiwer, soundfile as sf
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer

# test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
# dur = np.array([sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]).duration
#                 for a in test.cast_column("audio", Audio(decode=False))["audio"]])
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# ntok = np.array([len(tok(t).input_ids) for t in test["norm_text"]])
# P = {}
# for line in open("eval_controls/c4b_bfloat16_persample.jsonl", encoding="utf-8"):
#     r = json.loads(line); P[r["orig_idx"]] = r

# def show(mask, name, k=5):
#     rows = []
#     for i in np.where(mask)[0]:
#         r = P[i]; o = jiwer.process_words(r["reference"], r["prediction"])
#         rows.append((o.insertions, i, dur[i], len(r["reference"].split()) / dur[i],
#                      len(r["reference"].split()), len(r["prediction"].split()), r))
#     rows.sort(key=lambda x: -x[0])
#     tot = sum(x[0] for x in rows)
#     print(f"\n== {name}: n={len(rows)} | total insertions {tot} | top-5 share {sum(x[0] for x in rows[:5]) / max(tot, 1):.2f}")
#     for ins, i, d, rate, nr, nh, r in rows[:k]:
#         print(f"idx {i} | dur {d:.1f}s | ref {nr} words ({rate:.2f} w/s) | hyp {nh} | ins {ins}")
#         print("  REF:", r["reference"][:160]); print("  HYP:", r["prediction"][:160])

# show((dur > 30) & (ntok <= 448), "B: audio >30 s, reference complete")
# show((dur <= 30) & (ntok > 448), "D: audio <=30 s, reference truncated")
# import io, json, collections, numpy as np, jiwer, soundfile as sf
# from datasets import load_from_disk, Audio
# from transformers import WhisperTokenizer

# test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
# dur = np.array([sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]).duration
#                 for a in test.cast_column("audio", Audio(decode=False))["audio"]])
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# full_words = np.array([len(t.split()) for t in test["norm_text"]])      # full reference, untruncated
# ntok = np.array([len(tok(t).input_ids) for t in test["norm_text"]])
# P = {}
# for line in open("eval_controls/c4b_bfloat16_persample.jsonl", encoding="utf-8"):
#     r = json.loads(line); P[r["orig_idx"]] = r
# rate = full_words / dur

# def max_repeat(h, n=3):
#     w = h.split()
#     g = collections.Counter(tuple(w[k:k+n]) for k in range(len(w) - n + 1))
#     return max(g.values(), default=0)

# def stats(idx):
#     S = D = I = N = loops = 0
#     for i in idx:
#         r = P[i]; o = jiwer.process_words(r["reference"], r["prediction"])
#         S += o.substitutions; D += o.deletions; I += o.insertions
#         N += len(r["reference"].split())
#         loops += max_repeat(r["prediction"]) >= 4
#     E = max(S + D + I, 1)
#     return dict(n=len(idx), WER=round(100 * (S + D + I) / N, 1),
#                 ins=round(100 * I / E, 1), dele=round(100 * D / E, 1), loops=int(loops))

# everything = list(range(len(dur)))
# low  = [i for i in everything if rate[i] < 0.5]
# high = [i for i in everything if rate[i] > 2.5]
# mid  = [i for i in everything if 0.5 <= rate[i] <= 2.5]
# for name, ix in (("all", everything), ("rate<0.5 (reference far too short)", low),
#                  ("rate>2.5 (very fast)", high), ("0.5 to 2.5 (typical)", mid),
#                  ("all minus rate<0.5", mid + high)):
#     print(f"{name:38s}", stats(ix))

# # group D sorted by deletions, the failure mode that carries 20% of its errors
# D_idx = [i for i in everything if dur[i] <= 30 and ntok[i] > 448]
# def dels(i): return jiwer.process_words(P[i]["reference"], P[i]["prediction"]).deletions
# for i in sorted(D_idx, key=dels, reverse=True)[:3]:
#     print(f"\nidx {i} | dur {dur[i]:.1f}s | {full_words[i]} ref words ({rate[i]:.2f} w/s) | deletions {dels(i)}")
#     print("  REF:", P[i]["reference"][:160]); print("  HYP:", P[i]["prediction"][:160])

import io, numpy as np, soundfile as sf
from datasets import load_from_disk, Audio

test = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
audio = test.cast_column("audio", Audio(decode=False))["audio"]
words = np.array([len(t.split()) for t in test["norm_text"]])
dur, sil = np.zeros(len(audio)), np.zeros(len(audio))
for i, a in enumerate(audio):
    x, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
    if x.ndim > 1:
        x = x.mean(1)
    dur[i] = len(x) / sr
    n = int(0.025 * sr)
    fr = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((fr ** 2).mean(1)) + 1e-10
    sil[i] = (20 * np.log10(rms / rms.max()) < -40).mean()   # share of frames 40 dB below the peak
rate = words / dur
for name, m in (("rate<0.5", rate < 0.5),
                ("0.5-2.5", (rate >= 0.5) & (rate <= 2.5)),
                ("rate>2.5", rate > 2.5)):
    print(f"{name:9s} n={m.sum():5d} | silent frames: median {100*np.median(sil[m]):.0f}%, mean {100*sil[m].mean():.0f}%")