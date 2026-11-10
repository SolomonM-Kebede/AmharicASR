import pandas as pd
from datasets import load_from_disk

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

JOINT = "phase2_leyu/dataset_cache_joint_independent"
jt = {k: v.select_columns(["speaker_id", "dialect", "text", "norm_text"]).to_pandas()
      for k, v in load_from_disk(JOINT).items()}
tr, va, te = jt["train"], jt["validation"], jt["test"]

# 0. Does the pasted function reproduce the cache's norm_text?
print("normalizer matches cache norm_text (test):",
      (te.text.map(normalize_amharic) == te.norm_text).mean())      # expect 1.0

key = lambda s: s.str.replace("::", "_").str.replace(" ", "_")
m = pd.read_parquet("leyu_meta_raw.parquet")
m["key"] = key(m.speaker)
m["dialect"] = m.speaker.str.rsplit("_", n=1).str[0]
m["norm_text"] = m.text.map(normalize_amharic)

# 1. Reconcile the 38,948 vs 38,292 difference
print("parquet: utts", len(m), "| speakers", m.key.nunique())
for name, df in (("val", va), ("test", te)):
    ks = set(key(df.speaker_id))
    print(name, "| utts in parquet for these speakers:", int(m.key.isin(ks).sum()),
          "| utts in cache:", len(df))

held = set(key(va.speaker_id)) | set(key(te.speaker_id))
si_tr = m[~m.key.isin(held)]
print("SI train = all non-held-out utts:", len(si_tr), "(chapter: 38,948)",
      "| speakers:", si_tr.key.nunique())
print("of these, speakers absent from Joint train:",
      len(set(si_tr.key) - set(key(tr.speaker_id))))

# 2. Text overlap with normalized text on both sides
T = set(si_tr.norm_text)
print("SI: share of test utts with text in train:", round(te.norm_text.isin(T).mean(), 3),
      "(chapter ~0.96)")
print("SI: share of distinct test texts in train:",
      round(te.norm_text.drop_duplicates().isin(T).mean(), 3))
print("Joint: share of test utts with text in train:",
      te.norm_text.isin(set(tr.norm_text)).mean())

# 3. Per-dialect table (SI vs Joint)
t = pd.DataFrame({
    "SI_speakers":    si_tr.groupby("dialect").key.nunique(),
    "Joint_speakers": tr.groupby("dialect").speaker_id.nunique(),
    "SI_train":       si_tr.groupby("dialect").size(),
    "Joint_train":    tr.groupby("dialect").size(),
})
t["removed_pct"] = (1 - t.Joint_train / t.SI_train) * 100
print(t.round(1))
print(te.text.isin(set(m.text)).mean())   # ~1.0 means cache text = parquet text

import collections
pm = m.groupby("norm_text").text.first()
s = te.drop_duplicates("norm_text").copy()
s["pq_text"] = s.norm_text.map(pm)
d = s[s.text != s.pq_text]
print(len(d), "of", len(s), "distinct test texts differ")
for _, r in d.head(5).iterrows():
    print(repr(r.text[:70])); print(repr(r.pq_text[:70])); print()
c1 = collections.Counter(ch for t in d.text for ch in t)
c2 = collections.Counter(ch for t in d.pq_text for ch in t)
print("more in cache:", [(ch, hex(ord(ch)), n) for ch, n in (c1 - c2).most_common(12)])
print("more in parquet:", [(ch, hex(ord(ch)), n) for ch, n in (c2 - c1).most_common(12)])

test_pq = m[m.key.isin(set(key(te.speaker_id)))]
print("exact match :", round(test_pq.text.isin(set(si_tr.text)).mean(), 3))
print("normalized  :", round(test_pq.norm_text.isin(set(si_tr.norm_text)).mean(), 3))  # expect 0.964

print("raw       :", round(te.text.isin(set(va.text)).mean(), 3))
print("normalized:", round(te.norm_text.isin(set(va.norm_text)).mean(), 3))   # 0.67
print("parquet text == cache norm_text:", te.norm_text.isin(set(m.text)).mean())  # expect ~1.0

raw = "".join(te.text) + "".join(va.text) + "".join(tr.text)
print({hex(ord(c)): raw.count(c) for c in "\u1361\u1367\u1368"})

print([hex(ord(c)) for c in PUNCT if ord(c) > 127])   # expect 0x1362 to 0x1366 only
for ch in ("\u1361", "\u1367", "\u1368"):
    print(hex(ord(ch)),
          "in PUNCT:", ch in PUNCT,
          "| survives:", ch in normalize_amharic("ቃል" + ch + "ቃል"),
          "| in norm_text train/val/test:",
          [int(df.norm_text.str.count(ch).sum()) for df in (tr, va, te)])

import re
marks = "\u1361\u1367\u1368"
for n, df in (("train", tr), ("val", va), ("test", te)):
    print(n, "utts with a mark: %.1f%%" % (df.norm_text.str.contains(f"[{marks}]").mean() * 100))
t = " ".join(te.norm_text)
print("test, U+1361 attached on both sides:", len(re.findall(r"\S\u1361\S", t)),
      "| surrounded by spaces:", len(re.findall(r" \u1361 ", t)))

import json, re, jiwer
strip = lambda s: " ".join(re.sub(f"[{marks}]", " ", s).split())
files = {
  "C4 full FT joint": "phase_fullft_joint_indep/results/fullft_joint_indep_TEST_persample.jsonl",
  "composition ablation": "joint_composition_ablation/results_raw_persample.jsonl",
  # add C1, C2, C3 and the A-run per-sample files here
}
REF_KEYS = ["reference", "ref", "norm_ref", "target", "gold", "label"]
HYP_KEYS = ["prediction", "pred", "norm_pred", "hypothesis", "hyp"]
for name, path in files.items():
    rows = [json.loads(l) for l in open(path)]
    k = rows[0].keys()
    R = next((x for x in REF_KEYS if x in k), None); H = next((x for x in HYP_KEYS if x in k), None)
    print(name, "| keys:", list(k), "| using:", R, H)
    if not (R and H): continue
    refs, hyps = [r[R] for r in rows], [r[H] for r in rows]
    a = jiwer.wer(refs, hyps) * 100
    b = jiwer.wer([strip(x) for x in refs], [strip(x) for x in hyps]) * 100
    print(f"   WER as reported {a:.2f} | marks removed {b:.2f} | diff {b - a:+.2f}")