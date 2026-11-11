"""Diagnostics for the Joint and SI splits, the Standard-Amharic pool, and the
normalized text. Run from the workspace root after sourcing env.amh.sh.

Each section prints the figure the thesis states in its comment, so a
mismatch is visible at once.
"""
from common import config as cfg  # must load before transformers/datasets

import collections
import json
import re

import pandas as pd
from datasets import load_from_disk
from transformers import WhisperTokenizer

from common.bootstrap_utils import word_errors
from data.normalize import normalize_amharic

JOINT_CACHE = cfg.require(cfg.LEYU_JOINT_CACHE)
STANDARD_CACHE = cfg.require(cfg.JOINT_STANDARD_CACHE)
FULLFT_CACHE = cfg.require(cfg.DATA_DIR / "phase_fullft_joint_indep" / "dataset_cache")
META_PATH = cfg.require(cfg.DATA_DIR / "leyu_meta_raw.parquet")

MARKS = "\u1361\u1367\u1368"  # Ethiopic word separator and related marks

PERSAMPLE = {
    "C2 (LoRA, bf16)": cfg.PERSAMPLE / "results_raw_persample.jsonl",
    "C4b (full FT, bf16)": cfg.PERSAMPLE / "c4b_bfloat16_persample.jsonl",
    "C4 (full FT, fp16)": cfg.PERSAMPLE / "fullft_joint_indep_TEST_persample.jsonl",
}


def speaker_key(s):
    """The cache and the metadata write speaker IDs differently. Map both to one form."""
    return s.str.replace("::", "_", regex=False).str.replace(" ", "_", regex=False)


def strip_marks(text):
    return " ".join(re.sub(f"[{MARKS}]", " ", text).split())


def corpus_wer(refs, hyps):
    """Corpus WER with the same word counting as the bootstrap."""
    errors = words = 0
    for r, h in zip(refs, hyps):
        e, n = word_errors(r, h)
        errors += e
        words += n
    return 100.0 * errors / words


# ── Load the Joint split ─────────────────────────────────────────────────────
jt = {k: v.select_columns(["speaker_id", "dialect", "text", "norm_text"]).to_pandas()
      for k, v in load_from_disk(str(JOINT_CACHE)).items()}
tr, va, te = jt["train"], jt["validation"], jt["test"]

# ── 0. The shared normalizer reproduces the cached text (expect 1.0) ─────────
agree = (te.text.map(normalize_amharic) == te.norm_text.str.strip()).mean()
print(f"0. normalizer agrees with cached norm_text on test: {agree:.4f}")

# ── 1. Joint split: independence and overlap ─────────────────────────────────
print("\n1. Joint split")
print("   utterances  train/val/test:", len(tr), len(va), len(te))
print("   speakers    train/val/test:",
      tr.speaker_id.nunique(), va.speaker_id.nunique(), te.speaker_id.nunique())
print("   speaker overlap  train&test:", len(set(tr.speaker_id) & set(te.speaker_id)),
      "| train&val:", len(set(tr.speaker_id) & set(va.speaker_id)))

T_tr, T_va, T_te = set(tr.norm_text), set(va.norm_text), set(te.norm_text)
print("   text overlap     train&test:", len(T_tr & T_te),
      "| train&val:", len(T_tr & T_va))
print("   validation & test shared prompts:", len(T_va & T_te), "(thesis: 2,005)")
print("   test utterances with text in validation: %.3f (thesis: 0.67)"
      % te.norm_text.isin(T_va).mean())

# ── 2. SI split rebuilt from the metadata ────────────────────────────────────
meta = pd.read_parquet(META_PATH)
meta["key"] = speaker_key(meta.speaker)
meta["dialect"] = meta.key.str.rsplit("_", n=1).str[0].str.replace("_", " ")
meta["norm_text"] = meta.text.map(normalize_amharic)

held_out = set(speaker_key(va.speaker_id)) | set(speaker_key(te.speaker_id))
si_train = meta[~meta.key.isin(held_out)]

print("\n2. SI split")
print(f"   SI train utterances: {len(si_train):,} (thesis: 38,948)"
      f" | speakers: {si_train.key.nunique()}")
print("   speakers in SI train but absent from Joint train:",
      len(set(si_train.key) - set(speaker_key(tr.speaker_id))))

T_si = set(si_train.norm_text)
print("   test utterances, normalized text in SI train: %.3f (thesis: 0.964)"
      % te.norm_text.isin(T_si).mean())
print("   distinct test texts, in SI train: %.3f"
      % te.norm_text.drop_duplicates().isin(T_si).mean())
print("   test utterances, raw text in SI train: %.3f"
      % te.text.isin(set(si_train.text)).mean())
print("   test utterances, text in Joint train: %.3f (should be 0)"
      % te.norm_text.isin(T_tr).mean())

# ── 3. Per-dialect training counts, SI vs Joint (thesis: dialect table) ──────
print("\n3. Per-dialect training counts")
table = pd.DataFrame({
    "SI_speakers": si_train.groupby("dialect").key.nunique(),
    "Joint_speakers": tr.groupby("dialect").speaker_id.nunique(),
    "SI_train": si_train.groupby("dialect").size(),
    "Joint_train": tr.groupby("dialect").size(),
})
table["removed_pct"] = 100 * (1 - table.Joint_train / table.SI_train)
print(table.round(1).to_string())

# ── 4. Cache text vs metadata text ───────────────────────────────────────────
print("\n4. Cache text vs metadata text")
print("   test utterances whose raw text occurs in metadata: %.3f"
      % te.text.isin(set(meta.text)).mean())

pm = meta.groupby("norm_text").text.first()
distinct = te.drop_duplicates("norm_text")[["norm_text", "text"]].copy()
distinct["meta_text"] = distinct.norm_text.map(pm)
not_found = int(distinct.meta_text.isna().sum())
found = distinct.dropna(subset=["meta_text"])
differ = found[found.text != found.meta_text]
print(f"   distinct test texts: {len(distinct)} | not found in metadata: {not_found}"
      f" | raw text differs: {len(differ)}")
for _, row in differ.head(3).iterrows():
    print("     cache   :", repr(row.text[:70]))
    print("     metadata:", repr(row.meta_text[:70]))

c_cache = collections.Counter(ch for t in differ.text for ch in t)
c_meta = collections.Counter(ch for t in differ.meta_text for ch in t)
print("   more frequent in cache   :",
      [hex(ord(ch)) for ch, _ in (c_cache - c_meta).most_common(8)])
print("   more frequent in metadata:",
      [hex(ord(ch)) for ch, _ in (c_meta - c_cache).most_common(8)])

# ── 5. Ethiopic word separator and WER with marks removed ────────────────────
print("\n5. Ethiopic marks")
for ch in MARKS:
    survives = ch in normalize_amharic("ቃል" + ch + "ቃል")
    counts = [int(df.norm_text.str.count(ch).sum()) for df in (tr, va, te)]
    print(f"   {hex(ord(ch))} survives normalizer: {survives}"
          f" | count in norm_text train/val/test: {counts}")
for name, df in (("train", tr), ("val", va), ("test", te)):
    share = df.norm_text.str.contains(f"[{MARKS}]").mean()
    print(f"   {name}: utterances containing a mark: {100 * share:.1f}%")

for label, path in PERSAMPLE.items():
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    refs = [r["reference"] for r in rows]
    hyps = [r["prediction"] for r in rows]
    as_reported = corpus_wer(refs, hyps)
    stripped = corpus_wer([strip_marks(r) for r in refs], [strip_marks(h) for h in hyps])
    print(f"   {label}: as reported {as_reported:.2f} | marks removed {stripped:.2f}"
          f" | difference {stripped - as_reported:+.2f}")

# ── 6. Standard-Amharic pool ─────────────────────────────────────────────────
print("\n6. Standard-Amharic pool")
std = load_from_disk(str(STANDARD_CACHE))
sm = std.select_columns(["utterance_id", "speaker_id", "split", "dialect"]).to_pandas()
print(f"   utterances: {len(sm):,} (thesis: 11,428) | speakers: {sm.speaker_id.nunique():,}")
print("   split:", sm.split.value_counts(dropna=False).to_dict())
print("   dialect:", sm.dialect.value_counts(dropna=False).to_dict())
print("   missing values:", sm.isna().sum().to_dict())

tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL, language="amharic", task="transcribe")
labels = std.select_columns(["labels"])["labels"]
std_text = [normalize_amharic(tok.decode([i for i in x if i >= 0], skip_special_tokens=True))
            for x in labels]
held_text = T_va | T_te
print("   Standard utterances whose normalized text occurs in Leyu validation or test:",
      sum(t in held_text for t in std_text), "(expect 0)")

# ── 7. Full fine-tuning training set ─────────────────────────────────────────
print("\n7. Full fine-tuning training set")
ft = load_from_disk(str(FULLFT_CACHE))["train"].select_columns(["dialect"]).to_pandas()
print(f"   utterances: {len(ft):,} (thesis: 20,592)")
print(ft.dialect.value_counts().to_string())