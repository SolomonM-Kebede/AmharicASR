"""Joint and SI split diagnostics: independence, overlap, dialect counts, and the
Standard-Amharic pool. Run from the workspace root after sourcing env.amh.sh."""
from common import config as cfg  # must load before transformers/datasets

import pandas as pd
from datasets import load_from_disk
from transformers import WhisperTokenizer

from data.normalize import normalize_amharic

JOINT_CACHE = cfg.require(cfg.LEYU_JOINT_CACHE)
STANDARD_CACHE = cfg.require(cfg.JOINT_STANDARD_CACHE)
FULLFT_CACHE = cfg.require(cfg.DATA_DIR / "phase_fullft_joint_indep" / "dataset_cache")
META_PATH = cfg.require(cfg.DATA_DIR / "leyu_meta_raw.parquet")

# Joint split: independence and overlap 
cols = ["speaker_id", "dialect", "gender", "text", "norm_text"]
jt = {k: v.select_columns(cols).to_pandas() for k, v in load_from_disk(str(JOINT_CACHE)).items()}
tr, va, te = jt["train"], jt["validation"], jt["test"]

print("utts", len(tr), len(va), len(te))
print("speakers", tr.speaker_id.nunique(), va.speaker_id.nunique(), te.speaker_id.nunique())
print("speaker overlap train&test:", len(set(tr.speaker_id) & set(te.speaker_id)),
      "| train&val:", len(set(tr.speaker_id) & set(va.speaker_id)))

T_tr, T_va, T_te = set(tr.norm_text), set(va.norm_text), set(te.norm_text)
print("text overlap train&test:", len(T_tr & T_te), "| train&val:", len(T_tr & T_va))
print("val&test shared prompts:", len(T_va & T_te), "(text says 2,005)")
print("test utts with text in val:", round(te.norm_text.isin(T_va).mean(), 3), "(text says 0.67)")

for name, df in (("train", tr), ("val", va), ("test", te)):
    print(name, df.groupby("dialect").speaker_id.nunique().to_dict(),
          df.drop_duplicates("speaker_id").gender.value_counts().to_dict())

# SI split rebuilt from the metadata parquet 
# Speaker IDs in the parquet use "_" and "::" differently from the cache,
# so both sides are normalized to one key format before matching.
meta = pd.read_parquet(META_PATH)
key = lambda s: s.str.replace("::", "_").str.replace(" ", "_")
meta["key"] = key(meta.speaker)
si_tr = meta[meta.key.isin(set(key(tr.speaker_id)))].copy()
si_tr["dialect"] = si_tr.key.str.rsplit("_", n=1).str[0].str.replace("_", " ")

print("SI train:", len(si_tr), "(expect 38,948)")

# Compare normalized text on both sides. Raw text is a lower bound only.
si_norm = set(si_tr.text.map(normalize_amharic))
print("test utts with normalized text in SI train:",
      round(te.norm_text.isin(si_norm).mean(), 3), "(text says 0.96)")

# Table: dialect counts 
table = pd.DataFrame({
    "train_speakers": tr.groupby("dialect").speaker_id.nunique(),
    "SI_train": si_tr.groupby("dialect").size(),
    "Joint_train": tr.groupby("dialect").size(),
})
table["removed_pct"] = (1 - table.Joint_train / table.SI_train) * 100
print(table.round(1))

# Standard-Amharic pool 
std = load_from_disk(str(STANDARD_CACHE))
sm = std.select_columns(["utterance_id", "speaker_id", "split", "dialect"]).to_pandas()
print(len(sm), "utts,", sm.speaker_id.nunique(), "speakers")
print("missing values:", sm.isna().sum().to_dict())
print("split:", sm.split.value_counts(dropna=False).to_dict())
print("dialect:", sm.dialect.value_counts(dropna=False).to_dict())

# Does the Joint guarantee cover the Standard data? Decode labels and compare
# with Leyu validation and test text.
tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL)
labels = std.select_columns(["labels"])["labels"]
std_text = [tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip() for x in labels]
held = T_va | T_te
print("standard utts with text in Leyu val/test:", sum(s in held for s in std_text))

# Full fine-tuning training set (expect 20,592 total)
ft = load_from_disk(str(FULLFT_CACHE))["train"]
print("full-FT train by dialect:")
print(ft.select_columns(["dialect"]).to_pandas().dialect.value_counts().to_string())