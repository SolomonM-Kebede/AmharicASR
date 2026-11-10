import pandas as pd
from datasets import load_from_disk

JOINT = "phase2_leyu/dataset_cache_joint_independent"
cols = ["speaker_id", "dialect", "gender", "text", "norm_text"]
jt = {k: v.select_columns(cols).to_pandas() for k, v in load_from_disk(JOINT).items()}
tr, va, te = jt["train"], jt["validation"], jt["test"]

# --- Joint split: independence and overlap
print("utts", len(tr), len(va), len(te))
print("speakers", tr.speaker_id.nunique(), va.speaker_id.nunique(), te.speaker_id.nunique())
print("speaker overlap train&test:", len(set(tr.speaker_id) & set(te.speaker_id)),
      "| train&val:", len(set(tr.speaker_id) & set(va.speaker_id)))
T_tr, T_va, T_te = set(tr.norm_text), set(va.norm_text), set(te.norm_text)
print("text overlap train&test:", len(T_tr & T_te), "| train&val:", len(T_tr & T_va))
print("val&test shared prompts:", len(T_va & T_te), "(text says 2,005)")
print("test utts with text in val:", round(te.norm_text.isin(T_va).mean(), 3), "(text says 0.67)")
for n, df in (("train", tr), ("val", va), ("test", te)):
    print(n, df.groupby("dialect").speaker_id.nunique().to_dict(),
          df.drop_duplicates("speaker_id").gender.value_counts().to_dict())

# --- SI split rebuilt from the parquet
m = pd.read_parquet("leyu_meta_raw.parquet")
print(m.speaker.head(2).tolist(), tr.speaker_id.head(2).tolist())   # IDs must match in format
m["dialect"] = m.speaker.str.split("_").str[0]
si_tr = m[m.speaker.isin(set(tr.speaker_id))]
print("SI train utts:", len(si_tr), "(expect 38,948)")
print("SI: test utts whose RAW text occurs in SI train:",
      round(te.text.isin(set(si_tr.text)).mean(), 3), "(text says ~0.96; raw is a lower bound)")

# --- Table dialect_counts
t = pd.DataFrame({"train_speakers": tr.groupby("dialect").speaker_id.nunique(),
                  "SI_train": si_tr.groupby("dialect").size(),
                  "Joint_train": tr.groupby("dialect").size()})
t["removed_pct"] = (1 - t.Joint_train / t.SI_train) * 100
print(t.round(1))

# --- Standard-Amharic pool
std = load_from_disk("phase2b_leyu_standard/standard_speech_cache_ratio1.0")
sm = std.select_columns(["utterance_id", "speaker_id", "split", "dialect"]).to_pandas()
print(len(sm), "utts,", sm.speaker_id.nunique(), "speakers")
print(sm.split.value_counts().to_dict(), sm.dialect.value_counts().to_dict())

# Does the Joint guarantee cover ALFFA/CV? (decode labels, compare with Leyu val/test)
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
lab = std.select_columns(["labels"])["labels"]
txt = [tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip() for x in lab]
held = T_va | T_te
print("standard utts with text in Leyu val/test:", sum(s in held for s in txt))

# --- Which rows are in the full-FT training set? (20,592 vs 20,591)
ft = load_from_disk("phase_fullft_joint_indep/dataset_cache")
print(ft["train"].select_columns(["dialect"]).to_pandas().dialect.value_counts())

key = lambda s: s.str.replace("::", "_").str.replace(" ", "_")
m = pd.read_parquet("leyu_meta_raw.parquet")
m["key"] = key(m.speaker)
print(m.speaker.str.rsplit("_", n=1).str[0].value_counts())   # dialect prefixes

si_tr = m[m.key.isin(set(key(tr.speaker_id)))]
si_tr = si_tr.assign(dialect=si_tr.key.str.rsplit("_", n=1).str[0].str.replace("_", " "))
print("SI train:", len(si_tr), "(expect 38,948)")
print("test utts with RAW text in SI train:", round(te.text.isin(set(si_tr.text)).mean(), 3))

t = pd.DataFrame({"train_speakers": tr.groupby("dialect").speaker_id.nunique(),
                  "SI_train": si_tr.groupby("dialect").size(),
                  "Joint_train": tr.groupby("dialect").size()})
t["removed_pct"] = (1 - t.Joint_train / t.SI_train) * 100
print(t.round(1))

std = load_from_disk("phase2b_leyu_standard/standard_speech_cache_ratio1.0")
sm = std.select_columns(["utterance_id", "speaker_id", "split"]).to_pandas()
print(sm.isna().sum()); print(sm.split.value_counts(dropna=False))
print(sm.sample(8, random_state=0)); print(sm.utterance_id.str[:6].value_counts().head())
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
ft = load_from_disk("phase_fullft_joint_indep/dataset_cache")["train"]
d = ft.select_columns(["dialect", "labels"]).to_pandas()
ley = d[d.dialect != "standard"].labels.head(300)
dec = [tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip() for x in ley]
texts = list(set(tr.norm_text))
print(sum(any(t.startswith(s) for t in texts) for s in dec) / len(dec))  
from phase2b_leyu_alffa_train import normalize_amharic 
print(te[["text", "norm_text"]].head(3).to_string()); print(m.text.head(3).to_string())
si_norm = set(si_tr.text.map(normalize_amharic))
print("test utts with text in SI train:", te.norm_text.isin(si_norm).mean())