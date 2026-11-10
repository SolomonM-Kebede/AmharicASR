
# import re, pandas as pd
# from datasets import load_from_disk
# from transformers import WhisperTokenizer

# AMHARIC_CHAR_MAP = {
#     "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
#     "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
#     "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
#     "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
#     "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
# }
# PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

# def normalize_amharic(text: str) -> str:
#     for src, tgt in AMHARIC_CHAR_MAP.items():
#         text = text.replace(src, tgt)
#     for p in PUNCT:
#         text = text.replace(p, " ")
#     return " ".join(text.split()).strip()

# P = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/phase2b_leyu_standard/standard_speech_cache_ratio1.0"
# ds = load_from_disk(P)
# df = ds.select_columns(["utterance_id", "speaker_id", "split", "dialect"]).to_pandas()

# is_alffa = df.utterance_id.fillna("").str.match(r"^(tr|te)_\d+_(tr|te)\d+")
# print("ALFFA-style IDs:", int(is_alffa.sum()), "| no ID:", int(df.utterance_id.isna().sum()),
#       "| other:", int((~is_alffa & df.utterance_id.notna()).sum()))
# print(df[is_alffa].split.value_counts().to_dict())
# # ALFFA speaker = first two fields of the ID (assumption: verify against ALFFA's spk2utt)
# print("ALFFA speakers:", df[is_alffa].utterance_id.str.rsplit("_", n=1).str[0].nunique())

# # Do the 553 ID-less rows match Common Voice sentences?
# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# cv_dir = "cv-corpus-25.0-2026-03-09/am/"
# cv = {n: pd.read_csv(cv_dir + n + ".tsv", sep="\t", quoting=3, usecols=["client_id", "sentence"])
#       for n in ("train", "validated", "test")}
# for n, t in cv.items():
#     t["norm"] = t.sentence.fillna("").map(normalize_amharic)
#     print(n, "| clips:", len(t), "| speakers:", t.client_id.nunique())

# idx = list(df.index[df.utterance_id.isna()])
# lab = ds.select_columns(["labels"]).select(idx)["labels"]
# dec = [tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip() for x in lab]
# for n, t in cv.items():
#     print("share of the 553 found in CV", n, ":", round(sum(s in set(t.norm) for s in dec) / len(dec), 3))
import pandas as pd
d = "cv-corpus-25.0-2026-03-09/am/"
t = pd.read_csv(d + "train.tsv", sep="\t", quoting=3, usecols=["path", "client_id"])
cd = pd.read_csv(d + "clip_durations.tsv", sep="\t")
print(cd.columns.tolist())                                   # usually: clip, duration[ms]
m = t.merge(cd, left_on="path", right_on="clip")
print("used clips:", len(t), "| with duration:", len(m),
      "| hours:", round(m["duration[ms]"].sum() / 3.6e6, 2), "| speakers:", t.client_id.nunique())
print("whole 25.0 corpus: clips", len(cd), "| hours", round(cd["duration[ms]"].sum() / 3.6e6, 2))