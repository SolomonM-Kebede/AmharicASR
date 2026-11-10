"""
Amharic ASR — Phase 1 standard-dialect TEST split rebuild
============================================================
Reconstructs the original ALFFA+CV speaker-disjoint test split (dataset_cache
was under /tmp and did not survive). Mirrors the original Phase 1 training
script's combine/filter/split logic exactly, including SEED=42, so this
should reproduce the same held-out test set. Needs network access for the
ALFFA Hub download — run where internet is available (login node), not
inside an offline-mode GPU job.
"""

# import os, random
# from collections import defaultdict
# from pathlib import Path

# import pandas as pd
# from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, Audio
# from transformers import WhisperProcessor

# CUSTOM_CACHE = os.path.expanduser("~/AmharicASR/hf_cache")
# os.makedirs(CUSTOM_CACHE, exist_ok=True)
# os.environ["HF_HOME"] = CUSTOM_CACHE
# os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
# os.environ["HUGGINGFACE_HUB_CACHE"] = CUSTOM_CACHE   # legacy name, harmless if unused
# os.environ["TRANSFORMERS_CACHE"] = CUSTOM_CACHE       # legacy name, harmless if unused
# os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
# os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE

# MODEL_NAME   = "openai/whisper-large-v3"
# LOCAL_CV_DIR = "cv-corpus-25.0-2026-03-09/am"  
# OUT_DIR      = "phase1_curriculum/dataset_cache_rebuilt"

# MAX_TEXT_CHARS   = 350
# MAX_LABEL_LENGTH = 448
# SEED             = 42

# os.makedirs(OUT_DIR, exist_ok=True)

# print("Loading processor ...")
# processor = WhisperProcessor.from_pretrained(
#     MODEL_NAME, language="amharic", task="transcribe", cache_dir=CUSTOM_CACHE
# )
# DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

# AMHARIC_CHAR_MAP = {
#     "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
#     "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
#     "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
#     "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
#     "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
# }
# PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

# def normalize_amharic(text):
#     for src, tgt in AMHARIC_CHAR_MAP.items():
#         text = text.replace(src, tgt)
#     for p in PUNCT:
#         text = text.replace(p, " ")
#     return " ".join(text.split()).strip()

# print("Loading ALFFA ...")
# alffa = load_dataset("hadamard-2/alffa-amharic", split="train", cache_dir=CUSTOM_CACHE)
# alffa = alffa.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
# alffa = alffa.rename_column("transcript", "text")
# alffa = alffa.map(
#     lambda x, i: {"dialect": "standard", "gender": "unknown", "speaker_id": f"alffa_{i}"},
#     with_indices=True,
# )
# alffa = alffa.remove_columns([c for c in alffa.column_names
#                               if c not in ["audio", "text", "dialect", "gender", "speaker_id"]])

# print("Loading Common Voice ...")
# cv_tsv = os.path.join(LOCAL_CV_DIR, "train.tsv")
# if not os.path.exists(cv_tsv):
#     raise FileNotFoundError(f"Common Voice TSV not found at {cv_tsv} — locate the raw corpus first.")

# cv_df = pd.read_csv(cv_tsv, sep="\t")[["client_id", "path", "sentence", "gender"]]
# cv_df = cv_df.rename(columns={"client_id": "speaker_id", "sentence": "text"})
# cv_df["dialect"] = "standard"
# cv_df["gender"]  = cv_df["gender"].fillna("unknown")

# sample_bare   = os.path.join(LOCAL_CV_DIR, "clips", cv_df["path"].iloc[0])
# sample_direct = os.path.join(LOCAL_CV_DIR, cv_df["path"].iloc[0])
# if os.path.exists(sample_bare):
#     cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(LOCAL_CV_DIR, "clips", p))
# elif os.path.exists(sample_direct):
#     cv_df["audio"] = cv_df["path"].apply(lambda p: os.path.join(LOCAL_CV_DIR, p))
# else:
#     raise FileNotFoundError("Cannot resolve CV audio path.")

# cv_df = cv_df.drop(columns=["path"])
# cv_dataset = Dataset.from_dict(cv_df.to_dict("list"))
# cv_dataset = cv_dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

# combined = concatenate_datasets([alffa, cv_dataset])
# combined = combined.filter(lambda x: x["text"] and len(x["text"]) <= MAX_TEXT_CHARS)

# speakers = defaultdict(list)
# for i, row in enumerate(combined):
#     speakers[row["speaker_id"]].append(i)

# spk_ids = list(speakers.keys())
# random.Random(SEED).shuffle(spk_ids)
# n = len(spk_ids)
# n_test = max(1, int(n * 0.10))
# n_val  = max(1, int(n * 0.10))

# test_spk = set(spk_ids[:n_test])
# test_idx = [i for s, ii in speakers.items() if s in test_spk for i in ii]

# test_raw = combined.select(test_idx)
# print(f"Reconstructed test split: {len(test_raw)} examples")

# def prepare_batch(batch):
#     import librosa, io as _io
#     audio_data = batch["audio"]
#     try:
#         if audio_data.get("bytes") is not None:
#             audio_arr, _ = librosa.load(_io.BytesIO(audio_data["bytes"]), sr=16_000, mono=True)
#         elif audio_data.get("path") is not None:
#             audio_arr, _ = librosa.load(audio_data["path"], sr=16_000, mono=True)
#         else:
#             audio_arr = np.array(audio_data["array"], dtype="float32")
#     except Exception:
#         import numpy as np
#         audio_arr = np.zeros(16_000, dtype="float32")

#     batch["input_features"] = processor.feature_extractor(
#         audio_arr, sampling_rate=16_000, return_tensors="pt"
#     ).input_features[0]
#     batch["labels"] = processor.tokenizer(
#         normalize_amharic(batch["text"]), max_length=MAX_LABEL_LENGTH, truncation=True
#     ).input_ids
#     return batch

# print("Extracting features (num_proc=4) ...")
# test_dataset = test_raw.map(
#     prepare_batch,
#     remove_columns=["text", "audio", "dialect", "speaker_id", "gender"],
#     num_proc=4,
# )
# DatasetDict({"test": test_dataset}).save_to_disk(OUT_DIR)
# print(f"Saved to {OUT_DIR}")

from huggingface_hub import HfApi
api = HfApi()
for c in api.list_repo_commits("hadamard-2/alffa-amharic", repo_type="dataset")[:10]:
    print(c.created_at, c.title)