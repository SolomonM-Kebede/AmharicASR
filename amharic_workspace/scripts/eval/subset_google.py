from data.normalize import normalize_amharic
"""Select the 688-recording subset (<=30 s audio, <=448 label tokens) and score Google on it.

Missing Google outputs are reported two ways: excluded, and scored as full deletions.
"""
import json
from common import config as cfg  # must load before transformers/datasets
import evaluate
from transformers import WhisperTokenizer

MAN = "test_set_audio_export/manifest.jsonl"
RAW = "results/google/google_stt_raw_predictions.jsonl"
MAX_TOK = 448

CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text):
    for src, tgt in CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="amharic", task="transcribe")

man = [json.loads(l) for l in open(MAN, encoding="utf-8")]
subset = [m for m in man
          if m["duration_sec"] <= 30 and len(tok(m["reference"]).input_ids) <= MAX_TOK]

print(f"subset: {len(subset)}")
for d in sorted({m["dialect"] for m in subset}):
    print(f"  {d:12s} {sum(m['dialect'] == d for m in subset)}")
print("expected: 688 (Gojjam 185, Gonder 275, Wello 228)\n")

raw = {}
for l in open(RAW, encoding="utf-8"):
    r = json.loads(l)
    raw[r["path"]] = r

wer_m, cer_m = evaluate.load("wer"), evaluate.load("cer")

def score(rows, hyps):
    pred = [normalize_amharic(h) for h in hyps]
    ref = [normalize_amharic(m["reference"]) for m in rows]
    return (wer_m.compute(predictions=pred, references=ref) * 100,
            cer_m.compute(predictions=pred, references=ref) * 100)

have = [m for m in subset if m["path"] in raw]
missing = [m for m in subset if m["path"] not in raw]
w, c = score(have, [raw[m["path"]]["hypothesis"] for m in have])
print(f"Google, outputs only:        n={len(have)}  WER={w:.2f}  CER={c:.2f}")
w2, c2 = score(subset, [raw[m["path"]]["hypothesis"] if m["path"] in raw else "" for m in subset])
print(f"Google, missing = deletion:  n={len(subset)}  WER={w2:.2f}  CER={c2:.2f}")
print("missing from subset:", [m["path"] for m in missing])