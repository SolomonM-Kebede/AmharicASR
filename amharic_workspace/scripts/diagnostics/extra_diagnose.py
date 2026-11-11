from data.normalize import normalize_amharic
import json, jiwer

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text):
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()

path = "~/AmharicASR/amharic_workspace/phase2b_leyu_standard/results/gojjam_merged_fixed_persample.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
print(f"n={len(rows)}")

refs_raw  = [r["reference"] for r in rows]
preds_raw = [r["prediction"] for r in rows]
refs_norm  = [normalize_amharic(r) for r in refs_raw]
preds_norm = [normalize_amharic(p) for p in preds_raw]

print(f"WER (raw, no normalization)   : {jiwer.wer(refs_raw, preds_raw)*100:.2f}%")
print(f"WER (normalized, matches doc2): {jiwer.wer(refs_norm, preds_norm)*100:.2f}%")
print(f"CER (normalized)              : {jiwer.cer(refs_norm, preds_norm)*100:.2f}%")

# also check per-sample, sorted worst-first, to see if a handful of catastrophic
# samples are dragging the average up despite most looking fine
per_sample = sorted(
    [(jiwer.wer([r], [p]), i, r, p) for i, (r, p) in enumerate(zip(refs_norm, preds_norm))],
    reverse=True
)
print("\nWorst 5 samples by WER:")
for w, i, r, p in per_sample[:20]:
    print(f"[{i}] wer={w*100:.1f}%")
    print(f"  REF : {r}")
    print(f"  PRED: {p}")