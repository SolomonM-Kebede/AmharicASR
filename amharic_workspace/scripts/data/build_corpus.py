from data.normalize import normalize_amharic
import os
from common import config as cfg  # must load before transformers/datasets
from datasets import load_dataset
from tqdm import tqdm

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

print("Downloading Amharic Wikipedia...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.am", split="train")

output_file = "/tmp/amharic_workspace/amharic_corpus.txt"
print(f"⚙️ Normalizing and saving to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    for row in tqdm(wiki):
        # Split articles by newline to preserve sentence boundaries
        lines = row["text"].split('\n')
        for line in lines:
            norm_text = normalize_amharic(line)
            # Only save lines that actually contain words
            if len(norm_text) > 5: 
                f.write(norm_text + "\n")

print(f"✅ Corpus successfully built!")