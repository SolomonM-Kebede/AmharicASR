from data.normalize import normalize_amharic
import json, collections
from common import config as cfg  # must load before transformers/datasets
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

te = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
rows = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]

cache = collections.defaultdict(set)
for i in range(len(te)):
    cache[(te["dialect"][i], te["norm_text"][i])].add(te["speaker_id"][i])

speaker_map, missing, ambiguous = {}, 0, 0
for j, r in enumerate(rows):
    speakers = cache.get((r["dialect"], normalize_amharic(r["reference"])), set())
    if not speakers:
        missing += 1
    elif len(speakers) == 1:
        speaker_map[str(j)] = next(iter(speakers))
    else:
        ambiguous += 1

print("matched:", len(speaker_map), "| missing:", missing, "| ambiguous:", ambiguous)
if missing == 0 and ambiguous == 0:
    json.dump(speaker_map, open("speaker_map.json", "w", encoding="utf-8"))
    print("wrote speaker_map.json with", len(speaker_map), "entries")