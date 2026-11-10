import json
from transformers import WhisperTokenizer

MANIFEST = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl"  # replace with the path to the test manifest

rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
print("manifest rows:", len(rows))  # expect 4588

tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3", language="amharic", task="transcribe")
subset = [i for i, r in enumerate(rows)
          if r["duration_sec"] <= 30 and len(tok(r["reference"]).input_ids) <= 448]
print("subset size:", len(subset))  # expect 688

open("splits/subset_688_idx.txt", "w").write("\n".join(map(str, subset)) + "\n")
open("all_idx.txt", "w").write("\n".join(map(str, range(len(rows)))) + "\n")