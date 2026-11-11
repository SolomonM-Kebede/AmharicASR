
import json

man = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]
scored = {json.loads(l)["path"]
          for l in open("results/google/google_stt_raw_predictions.jsonl", encoding="utf-8")}

over60 = [m for m in man if m["duration_sec"] > 60]
under60 = [m for m in man if m["duration_sec"] <= 60]
failed = [m["path"] for m in under60 if m["path"] not in scored]

print("manifest:", len(man), "| >60 s:", len(over60), "| <=60 s:", len(under60), "| scored:", len(scored))
print("failed (<=60 s, no output):", len(failed))
for p in failed:
    print(p)