from data.normalize import normalize_amharic

"""
Amharic ASR — Google Speech-to-Text evaluation on the SAME test set 
"""

import os
import io
import json
import time
import threading
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("API_KEY") 
if not API_KEY:
    raise ValueError("API_KEY is not set!")
CUSTOM_CACHE = os.environ.get("STT_CACHE_DIR", os.path.expanduser("~/AmharicASR/clean_cache"))
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

MANIFEST_PATH  = os.environ.get("MANIFEST_PATH", "~/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl")
RESULTS_DIR    = os.environ.get("RESULTS_DIR", "~/AmharicASR/amharic_workspace/phase2_leyu/results")
LANGUAGE_CODE  = os.environ.get("GOOGLE_STT_LANGUAGE_CODE", "am-ET")
MAX_SYNC_SECONDS = 55
MAX_WORKERS    = int(os.environ.get("MAX_WORKERS", 5))   
MAX_RETRIES    = int(os.environ.get("MAX_RETRIES", 4))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 50))

os.makedirs(RESULTS_DIR, exist_ok=True)

from google.cloud import speech
from google.api_core import exceptions as gexc

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


print(f"Loading manifest from {MANIFEST_PATH} ...", flush=True)
entries = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entries.append(json.loads(line))
print(f"  {len(entries):,} utterances", flush=True)

client = speech.SpeechClient(client_options={"api_key": API_KEY})
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16_000,
    language_code=LANGUAGE_CODE,
)


def transcribe_one(entry):
    if entry["duration_sec"] > MAX_SYNC_SECONDS:
        return {"status": "skipped_too_long", "entry": entry}

    backoff = 1.0
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with io.open(entry["path"], "rb") as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            response = client.recognize(config=config, audio=audio)
            hyp = " ".join(r.alternatives[0].transcript for r in response.results) if response.results else ""
            return {"status": "ok", "entry": entry, "hyp": hyp}
        except gexc.ResourceExhausted as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
        except Exception as e:
            last_error = str(e)
            break

    return {"status": "failed", "entry": entry, "error": last_error}


print(f"\nTranscribing with {MAX_WORKERS} concurrent workers ...", flush=True)
progress_lock = threading.Lock()
completed = 0
total = len(entries)
t_start = time.time()

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(transcribe_one, entry) for entry in entries]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
        with progress_lock:
            completed += 1
            if completed % PROGRESS_EVERY == 0 or completed == total:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (total - completed) / rate if rate > 0 else float("nan")
                print(f"  [{completed}/{total}]  ({rate:.2f} req/s, {elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

skipped_too_long = sum(1 for r in results if r["status"] == "skipped_too_long")
failed = sum(1 for r in results if r["status"] == "failed")
print(f"\nDone. Skipped (too long for sync API): {skipped_too_long}  Failed (after retries): {failed}", flush=True)

# Save raw predictions to disk
raw_predictions_path = os.path.join(RESULTS_DIR, "google_stt_raw_predictions.jsonl")
with open(raw_predictions_path, "w", encoding="utf-8") as f:
    for r in results:
        if r["status"] == "ok":
            f.write(json.dumps({
                "dialect": r["entry"]["dialect"],
                "reference": r["entry"]["reference"],
                "hypothesis": r["hyp"],
                "path": r["entry"]["path"],
            }, ensure_ascii=False) + "\n")
print(f"Raw predictions saved to {raw_predictions_path} (safe from here on regardless of what happens next)", flush=True)

results_by_dialect = {}
for r in results:
    if r["status"] != "ok":
        continue
    entry = r["entry"]
    hyp_norm = normalize_amharic(r["hyp"])
    ref_norm = normalize_amharic(entry["reference"])
    d = entry["dialect"]
    results_by_dialect.setdefault(d, {"pred": [], "ref": []})
    results_by_dialect[d]["pred"].append(hyp_norm)
    results_by_dialect[d]["ref"].append(ref_norm)

from common import config as cfg  # must load before transformers/datasets
import evaluate as _eval
wer_metric = _eval.load("wer")
cer_metric = _eval.load("cer")

final_results = {}
all_pred, all_ref = [], []
for dialect, data in results_by_dialect.items():
    if not data["pred"]:
        continue
    wer = wer_metric.compute(predictions=data["pred"], references=data["ref"]) * 100
    cer = cer_metric.compute(predictions=data["pred"], references=data["ref"]) * 100
    final_results[dialect] = {"n_samples": len(data["pred"]), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"  {dialect:15s}  n={len(data['pred']):4d}  WER={wer:.2f}%  CER={cer:.2f}%", flush=True)
    all_pred.extend(data["pred"])
    all_ref.extend(data["ref"])

overall_wer = wer_metric.compute(predictions=all_pred, references=all_ref) * 100
overall_cer = cer_metric.compute(predictions=all_pred, references=all_ref) * 100
final_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}
final_results["_meta"] = {"skipped_too_long": skipped_too_long, "failed": failed, "language_code": LANGUAGE_CODE}

print(f"\n{'='*60}\n  GOOGLE STT — RESULTS (on your held-out test set)\n{'='*60}", flush=True)
for dialect, m in final_results.items():
    if dialect == "_meta":
        continue
    print(f"  {dialect:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%", flush=True)

out_path = os.path.join(RESULTS_DIR, "google_stt_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {out_path}", flush=True)