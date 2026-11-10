"""Rerun the two Google STT recordings that failed in the main run.

Same client, config and retry logic as google_stt.py. Runs sequentially.
Appends successes to google_stt_raw_predictions.jsonl (after a .bak copy)
and writes the outcome of each attempt to rerun_outcomes.json.
"""
import os, io, json, time, shutil
from dotenv import load_dotenv
from google.cloud import speech
from google.api_core import exceptions as gexc

load_dotenv()
API_KEY = os.environ["API_KEY"]
MANIFEST_PATH = os.path.expanduser(os.environ.get(
    "MANIFEST_PATH", "~/AmharicASR/amharic_workspace/test_set_audio_export/manifest.jsonl"))
RESULTS_DIR = os.path.expanduser(os.environ.get(
    "RESULTS_DIR", "~/AmharicASR/amharic_workspace/phase2_leyu/results"))
LANGUAGE_CODE = os.environ.get("GOOGLE_STT_LANGUAGE_CODE", "am-ET")
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 4))
TARGETS = ["gojjam/gojjam_00046.wav", "gonder/gonder_00631.wav"]

RAW = os.path.join(RESULTS_DIR, "google_stt_raw_predictions.jsonl")
OUT = os.path.join(RESULTS_DIR, "rerun_outcomes.json")

entries = [json.loads(l) for l in open(MANIFEST_PATH, encoding="utf-8")]
targets = [e for e in entries if any(e["path"].endswith(t) for t in TARGETS)]
assert len(targets) == 2, f"expected 2 manifest rows, found {len(targets)}"

client = speech.SpeechClient(client_options={"api_key": API_KEY})
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16_000,
    language_code=LANGUAGE_CODE,
)

def transcribe_one(entry):
    backoff, last_error = 1.0, None
    for attempt in range(MAX_RETRIES + 1):
        try:
            with io.open(entry["path"], "rb") as f:
                content = f.read()
            response = client.recognize(config=config,
                                        audio=speech.RecognitionAudio(content=content))
            hyp = " ".join(r.alternatives[0].transcript for r in response.results) \
                  if response.results else ""
            return {"status": "ok", "hyp": hyp}
        except gexc.ResourceExhausted as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
        except Exception as e:
            last_error = str(e)
            break
    return {"status": "failed", "error": last_error}

outcomes = {}
for e in targets:
    r = transcribe_one(e)
    outcomes[e["path"]] = {"duration_sec": e["duration_sec"], **r}
    print(e["path"], r["status"], repr(r.get("hyp", r.get("error", "")))[:200], flush=True)

existing = open(RAW, encoding="utf-8").read()
shutil.copy(RAW, RAW + ".bak")
appended = 0
with open(RAW, "a", encoding="utf-8") as f:
    for e in targets:
        r = outcomes[e["path"]]
        if r["status"] == "ok" and e["path"] not in existing:
            f.write(json.dumps({
                "dialect": e["dialect"],
                "reference": e["reference"],
                "hypothesis": r["hyp"],
                "path": e["path"],
            }, ensure_ascii=False) + "\n")
            appended += 1

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(outcomes, f, ensure_ascii=False, indent=2)
print(f"\nAppended {appended} row(s). Backup: {RAW}.bak. Outcomes: {OUT}", flush=True)