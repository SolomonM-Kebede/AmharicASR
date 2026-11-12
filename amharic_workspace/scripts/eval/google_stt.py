"""Google Speech-to-Text (synchronous API) on the Leyu test recordings.

Sends each test recording of at most cfg.GOOGLE_MAX_SYNC_SECONDS (55 s) to the
API, keeps the raw transcripts, and scores them with the shared normalization.
The API accepts at most 60 s per request; the 55 s rule leaves a margin. Of the
4,588 test recordings, 4,198 are sent and 390 are not (147 of them exceed 60 s).

Two modes:
  default     Sends requests and writes the raw transcripts. Each request is
              billed, so the mode refuses to overwrite an existing raw file.
  --rescore   Scores an existing raw file without calling the API. Use this to
              reproduce a reported figure from the committed raw predictions.

--subset restricts scoring to the 688 recordings that fit both the 30 s window
and the 448-token reference limit (splits/subset_688_idx.txt). Those are the
only recordings on which Google and the Whisper systems can be compared
directly. The same raw file serves the full set and the subset.

Outputs in results/google/:
  <name>_raw_predictions.jsonl   one row per returned recording: orig_idx,
                                 dialect, reference, hypothesis, path
                                 (written by the default mode only)
  <name>_results.json            per-dialect and overall WER and CER, with _meta

Run from the workspace root after sourcing env.amh.sh. The API key is read from
GOOGLE_API_KEY and is never written to a file.
"""
from common import config as cfg  # sets cache and offline flags; keep first

import argparse
import concurrent.futures
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import evaluate
from google.api_core import exceptions as gexc
from google.cloud import speech

from data.normalize import normalize_amharic

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED_TEST = 4588
EXPECTED_SENT = 4198
SUBSET_SIZE = 688
MAX_SYNC_SECONDS = cfg.GOOGLE_MAX_SYNC_SECONDS
LANGUAGE_CODE = os.environ.get("GOOGLE_STT_LANGUAGE_CODE", "am-ET")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="google_stt",
                    help="output prefix in results/google/")
    ap.add_argument("--rescore", action="store_true",
                    help="score an existing raw file without calling the API")
    ap.add_argument("--raw", default="google_stt_raw_predictions.jsonl",
                    help="raw file to rescore, relative to results/google/")
    ap.add_argument("--subset", action="store_true",
                    help="score only the 688 complete-recording subset")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--retries", type=int, default=4,
                    help="retries per request on quota errors")
    ap.add_argument("--allow-failures", action="store_true",
                    help="write results even if some requests failed")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def load_manifest():
    path = cfg.require(cfg.MANIFESTS / "test_manifest.jsonl")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != EXPECTED_TEST:
        raise SystemExit(f"manifest has {len(rows)} rows, expected {EXPECTED_TEST}")
    sent = sum(r["duration_sec"] <= MAX_SYNC_SECONDS for r in rows)
    if sent != EXPECTED_SENT:
        raise SystemExit(f"{sent} recordings at most {MAX_SYNC_SECONDS} s, expected {EXPECTED_SENT}")
    return rows


def load_subset(rows):
    """Indices of the 688 complete-recording subset, checked against the manifest."""
    path = cfg.require(cfg.SUBSET_688_IDX)
    idx = {int(x) for x in path.read_text().split()}
    if len(idx) != SUBSET_SIZE:
        raise SystemExit(f"{path} lists {len(idx)} indices, expected {SUBSET_SIZE}")
    too_long = [i for i in idx if rows[i]["duration_sec"] > 30]
    if too_long:
        raise SystemExit(f"{len(too_long)} subset recordings exceed 30 s; "
                         f"the index file does not match the manifest")
    return idx


def check_audio(rows):
    missing = [r["path"] for r in rows
               if r["duration_sec"] <= MAX_SYNC_SECONDS and not (cfg.DATA_DIR / r["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} recordings to send are missing, first: {missing[0]}")


def transcribe(client, config, idx, entry, retries):
    """One recording. Quota errors are retried with backoff; other errors are not."""
    if entry["duration_sec"] > MAX_SYNC_SECONDS:
        return idx, "too_long", None, None
    with open(cfg.DATA_DIR / entry["path"], "rb") as f:
        audio = speech.RecognitionAudio(content=f.read())
    delay, error = 1.0, None
    for attempt in range(retries + 1):
        try:
            response = client.recognize(config=config, audio=audio)
            hyp = (" ".join(r.alternatives[0].transcript for r in response.results)
                   if response.results else "")
            return idx, "ok", hyp, None
        except gexc.ResourceExhausted as e:
            error = str(e)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        except Exception as e:
            return idx, "failed", None, str(e)
    return idx, "failed", None, error


def fetch(rows, workers, retries):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set. Export it in your shell; "
                         "do not write it to a tracked file.")
    client = speech.SpeechClient(client_options={"api_key": api_key})
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16_000,
        language_code=LANGUAGE_CODE,
    )
    outcomes, t0 = {}, time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(transcribe, client, config, i, r, retries)
                   for i, r in enumerate(rows)]
        for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            idx, status, hyp, err = fut.result()
            outcomes[idx] = (status, hyp, err)
            if done % 50 == 0 or done == len(rows):
                print(f"  [{done}/{len(rows)}]  {done / (time.time() - t0):.2f} req/s", flush=True)
    return outcomes


def from_raw(rows, raw_path):
    """Rebuild outcomes from a raw file. Rows are matched by audio path."""
    index = {r["path"]: i for i, r in enumerate(rows)}
    outcomes = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            idx = index[o["path"]]
            if rows[idx]["duration_sec"] > MAX_SYNC_SECONDS:
                raise SystemExit(f"raw file contains a recording that was not sent: {o['path']}")
            outcomes[idx] = ("ok", o["hypothesis"], None)
    for i, r in enumerate(rows):
        if i not in outcomes:
            outcomes[i] = (("too_long", None, None) if r["duration_sec"] > MAX_SYNC_SECONDS
                           else ("failed", None, "absent from raw file"))
    return outcomes


def write_raw(rows, outcomes, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            status, hyp, _ = outcomes[i]
            if status == "ok":
                f.write(json.dumps({"orig_idx": i, "dialect": r["dialect"],
                                    "reference": r["reference"], "hypothesis": hyp,
                                    "path": r["path"]}, ensure_ascii=False) + "\n")
    print(f"raw predictions written to {path}")


def score(rows, outcomes, only=None):
    """Per-dialect and overall WER and CER. `only` limits rows to a set of indices."""
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    preds, refs = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        if only is not None and i not in only:
            continue
        status, hyp, _ = outcomes[i]
        if status != "ok":
            continue
        preds[r["dialect"]].append(normalize_amharic(hyp))
        refs[r["dialect"]].append(normalize_amharic(r["reference"]))

    unknown = set(preds) - set(DIALECTS)
    if unknown:
        raise SystemExit(f"unknown dialect labels: {sorted(unknown)}")

    results = {}
    for d in DIALECTS:
        if preds[d]:
            results[d] = {
                "n_samples": len(preds[d]),
                "wer": round(wer_metric.compute(predictions=preds[d], references=refs[d]) * 100, 2),
                "cer": round(cer_metric.compute(predictions=preds[d], references=refs[d]) * 100, 2),
            }
    all_p = [p for d in DIALECTS for p in preds[d]]
    all_r = [r for d in DIALECTS for r in refs[d]]
    results["OVERALL"] = {
        "n_samples": len(all_p),
        "wer": round(wer_metric.compute(predictions=all_p, references=all_r) * 100, 2),
        "cer": round(cer_metric.compute(predictions=all_p, references=all_r) * 100, 2),
    }
    return results


def main():
    args = parse_args()
    rows = load_manifest()

    name = args.name + ("_subset688" if args.subset else "")
    raw_path = cfg.GOOGLE / f"{args.name}_raw_predictions.jsonl"
    out_path = cfg.GOOGLE / f"{name}_results.json"

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} exists; pass --overwrite or choose --name")

    if args.rescore:
        outcomes = from_raw(rows, cfg.require(cfg.GOOGLE / args.raw))
    else:
        if raw_path.exists() and not args.overwrite:
            raise SystemExit(f"{raw_path} exists; pass --overwrite or choose --name")
        check_audio(rows)  # check before any request is billed
        outcomes = fetch(rows, args.workers, args.retries)

    counts = Counter(status for status, _, _ in outcomes.values())
    print(f"returned: {counts['ok']:,}  not sent (over {MAX_SYNC_SECONDS} s): {counts['too_long']:,}"
          f"  failed: {counts['failed']:,}")

    failed = sorted(i for i, (s, _, _) in outcomes.items() if s == "failed")
    if failed and not args.allow_failures:
        for i in failed[:5]:
            print(f"  failed: {rows[i]['path']}  ({outcomes[i][2]})")
        raise SystemExit(f"{len(failed)} requests failed; nothing written. "
                         f"Rerun, or pass --allow-failures.")
    if counts["ok"] != EXPECTED_SENT and not args.allow_failures:
        raise SystemExit(f"{counts['ok']} returned, expected {EXPECTED_SENT}")

    if not args.rescore:
        write_raw(rows, outcomes, raw_path)

    only = load_subset(rows) if args.subset else None
    results = score(rows, outcomes, only)
    results["_meta"] = {
        "mode": "rescore" if args.rescore else "api",
        "raw_file": str(cfg.GOOGLE / args.raw) if args.rescore else raw_path.name,
        "scope": f"688 complete-recording subset" if args.subset else "full test set (sent recordings)",
        "sent": counts["ok"],
        "not_sent_over_limit": counts["too_long"],
        "failed": counts["failed"],
        "max_sync_seconds": MAX_SYNC_SECONDS,
        "language_code": LANGUAGE_CODE,
        "metric": "evaluate wer/cer on normalized text, same counting as the Whisper runs",
    }
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    for d, m in results.items():
        if d != "_meta":
            print(f"  {d:<14} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()