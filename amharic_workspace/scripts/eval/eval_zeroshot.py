"""Zero-shot Whisper Large-v3 (no fine-tuning) on the Leyu test set.

Reads the audio and references written by export_test_audio.py, so this system
and the Google system are scored on the same recordings. Writes
results/persample/<name>_persample.jsonl and <name>.json.

Settings: openai/whisper-large-v3 as published. Weights load in the dtype stored
in the checkpoint config (float16 for this checkpoint). Inference runs under bf16
autocast, as the original run did, so the label is "float16 weights, bf16
autocast". Beam 5, 448-token limit, no repetition penalty.

The record of the reported run (174.34%) is in results/summaries/ with its log.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data.normalize import normalize_amharic

DIALECTS = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
EXPECTED_COUNTS = {"gojjam": 654, "gonder": 1237, "wello": 836, "shewa": 1101, "addis ababa": 760}
EXPECTED_TOTAL = 4588
NUM_BEAMS = 5


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=cfg.MANIFESTS / "test_manifest.jsonl")
    ap.add_argument("--name", default="zeroshot", help="output prefix")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != EXPECTED_TOTAL:
        raise SystemExit(f"manifest has {len(rows)} rows, expected {EXPECTED_TOTAL}")
    for i, r in enumerate(rows):
        r["orig_idx"] = i  # position in the manifest; the same index the bootstrap uses
    return rows


def transcribe(model, processor, rows, batch_size):
    hyps = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        wavs = []
        for r in chunk:
            wav, sr = sf.read(cfg.DATA_DIR / r["path"], dtype="float32")
            if sr != 16_000:
                raise ValueError(f"{r['path']} is {sr} Hz, expected 16 kHz")
            wavs.append(wav)
        feats = processor.feature_extractor(wavs, sampling_rate=16_000, return_tensors="pt").input_features
        feats = feats.to("cuda", dtype=model.dtype)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(feats)
        hyps += [normalize_amharic(s) for s in processor.batch_decode(ids, skip_special_tokens=True)]
        if (start // batch_size) % 50 == 0:
            print(f"  {start + len(chunk):5d}/{len(rows)}", flush=True)
    return hyps


def corpus_wer_cer(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)


def main():
    args = parse_args()
    persample_path = cfg.PERSAMPLE / f"{args.name}_persample.jsonl"
    summary_path = cfg.PERSAMPLE / f"{args.name}.json"
    for p in (persample_path, summary_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite or choose --name")

    rows = read_manifest(cfg.require(args.manifest))
    from collections import Counter
    counts = Counter(r["dialect"] for r in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"dialect counts {dict(counts)} differ from {EXPECTED_COUNTS}")

    processor = WhisperProcessor.from_pretrained(cfg.WHISPER_MODEL, language="amharic", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(cfg.WHISPER_MODEL).to("cuda").eval()
    print(f"weights in memory: {model.dtype}; inference: bf16 autocast")
    gc = model.generation_config
    gc.forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    gc.num_beams = NUM_BEAMS
    gc.repetition_penalty = 1.0
    gc.no_repeat_ngram_size = 0
    gc.max_length = cfg.MAX_LABEL_LENGTH
    gc.max_new_tokens = None
    gc.use_cache = True

    hyps = transcribe(model, processor, rows, args.batch_size)
    refs = [r["reference"] for r in rows]

    by_dialect = defaultdict(lambda: ([], []))
    for r, h in zip(rows, hyps):
        by_dialect[r["dialect"]][0].append(r["reference"])
        by_dialect[r["dialect"]][1].append(h)

    summary = {}
    for d in DIALECTS:
        wer, cer = corpus_wer_cer(*by_dialect[d])
        summary[d] = {"n": len(by_dialect[d][0]), "wer": round(wer, 2), "cer": round(cer, 2)}
        print(f"{d:12s} n={summary[d]['n']:5d}  WER={wer:6.2f}%  CER={cer:6.2f}%")

    wer, cer = corpus_wer_cer(refs, hyps)
    longest = max(len(processor.tokenizer(h, add_special_tokens=False).input_ids) for h in hyps)
    summary["OVERALL"] = {"n": len(refs), "wer": round(wer, 2), "cer": round(cer, 2),
                          "weights": "float16 as stored", "inference": "bf16 autocast",
                          "beams": NUM_BEAMS, "max_length": cfg.MAX_LABEL_LENGTH,
                          "longest_prediction_tokens": longest, "model": cfg.WHISPER_MODEL}
    print(f"{'OVERALL':12s} n={len(refs):5d}  WER={wer:6.2f}%  CER={cer:6.2f}%  longest={longest} tokens")

    with open(persample_path, "w", encoding="utf-8") as out:
        for r, h in zip(rows, hyps):
            out.write(json.dumps({"orig_idx": r["orig_idx"], "dialect": r["dialect"],
                                  "reference": r["reference"], "prediction": h}, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {persample_path}\nsaved {summary_path}")


if __name__ == "__main__":
    main()