"""Exports the Leyu test set as 16 kHz mono WAV files and a manifest, after
checking that the manifest matches the Joint test split row by row.

The manifest is the shared input for the zero-shot system, the Google system,
and the 688-recording subset, whose indices are positions in this manifest.
Row i must be row i of the Joint test split. The check runs before any audio
is written, and nothing is written if it fails.

Test speakers are reproduced with the original speaker split: seed 42,
stratified by gender within each dialect. Per-dialect counts must be
654, 1237, 836, 1101 and 760 (4,588 in total).

Writes <audio-dir>/<dialect>/<dialect>_<k>.wav and the manifest with fields
orig_idx, path, reference, dialect, duration_sec. Run from the workspace root
after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import io
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split

from data.normalize import normalize_amharic

DIALECTS = [
    ("leyu-amharic/leyu-amharic-gojjam-dialect", "gojjam"),
    ("leyu-amharic/leyu-amharic-gonder-dialect", "gonder"),
    ("leyu-amharic/leyu-amharic-wello-dialect", "wello"),
    ("leyu-amharic/leyu-amharic-shewa-dialect", "shewa"),
    ("leyu-amharic/leyu-amharic-addis-ababa-dialect", "addis ababa"),
]
EXPECTED_COUNTS = {"gojjam": 654, "gonder": 1237, "wello": 836, "shewa": 1101, "addis ababa": 760}
EXPECTED_TOTAL = 4588
TEST_SIZE = 0.10


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, default=cfg.ROOT / "test_set_audio_export")
    ap.add_argument("--manifest", type=Path, default=cfg.MANIFESTS / "test_manifest.jsonl")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def test_indices(ds):
    """Positions of the test speakers in one dialect, by the original speaker split."""
    spk = ds.select_columns(["speaker_id", "gender"]).to_pandas()
    spk = spk.drop_duplicates(subset="speaker_id").reset_index(drop=True)
    spk["gender"] = spk["gender"].fillna("unknown")
    ids, genders = spk["speaker_id"].tolist(), spk["gender"].tolist()
    try:
        _, test_ids = train_test_split(ids, test_size=TEST_SIZE, random_state=cfg.SEED, stratify=genders)
    except ValueError:
        _, test_ids = train_test_split(ids, test_size=TEST_SIZE, random_state=cfg.SEED)
    keep = set(test_ids)
    return [i for i, s in enumerate(ds["speaker_id"]) if s in keep]


def main():
    args = parse_args()
    if args.manifest.exists() and not args.overwrite:
        raise SystemExit(f"{args.manifest} exists; pass --overwrite or choose --manifest")

    selections = []
    for repo, dialect in DIALECTS:
        ds = load_dataset(repo, split="train", verification_mode="no_checks")
        idx = test_indices(ds)
        if len(idx) != EXPECTED_COUNTS[dialect]:
            raise SystemExit(f"{dialect}: {len(idx)} test utterances, expected {EXPECTED_COUNTS[dialect]}")
        texts = ds.select(idx).select_columns(["text"])["text"]
        selections.append((dialect, repo, idx, [normalize_amharic(t or "") for t in texts]))

    references = [r for _, _, _, refs in selections for r in refs]
    dialects = [d for d, _, idx, _ in selections for _ in idx]
    if len(references) != EXPECTED_TOTAL:
        raise SystemExit(f"{len(references)} test utterances, expected {EXPECTED_TOTAL}")

    joint = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"].select_columns(["dialect", "norm_text"])
    joint_text, joint_dialect = joint["norm_text"], joint["dialect"]
    if len(joint_text) != EXPECTED_TOTAL:
        raise SystemExit(f"Joint test split has {len(joint_text)} rows, expected {EXPECTED_TOTAL}")
    mismatched = [i for i in range(EXPECTED_TOTAL)
                  if references[i] != joint_text[i].strip() or dialects[i] != joint_dialect[i]]
    if mismatched:
        raise SystemExit(f"{len(mismatched)} rows differ from the Joint test split; "
                         f"first at index {mismatched[0]}. Nothing written.")
    print(f"manifest rows matching the Joint test split: {EXPECTED_TOTAL:,} of {EXPECTED_TOTAL:,}")

    entries, orig = [], 0
    for dialect, repo, idx, _ in selections:
        ds = load_dataset(repo, split="train", verification_mode="no_checks")
        ds = ds.cast_column("audio", Audio(decode=False)).select(idx)
        out_dir = args.audio_dir / dialect.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        for k, row in enumerate(ds):
            a = row["audio"]
            src = io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"]
            wav, _ = librosa.load(src, sr=16_000, mono=True)
            path = out_dir / f"{dialect.replace(' ', '_')}_{k:05d}.wav"
            sf.write(path, np.asarray(wav, dtype=np.float32), 16_000)
            entries.append({"orig_idx": orig, "path": str(path), "reference": references[orig],
                            "dialect": dialect, "duration_sec": round(len(wav) / 16_000, 2)})
            orig += 1
        print(f"  {dialect}: {len(idx)} files")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries):,} rows to {args.manifest}")


if __name__ == "__main__":
    main()