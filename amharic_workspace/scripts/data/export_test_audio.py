"""Exports the Leyu test set as 16 kHz mono WAV files and writes the test manifest.

Two modes:
  default           decode the test recordings, write the WAV files, write the manifest.
                    Refuses to run if WAV files already exist, unless --overwrite is given.
  --manifest-only   the WAV files already exist. Check them and write the manifest only.
                    Use this to rewrite the manifest with relative paths without
                    rewriting the audio.

Before anything is written, references and dialect labels are checked row by row against
the Joint test split. Row i of the manifest is then row i of the Joint test split, which
the 688-recording subset indices and the bootstrap pairing rely on.

Test speakers come from the original speaker split: seed 42, stratified by gender within
each dialect. Per-dialect counts must be 654, 1237, 836, 1101 and 760 (4,588 in total).

Manifest fields: orig_idx, path, reference, dialect, duration_sec. Paths are relative to
cfg.DATA_DIR, so the manifest contains no machine-specific paths. Readers join them with
cfg.DATA_DIR.

Run from the workspace root after sourcing env.amh.sh.
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
SAMPLE_RATE = 16_000
DEFAULT_AUDIO_DIR = cfg.DATA_DIR / "test_set_audio_export"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    ap.add_argument("--manifest", type=Path, default=cfg.MANIFESTS / "test_manifest.jsonl")
    ap.add_argument("--manifest-only", action="store_true",
                    help="write the manifest for WAV files that already exist")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing manifest or existing WAV files")
    return ap.parse_args()


def wav_relpath(dialect, k):
    """Path of recording k of a dialect, relative to the audio directory."""
    stem = dialect.replace(" ", "_")
    return f"{stem}/{stem}_{k:05d}.wav"


def rel_to_data(path):
    try:
        return path.resolve().relative_to(cfg.DATA_DIR.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f"{path} is outside {cfg.DATA_DIR}; set --audio-dir under it")


def entry(orig, path, reference, dialect, seconds):
    return {"orig_idx": orig, "path": rel_to_data(path), "reference": reference,
            "dialect": dialect, "duration_sec": round(seconds, 2)}


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


def select_test():
    """One entry per dialect: the dataset (audio undecoded), test positions, normalized references."""
    selections = []
    for repo, dialect in DIALECTS:
        ds = load_dataset(repo, split="train", verification_mode="no_checks")
        ds = ds.cast_column("audio", Audio(decode=False))
        idx = test_indices(ds)
        if len(idx) != EXPECTED_COUNTS[dialect]:
            raise SystemExit(f"{dialect}: {len(idx)} test utterances, expected {EXPECTED_COUNTS[dialect]}")
        refs = [normalize_amharic(t or "") for t in ds.select(idx)["text"]]
        selections.append({"dialect": dialect, "ds": ds, "idx": idx, "refs": refs})
    return selections


def check_against_joint(selections):
    references = [r for s in selections for r in s["refs"]]
    dialects = [s["dialect"] for s in selections for _ in s["idx"]]
    if len(references) != EXPECTED_TOTAL:
        raise SystemExit(f"{len(references)} test utterances, expected {EXPECTED_TOTAL}")
    joint = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"].select_columns(["dialect", "norm_text"])
    if len(joint) != EXPECTED_TOTAL:
        raise SystemExit(f"Joint test split has {len(joint)} rows, expected {EXPECTED_TOTAL}")
    j_text, j_dialect = joint["norm_text"], joint["dialect"]
    bad = [i for i in range(EXPECTED_TOTAL)
           if references[i] != j_text[i].strip() or dialects[i] != j_dialect[i]]
    if bad:
        raise SystemExit(f"{len(bad)} rows differ from the Joint test split; first at index {bad[0]}. "
                         f"Nothing written.")
    print(f"manifest rows matching the Joint test split: {EXPECTED_TOTAL:,} of {EXPECTED_TOTAL:,}")


def load_wav(audio):
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    wav, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
    return wav


def export_audio(selections, audio_dir, overwrite):
    existing = list(audio_dir.rglob("*.wav")) if audio_dir.exists() else []
    if existing and not overwrite:
        raise SystemExit(f"{len(existing)} WAV files already in {audio_dir}. Use --manifest-only to "
                         f"write the manifest for them, or --overwrite to replace them.")
    entries, orig = [], 0
    for s in selections:
        for k, row in enumerate(s["ds"].select(s["idx"])):
            path = audio_dir / wav_relpath(s["dialect"], k)
            path.parent.mkdir(parents=True, exist_ok=True)
            wav = load_wav(row["audio"])
            sf.write(path, np.asarray(wav, dtype=np.float32), SAMPLE_RATE)
            entries.append(entry(orig, path, s["refs"][k], s["dialect"], len(wav) / SAMPLE_RATE))
            orig += 1
        print(f"  {s['dialect']}: {len(s['idx'])} files")
    return entries


def manifest_from_files(selections, audio_dir):
    entries, orig, missing = [], 0, []
    for s in selections:
        for k in range(len(s["idx"])):
            path = audio_dir / wav_relpath(s["dialect"], k)
            if not path.exists():
                missing.append(path)
            else:
                entries.append(entry(orig, path, s["refs"][k], s["dialect"], sf.info(path).duration))
            orig += 1
    if missing:
        raise SystemExit(f"{len(missing)} WAV files missing, first {missing[0]}")
    print(f"found {len(entries):,} WAV files in {audio_dir}")
    return entries


def main():
    args = parse_args()
    if args.manifest.exists() and not args.overwrite:
        raise SystemExit(f"{args.manifest} exists; pass --overwrite or choose --manifest")

    selections = select_test()
    check_against_joint(selections)
    if args.manifest_only:
        entries = manifest_from_files(selections, args.audio_dir)
    else:
        entries = export_audio(selections, args.audio_dir, args.overwrite)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(entries):,} rows to {args.manifest}")


if __name__ == "__main__":
    main()