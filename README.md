# Amharic ASR: Dialectal Adaptation of Whisper Large-v3

Code, per-utterance predictions, and analysis for a bachelor thesis on adapting
Whisper Large-v3 to five regional varieties of Amharic (Gojjam, Gonder, Wello,
Shewa, Addis Ababa) with LoRA and full fine-tuning, using the Leyu corpus.

The thesis reports word error rates (WER) on held-out Leyu test sets under three
protocols: speaker-independent (SI), text-independent (TI), and jointly speaker-
and text-independent (Joint). The paired comparisons in the thesis can be checked
from the per-utterance files in this repository without a GPU.

All code lives in `amharic_workspace/`.

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/analysis/` | Paired bootstrap (`run_bootstrap.py`), per-dialect comparison, truncation analysis, figures |
| `scripts/data/` | Split and subset builders, test-audio export, text normalization |
| `scripts/train/` | Training scripts for the runs in the experimental grid |
| `scripts/eval/` | Evaluation scripts for Whisper runs, zero-shot baseline, Google Speech-to-Text |
| `scripts/merge/` | LoRA merge and precision handling |
| `scripts/diagnostics/` | Checks cited in the thesis (normalization, Common Voice durations, overlap) |
| `scripts/common/` | Configuration and shared bootstrap helpers |
| `results/persample/` | Per-utterance predictions for each evaluated system |
| `results/google/` | Google Speech-to-Text predictions and rescoring outputs |
| `results/summaries/` | Saved summaries and logs for each reported result |
| `splits/` | 688-recording subset index, split manifests, split counts |
| `figures/` | Figures used in the thesis |

## Setup

```bash
git clone https://github.com/SolomonM-Kebede/AmharicASR.git
cd AmharicASR/amharic_workspace
pip install -r requirements.txt
cp env.example.sh env.sh      # set the paths for your machine
source env.sh
```

Python 3.12 and PyTorch 2.13 (CUDA 13.0 build) were used. Scripts are run from
`amharic_workspace/` with `PYTHONPATH=scripts`.

The bootstrap and figure scripts read the Joint split cache, which is not stored
in this repository (see *Rebuilding the split caches*). The cache is needed for
speaker identities and for the check that each stored reference matches the
test text.

## Checking the reported comparisons

```bash
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py
```

Default comparison: LoRA (C2) against full fine-tuning (C4b), bfloat16 inference.

| Quantity | Value |
|---|---|
| WER, C2 (LoRA) | 32.06% |
| WER, C4b (full fine-tuning) | 31.16% |
| Difference (C2 − C4b) | +0.89 points |
| Utterance-level 95% interval | [0.68, 1.11] |
| Speaker-level 95% interval | [0.59, 1.20] |

The script prints WER(X) − WER(Y), so a positive difference means Y has the lower WER.
The speaker-level interval is the primary one, because utterances from the same
speaker are not independent.

Other scopes and comparisons:

```bash
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py --subset
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py --outside-subset
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py --exclude-val-overlap
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py --x a2_persample.jsonl --y d1_persample.jsonl
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py --x d1_persample.jsonl --y c1_persample.jsonl
PYTHONPATH=scripts python scripts/analysis/perdialect_bootstrap.py
```

Add `--json <path>` to any run to save its result. The printed sign is always
first minus second. The thesis tables use second minus first, so their signs are reversed.

| Scope or pair | Printed difference | Speaker-level interval (full set) |
|---|---|---|
| C2 vs C4b, full set | +0.89 | [0.59, 1.20] |
| C2 vs C4b, 688 subset | +1.24 | — (utterance-level [0.45, 2.04]) |
| C2 vs C4b, outside subset (3,900) | +0.87 | — (utterance-level [0.65, 1.09]) |
| C2 vs C4b, validation overlap excluded (3,992) | +0.87 | — (utterance-level [0.64, 1.11]) |
| A2 vs D1 | −13.24 | [−14.73, −11.35] |
| D1 vs C1 | −4.19 | [−4.62, −3.66] |
| A2 vs C1 | −17.43 | [−19.25, −15.15] |

The per-dialect comparison (`perdialect_bootstrap.py`) reports utterance-level
intervals for Gojjam, Gonder, Wello, Shewa and Addis Ababa.

## Rebuilding the split caches

The Leyu corpus is not in this repository, so the split caches are rebuilt from
the Leyu datasets on the Hugging Face Hub. Run these from `amharic_workspace/`:

```bash
PYTHONPATH=scripts python scripts/data/build_joint_split.py   # Joint cache and split counts
PYTHONPATH=scripts python scripts/data/build_ti_split.py      # TI cache, checked against the recorded manifest
PYTHONPATH=scripts python scripts/data/build_si_size_matched.py  # D1 cache (size-matched SI sample)
PYTHONPATH=scripts python scripts/data/build_subset.py        # 688-recording subset index
```

Each builder refuses to overwrite an existing cache unless `OVERWRITE=1` is set.
The Joint build should reproduce the counts in `splits/joint_split_counts.json`:
38,948 SI training utterances, 10,296 Joint training utterances, 5,942 validation,
and 4,588 test.

The SI training cache is not kept. Its test partition is identical to the Joint
test partition, so the SI results can be checked from the per-utterance files.

## Evaluation and training

Training and evaluation scripts in `scripts/train/` and `scripts/eval/` need a GPU
and the model checkpoints, which are not in this repository. Each evaluation
writes its per-utterance file and a summary to `results/persample/`, using the
same normalization and counting as the bootstrap.

Google Speech-to-Text (`scripts/eval/google_stt.py`) needs a `GOOGLE_API_KEY`
environment variable and is billed per request. Its `--rescore` and `--subset`
modes recompute the reported scores from the committed predictions without any API calls.

```bash
PYTHONPATH=scripts python scripts/eval/google_stt.py --rescore --subset --name google_stt_rescore --overwrite
```

## Configuration

Paths are defined in `scripts/common/config.py` and can be overridden with the
`AMHARIC_*` variables in `env.sh`. The Hugging Face cache is set on import, before
any transformers or datasets code loads. Inputs are checked when a script uses
them, so a missing input affects only the scripts that need it.

## Data

- **Leyu** (dialectal prompted read speech, five varieties, 49,478 utterances from
  295 speakers). Licensed under CC BY 4.0. This repository includes derived data
  (per-utterance predictions and split manifests) but not the audio. The audio
  and the full corpus are available from the Hugging Face datasets.
- **ALFFA** (Standard Amharic read speech) and **Mozilla Common Voice 25.0 Amharic**
  (CC0). Obtained from their public releases. See `DATA_LICENCES.md` for the
  licence of each source and of the derived files.

The Common Voice pool used by C2 and C3 combines all five clip-level files without
deduplication, so some clips appear more than once. This is documented in the
thesis.

## Known limitations

- The SI training cache is not kept (see above). The SI results can be checked from
  the per-utterance files, but the training run cannot be repeated from this repository.
- Results come from single training runs, except where stated.
- Checkpoint selection differs by run: 800 validation utterances for most LoRA runs,
  the full validation set for C4, and 800 utterances for C4b.
- The A4 checkpoint was not retained, so its result is reported from its saved summary only.
- Google Speech-to-Text was scored on the 4,198 recordings of at most 55 seconds,
  which is inside the API's 60-second limit.

## License

The code is released under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
The data files derived from the Leyu corpus are released under CC BY 4.0, and the
other data sources keep their own licences. See `amharic_workspace/DATA_LICENCES.md`.