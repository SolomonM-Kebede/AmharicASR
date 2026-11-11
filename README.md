# Amharic ASR: Dialectal Adaptation of Whisper Large-v3

Code, per-utterance predictions, and analysis for a bachelor thesis on adapting
Whisper Large-v3 to five regional varieties of Amharic (Gojjam, Gonder, Wello,
Shewa, Addis Ababa) using the Leyu corpus.

The thesis reports word error rates (WER) on a held-out Leyu test set under
three splits (speaker-independent, text-independent, and jointly
speaker- and text-independent). The results that the thesis reports can be
checked from the per-utterance files in this repository without a GPU.

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/analysis/` | Paired bootstrap, subset and per-dialect analyses, figures |
| `scripts/data/` | Corpus, split, and subset construction, text normalization |
| `scripts/train/` | Training scripts for each run in the experimental grid |
| `scripts/eval/` | Evaluation, zero-shot baseline, Google Speech-to-Text |
| `scripts/merge/` | LoRA merge and precision handling |
| `scripts/diagnostics/` | Checks and diagnostics cited in the thesis |
| `scripts/common/` | Shared configuration and bootstrap helpers |
| `results/persample/` | Per-utterance predictions for each evaluated system |
| `results/google/` | Google Speech-to-Text predictions |
| `splits/` | 688-recording subset index and split manifests |
| `figures/` | Figures used in the thesis |
| `archive/` | Superseded scripts, kept for reference |

## Reproducing the reported comparisons

The paired comparisons need only the per-utterance files and the Leyu
test cache. No GPU is required.

```bash
git clone https://github.com/SolomonM-Kebede/AmharicASR.git
cd AmharicASR/amharic_workspace
cp env.example.sh env.sh          # then set the paths for your machine
source env.sh
PYTHONPATH=scripts python scripts/analysis/run_bootstrap.py
PYTHONPATH=scripts python scripts/analysis/outside_subset.py
PYTHONPATH=scripts python scripts/analysis/perdialect_bootstrap.py
```

Expected output for `run_bootstrap.py` (LoRA C2 vs full fine-tuning C4b, bfloat16 inference):

| Quantity | Value |
|---|---|
| WER, C2 (LoRA) | 32.06% |
| WER, C4b (full fine-tuning) | 31.16% |
| Difference | 0.89 points |
| Utterance-level 95% interval | [0.68, 1.11] |
| Speaker-level 95% interval | [0.59, 1.20] |

The speaker-level interval is the primary one, because utterances from the
same speaker are not independent.

## Configuration

Paths are set in `scripts/common/config.py` and can be overridden through
environment variables, which `env.example.sh` lists. The Hugging Face cache
is set on import, before any transformers or datasets code loads. Inputs
are checked when a script uses them, so a missing input only affects the
scripts that need it.

## Data

- **Leyu** (dialectal, prompted read speech, five varieties, 49,478
  utterances from 295 speakers). Not included in this repository; obtain it
  from its authors and follow its license.
- **ALFFA** (Standard Amharic read speech) and **Mozilla Common Voice 25.0
  Amharic**. Obtained from their public releases.

The Common Voice component of C2 and C3 is built from all five clip-level
files without deduplication, so the pool contains repeated clips. This is
documented in the thesis and is not a bug in the repository.

## Known limitations

- The SI split cache (`dataset_cache_2026-08-expanded-spkindep`) is not kept
  in this repository. It can be rebuilt with [script name] from the corpus.
- Results are from single training runs, except where stated.
- The Joint-split LoRA runs select checkpoints on an 800-utterance validation
  subset. Full fine-tuning run C4 selected on the full validation set, and C4b
  on 800 utterances.
- Google Speech-to-Text was scored on recordings of at most 55 seconds, which
  is a sending rule inside the API's 60-second limit.

## Citing

## License

