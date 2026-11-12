"""Overview of all systems on the Leyu test set: WER and CER for each system.

Every bar comes from a committed source:
  persample  a file in results/persample/: WER and CER computed here, with the
             counting used by bootstrap_utils (empty references skipped)
  json       a saved summary in results/summaries/ or results/google/: WER and CER
             read from the file

Each WER is checked against the value in the thesis (expected_wer). A mismatch
stops the run, so the figure cannot disagree with the text. Systems without a
reproducible source (source=None) are listed and not drawn, so no unsourced
number reaches the figure.

Writes figures/fig_all_runs.png and .pdf. Run from the workspace root after
sourcing env.amh.sh.
"""
from common import config as cfg  # keep first

import json
from dataclasses import dataclass
from typing import Optional, Tuple

import re
import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common.bootstrap_utils import load, word_errors

SUMMARIES = cfg.DATA_DIR / "results" / "summaries"
COLOUR = {"base": "#C65D00", "lora": "#4c72b0", "fft": "#55a868"}
XMAX = 180
OVERALL_RE = re.compile(
    r"OVERALL\s+(?:n=\s*)?(\d+)\s+(?:WER=\s*)?([\d.]+)%?\s+(?:CER=\s*)?=?\s*([\d.]+)%")


@dataclass(frozen=True)
class Row:
    label: str
    group: str                          # base | lora | fft
    expected_wer: float                 # value in the thesis text
    source: Optional[Tuple]             # ("persample", path) | ("json", path, key or None) | None
    hatched: bool = False               # different test set or coverage


ROWS = [
    Row("Zero-shot Whisper Large-v3", "base", 174.34,
        ("json", SUMMARIES / "zeroshot_summary.json", "OVERALL")),
    Row("Google Speech-to-Text (4,198 rec.)", "base", 69.09,
        ("json", cfg.GOOGLE / "google_stt_rescore_results.json", "OVERALL"), hatched=True),
    Row("Phase 1, Leyu test set", "base", 98.52,
        ("log", SUMMARIES / "phase1_leyu_eval.log")),
    Row("A2  SI, LoRA (bf16 inference)", "lora", 16.04,
        ("persample", cfg.PERSAMPLE / "a2_persample.jsonl")),
    Row("A3  SI, LoRA + Standard", "lora", 15.26,
        ("json", SUMMARIES / "a3_si_merged_summary.json", "OVERALL")),
    Row("A4  SI, full FT (weights unverified)", "fft", 22.68,
        ("json", SUMMARIES / "a4_si_fullft_summary.json", "OVERALL")),
    Row("B1  TI, LoRA (own test set)", "lora", 26.51,
        ("json", SUMMARIES / "b1_summary.json", "OVERALL"), hatched=True),
    Row("D1  SI sample, LoRA (control)", "lora", 29.28,
        ("persample", cfg.PERSAMPLE / "d1_persample.jsonl")),
    Row("C1  Joint, LoRA, Phase 1 init.", "lora", 33.47,
        ("persample", cfg.PERSAMPLE / "c1_persample.jsonl")),
    Row("C3  Joint, LoRA, Phase 1 + Standard", "lora", 32.95,
        ("log", SUMMARIES / "c3_eval.log")),
    Row("C2  Joint, LoRA, raw init.", "lora", 32.06,
        ("persample", cfg.PERSAMPLE / "results_raw_persample.jsonl")),
    Row("C4  Joint, full FT (FP16 weights)", "fft", 35.60,
        ("persample", cfg.PERSAMPLE / "fullft_joint_indep_TEST_persample.jsonl")),
    Row("C4b Joint, full FT (FP32 weights)", "fft", 31.16,
        ("persample", cfg.PERSAMPLE / "c4b_bfloat16_persample.jsonl")),
]


def persample_scores(path):
    """WER from the bootstrap counting; CER over pairs with a non-empty hypothesis."""
    rows = load(cfg.require(path))
    errors = words = 0
    refs, hyps = [], []
    for r in rows.values():
        ref, hyp = r["reference"], r["prediction"]
        if not ref.strip():
            continue
        e, n = word_errors(ref, hyp)
        errors += e
        words += n
        if hyp.strip():
            refs.append(ref)
            hyps.append(hyp)
    return round(100 * errors / words, 2), round(100 * jiwer.cer(refs, hyps), 2)


def json_scores(path, key):
    d = json.loads(cfg.require(path).read_text(encoding="utf-8"))
    if key:
        d = d[key]
    return round(d["wer"], 2), round(d["cer"], 2)


def scores(row):
    if row.source[0] == "persample":
        return persample_scores(row.source[1])
    if row.source[0] == "log":
        return log_scores(row.source[1])
    return json_scores(row.source[1], row.source[2])

def log_scores(path, expected_n=4588):
    text = cfg.require(path).read_text(encoding="utf-8", errors="replace")
    found = OVERALL_RE.findall(text)
    if len(found) != 1:
        raise SystemExit(f"{path.name}: {len(found)} OVERALL lines, expected 1")
    n, wer, cer = found[0]
    if int(n) != expected_n:
        raise SystemExit(f"{path.name}: n={n}, expected {expected_n}")
    return round(float(wer), 2), round(float(cer), 2)

def draw(items):
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    n = len(items)
    for k, (row, wer, cer) in enumerate(items):
        y = n - 1 - k                       # first row at the top
        bar = min(wer, XMAX)
        ax.barh(y, bar, height=0.62, color=COLOUR[row.group], edgecolor="black",
                linewidth=0.4, hatch="////" if row.hatched else None)
        if wer > XMAX:
            ax.text(XMAX - 3, y, "//", ha="center", va="center", fontsize=8, color="white")
        ax.text(bar + 1.0, y, f"{wer:.2f} (CER {cer:.2f})", va="center", fontsize=6.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels([row.label for row, _, _ in items][::-1], fontsize=7)
    ax.set_xlim(0, XMAX + 42)
    ax.set_xticks(range(0, XMAX + 1, 15))
    ax.set_xlabel("WER (%)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Patch(facecolor=COLOUR["base"], edgecolor="black", label="Baseline"),
        Patch(facecolor=COLOUR["lora"], edgecolor="black", label="LoRA"),
        Patch(facecolor=COLOUR["fft"], edgecolor="black", label="Full fine-tuning"),
        Patch(facecolor="white", edgecolor="black", hatch="////",
              label="Different test set or coverage"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False)

    fig.tight_layout()
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = cfg.FIGURES / f"fig_all_runs.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {path}")
    plt.close(fig)


def main():
    drawn, pending = [], []
    print(f"{'system':42s} {'WER':>7s} {'CER':>7s}")
    for row in ROWS:
        if row.source is None:
            pending.append(row.label)
            continue
        wer, cer = scores(row)
        if round(wer, 2) != row.expected_wer:
            raise SystemExit(f"{row.label}: computed WER {wer:.2f}, thesis {row.expected_wer:.2f}; "
                             f"figure not drawn")
        drawn.append((row, wer, cer))
        print(f"{row.label:42s} {wer:7.2f} {cer:7.2f}")

    if pending:
        print("\nnot drawn (no reproducible source yet):")
        for label in pending:
            print(f"  {label}")
    draw(drawn)


if __name__ == "__main__":
    main()