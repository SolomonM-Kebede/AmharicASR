"""Figures for RQ3 and the per-dialect comparison, computed from the per-utterance files.

Every bar is computed here with the counting used by run_bootstrap.py, from
results/persample/. Dialect labels come from the Joint test split, indexed by
orig_idx. Before drawing, each system's overall WER is checked against the value
in the thesis; the script stops if any differs.

Writes figures/fig_rq3.png and figures/fig_per_dialect.png. Run from the workspace
root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

from common.bootstrap_utils import load, word_errors

DIALECTS = ["addis ababa", "gojjam", "gonder", "shewa", "wello"]
LABELS = ["Addis Ababa", "Gojjam", "Gonder", "Shewa", "Wello"]

# Per-utterance file for each system drawn.
SYSTEMS = {
    "c2": "results_raw_persample.jsonl",             # LoRA, BF16 weights, bf16 inference
    "c4b": "c4b_bfloat16_persample.jsonl",            # full FT, FP32 weights, bf16 inference
    "c4": "fullft_joint_indep_TEST_persample.jsonl",  # full FT, FP16 weights
    "a2": "a2_persample.jsonl",                       # SI, bf16 re-inference
    "d1": "d1_persample.jsonl",                       # SI size-matched sample
    "c1": "c1_persample.jsonl",                       # Joint, Phase 1 initialization
}
# Overall WER as reported in the thesis. A mismatch stops the script.
THESIS_OVERALL = {"c2": 32.06, "c4b": 31.16, "c4": 35.60, "a2": 16.04, "d1": 29.28, "c1": 33.47}


def dialect_of_index():
    """Dialect of each Joint test utterance, by position (the orig_idx)."""
    test = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"]
    return list(test["dialect"])


def system_stats(name, dialect_of):
    """Per-dialect and overall WER, with empty references skipped as in the bootstrap."""
    rows = load(cfg.require(cfg.PERSAMPLE / SYSTEMS[name]))
    errors = {d: 0 for d in DIALECTS}
    words = {d: 0 for d in DIALECTS}
    for i, r in rows.items():
        if not r["reference"].strip():
            continue
        d = dialect_of[i]
        e, n = word_errors(r["reference"], r["prediction"])
        errors[d] += e
        words[d] += n
    per = [100 * errors[d] / words[d] for d in DIALECTS]
    overall = 100 * sum(errors.values()) / sum(words.values())
    return per, overall


def draw_rq3(stats):
    lora_per, _ = stats["c2"]
    full_per, _ = stats["c4b"]
    overall = [
        ("LoRA\n(C2)", stats["c2"][1], "#4c72b0"),
        ("Full FT, FP16\n(C4)", stats["c4"][1], "#c44e52"),
        ("Full FT, FP32\n(C4b)", stats["c4b"][1], "#dd8452"),
    ]

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.3, 2.7), sharey=True, gridspec_kw={"width_ratios": [1, 2.3]})

    x = np.arange(len(overall))
    bars = ax_a.bar(x, [v for _, v, _ in overall], 0.6, color=[c for *_, c in overall])
    ax_a.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([n for n, *_ in overall], fontsize=7)
    ax_a.set_ylabel("WER (%)")
    ax_a.set_ylim(0, 46)
    ax_a.grid(axis="y", alpha=0.3)
    ax_a.set_axisbelow(True)
    ax_a.set_title("(a) Overall", fontsize=9)

    w = 0.36
    x = np.arange(len(LABELS))
    b1 = ax_b.bar(x - w / 2, lora_per, w, color="#4c72b0", label="LoRA (C2)")
    b2 = ax_b.bar(x + w / 2, full_per, w, color="#dd8452", label="Full fine-tuning (C4b)")
    ax_b.bar_label(b1, fmt="%.2f", fontsize=6, padding=1, rotation=90)
    ax_b.bar_label(b2, fmt="%.2f", fontsize=6, padding=1, rotation=90)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(LABELS, fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)
    ax_b.set_axisbelow(True)
    ax_b.set_title("(b) By dialect", fontsize=9)
    ax_b.legend(frameon=False, ncol=2, loc="upper center", fontsize=8)

    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "fig_rq3.png", dpi=300)
    plt.close(fig)


def draw_per_dialect(stats):
    runs = [
        ("A2: SI, 38,948 utt., bf16 re-inference", stats["a2"]),
        ("D1: SI sample, 10,296 utt.", stats["d1"]),
        ("C1: Joint, 10,296 utt.", stats["c1"]),
    ]
    labels = LABELS + ["All"]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    x = np.arange(len(labels))
    w = 0.26
    colors = ["#bdbdbd", "#6baed6", "#08519c"]
    for i, ((name, (per, overall)), color) in enumerate(zip(runs, colors)):
        bars = ax.bar(x + (i - 1) * w, per + [overall], w, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=1, rotation=90)
    ax.axvline(4.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("WER (%)")
    ax.set_ylim(0, 52)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)

    fig.tight_layout()
    fig.savefig(cfg.FIGURES / "fig_per_dialect.png", dpi=300)
    plt.close(fig)


def main():
    dialect_of = dialect_of_index()
    stats = {name: system_stats(name, dialect_of) for name in SYSTEMS}

    print("WER (%) used in the figures:")
    for name, (per, overall) in stats.items():
        detail = "  ".join(f"{l} {v:5.2f}" for l, v in zip(LABELS, per))
        print(f"  {name:4s} overall {overall:6.2f}   {detail}")
        if round(overall, 2) != THESIS_OVERALL[name]:
            raise SystemExit(f"{name}: computed {overall:.2f}, thesis {THESIS_OVERALL[name]:.2f}; "
                             f"figures not drawn")

    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    draw_rq3(stats)
    draw_per_dialect(stats)
    print(f"saved {cfg.FIGURES / 'fig_rq3.png'} and {cfg.FIGURES / 'fig_per_dialect.png'}")


if __name__ == "__main__":
    main()