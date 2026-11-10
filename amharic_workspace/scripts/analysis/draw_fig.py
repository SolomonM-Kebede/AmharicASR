"""Reproduce fig_rq3.png and fig_per_dialect.png from the evaluation summary files.

Run from amharic_workspace/:
    python make_figures.py

Reads per-dialect and overall WER from the eval_controls/*.json summaries, prints
the values used, and writes both figures at 300 dpi to the current directory.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = Path("eval_controls")

# Summary files (per-dialect WER plus OVERALL). Confirm the a2/c1 filenames.
FILES = {
    "lora": "c2fp32merge_bfloat16.json",   # C2, adapter merged into FP32, bf16 inference
    "full32": "c4b_bfloat16.json",         # C4b, FP32 weights, bf16 inference
    "a2": "a2.json",                       # A2, bf16 re-inference
    "d1": "d1.json",                       # D1, size-matched control
    "c1": "c1.json",                       # C1, Joint split, LoRA
}

# C4 (FP16 weights) is shown overall only. 35.60 is the value in the thesis table.
C4_OVERALL = 35.60

# Dialect keys in the summary files, and the labels used in the figures.
KEYS = ["addis ababa", "gojjam", "gonder", "shewa", "wello"]
LABELS = ["Addis Ababa", "Gojjam", "Gonder", "Shewa", "Wello"]


def load_summary(name):
    """Return (per-dialect WER list in KEYS order, overall WER)."""
    with open(EVAL_DIR / name, encoding="utf-8") as f:
        summary = json.load(f)
    per_dialect = [summary[k]["wer"] for k in KEYS]
    return per_dialect, summary["OVERALL"]["wer"]


def main():
    data = {name: load_summary(fname) for name, fname in FILES.items()}

    print("Values used:")
    for name, (per, overall) in data.items():
        print(f"  {name:7s} overall {overall:6.2f}  by dialect {per}")
    print(f"  c4      overall {C4_OVERALL:6.2f}  (fixed value)")

    plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})

    # ---- Figure 1: Joint split. (a) overall, (b) by dialect.
    lora_per, lora_all = data["lora"]
    full_per, full_all = data["full32"]
    overall = [
        ("LoRA\n(C2)", lora_all, "#4c72b0"),
        ("Full FP16\n(C4)", C4_OVERALL, "#c44e52"),
        ("Full FP32\n(C4b)", full_all, "#dd8452"),
    ]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.3, 2.7), sharey=True, gridspec_kw={"width_ratios": [1, 2.3]}
    )
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
    fig.savefig("fig_rq3.png", dpi=300)
    plt.close(fig)

    # ---- Figure 2: A2, D1, C1 by dialect, with overall as the last group.
    runs = [
        ("A2: SI, 38,948 utt.", data["a2"]),
        ("D1: SI sample, 10,296 utt.", data["d1"]),
        ("C1: Joint, 10,296 utt.", data["c1"]),
    ]
    labels = LABELS + ["All"]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    x = np.arange(len(labels))
    w = 0.26
    colors = ["#bdbdbd", "#6baed6", "#08519c"]
    for i, ((name, (per, overall_wer)), color) in enumerate(zip(runs, colors)):
        bars = ax.bar(x + (i - 1) * w, per + [overall_wer], w, color=color, label=name)
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
    fig.savefig("fig_per_dialect.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()