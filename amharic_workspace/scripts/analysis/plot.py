"""Training curves for the runs in the thesis.

  python scripts/analysis/plot.py --extract
      Reads the Slurm logs in slurm_logs/<run>/ (local only) and writes one JSON file
      per run to results/training_logs/. Run once, while the logs exist, then commit
      the JSON.

  python scripts/analysis/plot.py
      Reads results/training_logs/ and writes figures/fig_val_curves and
      figures/fig_train_loss (.png and .pdf). Prints the best validation checkpoint
      of each run. Needs no logs and no GPU.

Panels are split by data split. Validation WER is comparable only within one split:
the Joint runs share the Joint validation set, and the SI and TI runs share the SI
validation set. C4 was checkpoint-selected on the full Joint validation set (5,942
utterances); the other runs used the 800-utterance subset. Its curve is labelled.
"""
from common import config as cfg  # path convention; keep first

import argparse
import ast
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = cfg.DATA_DIR / "slurm_logs"
HISTORY_DIR = cfg.RESULTS / "training_logs"
FIGURES = cfg.FIGURES

JOINT = {
    "C1 (Phase 1 init)": "C1",
    "C2 (raw init)": "C2",
    "C3 (Phase 1 init)": "C3",
    "C4b (full FT, FP32)": "C4b",
    "C4 (full FT, FP16; full validation set)": "C4",
}
SI_TI = {
    "A1 (Phase 1)": "A1",
    "D1 (SI sample)": "D1",
    "A3 (SI, mixed)": "A3",
    "B1 (TI)": "B1",
}
ALL_RUNS = {**JOINT, **SI_TI}
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1))]
DICT_RE = re.compile(r"\{'[^{}]*\}")


# ---------- step 1: extract from the Slurm logs (run once) ----------

def job_number(path):
    return int(re.findall(r"\d+", path.stem)[-1])


def parse_out(path):
    entries = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for match in DICT_RE.finditer(line):
                try:
                    entry = ast.literal_eval(match.group())
                except (ValueError, SyntaxError):
                    continue
                if isinstance(entry, dict) and "epoch" in entry:
                    entries.append(entry)
    return entries


def load_trainer_state(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("log_history", []) if isinstance(data, dict) else data
    return [e for e in entries if isinstance(e, dict) and "epoch" in e]


def read_run(folder):
    """Return (entries, source). A run is resumed across jobs; a later segment
    replaces the earlier entries from the epoch where it starts."""
    directory = LOG_DIR / folder
    if not directory.is_dir():
        return [], "missing folder"
    json_files = sorted(directory.glob("*.json"))
    if json_files:
        return load_trainer_state(json_files[0]), json_files[0].name
    log, names = [], []
    for path in sorted(directory.glob("*.out"), key=job_number):
        segment = parse_out(path)
        if not segment:
            continue
        start = min(float(e["epoch"]) for e in segment)
        log = [e for e in log if float(e["epoch"]) < start]
        log.extend(segment)
        names.append(path.name)
    return log, ", ".join(names)


def extract():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    for folder in sorted(set(ALL_RUNS.values())):
        entries, source = read_run(folder)
        if not entries:
            print(f"[EMPTY] {folder}: {source}")
            continue
        out = HISTORY_DIR / f"{folder}.json"
        out.write_text(json.dumps({"run": folder, "source": source, "entries": entries},
                                  indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"[WROTE] {out.name}: {len(entries)} entries from {source}")


# ---------- step 2: draw from the committed JSON ----------

def load_history(folder):
    path = HISTORY_DIR / f"{folder}.json"
    if not path.exists():
        print(f"[NO HISTORY] {folder}: run with --extract first")
        return []
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def extract_points(entries, metric):
    points = []
    for e in entries:
        if metric not in e:
            continue
        try:
            epoch, value = float(e["epoch"]), float(e[metric])
        except (ValueError, TypeError):
            continue
        if math.isfinite(epoch) and math.isfinite(value):
            points.append((epoch, value))
    return points


def panel(ax, runs, metric, title):
    drawn = 0
    for (label, folder), style in zip(runs.items(), STYLES):
        points = extract_points(load_history(folder), metric)
        if not points:
            print(f"[NO {metric}] {label}")
            continue
        epochs, values = zip(*points)
        line, = ax.plot(epochs, values, linestyle=style, linewidth=1.3, label=label)
        if metric == "eval_wer":
            best = min(range(len(values)), key=values.__getitem__)
            ax.plot(epochs[best], values[best], marker="*", ms=9, ls="None",
                    color=line.get_color())
            print(f"[BEST] {label}: epoch {epochs[best]:.2f}, WER {values[best]:.2f} "
                  f"(last logged epoch {epochs[-1]:.2f})")
        drawn += 1
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=7, frameon=False)


def figure(metric, ylabel, stem):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    panel(ax_a, JOINT, metric, "(a) Joint split")
    panel(ax_b, SI_TI, metric, "(b) SI and TI splits")
    ax_a.set_ylabel(ylabel)
    ax_b.set_ylabel(ylabel)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIGURES / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extract", action="store_true", help="read the Slurm logs and write the JSON histories")
    args = parser.parse_args()
    if args.extract:
        extract()
        return
    figure("eval_wer", "Validation WER (%)", "fig_val_curves")
    figure("loss", "Training loss", "fig_train_loss")


if __name__ == "__main__":
    main()