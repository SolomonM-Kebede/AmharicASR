import ast
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "slurm_logs"   

# Label -> subfolder name in slurm_logs/
A = {
    "C1 (Phase 1 init)": "C1",
    "C2 (raw init)": "C2",
    "C3 (Phase 1 init)": "C3",
    "D1 (SI sample)": "D1",
    "C4b (full FT, FP32)": "C4b",
    "C4 (full FT, FP16)": "C4",
}
# C4 was validated on the full 5,942 utterances, the others on the 800-utterance subset
A_VAL = {k: v for k, v in A.items() if not k.startswith("C4 (")}

B = {
    "A1 (Phase 1)": "A1",
    "A3 (SI, mixed)": "A3",
    "B1 (TI)": "B1",
}

styles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1))]
DICT_RE = re.compile(r"\{'[^{}]*\}")


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


def load_json(path):
    """Read a Hugging Face trainer_state.json and keep the entries that have an epoch."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("log_history", []) if isinstance(data, dict) else data
    return [e for e in entries if isinstance(e, dict) and "epoch" in e]


def load_run(folder):
    """Load one run: its .json history if present, otherwise merge its .out files in job order."""
    directory = LOG_DIR / folder
    if not directory.is_dir():
        print(f"[NO FOLDER] {directory}")
        return []

    json_files = sorted(directory.glob("*.json"))
    if json_files:
        log = load_json(json_files[0])
        print(f"[LOADED] {json_files[0].name}: {len(log)} entries")
        return log

    log = []
    for path in sorted(directory.glob("*.out"), key=job_number):
        segment = parse_out(path)
        if not segment:
            continue
        start = min(float(e["epoch"]) for e in segment)
        log = [e for e in log if float(e["epoch"]) < start]
        log.extend(segment)
        print(f"[LOADED] {path.name}: {len(segment)} entries, from epoch {start:.2f}")
    return log

histories = {folder: load_run(folder) for folder in set(A.values()) | set(B.values())}
print("[A3 eval]", [(e["epoch"], e["eval_wer"]) for e in histories["A3"] if "eval_wer" in e])

def extract_points(log, metric):
    points = []
    for entry in log:
        if metric not in entry:
            continue
        try:
            epoch = float(entry["epoch"])
            value = float(entry[metric])
        except (ValueError, TypeError):
            continue
        if math.isfinite(epoch) and math.isfinite(value):
            points.append((epoch, value))
    return points


def panel(ax, runs, title, metric):
    plotted = 0
    for (label, folder), ls in zip(runs.items(), styles):
        points = extract_points(histories.get(folder, []), metric)
        if not points:
            print(f"[NO {metric}] {label}")
            continue
        epochs, values = zip(*points)
        line, = ax.plot(epochs, values, linestyle=ls, linewidth=1.3, label=label)
        if metric == "eval_wer":
            best = min(range(len(values)), key=lambda i: values[i])
            print(f"[RANGE {metric}] {label}: min={min(values):.2f} max={max(values):.2f}")
            ax.plot(epochs[best], values[best], marker="*", linestyle="None",
                    markersize=9, color=line.get_color())
        plotted += 1
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=7, frameon=False)


def create_figure(metric, ylabel, filename, runs_a):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    panel(ax_a, runs_a, "(a) Joint-split and full FT runs", metric)
    panel(ax_b, B, "(b) Phase 1, SI and TI runs", metric)
    ax_a.set_ylabel(ylabel)
    ax_b.set_ylabel(ylabel)
    fig.tight_layout()

    out = BASE_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out / f"{filename}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] {path}")
    plt.close(fig)


create_figure("eval_wer", "Validation WER (%)", "fig_val_curves", A_VAL)
create_figure("loss", "Training loss", "fig_train_loss", A)