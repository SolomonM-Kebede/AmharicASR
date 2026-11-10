import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

# label, WER, CER, colour group, hatched = not directly comparable
rows = [
    ("Zero-shot Whisper Large-v3",          174.34, 107.40, "base", False),
    ("Google Speech-to-Text (4,198 rec.)",   69.09,  43.79, "base", True),
    ("Phase 1, Leyu test set",               98.52,  32.15, "base", False),
    ("A2  SI, LoRA",                         16.01,   5.69, "lora", False),
    ("A3  SI, LoRA + Standard",              15.26,   5.43, "lora", False),
    ("A4  SI, full FT (FP16, probable)",     22.68,   7.71, "fft",  False),
    ("B1  TI, LoRA (own test set)",          26.54,   9.26, "lora", True),
    ("D1  SI sample, LoRA (control)",        29.28,   9.94, "lora", False),
    ("C1  Joint, LoRA, Phase 1 init.",       33.47,  11.45, "lora", False),
    ("C3  Joint, LoRA, Phase 1 + Standard",  32.95,  11.22, "lora", False),
    ("C2  Joint, LoRA, raw init.",           32.03,  10.86, "lora", False),
    ("C4  Joint, full FT (FP16)",            35.60,  12.17, "fft",  False),
    ("C4b Joint, full FT (FP32)",            31.16,  10.50, "fft",  False),
]
colour = {
    "base": "#C65D00",  
    "lora": "#4c72b0",  
    "fft":  "#55a868",  
}
XMAX = 180

fig, ax = plt.subplots(figsize=(6.3, 4.8))
n = len(rows)
for k, (label, wer, cer, grp, hatched) in enumerate(rows):
    y = n - 1 - k                     # first row at the top
    bar = min(wer, XMAX)
    ax.barh(y, bar, height=0.62, color=colour[grp], edgecolor="black", linewidth=0.4,
            hatch="////" if hatched else None)
    text = f"{wer:.2f} (CER {cer:.2f})"
    if wer > XMAX:
        ax.text(XMAX - 3, y, "//", ha="center", va="center", fontsize=8, color="white")
    ax.text(bar + 1.0, y, text, va="center", fontsize=6.5)

ax.set_yticks(range(n))
ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7)
ax.set_xlim(0, XMAX + 42)
ax.set_xticks(range(0, XMAX + 1, 15))
ax.set_xlabel("WER (%)")
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

handles = [Patch(facecolor=colour["base"], edgecolor="black", label="Baseline"),
           Patch(facecolor=colour["lora"], edgecolor="black", label="LoRA"),
           Patch(facecolor=colour["fft"], edgecolor="black", label="Full fine-tuning"),
           Patch(facecolor="white", edgecolor="black", hatch="////", label="Different test set or coverage")]
ax.legend(handles=handles, loc="lower right", fontsize=6.5, frameon=False)

fig.tight_layout()
fig.savefig("figures/fig_all_runs.png", dpi=300, bbox_inches="tight")
print("saved figures/fig_all_runs.png")