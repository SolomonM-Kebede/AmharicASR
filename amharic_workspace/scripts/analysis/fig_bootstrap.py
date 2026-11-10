"""Bootstrap figure for C2 (LoRA) vs. C4 (Full Fine-Tuning) on the Joint test set.
Difference = WER(Full FT) - WER(LoRA), in percentage points.
Run from the workspace folder:   python fig_bootstrap.py
Outputs: fig_bootstrap.pdf / .png and the numbers printed to the terminal.
"""
import json
import numpy as np
import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/phase2_leyu/dataset_cache_joint_independent"
C2 = "results/persample/results_raw_persample.jsonl"                       # LoRA, raw init
C4 = "results/persample/fullft_joint_indep_TEST_persample.jsonl"    # Full FT, raw init
B = 10_000
SEED = 0


def per_utt(path):
    """Errors (S+D+I) and reference word count for every utterance of a per-sample file."""
    e, n = [], []
    for line in open(path):
        r = json.loads(line)
        o = jiwer.process_words(r["reference"], r["prediction"])
        e.append(o.substitutions + o.deletions + o.insertions)
        n.append(o.substitutions + o.deletions + o.hits)
    return np.array(e, float), np.array(n, float)


def boot_utterances(e1, n1, e2, n2, B=B, seed=SEED, chunk=500):
    """Resample utterances; returns WER(system 1) - WER(system 2) for each resample."""
    rng = np.random.default_rng(seed)
    N = len(e1)
    out = np.empty(B)
    for s in range(0, B, chunk):
        k = min(chunk, B - s)
        idx = rng.integers(0, N, (k, N))
        out[s:s + k] = e1[idx].sum(1) / n1[idx].sum(1) - e2[idx].sum(1) / n2[idx].sum(1)
    return out


def boot_groups(e1, n1, e2, n2, groups, B=B, seed=SEED):
    """Resample whole groups (speakers); returns WER(system 1) - WER(system 2) for each resample."""
    rng = np.random.default_rng(seed)
    _, gid = np.unique(groups, return_inverse=True)
    G = gid.max() + 1
    ge1, gn1 = np.bincount(gid, e1, G), np.bincount(gid, n1, G)
    ge2, gn2 = np.bincount(gid, e2, G), np.bincount(gid, n2, G)
    idx = rng.integers(0, G, (B, G))
    return ge1[idx].sum(1) / gn1[idx].sum(1) - ge2[idx].sum(1) / gn2[idx].sum(1)


def summarize(d):
    return 100 * d.mean(), 100 * np.percentile(d, [2.5, 97.5]), (d > 0).mean()


def plot(d_utt, d_spk, observed, path="fig_bootstrap"):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=False)
    for ax, d, title in ((axes[0], d_utt, "Resampling utterances"),
                         (axes[1], d_spk, "Resampling speakers")):
        d = 100 * d
        lo, hi = np.percentile(d, [2.5, 97.5])
        ax.hist(d, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="red", linewidth=1.2)
        ax.axvline(observed, color="black", linestyle="--", linewidth=1.2)
        ax.axvspan(lo, hi, color="gray", alpha=0.18)
        ax.set_title(f"{title}\n95% interval [{lo:.2f}, {hi:.2f}]", fontsize=9)
        ax.set_xlabel("WER(Full FT) $-$ WER(LoRA) (points)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(min(-0.5, d.min() - 0.3), d.max() + 0.5)
    axes[0].set_ylabel("Number of resamples")
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=300)


if __name__ == "__main__":
    from datasets import load_from_disk
    test = load_from_disk(CACHE)["test"].select_columns(["dialect", "speaker_id"])   # no audio is read
    groups = np.array([f"{d}_{s}" for d, s in zip(test["dialect"], test["speaker_id"])])
    print("test utterances:", len(groups), " speakers:", len(set(groups)))

    e2, n2 = per_utt(C2)
    e4, n4 = per_utt(C4)
    assert len(e2) == len(groups) == len(e4), "row counts differ"
    assert np.array_equal(n2, n4), "reference word counts differ between the two files"
    print(f"WER LoRA {100*e2.sum()/n2.sum():.2f} | WER Full FT {100*e4.sum()/n4.sum():.2f}")
    observed = 100 * (e4.sum() / n4.sum() - e2.sum() / n2.sum())

    d_utt = boot_utterances(e4, n4, e2, n2)
    d_spk = boot_groups(e4, n4, e2, n2, groups)
    for name, d in (("utterance-level", d_utt), ("speaker-level", d_spk)):
        m, ci, p = summarize(d)
        print(f"{name}: mean {m:.2f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  LoRA better in {100*p:.2f}% of resamples")
    plot(d_utt, d_spk, observed)
    print("saved fig_bootstrap.pdf / .png")