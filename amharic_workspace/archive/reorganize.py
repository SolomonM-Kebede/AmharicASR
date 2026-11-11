#!/usr/bin/env python3
"""Pass 1: move files into the new folders. Dry run unless --apply is given.
Moves only; file contents are not changed. Run from the workspace root."""
import os, re, shutil, subprocess, sys

APPLY = "--apply" in sys.argv

KEEP = {"reorganize.py", "rewrite_paths.py", "config.py", "env.amh.sh",
        "requirements.txt", "README.md", ".gitignore"}

# Explicit destinations, checked before the rules
EXPLICIT = {
    "subset_688_idx.txt": "splits/subset_688_idx.txt",
    "manifests": "splits/manifests",
    "eval_controls": "results/persample",
    "joint_composition_ablation/results_raw_persample.jsonl":
        "results/persample/results_raw_persample.jsonl",
    "phase_fullft_joint_indep/results/fullft_joint_indep_TEST_persample.jsonl":
        "results/persample/fullft_joint_indep_TEST_persample.jsonl",
    "phase2_leyu/results/google_stt_raw_predictions.jsonl":
        "results/google/google_stt_raw_predictions.jsonl",
    "train_joint_composition_trainonly.py": "archive/train_joint_composition_trainonly.py",
    "dryrun_trainonly.log": "archive/dryrun_trainonly.log",
}

# Top-level file names: first matching rule wins
RULES = [
    ("scripts/analysis", r"(run_bootstrap|outside_subset|overlap_bootstrap|perdialect_bootstrap|"
                         r"coverage_check|bootstrap_pair|check_diff|check_prefix|draw_fig|plot|"
                         r"result_graph|fig_\w+)\.py"),
    ("scripts/app", r"(app|upload_model)\.py"),
    ("scripts/merge", r"(merge_\w+|manual_merge)\.py"),
    ("scripts/data", r"(build_\w+|prepare_\w+|create_text_independent_split|leakage_filter|"
                     r"make_speaker_map|join_speaker|normalize|rebuild_ti|overlap|phase1_test_rebuild)\.py"),
    ("scripts/train", r"(train_\w+|full_fine\w*|phase2b_leyu_alffa_train|text_speaker_indep_train)\.py"),
    ("scripts/eval", r"(eval\w*|evaluate_\w+|final_eval\w*|full_eval\w*|reeval\w*|"
                     r"phase1_standard_evalTrue|phase2_true_fixed_eval|zeroShot|google_stt|"
                     r"subset_google|score_google_raw|rerun_google_stt|fine_tune_phase3Eval)\.py"),
    ("scripts/diagnostics", r"(diagnose_\w+|dignose_\w+|further_diagnose|extra_diagnose|diff_test|"
                            r"check_\w+|verify\w*|token_check|todo_check|split_stats|leyu_stats|"
                            r"rescore_clean|cv_\w+_check)\.py"),
    ("figures", r"fig_\w+\.(png|pdf)"),
    ("thesis", r"\w+\.tex"),
]


def is_tracked(path):
    out = subprocess.run(["git", "ls-files", "--", path], capture_output=True, text=True).stdout
    return bool(out.strip())


def plan():
    moves, unsorted = [], []
    for src, dst in EXPLICIT.items():
        if os.path.exists(src):
            moves.append((src, dst))
    for name in sorted(os.listdir(".")):
        if name.startswith(".") or name in KEEP or name in EXPLICIT or os.path.isdir(name):
            continue
        for dst_dir, pattern in RULES:
            if re.fullmatch(pattern, name):
                moves.append((name, f"{dst_dir}/{name}"))
                break
        else:
            unsorted.append(name)
    return moves, unsorted


moves, unsorted = plan()
clash = [dst for _, dst in moves if os.path.exists(dst)]
if clash:
    sys.exit("destination already exists, resolve first:\n  " + "\n  ".join(clash))

for src, dst in moves:
    print(f"{src:65s} -> {dst}")
print(f"\n{len(moves)} move(s).")
print("Left in place (decide where these go):")
for name in unsorted:
    print("  ", name)

if not APPLY:
    print("\nDry run. Re-run with --apply to move.")
    sys.exit(0)

for src, dst in moves:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isfile(src) and is_tracked(src):
        subprocess.run(["git", "mv", src, dst], check=True)
    else:
        shutil.move(src, dst)
print("done.")