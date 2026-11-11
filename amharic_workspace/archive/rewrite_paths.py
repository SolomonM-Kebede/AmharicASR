#!/usr/bin/env python3
"""Pass 2: rewrite moved path literals and the bootstrap import in scripts/.
Dry run unless --apply is given. Run from the workspace root after reorganize.py."""
import difflib, glob, re, sys

APPLY = "--apply" in sys.argv

# Old path prefix -> new prefix. Matched only at the start of a quoted string literal.
PATH_MAP = [
    ("eval_controls/", "results/persample/"),
    ("manifests/", "splits/manifests/"),
    ("subset_688_idx.txt", "splits/subset_688_idx.txt"),
    ("joint_composition_ablation/results_raw_persample.jsonl",
     "results/persample/results_raw_persample.jsonl"),
    ("phase_fullft_joint_indep/results/fullft_joint_indep_TEST_persample.jsonl",
     "results/persample/fullft_joint_indep_TEST_persample.jsonl"),
    ("phase2_leyu/results/google_stt_raw_predictions.jsonl",
     "results/google/google_stt_raw_predictions.jsonl"),
    ("train_joint_composition_trainonly.py", "archive/train_joint_composition_trainonly.py"),
]
IMPORT_MAP = [("from run_bootstrap import", "from common.bootstrap_utils import")]

changed = 0
for path in sorted(glob.glob("scripts/**/*.py", recursive=True)):
    src = open(path, encoding="utf-8").read()
    out = src
    for old, new in PATH_MAP:
        out = re.sub(r"""(?<=["'])""" + re.escape(old), new, out)
    for old, new in IMPORT_MAP:
        out = out.replace(old, new)
    if out == src:
        continue
    changed += 1
    if APPLY:
        open(path, "w", encoding="utf-8").write(out)
    else:
        sys.stdout.writelines(difflib.unified_diff(
            src.splitlines(True), out.splitlines(True), path, path))

print(f"\n{changed} file(s) {'rewritten' if APPLY else 'would change'}.")