#!/usr/bin/env python3
"""Insert config and normalize imports where needed. Dry run unless --apply."""
import glob, re, sys, difflib

APPLY = "--apply" in sys.argv
CONFIG_LINE = "from common import config as cfg  # sets HF cache and offline flags; keep first\n"
NORM_LINE = "from data.normalize import normalize_amharic\n"

# Modules whose import must come after config (anything importing HF libraries)
HF_IMPORT = re.compile(r"^(import|from) (torch|librosa|datasets|transformers|peft|evaluate)\b", re.M)

for path in sorted(glob.glob("scripts/**/*.py", recursive=True)):
    if path.endswith(("config.py", "normalize.py", "__init__.py")):
        continue
    src = open(path, encoding="utf-8").read()
    out = src

    # Drop the old inline normalize import (the function is now shared)
    out = re.sub(r"^from normalize import .*\n", NORM_LINE, out, flags=re.M)
    out = re.sub(r"^import normalize\n", "", out, flags=re.M)

    # Add the normalize import if the name is used but not imported
    if "normalize_amharic(" in out and "from data.normalize import" not in out:
        out = NORM_LINE + out

    # Add the config import before the first heavy import, if not present
    if "from common import config" not in out and HF_IMPORT.search(out):
        m = HF_IMPORT.search(out)
        out = out[:m.start()] + CONFIG_LINE + out[m.start():]

    if out != src:
        if APPLY:
            open(path, "w", encoding="utf-8").write(out)
        else:
            sys.stdout.writelines(difflib.unified_diff(
                src.splitlines(True), out.splitlines(True), path, path, n=0))

print("applied" if APPLY else "dry run only; re-run with --apply")