"""
checking .out files to parase through meta data to get infos like wer, epoch...
run with python check_out.py path/of/the/file.out
"""

import ast
import re
import sys

text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
rows = []
for m in re.findall(r"\{'[^{}]*\}", text):
    try:
        rows.append(ast.literal_eval(m))
    except (ValueError, SyntaxError):
        continue
eval = [r for r in rows if "eval_wer" in r]
best = min(eval, key=lambda r: float(r["eval_wer"]))
print("eval points:", len(eval), "| best eval_wer:", best["eval_wer"], "at epoch", best["epoch"])