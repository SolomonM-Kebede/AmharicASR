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
ev = [r for r in rows if "eval_wer" in r]
best = min(ev, key=lambda r: float(r["eval_wer"]))
print("eval points:", len(ev), "| best eval_wer:", best["eval_wer"], "at epoch", best["epoch"])