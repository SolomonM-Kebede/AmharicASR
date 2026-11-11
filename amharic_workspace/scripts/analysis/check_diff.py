
import json, jiwer

X = {o["orig_idx"]: o for o in map(json.loads, open("results/persample/results_raw_persample.jsonl", encoding="utf-8"))}
SUB = [int(x) for x in open("splits/subset_688_idx.txt").read().split()]

refs = [X[i]["reference"] for i in SUB]
hyps = [X[i]["prediction"] for i in SUB]
print("subset size:", len(refs))
print("corpus WER (jiwer):", round(100 * jiwer.wer(refs, hyps), 2))