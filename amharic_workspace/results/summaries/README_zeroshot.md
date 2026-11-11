# Zero-shot Whisper Large-v3 (174.34% WER)

Summary: zeroshot_summary.json, from phase2_leyu/results/zeroshot_whisper_results.json.
Settings (from the git history of scripts/eval/zeroShot.py): generation_max_length
commented out on the trainer, model.generation_config.max_length = 448,
max_new_tokens = None, beam 5, no repetition penalty. Weights float16 as stored.
Inference under bf16 (trainer bf16=True).

Log for this summary: zeroshot_capped225_eval.log
Capped run (225-token cap, about 109% WER): zeroshot_capped225_eval.log.
