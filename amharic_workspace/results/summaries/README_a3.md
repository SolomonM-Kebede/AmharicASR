# A3: SI split, LoRA on Leyu + ALFFA + Common Voice (15.26% WER)

Source: scripts/eval/final_eval_merged.py (in git history). Output:
phase2b_merged_FIXED_final.json, copied here as a3_si_merged_summary.json.
Log: a3_si_merged_eval.log.

Settings, read from the scripts:
- Training: scripts/train/phase2b_leyu_alffa_train.py. Phase 1 initialization,
  speaker-independent split, LoRA merged into the base model. Common Voice from
  train.tsv only (553 clips). Training weights loaded in bfloat16.
- Model: phase2b_leyu_standard/models/phase2b_final_merged. Saved config
  torch_dtype: <fill from config check>.
- Test set: dataset_cache_2026-08-expanded-spkindep, 4,588 utterances.
- Inference: float32 (torch_dtype=float32, bf16=False).
- Decoding: beam 5 (set at runtime), max_length 448, no repetition penalty and
  no no-repeat constraint (transformers defaults; not recorded in the saved config).
- Batch size: EVAL_BATCH_SIZE, default 8. Not recorded in the log.

The SI cache is no longer in the workspace, and the per-utterance predictions
are not retained, so this result cannot be bootstrapped or rerun from here.
