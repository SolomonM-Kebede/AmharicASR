# A4: SI split, full fine-tuning (22.68% WER)

Source: scripts/eval/fine_tune_phase3Eval.py (in git history after the rename to
eval_si_fullft_a4.py). Run: 2026-08-30. Output: phase3_fft_TRUE_fixed.json,
copied here as a4_si_fullft_summary.json. Log: a4_si_fullft_eval.log.

Settings, read from the script:
- Model: phase3_full_finetune/models/phase3_full_finetuned_final (training log:
  Phase 1 init, SI split). The directory is no longer retained.
- Test set: dataset_cache_2026-08-expanded-spkindep, 4,588 utterances.
- Inference: float32 (torch_dtype=float32, bf16=False).
- Decoding: beam 5, max_length 448, max_new_tokens unset, no repetition
  penalty. Uncapped: no generation_max_length set on the trainer.
- Batch size: EVAL_BATCH_SIZE, default 8. The log does not record an override.
- Weight dtype at training: not verified. The checkpoint is gone.

Cannot be rerun, bootstrapped, or checked for weight precision. No
per-utterance predictions exist.
