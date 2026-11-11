# Phase 1 on the Standard-Amharic held-out set (25.91% WER, 7.51% CER)

Script: scripts/eval/phase1_standard_evalTrue.py (in git history). Summary:
phase1_standard_summary.json. Log: phase1_standard_eval.log. Log and summary agree (n=1087).

- Model: phase1_curriculum/models/phase1_best_model_merged. Weights stored as float16
  (config.json). Evaluated in float32 (torch_dtype=float32, bf16=False).
- Decoding: beam 5, max_length 448, no generation limit on the trainer, no repetition penalty.
- Test set: phase1_curriculum/dataset_cache_rebuilt, 1,087 utterances, all from ALFFA
  (checked by decoding the labels against the ALFFA transcripts). No Common Voice
  sentence occurs in the test set. The cache has no speaker column, so the speaker
  partition cannot be checked from it. Built by an earlier version of
  scripts/data/phase1_test_rebuild.py, whose split and save steps are now commented out.
- Per-utterance predictions are not retained, so the figure cannot be bootstrapped.
- The Leyu Phase 1 figure (98.52%) is a separate bf16 evaluation (eval_phase1.py).
