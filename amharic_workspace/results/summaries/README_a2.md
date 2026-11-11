# A2: SI split, LoRA on Leyu only

Model: phase2_leyu/models/phase2_final_merged (LoRA merged). Test set:
dataset_cache_2026-08-expanded-spkindep, 4,588 utterances. Beam 5, 448-token
limit, no repetition penalty.

| Result | WER | Weights | Inference | Per-utterance file | Producer |
|---|---|---|---|---|---|
| float32 | 16.01% | float32 | bf16 autocast | none | phase2_true_fixed_eval.py (log: a2_float32_eval.log) |
| bf16 | 16.04% | bf16 | bf16 | results/persample/a2_persample.jsonl | eval_any_merged.py (log: eval_ctrl_a2_eval_5518.out) |

The float32 run sets torch_dtype=float32 and bf16=True in the trainer, so
predictions were computed under bf16 autocast. The float32 label used elsewhere
should say "float32 weights with bf16 autocast."