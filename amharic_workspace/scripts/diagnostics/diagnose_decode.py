"""Decode-configuration diagnostic for the Phase 2 merged model. No training.

Runs the merged model on a subsample of the Leyu test set under four decoding
configurations, a 2x2 over decoding strategy (greedy, beam 5) and repetition
control (none, or repetition_penalty 1.2 with no_repeat_ngram_size 7):

  A  greedy, no repetition control
  B  greedy, with repetition control (the settings used in training-time eval)
  C  beam 5, no repetition control
  D  beam 5, with repetition control (the settings of the final per-dialect eval)

How to read the result:
  - B close to A, and D well above C: repetition control hurts beam search.
  - B also high: the problem is not beam search. Check the model or the data.

This is a diagnostic on a subsample, not a reported result.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import evaluate
import torch
from datasets import load_from_disk
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from data.normalize import normalize_amharic

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found; aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# Paths (from config; override with AMHARIC_* variables in env.amh.sh)
PHASE2_DIR = cfg.DATA_DIR / "phase2_leyu"
MERGED_DIR = cfg.require(PHASE2_DIR / "models" / "phase2_final_merged")
RESULTS_DIR = PHASE2_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_VERSION = os.environ.get("DATASET_VERSION", "2026-08-expanded-spkindep")
LEYU_CACHE = cfg.require(PHASE2_DIR / f"dataset_cache_{DATASET_VERSION}")

# Settings
DIAG_SUBSET_SIZE = int(os.environ.get("DIAG_SUBSET_SIZE", 600))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
N_SAMPLE_PRINTS = int(os.environ.get("N_SAMPLE_PRINTS", 3))
# 448 is the model's decoder limit. The earlier runs used 225, which truncated
# long utterances. Set MAX_NEW_TOKENS=225 to reproduce those earlier numbers.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", cfg.MAX_LABEL_LENGTH))
SEED = cfg.SEED

processor = WhisperProcessor.from_pretrained(str(MERGED_DIR), language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

model = WhisperForConditionalGeneration.from_pretrained(str(MERGED_DIR)).to("cuda").eval()

print(f"\nLoading test split from {LEYU_CACHE}")
test_dataset = load_from_disk(str(LEYU_CACHE))["test"]
print(f"  full test set: {len(test_dataset):,} examples")
if 0 < DIAG_SUBSET_SIZE < len(test_dataset):
    diag_test = test_dataset.shuffle(seed=SEED).select(range(DIAG_SUBSET_SIZE))
else:
    diag_test = test_dataset
print(f"  diagnostic subsample: {len(diag_test):,} examples")

WER = evaluate.load("wer")
CER = evaluate.load("cer")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch["labels"] = lids
        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=DECODER_START_TOKEN_ID,
)


def decode(ids):
    return [normalize_amharic(s) for s in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]


def compute_metrics(pred):
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str, label_str = decode(pred.predictions), decode(label_ids)
    return {
        "wer": round(WER.compute(predictions=pred_str, references=label_str) * 100, 2),
        "cer": round(CER.compute(predictions=pred_str, references=label_str) * 100, 2),
    }


class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()
        outputs = model(input_features=input_features, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        for key in ("decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                    "attention_mask", "dialect"):
            inputs.pop(key, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir=str(cfg.CHECKPOINT_DIR / "eval_scratch_diag"),
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=MAX_NEW_TOKENS,
    bf16=True,
    fp16=False,
    report_to=["none"],
    remove_unused_columns=False,
)

trainer = SafeWhisperTrainer(
    model=model,
    args=eval_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Named "settings", not "cfg", so the config module is not shadowed.
CONFIGS = [
    ("A_greedy_no_penalty",
     dict(num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0)),
    ("B_greedy_with_penalty_matches_train_eval",
     dict(num_beams=1, repetition_penalty=1.2, no_repeat_ngram_size=7)),
    ("C_beam5_no_penalty",
     dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0)),
    ("D_beam5_with_penalty_matches_final_eval",
     dict(num_beams=5, repetition_penalty=1.2, no_repeat_ngram_size=7)),
]

results = {}
for name, settings in CONFIGS:
    print(f"\n{'=' * 70}\nConfig: {name}  {settings}\n{'=' * 70}")

    gen_kwargs = dict(
        forced_decoder_ids=forced_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        decoder_start_token_id=model.config.decoder_start_token_id,
        num_beams=settings["num_beams"],
        repetition_penalty=settings["repetition_penalty"],
    )
    if settings["no_repeat_ngram_size"] > 0:
        gen_kwargs["no_repeat_ngram_size"] = settings["no_repeat_ngram_size"]
    model.generation_config = GenerationConfig(**gen_kwargs)

    prediction = trainer.predict(diag_test)
    label_ids = prediction.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str, label_str = decode(prediction.predictions), decode(label_ids)

    wer = WER.compute(predictions=pred_str, references=label_str) * 100
    cer = CER.compute(predictions=pred_str, references=label_str) * 100
    results[name] = {"wer": round(wer, 2), "cer": round(cer, 2), "n": len(diag_test)}
    print(f"  WER={wer:.2f}%  CER={cer:.2f}%")

    print("  Sample predictions:")
    for i in range(min(N_SAMPLE_PRINTS, len(pred_str))):
        print(f"    ref : {label_str[i][:120]}")
        print(f"    hyp : {pred_str[i][:120]}")

print(f"\n{'=' * 70}\nSUMMARY (n={len(diag_test)} test examples, max_new_tokens={MAX_NEW_TOKENS})\n{'=' * 70}")
for name, r in results.items():
    print(f"  {name:45s}  WER={r['wer']:6.2f}%  CER={r['cer']:6.2f}%")

out_path = RESULTS_DIR / f"decode_config_diagnostic_{DATASET_VERSION}_max{MAX_NEW_TOKENS}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")