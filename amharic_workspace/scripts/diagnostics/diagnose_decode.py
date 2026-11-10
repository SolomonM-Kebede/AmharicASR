"""
Amharic ASR — Phase 2 decode-config diagnostic (NO RETRAINING)
================================================================
Loads the already-trained, already-merged Phase 2 model and re-runs
inference on a subsample of the held-out TEST set under four decoding
configurations, to isolate why the final beam-search test WER (64.76%)
came back much higher than the in-training greedy-eval WER (~56.6%).

Small 2x2 factorial over (decoding strategy) x (repetition penalty):
  A. greedy (num_beams=1), no repetition penalty        — plain baseline
  B. greedy (num_beams=1), repetition_penalty=1.2 +
     no_repeat_ngram_size=7                              — matches the
     generation_config used during IN-TRAINING eval
  C. beam=5, no repetition penalty                        — plain beam search
  D. beam=5, repetition_penalty=1.2 + no_repeat_ngram=7   — matches the
     generation_config used for the FINAL per-dialect eval (64.76%)

How to read it:
  - If B ≈ A and D >> C: the repetition-penalty/no-repeat-ngram settings
    are specifically hurting beam search (the leading hypothesis).
  - If B is ALSO high (~64%) despite being greedy: the problem isn't beam
    search at all — look at a train/test distribution difference or a
    merge issue instead, since B exactly matches what training-time eval
    used and should reproduce ~56-58% if nothing else changed.
  - Sample predictions are printed per config so you can visually check
    for degenerate/looping output, which would directly confirm a
    decode-config problem rather than a modeling one.

Runs on a subsample (DIAG_SUBSET_SIZE) for speed — this is a diagnostic,
not the final thesis number. Once the winning config is identified, rerun
the full per-dialect eval with it for the number you actually report.
"""

import os

CUSTOM_CACHE = "/home/stud_homes/s8345520/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_METRICS_CACHE"] = CUSTOM_CACHE
os.environ["HF_EVALUATE_CACHE"] = CUSTOM_CACHE

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
)

print(f"torch  : {torch.__version__}")
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found — aborting.")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

BASE_DIR   = "./amharic_workspace"
PHASE2_DIR = os.path.join(BASE_DIR, "phase2_leyu")
MERGED_DIR = os.path.join(PHASE2_DIR, "models", "phase2_final_merged")

if not Path(MERGED_DIR).exists():
    raise FileNotFoundError(f"Merged model not found at {MERGED_DIR}")

DATASET_VERSION  = os.environ.get("DATASET_VERSION", "2026-08-expanded-spkindep")
LEYU_CACHE       = os.path.join(PHASE2_DIR, f"dataset_cache_{DATASET_VERSION}")
RESULTS_DIR      = os.path.join(PHASE2_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DIAG_SUBSET_SIZE = int(os.environ.get("DIAG_SUBSET_SIZE", 600))
SEED             = int(os.environ.get("SEED", 42))
EVAL_BATCH_SIZE  = int(os.environ.get("EVAL_BATCH_SIZE", 8))
N_SAMPLE_PRINTS  = int(os.environ.get("N_SAMPLE_PRINTS", 3))


AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


print(f"\nLoading processor/model from {MERGED_DIR} ...")
processor = WhisperProcessor.from_pretrained(MERGED_DIR, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR)
model = model.to("cuda").eval()

forced_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)

print(f"\nLoading cached test split from {LEYU_CACHE} ...")
cached = load_from_disk(LEYU_CACHE)
test_dataset = cached["test"]
print(f"  full test set: {len(test_dataset):,} examples")

if 0 < DIAG_SUBSET_SIZE < len(test_dataset):
    diag_test = test_dataset.shuffle(seed=SEED).select(range(DIAG_SUBSET_SIZE))
else:
    diag_test = test_dataset
print(f"  diagnostic subsample: {len(diag_test):,} examples")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]}              for f in features]

        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
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


def compute_metrics(pred):
    import evaluate as _eval
    _wer = _eval.load("wer")
    _cer = _eval.load("cer")

    pred_ids  = pred.predictions
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    wer = _wer.compute(predictions=pred_str, references=label_str) * 100
    cer = _cer.compute(predictions=pred_str, references=label_str) * 100
    return {"wer": round(wer, 2), "cer": round(cer, 2)}


class SafeWhisperTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels         = inputs.pop("labels")
        input_features = inputs.pop("input_features")
        inputs.clear()
        outputs = model(input_features=input_features, labels=labels)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        inputs.pop("decoder_input_ids",     None)
        inputs.pop("decoder_inputs_embeds", None)
        inputs.pop("input_ids",             None)
        inputs.pop("attention_mask",        None)
        inputs.pop("dialect",               None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                dtype=next(model.parameters()).dtype
            )
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


eval_args = Seq2SeqTrainingArguments(
    output_dir                 = "/tmp/eval_scratch_diag",
    per_device_eval_batch_size = EVAL_BATCH_SIZE,
    predict_with_generate      = True,
    generation_max_length      = 225,
    bf16                       = True,
    fp16                       = False,
    report_to                  = ["none"],
    remove_unused_columns      = False,
)

trainer = SafeWhisperTrainer(
    model           = model,
    args            = eval_args,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

CONFIGS = [
    ("A_greedy_no_penalty",                     dict(num_beams=1, repetition_penalty=1.0, no_repeat_ngram_size=0)),
    ("B_greedy_with_penalty_matches_train_eval", dict(num_beams=1, repetition_penalty=1.2, no_repeat_ngram_size=7)),
    ("C_beam5_no_penalty",                       dict(num_beams=5, repetition_penalty=1.0, no_repeat_ngram_size=0)),
    ("D_beam5_with_penalty_matches_final_eval",  dict(num_beams=5, repetition_penalty=1.2, no_repeat_ngram_size=7)),
]

results = {}

for name, cfg in CONFIGS:
    print(f"\n{'='*70}\nConfig: {name}  {cfg}\n{'='*70}")

    gen_kwargs = dict(
        forced_decoder_ids     = forced_ids,
        max_new_tokens         = 225,
        use_cache               = True,
        decoder_start_token_id  = model.config.decoder_start_token_id,
        num_beams               = cfg["num_beams"],
        repetition_penalty      = cfg["repetition_penalty"],
    )
    if cfg["no_repeat_ngram_size"] > 0:
        gen_kwargs["no_repeat_ngram_size"] = cfg["no_repeat_ngram_size"]

    model.generation_config = GenerationConfig(**gen_kwargs)

    result = trainer.predict(diag_test)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(result.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in
                 processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

    results[name] = {"wer": round(wer, 2), "cer": round(cer, 2), "n": len(diag_test)}
    print(f"  WER={wer:.2f}%  CER={cer:.2f}%")

    print("  Sample predictions:")
    for i in range(min(N_SAMPLE_PRINTS, len(pred_str))):
        print(f"    ref : {label_str[i][:120]}")
        print(f"    hyp : {pred_str[i][:120]}")

print(f"\n{'='*70}\nSUMMARY — decode-config diagnostic (n={len(diag_test)} test examples)\n{'='*70}")
for name, r in results.items():
    print(f"  {name:45s}  WER={r['wer']:6.2f}%  CER={r['cer']:6.2f}%")

out_path = os.path.join(RESULTS_DIR, f"decode_config_diagnostic_{DATASET_VERSION}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")