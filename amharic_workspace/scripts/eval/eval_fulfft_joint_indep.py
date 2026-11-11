from data.normalize import normalize_amharic
"""
Amharic ASR - final TEST evaluation of the full fine-tuned (no LoRA) model on the
joint speaker+text-independent split.
"""

import os

CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from dataclasses import dataclass
from typing import Any, Dict, List

import jiwer
from common import config as cfg  # must load before transformers/datasets
import torch
from datasets import load_from_disk
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)

BASE_DIR    = "~/AmharicASR/amharic_workspace"
PHASE_DIR   = os.path.join(BASE_DIR, "phase_fullft_joint_indep")
MODEL_DIR   = os.path.join(PHASE_DIR, "models", "fullft_joint_indep_final")
TEST_CACHE  = os.path.join(PHASE_DIR, "dataset_cache")      
RESULTS_DIR = os.path.join(PHASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 8))
NUM_BEAMS       = int(os.environ.get("NUM_BEAMS", 5))
DIALECTS        = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]

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

print(f"Loading model + processor from {MODEL_DIR} ...", flush=True)
processor = WhisperProcessor.from_pretrained(MODEL_DIR, language="amharic", task="transcribe")
DECODER_START_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

# OPTIMIZATION 1: Use bfloat16 and SDPA (Flash Attention) for massive speedup
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="sdpa"
).to("cuda").eval()

model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length = 448
model.generation_config.max_new_tokens = None
model.generation_config.num_beams = NUM_BEAMS
model.generation_config.use_cache = True

print(f"Loading test split from {TEST_CACHE} ...", flush=True)
test_dataset = load_from_disk(TEST_CACHE)["test"]
test_dataset = test_dataset.add_column("orig_idx", list(range(len(test_dataset))))
print(f"  test utterances: {len(test_dataset):,}", flush=True)

# OPTIMIZATION 2: Extract dialect list instantly in Python to avoid 30 min of disk I/O
print("Pre-computing dialect indices to bypass disk I/O bottleneck...", flush=True)
all_dialects = test_dataset["dialect"]

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

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids",
                  "attention_mask", "dialect", "orig_idx"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            # OPTIMIZATION 3: Ensure inputs are cast to the model's bfloat16 dtype
            inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# OPTIMIZATION 4: Enable bf16 and dataloader workers
eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_fullft_test",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    predict_with_generate=True,
    bf16=True, fp16=False,
    dataloader_num_workers=4,
    report_to=["none"], remove_unused_columns=False,
)
trainer = SafeWhisperTrainer(
    model=model, args=eval_args,
    data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor, DECODER_START_TOKEN_ID),
)

persample_path = os.path.join(RESULTS_DIR, "fullft_joint_indep_TEST_persample.jsonl")
results_path = os.path.join(RESULTS_DIR, "fullft_joint_indep_TEST.json")
persample = open(persample_path, "w", encoding="utf-8")

dialect_results, all_refs, all_hyps = {}, [], []
n_empty_ref = 0

def score(refs, hyps):
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs: return 0.0, 0.0
    return (jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) * 100,
            jiwer.cer([r for r, _ in pairs], [h for _, h in pairs]) * 100)

for d_name in DIALECTS:
    # OPTIMIZATION 5: Instant selection via indices instead of .filter()
    d_indices = [i for i, d in enumerate(all_dialects) if d == d_name]
    if not d_indices:
        continue
    d_test = test_dataset.select(d_indices)
    
    print(f"\nRunning {d_name} (n={len(d_test)}) ...", flush=True)
    result = trainer.predict(d_test)

    pred_ids = result.predictions.copy()
    pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    hyps = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    refs = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    for idx, r, h in zip(d_test["orig_idx"], refs, hyps):
        persample.write(json.dumps({"orig_idx": int(idx), "dialect": d_name,
                                    "reference": r, "prediction": h}, ensure_ascii=False) + "\n")
    persample.flush()

    n_empty_ref += sum(1 for r in refs if not r)
    wer, cer = score(refs, hyps)
    runaway = sum(1 for r, h in zip(refs, hyps) if r and len(h) > 2 * len(r))
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2),
                               "hyp_over_2x_ref_chars": runaway}
    print(f"  {d_name:13s} n={len(d_test):4d}  WER={wer:.2f}%  CER={cer:.2f}%  "
          f"(hyps >2x reference length: {runaway})", flush=True)
    all_refs.extend(refs)
    all_hyps.extend(hyps)

persample.close()

wer, cer = score(all_refs, all_hyps)
dialect_results["OVERALL"] = {"n_samples": len(all_refs), "wer": round(wer, 2), "cer": round(cer, 2),
                              "empty_references_skipped": n_empty_ref}

print(f"\n{'=' * 62}\n  FULL FINE-TUNE (joint speaker+text-independent split) - TEST, beam={NUM_BEAMS}\n{'=' * 62}")
for d, m in dialect_results.items():
    print(f"  {d:<13} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%"
          + ("  <-" if d == "OVERALL" else ""))
print("=" * 62)
print("  LoRA reference points (same split): Raw+ALFFA+CV 32.06% | Phase1+ALFFA+CV 32.95% | Leyu-only 33.47%")

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved {results_path}\nSaved {persample_path}")