import os

CUSTOM_CACHE = os.environ.get("STT_CACHE_DIR", os.path.expanduser("~/AmharicASR/clean_cache"))
os.makedirs(CUSTOM_CACHE, exist_ok=True)
os.environ["HF_HOME"] = CUSTOM_CACHE
os.environ["HF_DATASETS_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_CACHE"] = CUSTOM_CACHE
os.environ["HF_MODULES_CACHE"] = CUSTOM_CACHE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json, torch
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List
import jiwer  # pure-python, no hub/cache dependency — pip install jiwer if not already present

BASE_DIR      = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
MERGED_DIR    = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_clean_merged_fp32")
LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
RESULTS_DIR   = os.path.join(BASE_DIR, "phase2b_leyu_standard/results")
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_JSONL = os.path.join(RESULTS_DIR, "gojjam_merged_fixed_persample.jsonl")

processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
DECODER_START = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

merged = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to("cuda").eval()
merged.generation_config.forced_decoder_ids = forced_ids
merged.generation_config.max_new_tokens = 225
merged.generation_config.num_beams = 5
merged.generation_config.use_cache = True

@dataclass
class Collator:
    processor: Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        lids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (lids[:, 0] == self.decoder_start_token_id).all().cpu().item():
            lids = lids[:, 1:]
        batch["labels"] = lids
        return batch

collator = Collator(processor=processor, decoder_start_token_id=DECODER_START)

all_pred, all_ref = [], []
f_out = open(OUT_JSONL, "w", encoding="utf-8")

def compute_metrics(pred):
    pred_ids = pred.predictions.copy(); pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
    label_ids = pred.label_ids.copy(); label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    for p, r in zip(pred_str, label_str):
        f_out.write(json.dumps({"reference": r, "prediction": p}, ensure_ascii=False) + "\n")
        f_out.flush()
        all_pred.append(p); all_ref.append(r)
    return {"wer_running": round(jiwer.wer(label_str, pred_str) * 100, 2)}  # per-batch, informational only

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect", "path", "duration_sec"]:
            inputs.pop(k, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

test_dataset = load_from_disk(LEYU_CACHE)["test"]
gojjam = test_dataset.filter(lambda x: x["dialect"] == "gojjam")

eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch_merged_fixed", per_device_eval_batch_size=8,
    predict_with_generate=True, bf16=False, fp16=False,
    report_to=["none"], remove_unused_columns=False,
)
trainer = SafeWhisperTrainer(model=merged, args=eval_args, data_collator=collator, compute_metrics=compute_metrics)
trainer.predict(gojjam)
f_out.close()

overall_wer = jiwer.wer(all_ref, all_pred) * 100
overall_cer = jiwer.cer(all_ref, all_pred) * 100
print(f"\nGOJJAM — n={len(all_pred)}  WER={overall_wer:.2f}%  CER={overall_cer:.2f}%")
print(f"Per-sample predictions saved to {OUT_JSONL}")