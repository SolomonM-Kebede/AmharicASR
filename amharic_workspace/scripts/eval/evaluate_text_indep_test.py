import os
import io
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_from_disk, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# --- Environment ---
CUSTOM_CACHE = "~/AmharicASR/hf_cache"
os.environ["HF_HOME"] = CUSTOM_CACHE

BASE_DIR = "~/AmharicASR/amharic_workspace"
DATASET_DIR = os.path.join(BASE_DIR, "phase2_leyu/models/joint_indep_lora_run/final_merged")
MERGED_DIR = os.path.join(BASE_DIR, "phase2_leyu/models/text_indep_lora_run/final_merged")

print(f"Loading processor and Phase 1 merged model from {MERGED_DIR}...")
processor = WhisperProcessor.from_pretrained(MERGED_DIR)

# --- Load & Prepare TEST Dataset ONLY ---
print("Loading TEST dataset...")
raw_ds = load_from_disk(DATASET_DIR)
raw_test = raw_ds["test"]
raw_test = raw_test.cast_column("audio", Audio(decode=False)) 

def prepare_dataset(batch):
    audio = batch["audio"]
    if audio.get("bytes") is not None:
        arr, _ = librosa.load(io.BytesIO(audio["bytes"]), sr=16_000, mono=True)
    elif audio.get("path") is not None:
        arr, _ = librosa.load(audio["path"], sr=16_000, mono=True)
    else:
        arr = audio["array"]
        
    batch["input_features"] = processor.feature_extractor(arr, sampling_rate=16_000).input_features[0]
    batch["labels"] = processor.tokenizer(batch["norm_text"], max_length=448, truncation=True).input_ids
    return batch

cols_to_remove = [c for c in raw_test.column_names if c != "dialect"]
print(f"Extracting log-Mel features for {len(raw_test)} test samples...")
processed_test = raw_test.map(
    prepare_dataset, 
    remove_columns=cols_to_remove,
    num_proc=4 
)
print(f"Loading model from {MERGED_DIR}...")
model = WhisperForConditionalGeneration.from_pretrained(
    MERGED_DIR, 
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to("cuda")

# --- IN-PLACE MUTATION (Matches Phase 2 True Eval) ---
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
print(forced_ids)
model.generation_config.forced_decoder_ids = forced_ids
model.generation_config.max_length         = 448   
model.generation_config.num_beams          = 5
model.generation_config.repetition_penalty = 1.0
model.generation_config.no_repeat_ngram_size = 0
model.config.use_cache                     = True

# --- Normalization ---
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



# --- Collator & Metrics ---
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

class SafeWhisperTrainer(Seq2SeqTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        for key in ["decoder_input_ids", "decoder_inputs_embeds", "input_ids", "attention_mask", "dialect"]:
            inputs.pop(key, None)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype=next(model.parameters()).dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

# --- Evaluate ---
eval_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/eval_scratch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    bf16=True,
    report_to=["none"],
    remove_unused_columns=False,
)

eval_trainer = SafeWhisperTrainer(
    model=model,
    args=eval_args,
    data_collator=data_collator,
)

# --- Per-Dialect & Overall Evaluation ---
print("\nRunning per-dialect evaluation on the full test set...")

dialect_results = {}
unique_dialects = sorted(list(set(processed_test["dialect"])))

for dialect in unique_dialects:
    print(f"  Evaluating {dialect}...")
    dialect_subset = processed_test.filter(lambda x: x["dialect"] == dialect)
    
    # Extract manual predictions to match Phase 2 logic
    result = eval_trainer.predict(dialect_subset)
    label_ids = result.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(result.predictions, skip_special_tokens=True)]
    label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    import evaluate as _eval
    wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
    cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

    dialect_results[dialect] = {
        "n_samples": len(dialect_subset),
        "wer": wer,
        "cer": cer,
    }

print("  Evaluating OVERALL...")
overall_result = eval_trainer.predict(processed_test)
label_ids = overall_result.label_ids.copy()
label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

pred_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(overall_result.predictions, skip_special_tokens=True)]
label_str = [normalize_amharic(s) for s in processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

overall_wer = _eval.load("wer").compute(predictions=pred_str, references=label_str) * 100
overall_cer = _eval.load("cer").compute(predictions=pred_str, references=label_str) * 100

dialect_results["OVERALL"] = {
    "n_samples": len(processed_test),
    "wer": overall_wer,
    "cer": overall_cer,
}

# --- Format Output ---
print(f"\n{'='*55}")
print(f"  TEXT-INDEPENDENT ABLATION — FINAL TEST RESULTS (n_beams=5)")
print(f"{'='*55}")
print(f"  {'Dialect':<18} {'Samples':>7}  {'WER':>8}  {'CER':>8}")
print(f"  {'-'*46}")
for dialect in unique_dialects:
    metrics = dialect_results[dialect]
    print(f"  {dialect:<18} {metrics['n_samples']:>7d}  {metrics['wer']:>7.2f}%  {metrics['cer']:>7.2f}%")
print(f"  {'-'*46}")
overall_metrics = dialect_results["OVERALL"]
print(f"  {'OVERALL':<18} {overall_metrics['n_samples']:>7d}  {overall_metrics['wer']:>7.2f}%  {overall_metrics['cer']:>7.2f}% ←")
print(f"{'='*55}")