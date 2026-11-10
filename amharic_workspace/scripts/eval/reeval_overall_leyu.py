import os
import json
import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import evaluate

# 1. Strict Offline Mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = "~/AmharicASR/amharic_workspace"

# 2. CRITICAL FIX: Pointing to Phase 2b Joint Training Merged Model!
MERGED_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_clean_merged_fp32")

# We still test on the Leyu-only test cache
CACHE_DIR = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")
RESULTS_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/results")

os.makedirs(RESULTS_DIR, exist_ok=True)

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

print(f"Loading processor and model from {MERGED_DIR}...", flush=True)
processor = WhisperProcessor.from_pretrained(MERGED_DIR, language="amharic", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR).to("cuda").eval()
model_dtype = next(model.parameters()).dtype

# Proper in-place generation config
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
model.generation_config.max_new_tokens = 225
model.generation_config.num_beams = 5
model.generation_config.use_cache = True
model.generation_config.repetition_penalty = 1.0
model.generation_config.no_repeat_ngram_size = 0

print(f"Loading test dataset from {CACHE_DIR}...", flush=True)
test_dataset = load_from_disk(CACHE_DIR)["test"]

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

DIALECT_NAMES = ["gojjam", "gonder", "wello", "shewa", "addis ababa"]
dialect_results = {}
all_pred, all_ref = [], []

print("\nStarting Pure PyTorch Inference...", flush=True)

for d_name in DIALECT_NAMES:
    d_test = test_dataset.filter(lambda x: x["dialect"] == d_name)
    if len(d_test) == 0: continue
    
    print(f"\nRunning {d_name} (n={len(d_test)})...", flush=True)
    predictions, references = [], []
    
    with torch.no_grad():
        for item in tqdm(d_test, desc=d_name):
            # Dynamic dtype casting to prevent FP16/FP32 mismatch crashes
            input_features = torch.tensor(item["input_features"]).unsqueeze(0).to(device="cuda", dtype=model_dtype)
            
            # Generate!
            pred_ids = model.generate(input_features)
            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            
            # Get Ground Truth
            label_ids = [t for t in item["labels"] if t != -100]
            label_str = processor.decode(label_ids, skip_special_tokens=True)
            
            # Normalize
            predictions.append(normalize_amharic(pred_str))
            references.append(normalize_amharic(label_str))
            
    wer = wer_metric.compute(predictions=predictions, references=references) * 100
    cer = cer_metric.compute(predictions=predictions, references=references) * 100
    
    dialect_results[d_name] = {"n_samples": len(d_test), "wer": round(wer, 2), "cer": round(cer, 2)}
    print(f"--> {d_name.upper()} RESULTS: WER={wer:.2f}% | CER={cer:.2f}%", flush=True)
    
    all_pred.extend(predictions)
    all_ref.extend(references)

overall_wer = wer_metric.compute(predictions=all_pred, references=all_ref) * 100
overall_cer = cer_metric.compute(predictions=all_pred, references=all_ref) * 100
dialect_results["OVERALL"] = {"n_samples": len(all_pred), "wer": round(overall_wer, 2), "cer": round(overall_cer, 2)}


print(f"  PHASE 2B MERGED MODEL FINAL RESULTS", flush=True)
for d, m in dialect_results.items():
    print(f"  {d:<18} n={m['n_samples']:>5}  WER={m['wer']:>6.2f}%  CER={m['cer']:>6.2f}%", flush=True)
out_file = os.path.join(RESULTS_DIR, "phase2b_clean_merged_pure_inference.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(dialect_results, f, ensure_ascii=False, indent=2)
print(f"Results saved to {out_file}", flush=True)