import os
import json
import torch
import random
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# --- STRICT OFFLINE FLAGS ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = "~/AmharicASR/amharic_workspace"

# Model Paths
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
ADAPTER_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_lora_adapter")
MERGED_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_clean_merged_fp32")
LEYU_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. LOAD PROCESSOR FROM BASE MODEL
print(f"Loading processor from {PHASE1_MERGED}...")
processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")

def setup_config(model):
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.max_new_tokens = 225
    model.generation_config.num_beams = 5
    model.generation_config.use_cache = True
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.no_repeat_ngram_size = 0
    return model

# 2. LOAD UNMERGED
print("\nLoading UNMERGED model (Base + LoRA)...")
base_model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.float32)
unmerged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device).eval()
unmerged_model = setup_config(unmerged_model)

# 3. LOAD MERGED
print("\nLoading MERGED model...")
merged_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to(device).eval()
merged_model = setup_config(merged_model)

# 4. LOAD DATASET
print("\nLoading test dataset...")
test_split = load_from_disk(LEYU_CACHE)["test"]
gojjam_split = test_split.filter(lambda x: x["dialect"] == "gojjam")

# Pick 5 random indices
random.seed(42)
indices = random.sample(range(len(gojjam_split)), 5)

print("\nGenerating transcriptions for 5 random samples...")

results = []

for i, idx in enumerate(indices):
    sample = gojjam_split[idx]
    
    # Force audio to float32
    input_features = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get ground truth
    label_ids = [t for t in sample["labels"] if t != -100]
    ref_text = processor.decode(label_ids, skip_special_tokens=True)
    
    with torch.no_grad():
        unmerged_tokens = unmerged_model.generate(input_features)
        merged_tokens = merged_model.generate(input_features)
        
    unmerged_text = processor.batch_decode(unmerged_tokens, skip_special_tokens=True)[0]
    merged_text = processor.batch_decode(merged_tokens, skip_special_tokens=True)[0]
    
    sample_result = {
        "sample_number": i + 1,
        "dataset_index": idx,
        "reference": ref_text,
        "unmerged_hypothesis": unmerged_text,
        "merged_hypothesis": merged_text,
        "unmerged_token_count": len(unmerged_tokens[0]),
        "merged_token_count": len(merged_tokens[0]),
        "unmerged_raw_tokens": unmerged_tokens[0].tolist(),
        "merged_raw_tokens": merged_tokens[0].tolist(),
        "hit_max_tokens_warning": len(merged_tokens[0]) >= 225
    }
    results.append(sample_result)

# 5. SAVE AND PRINT JSON
output_file = os.path.join(BASE_DIR, "diagnose_samples.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\n" + "="*80)
print(f"  SAVED JSON TO: {output_file}")
print("="*80 + "\n")

# Print the JSON to the console so you can read it immediately
print(json.dumps(results, ensure_ascii=False, indent=4))