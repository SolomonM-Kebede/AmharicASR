import os
from common import config as cfg  # must load before transformers/datasets
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

BASE_DIR      = str(cfg.REPO_ROOT)
PHASE1_MERGED = BASE_DIR/ "phase1_curriculum/models/phase1_best_model_merged"
ADAPTER_DIR = BASE_DIR/ "phase2b_leyu_standard/models/phase2b_lora_adapter"
OUTPUT_DIR = BASE_DIR/ "phase2b_leyu_standard/models/phase2b_clean_merged_fp32"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load in strict FP32 (Your brilliant fix)
print("Loading base model in strict FP32...")
base = WhisperForConditionalGeneration.from_pretrained(
    PHASE1_MERGED,
    torch_dtype=torch.float32,
)

# 2. Load the LoRA adapter
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base,
    ADAPTER_DIR,
)

# 3. Let Hugging Face do the math, but now it's forced into FP32!
print("Merging weights (merge_and_unload)...")
merged_model = model.merge_and_unload(safe_merge=True)

# 4. Save the standalone model
print(f"Saving merged FP32 model to {OUTPUT_DIR}...")
merged_model.save_pretrained(OUTPUT_DIR)

processor = WhisperProcessor.from_pretrained(PHASE1_MERGED)
processor.save_pretrained(OUTPUT_DIR)

print("Merge complete! Ready for evaluation.")