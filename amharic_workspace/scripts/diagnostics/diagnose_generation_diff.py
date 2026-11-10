import os
import torch
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

BASE_DIR = "/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
ADAPTER_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_lora_adapter")
MERGED_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_clean_merged_fp32")
LEYU_CACHE = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Fast sample extraction (no full dataset filtering)
print("Loading dataset and finding first Gojjam sample...")
test_split = load_from_disk(LEYU_CACHE)["test"]
sample = None
for item in test_split:
    if item["dialect"] == "gojjam":
        sample = item
        break

if sample is None:
    raise ValueError("No Gojjam sample found in test dataset.")

processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
input_features = torch.tensor(sample["input_features"], dtype=torch.float32).unsqueeze(0).to(device)
reference_text = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True)

def setup_config(model):
    forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
    model.generation_config.forced_decoder_ids = forced_ids
    model.generation_config.max_new_tokens = 225
    model.generation_config.num_beams = 5
    model.generation_config.use_cache = True
    model.generation_config.repetition_penalty = 1.0
    model.generation_config.no_repeat_ngram_size = 0
    return model

# 2. Unmerged Model Generation
print("\n1. Generating with UNMERGED model...")
base_model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.float32)
unmerged_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device).eval()
unmerged_model = setup_config(unmerged_model)

with torch.no_grad():
    unmerged_tokens = unmerged_model.generate(input_features)
    unmerged_text = processor.batch_decode(unmerged_tokens, skip_special_tokens=True)[0]

# Free VRAM before loading the second model
del base_model, unmerged_model
torch.cuda.empty_cache()

# 3. Merged Model Generation
print("2. Generating with MERGED model...")
merged_model = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to(device).eval()
merged_model = setup_config(merged_model)

with torch.no_grad():
    merged_tokens = merged_model.generate(input_features)
    merged_text = processor.batch_decode(merged_tokens, skip_special_tokens=True)[0]

# 4. Display Results
print("\n" + "=" * 60)
print("  SIDE-BY-SIDE GENERATION DIFF")
print("=" * 60)
print(f"REFERENCE : {reference_text}")
print(f"UNMERGED  : {unmerged_text}")
print(f"MERGED    : {merged_text}")
print("=" * 60)
print(f"\nUnmerged Token Count : {len(unmerged_tokens[0])}")
print(f"Merged Token Count   : {len(merged_tokens[0])}")
print(f"Raw Token IDs (Unmerged): {unmerged_tokens[0].tolist()}")
print(f"Raw Token IDs (Merged)  : {merged_tokens[0].tolist()}")