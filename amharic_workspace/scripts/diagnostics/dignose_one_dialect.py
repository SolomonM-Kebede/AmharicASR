import os, torch
from common import config as cfg  # must load before transformers/datasets
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from peft import PeftModel
from dataclasses import dataclass
from typing import Any, Dict, List

BASE_DIR      = "~/AmharicASR/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
ADAPTER_DIR   = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_lora_adapter")
MERGED_DIR    = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_clean_merged_fp32")
LEYU_CACHE    = os.path.join(BASE_DIR, "phase2_leyu/dataset_cache_2026-08-expanded-spkindep")

device = "cuda"
BATCH_SIZE, N_BATCHES = 8, 2

processor = WhisperProcessor.from_pretrained(PHASE1_MERGED, language="amharic", task="transcribe")
forced_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
DECODER_START = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

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

print("Loading UNMERGED (base+adapter)...")
base_fp32 = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.float32)
unmerged = PeftModel.from_pretrained(base_fp32, ADAPTER_DIR).to(device).eval()
# in-place mutation — note this actually mutates base_fp32.generation_config under the
# hood (same object), which is the one PeftModel.generate() really reads
unmerged.generation_config.forced_decoder_ids = forced_ids
unmerged.generation_config.max_new_tokens = 225
unmerged.generation_config.num_beams = 5
unmerged.generation_config.use_cache = True

print("Loading MERGED...")
merged = WhisperForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.float32).to(device).eval()
merged.generation_config.forced_decoder_ids = forced_ids
merged.generation_config.max_new_tokens = 225
merged.generation_config.num_beams = 5
merged.generation_config.use_cache = True

test_dataset = load_from_disk(LEYU_CACHE)["test"]
gojjam = test_dataset.filter(lambda x: x["dialect"] == "gojjam")
print(f"Gojjam subset: {len(gojjam)} — using first {BATCH_SIZE * N_BATCHES}")

for b in range(N_BATCHES):
    items = [gojjam[i] for i in range(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)]
    batch = collator(items)
    input_features = batch["input_features"].to(device=device, dtype=torch.float32)
    print(f"\n=== BATCH {b} (size={len(items)}, keys={list(batch.keys())}) ===")

    with torch.no_grad():
        u_ids = unmerged.generate(input_features)
        m_ids = merged.generate(input_features)

    u_texts = processor.batch_decode(u_ids, skip_special_tokens=True)
    m_texts = processor.batch_decode(m_ids, skip_special_tokens=True)
    for j in range(len(items)):
        lbl = batch["labels"][j]
        lbl = lbl[lbl != -100]
        ref = processor.decode(lbl, skip_special_tokens=True)
        print(f"--- sample {b*BATCH_SIZE+j} ---")
        print(f"REF     : {ref}")
        print(f"UNMERGED: {u_texts[j]}")
        print(f"MERGED  : {m_texts[j]}")