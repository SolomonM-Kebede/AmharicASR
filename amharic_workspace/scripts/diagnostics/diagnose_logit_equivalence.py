import os
import torch
from transformers import WhisperForConditionalGeneration
from peft import PeftModel

BASE_DIR = "/amharic_workspace"
PHASE1_MERGED = os.path.join(BASE_DIR, "phase1_curriculum/models/phase1_best_model_merged")
ADAPTER_DIR = os.path.join(BASE_DIR, "phase2b_leyu_standard/models/phase2b_lora_adapter")

print("1. Loading unmerged model in pure FP32...")
base_model = WhisperForConditionalGeneration.from_pretrained(PHASE1_MERGED, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).eval()

# ─── TEST A: THE TIED EMBEDDINGS POINTER TEST (BEFORE MERGE) ───
decoder_embed_ptr_unmerged = peft_model.base_model.model.model.decoder.embed_tokens.weight.data_ptr()
lm_head_ptr_unmerged = peft_model.base_model.model.proj_out.weight.data_ptr()
print(f"\n[Unmerged] Decoder Embeddings Pointer: {decoder_embed_ptr_unmerged}")
print(f"[Unmerged] LM Head Output Pointer:     {lm_head_ptr_unmerged}")
print(f"Are they tied? {decoder_embed_ptr_unmerged == lm_head_ptr_unmerged}")

# Create deterministic dummy input
torch.manual_seed(42)
dummy_audio = torch.randn(1, 128, 3000, dtype=torch.float32)
dummy_decoder_ids = torch.tensor([[50258, 50259, 50359]], dtype=torch.long) # <|startoftranscript|> <|am|> <|transcribe|>

# ─── TEST B: CAPTURE UNMERGED LOGITS ───
print("\n2. Running forward pass on UNMERGED model...")
with torch.no_grad():
    out_unmerged = peft_model(input_features=dummy_audio, decoder_input_ids=dummy_decoder_ids)
    logits_unmerged = out_unmerged.logits

# ─── TEST C: THE PER-LAYER DELTA WEIGHT TEST ───
# Let's inspect the first self-attention key projection layer in the encoder
target_layer = peft_model.base_model.model.model.encoder.layers[0].self_attn.k_proj
base_weight = target_layer.base_layer.weight.data.clone()
delta_weight = target_layer.get_delta_weight("default").clone()
theoretical_merged_weight = base_weight + delta_weight

print("\n3. Merging weights via merge_and_unload()...")
merged_model = peft_model.merge_and_unload(safe_merge=True).eval()

# Check the exact same layer after merge
actual_merged_weight = merged_model.model.encoder.layers[0].self_attn.k_proj.weight.data
delta_diff = torch.max(torch.abs(theoretical_merged_weight - actual_merged_weight))
print(f"\n[Delta Test] Max difference between theoretical merge and actual merge: {delta_diff.item():.8f}")

# ─── TEST D: THE TIED EMBEDDINGS POINTER TEST (AFTER MERGE) ───
decoder_embed_ptr_merged = merged_model.model.decoder.embed_tokens.weight.data_ptr()
lm_head_ptr_merged = merged_model.proj_out.weight.data_ptr()
print(f"\n[Merged] Decoder Embeddings Pointer: {decoder_embed_ptr_merged}")
print(f"[Merged] LM Head Output Pointer:     {lm_head_ptr_merged}")
print(f"Are they tied? {decoder_embed_ptr_merged == lm_head_ptr_merged}")

# ─── TEST E: CAPTURE MERGED LOGITS AND COMPARE ───
print("\n4. Running forward pass on MERGED model...")
with torch.no_grad():
    out_merged = merged_model(input_features=dummy_audio, decoder_input_ids=dummy_decoder_ids)
    logits_merged = out_merged.logits

logit_diff = torch.max(torch.abs(logits_unmerged - logits_merged))
print(f"\n[Logit Equivalence Test] Maximum absolute difference in output logits: {logit_diff.item():.8f}")

if logit_diff < 1e-4:
    print(">>> SUCCESS: Models are mathematically equivalent!")
else:
    print(">>> FAILURE: merge_and_unload() broke the computational graph or weights!")
    