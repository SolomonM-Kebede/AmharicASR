"""Merge equivalence test for the Phase 2 LoRA adapter. No training.

Checks three things on a fixed random input:
  1. The merged weights equal base + delta for one attention projection.
  2. The merged model's logits match the unmerged LoRA model's logits.
  3. The tied decoder embedding and output projection share storage, before
     and after the merge.

Runs in float32 on purpose: the test is about exact equivalence, and bfloat16
would add rounding noise that hides a real difference.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

TOLERANCE = 1e-4

PHASE1_MERGED = cfg.require(cfg.PHASE1_MERGED)
ADAPTER_DIR = cfg.require(cfg.DATA_DIR / "phase2b_leyu_standard" / "models" / "phase2b_lora_adapter")

processor = WhisperProcessor.from_pretrained(str(PHASE1_MERGED), language="amharic", task="transcribe")
prefix = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)
decoder_start = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
# Prefix as (position, token) pairs; the decoder input starts with the start token.
decoder_ids = torch.tensor([[decoder_start] + [tok for _, tok in prefix]], dtype=torch.long)
print("decoder input ids:", decoder_ids.tolist())


def tied(a, b):
    return a.data_ptr() == b.data_ptr()


print("\n1. Unmerged LoRA model (float32)")
base = WhisperForConditionalGeneration.from_pretrained(str(PHASE1_MERGED), torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).eval()
inner = peft_model.base_model.model

print("   decoder embeddings tied to output projection:",
      tied(inner.model.decoder.embed_tokens.weight, inner.proj_out.weight))

torch.manual_seed(cfg.SEED)
audio = torch.randn(1, 128, 3000, dtype=torch.float32)  # fixed random features, not speech
with torch.no_grad():
    logits_unmerged = peft_model(input_features=audio, decoder_input_ids=decoder_ids).logits

    layer = inner.model.encoder.layers[0].self_attn.k_proj
    expected = layer.base_layer.weight.data + layer.get_delta_weight("default")

print("\n2. Merging with merge_and_unload(safe_merge=True)")
merged = peft_model.merge_and_unload(safe_merge=True).eval()

actual = merged.model.encoder.layers[0].self_attn.k_proj.weight.data
weight_gap = (expected - actual).abs().max().item()
print(f"   encoder layer 0, k_proj: max |base + delta - merged| = {weight_gap:.3e}")

print("\n3. Merged model")
print("   decoder embeddings tied to output projection:",
      tied(merged.model.decoder.embed_tokens.weight, merged.proj_out.weight))

with torch.no_grad():
    logits_merged = merged(input_features=audio, decoder_input_ids=decoder_ids).logits

logit_gap = (logits_unmerged - logits_merged).abs().max().item()
print(f"\n4. Max |logit difference| over the forward pass: {logit_gap:.3e}")
verdict = "EQUIVALENT" if logit_gap < TOLERANCE else "NOT EQUIVALENT"
print(f"   threshold {TOLERANCE:g}: {verdict}")