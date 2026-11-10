import os, sys, torch
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME="/home/stud_homes/s8345520/AmharicASR/hf_cache")
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
adapter, out = sys.argv[1], sys.argv[2]
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", torch_dtype=torch.bfloat16).float()
model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
model.save_pretrained(out); WhisperProcessor.from_pretrained(adapter).save_pretrained(out)
print("saved", out, next(model.parameters()).dtype)
