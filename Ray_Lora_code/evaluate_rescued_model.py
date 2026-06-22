import os
import torch
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk

# ── 1. Paths and Config ────────────────────────────────────────────────────────
BASE_MODEL_ID = "openai/whisper-large-v3"
LORA_MODEL_PATH = "/home/stud_homes/s8345520/AmharicASR/whisper-amharic-lora_32"
DATASET_CACHE_PATH = "/tmp/amharic_workspace/amharic_dataset_cache/amharic_dataset_cache_processed"

# ── 2. Normalization Function ──────────────────────────────────────────────────
AMHARIC_CHAR_MAP = {} 
PUNCT_STR = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text):
    for search_char, replace_char in AMHARIC_CHAR_MAP.items():
        text = text.replace(search_char, replace_char)
    text = text.translate(str.maketrans('', '', PUNCT_STR))
    return ' '.join(text.split()).strip()

# ── 3. Load Model, Processor, and Dataset ──────────────────────────────────────
print("Loading processor and base model (FP16)...")
processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID, language="amharic", task="transcribe")

base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

print(f"Loading LoRA adapters from {LORA_MODEL_PATH}...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
model.eval()

print(f"Loading test dataset from {DATASET_CACHE_PATH}...")
dataset = load_from_disk(DATASET_CACHE_PATH)
test_dataset = dataset["test"]

# ── 4. Set Up Generation Guardrails ────────────────────────────────────────────
# FIXED: no_timestamps=True is passed directly here, and the suppress list is gone!
forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ── 5. Evaluation Loop ─────────────────────────────────────────────────────────
print(f"Starting evaluation on {len(test_dataset)} test samples...\n")

predictions = []
references = []
batch_size = 8 
device = "cuda" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Decoding Audio"):
        batch = test_dataset[i : i + batch_size]
        
        input_features = torch.tensor(batch["input_features"]).to(device, dtype=torch.float16)
        labels = batch["labels"]
        labels_padded = [[l if l != -100 else processor.tokenizer.pad_token_id for l in label] for label in labels]
        decoded_labels = processor.tokenizer.batch_decode(labels_padded, skip_special_tokens=True)
        
        # FIXED: suppress_tokens removed completely
        generated_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
            use_cache=True
        )
        decoded_preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, ref in zip(decoded_preds, decoded_labels):
            predictions.append(normalize_amharic(pred))
            references.append(normalize_amharic(ref))

# ── 6. Final Calculation ───────────────────────────────────────────────────────
final_wer = wer_metric.compute(predictions=predictions, references=references) * 100
final_cer = cer_metric.compute(predictions=predictions, references=references) * 100

print("\n" + "="*50)
print(f"🎉 TRUE MODEL METRICS 🎉")
print(f"Final Word Error Rate (WER): {final_wer:.2f}%")
print(f"Final Char Error Rate (CER): {final_cer:.2f}%")
print("="*50)

print("\nSample Transcriptions:")
for i in range(min(5, len(predictions))):
    print(f"True : {references[i]}")
    print(f"Pred : {predictions[i]}\n")
