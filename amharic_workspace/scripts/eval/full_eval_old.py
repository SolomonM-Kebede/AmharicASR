from data.normalize import normalize_amharic
from common import config as cfg  # must load before transformers/datasets
import torch
import evaluate
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

# ── 1. Configuration ────────────────────────────────────────────────────────
MODEL_DIR = "/tmp/amharic_workspace/whisper-amharic-model-unzipped" 
DATASET_CACHE = "/tmp/amharic_workspace/amharic_dataset_cache/amharic_dataset_cache_processed"

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ":"ሰ", "ሡ":"ሱ", "ሢ":"ሲ", "ሣ":"ሳ", "ሤ":"ሴ", "ሥ":"ስ", "ሦ":"ሶ",
    "ፀ": "ጸ", "ፁ":"ጹ", "ፂ":"ጺ", "ፃ":"ጻ", "ፄ":"ጼ", "ፅ":"ጽ", "ፆ":"ጾ"
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text):
    for src, tgt in AMHARIC_CHAR_MAP.items(): text = text.replace(src, tgt)
    for p in PUNCT: text = text.replace(p, ' ')
    return " ".join(text.split()).strip()

# ── 2. Load Model & Data ──────────────────────────────────────────────────
print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map="auto")
model.eval()

dataset = load_from_disk(DATASET_CACHE)["test"]
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# ── 3. Robust Generation Config ───────────────────────────────────────────
forced_decoder_ids = processor.get_decoder_prompt_ids(language="amharic", task="transcribe", no_timestamps=True)

predictions, references = [], []
batch_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 4. Inference Loop ─────────────────────────────────────────────────────
print(f"Running Beam Search evaluation on {len(dataset)} samples...")
with torch.no_grad():
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i : i + batch_size]
        input_features = torch.tensor(batch["input_features"]).to(device, dtype=torch.float16)
        
        # Ground Truth
        labels = batch["labels"]
        labels_padded = [[l if l != -100 else processor.tokenizer.pad_token_id for l in label] for label in labels]
        decoded_labels = processor.tokenizer.batch_decode(labels_padded, skip_special_tokens=True)
        
        # The Fix: Beam Search + Repetition Penalty
        generated_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
            num_beams=5,                  # Explores 5 paths to avoid dead ends
            repetition_penalty=1.1,       # Slightly penalizes repeating the same word
            use_cache=True
        )
        decoded_preds = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for pred, ref in zip(decoded_preds, decoded_labels):
            predictions.append(normalize_amharic(pred))
            references.append(normalize_amharic(ref))

# ── 5. Results ────────────────────────────────────────────────────────────
final_wer = wer_metric.compute(predictions=predictions, references=references) * 100
final_cer = cer_metric.compute(predictions=predictions, references=references) * 100

print("\n" + "="*50)
print(f"🎉 REAL BEAM SEARCH METRICS 🎉")
print(f"Test WER: {final_wer:.2f}%")
print(f"Test CER: {final_cer:.2f}%")
print("="*50)

print("\nSample Output Check:")
for i in range(min(20, len(predictions))):
    print(f"True: {references[i]}\nPred: {predictions[i]}\n")