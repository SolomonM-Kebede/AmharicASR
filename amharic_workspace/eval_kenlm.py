"""
Amharic ASR — Final Evaluation with KenLM Shallow Fusion Rescoring
Target: Leyu Dialects Test Set
"""

import os
import torch
import kenlm
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperProcessor, WhisperForConditionalGeneration

print(f"torch  : {torch.__version__}")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# CONFIGURATION
BASE_DIR = "/tmp/amharic_workspace"
FINAL_MODEL = os.path.join(BASE_DIR, "phase2_leyu/models/phase2_final_merged")
KENLM_MODEL = os.path.join(BASE_DIR, "amharic_5gram.bin")

# Rescoring Hyperparameters 
ALPHA = 0.3  # Language Model Weight
BETA = 1.0   # Word Insertion Penalty


# NORMALIZATION
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


# INITIALIZATION
print("Loading Processor and Model...")
processor = WhisperProcessor.from_pretrained(FINAL_MODEL, language="amharic", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(FINAL_MODEL).to(device)
model.eval()

print("Loading KenLM 5-Gram Model...")
lm = kenlm.Model(KENLM_MODEL)

wer_metric = evaluate.load("wer")

print("Loading Leyu Test Sets...")
dialects = ["gojjam", "gonder", "wello", "shewa"]
test_dsets = []
for dialect in dialects:
    ds = load_dataset(f"leyu-amharic/leyu-amharic-{dialect}-dialect", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=True))
    test_dsets.append(ds)

test_dataset = concatenate_datasets(test_dsets)


# EVALUATION LOOP
raw_whisper_preds = []
rescored_preds = []
references = []

print(f"Starting Evaluation on {len(test_dataset)} samples...")

with torch.no_grad():
    for item in tqdm(test_dataset):
        audio = item["audio"]["array"]
        ref_text = normalize_amharic(item["sentence"])
        references.append(ref_text)
        
        # 1. Feature Extraction
        input_features = processor(audio, sampling_rate=16_000, return_tensors="pt").input_features.to(device)
        
        # 2. Whisper Beam Search (Generate Top 5)
        outputs = model.generate(
            input_features,
            max_new_tokens=225,
            num_beams=5,
            num_return_sequences=5,
            return_dict_in_generate=True,
            output_scores=True,
            language="amharic",
            task="transcribe"
        )
        
        best_rescored_text = ""
        best_final_score = -float("inf")
        
        # The first sequence is Whisper's #1 raw acoustic choice
        raw_whisper_preds.append(processor.decode(outputs.sequences[0], skip_special_tokens=True))
        
        # 3. N-Best Rescoring with KenLM
        for i in range(5):
            sequence = outputs.sequences[i]
            # Whisper Acoustic Log-Prob
            acoustic_score = outputs.sequences_scores[i].item()
            
            text = processor.decode(sequence, skip_special_tokens=True)
            norm_text = normalize_amharic(text)
            word_count = len(norm_text.split())
            
            # KenLM Grammatical Log-Prob
            lm_score = lm.score(norm_text)
            
            # The Rescoring Formula
            final_score = acoustic_score + (ALPHA * lm_score) + (BETA * word_count)
            
            if final_score > best_final_score:
                best_final_score = final_score
                best_rescored_text = norm_text
                
        rescored_preds.append(best_rescored_text)


# RESULTS
raw_whisper_preds = [normalize_amharic(t) for t in raw_whisper_preds]

baseline_wer = wer_metric.compute(predictions=raw_whisper_preds, references=references) * 100
rescored_wer = wer_metric.compute(predictions=rescored_preds, references=references) * 100

print(f"\n{'='*50}")
print(f"FINAL METRICS")
print(f"{'='*50}")
print(f"Raw Whisper WER      : {baseline_wer:.2f}%")
print(f"KenLM Rescored WER   : {rescored_wer:.2f}%")
print(f"Absolute Improvement : {(baseline_wer - rescored_wer):.2f}%")
print(f"{'='*50}")