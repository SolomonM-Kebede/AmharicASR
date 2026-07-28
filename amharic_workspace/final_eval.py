import torch
import numpy as np
import evaluate
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, GenerationConfig
from peft import PeftModel
from datasets import load_from_disk

MODEL_ID        = "openai/whisper-large-v3"
CHECKPOINT_PATH = "/tmp/amharic_workspace/whisper-amharic-lora_32_merged"
CACHE_FILE      = "/tmp/amharic_workspace/amharic_dataset_cache/amharic_dataset_cache_processed"
BATCH_SIZE      = 16

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ"
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'

def normalize_amharic(text: str) -> str:
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, ' ')
    return " ".join(text.split()).strip()

print("Loading dataset...")
dataset   = load_from_disk(CACHE_FILE)["test"]
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="amharic", task="transcribe")

print(f"Loading merged standalone model from {CHECKPOINT_PATH}...")
model = WhisperForConditionalGeneration.from_pretrained(
    CHECKPOINT_PATH, 
    torch_dtype=torch.float16, 
    device_map="auto"
).eval()

forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="amharic", task="transcribe", no_timestamps=True
)
gen_config = GenerationConfig(
    forced_decoder_ids=forced_decoder_ids,
    max_new_tokens=225,
    repetition_penalty=1.2,
    no_repeat_ngram_size=5,
    use_cache=True,
)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

predictions, references = [], []

print(f"Running inference on {len(dataset)} test samples...")
with torch.no_grad():
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]

        input_features = torch.from_numpy(
            np.array(batch["input_features"], dtype=np.float32)
        ).to("cuda", dtype=torch.float16)

        generated_ids = model.generate(
            input_features=input_features,
            generation_config=gen_config,
        )

        decoded_preds = processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        labels_clean = [
            [tok if tok != -100 else processor.tokenizer.pad_token_id for tok in lab]
            for lab in batch["labels"]
        ]
        decoded_labels = processor.tokenizer.batch_decode(
            labels_clean, skip_special_tokens=True
        )

        for pred, ref in zip(decoded_preds, decoded_labels):
            predictions.append(normalize_amharic(pred))
            references.append(normalize_amharic(ref))

final_wer = wer_metric.compute(predictions=predictions, references=references) * 100
final_cer = cer_metric.compute(predictions=predictions, references=references) * 100

print("\n" + "=" * 50)
print(f"  WER : {final_wer:.2f}%")
print(f"  CER : {final_cer:.2f}%")
print("=" * 50)

print("\nSample transcriptions (⚠️ = prediction >2× reference length):")
for i in range(min(20, len(predictions))):
    pred_words = len(predictions[i].split())
    ref_words  = len(references[i].split())
    flag = " ⚠️ LONG" if ref_words > 0 and pred_words > 2 * ref_words else ""
    print(f"  Ref  : {references[i]}")
    print(f"  Pred : {predictions[i]}{flag}\n")