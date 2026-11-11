import os
import zipfile
import shutil
import pandas as pd
from common import config as cfg  # must load before transformers/datasets
from datasets import load_dataset, Audio
from tqdm import tqdm

# ==========================================
# CONFIGURATION - UPDATE THESE!
# ==========================================
HF_DATASET_NAME = "snapwre/amharic-speech" 
CONFIG_NAME = None                             # Kept as None for datasets without configs
SPLIT = "test"                                 # "test", "validation", or "train"
TEXT_COLUMN = "sentence"                  # The name of the column containing the Amharic text
# ==========================================

def main():
    print(f"Downloading {HF_DATASET_NAME} - {SPLIT} split...")
    
    # Load the dataset
    if CONFIG_NAME:
        ds = load_dataset(HF_DATASET_NAME, CONFIG_NAME, split=SPLIT)
    else:
        ds = load_dataset(HF_DATASET_NAME, split=SPLIT)
        
    # --- CRITICAL FIX: Bypass torchcodec and FFmpeg by disabling decoding ---
    ds = ds.cast_column("audio", Audio(decode=False))
        
    print(f"Loaded {len(ds)} audio files. Preparing files...")

    # Create a temporary directory for the audio files
    os.makedirs("hf_temp_audio", exist_ok=True)
    manifest_data = []

    # Iterate through the dataset and save raw files
    for i, row in enumerate(tqdm(ds)):
        audio_info = row["audio"] 
        transcript = row[TEXT_COLUMN]
        
        # Grab the original file extension (e.g. .mp3, .wav), default to .wav
        orig_path = audio_info.get("path") or ""
        ext = os.path.splitext(orig_path)[1]
        if not ext:
            ext = ".wav"
            
        filename = f"utterance_{i:04d}{ext}"
        filepath = os.path.join("hf_temp_audio", filename)
        
        # Dump the raw audio bytes directly to disk
        if audio_info.get("bytes"):
            with open(filepath, "wb") as f:
                f.write(audio_info["bytes"])
        elif orig_path:
            shutil.copy(orig_path, filepath)
        else:
            print(f"Warning: Audio {i} missing bytes and path. Skipping.")
            continue
            
        # Add to our CSV tracking list
        manifest_data.append({
            "filename": filename,
            "reference": transcript
        })

    # Save the manifest.csv
    df = pd.DataFrame(manifest_data)
    df.to_csv("hf_manifest.csv", index=False)
    print("\nSaved: hf_manifest.csv")

    # Zip the audio files
    zip_name = "hf_audio_files.zip"
    print(f"Zipping audio into {zip_name}...")
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir("hf_temp_audio"):
            zf.write(os.path.join("hf_temp_audio", f), arcname=f)
            
    print(f"Saved: {zip_name}")
    print("\nDone! You can now upload 'hf_audio_files.zip' and 'hf_manifest.csv' to your Gradio app.")

if __name__ == "__main__":
    main()