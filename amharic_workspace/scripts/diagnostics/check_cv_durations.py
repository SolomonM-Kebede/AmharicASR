
import pandas as pd
from common import config as cfg  # must load before transformers/datasets
import librosa
from config import CV_DIR

def durations(tsv_name):
    df = pd.read_csv(CV_DIR / tsv_name, sep="\t", usecols=["path"])
    secs, missing = [], []
    for p in df["path"]:
        f = CV_DIR / "clips" / p
        if f.exists():
            secs.append(librosa.get_duration(path=str(f)))
        else:
            missing.append(p)
    return secs, missing

def report(name, secs, missing):
    total = sum(secs)
    print(f"{name}: {len(secs)} clips found, {len(missing)} missing")
    print(f"  total   {total:9.1f} s = {total / 3600:.3f} h")
    print(f"  mean    {total / len(secs):.3f} s   min {min(secs):.2f}   max {max(secs):.2f}")

# Control: datasheet says validated = 1,070 clips, 1.88 h, mean 6.345 s
val_secs, val_missing = durations("validated.tsv")
report("validated.tsv (control)", val_secs, val_missing)

# The number in question: train split
tr_secs, tr_missing = durations("train.tsv")
report("train.tsv", tr_secs, tr_missing)
print(f"  thesis claims 0.93 h; 553 x 6.345 s would give {553 * 6.345 / 3600:.3f} h")