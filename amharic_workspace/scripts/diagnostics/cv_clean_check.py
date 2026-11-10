import os
import pandas as pd

CV_DIR = "/home/stud_homes/s8345520/AmharicASR/amharic_workspace/cv-corpus-25.0-2026-03-09/am"
ALFFA_COUNT = 10_875   # ALFFA examples before quota (from log)
LEYU_TRAIN = 10_296    # quota (from log)
SEED = 42

def load(name):
    return pd.read_csv(os.path.join(CV_DIR, name), sep="\t", quoting=3)

# 1. Structure check: is validated.tsv the union of the split files?
tr, dv, te = (set(load(n)["path"]) for n in ["train.tsv", "dev.tsv", "test.tsv"])
oth = set(load("other.tsv")["path"])
val = set(load("validated.tsv")["path"])

print("== Structure ==")
print("validated == train | dev | test:", val == (tr | dv | te))
print("other disjoint from validated: ", not (oth & val))
print("other disjoint from train|dev|test:", not (oth & (tr | dv | te)))

# 2. Three candidate pools
def pool(files, dedup):
    df = pd.concat([load(f) for f in files], ignore_index=True)
    if dedup:
        df = df.drop_duplicates(subset="path", keep="first")
    return df

pools = {
    "as used (5 files, no dedup)": pool(["train.tsv", "dev.tsv", "test.tsv", "other.tsv", "validated.tsv"], dedup=False),
    "all 5 files, dedup by path": pool(["train.tsv", "dev.tsv", "test.tsv", "other.tsv", "validated.tsv"], dedup=True),
    "train.tsv only": pool(["train.tsv"], dedup=True),
}

# 3. Same quota rule as training, applied to each pool
print("\n== Quota sampling (same rule and seed as training) ==")
print(f"{'pool':32s} {'rows':>6s} {'distinct':>9s} {'keep':>7s} {'CV kept':>8s} {'share':>7s} {'repeats':>8s}")
for label, df in pools.items():
    n = len(df)
    total = ALFFA_COUNT + n
    f = LEYU_TRAIN / total
    n_keep = int(n * f)
    # Sample row indices with the same seeded shuffle; paths identify clips
    idx = list(range(n))
    import random
    rng = random.Random(SEED)
    rng.shuffle(idx)
    kept_paths = [df.iloc[i]["path"] for i in idx[:n_keep]]
    distinct = len(set(kept_paths))
    print(f"{label:32s} {n:>6,} {df['path'].nunique():>9,} {f:>7.4f} "
          f"{n_keep:>8,} {n_keep / LEYU_TRAIN:>7.1%} {n_keep - distinct:>8,}")