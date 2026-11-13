"""Split and graph diagnostics for the Leyu corpus.

Recomputes the figures in the split-diagnostics appendix from leyu_meta_raw.parquet
(normalized text per utterance) and the Joint cache:
  - the bipartite speaker-text graph: components, shares, speaker and dialect
    composition of the largest component, singleton components;
  - the within-dialect filter versus the global Joint filter, including the count of
    survivors whose prompt occurs in another dialect's held-out partitions;
  - prompts shared by validation and test, and the share of test utterances whose
    prompt occurs in validation.

Writes splits/split_diagnostics.json. Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import json
from collections import Counter

import pandas as pd
from datasets import load_from_disk

OUT = cfg.SPLITS / "split_diagnostics.json"


def parse_meta(speaker):
    """Metadata speakers are '<dialect>_<id>'; return the cache-style key '<dialect>::<id>'."""
    dialect, sid = str(speaker).rsplit("_", 1)
    return f"{dialect.replace('_', ' ')}::{sid}"


def components(keys, texts):
    """Connected components of the bipartite speaker-text graph (union-find)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, t in zip(keys, texts):
        a, b = find(("S", k)), find(("T", t))
        if a != b:
            parent[a] = b
    return [find(("S", k)) for k in keys]


def graph_stats(m):
    m = m.copy()
    m["comp"] = components(m["key"].tolist(), m["text"].tolist())
    sizes = Counter(m["comp"])
    big, big_n = sizes.most_common(1)[0]
    n = len(m)
    largest = m[m["comp"] == big]
    rest = m[m["comp"] != big]
    dialects_per_rest = rest.groupby("comp")["dialect"].nunique()
    return {
        "utterances": n,
        "components": len(sizes),
        "largest_share_pct": round(100 * big_n / n, 1),
        "top_shares_pct": [round(100 * c / n, 1) for _, c in sizes.most_common(6)],
        "largest_speakers": int(largest["key"].nunique()),
        "largest_utterances_by_dialect": {d: int(c) for d, c in largest["dialect"].value_counts().items()},
        "speakers_by_dialect_total": {d: int(c) for d, c in m.groupby("dialect")["key"].nunique().items()},
        "speakers_by_dialect_in_largest": {d: int(c) for d, c in largest.groupby("dialect")["key"].nunique().items()},
        "singleton_components": sum(1 for c in sizes.values() if c == 1),
        "non_largest_components_single_dialect": bool((dialects_per_rest <= 1).all()),
    }


def filter_stats(m):
    train = m[m["part"] == "train"]
    held = m[m["part"] != "train"]
    held_by_dialect = {d: set(g["text"]) for d, g in held.groupby("dialect")}
    all_held = set(held["text"])

    within_keep = [t not in held_by_dialect.get(d, set())
                   for d, t in zip(train["dialect"], train["text"])]
    within = train[within_keep]
    cross = sum(
        any(t in s for od, s in held_by_dialect.items() if od != d)
        for d, t in zip(within["dialect"], within["text"]))
    global_keep = sum(t not in all_held for t in train["text"])
    return {
        "si_train": len(train),
        "within_dialect_survivors": len(within),
        "survivors_with_prompt_in_other_dialect_heldout": int(cross),
        "cross_share_pct": round(100 * cross / len(within), 2),
        "global_joint_train": int(global_keep),
    }


def val_test_stats(cache):
    val = [t.strip() for t in cache["validation"]["norm_text"]]
    test = [t.strip() for t in cache["test"]["norm_text"]]
    shared = set(val) & set(test)
    return {
        "val_test_shared_texts": len(shared),
        "test_utt_with_text_in_val_pct": round(100 * sum(t in set(val) for t in test) / len(test), 1),
    }


def main():
    cache = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    meta = pd.read_parquet(cfg.require(cfg.DATA_DIR / "leyu_meta_raw.parquet"))
    meta["key"] = meta["speaker"].map(parse_meta)
    meta["dialect"] = meta["key"].str.split("::").str[0]
    meta["text"] = meta["text"].astype(str).str.strip()

    val = set(cache["validation"]["speaker_id"])
    test = set(cache["test"]["speaker_id"])
    meta["part"] = "train"
    meta.loc[meta["key"].isin(val), "part"] = "validation"
    meta.loc[meta["key"].isin(test), "part"] = "test"

    report = {
        "partitions": {k: int(v) for k, v in meta["part"].value_counts().items()},
        "graph": graph_stats(meta),
        "filters": filter_stats(meta),
        "val_test": val_test_stats(cache),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()