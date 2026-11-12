"""Character check for the shared normalization (data.normalize).

Counts characters in the normalized Leyu text (all splits of the Joint cache) and in the
normalized Standard-Amharic training labels, and reports:
  - variant letters still present after normalization (expected 0 in both)
  - the characters outside the Ethiopic syllable block and ASCII letters and digits,
    most frequent first
  - labialized syllable variants (HHW, SZW, XW)

Writes splits/normalization_check.json. Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import collections
import json
import re
import unicodedata

from datasets import load_from_disk
from transformers import WhisperTokenizer

from data.normalize import AMHARIC_CHAR_MAP, normalize_amharic

ETHIOPIC_SYLLABLES = (0x1200, 0x135A)
LABIALIZED = re.compile(r"SYLLABLE (HHW|SZW|XW)")
TOP_ODD = 25
OUT = cfg.SPLITS / "normalization_check.json"


def is_expected(ch):
    lo, hi = ETHIOPIC_SYLLABLES
    return lo <= ord(ch) <= hi or ch == " " or (ch.isascii() and ch.isalnum())


def summarize(name, counter):
    variants = sum(counter[ch] for ch in AMHARIC_CHAR_MAP)
    odd = sorted(((ch, hex(ord(ch)), unicodedata.name(ch, "?"), n)
                  for ch, n in counter.items() if not is_expected(ch)),
                 key=lambda r: -r[3])
    labialized = {ch: n for ch, n in counter.items()
                  if LABIALIZED.search(unicodedata.name(ch, ""))}
    print(f"== {name}: {sum(counter.values()):,} characters, {len(counter)} distinct")
    print(f"   variant letters still present: {variants}")
    for row in odd[:TOP_ODD]:
        print("  ", row)
    print(f"   labialized variants: {labialized}")
    return {"characters": sum(counter.values()), "distinct": len(counter),
            "variant_letters_present": variants,
            "unexpected_top": [{"char": c, "codepoint": h, "name": nm, "count": n}
                               for c, h, nm, n in odd[:TOP_ODD]],
            "labialized": labialized}


def leyu_counter():
    ds = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))
    counter = collections.Counter()
    for split in ds.values():
        for s in split["norm_text"]:
            counter.update(s)
    return counter


def standard_counter():
    tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL, language="amharic", task="transcribe")
    std = load_from_disk(str(cfg.require(cfg.JOINT_STANDARD_CACHE)))
    counter = collections.Counter()
    for labels in std.select_columns(["labels"])["labels"]:
        counter.update(tok.decode([i for i in labels if i >= 0], skip_special_tokens=True).strip())
    return counter


def main():
    report = {"leyu_joint_test_and_train": summarize("Leyu", leyu_counter()),
              "standard_training_labels": summarize("Standard", standard_counter())}
    # Sanity check: the normalizer leaves no variant letter in its own output.
    sample = "ሐ ዐ ፀ ሠ"
    if any(ch in normalize_amharic(sample) for ch in AMHARIC_CHAR_MAP):
        raise SystemExit("normalize_amharic leaves a variant letter in its output")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()