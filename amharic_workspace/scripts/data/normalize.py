"""Shared Amharic text normalization, applied to references and hypotheses before scoring.

Used by every evaluation and bootstrap script, so that all reported WER and CER values
are computed on the same text. Two steps:

1. Merge the variant letters of the standard orthography onto their base letters.
   These are the h, glottal, s and ts series, which are pronounced alike in standard
   Amharic. Labialized and other variant forms are not in the map and are left as they are.
2. Replace the ASCII and Ethiopic punctuation in PUNCT with spaces, then collapse runs of
   whitespace. str.split() also treats U+00A0 (non-breaking space) as whitespace.

Changing this module changes every reported WER. After any edit, rerun the default
bootstrap (expect 0.89) and the normalization check, then copy the module to
app/normalize.py, since the app keeps its own copy.
"""

AMHARIC_CHAR_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ፀ": "ጸ", "ፁ": "ጹ", "ፂ": "ጺ", "ፃ": "ጻ", "ፄ": "ጼ", "ፅ": "ጽ", "ፆ": "ጾ",
}
PUNCT = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~።፣፤፥፦'


def normalize_amharic(text: str) -> str:
    """Return the normalized form of text. Expects a str; callers pass "" for missing text."""
    for src, tgt in AMHARIC_CHAR_MAP.items():
        text = text.replace(src, tgt)
    for p in PUNCT:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()