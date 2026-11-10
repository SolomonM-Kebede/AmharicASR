from datasets import load_from_disk
from transformers import WhisperTokenizer
tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
std = load_from_disk("phase2b_leyu_standard/standard_speech_cache_ratio1.0")
for i in (0, 1, 2, 3, 4, 5000, 10000):
    print(i, tok.decode([t for t in std[i]["labels"] if t >= 0], skip_special_tokens=True)[:80])
    
import ast, re, unicodedata, collections
from datasets import load_from_disk
from transformers import WhisperTokenizer

tree = ast.parse(open("train_phase2_leyu.py", encoding="utf8").read())
MAP = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", None) == "AMHARIC_CHAR_MAP" for t in n.targets))

ds = load_from_disk("phase2_leyu/dataset_cache_joint_independent")
leyu = collections.Counter()
for sp in ds.values():
    for s in sp.select_columns(["norm_text"])["norm_text"]:
        leyu.update(s)

# tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
# std = load_from_disk("phase2b_leyu_standard/standard_speech_cache_ratio1.0")
stdc = collections.Counter()
for x in std.select_columns(["labels"])["labels"]:
    stdc.update(tok.decode([i for i in x if i >= 0], skip_special_tokens=True).strip())

ok = lambda ch: 0x1200 <= ord(ch) <= 0x135A or ch == " " or (ch.isascii() and ch.isalnum())
for name, c in (("Leyu", leyu), ("standard", stdc)):
    print(f"== {name}: {sum(c.values())} chars, {len(c)} distinct")
    print("   variant letters still present:", sum(c[ch] for ch in MAP))      # expect 0
    odd = sorted(((ch, hex(ord(ch)), unicodedata.name(ch, "?"), n) for ch, n in c.items() if not ok(ch)),
                 key=lambda r: -r[3])
    for r in odd[:25]:
        print("  ", r)
    lab = {ch: n for ch, n in c.items() if re.search(r"SYLLABLE (HHW|SZW|XW)", unicodedata.name(ch, ""))}
    print("   labialized variants (HHW/SZW/XW):", lab)