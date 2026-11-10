import json, collections
from datasets import load_from_disk

te = load_from_disk("phase2_leyu/dataset_cache_joint_independent")["test"]
rows = [json.loads(l) for l in open("test_set_audio_export/manifest.jsonl", encoding="utf-8")]
print("cache columns:", te.column_names)

cache = [(te["dialect"][i], te["text"][i].strip(), te["speaker_id"][i]) for i in range(len(te))]
man = [(r["dialect"], r["reference"].strip()) for r in rows]

# 1. Same order within each dialect?
for d in sorted({x[0] for x in man}):
    c = [x for x in cache if x[0] == d]
    m = [x for x in man if x[0] == d]
    same = sum(a[1] == b[1] for a, b in zip(c, m))
    print(f"{d!r}: cache {len(c)}, manifest {len(m)}, same text in order {same}")

# 2. Text join: how many manifest texts are found, and are they unique?
cnt = collections.Counter(t for _, t, _ in cache)
found = [t for _, t in man if t in cnt]
print("manifest texts found in cache:", len(found), "of", len(man))
print("manifest texts unique in cache:", sum(cnt[t] == 1 for t in found))