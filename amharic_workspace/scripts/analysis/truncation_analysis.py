"""Truncation and reference-coverage analysis of the full test set.

Uses the bf16 per-sample predictions of C4b. Four parts:
  1. How much of each recording the 30 s window and the 448-token reference cover.
  2. Error breakdown for four groups of audio length and reference completeness.
  3. Reference-coverage groups by reference words per second (the Limitations bullet).
  4. Silence in each coverage group, which shows whether insertions are speech.

Token counts use the same tokenizer call as build_subset.py. The run stops
unless the subset of recordings that fit both limits has exactly 688 members.
Pass --examples to print the highest-insertion and most-deleted utterances.
Run from the workspace root after sourcing env.amh.sh.
"""
from common import config as cfg  # must load before transformers/datasets

import argparse
import collections
import io

import jiwer
import numpy as np
import soundfile as sf
from datasets import Audio, load_from_disk
from transformers import WhisperTokenizer

from common.bootstrap_utils import cache_ok, load

PERSAMPLE = cfg.PERSAMPLE / "c4b_bfloat16_persample.jsonl"
WINDOW_S = 30
TOKEN_LIMIT = cfg.MAX_LABEL_LENGTH
SUBSET_SIZE = 688
RATE_LOW, RATE_HIGH = 0.5, 2.5


def audio_source(audio):
    return io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]


def load_test():
    test = load_from_disk(str(cfg.require(cfg.LEYU_JOINT_CACHE)))["test"]
    audio = list(test.cast_column("audio", Audio(decode=False))["audio"])
    seconds = np.array([sf.info(audio_source(a)).duration for a in audio])
    tok = WhisperTokenizer.from_pretrained(cfg.WHISPER_MODEL, language="amharic", task="transcribe")
    tokens = np.array([len(tok(t).input_ids) for t in test["norm_text"]])
    words = np.array([len(t.split()) for t in test["norm_text"]])
    return test, audio, seconds, tokens, words


def paired_predictions(test):
    preds = load(PERSAMPLE)
    if sorted(preds) != list(range(len(test))):
        raise SystemExit(f"{PERSAMPLE.name} has {len(preds)} rows; expected {len(test)}")
    norm = test.select_columns(["norm_text"])["norm_text"]
    bad = [i for i in preds if not cache_ok({"norm_text": norm[i]}, preds[i]["reference"])]
    if bad:
        raise SystemExit(f"{len(bad)} references are not prefixes of the cached text, first idx {bad[0]}")
    return preds


def error_counts(ref, hyp):
    """Substitutions, deletions, insertions, reference words. Same convention as word_errors."""
    ref_w, hyp_w = ref.split(), hyp.split()
    n = len(ref_w)
    if not hyp_w:
        return 0, n, 0, n
    o = jiwer.process_words(" ".join(ref_w), " ".join(hyp_w))
    return o.substitutions, o.deletions, o.insertions, n


def max_trigram_repeat(text):
    w = text.split()
    counts = collections.Counter(tuple(w[k:k + 3]) for k in range(len(w) - 2))
    return max(counts.values(), default=0)


def aggregate(preds, idx):
    S = D = I = N = loops = 0
    for i in idx:
        s, d, ins, n = error_counts(preds[i]["reference"], preds[i]["prediction"])
        S, D, I, N = S + s, D + d, I + ins, N + n
        loops += max_trigram_repeat(preds[i]["prediction"]) >= 4
    E = max(S + D + I, 1)
    return {"n": len(idx), "wer": 100 * (S + D + I) / max(N, 1),
            "ins": 100 * I / E, "del": 100 * D / E, "loops": int(loops)}


def report(label, s):
    print(f"{label:38s} n={s['n']:5d}  WER {s['wer']:5.1f}  insertions {s['ins']:4.1f}%  "
          f"deletions {s['del']:4.1f}%  loops {s['loops']}")


def silent_share(audio, frame_s=0.025, floor_db=-40):
    """Share of 25 ms frames more than 40 dB below the loudest frame."""
    x, sr = sf.read(audio_source(audio))
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = int(frame_s * sr)
    frames = x[: len(x) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-10
    return float((20 * np.log10(rms / rms.max()) < floor_db).mean())


def part1_coverage(seconds, tokens):
    print("1. Coverage of the 30 s window and the 448-token reference")
    within = (seconds <= WINDOW_S) & (tokens <= TOKEN_LIMIT)
    print(f"   utterances within both limits: {within.sum()} (subset file: {SUBSET_SIZE})")
    if int(within.sum()) != SUBSET_SIZE:
        raise SystemExit("subset does not reproduce; check the token rule against build_subset.py")
    print(f"   share of test audio inside the window: {np.minimum(seconds, WINDOW_S).sum() / seconds.sum():.3f}")
    print(f"   references longer than 448 tokens: {(tokens > TOKEN_LIMIT).mean():.3f}")
    # Approximation: assumes speech is spread evenly over the reference tokens.
    covered = seconds * np.minimum(1, TOKEN_LIMIT / tokens)
    print(f"   audio covered by the evaluated reference prefix (approx.): "
          f"{np.minimum(covered, WINDOW_S).sum() / seconds.sum():.3f}")
    print(f"   reference prefix extends past the window (approx.): {(covered > WINDOW_S).mean():.3f}")


def part2_groups(preds, seconds, tokens):
    print("\n2. Audio length and reference completeness")
    groups = {
        "A audio <=30 s, reference complete": (seconds <= WINDOW_S) & (tokens <= TOKEN_LIMIT),
        "B audio >30 s, reference complete": (seconds > WINDOW_S) & (tokens <= TOKEN_LIMIT),
        "C audio >30 s, reference truncated": (seconds > WINDOW_S) & (tokens > TOKEN_LIMIT),
        "D audio <=30 s, reference truncated": (seconds <= WINDOW_S) & (tokens > TOKEN_LIMIT),
    }
    for label, mask in groups.items():
        idx = np.where(mask)[0]
        s = aggregate(preds, idx)
        heard = np.mean(np.minimum(WINDOW_S, seconds[idx]) / seconds[idx]) * 100
        report(label, s)
        print(f"{'':38s} audio heard by the model: {heard:.0f}%")


def part3_rate(preds, seconds, words, silence):
    print("\n3. Reference coverage by reference words per second")
    rate = words / seconds
    everything = np.arange(len(seconds))
    low = everything[rate < RATE_LOW]
    high = everything[rate > RATE_HIGH]
    typical = everything[(rate >= RATE_LOW) & (rate <= RATE_HIGH)]
    rows = {
        "all": everything,
        f"rate <{RATE_LOW} (reference far too short)": low,
        f"rate >{RATE_HIGH} (very fast)": high,
        f"{RATE_LOW} to {RATE_HIGH} (typical)": typical,
        f"all minus rate <{RATE_LOW}": np.concatenate([typical, high]),
    }
    for label, idx in rows.items():
        report(label, aggregate(preds, idx))
        if len(idx):
            print(f"{'':38s} silent frames: median {100 * np.median(silence[idx]):.0f}%, "
                  f"mean {100 * silence[idx].mean():.0f}%")


def show_examples(preds, seconds, idx, key, k=3):
    rows = []
    for i in idx:
        s, d, ins, n = error_counts(preds[i]["reference"], preds[i]["prediction"])
        rows.append({"idx": i, "ins": ins, "del": d, "sec": seconds[i]})
    rows.sort(key=lambda r: -r[key])
    for r in rows[:k]:
        p = preds[r["idx"]]
        print(f"idx {r['idx']} | {r['sec']:.1f} s | insertions {r['ins']} | deletions {r['del']}")
        print("  REF:", p["reference"][:160])
        print("  HYP:", p["prediction"][:160])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", action="store_true", help="print example utterances")
    args = parser.parse_args()

    test, audio, seconds, tokens, words = load_test()
    preds = paired_predictions(test)
    silence = np.array([silent_share(a) for a in audio])

    part1_coverage(seconds, tokens)
    part2_groups(preds, seconds, tokens)
    part3_rate(preds, seconds, words, silence)

    if args.examples:
        print("\n4. Examples")
        group_b = np.where((seconds > WINDOW_S) & (tokens <= TOKEN_LIMIT))[0]
        group_d = np.where((seconds <= WINDOW_S) & (tokens > TOKEN_LIMIT))[0]
        print("Audio >30 s, reference complete (most insertions):")
        show_examples(preds, seconds, group_b, "ins")
        print("Audio <=30 s, reference truncated (most deletions):")
        show_examples(preds, seconds, group_d, "del")


if __name__ == "__main__":
    main()