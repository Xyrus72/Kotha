"""Measure English-vs-Bangla language detection on real speech.

Test sets (cached under .cache/lid):
  en_us  FLEURS English test (US read speech)
  en_sa  Common Voice English, South Asian accent (DTU54DL/common-accent)
  bn     FLEURS Bangla test
Each clip is also cut to its first ~2 s of speech ("short") because short phrases are the hard case.

Stage 1 (--collect) runs the models once per clip and saves every signal to .cache/lid/signals.json:
  Whisper large-v3 language probabilities, and both forced decodes (large-v3 'en', Bangla model 'bn')
  with their texts and average log-probabilities.
Stage 2 (default) scores decision rules on those saved signals, so rules can be compared in seconds.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DIR = ROOT / ".cache" / "lid"
SIGNALS = DIR / "signals.json"


def fetch() -> dict[str, list[Path]]:
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    from voice import runtime
    runtime.setup()
    from faster_whisper import decode_audio
    sets: dict[str, list[Path]] = {}
    # FLEURS clips are already on disk from the accuracy benchmark
    for name, src, n in (("en_us", "en", 120), ("bn", "bn", 200)):
        files = sorted((ROOT / ".cache" / "bench" / src).glob("*.wav"))[:n]
        sets[name] = files
    out = DIR / "en_sa"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.wav"))) < 150:
        fs = HfFileSystem()
        f = "datasets/DTU54DL/common-accent@refs%2Fconvert%2Fparquet/default/test/0000.parquet"
        table = pq.ParquetFile(fs.open(f, "rb")).read(columns=["audio", "accent"]).to_pylist()
        k = 0
        for row in table:
            if not row["accent"].startswith("India and South Asia"):
                continue
            a = decode_audio(io.BytesIO(row["audio"]["bytes"]), sampling_rate=16000)
            sf.write(out / f"{k:04d}.wav", a, 16000)
            k += 1
            if k >= 150:
                break
    sets["en_sa"] = sorted(out.glob("*.wav"))[:150]
    return sets


def collect() -> None:
    import numpy as np
    from voice import config
    from voice.stt import RATE, Transcriber, speech_chunks
    from faster_whisper import decode_audio
    sets = fetch()
    cfg = config.load()
    stt = Transcriber(cfg)
    stt.load()
    signals = {}
    for name, files in sets.items():
        for path in files:
            audio = decode_audio(str(path), sampling_rate=RATE)
            pieces = speech_chunks(audio)
            if not pieces:
                continue
            speech = np.concatenate(pieces)
            for variant, clip in (("full", speech), ("short", speech[: int(2.0 * RATE)])):
                if variant == "short" and len(speech) < 3.0 * RATE:
                    continue                      # already short: the full clip covers it
                _, _, probs = stt.en.detect_language(clip[: 30 * RATE], vad_filter=False,
                                                     language_detection_segments=1)
                en_text, en_lp = stt._decode(stt.en, [clip], "en")
                bn_text, bn_lp = stt._decode(stt.bn, [clip], "bn")
                signals[f"{name}/{variant}/{path.stem}"] = {
                    "set": name, "variant": variant, "truth": "bn" if name == "bn" else "en",
                    "seconds": len(clip) / RATE, "probs": dict(probs),
                    "en_text": en_text, "en_lp": en_lp, "bn_text": bn_text, "bn_lp": bn_lp}
            print(f"  {name}: {path.stem}", end="\r", flush=True)
        print(f"  {name}: done{' ' * 20}", flush=True)
    SIGNALS.write_text(json.dumps(signals, ensure_ascii=False), encoding="utf8")
    print("saved", len(signals), "signals")


BANGLA_FAMILY = ("bn", "as", "hi", "ne", "mr", "sa", "ur", "gu", "pa", "or")


def rules():
    def fold(s):
        p = s["probs"]
        return p.get("en", 0.0), sum(p.get(k, 0.0) for k in BANGLA_FAMILY)

    def current(s):                       # what the app does today (before the fix)
        en, bn = fold(s)
        if max(en, bn) < 4.0 * max(min(en, bn), 1e-6):
            return "en" if s["en_lp"] >= s["bn_lp"] else "bn"
        return "en" if en >= bn else "bn"

    def argmax_folded(s):
        en, bn = fold(s)
        return "en" if en >= bn else "bn"

    def bn_only(s):                       # only Bengali itself counts, no folding
        p = s["probs"]
        return "en" if p.get("en", 0) >= p.get("bn", 0) else "bn"

    def biased(k):
        def r(s):
            en, bn = fold(s)
            return "en" if math.log(max(en, 1e-9)) + k >= math.log(max(bn, 1e-9)) else "bn"
        r.__name__ = f"folded_bias_{k:+.1f}"
        return r

    def best_lp(s):
        return "en" if s["en_lp"] >= s["bn_lp"] else "bn"

    def script_check(s):
        """Probabilities first; for close calls, look at what the English decode produced."""
        en, bn = fold(s)
        return "en" if en >= bn else "bn"

    def combo(k, w):
        """Language-ID margin plus how much better each model's own decode fits."""
        def r(s):
            en, bn = fold(s)
            m = math.log(max(en, 1e-9)) - math.log(max(bn, 1e-9)) + k + w * (s["en_lp"] - s["bn_lp"])
            return "en" if m >= 0 else "bn"
        r.__name__ = f"combo_k{k:+.0f}_w{w:.0f}"
        return r

    return ([current, argmax_folded, bn_only, best_lp] + [biased(k) for k in (-1.0, 0.0, 1.0, 2.0, 3.0)]
            + [combo(k, w) for k in (0.0, 1.0, 2.0) for w in (2.0, 5.0, 10.0)])


def score() -> None:
    signals = json.loads(SIGNALS.read_text(encoding="utf8"))
    groups: dict[tuple, list] = {}
    for s in signals.values():
        groups.setdefault((s["set"], s["variant"]), []).append(s)
    keys = sorted(groups)
    print(f"{'rule':24}" + "".join(f"{a}/{b}:>14" .replace(':>14', '').rjust(14) for a, b in keys) + "   errors")
    for rule in rules():
        row, total = "", 0
        for k in keys:
            items = groups[k]
            wrong = sum(rule(s) != s["truth"] for s in items)
            total += wrong
            row += f"{wrong:>4}/{len(items):<4} ".rjust(14)
        print(f"{rule.__name__:24}{row}   {total}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf8")
    if args.collect:
        collect()
    score()
