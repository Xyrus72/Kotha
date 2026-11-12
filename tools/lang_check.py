"""Run the app's real Transcriber in Auto mode over the language test clips and count wrong languages."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf8")

import numpy as np  # noqa: E402

from voice import config, runtime  # noqa: E402
from voice.stt import RATE, Transcriber, speech_chunks  # noqa: E402

runtime.setup()
from faster_whisper import decode_audio  # noqa: E402

signals = json.loads((ROOT / ".cache/lid/signals.json").read_text(encoding="utf8"))
paths = {"en_us": ROOT / ".cache/bench/en", "bn": ROOT / ".cache/bench/bn", "en_sa": ROOT / ".cache/lid/en_sa"}
stt = Transcriber(config.load())
stt.load()
print("VoxLingua:", "on" if stt.vox else "OFF")
wrong, times, per = [], [], {}
for key, s in signals.items():
    name, variant, stem = key.split("/")
    audio = decode_audio(str(paths[name] / f"{stem}.wav"), sampling_rate=RATE)
    speech = np.concatenate(speech_chunks(audio))
    clip = speech if variant == "full" else speech[: int(2.0 * RATE)]
    r = stt.transcribe(clip, "auto")
    times.append(r.seconds)
    g = f"{name}/{variant}"
    per.setdefault(g, [0, 0])
    per[g][1] += 1
    if r.language != s["truth"]:
        per[g][0] += 1
        wrong.append((g, r.text[:70]))
for g, (w, n) in sorted(per.items()):
    print(f"  {g:12} {w:3}/{n}")
print(f"wrong {len(wrong)} of {len(signals)};  median {np.median(times):.2f} s, p95 {np.percentile(times, 95):.2f} s per clip")
for g, t in wrong:
    print("   ", g, "|", t)
