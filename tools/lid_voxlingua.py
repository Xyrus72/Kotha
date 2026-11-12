"""Add VoxLingua107 (SpeechBrain ECAPA language-ID) scores to .cache/lid/signals.json, then compare rules."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SIGNALS = ROOT / ".cache" / "lid" / "signals.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf8")
    import numpy as np
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    from voice import runtime
    runtime.setup()
    from faster_whisper import decode_audio
    from voice.stt import RATE, speech_chunks

    torch.set_num_threads(8)
    clf = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                         savedir=str(ROOT / "models" / "lid-voxlingua107"),
                                         run_opts={"device": "cpu"})
    labels = clf.hparams.label_encoder
    names = [labels.decode_ndim(i) for i in range(len(labels))]
    en_i = [i for i, n in enumerate(names) if n.split(":")[0] == "en"][0]
    bn_i = [i for i, n in enumerate(names) if n.split(":")[0] == "bn"][0]
    print("labels", names[en_i], names[bn_i], len(names))

    signals = json.loads(SIGNALS.read_text(encoding="utf8"))
    paths = {"en_us": ROOT / ".cache/bench/en", "bn": ROOT / ".cache/bench/bn", "en_sa": ROOT / ".cache/lid/en_sa"}
    t_total, n = 0.0, 0
    for key, s in signals.items():
        name, variant, stem = key.split("/")
        audio = decode_audio(str(paths[name] / f"{stem}.wav"), sampling_rate=RATE)
        speech = np.concatenate(speech_chunks(audio))
        clip = speech if variant == "full" else speech[: int(2.0 * RATE)]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = clf.classify_batch(torch.from_numpy(clip[: 30 * RATE]).unsqueeze(0))
        t_total += time.perf_counter() - t0
        n += 1
        logp = out[0][0].numpy()                 # log-posteriors over 107 languages
        s["vl_en"] = float(logp[en_i])
        s["vl_bn"] = float(logp[bn_i])
        s["vl_top"] = names[int(logp.argmax())].split(":")[0]
    SIGNALS.write_text(json.dumps(signals, ensure_ascii=False), encoding="utf8")
    print(f"{n} clips, {1000 * t_total / n:.0f} ms per clip on the CPU")


if __name__ == "__main__":
    main()
