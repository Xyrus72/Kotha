"""Measure real accuracy (word / character error rate) on public test speech.

Downloads N clips of the FLEURS test split (read speech by many speakers, never used for training here) and
runs each given model over them with the app's own decoding settings.

    .venv\\Scripts\\python.exe tools\\bench.py --lang bn --n 100 tugstugi=models/bn-tugstugi-medium
    .venv\\Scripts\\python.exe tools\\bench.py --lang en --n 100 large-v3=models/whisper-large-v3
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice import config, runtime  # noqa: E402
from voice.stt import Transcriber, _clean, speech_chunks  # noqa: E402

FLEURS = {"bn": "bn_in", "en": "en_us"}
API = "https://datasets-server.huggingface.co/rows?dataset=google%2Ffleurs&config={cfg}&split=test&offset={off}&length={n}"


def fetch(lang: str, n: int) -> list[tuple[Path, str]]:
    """The first n test clips, read straight out of the dataset's parquet file over HTTP range requests
    (only the row groups that are needed are downloaded)."""
    folder = ROOT / ".cache" / "bench" / lang
    folder.mkdir(parents=True, exist_ok=True)
    index = folder / "index.json"
    items = json.loads(index.read_text(encoding="utf8")) if index.exists() else []
    if len(items) < n:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        path = f"datasets/google/fleurs@refs%2Fconvert%2Fparquet/{FLEURS[lang]}/test/0000.parquet"
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            items = []
            for g in range(pf.num_row_groups):
                table = pf.read_row_group(g, columns=["id", "audio", "transcription"]).to_pylist()
                for row in table:
                    wav = folder / f"{len(items):04d}.wav"
                    wav.write_bytes(row["audio"]["bytes"])
                    items.append([wav.name, row["transcription"]])
                print(f"  fetched {len(items)} clips", flush=True)
                if len(items) >= n:
                    break
        index.write_text(json.dumps(items, ensure_ascii=False), encoding="utf8")
    return [(folder / name, ref) for name, ref in items[:n]]


def norm(text: str, lang: str) -> str:
    text = _clean(unicodedata.normalize("NFC", text.lower()), lang)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)   # punctuation, symbols
    return re.sub(r"\s+", " ", text).strip()


def edits(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("bn", "en"), required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", type=int, default=5, help="print this many of the worst clips per model")
    ap.add_argument("--beam", type=int, help="beam size (default: the app setting)")
    ap.add_argument("--fair-numbers", action="store_true",
                    help="Bangla: write numbers in the reference as words, as the model says them")
    ap.add_argument("models", nargs="+", help="name=path")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf8")
    runtime.setup()
    from faster_whisper import WhisperModel, decode_audio

    clips = fetch(args.lang, args.n)
    audio = [decode_audio(str(p), sampling_rate=16000) for p, _ in clips]
    if args.fair_numbers and args.lang == "bn":
        from voice.bn_numbers import spell_numbers
        refs = [norm(spell_numbers(ref), args.lang) for _, ref in clips]
    else:
        refs = [norm(ref, args.lang) for _, ref in clips]
    cfg = config.load()
    if args.beam:
        cfg["beam_size"] = args.beam
    helper = Transcriber(cfg)
    report = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model = WhisperModel(config.model_path(path), device=cfg["device"], compute_type=cfg["compute_type"])
        helper._decode(model, speech_chunks(audio[0]), args.lang)                      # warm-up
        we = wn = ce = cn = exact = 0
        rows = []
        t0 = time.perf_counter()
        for a, ref in zip(audio, refs):
            hyp = norm(helper._decode(model, speech_chunks(a), args.lang)[0], args.lang)
            e = edits(ref.split(), hyp.split())
            we += e; wn += len(ref.split())
            ce += edits(list(ref), list(hyp)); cn += len(ref)
            exact += ref == hyp
            rows.append((e / max(1, len(ref.split())), ref, hyp))
        dt = time.perf_counter() - t0
        secs = sum(len(a) for a in audio) / 16000
        report[name] = {"wer": we / wn, "cer": ce / cn, "exact": exact, "n": len(refs), "rtf": dt / secs}
        print(f"\n== {name}: WER {100 * we / wn:.2f} %  CER {100 * ce / cn:.2f} %  "
              f"perfect clips {exact}/{len(refs)}  ({dt:.0f} s for {secs:.0f} s of audio)", flush=True)
        for w, ref, hyp in sorted(rows, reverse=True)[: args.show]:
            print(f"   [{100 * w:.0f} %] ref: {ref}\n           hyp: {hyp}")
        del model
    out = ROOT / "logs" / f"bench_{args.lang}.json"
    old = json.loads(out.read_text(encoding="utf8")) if out.exists() else {}
    old.update(report)
    out.write_text(json.dumps(old, indent=1), encoding="utf8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
