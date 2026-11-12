"""Download a Hugging Face Whisper fine-tune and convert it to CTranslate2 (faster-whisper) format.

Run with the conversion venv (needs torch + transformers + ctranslate2), not the app venv:

    D:\\arc\\venv-convert\\Scripts\\python.exe tools\\convert_model.py ^
        bengaliAI/tugstugi_bengaliai-asr_whisper-medium models\\bn-tugstugi-medium

The base tokenizer is used when the fine-tune ships no tokenizer.json (faster-whisper needs it).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ctranslate2.dll needs the VC++ runtime; the msvc-runtime wheel drops it next to python.exe.
for d in {Path(sys.prefix), Path(sys.prefix) / "Scripts", Path(sys.executable).parent}:
    if d.is_dir():
        os.add_dll_directory(str(d))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    repo, out = sys.argv[1], Path(sys.argv[2])
    base = sys.argv[3] if len(sys.argv) > 3 else "openai/whisper-medium"
    if not out.is_absolute():
        out = ROOT / out

    from huggingface_hub import hf_hub_download, snapshot_download

    if Path(repo).is_dir():                       # an already-downloaded folder
        src = Path(repo)
    else:
        print(f"downloading {repo} ...", flush=True)
        src = Path(snapshot_download(repo, allow_patterns=[
            "*.json", "*.txt", "*.safetensors", "pytorch_model.bin", "*.model"]))
    names = sorted(p.name for p in src.iterdir())
    print("files:", names, flush=True)

    import ctranslate2

    copy = [n for n in ("tokenizer.json", "preprocessor_config.json") if (src / n).exists()]
    print(f"converting -> {out} (float16), copying {copy}", flush=True)
    conv = ctranslate2.converters.TransformersConverter(
        str(src), copy_files=copy, low_cpu_mem_usage=True)
    conv.convert(str(out), quantization="float16", force=True)

    for name in ("tokenizer.json", "preprocessor_config.json"):
        if not (out / name).exists():
            print(f"{name} missing in the fine-tune - taking it from {base}", flush=True)
            shutil.copy(hf_hub_download(base, name), out / name)

    print("done:", sorted(p.name for p in out.iterdir()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
