"""Make the native libraries loadable on Windows before ctranslate2 / onnxruntime are imported.

ctranslate2.dll and onnxruntime need msvcp140.dll (the msvc-runtime wheel puts it inside the venv, where
Windows does not look), and ctranslate2 finds cublas64_12.dll / cudnn64_9.dll through PATH (the nvidia-*-cu12
wheels put them under site-packages/nvidia/*/bin).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_dll_dirs: list = []          # keep the add_dll_directory handles alive
_done = False


def setup() -> None:
    global _done
    if _done:
        return
    _done = True
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if sys.platform != "win32":
        return
    root = Path(sys.prefix)
    dirs = [d for d in (root, root / "Scripts") if (d / "msvcp140.dll").exists()]
    nvidia = root / "Lib" / "site-packages" / "nvidia"
    cuda = [d for d in (nvidia / "cublas" / "bin", nvidia / "cudnn" / "bin", nvidia / "cuda_nvrtc" / "bin")
            if d.is_dir()]
    if cuda:
        os.environ["PATH"] = os.pathsep.join([str(d) for d in cuda] + [os.environ.get("PATH", "")])
    for d in map(str, dirs + cuda):
        try:
            _dll_dirs.append(os.add_dll_directory(d))
        except OSError:
            pass
