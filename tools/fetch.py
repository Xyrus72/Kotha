"""Parallel, resumable HTTP download:  fetch.py URL OUTFILE [connections]

Splits the file into 8 MB blocks fetched over several connections; finished blocks are recorded in
OUTFILE.blocks so an interrupted download continues where it stopped.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

BLOCK = 8 * 1024 * 1024


def main() -> int:
    url, out = sys.argv[1], Path(sys.argv[2])
    conns = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
        final = r.geturl()
    state = Path(str(out) + ".blocks")
    done = set(json.loads(state.read_text())) if state.exists() and out.exists() else set()
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.stat().st_size != total:
        with open(out, "wb") as f:
            f.truncate(total)
        done = set()
    todo = [i for i in range((total + BLOCK - 1) // BLOCK) if i not in done]
    lock = threading.Lock()
    f = open(out, "r+b")

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i = todo.pop(0)
            a, b = i * BLOCK, min(total, (i + 1) * BLOCK) - 1
            for attempt in range(50):
                try:
                    rq = urllib.request.Request(final, headers={"Range": f"bytes={a}-{b}", "User-Agent": "curl/8"})
                    with urllib.request.urlopen(rq, timeout=60) as r:
                        data = r.read()
                    if len(data) != b - a + 1:
                        raise IOError("short read")
                    with lock:
                        f.seek(a)
                        f.write(data)
                        done.add(i)
                        state.write_text(json.dumps(sorted(done)))
                    break
                except Exception:                     # noqa: BLE001
                    time.sleep(2)
            else:
                with lock:
                    todo.append(i)
                return

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(conns)]
    for t in threads:
        t.start()
    blocks = (total + BLOCK - 1) // BLOCK
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        print(f"{len(done) * BLOCK / 1e6:.0f} / {total / 1e6:.0f} MB", flush=True)
    f.close()
    if len(done) == blocks:
        state.unlink(missing_ok=True)
        print("complete", flush=True)
        return 0
    print("incomplete", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
