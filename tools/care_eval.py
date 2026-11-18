"""Play the customer-care test conversations (business/scenarios.json) against the reply model and save the
transcripts, so different prompts/models can be compared side by side.

    .venv\\Scripts\\python.exe tools\\care_eval.py baseline                  current assistant prompt
    .venv\\Scripts\\python.exe tools\\care_eval.py care --model qwen3.5:9b    customer-care mode, another model

Transcripts: logs/care/<name>.json  (judge them with an LLM-as-judge or by reading).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--mode", choices=("assistant", "care"), default=None,
                    help="default: 'assistant' for the name 'baseline', else 'care'")
    ap.add_argument("--model")
    ap.add_argument("--only", help="comma-separated scenario ids")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf8")

    from voice import config
    from voice.llm import Chat
    cfg = config.load()
    if args.model:
        cfg["llm_model"] = args.model
    mode = args.mode or ("assistant" if args.name == "baseline" else "care")
    if mode == "care":
        cfg["mode"] = "care"
    else:
        cfg["mode"] = "assistant"

    scenarios = json.loads((ROOT / "business" / "scenarios.json").read_text(encoding="utf8"))
    if args.only:
        keep = set(args.only.split(","))
        scenarios = [s for s in scenarios if s["id"] in keep]
    out = []
    for sc in scenarios:
        chat = Chat.from_config(cfg)
        turns = []
        for user in sc["turns"]:
            t0 = time.perf_counter()
            first = None
            parts = []
            for piece in chat.reply(user, sc["lang"]):
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(piece)
            turns.append({"user": user, "agent": "".join(parts).strip(),
                          "first_token_s": round(first or 0, 2), "total_s": round(time.perf_counter() - t0, 2)})
        out.append({**sc, "transcript": turns})
        print(f"[{sc['id']}]")
        for t in turns:
            print(f"  C: {t['user']}\n  A: {t['agent']}\n     ({t['first_token_s']} s to first word, {t['total_s']} s total)")
    path = ROOT / "logs" / "care" / f"{args.name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": cfg["llm_model"], "mode": mode, "scenarios": out},
                               ensure_ascii=False, indent=1), encoding="utf8")
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
