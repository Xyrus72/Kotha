"""Compare language-decision rules on .cache/lid/signals.json (Whisper + VoxLingua107 signals)."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf8")
S = list(json.loads((ROOT / ".cache/lid/signals.json").read_text(encoding="utf8")).values())
FAM = ("bn", "as", "hi", "ne", "mr", "sa", "ur", "gu", "pa", "or")


def L(p):
    return math.log(max(p, 1e-9))


def wl(s):
    p = s["probs"]
    return L(p.get("en", 0)) - L(sum(p.get(k, 0) for k in FAM))


def vl(s):
    return s["vl_en"] - s["vl_bn"]


rules = {"whisper folded": lambda s: wl(s) >= 0, "voxlingua en-vs-bn": lambda s: vl(s) >= 0}
for a in (0.25, 0.5, 1.0):
    for k in (-2, -1, 0, 1, 2):
        rules[f"vl + {a}*whisper {k:+d}"] = (lambda a, k: lambda s: vl(s) + a * wl(s) + k >= 0)(a, k)

groups = sorted({(s["set"], s["variant"]) for s in S})
print(f"{'rule':28}" + "".join(f"{g[0]}/{g[1]}".rjust(14) for g in groups) + "  total")
for name, r in rules.items():
    row, tot = "", 0
    for g in groups:
        items = [s for s in S if (s["set"], s["variant"]) == g]
        w = sum(r(s) != (s["truth"] == "en") for s in items)
        tot += w
        row += f"{w}/{len(items)}".rjust(14)
    print(f"{name:28}{row}  {tot}")
if len(sys.argv) > 1:
    r = rules[sys.argv[1]]
    for s in S:
        if r(s) != (s["truth"] == "en"):
            print(s["set"], s["variant"], round(s["seconds"], 1), "vl", round(vl(s), 1), "wh", round(wl(s), 1),
                  s["vl_top"], "|", s["en_text"][:50], "|", s["bn_text"][:40])
