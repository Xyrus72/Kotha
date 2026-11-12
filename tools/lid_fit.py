"""Cross-validated logistic rule on Whisper + VoxLingua + decode-fit signals (.cache/lid/signals.json)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf8")
S = list(json.loads((ROOT / ".cache/lid/signals.json").read_text(encoding="utf8")).values())


def L(p):
    return math.log(max(p, 1e-6))


def feats(s):
    p = s["probs"]
    en, bn = p.get("en", 0), p.get("bn", 0) + p.get("as", 0)
    near = sum(p.get(k, 0) for k in ("hi", "ur", "ne", "mr", "sa", "gu", "pa", "or", "si"))
    other = max(1e-6, 1 - en - bn - near)
    return {"wh_en_bn": L(en) - L(bn), "wh_en_near": L(en) - L(near), "wh_en_other": L(en) - L(other),
            "vl": max(-30, min(30, s["vl_en"] - s["vl_bn"])), "en_lp": s["en_lp"], "bn_lp": s["bn_lp"],
            "log_s": math.log(s["seconds"])}


NAMES = list(feats(S[0]))
X = np.array([[feats(s)[k] for k in NAMES] for s in S])
y = np.array([1.0 if s["truth"] == "en" else 0.0 for s in S])
G = np.array([s["set"] + "/" + s["variant"] for s in S])


def fit(Xt, yt, l2=0.01):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    Z = (Xt - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    cw = np.where(yt == 1, 0.5 / yt.mean(), 0.5 / (1 - yt.mean()))
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - yt) * cw
        w -= 0.5 * (Z.T @ g / len(yt) + l2 * w)
        b -= 0.5 * g.mean()
    return mu, sd, w, b


def predict(model, Xs):
    mu, sd, w, b = model
    return ((Xs - mu) / sd) @ w + b


def cv(cols, reps=5, K=5):
    Xc = X[:, cols]
    total = np.zeros(len(S))
    for r in range(reps):
        idx = np.random.default_rng(r).permutation(len(S))
        for k in range(K):
            te = idx[k::K]
            tr = np.setdiff1d(idx, te)
            m = fit(Xc[tr], y[tr])
            total[te] += (predict(m, Xc[te]) >= 0) != (y[te] == 1)
    return total / reps


for label, cols in [
    ("whisper only", ["wh_en_bn", "wh_en_near", "wh_en_other"]),
    ("whisper + en_lp", ["wh_en_bn", "wh_en_near", "wh_en_other", "en_lp"]),
    ("whisper + voxlingua", ["wh_en_bn", "wh_en_near", "wh_en_other", "vl"]),
    ("whisper + voxlingua + en_lp", ["wh_en_bn", "wh_en_near", "wh_en_other", "vl", "en_lp"]),
    ("all", NAMES),
]:
    e = cv([NAMES.index(c) for c in cols])
    per = {g: round(float(e[G == g].sum()), 1) for g in sorted(set(G)) if e[G == g].sum() > 0}
    print(f"{label:30} avg errors {e.sum():5.1f} of {len(S)}  {per}")

if len(sys.argv) > 1:                          # print the final model trained on everything
    cols = sys.argv[1].split(",")
    m = fit(X[:, [NAMES.index(c) for c in cols]], y)
    print(json.dumps({"features": cols, "mu": m[0].round(4).tolist(), "sd": m[1].round(4).tolist(),
                      "w": m[2].round(4).tolist(), "b": round(float(m[3]), 4)}))
