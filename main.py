# polymarket-faa-attribution-02 entry: EWMA-of-absolute-moves volatility forecast.
# Model: per-market sigma estimated from the case's own last 120 one-minute absolute price
# moves (EWMA), scaled down 0.32 to account for the 0.005 price-grid quantization inflating
# small moves; floored at 0.0005 and capped at 0.05 per sqrt-minute. Quantiles are a Gaussian
# random walk around the checkpoint price. Deterministic: pure function of the case.

import json, math, os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
LAM = 0.94
SCALE = 0.32
SIGMA_FLOOR = 0.0005
SIGMA_CAP = 0.05

def clamp6(x):
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0

def sigma_of(prices):
    diffs = [abs(b - a) for a, b in zip(prices, prices[1:])][-120:]
    if not diffs:
        return SIGMA_FLOOR
    var = 0.0
    w = 0.0
    for d in diffs:
        var = LAM * var + (1 - LAM) * d * d
        w = LAM * w + (1 - LAM)
    if w <= 0:
        return SIGMA_FLOOR
    return max(SIGMA_FLOOR, min(SIGMA_CAP, math.sqrt(var / w) * SCALE))

def forecast(m, horizon):
    prices = m["prices"]
    x = prices[-1]
    s = sigma_of(prices)
    return [[clamp6(x + z * s * math.sqrt(h)) for z in Z] for h in range(1, horizon + 1)]

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(INPUT_DIR).glob("*.in")):
        cid = f.name[:-3]
        case = json.loads(f.read_text())
        if case["quantiles"] != QUANTILES:
            raise SystemExit(f"quantile mismatch in {cid}")
        forecasts = [{"id": m["id"], "q": forecast(m, case["horizon_minutes"])} for m in case["markets"]]
        (out / f"{cid}.out").write_text(json.dumps({"forecasts": forecasts}))

main()
