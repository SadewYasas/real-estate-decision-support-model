"""API response time for the full analysis (POST /api/analyse) -> artefacts/metrics/performance.json.

100 requests with varied properties (random ZIPs from the lookup in the 29 states of the
sale data, random beds / baths / size; seed 42), measured two ways:
- in-process with the Flask test client (the server's own processing time);
- over HTTP to a local server (adds JSON transfer and the network stack on localhost).
5 warm-up requests are run first and not counted. Model loading (start-up) is reported
separately; it happens once, not per request.

Run:  python -m src.api.performance
"""
import json
import logging
import platform
import sys
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR, RANDOM_STATE

N, WARMUP = 100, 5


def requests_sample(n: int) -> list[dict]:
    rng = np.random.default_rng(RANDOM_STATE)
    zips = pd.read_csv(PROCESSED_DIR / "zip_lookup.csv", dtype={"zip": str})
    states = set(pd.read_csv(PROCESSED_DIR / "sale_clean.csv", usecols=["state_code"])["state_code"])
    zips = zips[zips["state"].isin(states) & (zips["population"] > 1000)]["zip"].to_numpy()
    out = []
    for _ in range(n):
        beds = int(rng.integers(1, 6))
        out.append({"zip": str(rng.choice(zips)), "beds": beds,
                    "baths": float(rng.choice([1, 1.5, 2, 2.5, 3])),
                    "sqft": int(rng.integers(600, 4000))})
    return out


def stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"n": len(a), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max()),
            "under_3s_share": float((a < 3000).mean())}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.perf_counter()
    from backend.app import app
    startup = time.perf_counter() - t0
    reqs = requests_sample(N + WARMUP)

    client = app.test_client()
    in_process, statuses = [], []
    for i, body in enumerate(reqs):
        t = time.perf_counter()
        r = client.post("/api/analyse", json=body)
        dt = (time.perf_counter() - t) * 1000
        statuses.append(r.status_code)
        if i >= WARMUP:
            in_process.append(dt)

    from werkzeug.serving import make_server
    logging.getLogger("werkzeug").setLevel(logging.ERROR)   # no per-request log lines
    server = make_server("127.0.0.1", 5055, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    over_http = []
    try:
        for i, body in enumerate(reqs):
            data = json.dumps(body).encode()
            req = urllib.request.Request("http://127.0.0.1:5055/api/analyse", data=data,
                                         headers={"Content-Type": "application/json"})
            t = time.perf_counter()
            with urllib.request.urlopen(req) as resp:
                resp.read()
            dt = (time.perf_counter() - t) * 1000
            if i >= WARMUP:
                over_http.append(dt)
    finally:
        server.shutdown()

    result = {
        "endpoint": "POST /api/analyse (price + rent + SHAP + forecasts + rent-vs-buy, 30-year table)",
        "requirement": "full analysis under 3 s",
        "requests": N, "warmup_excluded": WARMUP,
        "status_codes": {str(k): statuses.count(k) for k in sorted(set(statuses))},
        "startup_model_loading_s": round(startup, 3),
        "in_process": stats(in_process),
        "http_localhost": stats(over_http),
        "machine": {"python": platform.python_version(), "platform": platform.platform(),
                    "processor": platform.processor()},
        "server": "Flask development server (werkzeug), single process",
    }
    with open(METRICS_DIR / "performance.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("status_codes", "startup_model_loading_s",
                                             "in_process", "http_localhost")}, indent=2))


if __name__ == "__main__":
    main()
