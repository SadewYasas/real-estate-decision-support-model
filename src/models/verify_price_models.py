"""Reproducibility check for the saved sale and rent models (without the ~3 h retraining).

Rebuilds the grouped 80/20 split exactly as train_price_models.py does, from the cleaned
data now in data/processed/, loads the saved models, recomputes the test metrics and checks
they equal those stored in {sale,rent}_models.json. Equality shows that (a) the cleaning is
deterministic, (b) the split is reproduced row for row, and (c) the saved models are the
ones the reported metrics came from.

Output: artefacts/metrics/price_models_reproducibility.json

Run:  python -m src.models.verify_price_models
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from src.models.train_price_models import DATASETS, test_metrics

TOL = 1e-9


def check(name: str) -> dict:
    spec = DATASETS[name]
    stored = json.loads((METRICS_DIR / f"{name}_models.json").read_text())
    df = pd.read_csv(PROCESSED_DIR / spec["file"])
    features = spec["categorical"] + spec["numeric"]
    groups = df.groupby(spec["group_cols"]).ngroup()
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                 .split(df, groups=groups))
    X, y = df[features].iloc[te], np.log(df[spec["target"]].iloc[te])
    out = {"rows_total": len(df), "rows_test": len(te),
           "rows_match_stored": len(df) == stored["rows"]["total"] and len(te) == stored["rows"]["test"]}
    model = joblib.load(MODELS_DIR / f"{name}_model.joblib")
    now = test_metrics(y, model.predict(X))
    ref = stored["candidates"][stored["selected_model"]]["test"]
    out["selected_model"] = stored["selected_model"]
    out["test_metrics_recomputed"] = now
    out["max_abs_difference"] = max(abs(now[k] - ref[k]) / max(abs(ref[k]), 1) for k in ref)
    if stored.get("ablation"):
        ab = stored["ablation"]
        m = joblib.load(MODELS_DIR / f"{name}_model_{ab['name']}.joblib")
        now_ab = test_metrics(y, m.predict(X[ab["features"]]))
        out["ablation_max_abs_difference"] = max(abs(now_ab[k] - ab["test"][k]) / max(abs(ab["test"][k]), 1)
                                                for k in ab["test"])
    out["reproduced"] = bool(out["rows_match_stored"] and out["max_abs_difference"] < TOL
                             and out.get("ablation_max_abs_difference", 0) < TOL)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    res = {name: check(name) for name in ("sale", "rent")}
    res["note"] = ("Saved models re-scored on the test split rebuilt from freshly cleaned data; "
                   "full retraining: python -m src.models.train_price_models --dataset <name> --fresh")
    (METRICS_DIR / "price_models_reproducibility.json").write_text(json.dumps(res, indent=2))
    for name in ("sale", "rent"):
        r = res[name]
        print(f"{name}: {r['rows_test']:,} test rows, {r['selected_model']}, R2 log "
              f"{r['test_metrics_recomputed']['r2_log']:.4f}, max relative difference "
              f"{r['max_abs_difference']:.1e} -> {'REPRODUCED' if r['reproduced'] else 'MISMATCH'}")
    if not all(res[n]["reproduced"] for n in ("sale", "rent")):
        sys.exit(1)


if __name__ == "__main__":
    main()
