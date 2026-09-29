"""Leakage audit: evidence for excluding PPSq and the D1 macro columns.

(a) house_train_proc.csv: test R² (log price) with and without PPSq (= price / area).
(b) Final Dataset: within-state correlation of each macro column with log price.
    A genuine state-level macro series varies over time, not across houses listed in the
    same state, so it should carry almost no house-level price information within a state.
(c) Final Dataset: test R² before and after shuffling the macro columns within each state.
    Shuffling within state keeps each state's macro distribution (so any real state-level
    signal survives) but breaks the link between a macro value and an individual house.

Models: RandomForest and XGBoost (both reported, so the conclusion does not depend on one
learner), 80/20 split, random_state=42, target log(price).

Run:  python -m src.pipeline.leakage_audit
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import METRICS_DIR, RANDOM_STATE, RAW_DIR, ROOT, ensure_dirs
from src.pipeline.clean_sale import KEEP_COLUMNS, MACRO_COLUMNS, RAW_FILE, clean_sale

OLD_FILE = ROOT / "backend" / "data" / "house_train_proc.csv"


MODELS = {
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    "xgboost": lambda: XGBRegressor(
        n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, n_jobs=-1, random_state=RANDOM_STATE),
}


def test_r2(X: pd.DataFrame, y: pd.Series) -> dict:
    """Test-set R² on log price for each audit model (same split for all)."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    out = {}
    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        out[name] = float(r2_score(y_te, model.predict(X_te)))
    return out


def audit_ppsq() -> dict:
    df = pd.read_csv(OLD_FILE)
    df = df[df["ListedPrice"] > 0]
    y = np.log(df["ListedPrice"])
    base = ["Bedroom", "Bathroom", "Area", "LotArea", "Latitude", "Longitude"]
    X = df[base].copy()
    X["State"] = df["State"].astype("category").cat.codes

    r2_without = test_r2(X, y)
    r2_with = test_r2(X.assign(PPSq=df["PPSq"]), y)
    # Zipcode_AvgPrice is a ZIP mean of the target over the whole file: a second leak.
    r2_zip_avg = test_r2(X.assign(Zipcode_AvgPrice=df["Zipcode_AvgPrice"]), y)
    identity_error = float(np.nanmax(np.abs(df["PPSq"] * df["Area"] / df["ListedPrice"] - 1)))
    return {
        "file": "backend/data/house_train_proc.csv",
        "rows": len(df),
        "base_features": base + ["State"],
        "r2_log_without_ppsq": r2_without,
        "r2_log_with_ppsq": r2_with,
        "r2_log_with_zipcode_avgprice": r2_zip_avg,
        "max_abs_relative_error_ppsq_x_area_vs_price": identity_error,
    }


def audit_macro(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["log_price"] = np.log(df["price"])

    per_state = {}
    for col in MACRO_COLUMNS:
        per_state[col] = df.groupby("state").apply(
            lambda g: g[col].corr(g["log_price"]), include_groups=False)

    # Pooled within-state correlation: demean both variables by state first.
    demeaned = df[MACRO_COLUMNS + ["log_price"]] - df.groupby("state")[MACRO_COLUMNS + ["log_price"]].transform("mean")
    within_std_share = (df.groupby("state")[MACRO_COLUMNS].std().mean()
                        / df[MACRO_COLUMNS].std())

    result = {}
    for col in MACRO_COLUMNS:
        r = per_state[col].dropna()
        result[col] = {
            "pooled_within_state_corr": float(demeaned[col].corr(demeaned["log_price"])),
            "median_state_corr": float(r.median()),
            "min_state_corr": float(r.min()),
            "max_state_corr": float(r.max()),
            "share_states_abs_corr_gt_0.3": float((r.abs() > 0.3).mean()),
            "within_state_std_as_share_of_total_std": float(within_std_share[col]),
            "per_state_corr": {k: round(float(v), 4) for k, v in r.items()},
        }
    return result


def audit_shuffle(df: pd.DataFrame) -> dict:
    y = np.log(df["price"])
    X = df[[c for c in KEEP_COLUMNS if c not in ("price", "state")] + MACRO_COLUMNS].copy()
    X["state"] = df["state"].astype("category").cat.codes

    r2_before = test_r2(X, y)

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = X.copy()
    for _, idx in df.groupby("state").groups.items():
        perm = rng.permutation(len(idx))
        shuffled.loc[idx, MACRO_COLUMNS] = X.loc[idx, MACRO_COLUMNS].to_numpy()[perm]
    r2_after = test_r2(shuffled, y)

    r2_no_macro = test_r2(X.drop(columns=MACRO_COLUMNS), y)
    return {
        "rows": len(df),
        "features": list(X.columns),
        "r2_log_with_macro": r2_before,
        "r2_log_macro_shuffled_within_state": r2_after,
        "r2_log_without_macro": r2_no_macro,
    }


def summary(res: dict) -> str:
    a, b, c = res["a_ppsq"], res["b_macro_within_state_corr"], res["c_macro_shuffle"]

    def both(d):
        return "  ".join(f"{m} {d[m]:.3f}" for m in MODELS)

    lines = [
        "LEAKAGE AUDIT - plain-English summary",
        "",
        f"(a) PPSq in house_train_proc.csv is price divided by area (PPSq x Area reproduces "
        f"the price to within {a['max_abs_relative_error_ppsq_x_area_vs_price']:.1e}).",
        f"    Test R2 on log price without PPSq:  {both(a['r2_log_without_ppsq'])}",
        f"    Test R2 on log price with PPSq:     {both(a['r2_log_with_ppsq'])}",
        "    The jump comes from the model reading the answer back, not from learning.",
        "    Zipcode_AvgPrice (a ZIP mean of the target over the whole file) also inflates R2:",
        f"                                        {both(a['r2_log_with_zipcode_avgprice'])}",
        "    This file is not used for training.",
        "",
        "(b) Genuine state-level macro data would be identical for every house in a state at a",
        "    given date, and national series (CPI, mortgage rate, fed funds) identical in every",
        "    state. In the Final Dataset all six vary from house to house within a state, and",
        "    the houses carry no listing date that could explain it.",
        "    Pooled within-state correlation with log price (within-state spread as % of total):",
    ]
    for col, v in b.items():
        lines.append(f"      {col:<25} r = {v['pooled_within_state_corr']:+.3f}   "
                     f"({v['within_state_std_as_share_of_total_std']:.0%})")
    lines += [
        "    mortgage_rate and interest_rate are strongly tied to the individual house's price;",
        "    the other four show no within-state relationship but are still not real values.",
        "",
        "(c) Test R2 on log price:",
        f"      with the macro columns:             {both(c['r2_log_with_macro'])}",
        f"      macro shuffled within each state:   {both(c['r2_log_macro_shuffled_within_state'])}",
        f"      macro columns removed:              {both(c['r2_log_without_macro'])}",
        "    Shuffling within a state keeps any genuine state-level signal, so the accuracy lost",
        "    came from a house-level link between the macro values and price. Shuffled and",
        "    removed give almost the same R2: the columns add nothing real. That is leakage,",
        "    and the six columns are dropped from all models.",
    ]
    return "\n".join(lines)


def main():
    ensure_dirs()
    sale_with_macro, _ = clean_sale(pd.read_csv(RAW_FILE), keep_macro=True)
    results = {
        "models": {
            "random_forest": "RandomForestRegressor(n_estimators=300, min_samples_leaf=2)",
            "xgboost": "XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8)",
        },
        "protocol": "80/20 train/test split, random_state=42, target log(price), metric test R2",
        "a_ppsq": audit_ppsq(),
        "b_macro_within_state_corr": audit_macro(sale_with_macro),
        "c_macro_shuffle": audit_shuffle(sale_with_macro),
    }
    text = summary(results)
    results["summary"] = text
    with open(METRICS_DIR / "leakage_audit.json", "w") as f:
        json.dump(results, f, indent=2)
    print(text)
    print(f"\nSaved {METRICS_DIR / 'leakage_audit.json'}")


if __name__ == "__main__":
    main()
