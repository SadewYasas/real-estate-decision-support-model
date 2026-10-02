"""Train and compare price models (sale now; rent reuses this in Step 4).

Protocol (CLAUDE.md / Chapter 4):
- target log(price); 80/20 train/test split (random_state=42);
- candidates: LinearRegression (baseline), RandomForest, XGBoost, CatBoost;
- 5-fold CV on the training set; RandomizedSearchCV (n_iter=20) for the three tree models;
- select the candidate with the lowest mean CV RMSE (log scale, the scale the models fit);
- report MAE, RMSE, MAPE, R² in dollars and R² on the log scale on the test set,
  plus CV mean ± std for every metric.

Encoders sit inside each sklearn Pipeline, so they are fitted on training folds only.
The 80/20 split and the CV folds are grouped (GroupShuffleSplit / GroupKFold) on
"group_cols", so the same property re-listed, or an identical floor plan at the same
location, never sits on both sides of a split.
If it defines "ablation", the selected model (same hyper-parameters) is also trained
without those columns and reported, and saved as {name}_model_{ablation}.joblib.
Predictions are returned to dollars with exp(); this estimates the conditional median.

Checkpointing: after every candidate the fitted pipeline is saved to
artefacts/models/candidates/ and the results so far to {name}_models.json
("status": "partial"). A rerun with the same settings resumes from there; use --fresh
to retrain everything.

Outputs:
  artefacts/models/{name}_model.joblib    best fitted pipeline (refit on the full training set)
  artefacts/models/{name}_model_meta.json features, winner, library versions
  artefacts/metrics/{name}_models.json    every candidate x every metric
  artefacts/figures/{name}_pred_vs_actual.png, {name}_shap_summary.png

Run:  python -m src.models.train_price_models --dataset sale --log artefacts/logs/train_sale.log
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import catboost
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from catboost import CatBoostRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (make_scorer, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error, r2_score)
from sklearn.base import clone
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     RandomizedSearchCV, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, ensure_dirs

N_ITER = 20
CV_FOLDS = 5

DATASETS = {
    "sale": {
        "file": "sale_clean.csv",
        "target": "price",
        "categorical": ["state_code"],
        "numeric": ["beds", "baths", "living_space", "zip_code_density", "median_household_income"],
        # Right-skewed inputs are logged for the linear baseline only (trees are invariant).
        "log_for_linear": ["living_space", "zip_code_density", "median_household_income"],
        "label": "sale price",
        # D1 has no coordinates; density + income identify the ZIP. Same ZIP + same size,
        # beds and baths = the same property re-listed or an identical floor plan.
        "group_cols": ["zip_code_density", "median_household_income", "living_space",
                       "beds", "baths"],
    },
    "rent": {
        "file": "rent_clean.csv",
        "target": "rent",
        # Aligned with the sale property the user enters: state, beds, baths, floor area,
        # plus coordinates (from the ZIP lookup in Step 8). D2 has no income/density.
        "categorical": ["state"],
        "numeric": ["bedrooms", "bathrooms", "square_feet", "latitude", "longitude"],
        "log_for_linear": ["square_feet"],
        "label": "monthly rent",
        # Same location + same floor area = the same flat (re-listed) or the same floor plan.
        "group_cols": ["latitude", "longitude", "square_feet"],
        "ablation": {"name": "no_coords", "drop": ["latitude", "longitude"]},
    },
}

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_DARK = "#2a78d6", "#104281"


# ----------------------------------------------------------------------------- scoring
def _usd(y_log):
    return np.exp(np.asarray(y_log))


def rmse_usd(y_log, p_log):
    return float(np.sqrt(mean_squared_error(_usd(y_log), _usd(p_log))))


def mae_usd(y_log, p_log):
    return float(mean_absolute_error(_usd(y_log), _usd(p_log)))


def mape_usd(y_log, p_log):
    return float(mean_absolute_percentage_error(_usd(y_log), _usd(p_log)))


def r2_usd(y_log, p_log):
    return float(r2_score(_usd(y_log), _usd(p_log)))


SCORING = {
    "rmse_log": "neg_root_mean_squared_error",
    "r2_log": "r2",
    "rmse_usd": make_scorer(rmse_usd, greater_is_better=False),
    "mae_usd": make_scorer(mae_usd, greater_is_better=False),
    "mape_usd": make_scorer(mape_usd, greater_is_better=False),
    "r2_usd": make_scorer(r2_usd),
}
# Metrics where sklearn stores the negative value.
NEGATED = {"rmse_log", "rmse_usd", "mae_usd", "mape_usd"}


def test_metrics(y_log, p_log) -> dict:
    return {
        "mae_usd": mae_usd(y_log, p_log),
        "rmse_usd": rmse_usd(y_log, p_log),
        "mape_usd": mape_usd(y_log, p_log),
        "r2_usd": r2_usd(y_log, p_log),
        "r2_log": float(r2_score(y_log, p_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_log, p_log))),
    }


# ----------------------------------------------------------------------------- candidates
def build_candidates(spec: dict) -> dict:
    cat, num = spec["categorical"], spec["numeric"]
    logged = spec["log_for_linear"]
    plain = [c for c in num if c not in logged]

    linear_prep = ColumnTransformer([
        ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one"), logged),
        ("num", "passthrough", plain),
    ], verbose_feature_names_out=False)

    tree_prep = ColumnTransformer([
        ("state", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat),
        ("num", "passthrough", num),
    ], verbose_feature_names_out=False).set_output(transform="pandas")

    cat_prep = ColumnTransformer([("all", "passthrough", cat + num)],
                                 verbose_feature_names_out=False).set_output(transform="pandas")

    # Parallelism: only one level may be parallel, never both the search and the model
    # (nested parallelism multiplies workers x threads and can exhaust memory).
    # "search_jobs" is the n_jobs of RandomizedSearchCV / cross_validate for that model.
    return {
        "linear_regression": {
            "pipeline": Pipeline([("prep", linear_prep), ("model", LinearRegression())]),
            "space": None,
            "search_jobs": -1,
        },
        "random_forest": {
            "pipeline": Pipeline([("prep", tree_prep), ("model", RandomForestRegressor(
                random_state=RANDOM_STATE, n_jobs=1))]),
            "search_jobs": 4,  # each RF fit holds hundreds of deep trees in memory
            "space": {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": [None, 10, 15, 20, 30],
                "model__min_samples_leaf": randint(1, 10),
                "model__max_features": uniform(0.3, 0.7),
            },
        },
        "xgboost": {
            "pipeline": Pipeline([("prep", tree_prep), ("model", XGBRegressor(
                random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"))]),
            "search_jobs": -1,
            "space": {
                "model__n_estimators": randint(300, 1500),
                "model__learning_rate": loguniform(0.01, 0.2),
                "model__max_depth": randint(3, 11),
                "model__min_child_weight": randint(1, 11),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(1e-3, 10),
            },
        },
        "catboost": {
            "pipeline": Pipeline([("prep", cat_prep), ("model", CatBoostRegressor(
                cat_features=tuple(cat), random_seed=RANDOM_STATE, thread_count=-1,
                verbose=0, allow_writing_files=False))]),
            "search_jobs": 1,  # CatBoost parallelises internally over all cores
            "space": {
                "model__iterations": randint(300, 1500),
                "model__learning_rate": loguniform(0.02, 0.2),
                "model__depth": randint(4, 9),
                "model__l2_leaf_reg": loguniform(1, 10),
            },
        },
    }


def cv_summary(cv_results: dict, index: int | None = None) -> dict:
    """Mean and std across folds for every metric (for one search candidate if index given)."""
    out = {}
    for m in SCORING:
        if index is None:
            scores = np.asarray(cv_results[f"test_{m}"])
        else:
            scores = np.array([cv_results[f"split{k}_test_{m}"][index] for k in range(CV_FOLDS)])
        if m in NEGATED:
            scores = -scores
        out[m] = {"mean": float(scores.mean()), "std": float(scores.std(ddof=1))}
    return out


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)


def plot_pred_vs_actual(y_true, y_pred, name, label, model_name, metrics):
    fig, ax = plt.subplots(figsize=(6.4, 6.0), facecolor=SURFACE)
    _style(ax)
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    hb = ax.hexbin(y_true, y_pred, gridsize=60, xscale="log", yscale="log", bins="log",
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"]),
                   mincnt=1, extent=(np.log10(lo), np.log10(hi), np.log10(lo), np.log10(hi)))
    ax.plot([lo, hi], [lo, hi], color=TEXT, linewidth=1, linestyle="--")
    ax.text(hi / 1.15, hi / 1.05, "perfect prediction", color=TEXT_2, fontsize=8,
            ha="right", va="top", rotation=45, rotation_mode="anchor")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v/1e6:g}M" if v >= 1e6 else (f"${v/1e3:g}k" if v >= 1e3 else f"${v:g}"))
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(fmt)
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"Actual {label} (log scale)", color=TEXT_2, fontsize=9)
    ax.set_ylabel(f"Predicted {label} (log scale)", color=TEXT_2, fontsize=9)
    ax.set_title(f"Predicted vs actual {label}: {model_name}, test set (n = {len(y_true):,})",
                 loc="left", fontsize=10, fontweight="bold", color=TEXT)
    ax.text(0.03, 0.97,
            f"R² (log) {metrics['r2_log']:.3f}\nR² ($) {metrics['r2_usd']:.3f}\n"
            f"MAPE {metrics['mape_usd']:.1%}\nMAE ${metrics['mae_usd']:,.0f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=TEXT,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.4"))
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Properties per cell", color=TEXT_2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_pred_vs_actual.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def shap_summary(pipeline, X_test, name, model_name, label) -> dict:
    """TreeSHAP beeswarm on a fixed sample of the test set; returns mean |SHAP| per feature."""
    sample = X_test.sample(min(2000, len(X_test)), random_state=RANDOM_STATE)
    X_t = pipeline[:-1].transform(sample)
    model = pipeline[-1]
    explainer = shap.TreeExplainer(model)
    values = explainer(X_t)
    if model_name == "catboost":
        # Beeswarm colour needs numbers; show the state as its category code for colour only.
        values.data = X_t.assign(**{c: X_t[c].astype("category").cat.codes
                                    for c in X_t.select_dtypes(exclude="number")}).to_numpy(float)
    plt.figure(facecolor=SURFACE)
    shap.plots.beeswarm(values, max_display=len(X_t.columns), show=False, plot_size=(7.5, 4.2))
    fig = plt.gcf()
    fig.patch.set_facecolor(SURFACE)
    ax = plt.gca()
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(f"SHAP value: effect on log({label}) (0.1 ≈ +10.5%)", fontsize=9, color=TEXT_2)
    ax.set_title(f"SHAP summary: {model_name}, {len(sample):,} test properties",
                 loc="left", fontsize=10, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{name}_shap_summary.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    mean_abs = np.abs(values.values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return {X_t.columns[i]: float(mean_abs[i]) for i in order}


# ----------------------------------------------------------------------------- checkpoints
CANDIDATES_DIR = MODELS_DIR / "candidates"


def run_fingerprint(spec: dict, n_rows: int) -> str:
    """Identifies the settings a checkpoint was made with, so a stale one is never reused."""
    key = json.dumps({"spec": spec, "n_rows": n_rows, "n_iter": N_ITER, "folds": CV_FOLDS,
                      "seed": RANDOM_STATE}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def save_checkpoint(name, fingerprint, cand, model, results, protocol, features, n, n_train, n_test):
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATES_DIR / f"{name}_{cand}.joblib", compress=3)
    partial = {"dataset": name, "status": "partial", "run_fingerprint": fingerprint,
               "rows": {"total": n, "train": n_train, "test": n_test}, "features": features,
               "protocol": protocol, "candidates": results}
    with open(METRICS_DIR / f"{name}_models.json", "w") as f:
        json.dump(partial, f, indent=2)


def load_checkpoint(name, fingerprint, X_test, fresh) -> tuple[dict, dict]:
    path = METRICS_DIR / f"{name}_models.json"
    if fresh or not path.exists():
        return {}, {}
    with open(path) as f:
        saved = json.load(f)
    if saved.get("run_fingerprint") != fingerprint:
        print(f"[{name}] existing {path.name} was made with other settings - starting fresh", flush=True)
        return {}, {}
    results, fitted = {}, {}
    for cand, res in saved["candidates"].items():
        model_path = CANDIDATES_DIR / f"{name}_{cand}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            results[cand] = res
            fitted[cand] = (model, model.predict(X_test))
    return results, fitted


class Tee:
    """Write console output to a log file as well (also captures tracebacks)."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, attr):  # encoding, isatty(), fileno() ... come from the console
        return getattr(self.stream, attr)


# ----------------------------------------------------------------------------- main
def run(name: str, fresh: bool = False):
    ensure_dirs()
    spec = DATASETS[name]
    features = spec["categorical"] + spec["numeric"]
    df = pd.read_csv(PROCESSED_DIR / spec["file"])
    X = df[features]
    y = np.log(df[spec["target"]])
    groups = df.groupby(spec["group_cols"]).ngroup()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                  .split(X, y, groups))
    X_train, X_test, y_train, y_test = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    g_train = groups.iloc[tr]
    cv = GroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    protocol = {
        "split": f"80/20 GroupShuffleSplit on {', '.join(spec['group_cols'])}, random_state=42",
        "cv": f"{CV_FOLDS}-fold GroupKFold (shuffle, random_state=42) on the training set, same groups",
        "tuning": f"RandomizedSearchCV n_iter={N_ITER}, refit on lowest mean CV RMSE (log)",
        "parallelism": "search n_jobs: linear -1, random_forest 4, xgboost -1 (model n_jobs=1), "
                       "catboost 1 (model thread_count=-1)",
        "selection": "lowest mean CV RMSE on log scale",
        "back_transform": "exp(prediction) - estimates the conditional median",
        "n_groups": int(groups.nunique()),
        "rows_in_repeated_groups": int(groups.duplicated(keep=False).sum()),
    }
    fingerprint = run_fingerprint(spec, len(df))
    results, fitted = load_checkpoint(name, fingerprint, X_test, fresh)

    for cand, c in build_candidates(spec).items():
        if cand in results:
            print(f"[{name}] {cand}: loaded from checkpoint "
                  f"(CV RMSE(log) {results[cand]['cv']['rmse_log']['mean']:.4f})", flush=True)
            continue
        t0 = time.time()
        print(f"[{name}] {cand} ... started {time.strftime('%H:%M:%S')}", flush=True)
        if c["space"] is None:
            cvr = cross_validate(c["pipeline"], X_train, y_train, groups=g_train, cv=cv,
                                 scoring=SCORING, n_jobs=c["search_jobs"])
            model = c["pipeline"].fit(X_train, y_train)
            cv_stats, best_params = cv_summary(cvr), {}
        else:
            search = RandomizedSearchCV(
                c["pipeline"], c["space"], n_iter=N_ITER, cv=cv, scoring=SCORING,
                refit="rmse_log", n_jobs=c["search_jobs"], random_state=RANDOM_STATE,
                error_score="raise")
            search.fit(X_train, y_train, groups=g_train)
            model = search.best_estimator_
            cv_stats = cv_summary(search.cv_results_, search.best_index_)
            best_params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                           for k, v in search.best_params_.items()}
        pred = model.predict(X_test)
        results[cand] = {
            "cv": cv_stats,
            "test": test_metrics(y_test, pred),
            "best_params": best_params,
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[cand] = (model, pred)
        save_checkpoint(name, fingerprint, cand, model, results, protocol, features, len(df),
                        len(X_train), len(X_test))
        print(f"    CV RMSE(log) {cv_stats['rmse_log']['mean']:.4f} ± {cv_stats['rmse_log']['std']:.4f}"
              f" | test R²(log) {results[cand]['test']['r2_log']:.4f}"
              f" | {results[cand]['fit_seconds']}s", flush=True)

    best = min(results, key=lambda k: results[k]["cv"]["rmse_log"]["mean"])
    best_model, best_pred = fitted[best]
    tuned = {f"model__{k}": v for k, v in results[best]["best_params"].items()}

    ablation = None
    if spec.get("ablation"):
        ab = spec["ablation"]
        ab_spec = {**spec, "numeric": [c for c in spec["numeric"] if c not in ab["drop"]],
                   "log_for_linear": [c for c in spec["log_for_linear"] if c not in ab["drop"]]}
        ab_features = ab_spec["categorical"] + ab_spec["numeric"]
        ab_cand = build_candidates(ab_spec)[best]
        ab_pipe = ab_cand["pipeline"].set_params(**tuned)
        print(f"[{name}] ablation {ab['name']} ({best}, same hyper-parameters) ...", flush=True)
        ab_cv = cross_validate(clone(ab_pipe), X_train[ab_features], y_train, groups=g_train,
                               cv=cv, scoring=SCORING, n_jobs=ab_cand["search_jobs"])
        ab_model = ab_pipe.fit(X_train[ab_features], y_train)
        ablation = {
            "name": ab["name"], "dropped": ab["drop"], "model": best, "features": ab_features,
            "cv": cv_summary(ab_cv),
            "test": test_metrics(y_test, ab_model.predict(X_test[ab_features])),
        }
        joblib.dump(ab_model, MODELS_DIR / f"{name}_model_{ab['name']}.joblib")

    # How much would an ungrouped random split have flattered the selected model?
    print(f"[{name}] ungrouped random-split comparison ({best}) ...", flush=True)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    rs_model = build_candidates(spec)[best]["pipeline"].set_params(**tuned).fit(Xr_tr, yr_tr)
    random_split_check = {
        "note": "Selected model and hyper-parameters on an ungrouped 80/20 split, for comparison only.",
        "test": test_metrics(yr_te, rs_model.predict(Xr_te)),
    }

    # Test error by state for the selected model (used for the "error by state" discussion).
    test_df = X_test.assign(actual=np.exp(y_test), predicted=np.exp(best_pred))
    test_df["ape"] = (test_df["predicted"] - test_df["actual"]).abs() / test_df["actual"]
    by_state = (test_df.groupby(spec["categorical"][0])
                .agg(n=("ape", "size"), mape=("ape", "mean"), median_ape=("ape", "median"))
                .sort_values("mape"))

    plot_pred_vs_actual(np.exp(y_test.to_numpy()), np.exp(best_pred), name, spec["label"],
                        best, results[best]["test"])
    shap_importance = shap_summary(best_model, X_test, name, best, spec["label"]) \
        if best != "linear_regression" else None

    baseline = results["linear_regression"]["test"]
    output = {
        "dataset": name,
        "rows": {"total": len(df), "train": len(X_train), "test": len(X_test)},
        "features": features,
        "target": f"log({spec['target']})",
        "status": "complete",
        "run_fingerprint": fingerprint,
        "protocol": protocol,
        "selected_model": best,
        "candidates": results,
        "improvement_over_baseline_test": {
            "r2_log": results[best]["test"]["r2_log"] - baseline["r2_log"],
            "mape_usd": results[best]["test"]["mape_usd"] - baseline["mape_usd"],
        },
        "selected_model_test_error_by_state": {
            s: {"n": int(r.n), "mape": float(r.mape), "median_ape": float(r.median_ape)}
            for s, r in by_state.iterrows()},
        "shap_mean_abs_log_scale": shap_importance,
        "ablation": ablation,
        "random_split_check": random_split_check,
    }
    with open(METRICS_DIR / f"{name}_models.json", "w") as f:
        json.dump(output, f, indent=2)

    joblib.dump(best_model, MODELS_DIR / f"{name}_model.joblib")
    meta = {
        "dataset": name, "model": best, "features": features,
        "categorical": spec["categorical"], "target": f"log({spec['target']})",
        "best_params": results[best]["best_params"], "test": results[best]["test"],
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "catboost": catboost.__version__,
                     "shap": shap.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(MODELS_DIR / f"{name}_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print_report(output)


def print_report(o: dict):
    print(f"\n{o['dataset'].upper()} MODELS  (train {o['rows']['train']:,} / test {o['rows']['test']:,})")
    print(f"  {'model':<18}{'CV RMSE(log)':>18}{'CV R2(log)':>16}"
          f"{'test R2 log':>13}{'test R2 $':>11}{'MAPE':>8}{'MAE $':>11}{'RMSE $':>11}")
    for m, r in o["candidates"].items():
        cv, t = r["cv"], r["test"]
        flag = " *" if m == o["selected_model"] else ""
        print(f"  {m + flag:<18}{cv['rmse_log']['mean']:>11.4f} ±{cv['rmse_log']['std']:.4f}"
              f"{cv['r2_log']['mean']:>9.4f} ±{cv['r2_log']['std']:.3f}"
              f"{t['r2_log']:>13.4f}{t['r2_usd']:>11.4f}{t['mape_usd']:>8.1%}"
              f"{t['mae_usd']:>11,.0f}{t['rmse_usd']:>11,.0f}")
    print(f"  * selected by lowest mean CV RMSE (log)")
    if o.get("ablation"):
        a = o["ablation"]
        print(f"  ablation {a['name']} (drop {', '.join(a['dropped'])}): "
              f"CV RMSE(log) {a['cv']['rmse_log']['mean']:.4f}, test R2 log {a['test']['r2_log']:.4f}, "
              f"MAPE {a['test']['mape_usd']:.1%}")
    if o.get("random_split_check"):
        t = o["random_split_check"]["test"]
        print(f"  ungrouped random split (comparison only): test R2 log {t['r2_log']:.4f}, "
              f"MAPE {t['mape_usd']:.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="sale")
    ap.add_argument("--log", type=Path, help="also write all console output to this file")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoints and retrain everything")
    args = ap.parse_args()
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "a", encoding="utf-8")
        log_file.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  dataset={args.dataset}"
                       f"  fresh={args.fresh} =====\n")
        sys.stdout, sys.stderr = Tee(sys.stdout, log_file), Tee(sys.stderr, log_file)
    t_start = time.time()
    run(args.dataset, fresh=args.fresh)
    print(f"\nFinished in {(time.time() - t_start) / 60:.1f} min")
