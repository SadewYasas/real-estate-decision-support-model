"""Forecast models for house price appreciation (state HPI) and rent growth (national CPI rent).

Model: gradient boosting (sklearn HistGradientBoostingRegressor) on the panels built by
src/pipeline/build_panel.py. Baselines (CLAUDE.md):
  (a) persistence: next-year growth = past-year growth;
  (b) ARIMA per series on the log level (orders (1,1,0), (0,1,1), (1,1,1), (2,1,0) with
      drift, chosen by AIC at every origin);
  (c) the same gradient boosting model without the macro features.
HPI also has a constrained variant, gbm_macro_constrained: the same model with monotonic
constraints so that a higher mortgage rate, a rise in the mortgage rate, higher
unemployment or a rise in unemployment can never raise predicted HPI growth (all other
features unconstrained). It is used for the forecasts and scenarios if its backtest RMSE
is within 5% of the unconstrained model's; otherwise gbm_macro is kept.

Evaluation: expanding-window rolling origin from 2018. At origin t every model is
refitted on the rows whose target was already known at t (origin <= t - horizon), then
forecasts every series at t. Hyper-parameters are fixed in advance, not tuned on the
backtest, so the backtest stays out of sample. Metrics: MAE, RMSE (percentage points of
growth), directional accuracy (sign of the growth), and direction of change versus the
past-year growth (acceleration). Diebold-Mariano tests (Newey-West variance, lag
horizon-1, on the per-origin mean squared error) compare the model with each baseline.

Uncertainty band: 10th / 90th percentiles of the recommended method's out-of-sample
errors (actual - forecast), added to its point forecast.

Latest forecast: origin 2025Q3 for HPI and 2025-09 for rent (fred_final.csv ends Sep 2025),
using only rows whose target was known at that origin, as in the backtest.

Outputs:
  artefacts/metrics/forecast.json                 every method x every metric, DM tests
  artefacts/metrics/forecast_backtest_{hpi,rent}.csv   all out-of-sample forecasts
  artefacts/models/forecasts.json                 latest forecasts with 10/90 bands
  artefacts/models/{hpi,rent}_forecast_model.joblib    fitted model + feature list (scenarios)
  artefacts/figures/forecast_backtest_{hpi,rent}.png, forecast_error_by_state.png

Run:  python -m src.models.train_forecast
"""
import json
import sys
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, ensure_dirs
from src.pipeline.build_panel import (HPI_BASE, HPI_HORIZON, HPI_MACRO, RENT_BASE, RENT_HORIZON,
                                      RENT_MACRO, load_cpi_rent, load_fred, load_hpi)

ARIMA_ORDERS = [(1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 1, 0)]
METHOD_LABELS = {"gbm_macro": "Gradient boosting + macro",
                 "gbm_macro_constrained": "Gradient boosting + macro, monotonic",
                 "gbm_no_macro": "Gradient boosting, no macro",
                 "persistence": "Persistence", "arima": "ARIMA"}
# Negative effect only (-1): rates and unemployment can only lower predicted HPI growth.
HPI_MONOTONIC = {"mortgage": -1, "mortgage_chg12": -1, "unemp": -1, "unemp_chg12": -1}
CONSTRAINED_RMSE_TOLERANCE = 1.05   # use the constrained model if RMSE <= 1.05 x unconstrained

SPECS = {
    "hpi": {
        "label": "house price appreciation (FHFA state HPI, all-transactions)",
        "panel": "hpi_panel.csv", "time": "quarter", "freq": "Q", "id": "state",
        "horizon": HPI_HORIZON, "base": HPI_BASE, "macro": HPI_MACRO,
        "first_origin": "2018Q1", "latest_origin": "2025Q3", "arima_start": "2000Q1",
        "gbm": dict(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                    l2_regularization=1.0),
        # name: (uses macro features, monotonic constraints)
        "variants": {"gbm_macro": (True, None), "gbm_macro_constrained": (True, HPI_MONOTONIC),
                     "gbm_no_macro": (False, None)},
    },
    "rent": {
        "label": "rent growth (BLS CPI rent of primary residence, US, NSA)",
        "panel": "rent_panel.csv", "time": "month", "freq": "M", "id": None,
        "horizon": RENT_HORIZON, "base": RENT_BASE, "macro": RENT_MACRO,
        "first_origin": "2018-01", "latest_origin": "2025-09", "arima_start": "2000-01",
        # One national series: far fewer rows, so a smaller model.
        "gbm": dict(max_iter=200, learning_rate=0.05, max_depth=2, min_samples_leaf=8,
                    l2_regularization=1.0),
        "variants": {"gbm_macro": (True, None), "gbm_no_macro": (False, None)},
    },
}

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_DARK, INK = "#2a78d6", "#104281", "#52514e"


# ----------------------------------------------------------------------------- data
def load_panel(spec: dict) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / spec["panel"])
    df["period"] = pd.PeriodIndex(df[spec["time"]], freq=spec["freq"])
    df["t"] = df["period"].map(lambda p: p.ordinal)
    if spec["id"] is None:
        df["series"] = "US"
    else:
        df["series"] = pd.Categorical(df[spec["id"]], categories=sorted(df[spec["id"]].unique()))
    return df


def level_series(name: str) -> dict:
    """Log level of the target index per series, as {series: pd.Series indexed by ordinal}."""
    if name == "hpi":
        h = load_hpi()
        out = {}
        for s, g in h.groupby("state"):
            g = g.sort_values("quarter")
            out[s] = pd.Series(np.log(g["hpi"].to_numpy()), index=[q.ordinal for q in g["quarter"]])
        return out
    r = load_cpi_rent()
    return {"US": pd.Series(np.log(r.to_numpy()), index=[m.ordinal for m in r.index])}


def features(spec: dict, macro: bool) -> list[str]:
    cols = spec["base"] + (spec["macro"] if macro else [])
    return cols + (["series"] if spec["id"] is not None else [])


def methods(spec: dict) -> list[str]:
    return list(spec["variants"]) + ["persistence", "arima"]


def make_gbm(spec: dict, monotonic: dict | None = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**spec["gbm"], early_stopping=False,
                                         categorical_features="from_dtype",
                                         monotonic_cst=monotonic,
                                         random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------- ARIMA
def arima_growth(y_log: pd.Series, origin: int, start: int, h: int) -> float:
    """Forecast growth (%) over h periods from the ARIMA (by AIC) fitted on y[start..origin]."""
    y = y_log.loc[start:origin].to_numpy()
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDERS:
            try:
                res = ARIMA(y, order=order, trend="t").fit()
            except Exception:
                continue
            if best is None or res.aic < best.aic:
                best = res
    if best is None:
        return np.nan
    return float(100 * (np.exp(best.forecast(h)[-1] - y[-1]) - 1))


def arima_for_series(series: str, y_log: pd.Series, origins: list[int], start: int, h: int):
    return [(series, t, arima_growth(y_log, t, start, h)) for t in origins]


# ----------------------------------------------------------------------------- backtest
def backtest(name: str, spec: dict, df: pd.DataFrame) -> pd.DataFrame:
    h = spec["horizon"]
    first = pd.Period(spec["first_origin"], freq=spec["freq"]).ordinal
    known = df.dropna(subset=["target"])
    origins = sorted(known.loc[known["t"] >= first, "t"].unique())
    rows = []
    for t in origins:
        train = known[known["t"] <= t - h]
        test = known[known["t"] == t]
        out = test[["series", "period", "t", "target"]].copy()
        for method, (macro, mono) in spec["variants"].items():
            cols = features(spec, macro)
            out[method] = make_gbm(spec, mono).fit(train[cols], train["target"]).predict(test[cols])
        out["persistence"] = test[spec["base"][0]].to_numpy()
        out["past_growth"] = test[spec["base"][0]].to_numpy()
        out["n_train"] = len(train)
        rows.append(out)
    bt = pd.concat(rows, ignore_index=True)

    levels = level_series(name)
    start = pd.Period(spec["arima_start"], freq=spec["freq"]).ordinal
    print(f"[{name}] ARIMA: {len(levels)} series x {len(origins)} origins ...", flush=True)
    res = Parallel(n_jobs=-1)(
        delayed(arima_for_series)(s, levels[s], origins, start, h) for s in levels)
    ar = pd.DataFrame([r for chunk in res for r in chunk], columns=["series", "t", "arima"])
    ar["series"] = ar["series"].astype(str)
    bt["series"] = bt["series"].astype(str)
    return bt.merge(ar, on=["series", "t"], how="left").rename(columns={"target": "actual"})


# ----------------------------------------------------------------------------- metrics
def method_metrics(bt: pd.DataFrame, method: str) -> dict:
    e = bt["actual"] - bt[method]
    out = {
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "directional_accuracy": float((np.sign(bt[method]) == np.sign(bt["actual"])).mean()),
        "n": int(len(bt)),
    }
    if method != "persistence":  # persistence predicts no change by definition
        out["acceleration_accuracy"] = float(
            (np.sign(bt[method] - bt["past_growth"]) == np.sign(bt["actual"] - bt["past_growth"])).mean())
    return out


def dm_test(bt: pd.DataFrame, a: str, b: str, lag: int) -> dict:
    """Diebold-Mariano on per-origin mean squared errors; negative stat = a is more accurate."""
    d = (bt.assign(la=(bt["actual"] - bt[a]) ** 2, lb=(bt["actual"] - bt[b]) ** 2)
         .groupby("t")[["la", "lb"]].mean())
    d = (d["la"] - d["lb"]).to_numpy()
    T, dbar = len(d), d.mean()
    dc = d - dbar
    gamma = [np.sum(dc[k:] * dc[:T - k]) / T for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    stat = dbar / np.sqrt(var / T) if var > 0 else np.nan
    return {"mean_loss_difference": float(dbar), "dm_stat": float(stat),
            "p_value": float(2 * (1 - norm.cdf(abs(stat)))), "n_origins": int(T),
            "better": a if dbar < 0 else b}


def evaluate(name: str, spec: dict, bt: pd.DataFrame) -> dict:
    lag = spec["horizon"] - 1
    ms = methods(spec)
    metrics = {m: method_metrics(bt, m) for m in ms}

    # Which gradient-boosting + macro variant feeds the forecasts and scenarios.
    selection = {"gbm_variant": "gbm_macro"}
    if "gbm_macro_constrained" in ms:
        ratio = metrics["gbm_macro_constrained"]["rmse"] / metrics["gbm_macro"]["rmse"]
        use_constrained = ratio <= CONSTRAINED_RMSE_TOLERANCE
        selection = {
            "gbm_variant": "gbm_macro_constrained" if use_constrained else "gbm_macro",
            "rule": f"use gbm_macro_constrained if its RMSE <= {CONSTRAINED_RMSE_TOLERANCE} x gbm_macro RMSE",
            "rmse_ratio_constrained_to_unconstrained": float(ratio),
            "monotonic_constraints": HPI_MONOTONIC,
        }
    other_variant = {"gbm_macro", "gbm_macro_constrained"} - {selection["gbm_variant"]}
    pool = [m for m in ms if m not in other_variant]
    best = min(pool, key=lambda m: metrics[m]["rmse"])
    errors = bt["actual"] - bt[best]
    p10, p90 = float(errors.quantile(0.10)), float(errors.quantile(0.90))
    by_year = {}
    for year, g in bt.groupby(bt["period"].dt.year):
        by_year[int(year)] = {m: {"mae": float((g["actual"] - g[m]).abs().mean())} for m in ms}
    result = {
        "target": spec["label"],
        "horizon": f"{spec['horizon']} {'quarters' if spec['freq'] == 'Q' else 'months'} ahead, % growth",
        "origins": f"{bt['period'].min()} - {bt['period'].max()} ({bt['t'].nunique()} origins)",
        "n_series": int(bt["series"].nunique()),
        "features_macro": spec["macro"],
        "features_base": spec["base"] + (["state"] if spec["id"] else []),
        "gbm_params": spec["gbm"],
        "metrics": metrics,
        "base_rate_actual_growth_positive": float((bt["actual"] > 0).mean()),
        "dm_tests_vs_gbm_macro": {b: dm_test(bt, "gbm_macro", b, lag)
                                  for b in ms if b != "gbm_macro"},
        **({"dm_tests_vs_gbm_macro_constrained": {b: dm_test(bt, "gbm_macro_constrained", b, lag)
                                                  for b in ms if b != "gbm_macro_constrained"}}
           if "gbm_macro_constrained" in ms else {}),
        "dm_test_gbm_no_macro_vs_persistence": dm_test(bt, "gbm_no_macro", "persistence", lag),
        "macro_improves_on_no_macro": bool(metrics["gbm_macro"]["rmse"] < metrics["gbm_no_macro"]["rmse"]),
        "gbm_macro_beats_persistence": bool(metrics["gbm_macro"]["rmse"] < metrics["persistence"]["rmse"]),
        "gbm_selection": selection,
        "recommended_method": best,
        "recommended_rule": ("lowest backtest RMSE among the selected gbm + macro variant, "
                             "gbm_no_macro, persistence and ARIMA"),
        "error_band": {"method": best, "p10": p10, "p90": p90,
                       "empirical_coverage": float(((errors >= p10) & (errors <= p90)).mean())},
        "mae_by_origin_year": by_year,
    }
    if spec["id"] is not None:
        result["mae_by_state"] = {
            s: {m: float((g["actual"] - g[m]).abs().mean()) for m in ms}
            for s, g in bt.groupby("series")}
    return result


# ----------------------------------------------------------------------------- latest forecast
def latest_forecast(name: str, spec: dict, df: pd.DataFrame, ev: dict) -> dict:
    h = spec["horizon"]
    T = pd.Period(spec["latest_origin"], freq=spec["freq"])
    known = df.dropna(subset=["target"])
    train = known[known["t"] <= T.ordinal - h]          # as-of the origin, like the backtest
    now = df[df["t"] == T.ordinal].copy()
    assert len(now), f"no feature rows at {T}"
    preds = pd.DataFrame({"series": now["series"].astype(str).to_numpy()})
    models = {}
    for method, (macro, mono) in spec["variants"].items():
        cols = features(spec, macro)
        models[method] = (make_gbm(spec, mono).fit(train[cols], train["target"]), cols)
        preds[method] = models[method][0].predict(now[cols])
    preds["persistence"] = now[spec["base"][0]].to_numpy()
    levels = level_series(name)
    start = pd.Period(spec["arima_start"], freq=spec["freq"]).ordinal
    preds["arima"] = [arima_growth(levels[s], T.ordinal, start, h) for s in preds["series"]]

    band = ev["error_band"]
    best = band["method"]
    chosen = ev["gbm_selection"]["gbm_variant"]
    for variant in [v for v in spec["variants"] if v.startswith("gbm_macro")]:
        model, cols = models[variant]
        bundle = {"model": model, "features": cols, "variant": variant, "origin": str(T),
                  "horizon": h, "latest_features": now[cols].assign(
                      series=now["series"].astype(str)).to_dict(orient="records")}
        joblib.dump(bundle, MODELS_DIR / f"{name}_forecast_model_{variant}.joblib")
        if variant == chosen:  # the one scenarios use
            joblib.dump(bundle, MODELS_DIR / f"{name}_forecast_model.joblib")

    # Part of the forecast horizon has already happened: report it as a check, not a metric.
    out = {}
    for _, r in preds.iterrows():
        y = levels[r["series"]]
        last = y.dropna().index.max()
        realised = float(100 * (np.exp(y.loc[last] - y.loc[T.ordinal]) - 1))
        steps = last - T.ordinal
        out[r["series"]] = {
            "forecast": float(r[best]),
            "p10": float(r[best] + band["p10"]),
            "p90": float(r[best] + band["p90"]),
            **{m: float(r[m]) for m in methods(spec)},
            "realised_so_far": {"growth_pct": realised, "periods": int(steps),
                                "to": str(pd.Period(ordinal=last, freq=spec["freq"]))},
        }
    return {
        "origin": str(T),
        "horizon": ev["horizon"],
        "method": best,
        "scenario_model": chosen,
        "band": f"10th-90th percentile of {best} backtest errors",
        "training_rows": int(len(train)),
        "forecasts": out,
    }


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)


def plot_backtest(name: str, spec: dict, bt: pd.DataFrame, ev: dict):
    g = bt.groupby("t").agg(period=("period", "first"), actual=("actual", "mean"),
                            **{m: (m, "mean") for m in methods(spec)})
    x = g["period"].dt.to_timestamp()
    ms = methods(spec)
    nrows = (len(ms) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.4 * nrows), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax in axes.flat[len(ms):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, ms):
        _style(ax)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.plot(x, g["actual"], color=INK, linewidth=2, label="Actual")
        ax.plot(x, g[m], color=BLUE, linewidth=2, label="Forecast")
        mt = ev["metrics"][m]
        ax.set_title(f"{METHOD_LABELS[m]}   MAE {mt['mae']:.2f}  RMSE {mt['rmse']:.2f}",
                     loc="left", fontsize=9.5, fontweight="bold", color=TEXT)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    unit = "4-quarter" if spec["freq"] == "Q" else "12-month"
    what = "mean across 51 states" if spec["id"] else "US"
    fig.supylabel(f"Next {unit} growth, % ({what})", fontsize=9, color=TEXT_2)
    for ax in axes.flat[max(len(ms) - 2, 0):len(ms)]:
        ax.set_xlabel("Forecast origin", fontsize=8.5, color=TEXT_2)
        ax.tick_params(labelbottom=True)
    fig.suptitle(f"Rolling-origin backtest: {spec['label']}", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold", color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"forecast_backtest_{name}.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_error_by_state(ev: dict):
    model = ev["gbm_selection"]["gbm_variant"]
    st = pd.DataFrame(ev["mae_by_state"]).T.sort_values(model)
    fig, ax = plt.subplots(figsize=(7.5, 0.22 * len(st) + 1.4), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(st))
    ax.barh(y, st[model], height=0.7, color=BLUE, label=METHOD_LABELS[model])
    ax.scatter(st["persistence"], y, color=TEXT, s=14, zorder=3, label="Persistence")
    ax.set_yticks(y, st.index, fontsize=7.5)
    ax.set_ylim(-0.7, len(st) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Backtest MAE, percentage points of 4-quarter HPI growth", fontsize=8.5, color=TEXT_2)
    ax.set_title("HPI forecast error by state (2018-2025 origins)", loc="left",
                 fontsize=10.5, fontweight="bold", color=TEXT)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "forecast_error_by_state.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def print_summary(name: str, ev: dict):
    print(f"\n{name.upper()}  {ev['origins']}, {ev['n_series']} series")
    print(f"  {'method':<38}{'MAE':>7}{'RMSE':>7}{'dir.acc':>9}{'accel.acc':>11}")
    for m in ev["metrics"]:
        r = ev["metrics"][m]
        acc = f"{r['acceleration_accuracy']:.1%}" if "acceleration_accuracy" in r else "-"
        print(f"  {METHOD_LABELS[m]:<38}{r['mae']:>7.2f}{r['rmse']:>7.2f}"
              f"{r['directional_accuracy']:>9.1%}{acc:>11}")
    for b, d in ev["dm_tests_vs_gbm_macro"].items():
        print(f"  DM gbm_macro vs {b:<22} stat {d['dm_stat']:+.2f}  p {d['p_value']:.3f}  better: {d['better']}")
    for b, d in ev.get("dm_tests_vs_gbm_macro_constrained", {}).items():
        print(f"  DM constrained vs {b:<20} stat {d['dm_stat']:+.2f}  p {d['p_value']:.3f}  better: {d['better']}")
    sel = ev["gbm_selection"]
    if "rmse_ratio_constrained_to_unconstrained" in sel:
        print(f"  constrained / unconstrained RMSE = {sel['rmse_ratio_constrained_to_unconstrained']:.3f}"
              f" -> scenarios use {sel['gbm_variant']}")
    print(f"  recommended (lowest RMSE): {ev['recommended_method']}; band p10 {ev['error_band']['p10']:+.2f}"
          f" / p90 {ev['error_band']['p90']:+.2f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    t0 = time.time()
    evaluation, forecasts = {}, {}
    for name, spec in SPECS.items():
        df = load_panel(spec)
        bt = backtest(name, spec, df)
        bt.drop(columns="period").assign(origin=bt["period"].astype(str)).to_csv(
            METRICS_DIR / f"forecast_backtest_{name}.csv", index=False)
        ev = evaluate(name, spec, bt)
        evaluation[name] = ev
        forecasts[name] = latest_forecast(name, spec, df, ev)
        plot_backtest(name, spec, bt, ev)
        if spec["id"] is not None:
            plot_error_by_state(ev)
        print_summary(name, ev)

    fred = load_fred()
    latest_mortgage = fred.dropna(subset=["mortgage_rate"]).sort_values("month").iloc[-1]
    forecasts["mortgage_rate_latest"] = {"value_pct": float(latest_mortgage["mortgage_rate"]),
                                         "month": str(latest_mortgage["month"])}
    forecasts["note"] = ("Rent growth is national (CPI rent, US city average) and applies to every "
                         "state. Indicative only - not financial or valuation advice.")
    with open(METRICS_DIR / "forecast.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    with open(MODELS_DIR / "forecasts.json", "w") as f:
        json.dump(forecasts, f, indent=2)
    print(f"\nFinished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
