"""Decision-flip experiment (Step 7): does using forecasts change rent-vs-buy decisions?

Properties: the full sale test set from Step 3 (same grouped 80/20 split, 7,622 houses in
29 states), so no property was used to train the sale model.

Inputs per property:
- P: the sale model's predicted price (what the system would use). A robustness run uses
  the actual listing price instead.
- R: rent_model_no_coords (state, bedrooms, bathrooms, square feet) x the CPI rent factor to
  Sep 2025. The sale data has no coordinates, so the coordinate-based rent model cannot be
  used. The rent model was trained on apartments (to 8,000 sq ft, 0-9 beds); the few larger
  houses are extrapolations.
- r: the latest mortgage rate (Sep 2025); all other assumptions at the CLAUDE.md defaults.

Three ways of setting growth (each held constant over the holding period):
  (a) fixed:      g = 3%, q = 3% (typical online calculators);
  (b) historical: g = the state's average annual HPI growth over the last 10 years
                  (2015Q3-2025Q3, FHFA all-transactions), q = average annual national CPI
                  rent growth over the same 10 years (Sep 2015 - Sep 2025);
  (c) forecast:   g = the state's 4-quarter forecast (gbm_macro_constrained), q = the
                  national 12-month ARIMA forecast (Step 5, origin 2025Q3 / Sep 2025).

For H = 5, 7, 10: the recommendation (buy if Delta(H) >= 0) under each setting, the share of
properties whose decision flips between (a) and (c) and between (b) and (c), the direction of
flips, and the shift in break-even year (search 1-30; "never" if no break-even). Confidence
intervals come from a bootstrap that resamples whole states (growth is set per state, so
properties in a state are not independent).

Outputs: artefacts/metrics/decision_flip.json, decision_flip_properties.csv,
         artefacts/figures/decision_flip.png

Run:  python -m src.evaluation.decision_flip
"""
import json
import sys
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, ensure_dirs
from src.engine.rent_vs_buy import DISCLAIMER, Assumptions, analyse
from src.models.train_price_models import DATASETS
from src.pipeline.build_panel import load_cpi_rent, load_hpi

HORIZONS = [5, 7, 10]
SETTINGS = ["fixed", "historical", "forecast"]
LABELS = {"fixed": "(a) fixed 3% / 3%", "historical": "(b) 10-year state history",
          "forecast": "(c) forecast"}
COMPARISONS = [("fixed", "forecast"), ("historical", "forecast")]
HIST_FROM_Q, HIST_TO_Q = "2015Q3", "2025Q3"
HIST_FROM_M, HIST_TO_M = "2015-09", "2025-09"
N_BOOT = 2000


# ----------------------------------------------------------------------------- inputs
def test_properties() -> pd.DataFrame:
    spec = DATASETS["sale"]
    df = pd.read_csv(PROCESSED_DIR / spec["file"])
    groups = df.groupby(spec["group_cols"]).ngroup()
    _, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                 .split(df, groups=groups))
    return df.iloc[te].reset_index(drop=True)


def growth_settings(states: list[str]) -> dict:
    hpi = load_hpi().pivot(index="quarter", columns="state", values="hpi")
    g_hist = (hpi.loc[pd.Period(HIST_TO_Q)] / hpi.loc[pd.Period(HIST_FROM_Q)]) ** 0.1 - 1
    rent = load_cpi_rent()
    q_hist = float((rent[pd.Period(HIST_TO_M)] / rent[pd.Period(HIST_FROM_M)]) ** 0.1 - 1)
    fc = json.loads((MODELS_DIR / "forecasts.json").read_text())
    q_fc = fc["rent"]["forecasts"]["US"]["forecast"] / 100
    return {
        "fixed": {s: (0.03, 0.03) for s in states},
        "historical": {s: (float(g_hist[s]), q_hist) for s in states},
        "forecast": {s: (fc["hpi"]["forecasts"][s]["forecast"] / 100, q_fc) for s in states},
        "_meta": {"hpi_method": fc["hpi"]["method"], "rent_method": fc["rent"]["method"],
                  "q_historical": q_hist, "q_forecast": q_fc,
                  "r": fc["mortgage_rate_latest"]["value_pct"] / 100,
                  "r_month": fc["mortgage_rate_latest"]["month"]},
    }


def predict_inputs(props: pd.DataFrame) -> pd.DataFrame:
    sale = joblib.load(MODELS_DIR / "sale_model.joblib")
    meta = json.loads((MODELS_DIR / "sale_model_meta.json").read_text())
    P = np.exp(sale.predict(props[meta["features"]]))
    rent = joblib.load(MODELS_DIR / "rent_model_no_coords.joblib")
    rx = pd.DataFrame({"state": props["state_code"], "bedrooms": props["beds"],
                       "bathrooms": props["baths"], "square_feet": props["living_space"]})
    factor = json.loads((MODELS_DIR / "cpi_rent_factors.json").read_text())["listing_weighted_factor"]
    R = np.exp(rent.predict(rx)) * factor
    return props.assign(P_pred=P, R=R, P_actual=props["price"],
                        outside_rent_training_range=(props["living_space"] > 8000) | (props["beds"] > 9))


# ----------------------------------------------------------------------------- engine runs
def run_one(P: float, R: float, g: float, q: float, r: float) -> tuple:
    res = analyse(P, R, Assumptions(r=r, g=g, q=q, H=max(HORIZONS)))
    deltas = tuple(res["yearly"][H - 1]["delta"] for H in HORIZONS)
    return deltas + (res["break_even_year"],)


def run_chunk(rows: list) -> list:
    return [run_one(*row) for row in rows]


def evaluate(props: pd.DataFrame, settings: dict, price_col: str) -> pd.DataFrame:
    r = settings["_meta"]["r"]
    out = props[["state_code", price_col, "R"]].rename(columns={price_col: "P"}).copy()
    for s in SETTINGS:
        rows = [(P, R, *settings[s][st], r) for P, R, st in zip(out["P"], out["R"], out["state_code"])]
        chunks = [rows[i:i + 500] for i in range(0, len(rows), 500)]
        res = [x for c in Parallel(n_jobs=-1)(delayed(run_chunk)(c) for c in chunks) for x in c]
        res = np.array(res, dtype=object)
        out[f"g_{s}"] = [settings[s][st][0] for st in out["state_code"]]
        out[f"q_{s}"] = [settings[s][st][1] for st in out["state_code"]]
        for j, H in enumerate(HORIZONS):
            out[f"delta_{s}_H{H}"] = res[:, j].astype(float)
            out[f"buy_{s}_H{H}"] = out[f"delta_{s}_H{H}"] >= 0
        out[f"break_even_{s}"] = [np.nan if v is None else v for v in res[:, -1]]
    return out


# ----------------------------------------------------------------------------- summaries
def flip_stats(d: pd.DataFrame, a: str, c: str, H: int) -> dict:
    ba, bc = d[f"buy_{a}_H{H}"], d[f"buy_{c}_H{H}"]
    return {"flip_rate": float((ba != bc).mean()),
            "buy_to_rent": int((ba & ~bc).sum()), "rent_to_buy": int((~ba & bc).sum()),
            "n_flips": int((ba != bc).sum())}


def break_even_shift(d: pd.DataFrame, a: str, c: str) -> dict:
    ea, ec = d[f"break_even_{a}"], d[f"break_even_{c}"]
    both = ea.notna() & ec.notna()
    shift = (ec - ea)[both]
    return {"mean_shift_years": float(shift.mean()) if both.any() else None,
            "median_shift_years": float(shift.median()) if both.any() else None,
            "n_both_break_even": int(both.sum()),
            "never_under_first_only": int((ea.isna() & ec.notna()).sum()),
            "never_under_second_only": int((ea.notna() & ec.isna()).sum()),
            "never_under_both": int((ea.isna() & ec.isna()).sum()),
            "share_later_under_second": float((shift > 0).mean()) if both.any() else None,
            "share_earlier_under_second": float((shift < 0).mean()) if both.any() else None}


def state_bootstrap_ci(d: pd.DataFrame, a: str, c: str, H: int) -> list:
    """95% CI of the overall flip rate, resampling whole states."""
    rng = np.random.default_rng(RANDOM_STATE)
    flips = (d[f"buy_{a}_H{H}"] != d[f"buy_{c}_H{H}"]).groupby(d["state_code"]).agg(["sum", "size"])
    states = flips.index.to_numpy()
    rates = []
    for _ in range(N_BOOT):
        pick = flips.loc[rng.choice(states, len(states), replace=True)]
        rates.append(pick["sum"].sum() / pick["size"].sum())
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def summarise(d: pd.DataFrame, settings: dict) -> dict:
    out = {"n_properties": len(d), "n_states": int(d["state_code"].nunique()), "by_horizon": {}}
    for H in HORIZONS:
        h = {"share_buy": {s: float(d[f"buy_{s}_H{H}"].mean()) for s in SETTINGS}, "comparisons": {}}
        for a, c in COMPARISONS:
            h["comparisons"][f"{a}_vs_{c}"] = {**flip_stats(d, a, c, H),
                                              "flip_rate_95ci_state_bootstrap": state_bootstrap_ci(d, a, c, H)}
        out["by_horizon"][f"H{H}"] = h
    out["break_even"] = {
        "median_year": {s: (float(d[f"break_even_{s}"].median()) if d[f"break_even_{s}"].notna().any() else None)
                        for s in SETTINGS},
        "share_never": {s: float(d[f"break_even_{s}"].isna().mean()) for s in SETTINGS},
        **{f"shift_{a}_to_{c}": break_even_shift(d, a, c) for a, c in COMPARISONS},
    }
    by_state = {}
    for st, g in d.groupby("state_code"):
        row = {"n": len(g), **{f"g_{s}": float(g[f"g_{s}"].iloc[0]) for s in SETTINGS}}
        for H in HORIZONS:
            for a, c in COMPARISONS:
                row[f"flip_{a}_vs_{c}_H{H}"] = float((g[f"buy_{a}_H{H}"] != g[f"buy_{c}_H{H}"]).mean())
            for s in SETTINGS:
                row[f"share_buy_{s}_H{H}"] = float(g[f"buy_{s}_H{H}"].mean())
        for a, c in COMPARISONS:
            row[f"median_break_even_shift_{a}_to_{c}"] = break_even_shift(g, a, c)["median_shift_years"]
        by_state[st] = row
    out["by_state"] = by_state
    return out


# ----------------------------------------------------------------------------- figure
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_FIXED, C_HIST = "#2a78d6", "#eb6834"   # categorical slots 1, 2


def plot(summary: dict, path):
    fig = plt.figure(figsize=(11, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=8.5)
    pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)

    # Panel 1: overall flip rate by H with state-bootstrap 95% CIs.
    x = np.arange(len(HORIZONS))
    w = 0.36
    for k, ((a, c), color) in enumerate(zip(COMPARISONS, (C_FIXED, C_HIST))):
        vals = [summary["by_horizon"][f"H{H}"]["comparisons"][f"{a}_vs_{c}"] for H in HORIZONS]
        rates = [v["flip_rate"] for v in vals]
        lo = [v["flip_rate"] - v["flip_rate_95ci_state_bootstrap"][0] for v in vals]
        hi = [v["flip_rate_95ci_state_bootstrap"][1] - v["flip_rate"] for v in vals]
        pos = x + (k - 0.5) * (w + 0.04)
        ax1.bar(pos, rates, width=w, color=color, label=f"{LABELS[a]} vs {LABELS[c]}")
        ax1.errorbar(pos, rates, yerr=[lo, hi], fmt="none", ecolor=TEXT, elinewidth=1, capsize=3)
        for p_, r_ in zip(pos, rates):
            ax1.text(p_, r_ / 2, f"{r_:.0%}", ha="center", va="center", fontsize=8.5, color="white",
                     fontweight="bold")
    ax1.set_xticks(x, [f"H = {H} years" for H in HORIZONS])
    ax1.yaxis.set_major_formatter(pct)
    ax1.grid(axis="y", color=GRID, linewidth=0.6)
    ax1.set_ylabel("Share of properties whose decision flips", fontsize=9, color=TEXT_2)
    ax1.set_title("Flip rate by holding period", loc="left", fontsize=10.5, fontweight="bold",
                  color=TEXT, pad=18)
    ax1.text(0, 1.01, "Error bars: 95% CI, bootstrap over states", transform=ax1.transAxes,
             fontsize=7.5, color=TEXT_2)
    ax1.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))

    # Panel 2: flip rate by state at H = 7.
    H = 7
    st = pd.DataFrame(summary["by_state"]).T
    st = st.sort_values(f"flip_fixed_vs_forecast_H{H}")
    y = np.arange(len(st))
    ax2.hlines(y, st[[f"flip_fixed_vs_forecast_H{H}", f"flip_historical_vs_forecast_H{H}"]].min(axis=1),
               st[[f"flip_fixed_vs_forecast_H{H}", f"flip_historical_vs_forecast_H{H}"]].max(axis=1),
               color=GRID, linewidth=2, zorder=1)
    ax2.scatter(st[f"flip_fixed_vs_forecast_H{H}"], y, s=36, color=C_FIXED, zorder=3,
                edgecolor=SURFACE, linewidth=1.5, label="(a) fixed vs (c) forecast")
    ax2.scatter(st[f"flip_historical_vs_forecast_H{H}"], y, s=36, color=C_HIST, zorder=3,
                edgecolor=SURFACE, linewidth=1.5, label="(b) history vs (c) forecast")
    ax2.set_yticks(y, [f"{s}  (n={int(n):,})" for s, n in zip(st.index, st["n"])], fontsize=8)
    ax2.set_ylim(-0.7, len(st) - 0.3)
    ax2.xaxis.set_major_formatter(pct)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(axis="x", color=GRID, linewidth=0.6)
    ax2.set_xlabel(f"Share of properties whose decision flips (H = {H})", fontsize=9, color=TEXT_2)
    ax2.set_title(f"Flip rate by state, H = {H}", loc="left", fontsize=10.5, fontweight="bold", color=TEXT)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle(f"Rent-vs-buy decisions that change when growth comes from the forecast "
                 f"({summary['n_properties']:,} test properties, {summary['n_states']} states)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=7, color=TEXT_2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.12)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def print_summary(s: dict, title: str):
    print(f"\n{title}: {s['n_properties']:,} properties, {s['n_states']} states")
    print(f"  {'H':>3}  {'buy (a)':>8}{'buy (b)':>9}{'buy (c)':>9}   {'flip a-c':>9} {'[95% CI]':>15}"
          f"  {'b->r':>5}{'r->b':>6}   {'flip b-c':>9} {'[95% CI]':>15}  {'b->r':>5}{'r->b':>6}")
    for H in HORIZONS:
        h = s["by_horizon"][f"H{H}"]
        sb = h["share_buy"]
        ac, bc = h["comparisons"]["fixed_vs_forecast"], h["comparisons"]["historical_vs_forecast"]
        ci = lambda v: f"[{v[0]:.1%}, {v[1]:.1%}]"
        print(f"  {H:>3}  {sb['fixed']:>8.1%}{sb['historical']:>9.1%}{sb['forecast']:>9.1%}   "
              f"{ac['flip_rate']:>9.1%} {ci(ac['flip_rate_95ci_state_bootstrap']):>15}  "
              f"{ac['buy_to_rent']:>5}{ac['rent_to_buy']:>6}   "
              f"{bc['flip_rate']:>9.1%} {ci(bc['flip_rate_95ci_state_bootstrap']):>15}  "
              f"{bc['buy_to_rent']:>5}{bc['rent_to_buy']:>6}")
    be = s["break_even"]
    print(f"  break-even median year: {be['median_year']}; never: "
          f"{ {k: f'{v:.1%}' for k, v in be['share_never'].items()} }")
    for a, c in COMPARISONS:
        sh = be[f"shift_{a}_to_{c}"]
        print(f"  break-even shift {a} -> {c}: mean {sh['mean_shift_years']:+.2f} y, median "
              f"{sh['median_shift_years']:+.1f} y (n={sh['n_both_break_even']:,}); never only under "
              f"{a}: {sh['never_under_first_only']}, only under {c}: {sh['never_under_second_only']}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ensure_dirs()
    t0 = time.time()
    props = predict_inputs(test_properties())
    states = sorted(props["state_code"].unique())
    settings = growth_settings(states)

    main_run = evaluate(props, settings, "P_pred")
    robust = evaluate(props, settings, "P_actual")
    summary = summarise(main_run, settings)
    summary_actual = summarise(robust, settings)

    meta = settings["_meta"]
    result = {
        "design": {
            "properties": "full Step 3 sale test set (grouped 80/20 split, random_state=42)",
            "price": "sale model prediction (robustness: actual listing price)",
            "rent": "rent_model_no_coords (state, bedrooms, bathrooms, square_feet) x CPI rent "
                    "factor to Sep 2025; sale data has no coordinates",
            "assumptions": "CLAUDE.md defaults; r = latest mortgage rate",
            "r": meta["r"], "r_month": meta["r_month"],
            "settings": {
                "fixed": "g = 3%, q = 3%",
                "historical": f"g = state average annual HPI growth {HIST_FROM_Q}-{HIST_TO_Q}; "
                              f"q = national CPI rent average annual growth {HIST_FROM_M} to "
                              f"{HIST_TO_M} ({meta['q_historical']:.2%})",
                "forecast": f"g = state 4-quarter forecast ({meta['hpi_method']}); q = national "
                            f"12-month forecast ({meta['rent_method']}, {meta['q_forecast']:.2%})",
            },
            "growth_held_constant_over_H": True,
            "recommendation": "buy if Delta(H) = C_R - C_B >= 0",
            "break_even": "smallest year 1-30 with Delta >= 0; NaN/never otherwise",
            "ci": f"95% bootstrap over states ({N_BOOT} resamples)",
            "n_outside_rent_training_range": int(props["outside_rent_training_range"].sum()),
        },
        "inputs_summary": {
            "median_price_pred": float(props["P_pred"].median()),
            "median_rent": float(props["R"].median()),
            "g_by_setting_mean_over_properties": {s: float(main_run[f"g_{s}"].mean()) for s in SETTINGS},
        },
        "results": summary,
        "robustness_actual_price": {k: summary_actual[k] for k in ("by_horizon", "break_even")},
        "disclaimer": DISCLAIMER,
    }
    with open(METRICS_DIR / "decision_flip.json", "w") as f:
        json.dump(result, f, indent=2)
    main_run.to_csv(METRICS_DIR / "decision_flip_properties.csv", index=False)
    plot(summary, FIGURES_DIR / "decision_flip.png")
    print(f"r = {meta['r']:.4%}; q: fixed 3.00%, historical {meta['q_historical']:.2%}, "
          f"forecast {meta['q_forecast']:.2%}; mean g: "
          + ", ".join(f"{s} {main_run[f'g_{s}'].mean():.2%}" for s in SETTINGS))
    print_summary(summary, "MAIN (predicted price)")
    print_summary(summary_actual, "ROBUSTNESS (actual listing price)")
    print(f"\nFinished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
