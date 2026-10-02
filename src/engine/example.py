"""Worked example used for the Step 6 figures (cost over time, tornado, scenarios).

A 3-bed, 2-bath, 1,800 sq ft house in Austin, Texas:
- price P from the sale model; D1 has no ZIP lookup yet (Step 8), so the ZIP density and
  income inputs are the Texas medians from the cleaned sale data;
- monthly rent R from the rent model at the most common Austin coordinates in the rent
  data, brought from 2019 to Sep 2025 with the CPI rent factor;
- g and q from the latest forecasts, r = latest mortgage rate (Sep 2025), other
  assumptions at the CLAUDE.md defaults.
It illustrates the engine; it is not a valuation of a real property.

Run:  python -m src.engine.example     (-> artefacts/figures/cost_over_time_example.png)
"""
import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, PROCESSED_DIR
from src.engine.rent_vs_buy import DISCLAIMER, Assumptions, analyse

STATE, CITY = "TX", "Austin"
BEDS, BATHS, SQFT = 3, 2, 1_800


def example_case() -> dict:
    sale = pd.read_csv(PROCESSED_DIR / "sale_clean.csv")
    tx = sale[sale["state_code"] == STATE]
    sale_x = pd.DataFrame([{
        "state_code": STATE, "beds": BEDS, "baths": BATHS, "living_space": SQFT,
        "zip_code_density": tx["zip_code_density"].median(),
        "median_household_income": tx["median_household_income"].median(),
    }])
    P = float(np.exp(joblib.load(MODELS_DIR / "sale_model.joblib").predict(sale_x)[0]))

    rent = pd.read_csv(PROCESSED_DIR / "rent_clean.csv")
    lat, lon = (rent[(rent["state"] == STATE) & (rent["city"] == CITY)]
                .groupby(["latitude", "longitude"]).size().idxmax())
    rent_x = pd.DataFrame([{"state": STATE, "bedrooms": BEDS, "bathrooms": BATHS,
                            "square_feet": SQFT, "latitude": lat, "longitude": lon}])
    R_2019 = float(np.exp(joblib.load(MODELS_DIR / "rent_model.joblib").predict(rent_x)[0]))
    factors = json.loads((MODELS_DIR / "cpi_rent_factors.json").read_text())
    R = R_2019 * factors["listing_weighted_factor"]

    fc = json.loads((MODELS_DIR / "forecasts.json").read_text())
    g = fc["hpi"]["forecasts"][STATE]
    q = fc["rent"]["forecasts"]["US"]
    a = Assumptions(r=fc["mortgage_rate_latest"]["value_pct"] / 100,
                    g=g["forecast"] / 100, q=q["forecast"] / 100)
    return {
        "P": P, "R": R, "R_2019": R_2019, "state": STATE, "assumptions": a,
        "g_band": (g["p10"] / 100, g["p90"] / 100), "q_band": (q["p10"] / 100, q["p90"] / 100),
        "short": f"{BEDS}-bed {SQFT:,} sq ft house, {CITY} {STATE}",
        "description": (f"{BEDS}-bed, {BATHS}-bath, {SQFT:,} sq ft house in {CITY}, {STATE}: "
                        f"predicted price ${P:,.0f}; predicted rent ${R:,.0f}/month "
                        f"(${R_2019:,.0f} in 2019 x CPI factor {factors['listing_weighted_factor']:.3f} "
                        f"to {factors['reference_month']}); g {a.g:.2%} ({fc['hpi']['method']}), "
                        f"q {a.q:.2%} ({fc['rent']['method']}), r {a.r:.2%} "
                        f"({fc['mortgage_rate_latest']['month']}). Illustrative, not a valuation."),
    }


SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BUY, RENT = "#2a78d6", "#eb6834"   # categorical slots 1 and 2


def plot_cost_over_time(res: dict, title: str, path):
    rows = res["yearly"]
    t = [r["year"] for r in rows]
    buy = [r["pv_cost_buy"] for r in rows]
    rent = [r["pv_cost_rent"] for r in rows]
    H, be = res["inputs"]["H"], res["break_even_year"]

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(t, buy, color=BUY, linewidth=2, label="Buy: C_B(t)")
    ax.plot(t, rent, color=RENT, linewidth=2, label="Rent: C_R(t)")
    ax.text(t[-1] + 0.4, buy[-1], "Buy", color=TEXT, fontsize=9, va="center")
    ax.text(t[-1] + 0.4, rent[-1], "Rent", color=TEXT, fontsize=9, va="center")
    ax.axvline(H, color=TEXT_2, linewidth=1, linestyle=":")
    ax.text(H, ax.get_ylim()[1], f" holding period H = {H}", fontsize=8, color=TEXT_2, va="top")
    if be is not None:
        ax.scatter([be], [rent[be - 1]], s=60, color=TEXT, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"break-even: year {be}", (be, rent[be - 1]), xytext=(10, -22),
                    textcoords="offset points", fontsize=8.5, color=TEXT)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_xlim(0.5, t[-1] + 2)
    ax.set_xlabel("Years held (t): sell / stop renting at the end of year t", fontsize=8.5, color=TEXT_2)
    ax.set_ylabel("Present value of net cost", fontsize=8.5, color=TEXT_2)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TEXT)
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=7, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ex = example_case()
    res = analyse(ex["P"], ex["R"], ex["assumptions"])
    plot_cost_over_time(res, f"Cost of buying vs renting over time: {ex['short']}",
                        FIGURES_DIR / "cost_over_time_example.png")
    out = {k: v for k, v in res.items() if k != "yearly"}
    out["example"] = ex["description"]
    out["yearly"] = res["yearly"]
    with open(METRICS_DIR / "engine_example.json", "w") as f:
        json.dump(out, f, indent=2)
    print(ex["description"])
    print(f"M ${res['monthly_payment']:,.0f}/month | C_B(H) ${res['cost_buy']:,.0f} | "
          f"C_R(H) ${res['cost_rent']:,.0f} | Delta ${res['delta']:,.0f} -> {res['recommendation']} | "
          f"break-even {res['break_even_year']} | user cost ${res['user_cost']:,.0f} vs rent "
          f"${res['annual_rent']:,.0f} -> {res['user_cost_says']}")


if __name__ == "__main__":
    main()
