"""Exploratory data analysis of the cleaned sale (D1) and rent (D2) data.

Figures (artefacts/figures/):
  {sale,rent}_distributions.png   histograms of every model variable
  {sale,rent}_correlation.png     Spearman correlation matrix
  {sale,rent}_price_by_state.png  price / rent box plot per state, sorted by median
Table (artefacts/metrics/eda_summary.json):
  count, mean, median, std (plus min/max) per variable, and count/median per state.

Spearman (rank) correlation is used because price, rent, area and density are strongly
right-skewed; it measures monotonic association and is unaffected by the log transform.

Run:  python -m src.pipeline.eda   (after clean_sale and clean_rent)
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator

from src.config import FIGURES_DIR, METRICS_DIR, PROCESSED_DIR, ensure_dirs

# Reference palette (light mode): one hue for magnitude, blue<->gray<->red for polarity.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_DARK = "#104281"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#f0efec", "#e34948", "#9e2a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "figure.dpi": 100, "savefig.dpi": 200,
})

DOLLARS = FuncFormatter(lambda v, _: f"${v/1e6:g}M" if v >= 1e6 else
                        (f"${v/1e3:g}k" if v >= 1e3 else f"${v:g}"))
THOUSANDS = FuncFormatter(lambda v, _: f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}")


def log_axis(axis, formatter, lo, hi):
    """Log-scale ticks: 1-2-5 steps over a narrow range, powers of ten over a wide one."""
    subs = (1,) if np.log10(hi / lo) > 3 else (1, 2, 5)
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(formatter)
    axis.set_minor_formatter(FuncFormatter(lambda *_: ""))

SALE = {
    "name": "sale",
    "file": "sale_clean.csv",
    "target": "price",
    "target_label": "Sale price",
    "state_col": "state_code",
    "hist": [  # column, label, log-x, integer (bar chart)
        ("price", "Sale price (USD, log scale)", True, False),
        ("living_space", "Living area (sq ft, log scale)", True, False),
        ("beds", "Bedrooms", False, True),
        ("baths", "Bathrooms", False, True),
        ("zip_code_density", "ZIP population density (per sq mi, log scale)", True, False),
        ("median_household_income", "ZIP median household income (USD)", False, False),
    ],
    "corr": ["price", "living_space", "beds", "baths", "zip_code_density",
             "median_household_income"],
}
RENT = {
    "name": "rent",
    "file": "rent_clean.csv",
    "target": "rent",
    "target_label": "Monthly rent",
    "state_col": "state",
    "hist": [
        ("rent", "Monthly rent (USD, log scale)", True, False),
        ("square_feet", "Floor area (sq ft, log scale)", True, False),
        ("bedrooms", "Bedrooms (0 = studio)", False, True),
        ("bathrooms", "Bathrooms", False, True),
    ],
    "corr": ["rent", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"],
}


def histograms(df: pd.DataFrame, spec: dict):
    items = spec["hist"]
    ncols = 3 if len(items) > 4 else 2
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    for ax, (col, label, log_x, integer) in zip(axes.flat, items):
        s = df[col].dropna()
        if integer:
            counts = s.value_counts().sort_index()
            width = 0.8 if (counts.index % 1 == 0).all() else 0.4
            ax.bar(counts.index, counts.values, width=width, color=BLUE,
                   edgecolor=SURFACE, linewidth=1)
            ax.set_xticks(counts.index)
            ax.tick_params(axis="x", labelrotation=90 if len(counts) > 10 else 0)
            ax.grid(axis="x", visible=False)
        else:
            bins = np.geomspace(s.min(), s.max(), 50) if log_x else 50
            ax.hist(s, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
            if log_x:
                ax.set_xscale("log")
            med = s.median()
            ax.axvline(med, color=BLUE_DARK, linewidth=1.5, linestyle="--")
            money = col in (spec["target"], "median_household_income")
            fmt = DOLLARS if money else THOUSANDS
            ax.text(med, ax.get_ylim()[1] * 0.97, f" median {'$' if money else ''}{med:,.0f}",
                    color=TEXT, fontsize=8, va="top")
            if log_x:
                log_axis(ax.xaxis, fmt, s.min(), s.max())
            else:
                ax.xaxis.set_major_formatter(fmt)
            ax.grid(axis="x", visible=False)
        ax.set_xlabel(label)
        ax.set_ylabel("Listings")
        ax.yaxis.set_major_formatter(THOUSANDS)
    for ax in list(axes.flat)[len(items):]:
        ax.set_visible(False)
    fig.suptitle(f"{spec['name'].capitalize()} data: distributions (n = {len(df):,})",
                 fontsize=13, fontweight="bold", color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{spec['name']}_distributions.png")
    plt.close(fig)


def correlation(df: pd.DataFrame, spec: dict) -> dict:
    corr = df[spec["corr"]].corr(method="spearman")
    n = len(corr)
    fig, ax = plt.subplots(figsize=(1.1 * n + 2.2, 1.0 * n + 1.2))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(n), corr.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    ax.set_title(f"{spec['name'].capitalize()} data: Spearman correlation", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{spec['name']}_correlation.png")
    plt.close(fig)
    return {k: {kk: round(float(vv), 4) for kk, vv in row.items()}
            for k, row in corr.to_dict(orient="index").items()}


def by_state(df: pd.DataFrame, spec: dict) -> dict:
    target, state = spec["target"], spec["state_col"]
    groups = df.groupby(state)[target]
    order = groups.median().sort_values().index.tolist()
    data = [df.loc[df[state] == s, target].values for s in order]
    counts = groups.size()

    fig, ax = plt.subplots(figsize=(8, 0.26 * len(order) + 1.4))
    bp = ax.boxplot(data, orientation="horizontal", showfliers=False, widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor="#b7d3f6", edgecolor=BLUE_DARK, linewidth=0.8),
                    medianprops=dict(color=BLUE_DARK, linewidth=2),
                    whiskerprops=dict(color=TEXT_2, linewidth=0.8),
                    capprops=dict(color=TEXT_2, linewidth=0.8))
    ax.set_yticks(range(1, len(order) + 1), [f"{s}  (n={counts[s]:,})" for s in order])
    ax.set_xscale("log")
    log_axis(ax.xaxis, DOLLARS, df[target].min(), df[target].max())
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{spec['target_label']} (USD, log scale). Box = IQR, line = median, "
                  "whiskers = 1.5 x IQR, outliers not drawn")
    ax.set_title(f"{spec['target_label']} by state, sorted by median", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{spec['name']}_price_by_state.png")
    plt.close(fig)
    return {s: {"count": int(counts[s]), "median": float(groups.median()[s])}
            for s in reversed(order)}


def summary_table(df: pd.DataFrame, columns: list[str]) -> dict:
    out = {}
    for col in columns:
        s = df[col].dropna()
        out[col] = {"count": int(s.count()), "mean": round(float(s.mean()), 4),
                    "median": round(float(s.median()), 4), "std": round(float(s.std()), 4),
                    "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)}
    return out


def run(spec: dict) -> dict:
    df = pd.read_csv(PROCESSED_DIR / spec["file"])
    histograms(df, spec)
    corr = correlation(df, spec)
    states = by_state(df, spec)
    return {"rows": len(df), "n_states": len(states),
            "summary": summary_table(df, spec["corr"]),
            "spearman_correlation": corr,
            "by_state": states}


def print_table(name: str, summary: dict):
    print(f"\n{name.upper()}")
    print(f"  {'variable':<25}{'count':>8}{'mean':>14}{'median':>14}{'std':>14}")
    for col, v in summary.items():
        print(f"  {col:<25}{v['count']:>8,}{v['mean']:>14,.1f}{v['median']:>14,.1f}{v['std']:>14,.1f}")


def main():
    ensure_dirs()
    results = {"sale": run(SALE), "rent": run(RENT),
               "notes": "Correlation is Spearman (rank). Rent coordinates are listing "
                        "lat/long; sale data has no coordinates."}
    with open(METRICS_DIR / "eda_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    for name in ("sale", "rent"):
        print_table(name, results[name]["summary"])
    print(f"\nSaved 6 figures to {FIGURES_DIR} and {METRICS_DIR / 'eda_summary.json'}")


if __name__ == "__main__":
    main()
