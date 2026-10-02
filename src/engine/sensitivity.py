"""One-at-a-time sensitivity of the rent-vs-buy result (data for a tornado chart).

Each assumption is moved to a low and a high value while everything else stays at the base
case, and Delta(H) = C_R - C_B and the break-even year are recomputed. Bars are sorted by
swing |Delta(high) - Delta(low)|.

Default ranges are tied to the evidence where there is some:
- g and q: the 10th / 90th percentile forecast band (Step 5), when supplied;
- P and R: +/- the test MAPE of the sale and rent models (Steps 3-4), i.e. how wrong the
  predicted price and rent typically are;
- the rest: plausible fixed ranges around the CLAUDE.md defaults.

Run:  python -m src.engine.sensitivity     (example property -> JSON + tornado figure)
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, METRICS_DIR
from src.engine.rent_vs_buy import DISCLAIMER, Assumptions, break_even_year, delta, validate

LABELS = {
    "g": "House price growth g", "q": "Rent growth q", "r": "Mortgage rate r",
    "k": "Discount rate k", "d": "Down payment d", "tau": "Property tax tau",
    "m": "Maintenance m", "h": "Insurance h", "cb": "Buying costs cb", "cs": "Selling costs cs",
    "H": "Holding period H", "P": "Price P", "R": "Monthly rent R",
}


def model_errors() -> dict:
    """Test-set MAPE of the selected sale and rent models, if Steps 3-4 have been run."""
    out = {}
    for name in ("sale", "rent"):
        path = METRICS_DIR / f"{name}_models.json"
        if path.exists():
            d = json.loads(path.read_text())
            out[name] = d["candidates"][d["selected_model"]]["test"]["mape_usd"]
    return out


def default_ranges(P: float, R: float, a: Assumptions, g_band=None, q_band=None) -> dict:
    err = model_errors()
    pe, re_ = err.get("sale", 0.20), err.get("rent", 0.15)
    return {
        "g": g_band or (a.g - 0.02, a.g + 0.02),
        "q": q_band or (a.q - 0.02, a.q + 0.02),
        "r": (max(a.r - 0.01, 0.0), a.r + 0.01),
        "k": (max(a.k - 0.01, 0.0), a.k + 0.01),
        "d": (max(a.d - 0.10, 0.0), min(a.d + 0.10, 1.0)),
        "tau": (max(a.tau - 0.005, 0.0), a.tau + 0.005),
        "m": (max(a.m - 0.005, 0.0), a.m + 0.005),
        "h": (a.h * 0.5, a.h * 1.5),
        "cb": (max(a.cb - 0.01, 0.0), a.cb + 0.01),
        "cs": (max(a.cs - 0.02, 0.0), a.cs + 0.02),
        "H": (max(a.H - 3, 1), a.H + 3),
        "P": (P * (1 - pe), P * (1 + pe)),
        "R": (R * (1 - re_), R * (1 + re_)),
    }


def _evaluate(P, R, a, name, v):
    if name == "P":
        P = v
    elif name == "R":
        R = v
    else:
        a = a.replace(**{name: int(round(v)) if name == "H" else v})
    validate(P, R, a)
    return delta(P, R, a, a.H), break_even_year(P, R, a)


def tornado(P: float, R: float, a: Assumptions, ranges: dict | None = None,
            g_band=None, q_band=None) -> dict:
    validate(P, R, a)
    ranges = ranges or default_ranges(P, R, a, g_band, q_band)
    base_delta = delta(P, R, a, a.H)
    bars = []
    for name, (low, high) in ranges.items():
        d_lo, be_lo = _evaluate(P, R, a, name, low)
        d_hi, be_hi = _evaluate(P, R, a, name, high)
        bars.append({"parameter": name, "label": LABELS[name], "low": low, "high": high,
                     "delta_low": d_lo, "delta_high": d_hi, "swing": abs(d_hi - d_lo),
                     "break_even_low": be_lo, "break_even_high": be_hi,
                     "flips_recommendation": (d_lo >= 0) != (d_hi >= 0)})
    bars.sort(key=lambda b: -b["swing"])
    return {"base_delta": base_delta, "base_recommendation": "buy" if base_delta >= 0 else "rent",
            "base_break_even": break_even_year(P, R, a), "H": a.H, "bars": bars,
            "disclaimer": DISCLAIMER}


# ----------------------------------------------------------------------------- figure
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LOW_COLOR, HIGH_COLOR = "#2a78d6", "#eb6834"   # categorical slots 1 and 2


def _fmt_value(name, v):
    if name in ("P", "R"):
        return f"${v:,.0f}"
    if name == "H":
        return f"{int(round(v))} y"
    return f"{v * 100:.2f}%"


def plot_tornado(result: dict, title: str, path):
    bars = result["bars"][::-1]
    base = result["base_delta"]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(bars) + 1.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    values = [v for b in bars for v in (b["delta_low"], b["delta_high"])] + [base, 0.0]
    offset = 0.012 * (max(values) - min(values))
    for i, b in enumerate(bars):
        for val, color in ((b["delta_low"], LOW_COLOR), (b["delta_high"], HIGH_COLOR)):
            ax.barh(i, val - base, left=base, height=0.62, color=color, edgecolor=SURFACE, linewidth=1)
        for val, raw in ((b["delta_low"], b["low"]), (b["delta_high"], b["high"])):
            right = val >= base
            ax.text(val + (offset if right else -offset), i, _fmt_value(b["parameter"], raw),
                    va="center", ha="left" if right else "right", fontsize=7.5, color=TEXT_2,
                    zorder=5, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.6))
    ax.axvline(base, color=TEXT, linewidth=1)
    ax.axvline(0, color=TEXT_2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(bars)), [b["label"] for b in bars], fontsize=8.5)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{'-' if v < 0 else ''}${abs(v) / 1000:,.0f}k"))
    lo, hi = ax.get_xlim()
    pad = (hi - lo) * 0.12
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel(f"Delta(H={result['H']}) = PV cost of renting - PV cost of buying "
                  "(positive favours buying)", fontsize=8.5, color=TEXT_2)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LOW_COLOR),
                       plt.Rectangle((0, 0), 1, 1, color=HIGH_COLOR),
                       plt.Line2D([0], [0], color=TEXT), plt.Line2D([0], [0], color=TEXT_2, ls="--")],
              labels=["Low value", "High value", f"Base case (${base / 1000:,.1f}k)", "Break-even (0)"],
              frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TEXT)
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=7, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from src.engine.example import example_case
    ex = example_case()
    res = tornado(ex["P"], ex["R"], ex["assumptions"], g_band=ex["g_band"], q_band=ex["q_band"])
    res["example"] = ex["description"]
    with open(METRICS_DIR / "sensitivity_example.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_tornado(res, f"Sensitivity of the rent-vs-buy result: {ex['short']}",
                 FIGURES_DIR / "tornado_sensitivity.png")
    print(f"Base Delta(H={res['H']}) = {res['base_delta']:,.0f} ({res['base_recommendation']}), "
          f"break-even year {res['base_break_even']}")
    for b in res["bars"]:
        flag = "  <- flips the recommendation" if b["flips_recommendation"] else ""
        print(f"  {b['label']:<24} {_fmt_value(b['parameter'], b['low']):>10} -> {b['delta_low']:>11,.0f}"
              f"   {_fmt_value(b['parameter'], b['high']):>10} -> {b['delta_high']:>11,.0f}{flag}")


if __name__ == "__main__":
    main()
