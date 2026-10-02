"""Macro scenarios: shock the forecast inputs, re-forecast, recompute the rent-vs-buy result.

House price growth g: the HPI gradient-boosting model (gbm_macro, the method selected in
Step 5) is re-run on the state's latest features with the shock applied.

Rent growth q: the selected rent method is ARIMA, which uses no macro inputs and so cannot
respond to a shock. The scenario rent forecast is therefore
    q_scenario = q_ARIMA + (gbm_macro_rent(shocked features) - gbm_macro_rent(latest features)),
i.e. the change the rent gradient-boosting model predicts for the shock, added on top of
the ARIMA base forecast. The general rule for both targets is
    scenario = recommended base forecast + (gbm(shocked) - gbm(base)),
which for HPI is simply gbm(shocked) because gbm_macro is the recommended method.

Mortgage-rate shocks also change the buyer's mortgage rate r by the same amount.

Market-growth cases use the Step 5 forecast band:
    pessimistic = 10th percentile for both g and q, optimistic = 90th percentile for both.

Limitations (state these with the results): tree models respond in steps and do not
extrapolate beyond the range of the training data (2014-2025), so a shock can produce a
small or zero change, and shocks are applied to one quarter's features with everything
else held fixed (no feedback between variables).

Run:  python -m src.engine.scenario     (example property -> JSON)
"""
import json
import sys

import joblib
import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR
from src.engine.rent_vs_buy import DISCLAIMER, Assumptions, analyse

# Shocks are changes to the model features (percentage points; permits_g and gdp_g are
# growth rates in %). Each model only uses the keys among its own features. A change in a
# level also moves its 12-month change by the same amount.
SHOCKS = {
    "rate_rise": {
        "label": "Interest rates +1 pp (mortgage and fed funds)",
        "features": {"mortgage": 1.0, "mortgage_chg12": 1.0, "fed_funds": 1.0},
    },
    "rate_cut": {
        "label": "Interest rates -1 pp (mortgage and fed funds)",
        "features": {"mortgage": -1.0, "mortgage_chg12": -1.0, "fed_funds": -1.0},
    },
    "recession": {
        "label": "Recession: unemployment +2 pp, permits -20%, state GDP -3 pp, rates -0.5 pp",
        "features": {"unemp": 2.0, "unemp_chg12": 2.0, "permits_g": -20.0, "gdp_g": -3.0,
                     "mortgage": -0.5, "mortgage_chg12": -0.5, "fed_funds": -0.5},
    },
    "inflation": {
        "label": "Inflation +2 pp with rates +0.75 pp",
        "features": {"cpi_infl": 2.0, "mortgage": 0.75, "mortgage_chg12": 0.75, "fed_funds": 0.75},
    },
}


class Forecaster:
    """Latest forecasts plus the fitted gbm_macro models, loaded once."""

    def __init__(self):
        self.forecasts = json.loads((MODELS_DIR / "forecasts.json").read_text())
        self.models = {n: joblib.load(MODELS_DIR / f"{n}_forecast_model.joblib") for n in ("hpi", "rent")}
        self.mortgage_rate = self.forecasts["mortgage_rate_latest"]["value_pct"] / 100

    def _series(self, target: str, state: str) -> str:
        return state if target == "hpi" else "US"

    def base(self, target: str, state: str) -> dict:
        f = self.forecasts[target]["forecasts"][self._series(target, state)]
        return {"forecast": f["forecast"], "p10": f["p10"], "p90": f["p90"],
                "method": self.forecasts[target]["method"]}

    def _gbm(self, target: str, state: str, shock: dict | None) -> float:
        b = self.models[target]
        rows = pd.DataFrame(b["latest_features"])
        row = rows[rows["series"] == self._series(target, state)].copy()
        if row.empty:
            raise KeyError(f"no forecast features for {state}")
        for col, change in (shock or {}).items():
            if col in row.columns and col != "series":
                row[col] = row[col] + change
        if "series" in b["features"]:
            row["series"] = pd.Categorical(row["series"], categories=sorted(rows["series"].unique()))
        return float(b["model"].predict(row[b["features"]])[0])

    def shocked(self, target: str, state: str, shock: dict) -> dict:
        base = self.base(target, state)
        gbm_base, gbm_shock = self._gbm(target, state, None), self._gbm(target, state, shock)
        return {"forecast": base["forecast"] + (gbm_shock - gbm_base),
                "gbm_change": gbm_shock - gbm_base, "base_method": base["method"]}


def run_scenarios(P: float, R: float, state: str, a: Assumptions | None = None,
                  forecaster: Forecaster | None = None, custom_shock: dict | None = None) -> dict:
    """Base, pessimistic, optimistic and macro-shock cases for one property."""
    fc = forecaster or Forecaster()
    g0, q0 = fc.base("hpi", state), fc.base("rent", state)
    a = a or Assumptions(r=fc.mortgage_rate, g=g0["forecast"] / 100, q=q0["forecast"] / 100)

    def case(name, label, g, q, r, extra=None):
        res = analyse(P, R, a.replace(g=g, q=q, r=r))
        return {"scenario": name, "label": label, "g": g, "q": q, "r": r,
                "delta": res["delta"], "recommendation": res["recommendation"],
                "break_even_year": res["break_even_year"], **(extra or {})}

    cases = [
        case("base", "Base forecast", a.g, a.q, a.r),
        case("pessimistic", "Pessimistic market: 10th percentile growth (g and q)",
             g0["p10"] / 100, q0["p10"] / 100, a.r),
        case("optimistic", "Optimistic market: 90th percentile growth (g and q)",
             g0["p90"] / 100, q0["p90"] / 100, a.r),
    ]
    shocks = dict(SHOCKS)
    if custom_shock:
        shocks["custom"] = {"label": "Custom shock", "features": custom_shock}
    for name, spec in shocks.items():
        hs = fc.shocked("hpi", state, spec["features"])
        rs = fc.shocked("rent", state, spec["features"])
        dg, dq = hs["gbm_change"] / 100, rs["gbm_change"] / 100
        dr = spec["features"].get("mortgage", 0.0) / 100
        cases.append(case(name, spec["label"], a.g + dg, a.q + dq, max(a.r + dr, 0.0),
                          {"shock": spec["features"], "g_change": dg, "q_change": dq, "r_change": dr}))
    base = cases[0]
    for c in cases:
        c["flips_vs_base"] = c["recommendation"] != base["recommendation"]
        c["delta_change_vs_base"] = c["delta"] - base["delta"]
    return {
        "state": state, "price": P, "monthly_rent": R, "H": a.H,
        "method": {"g": f"{g0['method']} (HPI); shocks re-run gbm_macro",
                   "q": f"{q0['method']} base + change predicted by the rent gbm_macro model",
                   "bands": "pessimistic / optimistic = forecast p10 / p90"},
        "cases": cases,
        "disclaimer": DISCLAIMER,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from src.engine.example import example_case
    ex = example_case()
    res = run_scenarios(ex["P"], ex["R"], ex["state"], ex["assumptions"])
    res["example"] = ex["description"]
    with open(METRICS_DIR / "scenario_example.json", "w") as f:
        json.dump(res, f, indent=2)
    print(ex["description"])
    print(f"  {'scenario':<13}{'g':>8}{'q':>8}{'r':>8}{'Delta(H)':>12}  rec   break-even")
    for c in res["cases"]:
        print(f"  {c['scenario']:<13}{c['g']:>8.2%}{c['q']:>8.2%}{c['r']:>8.2%}{c['delta']:>12,.0f}"
              f"  {c['recommendation']:<5} {c['break_even_year']}"
              f"{'   <- flips' if c['flips_vs_base'] else ''}")


if __name__ == "__main__":
    main()
