"""Analysis service behind the API: loads every model and table once, then answers requests.

ZIP lookup: data/processed/zip_lookup.csv (Census 2020 ZCTA + ACS 2020-2024). A ZIP that is
not in the lookup, or a missing field, falls back to the state's population-weighted
median (the caller must then supply the state). Every fallback is reported in the result.

Price: the sale model, fed the ZIP's density and MEAN household income (D1's income column
matches ACS mean, not median; see artefacts/metrics/zip_lookup_comparison.json).
Rent: the rent model with coordinates; the no-coordinates model if coordinates are missing.
The 2019 rent is brought to Sep 2025 with the CPI rent factor (the rent forecast origin).
Growth: g = the state's HPI forecast, q = the national rent forecast, r = latest mortgage rate.
"""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DIR
from src.engine.rent_vs_buy import DISCLAIMER, Assumptions, analyse
from src.engine.scenario import Forecaster, run_scenarios
from src.engine.sensitivity import tornado
from src.models.explain import top_contributions
from src.pipeline.build_zip_lookup import SALE_MODEL_INCOME

RENT_TRAINING_MAX_SQFT, RENT_TRAINING_MAX_BEDS = 8_000, 9


class NotFound(LookupError):
    pass


class AnalysisService:
    def __init__(self):
        self.sale_model = joblib.load(MODELS_DIR / "sale_model.joblib")
        self.sale_meta = json.loads((MODELS_DIR / "sale_model_meta.json").read_text())
        self.rent_model = joblib.load(MODELS_DIR / "rent_model.joblib")
        self.rent_model_no_coords = joblib.load(MODELS_DIR / "rent_model_no_coords.joblib")
        self.cpi = json.loads((MODELS_DIR / "cpi_rent_factors.json").read_text())
        self.forecaster = Forecaster()
        zips = pd.read_csv(PROCESSED_DIR / "zip_lookup.csv", dtype={"zip": str})
        self.zips = zips.set_index("zip")
        self.state_medians = pd.read_csv(PROCESSED_DIR / "zip_lookup_state_medians.csv").set_index("state")
        sale = pd.read_csv(PROCESSED_DIR / "sale_clean.csv", usecols=["state_code"])
        self.sale_states = set(sale["state_code"].unique())
        self.model_mape = {}
        for name in ("sale", "rent"):
            m = json.loads((METRICS_DIR / f"{name}_models.json").read_text())
            self.model_mape[name] = m["candidates"][m["selected_model"]]["test"]["mape_usd"]

    # ------------------------------------------------------------------ location
    def locate(self, zip_code: str, state: str | None) -> tuple[dict, list[str]]:
        warnings = []
        fields = ["lat", "lon", "density", "median_income", "mean_income"]
        if zip_code in self.zips.index:
            row = self.zips.loc[zip_code]
            zip_state = row["state"]
            if state and state != zip_state:
                warnings.append(f"ZIP {zip_code} is in {zip_state}, not {state}; using {zip_state}.")
            st = zip_state
            loc = {"zip": zip_code, "state": st, "found": True}
            source = {}
            for f in fields:
                v = row[f]
                if pd.isna(v) or (f == "density" and v <= 0):
                    loc[f], source[f] = float(self.state_medians.loc[st, f]), "state_median"
                else:
                    loc[f], source[f] = float(v), "zip"
        else:
            if not state:
                raise NotFound(f"ZIP {zip_code} is not in the lookup; please also give the state.")
            if state not in self.state_medians.index:
                raise NotFound(f"unknown state {state}")
            st = state
            loc = {"zip": zip_code, "state": st, "found": False,
                   **{f: float(self.state_medians.loc[st, f]) for f in fields}}
            source = {f: "state_median" for f in fields}
        if any(s == "state_median" for s in source.values()):
            fell_back = [f for f, s in source.items() if s == "state_median"]
            warnings.append(f"State median used for: {', '.join(fell_back)}.")
        loc["source"] = source
        return loc, warnings

    # ------------------------------------------------------------------ predictions
    def predict_price(self, loc: dict, beds, baths, sqft) -> dict:
        X = pd.DataFrame([{"state_code": loc["state"], "beds": beds, "baths": baths,
                           "living_space": sqft, "zip_code_density": loc["density"],
                           "median_household_income": loc[SALE_MODEL_INCOME]}])[self.sale_meta["features"]]
        exp = top_contributions(self.sale_model, X)
        return {"predicted": exp["prediction"], "model": self.sale_meta["model"],
                "test_mape": self.model_mape["sale"], "explanation": exp}

    def predict_rent(self, loc: dict, beds, baths, sqft) -> dict:
        X = pd.DataFrame([{"state": loc["state"], "bedrooms": beds, "bathrooms": baths,
                           "square_feet": sqft, "latitude": loc["lat"], "longitude": loc["lon"]}])
        model, name = self.rent_model, "rent_model"
        if any(pd.isna(loc.get(c)) for c in ("lat", "lon")):
            X, model, name = X.drop(columns=["latitude", "longitude"]), self.rent_model_no_coords, "rent_model_no_coords"
        exp = top_contributions(model, X)
        factor = self.cpi["listing_weighted_factor"]
        return {"predicted_2019": exp["prediction"], "cpi_factor": factor,
                "cpi_reference_month": self.cpi["reference_month"],
                "predicted": exp["prediction"] * factor, "model": name,
                "test_mape": self.model_mape["rent"], "explanation": exp}

    def forecasts_for(self, state: str) -> dict:
        g, q = self.forecaster.base("hpi", state), self.forecaster.base("rent", state)
        f = self.forecaster.forecasts
        return {
            "g": {"value": g["forecast"] / 100, "p10": g["p10"] / 100, "p90": g["p90"] / 100,
                  "method": g["method"], "origin": f["hpi"]["origin"], "geography": state},
            "q": {"value": q["forecast"] / 100, "p10": q["p10"] / 100, "p90": q["p90"] / 100,
                  "method": q["method"], "origin": f["rent"]["origin"], "geography": "US"},
            "r": {"value": self.forecaster.mortgage_rate, "month": f["mortgage_rate_latest"]["month"]},
        }

    # ------------------------------------------------------------------ requests
    def _prepare(self, prop: dict) -> dict:
        """Location, price, rent, forecasts and assumptions for one property request."""
        loc, warnings = self.locate(prop["zip"], prop.get("state"))
        beds, baths, sqft = prop["beds"], prop["baths"], prop["sqft"]
        price = self.predict_price(loc, beds, baths, sqft)
        rent = self.predict_rent(loc, beds, baths, sqft)
        if loc["state"] not in self.sale_states:
            warnings.append(f"The price model had no {loc['state']} listings in training; "
                            "the price estimate is less reliable.")
        if sqft > RENT_TRAINING_MAX_SQFT or beds > RENT_TRAINING_MAX_BEDS:
            warnings.append("The rent model was trained on apartments up to 8,000 sq ft and 9 bedrooms; "
                            "this rent estimate is an extrapolation.")
        P = prop.get("price") or price["predicted"]
        R = prop.get("monthly_rent") or rent["predicted"]
        price["used"], price["source"] = P, "user" if prop.get("price") else "model"
        rent["used"], rent["source"] = R, "user" if prop.get("monthly_rent") else "model"
        fc = self.forecasts_for(loc["state"])
        user_a = {k: v for k, v in (prop.get("assumptions") or {}).items() if v is not None}
        a = Assumptions(**{"r": fc["r"]["value"], "g": fc["g"]["value"], "q": fc["q"]["value"], **user_a})
        assumption_source = {k: ("user" if k in user_a else "forecast" if k in ("g", "q")
                                 else "latest_data" if k == "r" else "default")
                             for k in ("r", "g", "q", "d", "T", "H", "tau", "m", "h", "cb", "cs", "k")}
        return {"loc": loc, "price": price, "rent": rent, "forecast": fc, "a": a,
                "assumption_source": assumption_source, "warnings": warnings, "P": P, "R": R}

    def analyse(self, prop: dict) -> dict:
        x = self._prepare(prop)
        result = analyse(x["P"], x["R"], x["a"])
        return {
            "property": {k: prop.get(k) for k in ("zip", "state", "beds", "baths", "sqft")},
            "location": x["loc"], "price": x["price"], "rent": x["rent"], "forecast": x["forecast"],
            "assumptions": {**result["inputs"], "source": x["assumption_source"]},
            "result": {k: v for k, v in result.items() if k not in ("inputs", "disclaimer")},
            "warnings": x["warnings"], "disclaimer": DISCLAIMER,
        }

    def compare(self, props: list[dict]) -> dict:
        rows = []
        for i, p in enumerate(props):
            r = self.analyse(p)
            rows.append({"index": i, "zip": p["zip"], "state": r["location"]["state"],
                         "beds": p["beds"], "baths": p["baths"], "sqft": p["sqft"],
                         "price": r["price"]["used"], "monthly_rent": r["rent"]["used"],
                         "g": r["assumptions"]["g"], "q": r["assumptions"]["q"],
                         "monthly_payment": r["result"]["monthly_payment"],
                         "delta": r["result"]["delta"], "recommendation": r["result"]["recommendation"],
                         "break_even_year": r["result"]["break_even_year"],
                         "warnings": r["warnings"]})
        best = max(rows, key=lambda r: r["delta"])
        return {"properties": rows, "largest_buy_advantage_index": best["index"],
                "note": "Delta = present-value cost of renting minus buying over H years; "
                        "larger favours buying.", "disclaimer": DISCLAIMER}

    def sensitivity(self, prop: dict) -> dict:
        x = self._prepare(prop)
        fc = x["forecast"]
        res = tornado(x["P"], x["R"], x["a"], g_band=(fc["g"]["p10"], fc["g"]["p90"]),
                      q_band=(fc["q"]["p10"], fc["q"]["p90"]))
        res.update({"price": x["P"], "monthly_rent": x["R"], "state": x["loc"]["state"],
                    "warnings": x["warnings"]})
        return res

    def scenario(self, prop: dict, custom_shock: dict | None = None) -> dict:
        x = self._prepare(prop)
        res = run_scenarios(x["P"], x["R"], x["loc"]["state"], x["a"], self.forecaster,
                            custom_shock=custom_shock)
        res["warnings"] = x["warnings"]
        return res

    def forecast(self, state: str) -> dict:
        f = self.forecaster.forecasts
        if state not in f["hpi"]["forecasts"]:
            raise NotFound(f"no forecast for state {state}")
        h, r = f["hpi"]["forecasts"][state], f["rent"]["forecasts"]["US"]
        return {
            "state": state,
            "house_price_growth": {"forecast_pct": h["forecast"], "p10_pct": h["p10"], "p90_pct": h["p90"],
                                   "method": f["hpi"]["method"], "origin": f["hpi"]["origin"],
                                   "horizon": f["hpi"]["horizon"],
                                   "all_methods_pct": {m: h[m] for m in h if m not in
                                                       ("forecast", "p10", "p90", "realised_so_far")},
                                   "realised_so_far": h["realised_so_far"]},
            "rent_growth_us": {"forecast_pct": r["forecast"], "p10_pct": r["p10"], "p90_pct": r["p90"],
                               "method": f["rent"]["method"], "origin": f["rent"]["origin"],
                               "horizon": f["rent"]["horizon"], "realised_so_far": r["realised_so_far"]},
            "mortgage_rate_latest": f["mortgage_rate_latest"],
            "disclaimer": DISCLAIMER,
        }

    def legacy_price(self, zip_code: str, state: str | None, beds, baths, sqft) -> float:
        loc, _ = self.locate(zip_code, state)
        return self.predict_price(loc, beds, baths, sqft)["predicted"]


def to_jsonable(obj):
    """numpy / pandas scalars -> plain Python, for JSON responses."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
