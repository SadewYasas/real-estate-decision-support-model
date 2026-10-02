"""Build the forecasting panels and the CPI rent adjustment factors.

1. HPI panel (state x quarter): FHFA all-transactions state HPI (D4, hpi_master.csv
   filtered to traditional / all-transactions / quarterly / State) + FRED state panel (D3).
2. Rent panel (national x month): BLS CPI "Rent of primary residence", US city average,
   not seasonally adjusted (CUUR0000SEHA) + the national FRED columns.
3. CPI rent factors: CPI rent in the latest month / CPI rent in each listing month of the
   UCI rent data (D2, Dec 2018 - Dec 2019), used to bring 2019 rents to current levels.

No-look-ahead conventions (the forecast date is when the origin period's target-series
value is published):
- HPI for quarter t is published about two months after the quarter ends; all monthly
  FRED values for the months of quarter t are published by then.
- CPI rent for month t is published mid-month t+1; FRED values for month t are too.
- Annual state real GDP for year Y is published around the end of Q1 of Y+1. The FRED
  file stamps it in January of Y (and carries 2024 forward through 2025), so it is lagged:
  at an origin in year Y the latest usable growth is Y-1 vs Y-2 from Q2 onwards, and
  Y-2 vs Y-3 in Q1. The carried-forward 2025 value is never used.

Data quality (checked, not assumed): the last month of fred_final.csv (Sep 2025) is
incomplete. State unemployment is one value (4.4, the national rate) for all 51 states,
state permits are one value (1258.4) for all states, and national CPI and national permits
repeat August exactly. These are placeholders, not observations, so they are set to
missing; each feature then uses its latest genuine month within the period (Aug 2025 for
those four, Sep 2025 for the mortgage and fed funds rates). The CPI rent series has no
value for Oct 2025 (BLS did not publish it); series are put on a full calendar before any
lag is taken so that shifts are always by date, never by row.

Run:  python -m src.pipeline.build_panel
"""
import json

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

HPI_HORIZON = 4      # quarters
RENT_HORIZON = 12    # months
RENT_FACTOR_REFERENCE = "2025-09"  # = rent forecast origin (latest FRED month)

HPI_MACRO = ["unemp", "unemp_chg12", "permits_g", "gdp_g", "mortgage", "mortgage_chg12",
             "fed_funds", "cpi_infl"]
HPI_BASE = ["hpi_g4", "hpi_g1"]
RENT_MACRO = ["mortgage", "mortgage_chg12", "fed_funds", "cpi_infl", "permits_g"]
RENT_BASE = ["rent_g12", "rent_g3"]


def pct(a, b):
    return 100 * (a / b - 1)


# ----------------------------------------------------------------------------- loaders
def load_hpi() -> pd.DataFrame:
    h = pd.read_csv(RAW_DIR / "hpi_master.csv", low_memory=False)
    h = h[(h["hpi_type"] == "traditional") & (h["hpi_flavor"] == "all-transactions")
          & (h["frequency"] == "quarterly") & (h["level"] == "State")]
    h = h.assign(quarter=pd.PeriodIndex.from_fields(year=h["yr"], quarter=h["period"], freq="Q"))
    return h.rename(columns={"place_id": "state", "index_nsa": "hpi"})[["state", "quarter", "hpi"]]


STATE_COLS = ["state_unemployment_rate", "state_building_permits"]


def find_placeholders(f: pd.DataFrame) -> dict:
    """Placeholder values in the FRED panel. Returns {column: [months]}.

    - A state series with one value in every state that month is a placeholder.
    - CPI (3 decimals) repeating the previous month exactly is a carry-forward.
    - National permits are whole thousands, so a repeat can be genuine; a repeat is only
      treated as a carry-forward in a month already shown to hold state placeholders.
    """
    found = {}
    for col in STATE_COLS:
        one_value = f.groupby("month")[col].nunique() == 1
        found[col] = [str(m) for m in one_value[one_value].index]
    bad_months = set(found["state_unemployment_rate"]) | set(found["state_building_permits"])
    nat = f.drop_duplicates("month").set_index("month").sort_index()
    rep = nat["cpi"].diff() == 0
    found["cpi"] = [str(m) for m in rep[rep].index]
    rep = nat["building_permits"].diff() == 0
    found["building_permits"] = [str(m) for m in rep[rep].index if str(m) in bad_months]
    return {k: v for k, v in found.items() if v}


def load_fred() -> pd.DataFrame:
    f = pd.read_csv(RAW_DIR / "fred_final.csv")
    f["month"] = pd.to_datetime(f["date"]).dt.to_period("M")
    f = f.sort_values(["state", "month"]).reset_index(drop=True)
    for col, months in find_placeholders(f).items():
        f.loc[f["month"].astype(str).isin(months), col] = np.nan
    return f


def load_cpi_rent() -> pd.Series:
    c = pd.read_csv(RAW_DIR / "CUUR0000SEHA.csv")
    c.index = pd.to_datetime(c["observation_date"]).dt.to_period("M")
    s = c["CUUR0000SEHA"].astype(float).rename("cpi_rent")
    s = s.loc[s.first_valid_index():]
    # Full monthly calendar: missing months stay NaN, so shift(k) is always k months.
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))


# ----------------------------------------------------------------------------- HPI panel
def annual_state_gdp_growth(fred: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year growth of annual state real GDP, one row per state x year."""
    last_real_year = 2024  # FRED file carries 2024 forward through 2025
    annual = (fred.assign(year=fred["month"].dt.year)
              .query("year <= @last_real_year")
              .groupby(["state", "year"])["state_real_gdp"].first().reset_index())
    annual["gdp_growth"] = annual.groupby("state")["state_real_gdp"].transform(lambda s: pct(s, s.shift(1)))
    return annual[["state", "year", "gdp_growth"]]


def build_hpi_panel() -> pd.DataFrame:
    hpi = load_hpi()
    fred = load_fred()

    # Monthly features, then the value at the last month of each quarter.
    g = fred.groupby("state")
    m = pd.DataFrame({
        "state": fred["state"],
        "month": fred["month"],
        "unemp": fred["state_unemployment_rate"],
        "unemp_chg12": fred["state_unemployment_rate"] - g["state_unemployment_rate"].shift(12),
        # 3-month sum of permits vs the same three months a year earlier (removes seasonality)
        "permits_g": pct(g["state_building_permits"].transform(lambda s: s.rolling(3).sum()),
                         g["state_building_permits"].transform(lambda s: s.rolling(3).sum().shift(12))),
        "mortgage": fred["mortgage_rate"],
        "mortgage_chg12": fred["mortgage_rate"] - g["mortgage_rate"].shift(12),
        "fed_funds": fred["interest_rate"],
        "cpi_infl": pct(fred["cpi"], g["cpi"].shift(12)),
    })
    # Value per quarter = latest genuine month in that quarter (groupby.last skips NaN).
    m["quarter"] = m["month"].dt.asfreq("Q")
    m = m.drop(columns="month").groupby(["state", "quarter"], as_index=False).last()

    gdp = annual_state_gdp_growth(fred)
    m["gdp_year"] = m["quarter"].dt.year - np.where(m["quarter"].dt.quarter >= 2, 1, 2)
    m = m.merge(gdp.rename(columns={"year": "gdp_year", "gdp_growth": "gdp_g"}),
                on=["state", "gdp_year"], how="left")

    # HPI features and target (HPI runs past FRED, so targets exist for late origins).
    hpi = hpi.sort_values(["state", "quarter"])
    hg = hpi.groupby("state")["hpi"]
    hpi["hpi_g4"] = pct(hpi["hpi"], hg.shift(4))
    hpi["hpi_g1"] = pct(hpi["hpi"], hg.shift(1))
    hpi["target"] = pct(hg.shift(-HPI_HORIZON), hpi["hpi"])

    panel = m.merge(hpi, on=["state", "quarter"], how="left")
    panel = panel[["state", "quarter", "hpi"] + HPI_BASE + HPI_MACRO + ["target"]]
    # First quarter with every feature (12-month changes need 2012; GDP lag needs 2012-2013).
    panel = panel.dropna(subset=HPI_BASE + HPI_MACRO).reset_index(drop=True)
    return panel


# ----------------------------------------------------------------------------- rent panel
def build_rent_panel() -> pd.DataFrame:
    rent = load_cpi_rent()
    nat = load_fred().drop_duplicates("month").set_index("month").sort_index()
    permits3 = nat["building_permits"].rolling(3).sum()
    feats = pd.DataFrame({
        "mortgage": nat["mortgage_rate"],
        "mortgage_chg12": nat["mortgage_rate"] - nat["mortgage_rate"].shift(12),
        "fed_funds": nat["interest_rate"],
        "cpi_infl": pct(nat["cpi"], nat["cpi"].shift(12)),
        "permits_g": pct(permits3, permits3.shift(12)),
    })
    r = pd.DataFrame({"cpi_rent": rent})
    r["rent_g12"] = pct(rent, rent.shift(12))
    r["rent_g3"] = pct(rent, rent.shift(3))
    r["target"] = pct(rent.shift(-RENT_HORIZON), rent)

    # Latest genuine value: a feature missing in a month (placeholder removed) takes the
    # previous month's value. Only the final month (Sep 2025) is affected.
    feats = feats.ffill(limit=1)
    panel = feats.join(r, how="left")
    panel.index.name = "month"
    panel = panel.reset_index()[["month", "cpi_rent"] + RENT_BASE + RENT_MACRO + ["target"]]
    return panel.dropna(subset=RENT_BASE + RENT_MACRO).reset_index(drop=True)


# ----------------------------------------------------------------------------- CPI factors
def cpi_rent_factors() -> dict:
    """CPI rent in the rent forecast origin month / CPI rent in each listing month.

    The reference month is the rent forecast origin (RENT_FACTOR_REFERENCE), not the latest
    CPI month: the rent level and the 12-month rent growth forecast then start from the
    same date, so growth already observed after the origin is not counted twice.
    """
    rent = load_cpi_rent()
    ref = pd.Period(RENT_FACTOR_REFERENCE, freq="M")
    listings = pd.read_csv(PROCESSED_DIR / "rent_clean.csv", usecols=["listed_date"])
    counts = pd.to_datetime(listings["listed_date"]).dt.to_period("M").value_counts().sort_index()
    factors = {str(m): float(rent[ref] / rent[m]) for m in counts.index}
    weighted = float(sum(factors[str(m)] * n for m, n in counts.items()) / counts.sum())
    return {
        "series": "CUUR0000SEHA (CPI rent of primary residence, US city average, NSA)",
        "reference_month": str(ref),
        "reference_index": float(rent[ref]),
        "reference_reason": "rent forecast origin; the rent growth forecast starts here",
        "factor_by_listing_month": factors,
        "listings_by_month": {str(m): int(n) for m, n in counts.items()},
        "listing_weighted_factor": weighted,
        "how_to_use": "current_rent = predicted_2019_rent x listing_weighted_factor (the rent "
                      "model is trained on listings pooled over Dec 2018 - Dec 2019).",
    }


def main():
    ensure_dirs()
    raw = pd.read_csv(RAW_DIR / "fred_final.csv")
    raw["month"] = pd.to_datetime(raw["date"]).dt.to_period("M")
    placeholders = find_placeholders(raw.sort_values(["state", "month"]))
    rent_series = load_cpi_rent()
    quality = {
        "fred_placeholders_set_to_missing": placeholders,
        "cpi_rent_missing_months_since_2012": [
            str(m) for m in rent_series[rent_series.isna()].index if m.year >= 2012],
    }
    with open(PROCESSED_DIR / "panel_data_quality.json", "w") as f:
        json.dump(quality, f, indent=2)
    print("Data quality:", json.dumps(quality))
    hpi = build_hpi_panel()
    rent = build_rent_panel()
    hpi.to_csv(PROCESSED_DIR / "hpi_panel.csv", index=False)
    rent.to_csv(PROCESSED_DIR / "rent_panel.csv", index=False)
    factors = cpi_rent_factors()
    with open(MODELS_DIR / "cpi_rent_factors.json", "w") as f:
        json.dump(factors, f, indent=2)
    print(f"HPI panel:  {len(hpi):,} rows, {hpi['state'].nunique()} states, "
          f"{hpi['quarter'].min()} - {hpi['quarter'].max()}, "
          f"targets to {hpi.dropna(subset=['target'])['quarter'].max()}")
    print(f"Rent panel: {len(rent):,} rows, {rent['month'].min()} - {rent['month'].max()}, "
          f"targets to {rent.dropna(subset=['target'])['month'].max()}")
    print(f"CPI rent factors to {factors['reference_month']}: "
          f"{min(factors['factor_by_listing_month'].values()):.3f} - "
          f"{max(factors['factor_by_listing_month'].values()):.3f}, "
          f"listing-weighted {factors['listing_weighted_factor']:.3f}")


if __name__ == "__main__":
    main()
