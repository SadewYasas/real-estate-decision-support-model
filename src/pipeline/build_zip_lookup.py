"""Build the ZIP lookup (ZIP -> state, coordinates, population density, household income).

Sources (downloaded into data/raw/census/ if missing):
- Census 2020 ZCTA Gazetteer: land area (sq mi) and internal point (lat, lon).
- Census 2020 ZCTA-to-county relationship file: state = state of the county holding the
  largest share of the ZCTA's land (137 ZCTAs cross a state line).
- ACS 2020-2024 5-year estimates by ZCTA, from the Census table-based summary files
  (the Census API now requires a key; the summary files hold the same estimates):
    B01003_001E total population            -> density = population / land sq mi
    B19013_001E median household income     -> median_income (top-coded at 250,001)
    B19025_001E aggregate household income  -> mean_income = B19025 / B11001
    B11001_001E households

Comparison with the D1 training columns (saved to artefacts/metrics/zip_lookup_comparison.json):
- zip_code_density is people per square mile: its median over listings (1,589) is close to
  the population-weighted ZCTA median per sq mi (~1,680), not per sq km (~650).
- median_household_income is NOT the ACS median: 33 D1 ZIP profiles exceed the ACS top-code
  (250,001) and its distribution is ~25% above the ACS median, but it matches the ACS mean
  household income closely. The sale model is therefore fed mean_income (SALE_MODEL_INCOME).

Fallback for a ZIP not in the lookup (or with a missing value): the state's population-
weighted median of each field, and the state's population-weighted mean coordinates.

Run:  python -m src.pipeline.build_zip_lookup
"""
import json
import urllib.request

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

CENSUS_DIR = RAW_DIR / "census"
ACS_YEAR = 2024
ACS_URL = ("https://www2.census.gov/programs-surveys/acs/summary_file/{y}/table-based-SF/data/"
           "5YRData/acsdt5y{y}-{t}.dat")
DOWNLOADS = {
    "2020_Gaz_zcta_national.zip": "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
                                  "2020_Gazetteer/2020_Gaz_zcta_national.zip",
    "tab20_zcta520_county20_natl.txt": "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
                                       "zcta520/tab20_zcta520_county20_natl.txt",
    **{f"acsdt5y{ACS_YEAR}-{t}.dat": ACS_URL.format(y=ACS_YEAR, t=t)
       for t in ("b01003", "b19013", "b19025", "b11001")},
}
SALE_MODEL_INCOME = "mean_income"   # matches the scale of D1 median_household_income

STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def download():
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOWNLOADS.items():
        path = CENSUS_DIR / name
        if not path.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
    gaz = CENSUS_DIR / "2020_Gaz_zcta_national.txt"
    if not gaz.exists():
        import zipfile
        zipfile.ZipFile(CENSUS_DIR / "2020_Gaz_zcta_national.zip").extractall(CENSUS_DIR)


def acs_table(table: str, column: str) -> pd.Series:
    d = pd.read_csv(CENSUS_DIR / f"acsdt5y{ACS_YEAR}-{table}.dat", sep="|", dtype={"GEO_ID": str})
    d = d[d["GEO_ID"].str.startswith("860Z200US")]
    s = pd.to_numeric(d[column], errors="coerce")
    s.index = d["GEO_ID"].str[-5:]
    return s.where(s >= 0)   # ACS uses large negative codes for "not available"


def weighted_quantile(x: pd.Series, w: pd.Series, q):
    ok = x.notna() & w.notna() & (w > 0)
    x, w = x[ok].to_numpy(), w[ok].to_numpy()
    i = np.argsort(x)
    c = np.cumsum(w[i]) / w.sum()
    return np.interp(q, c, x[i])


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    gaz = pd.read_csv(CENSUS_DIR / "2020_Gaz_zcta_national.txt", sep="\t", dtype={"GEOID": str})
    gaz.columns = [c.strip() for c in gaz.columns]
    rel = pd.read_csv(CENSUS_DIR / "tab20_zcta520_county20_natl.txt", sep="|", dtype=str,
                      encoding="utf-8-sig").dropna(subset=["GEOID_ZCTA5_20"])
    rel["part"] = rel["AREALAND_PART"].astype(float)
    state = (rel.sort_values("part").groupby("GEOID_ZCTA5_20").tail(1)
             .set_index("GEOID_ZCTA5_20")["GEOID_COUNTY_20"].str[:2].map(STATE_FIPS))

    z = pd.DataFrame({
        "zip": gaz["GEOID"], "lat": gaz["INTPTLAT"], "lon": gaz["INTPTLONG"],
        "land_sqmi": gaz["ALAND_SQMI"],
    }).set_index("zip")
    z["state"] = state
    z["population"] = acs_table("b01003", "B01003_E001")
    z["median_income"] = acs_table("b19013", "B19013_E001")
    z["mean_income"] = acs_table("b19025", "B19025_E001") / acs_table("b11001", "B11001_E001")
    z["density"] = np.where(z["land_sqmi"] > 0, z["population"] / z["land_sqmi"], np.nan)
    z = z[z["state"].notna()].reset_index()   # drops Puerto Rico and other territories

    rows = []
    for st, g in z.groupby("state"):
        w = g["population"]
        rows.append({
            "state": st,
            "lat": float(np.average(g["lat"], weights=w)), "lon": float(np.average(g["lon"], weights=w)),
            "density": float(weighted_quantile(g["density"], w, 0.5)),
            "median_income": float(weighted_quantile(g["median_income"], w, 0.5)),
            "mean_income": float(weighted_quantile(g["mean_income"], w, 0.5)),
            "n_zips": len(g),
        })
    state_medians = pd.DataFrame(rows)
    cols = ["zip", "state", "lat", "lon", "density", "median_income", "mean_income", "population", "land_sqmi"]
    return z[cols], state_medians


def compare_with_d1(z: pd.DataFrame) -> dict:
    d1 = pd.read_csv(PROCESSED_DIR / "sale_clean.csv")
    zz = z[z["state"].isin(d1["state_code"].unique()) & (z["population"] > 0)]
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    pw = lambda col: [round(float(v), 1) for v in weighted_quantile(zz[col], zz["population"], q)]
    dens_km2 = zz["population"] / (zz["land_sqmi"] * 2.589988)
    return {
        "note": "Quantiles 5/25/50/75/95%. D1 over listings; ZCTAs in D1's 29 states, weighted by population.",
        "density": {
            "d1_zip_code_density": [round(float(v), 1) for v in d1["zip_code_density"].quantile(q)],
            "zcta_people_per_sq_mile": pw("density"),
            "zcta_people_per_sq_km": [round(float(v), 1) for v in weighted_quantile(dens_km2, zz["population"], q)],
            "conclusion": "D1 is people per square mile (medians agree; per sq km is ~2.6x too low). "
                          "D1's spread is narrower: listing ZIPs exclude the most rural and densest areas.",
        },
        "income": {
            "d1_median_household_income": [round(float(v), 0) for v in d1["median_household_income"].quantile(q)],
            "acs_median_household_income_B19013": pw("median_income"),
            "acs_mean_household_income_B19025_over_B11001": pw("mean_income"),
            "d1_profiles_above_acs_median_topcode_250001": int(
                (d1.drop_duplicates(["state_code", "zip_code_density", "median_household_income"])
                 ["median_household_income"] > 250001).sum()),
            "conclusion": "Despite its name, D1's income column behaves like MEAN household income: it "
                          "exceeds the ACS median top-code and its distribution matches the ACS mean, "
                          "~25% above the ACS median. The sale model is fed mean_income.",
        },
        "sale_model_income_column": SALE_MODEL_INCOME,
        "acs_vintage": f"ACS {ACS_YEAR - 4}-{ACS_YEAR} 5-year",
    }


def main():
    ensure_dirs()
    download()
    z, state_medians = build()
    z.to_csv(PROCESSED_DIR / "zip_lookup.csv", index=False)
    state_medians.to_csv(PROCESSED_DIR / "zip_lookup_state_medians.csv", index=False)
    cmp = compare_with_d1(z)
    with open(METRICS_DIR / "zip_lookup_comparison.json", "w") as f:
        json.dump(cmp, f, indent=2)
    print(f"zip_lookup.csv: {len(z):,} ZIPs in {z['state'].nunique()} states; missing "
          f"density {z['density'].isna().sum()}, median income {z['median_income'].isna().sum()}, "
          f"mean income {z['mean_income'].isna().sum()}")
    print(json.dumps(cmp, indent=1))


if __name__ == "__main__":
    main()
