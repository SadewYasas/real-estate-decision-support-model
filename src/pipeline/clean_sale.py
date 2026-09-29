"""Clean D1 (Final Dataset.csv) sale listings.

Rules (CLAUDE.md):
- keep only price, beds, baths, living_space, state, zip_code_density, median_household_income;
- drop the six macro columns (not real FRED values, and they leak the price);
- drop duplicates; price 30k-5M; beds and baths 1-10; living_space 300-15,000.

Duplicates are judged on the kept columns: the raw file repeats the same house up to
five times with different macro values, so a full-row check would find none.

Run:  python -m src.pipeline.clean_sale
"""
import json

import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR, STATE_NAME_TO_CODE, ensure_dirs

RAW_FILE = RAW_DIR / "Final Dataset.csv"
OUT_FILE = PROCESSED_DIR / "sale_clean.csv"

KEEP_COLUMNS = ["price", "beds", "baths", "living_space", "state",
                "zip_code_density", "median_household_income"]
MACRO_COLUMNS = ["state_unemployment_rate", "state_building_permits", "state_real_gdp",
                 "mortgage_rate", "interest_rate", "cpi"]


def clean_sale(raw: pd.DataFrame, keep_macro: bool = False) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned frame and a step-by-step row-count log.

    keep_macro=True is only for the leakage audit; the modelling data never has them.
    """
    log = {"raw_rows": len(raw)}
    df = raw.copy()

    df = df.drop_duplicates(subset=KEEP_COLUMNS, keep="first")
    log["after_drop_duplicates"] = len(df)

    df = df[df["price"].between(30_000, 5_000_000)]
    log["after_price_30k_5M"] = len(df)
    df = df[df["beds"].between(1, 10) & df["baths"].between(1, 10)]
    log["after_beds_baths_1_10"] = len(df)
    df = df[df["living_space"].between(300, 15_000)]
    log["after_living_space_300_15000"] = len(df)

    # Rows with a missing feature are dropped, never filled.
    df = df.dropna(subset=KEEP_COLUMNS)
    log["after_drop_missing"] = len(df)

    df["state_code"] = df["state"].map(STATE_NAME_TO_CODE)
    unmapped = df.loc[df["state_code"].isna(), "state"].unique().tolist()
    if unmapped:
        raise ValueError(f"Unknown state names: {unmapped}")

    columns = KEEP_COLUMNS + ["state_code"] + (MACRO_COLUMNS if keep_macro else [])
    df = df[columns].reset_index(drop=True)
    log["final_rows"] = len(df)
    log["n_states"] = int(df["state"].nunique())
    return df, log


def main():
    ensure_dirs()
    raw = pd.read_csv(RAW_FILE)
    df, log = clean_sale(raw)
    df.to_csv(OUT_FILE, index=False)
    log["dropped_columns"] = MACRO_COLUMNS
    with open(METRICS_DIR / "cleaning_sale.json", "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print(f"Saved {len(df):,} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
