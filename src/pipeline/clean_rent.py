"""Clean D2 (UCI apartments_for_rent_classified_100K.csv) rent listings.

Rules (CLAUDE.md):
- keep category == housing/rent/apartment, price_type == Monthly, currency == USD;
- rent 200-15,000; square_feet 150-8,000.
Also: drop repeated listing ids, and drop rows missing state, beds, baths or coordinates
(dropped, never filled). Studios (bedrooms == 0) are kept.

The listing date is kept so Step 5 can inflate 2019 rents to the D1 reference year.

Run:  python -m src.pipeline.clean_rent
"""
import json

import pandas as pd

from src.config import METRICS_DIR, PROCESSED_DIR, RAW_DIR, ensure_dirs

RAW_FILE = RAW_DIR / "apartments_for_rent_classified_100K.csv"
OUT_FILE = PROCESSED_DIR / "rent_clean.csv"

REQUIRED = ["rent", "bedrooms", "bathrooms", "square_feet", "state", "latitude", "longitude"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_FILE, sep=";", encoding="cp1252", na_values=["null"], low_memory=False)


def clean_rent(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(raw)}
    df = raw.copy()

    df = df[(df["category"] == "housing/rent/apartment")
            & (df["price_type"] == "Monthly")
            & (df["currency"] == "USD")]
    log["after_category_monthly_usd"] = len(df)

    df = df[df["price"].between(200, 15_000)]
    log["after_rent_200_15000"] = len(df)
    df = df[df["square_feet"].between(150, 8_000)]
    log["after_sqft_150_8000"] = len(df)

    df = df.drop_duplicates(subset="id", keep="first")
    log["after_drop_duplicate_ids"] = len(df)

    df = df.rename(columns={"price": "rent", "cityname": "city"})
    df = df.dropna(subset=REQUIRED)
    log["after_drop_missing"] = len(df)

    df["listed_date"] = pd.to_datetime(df["time"], unit="s").dt.date
    df = df[["id", "rent", "bedrooms", "bathrooms", "square_feet", "city", "state",
             "latitude", "longitude", "listed_date"]].reset_index(drop=True)

    log["final_rows"] = len(df)
    log["n_states"] = int(df["state"].nunique())
    log["listed_date_min"] = str(df["listed_date"].min())
    log["listed_date_max"] = str(df["listed_date"].max())
    # Same unit re-listed under different ids; Step 4 should split by group to avoid
    # the same unit appearing in both train and test.
    log["repeated_units_same_address_size_rent"] = int(df.duplicated(
        subset=["rent", "square_feet", "bedrooms", "bathrooms", "latitude", "longitude"]).sum())
    return df, log


def main():
    ensure_dirs()
    df, log = clean_rent(load_raw())
    df.to_csv(OUT_FILE, index=False)
    with open(METRICS_DIR / "cleaning_rent.json", "w") as f:
        json.dump(log, f, indent=2)
    print(json.dumps(log, indent=2))
    print(f"Saved {len(df):,} rows to {OUT_FILE}")


if __name__ == "__main__":
    main()
