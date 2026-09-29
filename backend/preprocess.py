import numpy as np
import pandas as pd
import re
import json
import os

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PROCESSED_PATH = os.path.join(BASE_DIR, "data", "house_train_proc.csv")
FEATURE_COLUMNS_PATH = os.path.join(BASE_DIR, "data", "house_feature_cols.json")

# Constants
TARGET_COLUMN = "ListedPrice"
CATEGORICAL_CANDIDATES = ["State", "City", "Zipcode"]

# Load Data and Feature Schema
train_full_df = pd.read_csv(TRAIN_PROCESSED_PATH)

with open(FEATURE_COLUMNS_PATH) as feature_file:
    FEATURE_COLUMNS = json.load(feature_file)


# Helper Function: Normalize ZIP codes
def normalize_zipcode(zip_value):
    """
    Clean and standardize ZIP code values.
      - Remove non-digit characters.
      - Pad to 5 digits if short.
      - Replace missing/invalid entries with 'Unknown'.
    """
    if zip_value is None or (isinstance(zip_value, float) and np.isnan(zip_value)):
        return "Unknown"

    digits_only = re.sub(r"\D", "", str(zip_value)).strip()  # keep only digits

    if 0 < len(digits_only) <= 5:
        return digits_only.zfill(5)  # pad with zeros if shorter than 5 digits
    return digits_only if digits_only else "Unknown"


# Compute ZIP-based Statistics for Proxy Price Estimation
train_full_df = train_full_df.copy()
train_full_df["Zipcode_Normalized"] = train_full_df["Zipcode"].map(normalize_zipcode)

# Average price per ZIP code (used as fallback estimate)
zip_to_avg_price = train_full_df.groupby("Zipcode_Normalized")[TARGET_COLUMN].mean().to_dict()
global_average_price = float(train_full_df[TARGET_COLUMN].mean())

# Median price-per-square-foot per ZIP (used as main proxy)
eps = 1e-9  # small constant to avoid division by zero
price_per_sqft = train_full_df[TARGET_COLUMN] / pd.to_numeric(train_full_df["Area"], errors="coerce").clip(lower=eps)
train_full_df["PricePerSqFt_Temp"] = price_per_sqft

zip_to_median_ppsq = train_full_df.groupby("Zipcode_Normalized")["PricePerSqFt_Temp"].median().to_dict()
global_median_ppsq = (
    float(np.median(list(zip_to_median_ppsq.values())))
    if len(zip_to_median_ppsq)
    else float(price_per_sqft.median())
)

# Determine which categorical columns are actually in the schema
CATEGORICAL_IN_SCHEMA = [col for col in CATEGORICAL_CANDIDATES if col in FEATURE_COLUMNS]


# Function: Estimate Market Price if Missing
def proxy_market_estimate(zip_normalized: str, area_value: float) -> float:
    """
    Estimate the market price when 'MarketEstimate' is missing.

    Uses median price-per-square-foot for user entered ZIP code multiplied by area.
    Falls back to ZIP-average or global mean if data is missing.
    """
    median_ppsq = zip_to_median_ppsq.get(zip_normalized, global_median_ppsq)
    if not np.isfinite(median_ppsq):
        median_ppsq = global_median_ppsq

    area_value = float(pd.to_numeric(area_value, errors="coerce") or 0)
    estimated_price = median_ppsq * max(area_value, 0.0)

    if estimated_price <= 0:
        # Fallback to ZIP average or global average price
        estimated_price = zip_to_avg_price.get(zip_normalized, global_average_price)

    return float(estimated_price)


# Function: Convert JSON Payload → Model-Ready DataFrame
def build_model_input_frame(payload: dict) -> pd.DataFrame:
    """
    Convert an incoming JSON payload into a model-ready DataFrame.
    Ensures schema consistency, fills missing data, and normalizes formats.
    """
    # Initialize all expected features as NaN
    base_row = {col: np.nan for col in FEATURE_COLUMNS}

    # Fill in provided fields from payload
    for feature_name in ["State", "City", "Zipcode", "Bedroom", "Bathroom", "Area", "LotArea", "MarketEstimate"]:
        if feature_name in FEATURE_COLUMNS and feature_name in payload:
            base_row[feature_name] = payload[feature_name]

    model_input_df = pd.DataFrame([base_row])

    # Handle categorical columns
    if "State" in model_input_df:
        model_input_df["State"] = model_input_df["State"].astype(str).fillna("Unknown").replace("nan", "Unknown")

    if "City" in model_input_df:
        model_input_df["City"] = model_input_df["City"].astype(str).fillna("Unknown").replace("nan", "Unknown")

    if "Zipcode" in model_input_df:
        model_input_df["Zipcode"] = model_input_df["Zipcode"].map(normalize_zipcode)

    # Convert numeric columns
    for column_name in model_input_df.columns:
        if column_name not in CATEGORICAL_IN_SCHEMA:
            model_input_df[column_name] = pd.to_numeric(model_input_df[column_name], errors="coerce")

    # Auto-fill missing MarketEstimate using proxy function
    if "MarketEstimate" in model_input_df.columns and pd.isna(model_input_df.at[0, "MarketEstimate"]):
        zip_norm = model_input_df.at[0, "Zipcode"] if "Zipcode" in model_input_df.columns else "Unknown"
        area_val = model_input_df.at[0, "Area"] if "Area" in model_input_df.columns else 0
        model_input_df.at[0, "MarketEstimate"] = proxy_market_estimate(zip_norm, area_val)

    # Align DataFrame columns to expected schema
    model_input_df = model_input_df.reindex(columns=FEATURE_COLUMNS)

    # Final cleanup for categorical & numeric columns
    for column_name in model_input_df.columns:
        if column_name in CATEGORICAL_IN_SCHEMA:
            model_input_df[column_name] = model_input_df[column_name].fillna("Unknown")
        else:
            if column_name != "MarketEstimate":
                model_input_df[column_name] = (
                    pd.to_numeric(model_input_df[column_name], errors="coerce").fillna(0)
                )

    return model_input_df
