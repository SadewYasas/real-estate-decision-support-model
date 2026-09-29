"""Shared paths and constants for the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The raw folder is committed as "Data/Raw"; CLAUDE.md calls it "data/raw".
# Use whichever exists so the code works before and after a rename.
_DATA_DIR = next((ROOT / d for d in ("data", "Data") if (ROOT / d).is_dir()), ROOT / "data")
RAW_DIR = next((_DATA_DIR / d for d in ("raw", "Raw") if (_DATA_DIR / d).is_dir()), _DATA_DIR / "raw")
PROCESSED_DIR = _DATA_DIR / "processed"

ARTEFACTS_DIR = ROOT / "artefacts"
METRICS_DIR = ARTEFACTS_DIR / "metrics"
FIGURES_DIR = ARTEFACTS_DIR / "figures"
MODELS_DIR = ARTEFACTS_DIR / "models"

RANDOM_STATE = 42

STATE_NAME_TO_CODE = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def ensure_dirs():
    for d in (PROCESSED_DIR, METRICS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
