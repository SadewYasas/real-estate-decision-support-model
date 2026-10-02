"""Request validation for the API (pydantic). Rates are fractions: 0.05 = 5%."""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import STATE_NAME_TO_CODE

STATE_CODES = set(STATE_NAME_TO_CODE.values())
# Macro features a custom scenario shock may change, with the largest change allowed.
SHOCK_LIMITS = {"mortgage": 5, "mortgage_chg12": 5, "fed_funds": 5, "unemp": 10, "unemp_chg12": 10,
                "permits_g": 80, "gdp_g": 10, "cpi_infl": 10}


class AssumptionsIn(BaseModel):
    """Optional overrides; anything left out uses the CLAUDE.md default or the forecast."""
    model_config = ConfigDict(extra="forbid")
    d: Annotated[float, Field(ge=0, le=1)] | None = None
    T: Annotated[int, Field(ge=1, le=40)] | None = None
    H: Annotated[int, Field(ge=1, le=30)] | None = None
    tau: Annotated[float, Field(ge=0, le=0.1)] | None = None
    m: Annotated[float, Field(ge=0, le=0.1)] | None = None
    h: Annotated[float, Field(ge=0, le=0.05)] | None = None
    cb: Annotated[float, Field(ge=0, le=0.2)] | None = None
    cs: Annotated[float, Field(ge=0, le=0.2)] | None = None
    k: Annotated[float, Field(ge=0, le=0.2)] | None = None
    r: Annotated[float, Field(ge=0, le=0.25)] | None = None
    g: Annotated[float, Field(ge=-0.5, le=0.5)] | None = None
    q: Annotated[float, Field(ge=-0.5, le=0.5)] | None = None


class PropertyIn(BaseModel):
    """One property. Ranges follow the sale-data cleaning rules the models were trained on."""
    model_config = ConfigDict(extra="forbid")
    zip: Annotated[str, Field(pattern=r"^\d{5}$", description="5-digit US ZIP code")]
    state: Annotated[str | None, Field(description="2-letter state; needed only if the ZIP is unknown")] = None
    beds: Annotated[int, Field(ge=1, le=10)]
    baths: Annotated[float, Field(ge=1, le=10)]
    sqft: Annotated[int, Field(ge=300, le=15_000, description="living area, sq ft")]
    price: Annotated[float | None, Field(gt=0, le=50_000_000, description="override the predicted price")] = None
    monthly_rent: Annotated[float | None, Field(gt=0, le=100_000, description="override the predicted rent")] = None
    assumptions: AssumptionsIn | None = None

    @field_validator("zip", mode="before")
    @classmethod
    def zip_as_text(cls, v):
        return str(v).strip().zfill(5) if isinstance(v, int) else str(v).strip()

    @field_validator("state")
    @classmethod
    def known_state(cls, v):
        if v is None:
            return v
        v = v.strip().upper()
        if v not in STATE_CODES:
            raise ValueError(f"unknown state code {v}")
        return v

    @field_validator("baths")
    @classmethod
    def half_baths(cls, v):
        if round(v * 2) != v * 2:
            raise ValueError("bathrooms must be a whole or half number")
        return v


class CompareIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    properties: Annotated[list[PropertyIn], Field(min_length=2, max_length=5)]


class ScenarioIn(PropertyIn):
    custom_shock: dict[str, float] | None = None

    @field_validator("custom_shock")
    @classmethod
    def known_shock(cls, v):
        if v is None:
            return v
        for key, change in v.items():
            if key not in SHOCK_LIMITS:
                raise ValueError(f"unknown shock feature {key}; allowed: {sorted(SHOCK_LIMITS)}")
            if abs(change) > SHOCK_LIMITS[key]:
                raise ValueError(f"shock to {key} must be within +/-{SHOCK_LIMITS[key]}")
        return v
