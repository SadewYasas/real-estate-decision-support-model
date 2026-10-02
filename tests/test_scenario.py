"""Tests for sensitivity.py and scenario.py (need the Step 5 forecast artefacts)."""
import json

import pytest

from src.config import MODELS_DIR
from src.engine.rent_vs_buy import Assumptions, delta
from src.engine.sensitivity import tornado

P, R, STATE = 400_000, 2_200, "TX"
needs_forecasts = pytest.mark.skipif(not (MODELS_DIR / "forecasts.json").exists(),
                                     reason="run src.models.train_forecast first")


def test_tornado_base_and_sorting():
    a = Assumptions(r=0.0635, g=0.03, q=0.035)
    res = tornado(P, R, a)
    assert res["base_delta"] == pytest.approx(delta(P, R, a, a.H))
    swings = [b["swing"] for b in res["bars"]]
    assert swings == sorted(swings, reverse=True)
    g_bar = next(b for b in res["bars"] if b["parameter"] == "g")
    assert g_bar["delta_low"] < res["base_delta"] < g_bar["delta_high"]


def test_tornado_uses_supplied_forecast_band():
    a = Assumptions(r=0.0635, g=0.03, q=0.035)
    res = tornado(P, R, a, g_band=(-0.01, 0.12), q_band=(0.02, 0.07))
    g_bar = next(b for b in res["bars"] if b["parameter"] == "g")
    assert (g_bar["low"], g_bar["high"]) == (-0.01, 0.12)


@needs_forecasts
def test_scenarios_follow_documented_method():
    from src.engine.scenario import Forecaster, run_scenarios
    fc = Forecaster()
    f = json.loads((MODELS_DIR / "forecasts.json").read_text())
    res = run_scenarios(P, R, STATE, forecaster=fc)
    cases = {c["scenario"]: c for c in res["cases"]}
    # Base uses the recommended forecasts and the latest mortgage rate.
    assert cases["base"]["g"] == pytest.approx(f["hpi"]["forecasts"][STATE]["forecast"] / 100)
    assert cases["base"]["q"] == pytest.approx(f["rent"]["forecasts"]["US"]["forecast"] / 100)
    assert cases["base"]["r"] == pytest.approx(f["mortgage_rate_latest"]["value_pct"] / 100)
    # Pessimistic / optimistic = p10 / p90 of the forecast band.
    assert cases["pessimistic"]["g"] == pytest.approx(f["hpi"]["forecasts"][STATE]["p10"] / 100)
    assert cases["optimistic"]["q"] == pytest.approx(f["rent"]["forecasts"]["US"]["p90"] / 100)
    # Rent shock = ARIMA base + change predicted by the rent gbm_macro model.
    shock = {"mortgage": 1.0, "mortgage_chg12": 1.0, "fed_funds": 1.0}
    change = fc._gbm("rent", STATE, shock) - fc._gbm("rent", STATE, None)
    assert cases["rate_rise"]["q"] == pytest.approx(cases["base"]["q"] + change / 100)
    # Mortgage-rate shocks move the buyer's mortgage rate too.
    assert cases["rate_rise"]["r"] == pytest.approx(cases["base"]["r"] + 0.01)


@needs_forecasts
def test_zero_shock_changes_nothing():
    from src.engine.scenario import run_scenarios
    res = run_scenarios(P, R, STATE, custom_shock={"mortgage": 0.0})
    custom = next(c for c in res["cases"] if c["scenario"] == "custom")
    assert custom["delta"] == pytest.approx(res["cases"][0]["delta"])
    assert not custom["flips_vs_base"]


@needs_forecasts
@pytest.mark.parametrize("feature", ["mortgage", "mortgage_chg12", "unemp", "unemp_chg12"])
def test_constrained_hpi_model_is_monotonic(feature):
    """If the scenarios use the constrained model, raising a constrained feature alone can
    never raise predicted HPI growth, in any state."""
    from src.engine.scenario import Forecaster
    fc = Forecaster()
    if fc.models["hpi"].get("variant") != "gbm_macro_constrained":
        pytest.skip("scenarios use the unconstrained model")
    states = sorted(fc.forecasts["hpi"]["forecasts"])
    for state in states:
        preds = [fc._gbm("hpi", state, {feature: step}) for step in (-3, -1, 0, 1, 3)]
        assert all(b <= a + 1e-12 for a, b in zip(preds, preds[1:])), (state, preds)


@needs_forecasts
def test_hpi_shock_reproduces_stored_forecast_when_unshocked():
    from src.engine.scenario import Forecaster
    fc = Forecaster()
    f = json.loads((MODELS_DIR / "forecasts.json").read_text())
    variant = fc.models["hpi"]["variant"]
    for state in ("CA", "TX", "NY"):
        assert fc._gbm("hpi", state, None) == pytest.approx(f["hpi"]["forecasts"][state][variant])
