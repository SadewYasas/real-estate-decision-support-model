"""API tests (Flask test client; models load once per test session)."""
import math

import pytest

from backend.app import app
from src.engine.rent_vs_buy import DISCLAIMER

AUSTIN = {"zip": "78704", "beds": 3, "baths": 2, "sqft": 1800}


@pytest.fixture(scope="module")
def client():
    return app.test_client()


def post(client, path, body):
    r = client.post(path, json=body)
    return r.status_code, r.get_json()


# ----------------------------------------------------------------------------- analyse
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_analyse_full_result(client):
    status, d = post(client, "/api/analyse", AUSTIN)
    assert status == 200
    assert d["location"]["state"] == "TX" and d["location"]["found"]
    assert set(d["location"]["source"].values()) == {"zip"}
    assert d["price"]["source"] == "model" and d["price"]["used"] == pytest.approx(d["price"]["predicted"])
    assert d["rent"]["predicted"] == pytest.approx(d["rent"]["predicted_2019"] * d["rent"]["cpi_factor"])
    assert d["rent"]["model"] == "rent_model"
    assert len(d["result"]["yearly"]) == 30
    assert (d["result"]["delta"] >= 0) == (d["result"]["recommendation"] == "buy")
    assert d["assumptions"]["g"] == pytest.approx(d["forecast"]["g"]["value"])
    assert d["assumptions"]["source"]["g"] == "forecast"
    assert d["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize("part", ["price", "rent"])
def test_shap_top5_reconstructs_prediction(client, part):
    _, d = post(client, "/api/analyse", AUSTIN)
    exp = d[part]["explanation"]
    assert len(exp["top"]) == 5
    total = math.prod(1 + t["effect_pct"] for t in exp["top"]) * (1 + exp["other_features_effect_pct"])
    assert exp["baseline"] * total == pytest.approx(exp["prediction"], rel=1e-6)
    effects = [abs(t["shap_log"]) for t in exp["top"]]
    assert effects == sorted(effects, reverse=True)


def test_user_overrides_are_used(client):
    body = {**AUSTIN, "price": 500_000, "monthly_rent": 2_500, "assumptions": {"H": 10, "d": 0.1, "g": 0.02}}
    status, d = post(client, "/api/analyse", body)
    assert status == 200
    assert d["price"]["used"] == 500_000 and d["price"]["source"] == "user"
    assert d["rent"]["used"] == 2_500 and d["rent"]["source"] == "user"
    assert d["assumptions"]["H"] == 10 and d["assumptions"]["source"]["H"] == "user"
    assert d["assumptions"]["g"] == 0.02 and d["assumptions"]["source"]["g"] == "user"


def test_unknown_zip_uses_state_medians(client):
    status, d = post(client, "/api/analyse", {**AUSTIN, "zip": "00000", "state": "TX"})
    assert status == 200
    assert not d["location"]["found"]
    assert set(d["location"]["source"].values()) == {"state_median"}
    assert any("State median" in w for w in d["warnings"])


def test_unknown_zip_without_state_is_rejected(client):
    status, d = post(client, "/api/analyse", {**AUSTIN, "zip": "00000"})
    assert status == 404 and "state" in d["error"] and d["disclaimer"] == DISCLAIMER


def test_zip_state_mismatch_uses_zip_state(client):
    status, d = post(client, "/api/analyse", {**AUSTIN, "state": "CA"})
    assert status == 200 and d["location"]["state"] == "TX"
    assert any("not CA" in w for w in d["warnings"])


def test_state_outside_sale_training_warns(client):
    status, d = post(client, "/api/analyse", {**AUSTIN, "zip": "82001"})   # Cheyenne, WY
    assert status == 200 and d["location"]["state"] == "WY"
    assert any("no WY listings" in w for w in d["warnings"])


def test_large_house_rent_extrapolation_warns(client):
    status, d = post(client, "/api/analyse", {**AUSTIN, "sqft": 9_000, "beds": 6})
    assert status == 200 and any("extrapolation" in w for w in d["warnings"])


@pytest.mark.parametrize("change, field", [
    ({"zip": "7870"}, "zip"), ({"zip": "ABCDE"}, "zip"), ({"beds": 0}, "beds"),
    ({"beds": 11}, "beds"), ({"baths": 2.3}, "baths"), ({"sqft": 20_000}, "sqft"),
    ({"sqft": 100}, "sqft"), ({"state": "XX"}, "state"), ({"price": -5}, "price"),
    ({"assumptions": {"d": 1.5}}, "assumptions.d"), ({"assumptions": {"H": 0}}, "assumptions.H"),
    ({"assumptions": {"r": 0.5}}, "assumptions.r"), ({"assumptions": {"bogus": 1}}, "assumptions.bogus"),
    ({"colour": "red"}, "colour"),
])
def test_invalid_input_rejected(client, change, field):
    status, d = post(client, "/api/analyse", {**AUSTIN, **change})
    assert status == 422
    assert any(det["field"] == field for det in d["details"]), d
    assert d["disclaimer"] == DISCLAIMER


def test_missing_field_rejected(client):
    status, d = post(client, "/api/analyse", {"zip": "78704", "beds": 3, "baths": 2})
    assert status == 422 and any(det["field"] == "sqft" for det in d["details"])


def test_non_json_body_rejected(client):
    r = client.post("/api/analyse", data="not json", content_type="text/plain")
    assert r.status_code == 400 and r.get_json()["disclaimer"] == DISCLAIMER


# ----------------------------------------------------------------------------- other endpoints
def test_compare(client):
    body = {"properties": [AUSTIN, {"zip": "94110", "beds": 2, "baths": 1, "sqft": 1000},
                           {"zip": "44101", "beds": 3, "baths": 1.5, "sqft": 1400, "state": "OH"}]}
    status, d = post(client, "/api/compare", body)
    assert status == 200 and len(d["properties"]) == 3
    deltas = [p["delta"] for p in d["properties"]]
    assert d["largest_buy_advantage_index"] == deltas.index(max(deltas))


@pytest.mark.parametrize("n", [1, 6])
def test_compare_needs_2_to_5(client, n):
    status, _ = post(client, "/api/compare", {"properties": [AUSTIN] * n})
    assert status == 422


def test_sensitivity(client):
    status, d = post(client, "/api/sensitivity", AUSTIN)
    assert status == 200 and len(d["bars"]) == 13
    swings = [b["swing"] for b in d["bars"]]
    assert swings == sorted(swings, reverse=True)
    g = next(b for b in d["bars"] if b["parameter"] == "g")
    _, a = post(client, "/api/analyse", AUSTIN)
    assert (g["low"], g["high"]) == pytest.approx((a["forecast"]["g"]["p10"], a["forecast"]["g"]["p90"]))


def test_scenario_with_custom_shock(client):
    status, d = post(client, "/api/scenario", {**AUSTIN, "custom_shock": {"mortgage": 2.0, "mortgage_chg12": 2.0}})
    assert status == 200
    names = [c["scenario"] for c in d["cases"]]
    assert names[:3] == ["base", "pessimistic", "optimistic"] and "custom" in names
    custom = next(c for c in d["cases"] if c["scenario"] == "custom")
    assert custom["r"] == pytest.approx(d["cases"][0]["r"] + 0.02)
    assert custom["g"] <= d["cases"][0]["g"] + 1e-12     # monotonic HPI model: higher rates never raise g


@pytest.mark.parametrize("shock", [{"bogus": 1.0}, {"mortgage": 50.0}])
def test_scenario_rejects_bad_shock(client, shock):
    status, _ = post(client, "/api/scenario", {**AUSTIN, "custom_shock": shock})
    assert status == 422


def test_forecast_endpoint(client):
    r = client.get("/api/forecast/ca")
    d = r.get_json()
    assert r.status_code == 200 and d["state"] == "CA"
    hp = d["house_price_growth"]
    assert hp["p10_pct"] < hp["forecast_pct"] < hp["p90_pct"]
    assert d["rent_growth_us"]["method"] == "arima" and d["disclaimer"] == DISCLAIMER


def test_forecast_unknown_state(client):
    r = client.get("/api/forecast/XX")
    assert r.status_code == 404 and r.get_json()["disclaimer"] == DISCLAIMER


def test_unknown_route_and_method(client):
    assert client.get("/api/nothing").status_code == 404
    assert client.post("/predict", json={}).status_code == 404     # legacy route removed
    assert client.get("/api/analyse").status_code == 405


def test_old_ppsq_model_not_loaded():
    import sys
    assert "preprocess" not in sys.modules and "backend.preprocess" not in sys.modules
