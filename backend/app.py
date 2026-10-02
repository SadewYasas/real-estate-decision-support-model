"""Flask API for the rent-vs-buy decision support system.

Endpoints (CLAUDE.md API contract):
  POST /api/analyse         one property: price, rent, forecasts, rent-vs-buy result, SHAP
  POST /api/compare         2-5 properties side by side
  POST /api/sensitivity     tornado data for one property
  POST /api/scenario        base / pessimistic / optimistic / macro-shock cases
  GET  /api/forecast/<state>  latest house price and rent growth forecasts
  GET  /api/health

All models are loaded once at start-up. Every response carries the disclaimer.
The old PPSq-based model has been removed from the API (kept in legacy/backend/ for reference).

Run from the project root:  python backend/app.py    (http://localhost:5000)
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request  # noqa: E402
from flask_cors import CORS  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from backend.schemas import CompareIn, PropertyIn, ScenarioIn  # noqa: E402
from src.api.service import AnalysisService, NotFound, to_jsonable  # noqa: E402
from src.engine.rent_vs_buy import DISCLAIMER  # noqa: E402

app = Flask(__name__)
CORS(app)

_t0 = time.perf_counter()
service = AnalysisService()
STARTUP_SECONDS = time.perf_counter() - _t0


def ok(payload: dict, status: int = 200):
    payload = to_jsonable(payload)
    payload.setdefault("disclaimer", DISCLAIMER)
    return jsonify(payload), status


def error(status: int, message: str, details=None):
    body = {"error": message, "disclaimer": DISCLAIMER}
    if details is not None:
        body["details"] = details
    return jsonify(body), status


def body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def as_property(p) -> dict:
    d = p.model_dump()
    d["assumptions"] = p.assumptions.model_dump() if p.assumptions else None
    return d


@app.errorhandler(ValidationError)
def validation_failed(e: ValidationError):
    details = [{"field": ".".join(str(x) for x in err["loc"]), "message": err["msg"]} for err in e.errors()]
    return error(422, "invalid input", details)


@app.errorhandler(NotFound)
def not_found(e):
    return error(404, str(e))


@app.errorhandler(ValueError)
def bad_request(e):
    return error(400, str(e))


@app.errorhandler(404)
def no_route(e):
    return error(404, "no such endpoint")


@app.errorhandler(405)
def wrong_method(e):
    return error(405, "method not allowed")


@app.errorhandler(Exception)
def unexpected(e):
    app.logger.exception(e)
    return error(500, "internal error")


@app.get("/api/health")
def health():
    return ok({"status": "ok", "startup_seconds": round(STARTUP_SECONDS, 3)})


@app.post("/api/analyse")
def api_analyse():
    return ok(service.analyse(as_property(PropertyIn.model_validate(body()))))


@app.post("/api/compare")
def api_compare():
    req = CompareIn.model_validate(body())
    return ok(service.compare([as_property(p) for p in req.properties]))


@app.post("/api/sensitivity")
def api_sensitivity():
    return ok(service.sensitivity(as_property(PropertyIn.model_validate(body()))))


@app.post("/api/scenario")
def api_scenario():
    req = ScenarioIn.model_validate(body())
    prop = as_property(req)
    shock = prop.pop("custom_shock", None)
    return ok(service.scenario(prop, custom_shock=shock))


@app.get("/api/forecast/<state>")
def api_forecast(state: str):
    return ok(service.forecast(state.strip().upper()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
