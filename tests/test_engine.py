"""Tests for the rent-vs-buy engine.

1. Manual scenarios (src/engine/manual_scenarios.py): engine output vs hand-calculated values.
2. Every manual scenario vs an independent reference calculation written here: the loan is
   simulated month by month (balance = balance x (1+i) - M) with the payment found from
   the annuity sum, not from the closed forms the engine uses.
3. Properties: the loan is repaid exactly at T, Delta rises with appreciation and falls
   with the mortgage rate, input validation, and the disclaimer.

Results per manual scenario are written to artefacts/metrics/engine_tests.json by
tests/conftest.py.
"""
import pytest

from src.engine.manual_scenarios import SCENARIOS
from src.engine.rent_vs_buy import (DISCLAIMER, Assumptions, analyse, balance, delta,
                                    monthly_payment)

MONEY_TOL = 0.01          # cents, for hand-calculated values
REL_TOL = 1e-9            # engine vs reference calculation


# ----------------------------------------------------------------------------- reference
def reference(P: float, R: float, a: Assumptions, H: int) -> dict:
    """Independent month-by-month calculation of C_B(H), C_R(H) and Delta(H)."""
    L = (1 - a.d) * P
    i, n = a.r / 12, 12 * a.T
    if L == 0:
        M = 0.0
    elif i == 0:
        M = L / n
    else:
        M = L / sum((1 + i) ** -j for j in range(1, n + 1))   # annuity: PV of payments = L
    bal, value, rent = L, P, 12 * R
    pv_owner = pv_rent = 0.0
    for t in range(1, H + 1):
        paid = 0.0
        for month in range(12):
            if bal > 1e-9 and (t - 1) * 12 + month < n:
                bal = bal * (1 + i) - M
                paid += M
        if (t - 1) * 12 >= n:
            bal = 0.0
        running = (a.tau + a.m + a.h) * value          # on the value at the start of year t
        pv_owner += (paid + running) / (1 + a.k) ** t
        pv_rent += rent / (1 + a.k) ** t
        value *= 1 + a.g
        rent *= 1 + a.q
    bal = max(bal, 0.0) if abs(bal) < 1e-6 else bal
    cost_buy = (a.d + a.cb) * P + pv_owner - (value * (1 - a.cs) - bal) / (1 + a.k) ** H
    return {"monthly_payment": M, "balance_H": bal, "cost_buy": cost_buy,
            "cost_rent": pv_rent, "delta": pv_rent - cost_buy}


def reference_break_even(P, R, a):
    for H in range(1, 31):
        if reference(P, R, a, H)["delta"] >= 0:
            return H
    return None


def engine_values(s: dict) -> dict:
    a = Assumptions(**s["a"])
    r = analyse(s["P"], s["R"], a)
    return {"monthly_payment": r["monthly_payment"], "balance_H": balance(s["P"], a, a.H),
            "cost_buy": r["cost_buy"], "cost_rent": r["cost_rent"], "delta": r["delta"],
            "break_even_year": r["break_even_year"], "recommendation": r["recommendation"],
            "user_cost": r["user_cost"]}


def same(expected, got) -> bool:
    if isinstance(expected, (str, type(None))) or isinstance(got, (str, type(None))):
        return expected == got
    return abs(expected - got) <= MONEY_TOL


# ----------------------------------------------------------------------------- 1 + 2
@pytest.mark.parametrize("s", SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def test_manual_scenario(s, record_property):
    got = engine_values(s)
    a = Assumptions(**s["a"])
    ref = reference(s["P"], s["R"], a, a.H)
    ref["break_even_year"] = reference_break_even(s["P"], s["R"], a)

    checks = []
    for k, v in s["expected"].items():
        checks.append({"check": f"{k} vs hand calculation", "expected": v, "engine": got[k],
                       "pass": same(v, got[k])})
    for k in ("monthly_payment", "cost_buy", "cost_rent", "delta"):
        ok = abs(ref[k] - got[k]) <= max(REL_TOL * max(abs(ref[k]), 1.0), 1e-6)
        checks.append({"check": f"{k} vs independent monthly calculation", "expected": ref[k],
                       "engine": got[k], "pass": ok})
    checks.append({"check": "break_even_year vs independent monthly calculation",
                   "expected": ref["break_even_year"], "engine": got["break_even_year"],
                   "pass": ref["break_even_year"] == got["break_even_year"]})
    record_property("scenario", {"id": s["id"], "name": s["name"], "hand": s["hand"],
                                 "checks": checks})
    failed = [c for c in checks if not c["pass"]]
    assert not failed, failed


# ----------------------------------------------------------------------------- 3 properties
BASE = Assumptions(r=0.06, g=0.03, q=0.03)


@pytest.mark.parametrize("r", [0.0, 0.03, 0.065, 0.10])
@pytest.mark.parametrize("T", [5, 15, 30])
def test_loan_repaid_exactly_at_term(r, T):
    a = BASE.replace(r=r, T=T)
    assert balance(300_000, a, T) == 0
    # The closed form one month before the end leaves exactly one payment outstanding.
    i, M, L = r / 12, monthly_payment(300_000, a), 0.8 * 300_000
    left = L - M * (12 * T - 1) if i == 0 else L * (1 + i) ** (12 * T - 1) - M * ((1 + i) ** (12 * T - 1) - 1) / i
    assert left == pytest.approx(M / (1 + i), rel=1e-9)


def test_balance_never_negative_after_term():
    a = BASE.replace(T=10, H=20)
    assert all(balance(300_000, a, t) >= 0 for t in range(0, 31))


def test_delta_increases_with_appreciation():
    ds = [delta(400_000, 2_000, BASE.replace(g=g), 7) for g in (-0.02, 0.0, 0.03, 0.06)]
    assert ds == sorted(ds)


def test_delta_falls_with_mortgage_rate():
    ds = [delta(400_000, 2_000, BASE.replace(r=r), 7) for r in (0.03, 0.05, 0.07, 0.09)]
    assert ds == sorted(ds, reverse=True)


def test_yearly_table_matches_horizon_results():
    a = BASE.replace(H=12)
    res = analyse(400_000, 2_000, a)
    assert len(res["yearly"]) == 30
    assert res["yearly"][11]["delta"] == pytest.approx(res["delta"])
    assert res["yearly"][11]["delta"] == pytest.approx(delta(400_000, 2_000, a, 12))


def test_break_even_consistent_with_yearly_table():
    res = analyse(400_000, 2_000, BASE)
    be = res["break_even_year"]
    deltas = [row["delta"] for row in res["yearly"]]
    assert be is not None and deltas[be - 1] >= 0 and all(d < 0 for d in deltas[: be - 1])


@pytest.mark.parametrize("P, R, change", [
    (0, 1_000, {}), (300_000, 0, {}), (300_000, 1_000, {"d": 1.2}), (300_000, 1_000, {"d": -0.1}),
    (300_000, 1_000, {"T": 0}), (300_000, 1_000, {"H": 0}), (300_000, 1_000, {"H": 2.5}),
    (300_000, 1_000, {"r": -0.01}), (300_000, 1_000, {"g": -1.0}), (300_000, 1_000, {"cs": 1.5}),
])
def test_invalid_inputs_rejected(P, R, change):
    with pytest.raises(ValueError):
        analyse(P, R, BASE.replace(**change))


def test_disclaimer_in_every_result():
    assert analyse(400_000, 2_000, BASE)["disclaimer"] == DISCLAIMER
    assert "not financial or valuation advice" in DISCLAIMER
