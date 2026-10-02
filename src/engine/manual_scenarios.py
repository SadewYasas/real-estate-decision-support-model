"""Manual test scenarios for the rent-vs-buy engine.

One source for tests/test_engine.py, the Excel check workbook (export_excel.py) and
artefacts/metrics/engine_tests.json. Scenarios S01-S06 and S08 are small enough to work
out by hand; the working is in "hand". S07, S09 and S11 use realistic numbers: S07 checks
the mortgage payment against the standard published value, and all of them are checked
against an independent month-by-month reference calculation and the Excel formulas.
"""

# Assumptions with every cost switched off; scenarios switch on what they test.
ZERO = dict(d=1.0, r=0.0, T=30, H=1, g=0.0, q=0.0, tau=0.0, m=0.0, h=0.0, cb=0.0, cs=0.0, k=0.0)
DEFAULTS = dict(d=0.20, T=30, H=7, tau=0.010, m=0.010, h=0.0035, cb=0.03, cs=0.06, k=0.05)

SCENARIOS = [
    {
        "id": "S01", "name": "Cash purchase, zero growth, no costs",
        "P": 100_000, "R": 500, "a": {**ZERO, "H": 1},
        "expected": {"monthly_payment": 0.0, "cost_buy": 0.0, "cost_rent": 6_000.0,
                     "delta": 6_000.0, "break_even_year": 1, "recommendation": "buy",
                     "user_cost": 0.0},
        "hand": "C_B = (1+0)100,000 + 0 - (100,000 x 1 - 0) = 0. C_R = 12 x 500 = 6,000. "
                "Delta = 6,000 >= 0 in year 1.",
    },
    {
        "id": "S02", "name": "Zero growth with running and transaction costs (100% down)",
        "P": 200_000, "R": 1_000,
        "a": {**ZERO, "H": 5, "tau": 0.01, "m": 0.01, "cb": 0.03, "cs": 0.06, "r": 0.06},
        "expected": {"monthly_payment": 0.0, "cost_buy": 38_000.0, "cost_rent": 60_000.0,
                     "delta": 22_000.0, "break_even_year": 3, "recommendation": "buy",
                     "user_cost": 4_000.0},
        "hand": "Upfront (1+0.03)200,000 = 206,000. Running 0.02 x 200,000 = 4,000 a year. "
                "Sale 200,000 x 0.94 = 188,000. C_B(5) = 206,000 + 20,000 - 188,000 = 38,000. "
                "C_R(5) = 60,000. In general Delta(H) = 12,000H - (18,000 + 4,000H) = "
                "8,000H - 18,000: H=2 gives -2,000, H=3 gives +6,000, so break-even = 3. "
                "The mortgage rate is irrelevant because nothing is borrowed.",
    },
    {
        "id": "S03", "name": "Interest-free mortgage, holding period shorter than term",
        "P": 120_000, "R": 700, "a": {**ZERO, "d": 0.25, "T": 15, "H": 5},
        "expected": {"monthly_payment": 500.0, "balance_H": 60_000.0, "cost_buy": 0.0,
                     "cost_rent": 42_000.0, "delta": 42_000.0, "break_even_year": 1,
                     "recommendation": "buy", "user_cost": 0.0},
        "hand": "L = 0.75 x 120,000 = 90,000; r = 0 so M = 90,000 / 180 = 500. "
                "B_5 = 90,000 - 60 x 500 = 60,000. O_t = 6,000. "
                "C_B(5) = 30,000 + 30,000 - (120,000 - 60,000) = 0. C_R(5) = 5 x 8,400 = 42,000.",
    },
    {
        "id": "S04", "name": "Holding period longer than mortgage term (H > T)",
        "P": 120_000, "R": 700, "a": {**ZERO, "d": 0.25, "T": 5, "H": 7},
        "expected": {"monthly_payment": 1_500.0, "balance_H": 0.0, "cost_buy": 0.0,
                     "cost_rent": 58_800.0, "delta": 58_800.0, "break_even_year": 1,
                     "recommendation": "buy", "user_cost": 0.0},
        "hand": "M = 90,000 / 60 = 1,500; O_t = 18,000 for t = 1..5 and 0 for t = 6, 7 "
                "(loan repaid, no running costs). B_7 = 0. "
                "C_B(7) = 30,000 + 90,000 - 120,000 = 0. C_R(7) = 7 x 8,400 = 58,800.",
    },
    {
        "id": "S05", "name": "100% down payment with appreciation and discounting",
        "P": 100_000, "R": 1_000, "a": {**ZERO, "r": 0.065, "g": 0.10, "k": 0.10, "H": 2},
        "expected": {"monthly_payment": 0.0, "cost_buy": 0.0, "cost_rent": 20_826.45,
                     "delta": 20_826.45, "break_even_year": 1, "recommendation": "buy",
                     "user_cost": 0.0},
        "hand": "V_2 = 100,000 x 1.1^2 = 121,000; C_B(2) = 100,000 - 121,000 / 1.21 = 0. "
                "C_R(2) = 12,000/1.1 + 12,000/1.21 = 10,909.09 + 9,917.36 = 20,826.45. "
                "UC = (0.10 - 0.10) x P = 0.",
    },
    {
        "id": "S06", "name": "Rent growth compounding",
        "P": 500_000, "R": 1_000, "a": {**ZERO, "q": 0.05, "H": 3},
        "expected": {"cost_buy": 0.0, "cost_rent": 37_830.0, "delta": 37_830.0,
                     "break_even_year": 1, "recommendation": "buy"},
        "hand": "Rent_1 = 12,000; Rent_2 = 12,000 x 1.05 = 12,600; Rent_3 = 12,000 x 1.05^2 = "
                "13,230. C_R(3) = 37,830 (k = 0). C_B = 500,000 - 500,000 = 0.",
    },
    {
        "id": "S07", "name": "Standard 30-year mortgage (published payment check)",
        "P": 300_000, "R": 1_500, "a": {**DEFAULTS, "r": 0.06, "g": 0.03, "q": 0.03},
        "expected": {"monthly_payment": 1_438.92},
        "hand": "L = 240,000 at 6% over 30 years: M = 240,000 x 0.005 x 1.005^360 / "
                "(1.005^360 - 1) = 1,438.92, the standard published value. Costs and "
                "break-even are checked against the independent monthly calculation and Excel.",
    },
    {
        "id": "S08", "name": "Buying never breaks even (price very high relative to rent)",
        "P": 1_000_000, "R": 1_000, "a": {**DEFAULTS, "r": 0.06, "g": 0.0, "q": 0.0},
        "expected": {"break_even_year": None, "recommendation": "rent",
                     "user_cost": 73_500.0},
        "hand": "C_R(30) = 12,000 x 15.372 (30-year annuity factor at 5%) = 184,469. "
                "C_B(30) is at least the upfront 230,000 minus the discounted sale "
                "proceeds 940,000 / 1.05^30 = 217,497, plus positive running costs, so "
                "C_B > 12,503 + PV(mortgage 57,552 a year) >> C_R for every H <= 30. "
                "UC = (0.05 + 0.0235 - 0) x 1,000,000 = 73,500 > 12,000.",
    },
    {
        "id": "S09", "name": "User cost check with default assumptions",
        "P": 300_000, "R": 1_200, "a": {**DEFAULTS, "r": 0.06, "g": 0.03, "q": 0.03},
        "expected": {"user_cost": 13_050.0},
        "hand": "UC = (0.05 + 0.01 + 0.01 + 0.0035 - 0.03) x 300,000 = 0.0435 x 300,000 = "
                "13,050 < 12 x 1,200 = 14,400, so the user-cost rule says buy.",
    },
    {
        "id": "S10", "name": "Negative appreciation",
        "P": 200_000, "R": 1_000, "a": {**ZERO, "g": -0.05, "H": 2},
        "expected": {"cost_buy": 19_500.0, "cost_rent": 24_000.0, "delta": 4_500.0,
                     "break_even_year": 1, "recommendation": "buy", "user_cost": 10_000.0},
        "hand": "V_2 = 200,000 x 0.95^2 = 180,500; C_B(2) = 200,000 - 180,500 = 19,500. "
                "C_R(2) = 24,000. Year 1: C_B = 10,000 < C_R = 12,000. "
                "UC = (0 - (-0.05)) x 200,000 = 10,000.",
    },
    {
        "id": "S11", "name": "Typical case: defaults, latest mortgage rate and forecasts",
        "P": 400_000, "R": 2_000, "a": {**DEFAULTS, "r": 0.063525, "g": 0.0378, "q": 0.0357},
        "expected": {},
        "hand": "No closed form; checked against the independent monthly calculation and "
                "the Excel formulas (r = Sep 2025 mortgage rate, g and q = mean forecasts).",
    },
]
