"""Rent-versus-buy NPV engine (pure Python, no ML). Formulas exactly as in CLAUDE.md.

Notation: P price, R monthly rent, g appreciation, q rent growth, d down payment,
r mortgage rate, T term (years), tau property tax, m maintenance, h insurance,
cb / cs buying / selling costs, k discount rate, H holding period (years).
All rates are annual fractions (0.05 = 5%).

    L = (1-d)P ; i = r/12 ; n = 12T ; M = L*i*(1+i)^n / ((1+i)^n - 1)
    V_t = P(1+g)^t
    B_t = L(1+i)^(12t) - M((1+i)^(12t) - 1)/i
    O_t = 12M*[t<=T] + (tau+m+h)*V_{t-1}
    Rent_t = 12R(1+q)^(t-1)
    C_B(H) = (d+cb)P + sum_{t=1..H} O_t/(1+k)^t - (V_H(1-cs) - B_H)/(1+k)^H
    C_R(H) = sum_{t=1..H} Rent_t/(1+k)^t
    Delta(H) = C_R - C_B ; break-even = min H in 1..30 with Delta >= 0
    User cost UC = (k+tau+m+h-g)*P, compared with 12R

Edge cases (limits of the same formulas, stated so they can be tested):
- r = 0: M = L/n and B_t = L - 12tM;
- d = 1: L = 0, so M = 0 and B_t = 0;
- t >= T: the loan is repaid, B_t = 0 (the closed form gives 0 at t = T; it is not
  extended past T, where it would turn negative).

Recommendation at horizon H: "buy" if Delta(H) >= 0 (renting costs at least as much in
present value), otherwise "rent".
"""
from dataclasses import asdict, dataclass

DISCLAIMER = "Indicative decision support only – not financial or valuation advice"
BREAK_EVEN_SEARCH_YEARS = 30


@dataclass(frozen=True)
class Assumptions:
    """User assumptions; defaults from CLAUDE.md. r, g and q have no defaults here: the
    caller supplies the latest mortgage rate and the forecasts."""
    r: float                 # mortgage rate
    g: float                 # house price appreciation
    q: float                 # rent growth
    d: float = 0.20          # down payment share
    T: int = 30              # mortgage term, years
    H: int = 7               # holding period, years
    tau: float = 0.010       # property tax, share of value
    m: float = 0.010         # maintenance, share of value
    h: float = 0.0035        # insurance, share of value
    cb: float = 0.03         # buying costs, share of price
    cs: float = 0.06         # selling costs, share of sale value
    k: float = 0.05          # discount rate

    def replace(self, **changes) -> "Assumptions":
        return Assumptions(**{**asdict(self), **changes})


def validate(P: float, R: float, a: Assumptions):
    if not P > 0:
        raise ValueError("price P must be positive")
    if not R > 0:
        raise ValueError("monthly rent R must be positive")
    if not 0 <= a.d <= 1:
        raise ValueError("down payment d must be between 0 and 1")
    if int(a.T) != a.T or a.T < 1:
        raise ValueError("mortgage term T must be a whole number of years >= 1")
    if int(a.H) != a.H or not 1 <= a.H <= 50:
        raise ValueError("holding period H must be a whole number of years from 1 to 50")
    if a.r < 0:
        raise ValueError("mortgage rate r cannot be negative")
    for name in ("g", "q", "k"):
        if getattr(a, name) <= -1:
            raise ValueError(f"{name} must be greater than -100%")
    for name in ("tau", "m", "h", "cb", "cs"):
        if not 0 <= getattr(a, name) < 1:
            raise ValueError(f"{name} must be between 0 and 1")


# ----------------------------------------------------------------------------- formulas
def loan(P: float, a: Assumptions) -> float:
    return (1 - a.d) * P


def monthly_payment(P: float, a: Assumptions) -> float:
    L = loan(P, a)
    n = 12 * a.T
    if L == 0:
        return 0.0
    i = a.r / 12
    if i == 0:
        return L / n
    return L * i * (1 + i) ** n / ((1 + i) ** n - 1)


def balance(P: float, a: Assumptions, t: int) -> float:
    """Outstanding loan after t years (B_t)."""
    L = loan(P, a)
    if L == 0 or t >= a.T:
        return 0.0
    M = monthly_payment(P, a)
    i = a.r / 12
    if i == 0:
        return L - 12 * t * M
    return L * (1 + i) ** (12 * t) - M * ((1 + i) ** (12 * t) - 1) / i


def value(P: float, a: Assumptions, t: int) -> float:
    return P * (1 + a.g) ** t


def owner_cost(P: float, a: Assumptions, t: int) -> float:
    """O_t: mortgage payments in year t (while t <= T) plus tax, maintenance and insurance
    on the value at the start of the year."""
    mortgage = 12 * monthly_payment(P, a) if t <= a.T else 0.0
    return mortgage + (a.tau + a.m + a.h) * value(P, a, t - 1)


def rent_cost(R: float, a: Assumptions, t: int) -> float:
    return 12 * R * (1 + a.q) ** (t - 1)


def discount(a: Assumptions, t: int) -> float:
    return (1 + a.k) ** t


def cost_buy(P: float, a: Assumptions, H: int) -> float:
    """C_B(H): present value of owning for H years net of the sale proceeds."""
    upfront = (a.d + a.cb) * P
    running = sum(owner_cost(P, a, t) / discount(a, t) for t in range(1, H + 1))
    equity = (value(P, a, H) * (1 - a.cs) - balance(P, a, H)) / discount(a, H)
    return upfront + running - equity


def cost_rent(R: float, a: Assumptions, H: int) -> float:
    """C_R(H): present value of renting for H years."""
    return sum(rent_cost(R, a, t) / discount(a, t) for t in range(1, H + 1))


def delta(P: float, R: float, a: Assumptions, H: int) -> float:
    """Delta(H) = C_R - C_B. Positive: buying is cheaper in present value."""
    return cost_rent(R, a, H) - cost_buy(P, a, H)


def break_even_year(P: float, R: float, a: Assumptions,
                    max_years: int = BREAK_EVEN_SEARCH_YEARS) -> int | None:
    """Smallest H (1..max_years) with Delta(H) >= 0, or None if buying never breaks even."""
    for H in range(1, max_years + 1):
        if delta(P, R, a, H) >= 0:
            return H
    return None


def user_cost(P: float, a: Assumptions) -> float:
    """Annual user cost of owning, UC = (k + tau + m + h - g) P."""
    return (a.k + a.tau + a.m + a.h - a.g) * P


# ----------------------------------------------------------------------------- analysis
def analyse(P: float, R: float, a: Assumptions) -> dict:
    """Full result for one property: yearly table to max(H, 30), break-even, recommendation."""
    validate(P, R, a)
    years = max(a.H, BREAK_EVEN_SEARCH_YEARS)
    rows, pv_owner, pv_rent = [], 0.0, 0.0
    for t in range(1, years + 1):
        o, rt = owner_cost(P, a, t), rent_cost(R, a, t)
        pv_owner += o / discount(a, t)
        pv_rent += rt / discount(a, t)
        c_b = (a.d + a.cb) * P + pv_owner - (value(P, a, t) * (1 - a.cs) - balance(P, a, t)) / discount(a, t)
        rows.append({
            "year": t,
            "home_value": value(P, a, t),
            "loan_balance": balance(P, a, t),
            "owner_cost": o,
            "rent_cost": rt,
            "pv_cost_buy": c_b,       # C_B(t): selling at the end of year t
            "pv_cost_rent": pv_rent,  # C_R(t)
            "delta": pv_rent - c_b,
        })
    d_H = rows[a.H - 1]["delta"]
    uc = user_cost(P, a)
    return {
        "inputs": {"price": P, "monthly_rent": R, **asdict(a)},
        "monthly_payment": monthly_payment(P, a),
        "loan": loan(P, a),
        "cost_buy": rows[a.H - 1]["pv_cost_buy"],
        "cost_rent": rows[a.H - 1]["pv_cost_rent"],
        "delta": d_H,
        "recommendation": "buy" if d_H >= 0 else "rent",
        "break_even_year": break_even_year(P, R, a),
        "user_cost": uc,
        "annual_rent": 12 * R,
        "user_cost_says": "buy" if uc <= 12 * R else "rent",
        "yearly": rows,
        "disclaimer": DISCLAIMER,
    }
