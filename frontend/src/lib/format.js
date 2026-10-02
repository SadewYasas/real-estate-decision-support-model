const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const money = (v) => (v == null || Number.isNaN(v) ? "–" : usd0.format(v));

/** $412k / $1.2M style, for chart axes and compact labels. */
export function moneyShort(v) {
    if (v == null || Number.isNaN(v)) return "–";
    const sign = v < 0 ? "-" : "";
    const a = Math.abs(v);
    if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
    if (a >= 1e3) return `${sign}$${Math.round(a / 1e3)}k`;
    return `${sign}$${Math.round(a)}`;
}

/** Fraction -> percent text: 0.0312 -> "3.1%". */
export const pct = (v, digits = 1) => (v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`);

/** Signed percent: +3.1% / -2.0%. */
export const signedPct = (v, digits = 1) =>
    v == null || Number.isNaN(v) ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(digits)}%`;

export const number = (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-US"));

export const METHOD_NAMES = {
    gbm_macro_constrained: "gradient boosting with economic indicators",
    gbm_macro: "gradient boosting with economic indicators",
    gbm_no_macro: "gradient boosting",
    arima: "ARIMA time-series model",
    persistence: "last year's growth",
};

export const methodName = (m) => METHOD_NAMES[m] || m;

/** Plain-language value for a SHAP feature row. */
export function featureValue(item) {
    const v = item.value;
    switch (item.feature) {
        case "zip_code_density":
            return `${number(v)} people / sq mi`;
        case "median_household_income":
            return money(v);
        case "living_space":
        case "square_feet":
            return `${number(v)} sq ft`;
        case "latitude":
        case "longitude":
            return Number(v).toFixed(2);
        default:
            return String(v);
    }
}
