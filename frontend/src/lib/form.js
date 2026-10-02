// Property form: validation limits, advanced assumptions and the request builder.

// Ranges match the API validation (the sale data the models were trained on).
const LIMITS = { beds: [1, 10], baths: [1, 10], sqft: [300, 15000] };

// Advanced assumptions: shown as percentages, sent to the API as fractions.
export const ADVANCED = [
    { key: "d", label: "Down payment", unit: "%", min: 0, max: 100, step: 1, placeholder: "20" },
    { key: "r", label: "Mortgage rate", unit: "%", min: 0, max: 25, step: 0.05, placeholder: "latest (6.35)" },
    { key: "T", label: "Mortgage term", unit: "years", min: 1, max: 40, step: 1, placeholder: "30", whole: true },
    { key: "H", label: "Years you'd stay", unit: "years", min: 1, max: 30, step: 1, placeholder: "7", whole: true },
    { key: "k", label: "Discount rate", unit: "%", min: 0, max: 20, step: 0.1, placeholder: "5" },
    { key: "tau", label: "Property tax", unit: "% of value", min: 0, max: 10, step: 0.05, placeholder: "1" },
    { key: "m", label: "Maintenance", unit: "% of value", min: 0, max: 10, step: 0.05, placeholder: "1" },
    { key: "h", label: "Insurance", unit: "% of value", min: 0, max: 5, step: 0.05, placeholder: "0.35" },
    { key: "cb", label: "Buying costs", unit: "% of price", min: 0, max: 20, step: 0.1, placeholder: "3" },
    { key: "cs", label: "Selling costs", unit: "% of price", min: 0, max: 20, step: 0.1, placeholder: "6" },
];

/** Builds the API request body from the form; returns { body, errors }. */
export function toRequest(form) {
    const errors = {};
    const zip = form.zip.trim();
    if (!/^\d{5}$/.test(zip)) errors.zip = "Enter a 5-digit ZIP code";
    const num = (k) => Number(form[k]);
    for (const [k, [lo, hi]] of Object.entries(LIMITS)) {
        const v = num(k);
        if (form[k] === "" || Number.isNaN(v) || v < lo || v > hi) errors[k] = `Between ${lo.toLocaleString()} and ${hi.toLocaleString()}`;
    }
    if (form.baths !== "" && Math.round(num("baths") * 2) !== num("baths") * 2) errors.baths = "Whole or half numbers only";
    if (form.beds !== "" && !Number.isInteger(num("beds"))) errors.beds = "Whole number";

    const assumptions = {};
    for (const a of ADVANCED) {
        const raw = form.advanced[a.key];
        if (raw === "" || raw == null) continue;
        const v = Number(raw);
        if (Number.isNaN(v) || v < a.min || v > a.max || (a.whole && !Number.isInteger(v))) {
            errors[a.key] = `Between ${a.min} and ${a.max}${a.whole ? ", whole years" : ""}`;
            continue;
        }
        assumptions[a.key] = a.unit.startsWith("%") ? v / 100 : v;
    }
    const body = { zip, beds: num("beds"), baths: num("baths"), sqft: num("sqft") };
    if (form.state) body.state = form.state;
    if (form.price) body.price = Number(form.price);
    if (form.monthly_rent) body.monthly_rent = Number(form.monthly_rent);
    if (Object.keys(assumptions).length) body.assumptions = assumptions;
    return { body, errors };
}

export const EMPTY_FORM = { zip: "", state: "", beds: "3", baths: "2", sqft: "1800", price: "", monthly_rent: "", advanced: {} };
