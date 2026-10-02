import React, { useState } from "react";
import { money, signedPct, featureValue } from "../lib/format";

const RAISE = "#2a78d6"; // blue: raises the estimate
const LOWER = "#e34948"; // red: lowers the estimate

/** Horizontal diverging bars of SHAP effects (top 5) for one model. */
function ShapChart({ title, exp, unit }) {
    const [hover, setHover] = useState(null);
    const maxAbs = Math.max(...exp.top.map((t) => Math.abs(t.shap_log)), 1e-9);

    return (
        <figure className="min-w-0">
            <figcaption className="text-sm font-semibold text-gray-900">{title}</figcaption>
            <p className="text-xs text-gray-500 mt-0.5 mb-3">
                Starts from the average estimate of {money(exp.baseline)}{unit}; each feature moves it up or down.
            </p>
            <ul className="space-y-2.5">
                {exp.top.map((t, i) => {
                    const w = (Math.abs(t.shap_log) / maxAbs) * 50; // % of the row; 50% = one side
                    const up = t.shap_log >= 0;
                    return (
                        <li key={t.feature} className="text-sm"
                            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                            onFocus={() => setHover(i)} onBlur={() => setHover(null)} tabIndex={0}
                            aria-label={`${t.label} ${featureValue(t)}: ${signedPct(t.effect_pct)}`}>
                            <div className="flex justify-between gap-2 text-gray-700">
                                <span className="truncate">{t.label} <span className="text-gray-500">({featureValue(t)})</span></span>
                                <span className="font-semibold tabular-nums text-gray-900">{signedPct(t.effect_pct)}</span>
                            </div>
                            <div className="relative h-5 mt-1 rounded bg-gray-50">
                                <div className="absolute inset-y-0 left-1/2 w-px bg-gray-300" aria-hidden="true" />
                                <div
                                    className="absolute inset-y-0.5 transition-opacity"
                                    style={{
                                        background: up ? RAISE : LOWER,
                                        width: `${w}%`,
                                        left: up ? "50%" : `${50 - w}%`,
                                        borderRadius: up ? "0 4px 4px 0" : "4px 0 0 4px",
                                        opacity: hover === null || hover === i ? 1 : 0.45,
                                    }}
                                    aria-hidden="true"
                                />
                            </div>
                        </li>
                    );
                })}
            </ul>
            <dl className="mt-3 text-sm text-gray-600 space-y-0.5">
                <div className="flex justify-between"><dt>All other features</dt><dd className="tabular-nums">{signedPct(exp.other_features_effect_pct)}</dd></div>
                <div className="flex justify-between font-semibold text-gray-900 border-t border-gray-200 pt-1">
                    <dt>This estimate</dt><dd className="tabular-nums">{money(exp.prediction)}{unit}</dd>
                </div>
            </dl>
        </figure>
    );
}

export default function ExplainPanel({ data }) {
    return (
        <div>
            <div className="flex flex-wrap gap-4 text-xs text-gray-600 mb-4" aria-hidden="true">
                <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: RAISE }} />raises the estimate</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: LOWER }} />lowers the estimate</span>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <ShapChart title="Price estimate" exp={data.price.explanation} unit="" />
                <ShapChart title={`Rent estimate (2019 level, before the ×${data.rent.cpi_factor.toFixed(2)} CPI update)`}
                    exp={data.rent.explanation} unit="/mo" />
            </div>
            <p className="mt-4 text-xs text-gray-500">
                SHAP values (TreeSHAP) for the five most influential features. Effects multiply: they are
                percentage changes on the model's log scale, so they combine exactly into the estimate.
            </p>
        </div>
    );
}
