import React, { useMemo, useState } from "react";
import { money, moneyShort, pct } from "../lib/format";
import { BUY, RENT } from "../lib/chart";

const LOW = BUY;    // blue: low value of the assumption
const HIGH = RENT;  // orange: high value

function fmtParam(p, v) {
    if (p === "P" || p === "R") return money(v);
    if (p === "H") return `${Math.round(v)} yrs`;
    return pct(v, 2);
}

function niceTicks(lo, hi, n = 5) {
    const span = hi - lo || 1;
    const step0 = span / n;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0);
    const ticks = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(t);
    return ticks;
}

export default function SensitivityTab({ data }) {
    const [hover, setHover] = useState(null);
    const { bars, base_delta: base } = data;
    const [lo, hi] = useMemo(() => {
        const vals = bars.flatMap((b) => [b.delta_low, b.delta_high]).concat([base, 0]);
        const a = Math.min(...vals), b = Math.max(...vals), pad = (b - a) * 0.04;
        return [a - pad, b + pad];
    }, [bars, base]);
    const x = (v) => ((v - lo) / (hi - lo)) * 100;
    const ticks = niceTicks(lo, hi);

    return (
        <div>
            <p className="text-sm text-gray-700">
                How the result changes when one assumption at a time moves to a low or high value. The bar shows the
                advantage of buying (present-value cost of renting minus buying over {data.H} years); right of the
                dashed line favours buying. Growth ranges are the forecast's 80% range; price and rent ranges are the
                models' typical errors.
            </p>
            <div className="flex flex-wrap gap-4 text-xs text-gray-600 mt-3 mb-4">
                <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: LOW }} aria-hidden="true" />Low value</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-3 h-3 rounded-sm" style={{ background: HIGH }} aria-hidden="true" />High value</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-px h-3 bg-gray-900" aria-hidden="true" />Your result ({moneyShort(base)})</span>
                <span className="inline-flex items-center gap-1.5"><span className="w-px h-3 border-l border-dashed border-gray-500" aria-hidden="true" />Break-even (0)</span>
            </div>

            <ul className="space-y-1">
                {bars.map((b, i) => {
                    const bLo = x(Math.min(base, b.delta_low)), wLo = Math.abs(x(b.delta_low) - x(base));
                    const bHi = x(Math.min(base, b.delta_high)), wHi = Math.abs(x(b.delta_high) - x(base));
                    const active = hover === i;
                    return (
                        <li key={b.parameter} tabIndex={0}
                            onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                            onFocus={() => setHover(i)} onBlur={() => setHover(null)}
                            className={`grid grid-cols-1 sm:grid-cols-[12.5rem_1fr] gap-x-3 items-center rounded-lg px-2 py-1 outline-none ${active ? "bg-gray-50" : ""}`}
                            aria-label={`${b.label}: ${fmtParam(b.parameter, b.low)} gives ${money(b.delta_low)}, ${fmtParam(b.parameter, b.high)} gives ${money(b.delta_high)}`}>
                            <span className="text-sm text-gray-800 flex flex-col items-start leading-tight">
                                {b.label}
                                {b.flips_recommendation && (
                                    <span className="mt-0.5 text-[10px] font-semibold uppercase tracking-wide bg-amber-100 text-amber-900 rounded px-1.5 py-0.5 whitespace-nowrap">changes decision</span>
                                )}
                            </span>
                            <div className="relative h-7">
                                <div className="absolute inset-y-0 border-l border-dashed border-gray-400" style={{ left: `${x(0)}%` }} aria-hidden="true" />
                                <div className="absolute top-1 h-5" style={{ left: `${bLo}%`, width: `${wLo}%`, background: LOW, borderRadius: b.delta_low < base ? "4px 0 0 4px" : "0 4px 4px 0" }} aria-hidden="true" />
                                <div className="absolute top-1 h-5" style={{ left: `${bHi}%`, width: `${wHi}%`, background: HIGH, borderRadius: b.delta_high < base ? "4px 0 0 4px" : "0 4px 4px 0" }} aria-hidden="true" />
                                <div className="absolute inset-y-0 w-px bg-gray-900" style={{ left: `${x(base)}%` }} aria-hidden="true" />
                                {[[b.delta_low, b.low], [b.delta_high, b.high]].map(([d, v], k) => (
                                    <span key={k} className="absolute top-1.5 text-[11px] text-gray-600 whitespace-nowrap px-1 bg-white/80 rounded"
                                        style={d >= base ? { left: `calc(${x(d)}% + 2px)` } : { right: `calc(${100 - x(d)}% + 2px)` }}>
                                        {fmtParam(b.parameter, v)}
                                    </span>
                                ))}
                            </div>
                            {active && (
                                <p className="sm:col-start-2 text-xs text-gray-700 pb-1">
                                    {fmtParam(b.parameter, b.low)}: buying advantage {money(b.delta_low)}, break-even{" "}
                                    {b.break_even_low ? `year ${b.break_even_low}` : "never"} · {fmtParam(b.parameter, b.high)}:{" "}
                                    {money(b.delta_high)}, break-even {b.break_even_high ? `year ${b.break_even_high}` : "never"}
                                </p>
                            )}
                        </li>
                    );
                })}
            </ul>
            <div className="grid grid-cols-1 sm:grid-cols-[12.5rem_1fr] gap-x-3 px-2">
                <span className="hidden sm:block" />
                <div className="relative h-5 border-t border-gray-300 text-[11px] text-gray-600">
                    {ticks.map((t) => (
                        <span key={t} className="absolute top-1 -translate-x-1/2" style={{ left: `${x(t)}%` }}>{moneyShort(t)}</span>
                    ))}
                </div>
            </div>
        </div>
    );
}
