import React, { useState } from "react";
import {
    CartesianGrid, Legend, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { RotateCcw } from "lucide-react";
import { money, moneyShort, pct } from "../lib/format";
import { BUY, RENT, SLIDERS } from "../lib/chart";
import { Spinner } from "./ui";

const GRID = "#e4e3df";
const INK = "#52514e";

const toDisplay = (s, v) => (s.unit === "%" ? +(v * 100).toFixed(2) : v);
const fromDisplay = (s, v) => (s.unit === "%" ? v / 100 : v);

function CostTooltip({ active, payload, label }) {
    if (!active || !payload?.length) return null;
    const row = payload[0].payload;
    return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-md px-3 py-2 text-xs">
            <p className="font-semibold text-gray-900 mb-1">Selling / stopping after year {label}</p>
            {[["Buy", row.buy, BUY], ["Rent", row.rent, RENT]].map(([name, v, c]) => (
                <p key={name} className="flex items-center gap-2 text-gray-600">
                    <span className="inline-block w-3 h-0.5" style={{ background: c }} aria-hidden="true" />
                    <span className="font-semibold text-gray-900 tabular-nums">{money(v)}</span> {name}
                </p>
            ))}
            <p className="mt-1 text-gray-700">{row.rent >= row.buy ? "Buying" : "Renting"} cheaper by{" "}
                <span className="font-semibold">{money(Math.abs(row.rent - row.buy))}</span></p>
        </div>
    );
}

function CostChart({ yearly, H, breakEven }) {
    const [table, setTable] = useState(false);
    const rows = yearly.map((y) => ({ year: y.year, buy: y.pv_cost_buy, rent: y.pv_cost_rent }));
    const be = breakEven ? rows[breakEven - 1] : null;
    return (
        <figure>
            <div className="flex items-center justify-between gap-2">
                <figcaption className="text-sm font-semibold text-gray-900">Total cost over time, in today's money</figcaption>
                <button onClick={() => setTable((t) => !t)} className="text-xs font-semibold text-indigo-600 hover:underline cursor-pointer">
                    {table ? "Show chart" : "Show as table"}
                </button>
            </div>
            <p className="text-xs text-gray-500 mb-2">
                Net present cost if you sell (buy) or stop renting at the end of each year. Buying includes the
                down payment and costs, minus what you'd get from selling.
            </p>
            {table ? (
                <div className="max-h-72 overflow-auto border border-gray-200 rounded-lg">
                    <table className="w-full text-sm tabular-nums">
                        <thead className="bg-gray-50 sticky top-0">
                            <tr className="text-left text-gray-600">
                                <th className="px-3 py-2 font-semibold">Year</th>
                                <th className="px-3 py-2 font-semibold text-right">Buy</th>
                                <th className="px-3 py-2 font-semibold text-right">Rent</th>
                                <th className="px-3 py-2 font-semibold text-right">Rent − Buy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows.map((r) => (
                                <tr key={r.year} className={`border-t border-gray-100 ${r.year === H ? "bg-indigo-50 font-semibold" : ""}`}>
                                    <td className="px-3 py-1.5">{r.year}</td>
                                    <td className="px-3 py-1.5 text-right">{money(r.buy)}</td>
                                    <td className="px-3 py-1.5 text-right">{money(r.rent)}</td>
                                    <td className="px-3 py-1.5 text-right">{money(r.rent - r.buy)}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: 0 }}>
                            <CartesianGrid stroke={GRID} vertical={false} />
                            <XAxis dataKey="year" tick={{ fontSize: 12, fill: INK }} tickLine={false}
                                axisLine={{ stroke: GRID }} ticks={[1, 5, 10, 15, 20, 25, 30]} interval={0}
                                tickFormatter={(v) => `yr ${v}`} />
                            <YAxis tickFormatter={moneyShort} tick={{ fontSize: 12, fill: INK }} tickLine={false}
                                axisLine={false} width={56} />
                            <Tooltip content={<CostTooltip />} cursor={{ stroke: INK, strokeWidth: 1 }} />
                            <Legend verticalAlign="bottom" align="left" height={24} iconType="plainline"
                                wrapperStyle={{ fontSize: 12, color: INK }} />
                            <ReferenceLine x={H} stroke={INK} strokeDasharray="3 3"
                                label={{ value: `stay ${H} yrs`, position: "top", fontSize: 11, fill: INK }} />
                            <Line type="monotone" dataKey="buy" name="Buy" stroke={BUY} strokeWidth={2} dot={false}
                                activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2 }} isAnimationActive={false} />
                            <Line type="monotone" dataKey="rent" name="Rent" stroke={RENT} strokeWidth={2} dot={false}
                                activeDot={{ r: 5, stroke: "#fff", strokeWidth: 2 }} isAnimationActive={false} />
                            {be && <ReferenceDot x={be.year} y={be.rent} r={6} fill="#0b0b0b" stroke="#fff" strokeWidth={2}
                                label={{ value: `break-even, year ${be.year}`, position: "bottom", fontSize: 11, fill: "#0b0b0b" }} />}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}
        </figure>
    );
}

function Slider({ s, value, onChange, forecast }) {
    const id = `slider-${s.key}`;
    return (
        <div>
            <div className="flex items-baseline justify-between gap-2">
                <label htmlFor={id} className="text-sm font-medium text-gray-700">{s.label}</label>
                <output htmlFor={id} className="text-sm font-semibold text-gray-900 tabular-nums">
                    {s.unit === "%" ? `${Number(value).toFixed(s.step < 0.1 ? 2 : 1)}%` : `${value} ${s.unit}`}
                </output>
            </div>
            <input id={id} type="range" min={s.min} max={s.max} step={s.step} value={value}
                onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-indigo-600 cursor-pointer" />
            {forecast && (
                <p className="text-xs text-gray-500">
                    Forecast {pct(forecast.value)} (80% range {pct(forecast.p10)} to {pct(forecast.p90)})
                </p>
            )}
        </div>
    );
}

export default function DecisionPanel({ data, sliderValues, onSlider, onReset, updating }) {
    const res = data.result;
    const a = data.assumptions;
    const buy = res.recommendation === "buy";
    const H = a.H;
    return (
        <div className="space-y-6">
            <div className={`rounded-2xl p-5 sm:p-6 border ${buy ? "bg-blue-50 border-blue-200" : "bg-orange-50 border-orange-200"}`}>
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                    <div>
                        <p className="text-sm font-medium text-gray-600">Recommendation for {H} {H === 1 ? "year" : "years"}</p>
                        <p className="text-3xl sm:text-4xl font-bold text-gray-900 mt-1 flex items-center gap-3">
                            <span className="inline-block w-3 h-3 rounded-full" style={{ background: buy ? BUY : RENT }} aria-hidden="true" />
                            {buy ? "Buying looks cheaper" : "Renting looks cheaper"}
                        </p>
                        <p className="text-gray-700 mt-2">
                            {buy ? "Buying" : "Renting"} saves about <strong>{money(Math.abs(res.delta))}</strong> in today's money over {H} years.
                        </p>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-1 gap-3 sm:text-right">
                        <div>
                            <p className="text-xs text-gray-600">Break-even</p>
                            <p className="text-xl font-semibold text-gray-900">
                                {res.break_even_year ? `Year ${res.break_even_year}` : "Not within 30 years"}
                            </p>
                        </div>
                        <div>
                            <p className="text-xs text-gray-600">Mortgage payment</p>
                            <p className="text-xl font-semibold text-gray-900">{money(res.monthly_payment)}<span className="text-sm font-normal text-gray-500">/mo</span></p>
                        </div>
                    </div>
                </div>
                <p className="text-xs text-gray-600 mt-3">
                    Quick check: owning costs about {money(res.user_cost)} a year (user cost) against {money(res.annual_rent)} a year
                    of rent, which also points to <strong>{res.user_cost_says}</strong>.
                </p>
            </div>

            <div className={`grid grid-cols-1 lg:grid-cols-3 gap-6 transition-opacity ${updating ? "opacity-60" : ""}`}>
                <div className="lg:col-span-2 min-w-0">
                    <CostChart yearly={res.yearly} H={H} breakEven={res.break_even_year} />
                </div>
                <div className="space-y-4">
                    <div className="flex items-center justify-between">
                        <p className="text-sm font-semibold text-gray-900">Try different assumptions</p>
                        {updating ? <Spinner label="Updating…" className="text-xs text-gray-500" /> : (
                            <button onClick={onReset} className="inline-flex items-center gap-1 text-xs font-semibold text-indigo-600 hover:underline cursor-pointer">
                                <RotateCcw className="w-3 h-3" aria-hidden="true" /> Reset
                            </button>
                        )}
                    </div>
                    {SLIDERS.map((s) => (
                        <Slider key={s.key} s={s} value={toDisplay(s, sliderValues[s.key])}
                            onChange={(v) => onSlider(s.key, fromDisplay(s, v))}
                            forecast={s.forecast ? data.forecast[s.forecast] : null} />
                    ))}
                </div>
            </div>
        </div>
    );
}
