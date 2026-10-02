import React from "react";
import { money, pct, signedPct } from "../lib/format";
import { BUY, RENT } from "../lib/chart";

export default function ScenarioTab({ data }) {
    const base = data.cases[0];
    return (
        <div>
            <p className="text-sm text-gray-700">
                The same property under different economic conditions over {data.H} years. Pessimistic and optimistic use
                the low and high ends of the forecast range. The four shocks change the economic indicators and re-run
                the forecasts; rate shocks also change your mortgage rate. Rows that change the decision are highlighted.
            </p>
            <p className="mt-3 text-xs text-gray-500 sm:hidden">Scroll sideways to see all columns.</p>
            <div className="mt-2 sm:mt-4 overflow-x-auto -mx-1">
                <table className="w-full min-w-[640px] text-sm tabular-nums">
                    <thead>
                        <tr className="text-left text-gray-600 border-b border-gray-200">
                            <th className="px-2 py-2 font-semibold">Scenario</th>
                            <th className="px-2 py-2 font-semibold text-right">House price growth</th>
                            <th className="px-2 py-2 font-semibold text-right">Rent growth</th>
                            <th className="px-2 py-2 font-semibold text-right">Mortgage rate</th>
                            <th className="px-2 py-2 font-semibold text-right">Buying advantage</th>
                            <th className="px-2 py-2 font-semibold">Decision</th>
                            <th className="px-2 py-2 font-semibold text-right">Break-even</th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.cases.map((c) => {
                            const isBase = c.scenario === "base";
                            return (
                                <tr key={c.scenario}
                                    className={`border-b border-gray-100 ${c.flips_vs_base ? "bg-amber-50" : ""} ${isBase ? "font-semibold" : ""}`}>
                                    <td className="px-2 py-2.5 text-gray-900">
                                        <span className="block">{c.label}</span>
                                        {c.flips_vs_base && (
                                            <span className="text-[10px] font-semibold uppercase tracking-wide bg-amber-200 text-amber-900 rounded px-1.5 py-0.5">changes decision</span>
                                        )}
                                    </td>
                                    <td className="px-2 py-2.5 text-right">{pct(c.g)}
                                        {!isBase && c.g !== base.g && <span className="block text-xs text-gray-500">{signedPct(c.g - base.g)}</span>}</td>
                                    <td className="px-2 py-2.5 text-right">{pct(c.q)}
                                        {!isBase && c.q !== base.q && <span className="block text-xs text-gray-500">{signedPct(c.q - base.q)}</span>}</td>
                                    <td className="px-2 py-2.5 text-right">{pct(c.r, 2)}</td>
                                    <td className="px-2 py-2.5 text-right">{money(c.delta)}
                                        {!isBase && <span className="block text-xs text-gray-500">{c.delta_change_vs_base >= 0 ? "+" : "−"}{money(Math.abs(c.delta_change_vs_base))}</span>}</td>
                                    <td className="px-2 py-2.5">
                                        <span className="inline-flex items-center gap-1.5">
                                            <span className="w-2.5 h-2.5 rounded-full" style={{ background: c.recommendation === "buy" ? BUY : RENT }} aria-hidden="true" />
                                            {c.recommendation === "buy" ? "Buy" : "Rent"}
                                        </span>
                                    </td>
                                    <td className="px-2 py-2.5 text-right">{c.break_even_year ? `Year ${c.break_even_year}` : "Never"}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
            <p className="mt-3 text-xs text-gray-500">
                Shocks show what the forecasting models learned from 2014–2025 data, not guaranteed cause and effect.
                The house price model is constrained so that higher mortgage rates or unemployment never raise predicted
                growth; other indicators are unconstrained. Rent growth = ARIMA forecast plus the change predicted by the
                rent model with economic indicators.
            </p>
        </div>
    );
}
