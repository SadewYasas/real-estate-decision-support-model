import React from "react";
import { Home, KeyRound, TrendingUp, LineChart } from "lucide-react";
import { money, pct, methodName } from "../lib/format";

function StatCard({ icon, label, value, unit, range, rangeLabel, note }) {
    return (
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5 flex flex-col">
            <p className="flex items-center gap-2 text-sm font-medium text-gray-600">
                <span className="text-indigo-600" aria-hidden="true">{icon}</span>{label}
            </p>
            <p className="mt-2 text-3xl font-semibold text-gray-900 tracking-tight">
                {value}{unit && <span className="text-base font-medium text-gray-500"> {unit}</span>}
            </p>
            {range && (
                <p className="mt-1 text-sm text-gray-700">
                    <span className="text-gray-500">{rangeLabel}: </span>{range}
                </p>
            )}
            {note && <p className="mt-2 text-xs text-gray-500 leading-relaxed">{note}</p>}
        </div>
    );
}

export default function ResultCards({ data, forecastDetail }) {
    const { price, rent, forecast } = data;
    const pm = price.test_mape, rm = rent.test_mape;
    const hp = forecastDetail?.house_price_growth;
    const rg = forecastDetail?.rent_growth_us;
    const realised = (r) => r ? `So far: ${pct(r.growth_pct / 100)} in ${r.periods} ${r.periods === 1 ? "period" : "periods"} to ${r.to}.` : "";

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            <StatCard
                icon={<Home className="w-4 h-4" />}
                label="Estimated price"
                value={money(price.predicted)}
                range={`${money(price.predicted * (1 - pm))} – ${money(price.predicted * (1 + pm))}`}
                rangeLabel={`Typical error ±${Math.round(pm * 100)}%`}
                note={price.source === "user"
                    ? `The decision uses your price, ${money(price.used)}.`
                    : "From the sale-price model (CatBoost); range = average test error."}
            />
            <StatCard
                icon={<KeyRound className="w-4 h-4" />}
                label="Estimated rent"
                value={money(rent.predicted)}
                unit="/ month"
                range={`${money(rent.predicted * (1 - rm))} – ${money(rent.predicted * (1 + rm))}`}
                rangeLabel={`Typical error ±${Math.round(rm * 100)}%`}
                note={rent.source === "user"
                    ? `The decision uses your rent, ${money(rent.used)}.`
                    : `2019 rent model × ${rent.cpi_factor.toFixed(3)} CPI rent change to ${rent.cpi_reference_month}.`}
            />
            <StatCard
                icon={<TrendingUp className="w-4 h-4" />}
                label={`House price growth, ${forecast.g.geography}`}
                value={pct(forecast.g.value)}
                unit="next 12 months"
                range={`${pct(forecast.g.p10)} to ${pct(forecast.g.p90)}`}
                rangeLabel="80% range"
                note={`Forecast from ${forecast.g.origin} (${methodName(forecast.g.method)}). ${realised(hp?.realised_so_far)}`}
            />
            <StatCard
                icon={<LineChart className="w-4 h-4" />}
                label="Rent growth, US"
                value={pct(forecast.q.value)}
                unit="next 12 months"
                range={`${pct(forecast.q.p10)} to ${pct(forecast.q.p90)}`}
                rangeLabel="80% range"
                note={`Forecast from ${forecast.q.origin} (${methodName(forecast.q.method)}). ${realised(rg?.realised_so_far)}`}
            />
        </div>
    );
}
