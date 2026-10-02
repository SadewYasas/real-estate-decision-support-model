import React from "react";
import { useNavigate } from "react-router-dom";
import { Home, TrendingUp, Calculator, BarChart3 } from "lucide-react";
import { DISCLAIMER } from "../lib/api";

// Figures from artefacts/metrics (sale_models.json, rent_models.json, forecast.json,
// decision_flip.json). Update them if the models are retrained.
const STATS = [
    { value: "0.86", label: "R² of the price model on 7,622 unseen homes (log scale)" },
    { value: "51", label: "states with a house price growth forecast" },
    { value: "1 in 10", label: "buy-or-rent decisions change when growth comes from our forecast instead of a fixed 3%" },
];

const FEATURES = [
    {
        icon: <Calculator className="w-8 h-8" />,
        title: "Price and rent estimates",
        description: "Machine-learning models trained on 38,000 sale listings and 99,000 rental listings, with a typical error of 22% (price) and 14% (rent).",
    },
    {
        icon: <TrendingUp className="w-8 h-8" />,
        title: "Market forecasts",
        description: "Next-year house price growth for every state and national rent growth, from economic indicators, with an 80% range.",
    },
    {
        icon: <BarChart3 className="w-8 h-8" />,
        title: "Rent-or-buy comparison",
        description: "A transparent present-value calculation with break-even year, sensitivity analysis and economic scenarios.",
    },
];

export default function HomePage() {
    const navigate = useNavigate();
    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 sm:pt-20 pb-16 text-center">
                <div className="flex justify-center mb-8">
                    <div className="bg-gradient-to-br from-indigo-500 to-blue-600 p-4 rounded-2xl shadow-xl">
                        <Home className="w-14 h-14 text-white" aria-hidden="true" />
                    </div>
                </div>
                <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold text-gray-900 mb-6 tracking-tight">
                    Should you rent
                    <span className="block bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent mt-2">
                        or buy?
                    </span>
                </h1>
                <p className="max-w-3xl mx-auto text-lg sm:text-xl text-gray-600 mb-10 leading-relaxed">
                    Enter a US home's ZIP code and size. We estimate its price and rent, forecast house prices and rents
                    for the next year, and show whether renting or buying is likely to cost less.
                </p>
                <button
                    onClick={() => navigate("/analyse")}
                    className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-semibold text-lg rounded-xl shadow-lg hover:shadow-2xl hover:scale-105 transition-all duration-300 cursor-pointer"
                >
                    Start an analysis
                </button>
                <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                    {STATS.map((s) => (
                        <div key={s.value} className="bg-white rounded-2xl p-6 shadow-md border border-gray-100">
                            <div className="text-4xl font-bold text-indigo-600 mb-2">{s.value}</div>
                            <div className="text-gray-600 text-sm">{s.label}</div>
                        </div>
                    ))}
                </div>
            </div>

            <div className="py-20 bg-white">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
                    <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-12 text-center">What you get</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {FEATURES.map((f) => (
                            <div key={f.title} className="bg-gradient-to-br from-gray-50 to-white p-8 rounded-2xl border border-gray-200">
                                <div className="bg-gradient-to-br from-indigo-100 to-blue-100 w-16 h-16 rounded-xl flex items-center justify-center mb-6 text-indigo-600">
                                    {f.icon}
                                </div>
                                <h3 className="text-xl font-bold text-gray-900 mb-3">{f.title}</h3>
                                <p className="text-gray-600 leading-relaxed">{f.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <footer className="py-8 text-center text-sm text-gray-500 px-4">
                {DISCLAIMER}. Final-year research project (Coventry University / NIBM).
            </footer>
        </div>
    );
}
