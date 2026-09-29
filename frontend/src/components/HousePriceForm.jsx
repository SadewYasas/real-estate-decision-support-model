import React, { useState } from "react";
import Select from "react-select";
import states from "../data/states.json";
import cities from "../data/cities.json";

const BACKEND_URL = "http://localhost:5000";

export default function HousePriceForm() {
    const [form, setForm] = useState({
        State: "",
        City: "",
        Zipcode: "",
        Bedroom: "",
        Bathroom: "",
        Area: "",
        LotArea: "",
    });

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");

    const stateOptions = Object.entries(states).map(([abbr, name]) => ({
        value: abbr,
        label: name,
    }));
    const cityOptions = cities.map((city) => ({ value: city, label: city }));

    const update = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError("");
        setResult(null);

        try {
            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...form,
                    State: form.State.value || form.State,
                    City: form.City.value || form.City,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || "Request failed");
            setResult(data);
        } catch (err) {
            setError(err.message || String(err));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 flex items-center justify-center px-4 py-12">
            <div className="w-full max-w-4xl bg-white rounded-3xl shadow-lg border border-gray-200 p-10">
                {/* Header */}
                <div className="text-center mb-10">
                    <h1 className="text-4xl font-bold text-gray-900 mb-3 tracking-tight">
                        House Price Prediction
                    </h1>
                    <p className="text-gray-500 text-lg max-w-2xl mx-auto leading-relaxed">
                        Estimate your property’s value using our AI-powered model trained on national housing and economic trends.
                        Enter your property details below to get an accurate prediction.
                    </p>
                </div>

                {/* Form */}
                <form
                    onSubmit={handleSubmit}
                    className="grid grid-cols-1 md:grid-cols-2 gap-6"
                >
                    {/* State */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            State
                        </label>
                        <Select
                            options={stateOptions}
                            onChange={(opt) => update("State", opt)}
                            placeholder="Select a state"
                            className="rounded-md shadow-sm"
                        />
                    </div>

                    {/* City */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            City
                        </label>
                        <Select
                            options={cityOptions}
                            onChange={(opt) => update("City", opt)}
                            placeholder="Select a city"
                            className="rounded-md shadow-sm"
                        />
                    </div>

                    {/* Zipcode */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            ZIP Code
                        </label>
                        <input
                            type="text"
                            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none transition-all"
                            value={form.Zipcode}
                            onChange={(e) => update("Zipcode", e.target.value)}
                            placeholder="e.g. 94107"
                        />
                    </div>

                    {/* Bedroom */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            Bedrooms
                        </label>
                        <input
                            type="number"
                            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none transition-all"
                            value={form.Bedroom}
                            onChange={(e) => update("Bedroom", Number(e.target.value))}
                            placeholder="e.g. 3"
                        />
                    </div>

                    {/* Bathroom */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            Bathrooms
                        </label>
                        <input
                            type="number"
                            step="0.5"
                            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none transition-all"
                            value={form.Bathroom}
                            onChange={(e) => update("Bathroom", Number(e.target.value))}
                            placeholder="e.g. 2"
                        />
                    </div>

                    {/* Area */}
                    <div>
                        <label className="block text-gray-700 font-medium mb-2">
                            Living Area (sqft)
                        </label>
                        <input
                            type="number"
                            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none transition-all"
                            value={form.Area}
                            onChange={(e) => update("Area", Number(e.target.value))}
                            placeholder="e.g. 1800"
                        />
                    </div>

                    {/* Lot Area */}
                    <div className="md:col-span-2">
                        <label className="block text-gray-700 font-medium mb-2">
                            Lot Area
                        </label>
                        <input
                            type="number"
                            className="w-full border border-gray-300 rounded-lg p-3 focus:ring-2 focus:ring-indigo-400 outline-none transition-all"
                            value={form.LotArea}
                            onChange={(e) => update("LotArea", Number(e.target.value))}
                            placeholder="e.g. 5000"
                        />
                    </div>

                    {/* Button */}
                    <div className="col-span-1 md:col-span-2 flex justify-center mt-6">
                        <button
                            disabled={loading}
                            className="w-full md:w-auto bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 transition-all duration-300 text-white font-semibold text-lg px-10 py-3 rounded-xl shadow-md hover:shadow-lg focus:ring-4 focus:ring-indigo-300 disabled:opacity-60"
                        >
                            {loading ? "Processing..." : "Predict Price"}
                        </button>
                    </div>
                </form>

                {/* Error */}
                {error && (
                    <div className="mt-6 text-center">
                        <p className="text-red-600 font-medium">{error}</p>
                    </div>
                )}

                {/* Result */}
                {result && (
                    <div className="mt-10 bg-gradient-to-br from-green-50 via-white to-green-100 p-8 rounded-2xl border border-green-200 shadow-sm">
                        <h2 className="text-2xl font-semibold text-gray-800 text-center mb-3">
                            Estimated Price
                        </h2>
                        <p className="text-center text-4xl font-bold text-green-700 mb-4">
                            ${result.prediction_usd.toLocaleString()}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
