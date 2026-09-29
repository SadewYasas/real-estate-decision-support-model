import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Select from "react-select";
import { ArrowLeft, Home, MapPin, Hash, Bed, Bath, Maximize, Map } from "lucide-react";
import states from "../data/states.json";
import cities from "../data/cities.json";

const BACKEND_URL = "http://localhost:5000";

export default function PredictPage() {
    const navigate = useNavigate();

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
    const [error, setError] = useState("");
    const [loadingMessage, setLoadingMessage] = useState("");

    const runPredictionSteps = async () => {
        const steps = [
            { message: "Analyzing property details...", delay: 1200 },
            { message: "Evaluating neighborhood market data...", delay: 1200 },
            { message: "Running price prediction...", delay: 1500 },
        ];

        for (const step of steps) {
            setLoadingMessage(step.message);
            await new Promise((res) => setTimeout(res, step.delay));
        }
    };

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
        setLoadingMessage("Initializing prediction...");

        try {
            await runPredictionSteps();

            const res = await fetch(`${BACKEND_URL}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...form,
                    State: form.State.value || form.State,
                    City: form.City?.label || form.City,
                }),
            });

            const data = await res.json();
            if (!res.ok) throw new Error(data?.error || "Request failed");

            // Navigate to result page with the prediction data
            navigate("/result", { state: { prediction: data, formData: form } });
        } catch (err) {
            setError(err.message || String(err));
            setLoading(false);
            setLoadingMessage("");
        }
    };

    const customSelectStyles = {
        control: (base) => ({
            ...base,
            padding: "0.5rem",
            borderRadius: "0.75rem",
            borderColor: "#d1d5db",
            boxShadow: "0 1px 2px 0 rgb(0 0 0 / 0.05)",
            "&:hover": {
                borderColor: "#818cf8",
            },
            "&:focus-within": {
                borderColor: "#818cf8",
                boxShadow: "0 0 0 3px rgba(129, 140, 248, 0.1)",
            },
        }),
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
            {/* Header */}
            <div className="bg-white shadow-sm border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <button
                        onClick={() => navigate("/")}
                        className="flex items-center cursor-pointer gap-2 text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        Back to Home
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                <div className="bg-white rounded-3xl shadow-xl border border-gray-200 overflow-hidden">
                    {/* Form Header */}
                    <div className="bg-gradient-to-r from-indigo-600 to-blue-600 px-8 py-10 text-center">
                        <div className="flex justify-center mb-4">
                            <div className="bg-white/20 backdrop-blur-sm p-3 rounded-xl">
                                <Home className="w-10 h-10 text-white" />
                            </div>
                        </div>
                        <h1 className="text-4xl font-bold text-white mb-3">
                            Property Valuation
                        </h1>
                        <p className="text-indigo-100 text-lg max-w-2xl mx-auto">
                            Enter your property details below to receive an AI-powered price estimate
                        </p>
                    </div>

                    {/* Form */}
                    <form onSubmit={handleSubmit} className="p-8 md:p-10">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* State */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <MapPin className="w-4 h-4 text-indigo-600" />
                                    State
                                </label>
                                <Select
                                    options={stateOptions}
                                    onChange={(opt) => update("State", opt)}
                                    placeholder="Select a state"
                                    styles={customSelectStyles}
                                    value={form.State}
                                />
                            </div>

                            {/* City - Optimized Select with virtualization */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Map className="w-4 h-4 text-indigo-600" />
                                    City
                                </label>
                                <Select
                                    options={cityOptions}
                                    onChange={(opt) => update("City", opt)}
                                    placeholder="Select or type a city"
                                    styles={customSelectStyles}
                                    value={form.City}
                                    filterOption={(option, inputValue) => {
                                        return option.label.toLowerCase().includes(inputValue.toLowerCase());
                                    }}
                                    menuPortalTarget={document.body}
                                    menuPosition="fixed"
                                    maxMenuHeight={200}
                                    pageSize={20}
                                />
                            </div>

                            {/* Zipcode */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Hash className="w-4 h-4 text-indigo-600" />
                                    ZIP Code
                                </label>
                                <input
                                    type="text"
                                    className="w-full border border-gray-300 rounded-xl px-4 py-3.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all shadow-sm"
                                    value={form.Zipcode}
                                    onChange={(e) => update("Zipcode", e.target.value)}
                                    placeholder="e.g. 94107"
                                    required
                                />
                            </div>

                            {/* Bedrooms */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Bed className="w-4 h-4 text-indigo-600" />
                                    Bedrooms
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    className="w-full border border-gray-300 rounded-xl px-4 py-3.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all shadow-sm"
                                    value={form.Bedroom}
                                    onChange={(e) => update("Bedroom", Number(e.target.value))}
                                    placeholder="e.g. 3"
                                    required
                                />
                            </div>

                            {/* Bathrooms */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Bath className="w-4 h-4 text-indigo-600" />
                                    Bathrooms
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    step="0.5"
                                    className="w-full border border-gray-300 rounded-xl px-4 py-3.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all shadow-sm"
                                    value={form.Bathroom}
                                    onChange={(e) => update("Bathroom", Number(e.target.value))}
                                    placeholder="e.g. 2"
                                    required
                                />
                            </div>

                            {/* Living Area */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Maximize className="w-4 h-4 text-indigo-600" />
                                    Living Area (sqft)
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    className="w-full border border-gray-300 rounded-xl px-4 py-3.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all shadow-sm"
                                    value={form.Area}
                                    onChange={(e) => update("Area", Number(e.target.value))}
                                    placeholder="e.g. 1800"
                                    required
                                />
                            </div>

                            {/* Lot Area */}
                            <div className="md:col-span-2 space-y-2">
                                <label className="flex items-center gap-2 text-gray-700 font-semibold text-sm uppercase tracking-wide">
                                    <Maximize className="w-4 h-4 text-indigo-600" />
                                    Lot Area (sqft)
                                </label>
                                <input
                                    type="number"
                                    min="0"
                                    className="w-full border border-gray-300 rounded-xl px-4 py-3.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none transition-all shadow-sm"
                                    value={form.LotArea}
                                    onChange={(e) => update("LotArea", Number(e.target.value))}
                                    placeholder="e.g. 5000"
                                    required
                                />
                            </div>
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="mt-6 bg-red-50 border border-red-200 rounded-xl p-4">
                                <p className="text-red-700 font-medium text-center">{error}</p>
                            </div>
                        )}

                        {/* Submit Button */}
                        <div className="mt-8 flex justify-center">
                            <button
                                type="submit"
                                disabled={loading}
                                className="group cursor-pointer relative px-12 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-bold text-lg rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-300 disabled:opacity-60 disabled:cursor-not-allowed disabled:transform-none"
                            >
                                {loading ? (
                                    <span className="flex items-center gap-3">
                                        <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                            <circle
                                                className="opacity-25"
                                                cx="12"
                                                cy="12"
                                                r="10"
                                                stroke="currentColor"
                                                strokeWidth="4"
                                                fill="none"
                                            />
                                            <path
                                                className="opacity-75"
                                                fill="currentColor"
                                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                                            />
                                        </svg>
                                        {loadingMessage || "Processing..."}
                                    </span>
                                ) : (
                                    "Calculate Property Value"
                                )}
                            </button>
                        </div>
                    </form>
                </div>

                {/* Info Cards */}
                <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
                        <p className="text-sm text-gray-600">
                            <span className="font-semibold text-indigo-600">Tip:</span> Accurate details lead to better predictions
                        </p>
                    </div>
                    <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
                        <p className="text-sm text-gray-600">
                            <span className="font-semibold text-indigo-600">Fast:</span> Get results in under a few seconds
                        </p>
                    </div>
                    <div className="bg-white rounded-xl p-4 shadow-sm border border-gray-200">
                        <p className="text-sm text-gray-600">
                            <span className="font-semibold text-indigo-600">Secure:</span> Your data is never stored
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}