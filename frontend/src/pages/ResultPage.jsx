import React from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Home, ArrowLeft, TrendingUp, MapPin, Bed, Bath, Maximize, CheckCircle2 } from "lucide-react";

export default function ResultPage() {
    const navigate = useNavigate();
    const location = useLocation();
    const { prediction, formData } = location.state || {};

    // Redirect to predict page if no data
    if (!prediction || !formData) {
        React.useEffect(() => {
            navigate("/predict");
        }, [navigate]);
        return null;
    }

    const propertyDetails = [
        {
            icon: <MapPin className="w-5 h-5" />,
            label: "Location",
            value: `${formData.City?.label || formData.City}, ${formData.State?.label || formData.State}`,
        },
        {
            icon: <Bed className="w-5 h-5" />,
            label: "Bedrooms",
            value: formData.Bedroom,
        },
        {
            icon: <Bath className="w-5 h-5" />,
            label: "Bathrooms",
            value: formData.Bathroom,
        },
        {
            icon: <Maximize className="w-5 h-5" />,
            label: "Living Area",
            value: `${formData.Area?.toLocaleString()} sqft`,
        },
        {
            icon: <Maximize className="w-5 h-5" />,
            label: "Lot Area",
            value: `${formData.LotArea?.toLocaleString()} sqft`,
        },

        {
            icon: <MapPin className="w-5 h-5" />,
            label: "ZIP Code",
            value: `${formData.Zipcode} sqft`,
        },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
            {/* Header */}
            <div className="bg-white shadow-sm border-b border-gray-200">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
                    <button
                        onClick={() => navigate("/")}
                        className="flex items-center gap-2 cursor-pointer text-indigo-600 hover:text-indigo-700 font-medium transition-colors"
                    >
                        <ArrowLeft className="w-5 h-5" />
                        Back to Home
                    </button>
                </div>
            </div>

            {/* Main Content */}
            <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
                {/* Success Animation */}
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-20 h-20 bg-green-100 rounded-full mb-4 animate-bounce">
                        <CheckCircle2 className="w-12 h-12 text-green-600" />
                    </div>
                    <h1 className="text-3xl font-bold text-gray-900">
                        Prediction Complete!
                    </h1>
                </div>

                {/* Result Card */}
                <div className="bg-white rounded-3xl shadow-2xl border border-gray-200 overflow-hidden mb-8">
                    {/* Price Display */}
                    <div className="bg-gradient-to-br from-green-500 via-emerald-500 to-teal-500 px-8 py-12 text-center relative overflow-hidden">
                        {/* Decorative elements */}
                        <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-32 -mt-32"></div>
                        <div className="absolute bottom-0 left-0 w-48 h-48 bg-white/10 rounded-full -ml-24 -mb-24"></div>

                        <div className="relative z-10">
                            <div className="flex justify-center mb-4">
                                <div className="bg-white/20 backdrop-blur-sm p-3 rounded-xl">
                                    <TrendingUp className="w-10 h-10 text-white" />
                                </div>
                            </div>
                            <h2 className="text-2xl font-semibold text-white/90 mb-3">
                                Estimated Property Value
                            </h2>
                            <p className="text-6xl md:text-7xl font-extrabold text-white mb-2 tracking-tight">
                                ${prediction.prediction_usd?.toLocaleString()}
                            </p>
                            <p className="text-white/80 text-lg">
                                Based on current market analysis
                            </p>
                        </div>
                    </div>

                    {/* Property Details */}
                    <div className="p-8">
                        <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center gap-2">
                            <Home className="w-6 h-6 text-indigo-600" />
                            Property Details
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {propertyDetails.map((detail, index) => (
                                <div
                                    key={index}
                                    className="bg-gradient-to-br from-gray-50 to-white p-4 rounded-xl border border-gray-200 hover:border-indigo-300 transition-all"
                                >
                                    <div className="flex items-center gap-3">
                                        <div className="bg-indigo-100 p-2 rounded-lg text-indigo-600">
                                            {detail.icon}
                                        </div>
                                        <div>
                                            <p className="text-sm text-gray-500 font-medium">
                                                {detail.label}
                                            </p>
                                            <p className="text-lg font-bold text-gray-900">
                                                {detail.value}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                    <button
                        onClick={() => navigate("/predict")}
                        className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white cursor-pointer font-semibold text-lg rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
                    >
                        Calculate Another Property
                    </button>

                    <button
                        onClick={() => navigate("/")}
                        className="px-8 py-4 bg-white text-indigo-600 font-semibold text-lg cursor-pointer rounded-xl shadow-md hover:shadow-lg border-2 border-indigo-200 hover:border-indigo-300 transition-all duration-300"
                    >
                        Return to Home
                    </button>
                </div>

                {/* Additional Info */}
                <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 text-center">
                        <div className="text-3xl font-bold text-indigo-600 mb-2">AI-Powered</div>
                        <p className="text-gray-600 text-sm">Advanced machine learning algorithms</p>
                    </div>
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 text-center">
                        <div className="text-3xl font-bold text-indigo-600 mb-2">Real-Time</div>
                        <p className="text-gray-600 text-sm">Based on current market data</p>
                    </div>
                    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-200 text-center">
                        <div className="text-3xl font-bold text-indigo-600 mb-2">Accurate</div>
                        <p className="text-gray-600 text-sm">Trained on thousands of properties</p>
                    </div>
                </div>
            </div>
        </div>
    );
}