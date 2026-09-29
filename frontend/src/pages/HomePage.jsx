import React from "react";
import { useNavigate } from "react-router-dom";
import { Home, TrendingUp, Calculator, BarChart3 } from "lucide-react";

export default function HomePage() {
    const navigate = useNavigate();

    const features = [
        {
            icon: <Calculator className="w-8 h-8" />,
            title: "AI-Powered Predictions",
            description: "Our advanced machine learning model analyzes thousands of data points to provide accurate estimates."
        },
        {
            icon: <TrendingUp className="w-8 h-8" />,
            title: "Market Trends",
            description: "Stay updated with real-time market trends and historical data from across the nation."
        },
        {
            icon: <BarChart3 className="w-8 h-8" />,
            title: "Detailed Analysis",
            description: "Get comprehensive insights into property values based on location, size, and amenities."
        }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                {/* Background decoration */}
                <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
                    <div className="text-center">
                        {/* Logo/Icon */}
                        <div className="flex justify-center mb-8">
                            <div className="bg-gradient-to-br from-indigo-500 to-blue-600 p-4 rounded-2xl shadow-xl">
                                <Home className="w-16 h-16 text-white" />
                            </div>
                        </div>

                        {/* Main Heading */}
                        <h1 className="text-5xl md:text-6xl lg:text-7xl font-extrabold text-gray-900 mb-6 tracking-tight">
                            Discover Your Home's
                            <span className="block bg-gradient-to-r from-indigo-600 to-blue-600 bg-clip-text text-transparent mt-2">
                                True Value
                            </span>
                        </h1>

                        {/* Subheading */}
                        <p className="max-w-3xl mx-auto text-xl md:text-2xl text-gray-600 mb-12 leading-relaxed">
                            Get instant, AI-powered property valuations based on comprehensive market data
                            and economic trends across the United States.
                        </p>

                        {/* CTA Buttons */}
                        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
                            <button
                                onClick={() => navigate("/predict")}
                                className="group relative cursor-pointer px-8 py-4 bg-gradient-to-r from-indigo-600 to-blue-600 text-white font-semibold text-lg rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
                            >
                                <span className="relative z-10">Start Prediction</span>
                                <div className="absolute inset-0 bg-gradient-to-r from-indigo-700 to-blue-700 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                            </button>
                        </div>

                        {/* Stats */}
                        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
                            <div className="bg-white rounded-2xl p-6 shadow-md border border-gray-100">
                                <div className="text-4xl font-bold text-indigo-600 mb-2">50+</div>
                                <div className="text-gray-600 font-medium">US States Covered</div>
                            </div>
                            <div className="bg-white rounded-2xl p-6 shadow-md border border-gray-100">
                                <div className="text-4xl font-bold text-indigo-600 mb-2">1M+</div>
                                <div className="text-gray-600 font-medium">Properties Analyzed</div>
                            </div>
                            <div className="bg-white rounded-2xl p-6 shadow-md border border-gray-100">
                                <div className="text-4xl font-bold text-indigo-600 mb-2">95%</div>
                                <div className="text-gray-600 font-medium">Accuracy Rate</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div id="features" className="py-24 bg-white">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl font-bold text-gray-900 mb-4">
                            Why Choose Our Platform?
                        </h2>
                        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                            Advanced technology meets real estate expertise to deliver the most accurate predictions
                        </p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        {features.map((feature, index) => (
                            <div
                                key={index}
                                className="group bg-gradient-to-br from-gray-50 to-white p-8 rounded-2xl border border-gray-200 hover:border-indigo-300 hover:shadow-xl transition-all duration-300"
                            >
                                <div className="bg-gradient-to-br from-indigo-100 to-blue-100 w-16 h-16 rounded-xl flex items-center justify-center mb-6 text-indigo-600 group-hover:scale-110 transition-transform duration-300">
                                    {feature.icon}
                                </div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-3">
                                    {feature.title}
                                </h3>
                                <p className="text-gray-600 leading-relaxed">
                                    {feature.description}
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* CTA Section */}
            <div className="py-20 bg-gradient-to-br from-indigo-600 to-blue-600">
                <div className="max-w-4xl mx-auto text-center px-4">
                    <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
                        Ready to Get Started?
                    </h2>
                    <p className="text-xl text-indigo-100 mb-8">
                        Enter your property details and get an instant valuation in seconds
                    </p>
                    <button
                        onClick={() => navigate("/predict")}
                        className="px-10 py-4 bg-white cursor-pointer text-indigo-600 font-semibold text-lg rounded-xl shadow-lg hover:shadow-2xl transform hover:scale-105 transition-all duration-300"
                    >
                        Get Your Estimate Now
                    </button>
                </div>
            </div>
        </div>
    );
}