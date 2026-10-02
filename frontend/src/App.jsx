import React from "react";
import { BrowserRouter as Router, Navigate, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import AnalysePage from "./pages/AnalysePage";

export default function App() {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/analyse" element={<AnalysePage />} />
                {/* Old single-price pages were replaced by /analyse; keep their URLs working. */}
                <Route path="/predict" element={<Navigate to="/analyse" replace />} />
                <Route path="/result" element={<Navigate to="/analyse" replace />} />
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </Router>
    );
}
