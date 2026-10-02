// Calls to the Flask API. In development Vite proxies /api to http://127.0.0.1:5000;
// set VITE_API_URL to call a back end on another host instead.
const BASE = import.meta.env.VITE_API_URL ?? "";

export const DISCLAIMER = "Indicative decision support only – not financial or valuation advice";

export class ApiError extends Error {
    constructor(message, status, details = []) {
        super(message);
        this.status = status;
        this.details = details; // [{ field, message }] for invalid input (HTTP 422)
    }
}

async function request(path, options = {}) {
    let res;
    try {
        res = await fetch(`${BASE}${path}`, {
            ...options,
            headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        });
    } catch (err) {
        if (err.name === "AbortError") throw err;
        throw new ApiError(
            "Can't reach the analysis server. Is the back end running (python backend/app.py)?", 0);
    }
    let data = null;
    try {
        data = await res.json();
    } catch {
        // non-JSON reply (e.g. proxy error page)
    }
    if (!res.ok) {
        throw new ApiError(data?.error || `Request failed (HTTP ${res.status})`, res.status, data?.details || []);
    }
    return data;
}

const post = (path, body, signal) => request(path, { method: "POST", body: JSON.stringify(body), signal });

export const analyse = (body, signal) => post("/api/analyse", body, signal);
export const scenario = (body, signal) => post("/api/scenario", body, signal);
export const sensitivity = (body, signal) => post("/api/sensitivity", body, signal);
export const forecast = (state, signal) => request(`/api/forecast/${encodeURIComponent(state)}`, { signal });
