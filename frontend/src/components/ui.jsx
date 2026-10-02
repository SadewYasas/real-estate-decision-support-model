import React from "react";
import { AlertTriangle, Info, XCircle } from "lucide-react";
import { DISCLAIMER } from "../lib/api";

export function Card({ title, subtitle, children, className = "", action }) {
    return (
        <section className={`bg-white rounded-2xl border border-gray-200 shadow-sm p-5 sm:p-6 ${className}`}>
            {(title || action) && (
                <div className="flex items-start justify-between gap-3 mb-4">
                    <div>
                        {title && <h2 className="text-lg font-semibold text-gray-900">{title}</h2>}
                        {subtitle && <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>}
                    </div>
                    {action}
                </div>
            )}
            {children}
        </section>
    );
}

export function Disclaimer({ text = DISCLAIMER, className = "" }) {
    return (
        <p role="note" className={`flex items-start gap-2 text-sm text-amber-900 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 ${className}`}>
            <Info className="w-4 h-4 mt-0.5 shrink-0" aria-hidden="true" />
            <span><strong className="font-semibold">{text}.</strong> Estimates come from statistical models and can be wrong.</span>
        </p>
    );
}

export function Warnings({ items }) {
    if (!items?.length) return null;
    return (
        <div role="status" className="bg-orange-50 border border-orange-200 rounded-xl px-4 py-3">
            <p className="flex items-center gap-2 text-sm font-semibold text-orange-900 mb-1">
                <AlertTriangle className="w-4 h-4" aria-hidden="true" /> Please note
            </p>
            <ul className="list-disc pl-6 space-y-0.5 text-sm text-orange-900">
                {items.map((w) => <li key={w}>{w}</li>)}
            </ul>
        </div>
    );
}

export function ErrorBox({ error, onRetry }) {
    if (!error) return null;
    return (
        <div role="alert" className="bg-red-50 border border-red-200 rounded-xl px-4 py-3 text-sm text-red-800">
            <p className="flex items-center gap-2 font-semibold">
                <XCircle className="w-4 h-4" aria-hidden="true" /> {error.message}
            </p>
            {error.details?.length > 0 && (
                <ul className="list-disc pl-6 mt-1">
                    {error.details.map((d) => <li key={d.field + d.message}>{d.field}: {d.message}</li>)}
                </ul>
            )}
            {onRetry && (
                <button onClick={onRetry} className="mt-2 font-semibold underline cursor-pointer">Try again</button>
            )}
        </div>
    );
}

export function Spinner({ label = "Loading…", className = "" }) {
    return (
        <span className={`inline-flex items-center gap-2 ${className}`} role="status">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            {label}
        </span>
    );
}

export function Tabs({ tabs, active, onChange }) {
    return (
        <div role="tablist" className="flex gap-1 bg-gray-100 rounded-xl p-1 w-full sm:w-fit">
            {tabs.map((t) => (
                <button
                    key={t.id}
                    role="tab"
                    aria-selected={active === t.id}
                    onClick={() => onChange(t.id)}
                    className={`flex-1 sm:flex-none px-4 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${active === t.id ? "bg-white text-gray-900 shadow-sm" : "text-gray-600 hover:text-gray-900"
                        }`}
                >
                    {t.label}
                </button>
            ))}
        </div>
    );
}
