import React, { useState } from "react";
import { ChevronDown, Search } from "lucide-react";
import states from "../data/states.json";
import { ADVANCED, toRequest } from "../lib/form";
import { Spinner } from "./ui";

const inputCls =
    "w-full border border-gray-300 rounded-xl px-3.5 py-2.5 focus:ring-2 focus:ring-indigo-400 focus:border-indigo-400 outline-none shadow-sm";

function Field({ id, label, hint, error, children }) {
    return (
        <div className="space-y-1">
            <label htmlFor={id} className="block text-sm font-semibold text-gray-700">{label}</label>
            {children}
            {hint && !error && <p className="text-xs text-gray-500">{hint}</p>}
            {error && <p className="text-xs text-red-700" role="alert">{error}</p>}
        </div>
    );
}

export default function PropertyForm({ form, setForm, onSubmit, loading, serverErrors = {}, needState }) {
    const [open, setOpen] = useState(false);
    const [errors, setErrors] = useState({});
    const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));
    const setAdv = (k, v) => setForm((f) => ({ ...f, advanced: { ...f.advanced, [k]: v } }));
    const err = (k) => errors[k] || serverErrors[k] || serverErrors[`assumptions.${k}`];

    const submit = (e) => {
        e.preventDefault();
        const { body, errors: errs } = toRequest(form);
        setErrors(errs);
        if (Object.keys(errs).length) {
            if (ADVANCED.some((a) => errs[a.key])) setOpen(true);
            return;
        }
        onSubmit(body);
    };

    return (
        <form onSubmit={submit} noValidate className="space-y-5">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div className="col-span-2 lg:col-span-1">
                    <Field id="zip" label="ZIP code" error={err("zip")} hint="US 5-digit ZIP">
                        <input id="zip" inputMode="numeric" autoComplete="postal-code" maxLength={5} className={inputCls}
                            value={form.zip} onChange={(e) => set("zip", e.target.value.replace(/\D/g, ""))} placeholder="e.g. 78704" />
                    </Field>
                </div>
                <Field id="beds" label="Bedrooms" error={err("beds")}>
                    <input id="beds" type="number" min="1" max="10" step="1" className={inputCls}
                        value={form.beds} onChange={(e) => set("beds", e.target.value)} />
                </Field>
                <Field id="baths" label="Bathrooms" error={err("baths")}>
                    <input id="baths" type="number" min="1" max="10" step="0.5" className={inputCls}
                        value={form.baths} onChange={(e) => set("baths", e.target.value)} />
                </Field>
                <div className="col-span-2 lg:col-span-1">
                    <Field id="sqft" label="Living area (sq ft)" error={err("sqft")}>
                        <input id="sqft" type="number" min="300" max="15000" step="10" className={inputCls}
                            value={form.sqft} onChange={(e) => set("sqft", e.target.value)} />
                    </Field>
                </div>
            </div>

            {(needState || form.state) && (
                <div className="max-w-xs">
                    <Field id="state" label="State" error={err("state")}
                        hint="Needed because this ZIP isn't in our lookup; state averages are used.">
                        <select id="state" className={inputCls} value={form.state} onChange={(e) => set("state", e.target.value)}>
                            <option value="">Select a state</option>
                            {Object.entries(states).map(([code, name]) => <option key={code} value={code}>{name}</option>)}
                        </select>
                    </Field>
                </div>
            )}

            <div className="border border-gray-200 rounded-xl">
                <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
                    className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-gray-700 cursor-pointer">
                    Advanced assumptions (optional)
                    <ChevronDown className={`w-4 h-4 transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
                </button>
                {open && (
                    <div className="px-4 pb-4 space-y-4">
                        <p className="text-xs text-gray-500">
                            Leave blank to use the defaults (shown as placeholders). Growth rates come from the forecast;
                            you can change them with the sliders after the first result.
                        </p>
                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                            {ADVANCED.map((a) => (
                                <Field key={a.key} id={`adv-${a.key}`} label={`${a.label} (${a.unit})`} error={err(a.key)}>
                                    <input id={`adv-${a.key}`} type="number" min={a.min} max={a.max} step={a.step} className={inputCls}
                                        placeholder={a.placeholder} value={form.advanced[a.key] ?? ""}
                                        onChange={(e) => setAdv(a.key, e.target.value)} />
                                </Field>
                            ))}
                        </div>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-xl">
                            <Field id="price" label="Asking price ($)" hint="Optional: use instead of our estimate">
                                <input id="price" type="number" min="1" className={inputCls} placeholder="our estimate"
                                    value={form.price} onChange={(e) => set("price", e.target.value)} />
                            </Field>
                            <Field id="rent" label="Monthly rent ($)" hint="Optional: use instead of our estimate">
                                <input id="rent" type="number" min="1" className={inputCls} placeholder="our estimate"
                                    value={form.monthly_rent} onChange={(e) => set("monthly_rent", e.target.value)} />
                            </Field>
                        </div>
                    </div>
                )}
            </div>

            <button type="submit" disabled={loading}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-8 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-xl shadow-sm transition-colors disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer">
                {loading ? <Spinner label="Analysing…" /> : <><Search className="w-4 h-4" aria-hidden="true" /> Analyse</>}
            </button>
        </form>
    );
}
