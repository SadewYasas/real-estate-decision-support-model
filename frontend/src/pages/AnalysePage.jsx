import React, { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Home, MapPin } from "lucide-react";
import * as api from "../lib/api";
import { money, number } from "../lib/format";
import PropertyForm from "../components/PropertyForm";
import { EMPTY_FORM, toRequest } from "../lib/form";
import ResultCards from "../components/ResultCards";
import ExplainPanel from "../components/ExplainPanel";
import DecisionPanel from "../components/DecisionPanel";
import { SLIDERS } from "../lib/chart";
import SensitivityTab from "../components/SensitivityTab";
import ScenarioTab from "../components/ScenarioTab";
import { Card, Disclaimer, ErrorBox, Spinner, Tabs, Warnings } from "../components/ui";

const EXAMPLES = [
    { label: "Austin, TX", zip: "78704", beds: "3", baths: "2", sqft: "1800" },
    { label: "Columbus, OH", zip: "43215", beds: "3", baths: "2", sqft: "1600" },
    { label: "Phoenix, AZ", zip: "85016", beds: "4", baths: "2.5", sqft: "2200" },
];
const TABS = [{ id: "sensitivity", label: "Sensitivity" }, { id: "scenario", label: "Economic scenarios" }];
const DEBOUNCE_MS = 300;
const URL_FIELDS = ["zip", "state", "beds", "baths", "sqft"];

/** Property + advanced assumptions from the form, with slider overrides on top. */
function buildBody(baseBody, overrides) {
    if (!baseBody) return null;
    const assumptions = { ...(baseBody.assumptions || {}), ...overrides };
    const body = { ...baseBody };
    if (Object.keys(assumptions).length) body.assumptions = assumptions;
    else delete body.assumptions;
    return body;
}

function LocationLine({ loc }) {
    const fallback = Object.values(loc.source).includes("state_median");
    return (
        <p className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-gray-600">
            <span className="inline-flex items-center gap-1 font-semibold text-gray-900">
                <MapPin className="w-4 h-4 text-indigo-600" aria-hidden="true" /> ZIP {loc.zip}, {loc.state}
            </span>
            <span>{number(loc.density)} people / sq mi</span>
            <span>household income {money(loc.mean_income)} (mean), {money(loc.median_income)} (median)</span>
            {fallback && <span className="text-orange-800">(state averages used where ZIP data is missing)</span>}
        </p>
    );
}

export default function AnalysePage() {
    // A property in the URL (/analyse?zip=78704&beds=3&baths=2&sqft=1800) is analysed on load,
    // so a result can be bookmarked or shared.
    const [params, setParams] = useSearchParams();
    const [form, setForm] = useState(() => ({
        ...EMPTY_FORM,
        ...Object.fromEntries(URL_FIELDS.filter((k) => params.get(k)).map((k) => [k, params.get(k)])),
    }));
    const [baseBody, setBaseBody] = useState(null);
    const [overrides, setOverrides] = useState({});
    const [data, setData] = useState(null);
    const [status, setStatus] = useState("idle"); // idle | loading | updating
    const [error, setError] = useState(null);
    const [serverErrors, setServerErrors] = useState({});
    const [needState, setNeedState] = useState(false);
    const [forecastDetail, setForecastDetail] = useState(null);
    const [tab, setTab] = useState(params.get("tab") === "scenario" ? "scenario" : "sensitivity");
    const [tabState, setTabState] = useState({}); // id -> { key, data, error, loading }
    const propertyKey = useRef(null);
    const resultsRef = useRef(null);

    const body = useMemo(() => buildBody(baseBody, overrides), [baseBody, overrides]);
    const requestKey = body ? JSON.stringify(body) : null;

    // Main analysis: runs on submit and (debounced) whenever a slider moves.
    useEffect(() => {
        if (!body) return;
        const controller = new AbortController();
        const newProperty = propertyKey.current !== JSON.stringify(baseBody);
        setStatus(newProperty ? "loading" : "updating");
        const timer = setTimeout(async () => {
            try {
                const res = await api.analyse(body, controller.signal);
                propertyKey.current = JSON.stringify(baseBody);
                setData(res);
                setError(null);
                setServerErrors({});
                setNeedState(false);
                setStatus("idle");
                if (newProperty) setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
            } catch (err) {
                if (err.name === "AbortError") return;
                setStatus("idle");
                setError(err);
                setServerErrors(Object.fromEntries((err.details || []).map((d) => [d.field, d.message])));
                if (err.status === 404 && /state/i.test(err.message)) setNeedState(true);
            }
        }, newProperty ? 0 : DEBOUNCE_MS);
        return () => { clearTimeout(timer); controller.abort(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [requestKey]);

    // Forecast details (method, growth observed so far) for the property's state.
    const state = data?.location.state;
    useEffect(() => {
        if (!state) return;
        const controller = new AbortController();
        api.forecast(state, controller.signal).then(setForecastDetail).catch(() => setForecastDetail(null));
        return () => controller.abort();
    }, [state]);

    // Sensitivity / scenario tab: fetched when shown, refetched when the request changes.
    useEffect(() => {
        if (!data || !body || tabState[tab]?.key === requestKey) return;
        const controller = new AbortController();
        setTabState((s) => ({ ...s, [tab]: { ...s[tab], loading: true, error: null } }));
        const timer = setTimeout(async () => {
            try {
                const res = await (tab === "scenario" ? api.scenario : api.sensitivity)(body, controller.signal);
                setTabState((s) => ({ ...s, [tab]: { key: requestKey, data: res, loading: false, error: null } }));
            } catch (err) {
                if (err.name === "AbortError") return;
                setTabState((s) => ({ ...s, [tab]: { ...s[tab], loading: false, error: err } }));
            }
        }, DEBOUNCE_MS);
        return () => { clearTimeout(timer); controller.abort(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [tab, requestKey, data]);

    useEffect(() => {
        if (!params.get("zip")) return;
        const { body: b, errors } = toRequest(form);
        if (!Object.keys(errors).length) setBaseBody(b);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const submit = (b) => {
        setParams(Object.fromEntries(URL_FIELDS.filter((k) => b[k] != null).map((k) => [k, String(b[k])])), { replace: true });
        setOverrides({});
        setTabState({});
        setBaseBody({ ...b });
    };
    const retry = () => setBaseBody((b) => (b ? { ...b } : b));

    const sliderValues = useMemo(() => {
        if (!data) return {};
        return Object.fromEntries(SLIDERS.map((s) => [s.key, overrides[s.key] ?? data.assumptions[s.key]]));
    }, [data, overrides]);

    const current = tabState[tab];

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-blue-50">
            <header className="bg-white border-b border-gray-200">
                <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 font-semibold text-gray-900">
                        <span className="bg-indigo-600 p-1.5 rounded-lg"><Home className="w-4 h-4 text-white" aria-hidden="true" /></span>
                        Rent or Buy?
                    </Link>
                    <span className="text-xs text-gray-500 hidden sm:block">Research prototype · US homes</span>
                </div>
            </header>

            <main className="max-w-6xl mx-auto px-4 sm:px-6 py-6 sm:py-10 space-y-6">
                <Card title="Describe the home" subtitle="We estimate its price and rent, forecast the market, and compare renting with buying.">
                    <PropertyForm form={form} setForm={setForm} onSubmit={submit}
                        loading={status === "loading"} serverErrors={serverErrors} needState={needState} />
                    <p className="mt-4 text-xs text-gray-500 flex flex-wrap items-center gap-2">
                        Try an example:
                        {EXAMPLES.map((e) => (
                            <button key={e.zip} type="button" className="px-2 py-1 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-700 cursor-pointer"
                                onClick={() => { setForm({ ...EMPTY_FORM, ...e, label: undefined }); setNeedState(false); }}>
                                {e.label}
                            </button>
                        ))}
                    </p>
                </Card>

                <ErrorBox error={error} onRetry={error?.status === 0 || error?.status >= 500 ? retry : null} />

                {!data && status === "loading" && (
                    <Card><Spinner label="Estimating price and rent, forecasting growth…" className="text-gray-600" /></Card>
                )}

                {data && (
                    <div ref={resultsRef} className={`space-y-6 scroll-mt-4 transition-opacity ${status === "loading" ? "opacity-50" : ""}`}>
                        <Disclaimer text={data.disclaimer} />
                        <Warnings items={data.warnings} />
                        <LocationLine loc={data.location} />
                        <ResultCards data={data} forecastDetail={forecastDetail} />

                        <Card title="Rent or buy?" subtitle={`Present-value comparison over ${data.assumptions.H} years with your assumptions.`}>
                            <DecisionPanel data={data} sliderValues={sliderValues} updating={status === "updating"}
                                onSlider={(k, v) => setOverrides((o) => ({ ...o, [k]: v }))}
                                onReset={() => setOverrides({})} />
                        </Card>

                        <Card title="Why this estimate?" subtitle="The features that moved the price and rent estimates most.">
                            <ExplainPanel data={data} />
                        </Card>

                        <Card>
                            <Tabs tabs={TABS} active={tab} onChange={setTab} />
                            <div className="mt-5" role="tabpanel">
                                {current?.error && <ErrorBox error={current.error}
                                    onRetry={() => setTabState((s) => ({ ...s, [tab]: undefined }))} />}
                                {current?.loading && !current?.data && <Spinner className="text-gray-600" />}
                                {current?.data && (
                                    <div className={`transition-opacity ${current.loading ? "opacity-60" : ""}`}>
                                        {tab === "scenario" ? <ScenarioTab data={current.data} /> : <SensitivityTab data={current.data} />}
                                        <Warnings items={current.data.warnings?.filter((w) => !data.warnings.includes(w))} />
                                        <Disclaimer text={current.data.disclaimer} className="mt-5" />
                                    </div>
                                )}
                            </div>
                        </Card>
                    </div>
                )}

                {!data && status !== "loading" && (
                    <p className="text-center text-sm text-gray-500">
                        Results appear here: estimated price and rent, growth forecasts, and a buy-or-rent recommendation.
                    </p>
                )}
            </main>
            <footer className="max-w-6xl mx-auto px-4 sm:px-6 pb-8 text-xs text-gray-500">
                {api.DISCLAIMER}. Final-year research project (Coventry University / NIBM).
            </footer>
        </div>
    );
}
