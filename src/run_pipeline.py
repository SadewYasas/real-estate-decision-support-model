"""Run the whole research pipeline in order, from the raw data to every Chapter 5 output.

Each step runs as its own process; its output goes to artefacts/logs/pipeline/<step>.log.
Before running, every metrics JSON is snapshotted; afterwards the new results are compared
with the snapshot (timings and timestamps ignored) to show the pipeline reproduces itself.
The run record is saved to artefacts/metrics/pipeline_run.json.

    python -m src.run_pipeline            # everything except the ~3 h price-model training;
                                          # the saved price models are re-scored instead
    python -m src.run_pipeline --train    # also retrain the sale and rent models from scratch

Requires Microsoft Excel for the Excel recalculation step (skipped gracefully otherwise).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.config import METRICS_DIR, ROOT

LOG_DIR = ROOT / "artefacts" / "logs" / "pipeline"
PY = sys.executable

STEPS = [  # (name, command, what it produces)
    ("clean_sale", ["-m", "src.pipeline.clean_sale"], "data/processed/sale_clean.csv"),
    ("clean_rent", ["-m", "src.pipeline.clean_rent"], "data/processed/rent_clean.csv"),
    ("leakage_audit", ["-m", "src.pipeline.leakage_audit"], "leakage_audit.json"),
    ("eda", ["-m", "src.pipeline.eda"], "eda_summary.json + 6 figures"),
    ("price_models", None, "sale/rent models (train or verify)"),
    ("build_panel", ["-m", "src.pipeline.build_panel"], "HPI + rent panels, CPI factors"),
    ("train_forecast", ["-m", "src.models.train_forecast"], "forecast.json, forecasts.json"),
    ("build_zip_lookup", ["-m", "src.pipeline.build_zip_lookup"], "zip_lookup.csv"),
    ("tests", ["-m", "pytest", "tests", "-q"], "engine_tests.json (+ API tests)"),
    ("export_excel", ["-m", "src.engine.export_excel"], "engine_test_scenarios.xlsx"),
    ("engine_example", ["-m", "src.engine.example"], "cost_over_time_example.png"),
    ("scenario_example", ["-m", "src.engine.scenario"], "scenario_example.json"),
    ("sensitivity_example", ["-m", "src.engine.sensitivity"], "tornado_sensitivity.png"),
    ("decision_flip", ["-m", "src.evaluation.decision_flip"], "decision_flip.json"),
    ("performance", ["-m", "src.api.performance"], "performance.json"),
]

# Keys whose values legitimately change between runs.
VOLATILE = {"run_at", "fit_seconds", "startup_seconds", "startup_model_loading_s", "in_process",
            "http_localhost", "machine", "python", "seconds"}
SKIP_FILES = {"performance.json", "pipeline_run.json"}


def snapshot() -> dict:
    return {p.name: json.loads(p.read_text()) for p in sorted(METRICS_DIR.glob("*.json"))
            if p.name not in SKIP_FILES}


def compare(a, b, path="", out=None, tol=1e-9):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in only one run")
            else:
                compare(a[k], b[k], f"{path}/{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: list length {len(a)} -> {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
            out.append(f"{path}: {a} -> {b}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]} -> {str(b)[:60]}")
    return out


def run_step(name, args) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(LOG_DIR / f"{name}.log", "w", encoding="utf-8") as log:
        proc = subprocess.run([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    return {"step": name, "returncode": proc.returncode, "seconds": round(time.time() - t0, 1)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="retrain the sale and rent models (~3 h)")
    args = ap.parse_args()

    before = snapshot()
    results, t_start = [], time.time()
    for name, cmd, produces in STEPS:
        if name == "price_models":
            if args.train:
                for ds in ("sale", "rent"):
                    r = run_step(f"train_{ds}", ["-m", "src.models.train_price_models", "--dataset", ds,
                                                 "--fresh", "--log", f"artefacts/logs/train_{ds}.log"])
                    results.append({**r, "produces": f"{ds}_models.json"})
                    print(f"{r['step']:<22}{'OK' if r['returncode'] == 0 else 'FAILED'}  {r['seconds']:>7.1f}s", flush=True)
                continue
            name, cmd, produces = "verify_price_models", ["-m", "src.models.verify_price_models"], \
                "price_models_reproducibility.json"
        r = {**run_step(name, cmd), "produces": produces}
        results.append(r)
        print(f"{name:<22}{'OK' if r['returncode'] == 0 else 'FAILED'}  {r['seconds']:>7.1f}s  -> {produces}", flush=True)

    after = snapshot()
    diffs = {}
    for fname in sorted(set(before) | set(after)):
        if fname not in before:
            diffs[fname] = ["new file"]
        elif fname not in after:
            diffs[fname] = ["missing after run"]
        else:
            d = compare(before[fname], after[fname])
            if d:
                diffs[fname] = d[:20] + ([f"... {len(d) - 20} more"] if len(d) > 20 else [])
    record = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrained_price_models": args.train,
        "total_seconds": round(time.time() - t_start, 1),
        "steps": results,
        "all_steps_ok": all(r["returncode"] == 0 for r in results),
        "metrics_files_compared": len(set(before) & set(after)),
        "metrics_files_identical": sorted(f for f in set(before) & set(after) if f not in diffs),
        "differences_vs_previous_run": diffs,
    }
    (METRICS_DIR / "pipeline_run.json").write_text(json.dumps(record, indent=2))
    print(f"\nAll steps OK: {record['all_steps_ok']} | {record['total_seconds'] / 60:.1f} min | "
          f"{len(record['metrics_files_identical'])}/{record['metrics_files_compared']} metrics files "
          f"identical to the previous run")
    for f, d in diffs.items():
        print(f"  {f}: {len(d)} difference(s), e.g. {d[0]}")
    sys.exit(0 if record["all_steps_ok"] else 1)


if __name__ == "__main__":
    main()
