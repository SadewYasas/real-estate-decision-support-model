"""Writes artefacts/metrics/engine_tests.json (pass/fail per manual scenario) after a run
that includes tests/test_engine.py."""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_scenarios = []
_other = {"passed": 0, "failed": 0}


def pytest_runtest_logreport(report):
    if "test_engine.py" not in report.nodeid or report.when != "call":
        return
    props = dict(report.user_properties)
    if "scenario" in props:
        _scenarios.append({**props["scenario"], "outcome": report.outcome})
    else:
        _other["passed" if report.passed else "failed"] += 1


def pytest_sessionfinish(session, exitstatus):
    if not _scenarios:
        return
    out = {
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "manual_scenarios": {"total": len(_scenarios),
                             "passed": sum(s["outcome"] == "passed" for s in _scenarios),
                             "failed": sum(s["outcome"] != "passed" for s in _scenarios)},
        "property_and_validation_tests": _other,
        "scenarios": sorted(_scenarios, key=lambda s: s["id"]),
        "excel_check_workbook": "artefacts/engine_test_scenarios.xlsx",
    }
    path = ROOT / "artefacts" / "metrics" / "engine_tests.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():  # keep the Excel recalculation check written by src.engine.export_excel
        previous = json.loads(path.read_text())
        if "excel_check" in previous:
            out["excel_check"] = previous["excel_check"]
    path.write_text(json.dumps(out, indent=2, default=str))
