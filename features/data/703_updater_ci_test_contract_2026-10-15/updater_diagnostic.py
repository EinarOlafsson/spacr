import json
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        from spacr.qt.bridge import registry
        modals = item.funcargs.get("modals")
        window = item.funcargs.get("win")
        print("UPDATER_DIAGNOSTIC " + json.dumps({"outcome": report.outcome, "information": getattr(modals, "information", None), "warning": getattr(modals, "warning", None), "questions": getattr(modals, "questions", None), "closing": getattr(window, "_closing", None), "registered": [str(getattr(h, "app_key", "")) for h in registry().active()]}), flush=True)
