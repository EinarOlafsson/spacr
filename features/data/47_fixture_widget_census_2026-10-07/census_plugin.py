"""Observe the real Qt widget population at selected pytest boundaries."""

import json
import os
from collections import Counter


def _snapshot(event, item):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    widgets = list(app.allWidgets()) if app is not None else []
    kinds = Counter(type(widget).__name__ for widget in widgets)
    tops = [widget for widget in widgets if widget.parentWidget() is None]
    record = {
        "event": event,
        "nodeid": item.nodeid,
        "all_widgets": len(widgets),
        "top_level_widgets": len(tops),
        "kinds": dict(sorted(kinds.items())),
        "top_level_kinds": dict(sorted(Counter(type(widget).__name__ for widget in tops).items())),
    }
    with open(os.environ["SPACR_WIDGET_CENSUS"], "a", encoding="utf-8") as out:
        out.write(json.dumps(record, sort_keys=True) + "\n")


def pytest_runtest_setup(item):
    _snapshot("next_test_setup", item)


def pytest_runtest_teardown(item, nextitem):
    _snapshot("test_teardown_hook", item)
