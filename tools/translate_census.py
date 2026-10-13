"""Record every string a module-screen build renders, and what it cost.

380's condition for touching the triple-translation: "capture every
translated widget's text after a build, apply the change, compare the sets".
This is that capture, and the cost meter with it -- `_translate_qt_text` is
the per-widget unit of work, so counting its calls counts the passes' real
size rather than what they could have walked.
"""
from __future__ import annotations

import json
import os
import sys
import time

# THE REPOSITORY, NOT THE INSTALLED COPY. A Python script puts its OWN
# directory on `sys.path`, never the working directory, so running this from
# a checkout measured whatever `pip install spacr` had left in site-packages
# -- identical numbers before and after an edit, with nothing in the output
# to say the edit had not been loaded. Cost an hour on 2026-09-11.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
KEY = sys.argv[1] if len(sys.argv) > 1 else "measure"
os.environ["SPACR_LANGUAGE"] = sys.argv[2] if len(sys.argv) > 2 else "sv"
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/census.json"

from PySide6.QtWidgets import QApplication, QWidget      # noqa: E402

import spacr.qt.i18n as i18n                             # noqa: E402

COUNT = {"fields": 0, "widgets": 0, "passes": 0}
_real_field = i18n._translate_qt_text
_real_pass = i18n.retranslate_widget_tree
_seen = set()


def counted_field(obj, *a, **k):
    COUNT["fields"] += 1
    _seen.add(id(obj))
    return _real_field(obj, *a, **k)


def counted_pass(root, *a, **k):
    COUNT["passes"] += 1
    return _real_pass(root, *a, **k)


i18n._translate_qt_text = counted_field
i18n.retranslate_widget_tree = counted_pass


def path_of(widget):
    parts = []
    node = widget
    while node is not None:
        parent = node.parent()
        if parent is None:
            parts.append(type(node).__name__)
            break
        try:
            index = list(parent.children()).index(node)
        except ValueError:
            index = -1
        parts.append(f"{type(node).__name__}#{index}")
        node = parent
    return "/".join(reversed(parts))


#: Live readings, not captions. They move between two runs of this script
#: for reasons that have nothing to do with translation: `UsageBar` is the
#: machine's CPU/RAM/disk, and `ActivitySpinner` carries a caption only while
#: it is spinning.
LIVE = ("UsageBar", "ActivitySpinner")


def census(root):
    out = {}
    for widget in [root] + root.findChildren(QWidget):
        path = path_of(widget)
        if any(name in path for name in LIVE):
            continue
        record = {}
        for name in ("text", "toolTip", "windowTitle", "placeholderText",
                     "title", "accessibleName", "accessibleDescription"):
            getter = getattr(widget, name, None)
            if callable(getter):
                try:
                    value = str(getter() or "")
                except Exception:
                    continue
                if value:
                    record[name] = value
        if record:
            out[path] = record
    return out


def main():
    app = QApplication.instance() or QApplication([])
    from spacr.qt.screens.app_screen import AppScreen

    started = time.perf_counter()
    screen = AppScreen(KEY)
    screen.resize(1600, 1000)
    screen.show()
    counted_pass(screen)                 # what MainWindow does, once
    for _ in range(60):
        app.processEvents()
    elapsed = time.perf_counter() - started
    COUNT["widgets"] = len(_seen)
    rows = census(screen)
    result = {
        "app": KEY, "language": os.environ["SPACR_LANGUAGE"],
        "passes": COUNT["passes"], "field_calls": COUNT["fields"],
        "widgets_touched": COUNT["widgets"], "strings": len(rows),
        "build_seconds": round(elapsed, 3),
    }
    print(json.dumps(result))
    with open(OUT, "w") as fh:
        json.dump({"rows": rows, **result}, fh, indent=1, sort_keys=True)


main()
