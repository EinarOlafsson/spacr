"""A tab bar's close-mark watcher lets the bar go without raising.

``install_close_marks`` parents a ``_CloseMarkWatcher`` to each tab bar,
stores it on the bar and installs it as the bar's event filter. The watcher
kept the bar in ``_bar``, so the two formed a reference cycle that only
Python's cycle collector could free. The collector clears the watcher's
``__dict__`` before the bar's C++ object is destroyed; the bar's destructor
then removes its children, notifies its event filter, and
``_CloseMarkWatcher.eventFilter`` raised ``AttributeError: '_CloseMarkWatcher'
object has no attribute '_bar'`` inside the Qt event loop -- the shape
7bc459287 fixed in the backdrops, found here by the same sweep, 2026-09-15.

Two repairs, tested apart:

* the watcher holds the bar weakly, so no cycle forms and reference counting
  frees the bar while every attribute is still in place;
* the filter and the sweep pass when the bar is gone, because a cycle made
  anywhere else through the bar reopens the same path.

The collector runs in a CHILD interpreter. conftest records that collecting
a live Qt heap in the middle of a run can segfault, and a child turns any
such crash into a failed assertion instead of a dead pytest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

_CHILD = r"""
import gc, json, sys, weakref

import spacr
from PySide6.QtWidgets import QApplication, QTabBar

from spacr.qt.theme import install_close_marks

app = QApplication.instance() or QApplication([])
extra_cycle = sys.argv[1] == "extra_cycle"

bar = QTabBar()
bar.addTab("one")
bar.addTab("two")
install_close_marks(bar)
bar.show()
app.processEvents()
watched = getattr(bar, "_spacr_close_mark_watcher", None) is not None
if extra_cycle:
    bar._keeps_itself = [bar]

gone = weakref.ref(bar)
gc.collect()
gc.disable()
del bar
freed_without_collector = gone() is None
sys.stderr.write("---- COLLECT ----\n")
sys.stderr.flush()
gc.collect()
app.processEvents()
print("REPORT " + json.dumps({
    "spacr": spacr.__file__,
    "watched": watched,
    "freed_without_collector": freed_without_collector,
    "freed": gone() is None,
}), flush=True)
"""


def _run_child(case):
    import spacr

    # Rooted at the checkout this test imported, so the child runs the code
    # under test and not an installed copy.
    root = Path(spacr.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", _CHILD,
                           case], cwd=root, env=env, capture_output=True,
                          text=True, timeout=300)
    assert done.returncode == 0, (
        f"the child died (exit {done.returncode})\n{done.stderr[-3000:]}")
    lines = [line for line in done.stdout.splitlines()
             if line.startswith("REPORT ")]
    assert lines, f"the child reported nothing\n{done.stderr[-3000:]}"
    report = json.loads(lines[-1][len("REPORT "):])
    assert Path(report["spacr"]).resolve().parent.parent == root
    after = done.stderr.split("---- COLLECT ----", 1)[-1]
    problems = [line.strip() for line in after.splitlines()
                if "Error" in line or "Fatal" in line]
    return report, problems


def test_the_watcher_does_not_keep_its_bar_alive():
    """With the collector OFF, dropping the last name frees the bar, and
    nothing raises when the collector runs afterwards."""
    report, problems = _run_child("plain")

    assert report["watched"], "install_close_marks installed no watcher"
    assert report["freed_without_collector"], (
        "the tab bar outlived its last name: a reference cycle holds it")
    assert report["freed"]
    assert not problems, problems


def test_a_bar_freed_by_the_collector_does_not_reach_an_emptied_watcher():
    """A cycle through the bar that is not the watcher's own still leaves
    the collector to free it; the emptied watcher must pass the teardown
    events on quietly."""
    report, problems = _run_child("extra_cycle")

    assert report["watched"], "install_close_marks installed no watcher"
    assert not report["freed_without_collector"], (
        "the extra cycle did not hold the bar, so this case tests nothing")
    assert report["freed"], "the collector did not free the bar"
    assert not problems, problems


def test_a_sweep_that_meets_a_deleted_bar_passes_quietly():
    """The sweep is queued, and Qt can delete the bar before it runs.

    Holding the bar weakly usually leaves the sweep no bar at all. But a
    Python name can outlive the C++ bar -- each close mark's click handler
    keeps one -- and then the sweep is handed a wrapper whose every Qt call
    raises RuntimeError. That has to end in the sweep, not in the event
    loop. In process: no collector runs here.
    """
    import shiboken6
    from PySide6.QtWidgets import QTabWidget, QWidget

    from spacr.qt.theme import install_close_marks, mark_tab_bar

    tabs = QTabWidget()
    tabs.addTab(QWidget(), "one")
    tabs.setTabsClosable(True)
    install_close_marks(tabs)
    bar = tabs.tabBar()
    watcher = bar._spacr_close_mark_watcher
    shiboken6.delete(tabs)
    assert not shiboken6.isValid(bar)

    # The path is real: marking a bar Qt has deleted raises.
    with pytest.raises(RuntimeError):
        mark_tab_bar(bar)

    watcher._sweep()

    assert watcher._pending is False
