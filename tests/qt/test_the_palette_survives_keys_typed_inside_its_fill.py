"""Ctrl+K survives keys, filters, closes and deletes that land inside a fill.

The lesson 05 capture on the workstation died with SIGSEGV a second time on
2026-10-12, after the first fix (874223b49), now inside
``CommandPalette._fill`` at ``self._list.clear()``: the capture driver typed
into the palette from a timer in the palette's own ``exec`` loop while the
spaCR field shaded on its own threads. A list widget's ``clear`` and
``addItem`` hand the interpreter lock to every other Python thread for as long
as they run (measured: another thread ran over a million steps during one
3,000-item ``clear``), and every item ``clear`` deletes is a C++ object with a
Python wrapper and a Python command in its data, so the teardown of rows the
view still knew about ran interleaved with the field's Python.

The palette now builds its rows as plain Python objects and points the view
at a new model holding them in one call; nothing the view shows is ever
cleared or deleted under it. This drives the shape of the second crash many
times over, in a child interpreter with ``faulthandler`` on:

* the real Ctrl+K route, the driver nested in the palette's ``exec`` loop;
* the field animating, the interpreter switching threads every millisecond
  as it does while a worker runs, and a thread allocating and collecting
  garbage the whole time;
* from INSIDE the fill, while its rows are being built: keys typed into the
  input, a filter asked for directly, the event loop pumped, the palette
  closed, and the palette deleted outright.

After every key the list has to be exactly the rows the typed text selects,
the input and the list must not be left silent, and every closed palette has
to be released.
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
import faulthandler, gc, json, random, sys, threading

faulthandler.enable()

import spacr
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow
from shiboken6 import isValid

app = QApplication.instance() or QApplication([])

from spacr import run_journal
from spacr.qt import shortcuts as sc
from spacr.qt.command_palette import CommandPalette
from spacr.qt.widgets import ambient

run_journal.recent_runs = lambda limit=10: []

rounds = int(sys.argv[1])
rng = random.Random(int(sys.argv[2]))
sys.setswitchinterval(0.001)

window = QMainWindow()
window.resize(900, 600)
window.show()
field = ambient.AmbientWidget(window, theme="data_art_impulse_lens",
                              density=1, seed=23)
field.resize(window.size())
field.show()
field.lower()

stop = threading.Event()


def churn():
    heap = []
    while not stop.wait(0.0005):
        heap.append([{"k": i} for i in range(200)])
        if len(heap) > 40:
            heap.clear()
            gc.collect(1)


churner = threading.Thread(target=churn, daemon=True)
churner.start()

WORDS = ["Meas", "mask", "pref", "short", "zz", "open", "", "e", "go"]
state = {"wrong": [], "rounds": 0, "inside": 0, "typed_inside": 0,
         "closed_inside": 0, "deleted_inside": 0, "silent": 0}


def expected(pal):
    needle = pal._input.text().strip().lower()
    cmds = [c for c in pal._commands
            if not needle or needle in c.label.lower()
            or any(needle in k.lower() for k in c.keywords)]
    rows, section = [], None
    for cmd in cmds:
        if cmd.section != section:
            rows.append(cmd.section.upper())
            section = cmd.section
        rows.append(cmd.label)
    return rows


def shown(pal):
    return [pal._list.item(i).text() for i in range(pal._list.count())]


def alive(pal):
    return isValid(pal) and pal.isVisible()


def check(pal, where):
    if not alive(pal):
        return
    if pal._input.signalsBlocked() or pal._list.signalsBlocked():
        state["silent"] += 1
    rows = shown(pal)
    if rows != expected(pal):
        state["wrong"].append([state["rounds"], where, pal._input.text(),
                               len(rows), len(expected(pal))])
        return
    for i in range(pal._list.count()):
        item = pal._list.item(i)
        if item.flags() != Qt.NoItemFlags and item.data(Qt.UserRole) is None:
            state["wrong"].append([state["rounds"], where, "no command", i])


def inside_the_fill(pal, plan):
    build = pal._rows_for
    budget = plan["budget"]

    def built(cmds):
        if budget["n"] > 0 and isValid(pal):
            budget["n"] -= 1
            state["inside"] += 1
            before = pal._input.text()
            QTest.keyClick(pal._input, rng.choice("aeimnoprst"))
            if pal._input.text() != before:
                state["typed_inside"] += 1
            if rng.random() < .4:
                pal._on_filter(rng.choice(WORDS))
            if rng.random() < .6:
                app.processEvents()
            if plan["close"] and budget["n"] == 3:
                state["closed_inside"] += 1
                pal.reject()
            if plan["delete"]:
                plan["delete"] = False
                state["deleted_inside"] += 1
                pal.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        return build(cmds)

    pal._rows_for = built


def drive():
    pal = [w for w in app.topLevelWidgets()
           if isinstance(w, CommandPalette) and w.isVisible()][-1]
    try:
        type_into(pal)
    except Exception as exc:
        state["wrong"].append([state["rounds"], "raised", repr(exc)])
    finally:
        state["rounds"] += 1
        if isValid(pal):
            pal.reject()


def type_into(pal):
    plan = {"close": state["rounds"] % 5 == 3, "delete": False,
            "budget": {"n": 7}}
    inside_the_fill(pal, plan)
    for _ in range(3):
        for ch in rng.choice(WORDS):
            if not alive(pal):
                return
            QTest.keyClick(pal._input, ch)
            check(pal, "key")
            if rng.random() < .5:
                app.processEvents()
                check(pal, "events")
            if alive(pal) and rng.random() < .3:
                QTest.keyClick(pal, Qt.Key_Down)
            if alive(pal) and rng.random() < .2:
                pal._on_filter(pal._input.text())
                check(pal, "filter")
        for _ in range(rng.randrange(3)):
            if alive(pal):
                QTest.keyClick(pal._input, Qt.Key_Backspace)
                check(pal, "backspace")
    if alive(pal) and state["rounds"] % 5 == 4:
        plan["delete"] = True
        plan["budget"]["n"] = 1
        pal._on_filter(rng.choice(WORDS))
        if isValid(pal):
            state["wrong"].append([state["rounds"], "not deleted"])
    if rng.random() < .2:
        gc.collect()


for _ in range(rounds):
    QTimer.singleShot(0, drive)
    sc._open_palette(window)
    app.processEvents()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
app.processEvents()
left = len(window.findChildren(CommandPalette))
stop.set()
churner.join(5)
field._timer.stop()
state["wrong"] = state["wrong"][:5]
state["alive"] = left
state["spacr"] = spacr.__file__
print("REPORT " + json.dumps(state), flush=True)
"""


def _run_child(rounds, seed):
    import spacr

    root = Path(spacr.__file__).resolve().parent.parent
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", _CHILD,
                           str(rounds), str(seed)], cwd=root, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, (
        f"the child died (exit {done.returncode})\n{done.stderr[-3000:]}")
    lines = [line for line in done.stdout.splitlines()
             if line.startswith("REPORT ")]
    assert lines, f"the child reported nothing\n{done.stderr[-3000:]}"
    report = json.loads(lines[-1][len("REPORT "):])
    assert Path(report["spacr"]).resolve().parent.parent == root
    return report


@pytest.mark.parametrize("seed", [3, 4])
def test_keys_closes_and_deletes_inside_a_fill_leave_the_palette_whole(seed):
    report = _run_child(100, seed)
    assert report["rounds"] == 100
    assert report["inside"] > 100, "the fill was hardly ever entered"
    assert report["typed_inside"] > 0, "no key was typed inside a fill"
    assert report["closed_inside"] > 0, "no palette closed inside a fill"
    assert report["deleted_inside"] >= 10, "no palette deleted inside a fill"
    assert report["wrong"] == [], report["wrong"]
    assert report["silent"] == 0, "a fill left the input or the list silent"
    assert report["alive"] == 0, "closed palettes stay under the window"
