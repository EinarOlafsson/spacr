"""Ctrl+K survives keys that arrive while its list is being rebuilt.

A tutorial capture on the workstation died with SIGSEGV in
``CommandPalette._render`` at ``self._list.clear()``, 2026-10-12. The
capture driver typed from inside the palette's open callback, nested in its
``exec`` loop, with the field shading on its own thread. Every key rebuilt
the whole list, and nothing stopped a key dispatched from inside a rebuild
-- from a model signal, a list signal or an event filter -- from starting a
second rebuild in the middle of the first. The inner ``clear`` deletes items
the outer rebuild is still iterating and adding, and PySide only invalidates
the Python wrappers of the items it saw at the start of a ``clear``, so the
outer one leaves wrappers to freed items behind.

This drives that shape many times over: the real Ctrl+K route, a driver
nested in the open callback, keys dispatched from the list model's
``modelAboutToBeReset`` and ``rowsInserted`` and from the list's
``currentRowChanged`` while a rebuild runs, arrow keys, rows picked under the
pointer, a palette closed from inside a rebuild and typed into afterwards,
and a fresh palette opened on every round. After every key the list has to
be exactly the rows the typed text selects.

It runs in a CHILD interpreter, so the native crash the old code could take
is a failed assertion and not a dead pytest.
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
import gc, json, random, sys

import spacr
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow

app = QApplication.instance() or QApplication([])

from spacr.qt import shortcuts as sc
from spacr.qt.command_palette import CommandPalette
from spacr.qt.widgets import ambient

rounds = int(sys.argv[1])
rng = random.Random(int(sys.argv[2]))

window = QMainWindow()
window.resize(900, 600)
window.show()
field = ambient.AmbientWidget(window, theme="data_art_impulse_lens",
                              density=1, seed=19)
field.resize(window.size())
field.show()
field.lower()

WORDS = ["Meas", "mask", "pref", "short", "zz", "open", "", "e"]
state = {"wrong": [], "rounds": 0, "nested": 0, "after_close": 0}


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


def check(pal, where):
    if not pal.isVisible():
        return
    rows = shown(pal)
    if rows != expected(pal):
        state["wrong"].append([state["rounds"], where, pal._input.text(),
                               len(rows), len(expected(pal))])
        return
    for i in range(pal._list.count()):
        item = pal._list.item(i)
        if item.flags() != Qt.NoItemFlags and item.data(Qt.UserRole) is None:
            state["wrong"].append([state["rounds"], where, "no command", i])


def hook(pal, close_inside):
    depth = {"n": 0, "budget": 6}

    def key_inside(*_args):
        if depth["n"] or depth["budget"] <= 0:
            return
        depth["n"] += 1
        depth["budget"] -= 1
        state["nested"] += 1
        try:
            if close_inside and depth["budget"] == 3:
                pal.reject()
            QTest.keyClick(pal._input, rng.choice("aeimnoprst"))
        finally:
            depth["n"] -= 1

    model = pal._list.model()
    model.modelAboutToBeReset.connect(key_inside)
    model.rowsInserted.connect(key_inside)
    pal._list.currentRowChanged.connect(key_inside)


def drive():
    pal = [w for w in app.topLevelWidgets()
           if isinstance(w, CommandPalette) and w.isVisible()][-1]
    try:
        type_into(pal)
    except Exception as exc:
        state["wrong"].append([state["rounds"], "raised", repr(exc)])
    finally:
        state["rounds"] += 1
        pal.reject()


def type_into(pal):
    close_inside = state["rounds"] % 4 == 3
    hook(pal, close_inside)
    for _ in range(3):
        for ch in rng.choice(WORDS):
            QTest.keyClick(pal._input, ch)
            check(pal, "key")
            if rng.random() < .5:
                app.processEvents()
                check(pal, "events")
            if rng.random() < .3:
                QTest.keyClick(pal, Qt.Key_Down)
            if rng.random() < .3 and pal._list.count() > 1:
                pal._list.setCurrentRow(rng.randrange(pal._list.count()))
                QTest.mouseMove(pal._list.viewport(),
                                QPoint(20, 10 + 20 * rng.randrange(8)))
                check(pal, "pick")
        for _ in range(rng.randrange(4)):
            QTest.keyClick(pal._input, Qt.Key_Backspace)
            check(pal, "backspace")
    if not pal.isVisible():
        rows = shown(pal)
        QTest.keyClicks(pal._input, "zz")
        app.processEvents()
        if shown(pal) != rows:
            state["after_close"] += 1
    if rng.random() < .2:
        gc.collect()


for _ in range(rounds):
    QTimer.singleShot(0, drive)
    sc._open_palette(window)
    app.processEvents()
QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
app.processEvents()
alive = len(window.findChildren(CommandPalette))
field._timer.stop()
print("REPORT " + json.dumps({
    "spacr": spacr.__file__, "rounds": state["rounds"],
    "nested": state["nested"], "wrong": state["wrong"][:5],
    "after_close": state["after_close"], "alive": alive,
}), flush=True)
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


@pytest.mark.parametrize("seed", [1, 2])
def test_keys_typed_into_a_rebuild_leave_the_list_whole(seed):
    report = _run_child(60, seed)
    assert report["rounds"] == 60
    assert report["nested"] > 0, "no key ever arrived inside a rebuild"
    assert report["wrong"] == [], report["wrong"]
    assert report["after_close"] == 0
    assert report["alive"] == 0, "closed palettes stay under the window"
