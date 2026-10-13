"""The shortcut editor opens from the ``?`` sheet and closes, many times over.

The lesson 08 capture (scene 37, "change shortcuts") died with SIGSEGV on
its first take, right after the Cancel click that closed the editor: Escape,
Help > Keyboard shortcuts, "Change shortcuts..." on the sheet, thirteen
seconds of pointer moves over the editor, then a click on one of its
buttons. The sheet deletes itself from inside its own button's click and the
editor opens in its place, over a window whose field is shading on its own
thread.

Every editor that closed used to stay alive, hidden under the window, with
its table, its key editors and its popup backdrop: a session that visited it
twenty times carried twenty of them. The editor is now deleted when it
closes, one at a time per window.

This drives that sequence in a child interpreter with ``faulthandler`` on,
the field animating, the interpreter switching threads every millisecond
and a thread allocating and collecting garbage the whole time, and closes
the editor by every route: Cancel, Save and Escape.
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
import faulthandler, gc, json, random, sys, threading, time

faulthandler.enable()

import spacr
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QApplication, QDialog, QMainWindow,
                               QPushButton, QWidget)
from shiboken6 import isValid

app = QApplication.instance() or QApplication([])

from spacr.qt import shortcuts as sc
from spacr.qt.app import install_the_dialog_filters
from spacr.qt.widgets import ambient

rounds = int(sys.argv[1])
rng = random.Random(int(sys.argv[2]))
sys.setswitchinterval(0.001)

window = QMainWindow()
window.resize(1100, 760)
window.setCentralWidget(QWidget())
window.show()
field = ambient.AmbientWidget(window, theme="data_art_impulse_lens",
                              density=1, seed=23)
field.resize(window.size())
field.show()
field.lower()
install_the_dialog_filters(app)

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


def pump(n=3):
    for _ in range(n):
        app.processEvents()


def editors():
    return [d for d in window.findChildren(QDialog)
            if d.objectName() == "KeymapDialog"]


state = {"rounds": 0, "opened": 0, "most_alive": 0, "wrong": [],
         "routes": {"Cancel": 0, "Save": 0, "Escape": 0}}
for index in range(rounds):
    QTest.keyClick(window, Qt.Key_Escape)
    pump()
    sc.show_cheat_sheet(window)
    pump()
    buttons = [b for b in window.findChildren(QPushButton, "ShortcutOverlayEdit")
               if b.isVisible()]
    if not buttons:
        state["wrong"].append([index, "no Change shortcuts button"])
        continue
    QTest.mouseMove(buttons[-1])
    QTest.mouseClick(buttons[-1], Qt.LeftButton)
    pump()
    shown = [d for d in editors() if d.isVisible()]
    if len(shown) != 1:
        state["wrong"].append([index, f"{len(shown)} editors showing"])
        continue
    state["opened"] += 1
    dialog = shown[0]
    parts = [w for w in dialog.findChildren(QWidget) if w.isVisible()]
    for _ in range(rng.randint(5, 25)):
        part = rng.choice(parts)
        if isValid(part):
            QTest.mouseMove(part, QPoint(rng.randint(0, max(0, part.width() - 1)),
                                         rng.randint(0, max(0, part.height() - 1))))
        pump(1)
    route = rng.choice(["Cancel", "Save", "Escape"])
    if route == "Escape":
        QTest.keyClick(dialog, Qt.Key_Escape)
    else:
        named = {"Save": "KeymapSave"}
        target = [b for b in dialog.findChildren(QPushButton)
                  if (b.objectName() == named.get(route)
                      or b.text().replace("&", "") == route)]
        QTest.mouseMove(target[0])
        QTest.mouseClick(target[0], Qt.LeftButton)
    state["routes"][route] += 1
    pump(3)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    pump(2)
    state["most_alive"] = max(state["most_alive"], len(editors()))
    if rng.random() < .2:
        gc.collect()
    state["rounds"] += 1

stop.set()
churner.join(5)
field._timer.stop()
state["alive"] = len(editors())
state["wrong"] = state["wrong"][:5]
state["spacr"] = spacr.__file__
print("REPORT " + json.dumps(state), flush=True)
"""


def _run_child(rounds, seed, tmp_path):
    import spacr

    root = Path(spacr.__file__).resolve().parent.parent
    env = dict(os.environ)
    env.update(QT_QPA_PLATFORM="offscreen", CUDA_VISIBLE_DEVICES="",
               SPACR_NO_GL="1", SPACR_HOME=str(tmp_path / "spacr-home"),
               XDG_CONFIG_HOME=str(tmp_path / "config"),
               SPACR_LOG_DIR=str(tmp_path / "logs"))
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", _CHILD,
                           str(rounds), str(seed)], cwd=root, env=env,
                          capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, (
        f"the child died (exit {done.returncode})\n{done.stderr[-3000:]}")
    lines = [line for line in done.stdout.splitlines()
             if line.startswith("REPORT ")]
    assert lines, f"the child reported nothing\n{done.stderr[-3000:]}"
    report = json.loads(lines[-1][len("REPORT "):])
    assert Path(report["spacr"]).resolve().parent.parent == root
    return report


@pytest.mark.slow
@pytest.mark.parametrize("seed", [8, 37])
def test_opening_and_closing_the_shortcut_editor_never_crashes(seed, tmp_path):
    rounds = 60
    report = _run_child(rounds, seed, tmp_path)
    assert report["wrong"] == [], report["wrong"]
    assert report["rounds"] == rounds
    assert report["opened"] == rounds
    assert all(report["routes"].values()), report["routes"]
    assert report["most_alive"] == 0, (
        "a closed shortcut editor stayed alive under the window")
    assert report["alive"] == 0
