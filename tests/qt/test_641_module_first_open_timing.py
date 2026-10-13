"""Measure first module opens serially, without coverage or sibling workers.

The unchanged per-module budget is deliberately generous: the startup
benchmark measured first opens between 0.1 and 2.0 seconds on a workstation
at load 15-30. Crossing ten seconds means added seconds of GUI-thread work.
Untimed navigation and structural guards remain in the companion file.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

BUDGET_S = 10.0


def _keys():
    from spacr.qt.app import APPS

    return [key for key, _name, _description, _section in APPS]


@pytest.fixture(scope="module")
def window(qapp):
    from spacr.qt.app import MainWindow

    win = MainWindow(initial_app="__home__")
    win.resize(1400, 900)
    win.show()
    for _ in range(5):
        qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("key", _keys())
def test_a_module_first_open_stays_inside_its_budget(window, qapp, key):
    assert key not in window._screens
    started = time.perf_counter()
    window._on_nav_selected(key)
    for _ in range(3):
        qapp.processEvents()
    elapsed = time.perf_counter() - started
    assert window._stack.currentWidget() is window._screens.get(key)
    assert elapsed < BUDGET_S, f"{key} first open took {elapsed:.2f} s"
