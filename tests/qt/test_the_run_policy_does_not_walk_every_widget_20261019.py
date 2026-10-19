"""A job starting or ending finds the backdrops without walking every widget.

The animation policy paused fractals and re-paced backdrops by walking
``QApplication.allWidgets()``, which wraps every widget of the window in
Python; when lesson 13's run ended that walk was a fifth of the GUI thread's
samples. Backdrops and fractals now register themselves. Counted, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def no_walks(monkeypatch, qapp):
    walks = []
    real = type(qapp).allWidgets

    def counting():
        walks.append(1)
        return real()

    monkeypatch.setattr(type(qapp), "allWidgets", staticmethod(counting))
    return walks


def test_a_backdrop_is_found_and_re_paced_without_a_walk(qtbot, no_walks):
    host = QWidget()
    qtbot.addWidget(host)
    backdrop = ambient.AmbientWidget(host)
    synced = []
    backdrop._sync_run_state = lambda: synced.append(True)

    assert backdrop in ambient._live_ambient_widgets()
    policy = ambient._AnimationResourcePolicy()
    policy.cpu_busy = True
    policy._apply()

    assert synced == [True]
    assert no_walks == []


def test_a_fractal_is_found_without_a_walk(qtbot, no_walks):
    host = QWidget()
    qtbot.addWidget(host)
    fractal = QWidget(host)
    fractal._spaceout_built_from = ("test",)
    ambient._LIVE_SPACEOUT_FRACTALS.add(fractal)
    loose = QWidget()
    qtbot.addWidget(loose)
    loose._spaceout_built_from = ("test",)
    ambient._LIVE_SPACEOUT_FRACTALS.add(loose)

    found = ambient._live_spaceout_fractals()

    assert fractal in found
    assert loose not in found
    assert no_walks == []


def test_a_deleted_backdrop_leaves_the_registry(qtbot):
    host = QWidget()
    backdrop = ambient.AmbientWidget(host)
    assert backdrop in ambient._live_ambient_widgets()
    from shiboken6 import delete
    delete(host)
    assert all(widget is not backdrop for widget in ambient._live_ambient_widgets())
