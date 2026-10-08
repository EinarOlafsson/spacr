"""The spaceout backdrop and the resonance plate at their edges.

Pinned here, each as what the user sees or gets:

* a resonance plate asked for a zero-sized canvas has no grains and paints
  nothing;
* a heavy import that holds the GL lock is passed up to the caller, which
  retries, instead of being swallowed as a broken backdrop;
* the search for running backdrops finds none without an application and
  skips a widget that cannot answer;
* moving the references from an old backdrop to a new one survives a host
  that cannot name its window and a holder with no attribute dictionary;
* retiring an old backdrop carries on through each step that fails:
  it is still unparented when stopping it fails, and nothing escapes when
  its filter or its parent cannot be dropped;
* a rebuild with unreadable settings rebuilds nothing, one whose old
  backdrop cannot say whether it is paused builds a running one, and one
  whose new backdrop refuses to pause still replaces the old.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

import spacr.qt.theme as theme
from spacr.qt import preferences as P
from spacr.qt.widgets import ambient as A
from spacr.qt.widgets import fractal_travel as ft


@pytest.fixture
def spaceout(tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(P, "_SAFE_MODE", False, raising=False)
    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    monkeypatch.setattr(ft, "_LIVE_CONTROLS", [])
    P.set_fractal_settings(pattern="orbit", backend="cpu",
                           quality="balanced", scale=1.0)
    P.set_ambient_animation(A.SPACEOUT_THEME)
    made = []
    yield made
    for host in made:
        A._retire_fractals_on(host)
        host.deleteLater()
    QApplication.processEvents()


def _window_with_a_backdrop(made):
    window = QWidget()
    host = QWidget(window)
    window.resize(400, 300)
    host.resize(400, 300)
    made.append(window)
    window._dock_backdrop = A.install_ambient(
        host, None, theme=A.SPACEOUT_THEME, palette=A.SPACEOUT_PALETTE)
    assert window._dock_backdrop is not None
    return window, host


# ---------------------------------------------------------------------------
# The resonance plate on a canvas with no area
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width, height", [(0, 200), (320, 0)])
def test_a_plate_with_no_area_has_no_grains_and_paints_nothing(
        qapp, width, height):
    engine = A.ResonanceEngine(A.PALETTE_SETS["spacr"].colors,
                               "#101418", seed=7)
    assert engine.geometry(width, height) == ()

    image = QImage(8, 8, QImage.Format_RGB32)
    image.fill(QColor("#101418"))
    before = image.copy()
    painter = QPainter(image)
    engine._paint_field(painter, width, height)
    painter.end()
    assert image == before
    assert engine._floor is None


# ---------------------------------------------------------------------------
# Building one
# ---------------------------------------------------------------------------

def test_a_heavy_import_is_handed_up_for_a_retry(spaceout, monkeypatch):
    def busy(*_a, **_k):
        raise ft._HeavyImportInProgress("torch is importing")

    monkeypatch.setattr(A, "_build_the_spaceout_fractal", busy)
    host = QWidget()
    spaceout.append(host)
    with pytest.raises(ft._HeavyImportInProgress) as caught:
        A._the_spaceout_fractal(host)
    assert A._the_backdrop_wants_a_retry(caught.value)
    assert A._live_spaceout_fractals() == []


# ---------------------------------------------------------------------------
# Finding the running ones
# ---------------------------------------------------------------------------

def test_without_an_application_there_are_no_backdrops(qapp, monkeypatch):
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    assert A._live_spaceout_fractals() == []


def test_a_widget_that_cannot_answer_is_skipped(spaceout):
    class Broken(QWidget):
        _spaceout_built_from = ("orbit",)

        def parentWidget(self):  # noqa: N802 - Qt naming
            raise RuntimeError("Internal C++ object already deleted.")

    broken = Broken()
    spaceout.append(broken)
    window, _host = _window_with_a_backdrop(spaceout)
    assert A._live_spaceout_fractals() == [window._dock_backdrop]


# ---------------------------------------------------------------------------
# Moving the references
# ---------------------------------------------------------------------------

def test_a_host_that_cannot_name_its_window_still_gets_the_new_one():
    old, new = object(), object()

    class Host:
        def window(self):
            raise RuntimeError("Internal C++ object already deleted.")

    host = Host()
    host._ambient = old
    assert A._point_holders_at(old, new, host) == 1
    assert host._ambient is new


def test_a_holder_without_attributes_moves_nothing():
    old, new = object(), object()

    class Slotted:
        __slots__ = ()

        def window(self):
            return self

    assert A._point_holders_at(old, new, Slotted()) == 0


# ---------------------------------------------------------------------------
# Retiring an old one
# ---------------------------------------------------------------------------

def test_a_backdrop_that_will_not_stop_is_still_taken_off_screen(
        spaceout, caplog):
    window, host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop

    def stuck():
        raise RuntimeError("renderer hung")

    old.shutdown = stuck
    with caplog.at_level(logging.DEBUG, logger=A.LOG.name):
        A._retire_one_fractal(old, host)
    assert old.parentWidget() is None
    assert all(f._widget is None
               for f in host.findChildren(A._FractalTracksItsHost))
    assert "could not stop an old fractal" in caplog.text


def test_a_host_whose_children_cannot_be_listed_still_lets_go(caplog):
    unparented = []

    class Old:
        def shutdown(self):
            pass

        def setParent(self, parent):  # noqa: N802 - Qt naming
            unparented.append(parent)

        def deleteLater(self):  # noqa: N802 - Qt naming
            unparented.append("freed")

    class Host:
        def findChildren(self, _kind):  # noqa: N802 - Qt naming
            raise RuntimeError("Internal C++ object already deleted.")

    with caplog.at_level(logging.DEBUG, logger=A.LOG.name):
        A._retire_one_fractal(Old(), Host())
    assert unparented == [None, "freed"]
    assert "could not drop an old fractal's filter" in caplog.text


def test_a_backdrop_already_freed_is_retired_quietly(qapp):
    class Freed:
        def shutdown(self):
            pass

        def setParent(self, parent):  # noqa: N802 - Qt naming
            raise RuntimeError("Internal C++ object already deleted.")

    host = QObject()
    A._retire_one_fractal(Freed(), host)
    assert host.findChildren(A._FractalTracksItsHost) == []


# ---------------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------------

def test_unreadable_settings_rebuild_nothing(spaceout, monkeypatch, caplog):
    window, _host = _window_with_a_backdrop(spaceout)
    running = window._dock_backdrop

    def unreadable():
        raise OSError("settings file locked")

    monkeypatch.setattr(P, "get_fractal_settings", unreadable)
    with caplog.at_level(logging.DEBUG, logger=A.LOG.name):
        assert A.rebuild_the_spaceout_backdrops() == 0
    assert window._dock_backdrop is running
    assert "could not read the fractal settings" in caplog.text


def test_a_backdrop_that_cannot_say_it_is_paused_is_rebuilt_running(
        spaceout):
    window, _host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop

    def unknown():
        raise RuntimeError("renderer gone")

    old.is_paused = unknown
    P.set_fractal_settings(pattern="cascade")
    assert A.rebuild_the_spaceout_backdrops() == 1
    new = window._dock_backdrop
    assert new is not old
    assert not new.is_paused()


def test_a_new_backdrop_that_refuses_to_pause_still_replaces_the_old(
        spaceout, monkeypatch):
    window, host = _window_with_a_backdrop(spaceout)
    old = window._dock_backdrop
    old.pause()
    real_build = A._build_the_spaceout_fractal
    built = []

    def build(values, controls=None):
        widget = real_build(values, controls)

        def refuse():
            raise RuntimeError("no GL context yet")

        widget.pause = refuse
        built.append(widget)
        return widget

    monkeypatch.setattr(A, "_build_the_spaceout_fractal", build)
    P.set_fractal_settings(pattern="cascade")
    assert A.rebuild_the_spaceout_backdrops() == 1
    assert window._dock_backdrop is built[0]
    assert built[0].parentWidget() is host
    assert old.parentWidget() is None
