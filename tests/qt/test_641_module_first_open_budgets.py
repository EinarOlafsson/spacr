"""Item 641, round 3: fresh module navigation and structural startup guards.

The separate ``test_641_module_first_open_timing.py`` measures the unchanged
first-open wall-clock budget in the serial timing tail. These tests retain
real per-module navigation under coverage and pin the causes by count:

* a module open reads every preference from one ``QSettings`` store;
* the cursor policy passes over ``Show`` events of widgets with no cursor
  of their own;
* ``categories_for_app`` is memoised and hands out fresh copies;
* a close mark's glyph is measured once per font;
* PySide's per-import probe reads raw bytes, not ``inspect.getsource``.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

def _keys():
    from spacr.qt.app import APPS

    return [key for key, _name, _description, _section in APPS]


@pytest.fixture(scope="module")
def window(qapp):
    from spacr.qt.app import MainWindow

    win = MainWindow()
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
def test_a_fresh_module_open_selects_its_new_screen(window, qapp, key):
    assert key not in window._screens
    window._on_nav_selected(key)
    for _ in range(3):
        qapp.processEvents()
    assert window._stack.currentWidget() is window._screens.get(key)


def test_a_module_open_reads_preferences_from_one_store(qapp, monkeypatch):
    from spacr.qt import preferences

    made = []

    class Counting(preferences.QSettings):
        def __init__(self, *args):
            made.append(args)
            super().__init__(*args)

    monkeypatch.setattr(preferences, "QSettings", Counting)
    for _ in range(50):
        preferences.get_font_scale()
        preferences.resolve_effective_theme()
    outside = len(made)
    made.clear()
    with preferences._one_store():
        for _ in range(50):
            preferences.get_font_scale()
            preferences.resolve_effective_theme()
            preferences._get_show_alpha_features()
    assert outside >= 100
    assert len(made) == 1


def test_the_window_opens_a_module_inside_one_store():
    import inspect

    from spacr.qt.app import MainWindow

    source = inspect.getsource(MainWindow._on_nav_selected)
    assert "_one_store()" in source


def test_a_show_of_a_widget_without_a_cursor_reads_no_cursor(qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.cursor_policy import _CursorPolicy

    class Probe(QWidget):
        reads = 0

        def cursor(self):
            Probe.reads += 1
            return super().cursor()

    policy = _CursorPolicy(qapp)
    plain = Probe()
    policy.eventFilter(plain, QEvent(QEvent.Show))
    assert Probe.reads == 0
    pointed = Probe()
    pointed.setCursor(Qt.PointingHandCursor)
    policy.eventFilter(pointed, QEvent(QEvent.Show))
    assert Probe.reads == 1
    policy.eventFilter(plain, QEvent(QEvent.Enter))
    assert Probe.reads == 2
    assert pointed.cursor().shape() == Qt.ArrowCursor


def test_categories_for_app_is_memoised_and_returns_fresh_copies(monkeypatch):
    from spacr.qt.screens import settings_model

    calls = []
    real = settings_model._categories_for_app_uncached

    def counting(*args):
        calls.append(args[0])
        return real(*args)

    monkeypatch.setattr(settings_model, "_categories_for_app_uncached",
                        counting)
    settings_model._CATEGORIES_FOR_APP_MEMO.clear()
    cats = settings_model.get_categories()
    first = settings_model.categories_for_app("mask", cats)
    first[next(iter(first))].append("not_a_setting")
    second = settings_model.categories_for_app("mask", cats)
    assert calls == ["mask"]
    assert "not_a_setting" not in second[next(iter(second))]
    assert first.keys() == second.keys()


def test_a_close_mark_is_measured_once_per_font(qapp, monkeypatch):
    from PySide6.QtWidgets import QToolButton

    from spacr.qt import theme

    theme._CLOSE_MARK_INK.clear()
    button = QToolButton()
    sides = {theme.close_mark_side(button) for _ in range(20)}
    assert len(sides) == 1
    assert len(theme._CLOSE_MARK_INK) == 1


def test_the_pyside_import_probe_reads_bytes(tmp_path, monkeypatch):
    import sys
    import types

    from spacr.qt import app

    feature = types.ModuleType("shibokensupport.feature")
    asked = []
    feature._mod_uses_pyside = lambda module: asked.append(module) or False
    monkeypatch.setitem(sys.modules, "shibokensupport.feature", feature)
    assert app._quicken_the_pyside_import_probe() is True
    assert app._quicken_the_pyside_import_probe() is False
    uses = tmp_path / "uses.py"
    uses.write_text("from PySide6 import QtCore\n")
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n")
    probe = feature._mod_uses_pyside
    assert probe(types.SimpleNamespace(__file__=str(uses))) is True
    assert probe(types.SimpleNamespace(__file__=str(plain))) is False
    assert asked == []
    builtin = types.SimpleNamespace(__file__=None)
    assert probe(builtin) is False
    assert asked == [builtin]
