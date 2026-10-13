"""Categories that wait to be opened: when they build, and when they refuse.

Pins the bookkeeping of a settings category whose rows are left until it
is opened:

* the Essentials lookup falls back to "nothing is essential" when the
  settings search cannot be asked;
* a spec row the panel does not own holds nothing, and a control the model
  cannot build is left out of the opened category;
* a waiting heading is still recorded, with its keys, on a model that
  cannot remember section rows;
* a heading whose C++ half is gone is not waiting;
* opening, or taking a step of, a heading that has nothing to build, or
  that is already being built, does nothing -- and a build whose heading
  goes away mid-way is abandoned rather than resumed.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLineEdit, QWidget  # noqa: E402

from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.screens.settings_model import (  # noqa: E402
    SettingsSection,
    _ControlToCome,
)
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    widget = AppScreen("regression")
    try:
        yield widget
    finally:
        retire_pyqtgraph_menus(widget)
        widget.close()
        widget.deleteLater()


@pytest.fixture
def swap_model(screen):
    real = screen._settings_model

    def swap(model):
        screen._settings_model = model
        screen._widget_key_stamp = None
        return model

    yield swap
    screen._settings_model = real
    screen._widget_key_stamp = None


def test_an_unaskable_search_means_no_essentials(screen, monkeypatch):
    def refuse(_app_key):
        raise LookupError("no disclosure recorded")

    monkeypatch.setattr("spacr.qt.settings_search.disclosure_for", refuse)
    screen._essentials_it_opens = None
    assert screen._settings_the_essentials_view_opens() == set()
    assert screen._essentials_it_opens == set()


def test_a_row_the_panel_does_not_own_holds_nothing(screen, qtbot):
    stray = QWidget()
    qtbot.addWidget(stray)
    assert screen._spec_holds_anything(("Stray", [("Stray", stray)])) is False


def test_a_control_the_model_cannot_build_is_left_out(screen, swap_model,
                                                      qtbot):
    kept = QLineEdit()
    qtbot.addWidget(kept)
    asked = []

    class _Widgets:
        def build(self, keys):
            asked.extend(keys)

        def built(self, key):
            return None

    swap_model(types.SimpleNamespace(_widgets=_Widgets()))
    spec = SettingsSection("Cat", [("Ghost", _ControlToCome("ghost")),
                                   ("Kept", kept)])
    real = screen._with_the_controls(spec)
    assert asked == ["ghost"]
    assert [label for label, _w in real[1]] == ["Kept"]
    assert real[1][0][1] is kept


def test_a_waiting_heading_is_recorded_without_row_memory(screen,
                                                          swap_model, qtbot):
    stray = QWidget()
    qtbot.addWidget(stray)
    swap_model(types.SimpleNamespace(_widgets={}))
    spec = SettingsSection("Later", [("Ghost", _ControlToCome("ghost")),
                                     ("Stray", stray)])
    before = len(screen._settings_sections)
    section = screen._build_a_waiting_heading(spec)
    try:
        assert section.property("settingsCategorySource") == "Later"
        assert screen._waiting_heading_of["ghost"] is section
        assert screen._heading_is_waiting(section) is True
        assert len(screen._settings_sections) == before + 1
    finally:
        screen._settings_sections.remove(section)
        screen._waiting_heading_of.pop("ghost", None)
        section.deleteLater()


def test_a_heading_whose_c_half_is_gone_is_not_waiting(screen):
    class _Gone:
        @property
        def _spacr_waiting_spec(self):
            raise RuntimeError("Internal C++ object already deleted.")

    assert screen._heading_is_waiting(_Gone()) is False


class _Heading:
    """What the opening code reads from a heading, and nothing else."""

    def __init__(self, spec=None):
        self._spacr_waiting_spec = spec

    def property(self, _name):                              # noqa: A003
        return "Cat"


def test_a_built_heading_does_not_open_again(screen):
    heading = _Heading()
    assert screen._open_a_waiting_heading(heading) is False
    assert screen._run_a_step_of(heading) is False
    assert "_spacr_opening" not in heading.__dict__


def test_a_heading_already_opening_is_not_reentered(screen):
    steps = []

    def gen():
        steps.append("ran")
        yield

    heading = _Heading(spec=object())
    heading._spacr_opening = gen()
    heading._spacr_opening_now = True
    assert screen._open_a_waiting_heading(heading) is False
    assert screen._run_a_step_of(heading) is True
    assert steps == []


def test_a_heading_that_goes_away_mid_build_is_abandoned(screen):
    def gen():
        raise RuntimeError("Internal C++ object already deleted.")
        yield

    heading = _Heading(spec=object())
    heading._spacr_opening = gen()
    assert screen._run_a_step_of(heading) is False
    assert "_spacr_opening" not in heading.__dict__
    assert heading._spacr_opening_now is False


# --------------------------------------------------------------------------
# the settings search strip, asked after rows arrive


class _Strip:
    def __init__(self, apply_error=None, hides_error=None):
        self.apply_error = apply_error
        self.hides_error = hides_error
        self.applied = []
        self.indexed = 0

    def apply(self, reopen=True):
        self.applied.append(reopen)
        if self.apply_error is not None:
            raise self.apply_error("strip refused")

    def keys_it_hides(self):
        if self.hides_error is not None:
            raise self.hides_error("strip refused")
        return {"hidden_by_strip"}

    def _build_index(self):
        self.indexed += 1


@pytest.fixture
def strip(screen):
    real = screen.__dict__.get("_settings_search")

    def install(**kwargs):
        screen._settings_search = _Strip(**kwargs)
        return screen._settings_search

    yield install
    screen._settings_search = real


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
def test_a_refilter_that_fails_lets_the_next_one_run(screen, strip, error):
    bar = strip(apply_error=error)
    screen._refilter_the_settings_search()
    screen._refilter_the_settings_search()
    assert bar.applied == [False, False]
    assert screen._refiltering_settings is False


def test_a_strip_and_switches_that_cannot_be_asked_hide_only_alpha_rows(
        screen, strip, monkeypatch):
    strip(hides_error=KeyError)

    def refuse():
        raise ValueError("no dimension rows")

    monkeypatch.setattr(screen, "_dimension_rows", refuse)
    monkeypatch.setattr(screen, "_alpha_hidden_keys", lambda: {"alpha_row"})
    assert screen._rows_the_filters_hide() == {"alpha_row"}


def test_the_strip_is_asked_what_it_hides(screen, strip, monkeypatch):
    strip()
    monkeypatch.setattr(screen, "_alpha_hidden_keys", lambda: set())
    assert "hidden_by_strip" in screen._rows_the_filters_hide()


def test_a_strip_that_cannot_apply_after_reindexing_is_survived(screen,
                                                                strip):
    bar = strip(apply_error=TypeError)
    screen._the_rows_moved(judge_them=False)
    assert bar.indexed == 1
    assert bar.applied == [False]
