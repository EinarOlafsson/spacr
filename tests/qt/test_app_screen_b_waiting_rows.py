"""Rows the object rule hides wait uncaptioned until something asks for them.

A heading built while the run has no object for a key mounts that key's field
without its caption and records it in ``_rows_awaiting_layout``; reading the
heading's rows, the coarse caption pass or the rule bringing the key back is
what captions it. ``_the_rows_moved`` then puts every heading back in declared
order and re-runs the passes that depend on rows (object rule, dimension
switches, search index, late translation), each of which may fail alone.

A heading with a waiting row is built through ``_build_settings_section`` on a
FRESH screen per case: the build records the section on the screen and in the
settings model, and a queued visibility pass would caption the row if the
event loop ran in between.
"""
from __future__ import annotations

import logging
import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QLineEdit, QWidget                 # noqa: E402

from spacr.qt.screens import app_screen as aps                   # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt


def _refusing(*_args, **_kwargs):
    raise RuntimeError("the collaborator is half-built")


class _Screens:
    """Builds regression screens and tears every one down afterwards."""

    def __init__(self):
        self.built = []

    def __call__(self):
        screen = AppScreen("regression")
        self.built.append(screen)
        return screen

    def close(self):
        for screen in self.built:
            retire_pyqtgraph_menus(screen)
            screen.close()
            screen.deleteLater()


@pytest.fixture
def screens(qtbot):
    made = _Screens()
    try:
        yield made
    finally:
        made.close()


def _a_heading_with_one_waiting_row(screen):
    """One heading holding one row the object rule has decided to hide."""
    field = QLineEdit()
    screen._settings_model._widgets["ghost_setting"] = field
    screen._widget_key_stamp = None
    screen._run_has_no_object_for = {"ghost_setting"}
    section = screen._build_settings_section(
        ("Ghost", [("Ghost setting", field)]))
    return section, field


def _rows_of(section):
    """The heading's rows -- which is what asks for the waiting ones."""
    return [widget for _label, widget in section.__dict__["_row_widgets"]]


# -- which objects the run has, and a heading built from that answer ---------

def _objects_refusing_model_hides_nothing(screen, caplog):
    screen._run_has_no_object_for = None
    screen._settings_model = types.SimpleNamespace(
        keys_whose_object_the_run_lacks=lambda: {"pathogen_channel"})
    assert screen._keys_the_run_has_no_object_for() == {"pathogen_channel"}
    screen._run_has_no_object_for = None
    screen._settings_model = types.SimpleNamespace(
        keys_whose_object_the_run_lacks=_refusing)
    assert screen._keys_the_run_has_no_object_for() == set()
    assert "could not decide which objects the run has" in caplog.text


def _objects_no_model_is_a_set(screen, caplog):
    screen._run_has_no_object_for = None
    screen._settings_model = None
    assert screen._keys_the_run_has_no_object_for() == set()


def _objects_answer_kept_for_the_build(screen, caplog):
    asked = []

    def once():
        asked.append(1)
        return {"nucleus_channel"}

    screen._run_has_no_object_for = None
    screen._settings_model = types.SimpleNamespace(
        keys_whose_object_the_run_lacks=once)
    first = screen._keys_the_run_has_no_object_for()
    assert screen._keys_the_run_has_no_object_for() is first
    assert asked == [1]


def _heading_from_an_old_model(screen, caplog):
    field = QLineEdit()
    screen._run_has_no_object_for = set()
    screen._settings_model = types.SimpleNamespace(_widgets={})
    section = screen._build_settings_section(
        ("Old model", [("A setting", field)]))
    assert section.title().lower() == "old model"
    assert _rows_of(section) == [field]


def _heading_hidden_row_waits_uncaptioned(screen, caplog):
    section, field = _a_heading_with_one_waiting_row(screen)
    assert screen._rows_awaiting_layout == {"ghost_setting": section}
    assert section._spacr_declared_rows == (
        ("ghost_setting", "Ghost setting", field),)
    assert not hasattr(field, "_spacr_setting_label")


def _heading_read_back_hands_over_caption(screen, caplog):
    section, field = _a_heading_with_one_waiting_row(screen)
    rows = section.__dict__["_row_widgets"]
    assert isinstance(rows, aps._RowsBuiltWhenTheyAreAskedFor)
    assert [widget for _label, widget in rows] == [field]
    assert screen._rows_awaiting_layout == {}


def _back_still_hidden_stays_uncaptioned(screen, caplog):
    section, field = _a_heading_with_one_waiting_row(screen)
    screen._lay_out_the_rows_that_are_back({"ghost_setting"})
    assert screen._rows_awaiting_layout == {"ghost_setting": section}
    screen._lay_out_the_rows_that_are_back(())
    assert screen._rows_awaiting_layout == {}
    assert _rows_of(section) == [field]


def _back_nothing_waiting(screen, caplog):
    screen._rows_awaiting_layout = {}
    screen._lay_out_the_rows_that_are_back(())
    assert screen._rows_awaiting_layout == {}


def _every_only_its_own_heading(screen, caplog):
    from spacr.qt.widgets.section import Section

    section, field = _a_heading_with_one_waiting_row(screen)
    screen._lay_out_every_waiting_row(Section("Elsewhere"))
    assert screen._rows_awaiting_layout == {"ghost_setting": section}
    screen._lay_out_every_waiting_row(section)
    assert screen._rows_awaiting_layout == {}
    assert _rows_of(section) == [field]
    # Asked twice: nothing is left to hand over.
    screen._lay_out_every_waiting_row(section)
    assert screen._rows_awaiting_layout == {}


def _coarse_pass_captions_everything(screen, caplog):
    _section, field = _a_heading_with_one_waiting_row(screen)
    screen._caption_every_waiting_row()
    assert screen._rows_awaiting_layout == {}
    assert field._spacr_setting_label.text() == "Ghost setting"


def _one_unknown_key_left_alone(screen, caplog):
    section, _field = _a_heading_with_one_waiting_row(screen)
    screen._lay_out_one_waiting_row("not_a_waiting_key")
    assert screen._rows_awaiting_layout == {"ghost_setting": section}
    screen._lay_out_one_waiting_row("ghost_setting")
    assert screen._rows_awaiting_layout == {}


def _one_row_no_longer_declared(screen, caplog):
    section, _field = _a_heading_with_one_waiting_row(screen)
    section._spacr_declared_rows = ()
    screen._lay_out_one_waiting_row("ghost_setting")
    assert screen._rows_awaiting_layout == {}
    assert list(section.__dict__["_row_widgets"]) == []


def _one_heading_without_form(screen, caplog):
    section, _field = _a_heading_with_one_waiting_row(screen)
    section._form = None
    screen._lay_out_one_waiting_row("ghost_setting")
    assert screen._rows_awaiting_layout == {}
    assert list(section.__dict__["_row_widgets"]) == []


def _one_field_off_the_form_captioned_at_end(screen, caplog):
    section, field = _a_heading_with_one_waiting_row(screen)
    section._form.takeRow(field)
    screen._lay_out_one_waiting_row("ghost_setting")
    assert _rows_of(section) == [field]
    assert field._spacr_setting_label.text() == "Ghost setting"


def _one_heading_destroyed_is_logged(screen, caplog):
    section, _field = _a_heading_with_one_waiting_row(screen)

    def gone(_widget):
        raise RuntimeError("Internal C++ object already deleted.")

    section._form.getWidgetPosition = gone
    screen._lay_out_one_waiting_row("ghost_setting")
    assert "no heading left to caption" in caplog.text
    assert screen._rows_awaiting_layout == {}


def _one_first_late_caption_opens_record(screen, caplog):
    section, _field = _a_heading_with_one_waiting_row(screen)
    screen._captioned_late = None
    screen._lay_out_one_waiting_row("ghost_setting")
    assert screen._captioned_late == {section}


WAITING_ROW_CASES = {
    "objects: refusing model hides nothing": _objects_refusing_model_hides_nothing,
    "objects: no model": _objects_no_model_is_a_set,
    "objects: answer kept": _objects_answer_kept_for_the_build,
    "heading: old model": _heading_from_an_old_model,
    "heading: hidden row waits": _heading_hidden_row_waits_uncaptioned,
    "heading: read back captions": _heading_read_back_hands_over_caption,
    "back: still hidden": _back_still_hidden_stays_uncaptioned,
    "back: nothing waiting": _back_nothing_waiting,
    "every: only its own heading": _every_only_its_own_heading,
    "coarse pass": _coarse_pass_captions_everything,
    "one: unknown key": _one_unknown_key_left_alone,
    "one: no longer declared": _one_row_no_longer_declared,
    "one: heading without form": _one_heading_without_form,
    "one: field off the form": _one_field_off_the_form_captioned_at_end,
    "one: heading destroyed": _one_heading_destroyed_is_logged,
    "one: first late caption": _one_first_late_caption_opens_record,
}


def test_a_waiting_row_is_captioned_once_and_only_when_asked(
        subtests, screens, caplog):
    """The object answer is taken once per build and never hides everything on
    failure; a hidden key's field is on the form but uncaptioned until its own
    heading is read, the rule releases it, or the coarse pass runs; each caption
    path leaves ``_rows_awaiting_layout`` empty afterwards, even when the row is
    no longer declared, the heading lost its form or was destroyed, and the
    first late caption is recorded for the language pass."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    completed = []
    for name, case in WAITING_ROW_CASES.items():
        caplog.clear()
        with subtests.test(case=name):
            case(screens(), caplog)
            completed.append(name)
    assert completed == list(WAITING_ROW_CASES), "a case did not complete"


# -- putting the panel's row-shaped answers back in step ---------------------

class _Heading:
    def __init__(self, rows, declared):
        self._row_widgets = list(rows)
        self._spacr_declared_rows = tuple(declared)


def _search_recorder(env):
    indexed = []
    env.mp.setattr(env.screen, "_settings_search", types.SimpleNamespace(
        _build_index=lambda: indexed.append("index"),
        apply=lambda reopen=True: indexed.append("apply")), raising=False)
    return indexed


def _moved_declared_order_restored(env):
    first, second = QLineEdit(), QLineEdit()
    holder = QWidget()
    for widget in (first, second, holder):
        env.qtbot.addWidget(widget)
    placeable = _Heading(rows=[("Second", second), ("First", first)],
                         declared=[("a", "First", first),
                                   ("b", "Second", second)])
    unplaceable = _Heading(rows=[("Second", second), ("Held", holder)],
                           declared=[("a", "First", first),
                                     ("b", "Second", second)])
    env.mp.setattr(env.screen, "_settings_sections", [placeable, unplaceable])
    env.screen._the_rows_moved(judge_them=False)
    assert placeable._row_widgets == [("First", first), ("Second", second)]
    assert unplaceable._row_widgets == [("Second", second), ("Held", holder)]


def _moved_rule_runs_only_when_judging(env):
    judged = []
    env.mp.setattr(env.screen, "_settings_model", types.SimpleNamespace(
        refresh_object_visibility=lambda: judged.append(1)))
    env.screen._the_rows_moved(judge_them=False)
    assert judged == []
    env.screen._the_rows_moved(judge_them=True)
    assert judged == [1]


def _moved_no_model(env):
    env.mp.setattr(env.screen, "_settings_model", None)
    env.screen._the_rows_moved(judge_them=True)
    assert env.screen._captioned_late == set()


def _moved_object_rule_throws(env):
    env.mp.setattr(env.screen, "_settings_model", types.SimpleNamespace(
        refresh_object_visibility=_refusing))
    indexed = _search_recorder(env)
    env.screen._the_rows_moved(judge_them=True)
    assert "could not re-decide the object rows" in env.caplog.text
    assert indexed == ["index", "apply"]


def _moved_dimension_switches_throw(env):
    env.mp.setattr(env.screen, "_apply_dimension_visibility", _refusing)
    indexed = _search_recorder(env)
    env.screen._the_rows_moved(judge_them=False)
    assert "could not re-apply the dimension switches" in env.caplog.text
    assert indexed == ["index", "apply"]


def _moved_late_caption_translated(env):
    from spacr.qt import i18n

    walked = []
    env.mp.setattr(i18n, "retranslate_widget_tree", walked.append)
    env.mp.setattr(env.screen, "_captioned_late", set(), raising=False)
    env.screen._the_rows_moved(judge_them=False)
    assert walked == []
    heading = QWidget()
    env.qtbot.addWidget(heading)
    env.screen._captioned_late = {heading}
    env.screen._the_rows_moved(judge_them=False)
    assert walked == [heading]
    assert env.screen._captioned_late == set()


def _moved_language_pass_throws(env):
    from spacr.qt import i18n

    def refusing(_widget):
        raise ValueError("no catalog for this locale")

    heading = QWidget()
    env.qtbot.addWidget(heading)
    env.mp.setattr(i18n, "retranslate_widget_tree", refusing)
    env.mp.setattr(env.screen, "_captioned_late", {heading}, raising=False)
    env.screen._the_rows_moved(judge_them=False)
    assert "could not translate a caption that arrived late" in env.caplog.text


def _moved_search_will_not_index(env):
    env.mp.setattr(env.screen, "_settings_search", types.SimpleNamespace(
        _build_index=_refusing, apply=lambda reopen=True: None), raising=False)
    env.screen._the_rows_moved(judge_them=False)
    assert "could not re-index the settings search" in env.caplog.text


def _moved_no_search_strip(env):
    env.mp.setattr(env.screen, "_settings_search", None, raising=False)
    env.screen._the_rows_moved(judge_them=False)
    assert env.screen._settings_search is None


ROWS_MOVED_CASES = {
    "declared order restored": _moved_declared_order_restored,
    "rule runs only when judging": _moved_rule_runs_only_when_judging,
    "no model": _moved_no_model,
    "object rule throws": _moved_object_rule_throws,
    "dimension switches throw": _moved_dimension_switches_throw,
    "late caption translated": _moved_late_caption_translated,
    "language pass throws": _moved_language_pass_throws,
    "search will not index": _moved_search_will_not_index,
    "no search strip": _moved_no_search_strip,
}


def test_the_rows_moved_resyncs_each_pass_independently(
        subtests, monkeypatch, qtbot, caplog, screens):
    """Headings return to declared order when every row is declared (otherwise
    keep theirs), the object rule runs only when asked to judge and only if a
    model exists, and a failing object rule, dimension pass, language pass or
    search re-index is logged without stopping the passes after it."""
    caplog.set_level(logging.DEBUG, logger=aps.LOG.name)
    screen = screens()
    completed = []
    for name, case in ROWS_MOVED_CASES.items():
        caplog.clear()
        with subtests.test(case=name), monkeypatch.context() as mp:
            mp.setattr(screen, "_settings_sections", [])
            case(types.SimpleNamespace(screen=screen, mp=mp, qtbot=qtbot,
                                       caplog=caplog))
            completed.append(name)
    assert completed == list(ROWS_MOVED_CASES), "a case did not complete"
