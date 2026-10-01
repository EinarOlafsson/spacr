"""Raising the organelle count, and the form-shaping commits, in place.

Pins how a settings screen answers a committed value that changes which
objects the run has, without rebuilding the whole screen:

* the in-place visibility pass survives a missing model, a model that
  cannot decide and a form shape that cannot be read;
* the organelle count falls back to a whole-form rebuild when growing the
  slots in place raises, answers "done" while a rebuild is already running,
  refuses an unreadable count, falls back when a category has no heading,
  and keeps the per-object grid's seed failure to itself;
* the new slot's rows find their category (skipping hidden ones, refusing
  a shared parent with no heading, sending uncategorised rows to "Other");
* rows are inserted at the declared position of their heading, a split
  category gets its object sub-headings, and a heading with no form is
  left alone;
* a spawned switch is followed through the first signal that accepts the
  connection, and a widget with none is skipped;
* a committed ``number_of_organelles`` is followed by the organelle-count
  slot, a Cellpose 3 model chooser by the delayed visibility pass, and a
  rebuild is not started while settings are being applied.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QFormLayout, QLineEdit, QSpinBox, QWidget  # noqa: E402

from spacr.qt.screens import settings_model as sm  # noqa: E402
from spacr.qt.screens.app_screen import AppScreen  # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402
from spacr.qt.widgets.section import Section  # noqa: E402

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
    """Put a stand-in model on the screen for one test, then the real one."""
    real = screen._settings_model

    def swap(model):
        screen._settings_model = model
        return model

    yield swap
    screen._settings_model = real


class _Signal:
    def __init__(self, refuse=False):
        self.refuse = refuse
        self.slots = []

    def connect(self, slot):
        if self.refuse:
            raise TypeError("cannot connect")
        self.slots.append(slot)

    def fire(self, *args):
        for slot in self.slots:
            slot(*args)


def _raise(error=ValueError):
    def go(*_args, **_kwargs):
        raise error("refused")
    return go


# --------------------------------------------------------------------------
# the in-place visibility pass


def test_with_no_model_nothing_is_decided(screen, swap_model):
    swap_model(None)
    screen._run_has_no_object_for = {"x"}
    screen._show_the_objects_the_run_has()
    assert screen._run_has_no_object_for is None


def test_a_model_that_cannot_decide_or_be_read_is_survived(
        screen, swap_model, monkeypatch):
    shape_before = screen._form_shape_on_screen
    swap_model(types.SimpleNamespace(
        refresh_object_visibility=_raise()))
    monkeypatch.setattr(screen, "_form_shape", _raise(KeyError))
    screen._show_the_objects_the_run_has()
    assert screen._form_shape_on_screen == shape_before


# --------------------------------------------------------------------------
# following the organelle count


def test_a_failed_in_place_growth_rebuilds_the_form(screen, monkeypatch):
    rebuilt = []
    monkeypatch.setattr(screen, "_grow_the_organelle_slots_in_place",
                        _raise(RuntimeError))
    monkeypatch.setattr(screen, "_rebuild_the_form",
                        lambda *a: rebuilt.append(True))
    screen._follow_the_organelle_count(4)
    assert rebuilt == [True]


def test_a_rebuild_under_way_answers_the_count(screen):
    screen._rebuilding_the_form = True
    try:
        assert screen._grow_the_organelle_slots_in_place() is True
    finally:
        screen._rebuilding_the_form = False


class _OrganelleModel:
    def __init__(self, count="2", wanted=(), spawned=()):
        self.count = count
        self.wanted = list(wanted)
        self.spawned = list(spawned)
        self.refreshed = 0
        self.hidden = []
        self._widgets = {}

    def spawn_organelle_slots(self, count):
        return list(self.spawned)

    def _setting_value(self, key):
        return self.count

    def organelle_keys_to_spawn(self, count):
        return list(self.wanted)

    def refresh_object_visibility(self):
        self.refreshed += 1

    def hide_the_rows_the_grid_speaks_for(self, keys):
        self.hidden.append(tuple(keys))

    def collect(self):
        return {}


@pytest.fixture
def growable(screen, swap_model, monkeypatch):
    monkeypatch.setattr(screen, "_worker_thread_is_running", lambda: False)
    screen._deferred_form_values = None
    screen._settings_tabs = None

    def make(**kwargs):
        return swap_model(_OrganelleModel(**kwargs))

    return make


def test_an_unreadable_count_asks_for_a_rebuild(screen, growable):
    growable(count="many")
    assert screen._grow_the_organelle_slots_in_place() is False


def test_a_category_with_no_heading_asks_for_a_rebuild(
        screen, growable, monkeypatch):
    growable(wanted=["organelle_5_channel"])
    monkeypatch.setattr(screen, "_categories_holding", lambda wanted: None)
    assert screen._grow_the_organelle_slots_in_place() is False


def test_nothing_spawned_still_reruns_the_rule_and_a_grid_failure_is_kept(
        screen, growable, monkeypatch):
    model = growable(wanted=["organelle_5_channel"], spawned=[])
    laid = []
    monkeypatch.setattr(screen, "_categories_holding", lambda wanted: [])
    monkeypatch.setattr(screen, "_lay_out_spawned_rows",
                        lambda *a: laid.append(a))
    screen._object_grid_binding = types.SimpleNamespace(seed=_raise())
    try:
        assert screen._grow_the_organelle_slots_in_place() is True
    finally:
        screen._object_grid_binding = None
    assert laid == []
    assert model.refreshed == 1
    assert model.hidden == []


# --------------------------------------------------------------------------
# which heading the new rows land in


@pytest.fixture
def categories(monkeypatch):
    def install(cats, hidden=(), parents=None):
        monkeypatch.setattr(sm, "categories_for_app", lambda app, _all: cats)
        monkeypatch.setattr(sm, "_APP_HIDDEN_CATEGORIES",
                            {"regression": set(hidden)})
        monkeypatch.setattr(sm, "_shared_category_parents",
                            lambda: dict(parents or {}))
    return install


def test_a_hidden_category_is_passed_over_and_the_rest_goes_to_other(
        screen, categories, monkeypatch):
    categories({"Hidden": ["a"], "Shown": ["b"]}, hidden={"Hidden"})
    other = object()
    shown = object()
    monkeypatch.setattr(
        screen, "_heading_of_category",
        lambda title, keys: {"Other": other, "Shown": shown}.get(title))
    out = screen._categories_holding(["a", "b", "c"])
    assert out == [("Shown", ["b"], shown, ["b"]),
                   ("Other", ["a", "c"], other, ["a", "c"])]


def test_a_shared_parent_without_a_heading_asks_for_a_rebuild(
        screen, categories, monkeypatch):
    categories({"Organelles": ["a"]}, parents={"Organelles": "Objects"})
    monkeypatch.setattr(screen, "_heading_of_category", lambda *a: None)
    assert screen._categories_holding(["a"]) is None


def test_no_other_heading_asks_for_a_rebuild(screen, categories,
                                             monkeypatch):
    categories({"Shown": ["b"]})
    monkeypatch.setattr(screen, "_heading_of_category", lambda *a: None)
    assert screen._categories_holding(["loose"]) is None


def test_a_heading_that_goes_away_while_looked_up_is_skipped(
        screen, monkeypatch):
    title = str(screen.rendered_settings_sections()[0].property(
        "settingsCategorySource"))
    monkeypatch.setattr(screen, "_heading_is_waiting", _raise(RuntimeError))
    assert screen._heading_of_category(title, ["whatever"]) is None


def test_a_same_named_heading_without_the_keys_is_not_the_category(screen):
    for section in screen.rendered_settings_sections():
        if not section._nested_sections():
            title = str(section.property("settingsCategorySource"))
            break
    else:
        pytest.skip("every regression heading has sub-headings")
    assert screen._heading_of_category(title, ["no_such_key"]) is None


# --------------------------------------------------------------------------
# laying the new rows out


class _RowModel:
    def __init__(self, **widgets):
        self._widgets = widgets

    def _label_for(self, key):
        return key.replace("_", " ").title()


def test_a_split_category_gets_a_sub_heading_per_object(screen):
    model = screen._settings_model
    keys = [key for key in ("src", "dependent_variable")
            if key in model._widgets] or list(model._widgets)[:1]
    layout = screen._settings_layout
    before = layout.count()
    section = screen._add_a_category_heading("Brand New", keys,
                                             {"Brand New"})
    assert layout.count() == before + 1
    assert layout.indexOf(section) >= 0
    assert section._settings_top_level_order % 1 == 0.5


def test_a_heading_without_a_form_is_left_alone(screen, swap_model):
    swap_model(_RowModel(a=QLineEdit()))
    heading = types.SimpleNamespace(_form=None)
    screen._insert_waiting_rows(heading, ["a"], ["a"])
    assert "a" not in screen._rows_awaiting_layout
    assert not hasattr(heading, "_spacr_declared_rows")


def test_a_new_row_goes_above_the_declared_row_it_comes_before(
        screen, swap_model, qtbot):
    new, old = QLineEdit(), QLineEdit()
    swap_model(_RowModel(a=new, b=old))
    heading = Section("Cat")
    qtbot.addWidget(heading)
    heading.add_prose(old)
    heading._spacr_declared_rows = (("b", "B", old),)
    screen._insert_waiting_rows(heading, ["a", "b"], ["a"])
    assert [key for key, _l, _w in heading._spacr_declared_rows] == ["a", "b"]
    assert screen._form_row_holding(heading._form, new) == 0
    assert screen._form_row_holding(heading._form, old) == 1
    assert screen._rows_awaiting_layout["a"] is heading
    from spacr.qt.screens.app_screen import _RowsBuiltWhenTheyAreAskedFor

    assert isinstance(heading.__dict__["_row_widgets"],
                      _RowsBuiltWhenTheyAreAskedFor)


def test_a_first_row_in_an_empty_heading_goes_at_the_end(
        screen, swap_model, qtbot):
    new = QLineEdit()
    swap_model(_RowModel(a=new))
    heading = Section("Cat")
    qtbot.addWidget(heading)
    screen._insert_waiting_rows(heading, ["a"], ["a"])
    assert screen._form_row_holding(heading._form, new) == (
        heading._form.rowCount() - 1)
    assert heading._spacr_declared_rows == (("a", "A", new),)


def test_a_widget_on_no_row_is_not_found(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    form.addRow("x", QLineEdit())
    assert AppScreen._form_row_holding(form, QLineEdit()) == -1


# --------------------------------------------------------------------------
# following the switches a new slot brings


def test_a_spawned_switch_is_followed_through_the_signal_that_accepts(
        screen, swap_model, monkeypatch):
    commit_refused = types.SimpleNamespace(
        editingFinished=_Signal(refuse=True), valueChanged=None,
        currentIndexChanged=_Signal())
    all_refuse = types.SimpleNamespace(
        valueChanged=_Signal(refuse=True),
        currentIndexChanged=_Signal(refuse=True))
    old = types.SimpleNamespace(editingFinished=_Signal())
    swap_model(types.SimpleNamespace(_widgets={
        "commit_refused": commit_refused, "all_refuse": all_refuse,
        "old": old}))
    monkeypatch.setattr(
        screen, "_object_switches_on_this_form",
        lambda: ("gone", "commit_refused", "all_refuse", "old"))
    screen._follow_the_spawned_switches(
        ["gone", "commit_refused", "all_refuse"])
    assert commit_refused.currentIndexChanged.slots == [
        screen._show_the_objects_the_run_has]
    assert all_refuse.valueChanged.slots == []
    assert all_refuse.currentIndexChanged.slots == []
    assert old.editingFinished.slots == []


# --------------------------------------------------------------------------
# the commits that shape the form


def test_a_committed_organelle_count_is_followed(screen, swap_model,
                                                 monkeypatch, qtbot):
    count = QSpinBox()
    qtbot.addWidget(count)
    swap_model(types.SimpleNamespace(_widgets={"number_of_organelles": count}))
    followed = []
    monkeypatch.setattr(screen, "_follow_the_organelle_count",
                        lambda *a: followed.append(True))
    monkeypatch.setattr(screen, "_object_switches_on_this_form", lambda: ())
    monkeypatch.setattr(screen, "_form_shaping_keys",
                        lambda: ("number_of_organelles",))
    screen._watch_the_settings_that_decide_the_form()
    count.editingFinished.emit()
    assert followed == [True]


def test_a_cellpose3_chooser_is_followed_by_the_first_signal_that_takes(
        screen):
    chooser = types.SimpleNamespace(textChanged=_Signal(refuse=True),
                                    valueChanged=_Signal())
    screen._watch_the_cellpose3_choosers(
        types.SimpleNamespace(_widgets={"cell_model_name": chooser}))
    timer = screen._cellpose3_rows_timer
    assert not timer.isActive()
    chooser.valueChanged.fire("cellpose3:cyto3")
    assert timer.isActive()
    timer.stop()


def test_no_rebuild_while_settings_are_being_applied(screen, swap_model,
                                                     monkeypatch):
    asked = []
    swap_model(types.SimpleNamespace(_applying_settings=True))
    monkeypatch.setattr(screen, "_worker_thread_is_running",
                        lambda: asked.append(True) or False)
    screen._rebuild_the_form()
    assert asked == []
    assert not getattr(screen, "_form_rebuild_deferred", False)


def test_a_committed_shape_key_rebuilds_the_form(screen, swap_model,
                                                 monkeypatch, qtbot):
    shaper = QLineEdit()
    qtbot.addWidget(shaper)
    swap_model(types.SimpleNamespace(_widgets={"some_shape": shaper}))
    rebuilt = []
    monkeypatch.setattr(screen, "_rebuild_the_form",
                        lambda *a: rebuilt.append(True))
    monkeypatch.setattr(screen, "_object_switches_on_this_form", lambda: ())
    monkeypatch.setattr(screen, "_form_shaping_keys", lambda: ("some_shape",))
    screen._watch_the_settings_that_decide_the_form()
    shaper.editingFinished.emit()
    assert rebuilt == [True]


def test_a_chooser_with_no_signal_to_follow_is_skipped(screen):
    silent = types.SimpleNamespace()
    followed = types.SimpleNamespace(currentIndexChanged=_Signal())
    screen._watch_the_cellpose3_choosers(types.SimpleNamespace(_widgets={
        "cell_model_name": silent, "nucleus_model_name": followed}))
    assert len(followed.currentIndexChanged.slots) == 1


def test_a_switch_the_panel_does_not_hold_is_not_followed(
        screen, swap_model, monkeypatch):
    swap_model(types.SimpleNamespace(_widgets={"cell_channel": QLineEdit()}))
    monkeypatch.setattr(screen, "_form_shaping_keys",
                        lambda: ("nucleus_channel", "cell_channel"))
    assert screen._object_switches_on_this_form() == ("cell_channel",)


def test_a_layout_entry_with_none_of_the_spawned_keys_is_passed_over(
        screen, monkeypatch):
    added = []
    monkeypatch.setattr(screen, "_add_a_category_heading",
                        lambda *a: added.append(a))
    screen._run_has_no_object_for = {"x"}
    screen._lay_out_spawned_rows(["spawned"], [("T", ["k"], None, ["other"])])
    assert added == []
    assert screen._run_has_no_object_for is None


def test_a_heading_whose_rows_are_a_tuple_keeps_them(screen, swap_model,
                                                     qtbot):
    new = QLineEdit()
    swap_model(_RowModel(a=new))
    heading = Section("Cat")
    qtbot.addWidget(heading)
    heading._row_widgets = ()
    screen._insert_waiting_rows(heading, ["a"], ["a"])
    assert heading.__dict__["_row_widgets"] == ()
    assert heading._spacr_declared_rows == (("a", "A", new),)
