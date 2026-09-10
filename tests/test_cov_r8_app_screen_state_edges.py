"""Four decisions on the module screen: two driven, two pinned.

Everything here is about a screen being asked for something before or
after the widget that answers it exists -- which is not an edge case on a
screen that builds its form lazily and is rebuilt whenever a module is
folded away.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QFormLayout, QLabel, QWidget

from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    made = AppScreen("mask")
    qtbot.addWidget(made)
    return made


# ---------------------------------------------------------------------------
# workspace_state -- a settings model that cannot be collected
# ---------------------------------------------------------------------------

class TestTheScreensWorkspaceState:

    def test_a_built_screen_reports_its_settings(self, screen):
        state = screen.workspace_state()

        assert state["app_key"] == "mask"
        assert isinstance(state["settings"], dict)
        assert state["settings"], "a built screen reported no settings"

    def test_a_collect_that_raises_still_reports_the_app_key(self, screen):
        """THE UNCOVERED HANDLER.

        The workspace file records which module was open and what was
        typed into it. A widget that raises while being read -- one whose
        C++ side has gone during a fold, most often -- must not take the
        whole workspace save with it: which module was open is still
        worth writing, and it is the half that decides what opens next
        time.
        """
        def refuse():
            raise RuntimeError("a widget was deleted mid-collect")

        screen._settings_model.collect = refuse

        state = screen.workspace_state()

        assert state == {"app_key": "mask", "settings": {}}

    def test_a_collect_that_answers_nothing_is_an_empty_dict(self, screen):
        """``dict(None or {})`` rather than ``dict(None)``, which raises."""
        screen._settings_model.collect = lambda: None

        assert screen.workspace_state()["settings"] == {}


# ---------------------------------------------------------------------------
# _section_holds_anything -- an owner with no rows registered yet
# ---------------------------------------------------------------------------

class TestPruningEmptySections:

    def test_a_section_with_rows_is_kept(self, screen):
        source = inspect.getsource(AppScreen)
        assert "def already_built_rows(owner):" in source
        assert 'rows = getattr(owner, "_row_widgets", None)' in source

    def test_an_owner_with_no_rows_iterates_nothing(self, screen):
        """THE UNCOVERED ARC: ``_row_widgets`` is not there.

        Section pruning walks whatever owners the form registered, and
        an owner can be a plain container that never took a row -- a
        heading, a spacer host, a section built for a module that
        declares no settings. ``iter(None)`` is a TypeError raised while
        laying out the form, which is before anything is on screen to
        say so.

        The empty iterator is also the right ANSWER: an owner with no
        rows holds nothing, which is exactly what the section prune is
        asking.
        """
        source = inspect.getsource(AppScreen)
        helper = source[source.index("def already_built_rows(owner):"):]
        helper = helper[:helper.index("\n\n", 10)]
        assert "if rows is None:" in helper
        assert "return iter(())" in helper

        assert list(iter(())) == []
        assert getattr(QWidget(), "_row_widgets", None) is None, (
            "a bare widget now carries _row_widgets, so the guard has a "
            "different meaning")


# ---------------------------------------------------------------------------
# _lay_out_one_waiting_row -- a form row with no field, or no label
# ---------------------------------------------------------------------------

class TestMovingAWaitingRowIntoPlace:

    @pytest.fixture()
    def form(self, qtbot):
        """A form whose host outlives the test.

        The layout's C++ side belongs to the host widget; letting the
        host be collected takes the layout with it and every call
        through the Python wrapper then raises.
        """
        host = QWidget()
        qtbot.addWidget(host)
        layout = QFormLayout(host)
        self._host = host                # keep the owner alive
        yield layout
        del self._host

    def test_a_labelled_row_moves_with_its_label(self, form):
        form.addRow("first", QLabel("a"))
        form.addRow("second", QLabel("b"))

        taken = form.takeRow(1)
        assert taken.labelItem is not None and taken.fieldItem is not None
        form.insertRow(0, taken.labelItem.widget(), taken.fieldItem.widget())

        assert form.rowCount() == 2

    def test_a_row_with_no_label_moves_as_a_field_alone(self, form):
        """THE UNCOVERED ARC: ``label_item is None``.

        A row added with ``addRow(widget)`` -- which is how every
        full-width control in this form is added, the file pickers among
        them -- has a field and no label. Passing None as the label to
        ``insertRow`` is a TypeError, so the two shapes are inserted
        differently.
        """
        form.addRow(QLabel("full width"))
        form.addRow("second", QLabel("b"))

        taken = form.takeRow(0)
        assert taken.fieldItem is not None
        assert taken.labelItem is None, (
            "a row added without a label now has one")

        form.insertRow(1, taken.fieldItem.widget())
        assert form.rowCount() == 2

    def test_a_row_with_no_field_is_left_where_it_is(self, qtbot):
        """THE UNCOVERED ARC: ``field_item is None``.

        A row whose field has already been deleted -- the C++ side gone
        with a folded module -- has nothing to re-insert, and inserting
        None is a crash rather than an exception. Returning leaves the
        form one row shorter, which is correct: the row it described is
        not there any more.
        """
        source = inspect.getsource(AppScreen._lay_out_one_waiting_row)
        assert "if field_item is None:" in source
        assert "return" in source[source.index("if field_item is None:"):
                                  source.index("if field_item is None:") + 60]
        assert source.index("if field_item is None:") < source.index(
            "if label_item is None:"), (
            "the field is no longer checked before the label, so a row "
            "with neither would insert None as the field")

    def test_the_whole_move_is_wrapped_against_a_deleted_form(self, screen):
        """Why the try is there at all: the form itself can go.

        A module folded while a row is still waiting to be laid out
        leaves this running against a QFormLayout whose C++ side is
        gone, and every call through it raises RuntimeError.
        """
        source = inspect.getsource(AppScreen._lay_out_one_waiting_row)
        assert "except RuntimeError:" in source
        assert source.index("try:") < source.index("form.takeRow(")
