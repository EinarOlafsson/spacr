"""Settings-panel helpers on the inputs a real settings file actually holds.

A settings CSV is written by hand, by an older version of spaCR, and by a run
that was interrupted, so every one of these helpers is handed something it
cannot use sooner or later: a channel holding a dye name instead of a plane, a
diameter someone typed "large" into, an organelle type that no longer exists,
a control whose C++ object went away while the panel was reading it.

None of them may raise on the GUI thread and none of them may guess. What each
one does instead is what is pinned here.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shiboken6                                                 # noqa: E402
from PySide6.QtCore import QEvent                                # noqa: E402
from PySide6.QtWidgets import (                                  # noqa: E402
    QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget,
)

from spacr.qt.screens import settings_model as sm                # noqa: E402
from spacr.qt.widgets.section import Section                     # noqa: E402

pytestmark = pytest.mark.qt


# -- a control that cannot announce an edit ----------------------------------

class TestConnectingToAnEdit:

    def test_a_field_with_a_change_signal_reports_that_it_was_connected(
            self, qtbot):
        """The handler is really wired, and the caller is told so."""
        field = QLineEdit()
        qtbot.addWidget(field)
        heard = []

        assert sm._connect_value_changed(field, heard.append) is True
        field.setText("dapi")
        assert heard == ["dapi"]

    def test_a_control_with_no_change_signal_says_it_could_not_be_wired(
            self, qtbot):
        """A rule gated on this control would go stale in silence otherwise.

        The caller uses the answer to decide whether the rule has to be
        re-read from somewhere else; a bare ``None`` return would leave a
        greying rule that only updates when an unrelated field moves.
        """
        mute = QWidget()
        qtbot.addWidget(mute)

        assert sm._connect_value_changed(mute, lambda *_: None) is False


# -- "does this setting name a plane of the stack?" ---------------------------

class TestNamingAPlane:

    @pytest.mark.parametrize("value", ["2", 2, "1.5", 0])
    def test_a_number_names_a_plane(self, value):
        assert sm._names_a_plane(value) is True

    @pytest.mark.parametrize("value", ["dapi", "chan_1", "", "  ", "none",
                                       None, False, True])
    def test_anything_that_is_not_a_number_does_not(self, value):
        """A channel holding a dye name is not channel zero.

        ``int("dapi")`` is an exception and ``int(False)`` is 0 -- the second
        is the dangerous one, because plane zero is a real plane and reading
        "no" as it would switch an object into the run.
        """
        assert sm._names_a_plane(value) is False


# -- which morphology a slot is in -------------------------------------------

class TestTheMorphologyOfASlot:

    def test_a_type_the_table_never_heard_of_falls_back_to_the_slot(self):
        """A retired or mistyped organelle type is not a reason to guess.

        ``resolve_type`` refuses an unknown name rather than reading it as
        'custom', so the panel has to be the one to fall back -- to the
        slot's own morphology, which is also what a settings file written
        before the types existed carries.
        """
        answer = sm.organelle_morphology_now("organelle", {
            "organelle_type": "cytoskeleton",
            "organelle_morphology": "ring",
        })

        assert answer == "ring"

    def test_a_diameter_that_is_not_a_number_leaves_the_type_to_decide(self):
        """"large" in the diameter box must not lose the type's answer."""
        answer = sm.organelle_morphology_now("organelle", {
            "organelle_type": "vesicular",
            "organelle_diameter": "large",
        })

        assert answer == "spots"

    def test_an_unknown_type_and_an_unknown_morphology_narrow_nothing(self):
        """Neither field names anything: None, rather than a guess."""
        assert sm.organelle_morphology_now("organelle", {
            "organelle_type": "cytoskeleton",
            "organelle_morphology": "blobby",
        }) is None


# -- has this heading got anything left under it? ----------------------------

class TestWhetherAHeadingStillShowsSomething:

    def test_something_that_is_not_a_settings_card_is_never_judged_empty(
            self, qtbot):
        """A prose panel or a foreign widget was never carrying rows to lose."""
        stranger = QWidget()
        qtbot.addWidget(stranger)

        assert sm.section_shows_anything(stranger) is True

    def test_a_row_holding_no_widget_is_not_evidence_of_anything_on_screen(
            self, qtbot):
        """A form row whose field side is a layout has nothing to count.

        The walk asks each row for its field widget; a row that has none
        cannot say whether it is showing a control, so it is skipped rather
        than counted as one.
        """
        section = Section("Cell")
        qtbot.addWidget(section)
        section.add_row("Diameter", QLineEdit())
        section._form.addRow(QLabel("composite"), QHBoxLayout())
        section._form.setRowVisible(0, False)

        assert sm.section_shows_anything(section) is False

    def test_a_heading_whose_own_rows_are_hidden_survives_on_its_children(
            self, qtbot):
        """The sub-heading is the content; hiding the parent's rows is not.

        "Advanced settings" owns almost no rows of its own -- the families
        underneath it own them -- so judging it on its own rows alone would
        hide the umbrella and every object sub-heading with it.
        """
        parent = Section("Advanced settings")
        qtbot.addWidget(parent)
        parent.add_row("Diameter", QLineEdit())
        child = Section("Cell")
        child.add_row("Minimum size", QLineEdit())
        parent.add_prose(child)
        for row in range(parent._form.rowCount()):
            parent._form.setRowVisible(row, False)

        assert sm.section_shows_anything(child) is True
        assert sm.section_shows_anything(parent) is True

    def test_a_heading_with_every_row_hidden_and_no_children_is_empty(
            self, qtbot):
        """The case the rule exists for: a heading that opens onto nothing."""
        section = Section("Pathogen")
        qtbot.addWidget(section)
        section.add_row("Diameter", QLineEdit())
        section._form.setRowVisible(0, False)

        assert sm.section_shows_anything(section) is False


# -- the older (title, rows) pair still gets its help -------------------------

def test_a_plain_title_rows_pair_is_looked_up_by_its_title():
    """``build_sections`` answers pairs; a caller may still hold one.

    The tree object carries its own title, a bare pair does not, and the
    lookup has to find the same written blurb either way or the same heading
    would have help in one panel and none in another.
    """
    assert sm.section_tooltip_is_curated("mask", ("Paths", [])) is True
    assert sm.section_tooltip_is_curated(
        "mask", ("Nothing Is Written About This", [])) is False


# -- the backend field after its combo has gone ------------------------------

def test_an_event_reaching_a_backend_field_with_no_combo_is_handed_on(qtbot):
    """The filter is installed on widgets that outlive the field's parts.

    A filter that assumed its combo was still there would raise inside Qt's
    event delivery -- on the GUI thread, from a teardown the user cannot see.
    """
    field = sm._RegressionBackendField(regression_type="ols")
    qtbot.addWidget(field)
    del field.combo

    handled = field.eventFilter(field, QEvent(QEvent.Type.ToolTip))

    assert handled is False


# -- reading the panel's own controls for the visibility rule -----------------

class TestReadingTheControlsTheVisibilityRuleNeeds:

    def test_a_control_that_has_been_destroyed_falls_back_to_the_default(
            self, qtbot):
        """The rule runs on every keystroke, including during teardown.

        A widget whose C++ object has gone raises on the first read. The
        value the run will actually use is the default, so that is what the
        rule is given rather than an exception on the GUI thread.
        """
        panel = sm.SettingsWidgets("mask")
        panel.build_sections()
        key = next(name for name in panel._object_visibility_keys()
                   if name in panel._widgets)
        shiboken6.delete(panel._widgets[key])

        state = panel._object_visibility_settings()

        assert key in state
        assert state[key] == panel._defaults.get(key)

    def test_a_key_this_app_does_not_have_is_not_an_error(self, qtbot):
        """The rule is shared by every screen; not every screen has the key."""
        panel = sm.SettingsWidgets("mask")
        panel.build_sections()

        panel._set_row_visible("no_such_setting_anywhere", False)

    def test_a_field_with_no_row_yet_is_hidden_along_with_its_name(self,
                                                                   qtbot):
        """Hiding the field alone strands its label on an empty row.

        The screen builds the label and the form after ``build_sections``
        hands the rows back, so between those two moments there is a widget
        and a label but no row to hide.
        """
        panel = sm.SettingsWidgets("mask")
        host = QWidget()
        qtbot.addWidget(host)
        QVBoxLayout(host)
        field = QLineEdit(host)
        label = QLabel("Cell channel", host)
        field._spacr_setting_label = label
        panel._widgets["cell_channel"] = field

        panel._set_row_visible("cell_channel", False)

        assert field.isVisibleTo(host) is False
        assert label.isVisibleTo(host) is False
