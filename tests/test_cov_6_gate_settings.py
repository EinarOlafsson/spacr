"""Two guards in the gate-settings dialog: an absent note and an unknown mode.

The selection note and the mode dropdown are both written to from handlers
that can fire before, or after, the widget they write to exists -- a signal
connected during construction, a dialog being torn down. Each guard turns
that into a no-op instead of an ``AttributeError`` that would surface as the
settings window dying under the user's hands.

``set_mode`` is the other half of a two-view setting: the 2D/3D/xD buttons on
the editor and this dropdown show one value. A mode the dropdown does not
carry must leave both views alone rather than blanking the combo.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.gate_settings import (  # noqa: E402
    GATE_MODES, GateEditorSettings, GateSettingsDialog,
)

_COLUMNS = ("cell_area", "cell_channel_1_mean_intensity",
            "nucleus_area", "prcfo")


@pytest.fixture
def dialog(qtbot):
    widget = GateSettingsDialog(GateEditorSettings(), columns=_COLUMNS)
    qtbot.addWidget(widget)
    return widget


def test_the_selection_note_survives_the_label_not_being_there(dialog):
    """The handler runs on a torn-down dialog; it must not raise."""
    before = dialog.settings()
    dialog._selection_note = None
    assert dialog._refresh_selection_note() is None
    assert dialog._selection_note is None, "nothing was put back in its place"
    assert dialog.settings() == before, "the settings were left alone"


def test_an_absent_note_does_not_stop_the_setting_from_changing(dialog):
    """The note is a courtesy; losing it must not lose the edit behind it."""
    dialog._selection_note = None
    dialog._explicit.setText("cell_area, nucleus_area")
    dialog._on_explicit_changed()
    assert dialog.settings().reduction_columns == ("cell_area", "nucleus_area")


def test_the_note_says_how_many_measurements_were_picked(dialog):
    """With the label present the guard must not swallow the real update."""
    dialog._explicit.setText("cell_area")
    dialog._on_explicit_changed()
    assert dialog._selection_note.text().strip()


def test_a_mode_the_dropdown_does_not_carry_is_ignored(dialog):
    """Blanking the combo would show a state the editor is not in."""
    before = dialog._mode.currentText()
    dialog.set_mode("7D")
    assert dialog._mode.currentText() == before
    assert dialog.settings().gate_mode == before


def test_an_empty_mode_is_ignored_too(dialog):
    """An unset value from a caller that has not chosen yet is not a mode."""
    before = dialog._mode.currentText()
    dialog.set_mode("")
    assert dialog._mode.currentText() == before


def test_a_rejected_mode_does_not_leave_the_dialog_deaf(dialog):
    """The guard returns before ``_live`` is cleared, so it must still emit."""
    dialog.set_mode("7D")
    seen = []
    dialog.settings_changed.connect(seen.append)
    dialog._mode.setCurrentText("3D")
    assert [s.gate_mode for s in seen] == ["3D"]


@pytest.mark.parametrize("mode", GATE_MODES)
def test_every_real_mode_is_shown_without_being_echoed_back(dialog, mode):
    """Echoing a mode chosen elsewhere back out would be a signal loop."""
    seen = []
    dialog.settings_changed.connect(seen.append)
    dialog.set_mode(mode)
    assert dialog._mode.currentText() == mode
    assert seen == []


def test_the_dropdown_shows_xd_while_the_settings_record_what_it_means(dialog):
    """``xD`` is a legacy spelling of "3D with a projected Z", and stays one.

    :class:`GateEditorSettings` folds it into ``gate_mode='3D'`` plus
    ``xd_projection=True`` so nothing downstream has to know a third mode
    ever existed. The dropdown still offers the word, so the two views
    legitimately read differently here -- asserting they match would pin the
    wrong behaviour.
    """
    dialog.set_mode("xD")
    assert dialog._mode.currentText() == "xD"
    settings = dialog.settings()
    assert settings.gate_mode == "3D"
    assert settings.xd_projection is True


@pytest.mark.parametrize("mode", ["2D", "3D"])
def test_a_drawable_mode_reaches_the_settings_unchanged(dialog, mode):
    """Only the legacy spelling is rewritten; the real modes are not."""
    dialog.set_mode(mode)
    assert dialog.settings().gate_mode == mode
