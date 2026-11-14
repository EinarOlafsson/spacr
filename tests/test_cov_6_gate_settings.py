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
    GateEditorSettings, GateSettingsDialog,
)

_COLUMNS = ("cell_area", "cell_channel_1_mean_intensity",
            "nucleus_area", "prcfo")


@pytest.fixture
def dialog(qtbot):
    widget = GateSettingsDialog(GateEditorSettings(), columns=_COLUMNS)
    qtbot.addWidget(widget)
    return widget


def test_the_note_says_how_many_measurements_were_picked(dialog):
    """With the label present the guard must not swallow the real update."""
    dialog._explicit.setText("cell_area")
    dialog._on_explicit_changed()
    assert dialog._selection_note.text().strip()


