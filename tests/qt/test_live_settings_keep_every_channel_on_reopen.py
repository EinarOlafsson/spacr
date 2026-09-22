"""Live settings offers the pathogen channel every time it opens.

Reported by the maintainer on 2026-09-19: "in mask generation live settings,
with per object settings on and pathogen chosen i dont have the option to
choose pathogen channel in the live settings, only in the per object
settings."

Measured before the fix, on a built Mask screen: the first open showed a
spin box beside "Pathogen channel"; the second open showed the caption over
an empty field, for pathogen and organelle alike. The dialog hid every
control it borrowed when it closed, and re-showed only the ones its list
named -- which did not name those two. The existing check read the row's
CAPTION, which was there both times, so it passed throughout.

Everything here is asked of the widget the user would click.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _choose(panel, role: str) -> None:
    """Pick an object in the dialog's "Primary object" dropdown."""
    index = panel._object_box.findData(role)
    assert index >= 0, f"{role} is not offered"
    panel._object_box.setCurrentIndex(index)


def _row_of(dialog, widget):
    """The ``(form, row)`` ``widget`` sits on in ``dialog``, or ``None``."""
    from PySide6.QtWidgets import QFormLayout

    for form in dialog.findChildren(QFormLayout):
        row, _role = form.getWidgetPosition(widget)
        if row >= 0:
            return form, row
    return None


@pytest.mark.parametrize("attr", ["_pathogen_channel", "_organelle_channel"])
def test_the_channel_can_be_set_on_every_opening(qtbot, attr):
    """Open, close, open again: the spin box is on its row each time."""
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.show()
    for opening in range(3):
        panel.open_live_settings()
        dialog = panel._live_settings_dialog
        _choose(panel, "pathogen")
        spin = getattr(panel, attr)
        placed = _row_of(dialog, spin)
        assert placed is not None, f"{attr} is on no row (opening {opening})"
        form, row = placed
        assert form.isRowVisible(row)
        assert spin.isVisible(), (
            f"opening {opening + 1}: the row shows its caption and no "
            f"control -- {attr} was left hidden by the last close")
        dialog.close()
        qtbot.waitUntil(lambda: panel._live_settings_dialog is None
                        or not dialog.isVisible())


def test_no_control_on_a_row_is_left_hidden_after_a_reopen(qtbot):
    """The class, not the two instances: every row holds a live control."""
    from PySide6.QtWidgets import QFormLayout

    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.show()
    panel.open_live_settings()
    panel._live_settings_dialog.close()
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    stranded = []
    for form in dialog.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            field = form.itemAt(row, QFormLayout.FieldRole)
            widget = field.widget() if field is not None else None
            if widget is not None and widget.isHidden():
                label = form.itemAt(row, QFormLayout.LabelRole)
                stranded.append(label.widget().text()
                                if label is not None and label.widget()
                                else repr(widget))
    dialog.close()
    assert stranded == []


@pytest.fixture
def grid_on():
    """The per-object table on, as the maintainer had it; restored after."""
    from spacr.qt import preferences as prefs

    was = prefs.get_object_grid_enabled()
    prefs.set_object_grid_enabled(True)
    yield prefs
    prefs.set_object_grid_enabled(was)


def test_a_pathogen_channel_set_in_live_settings_reaches_the_run(
        qtbot, grid_on):
    """With the per-object table on, pathogen chosen and Propagate on.

    The value the user types into Live settings is what ``collect()`` hands
    the run, and the per-object table shows it too.
    """
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    QApplication.processEvents()
    assert getattr(screen, "_object_grid", None) is not None
    panel = screen._live_preview
    screen._on_preview_switch(True)
    panel.open_live_settings()
    panel._live_settings_dialog.close()
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    _choose(panel, "pathogen")
    dialog._propagate_btn.setChecked(True)

    assert panel._pathogen_channel.isVisible()
    assert panel._pathogen_channel.isEnabled()
    panel._pathogen_channel.setValue(3)
    QApplication.processEvents()

    assert screen._settings_model.collect()["pathogen_channel"] == 3
    shown = screen._object_grid.settings().get("pathogen_channel")
    assert str(shown) == "3", f"the per-object table shows {shown!r}"
    dialog.close()
