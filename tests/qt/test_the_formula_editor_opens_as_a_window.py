"""The computed-column editor in a window of its own.

:class:`~spacr.qt.widgets.formula_editor.FormulaDialog` exists for screens
with no room to embed :class:`~spacr.qt.widgets.formula_editor.FormulaPanel`
inline. Three things have to hold for it to be usable at all:

* it is **not modal** -- the whole point is that the chart behind it redraws
  as columns are added, which a modal dialog would make impossible to watch;
* it **hosts the panel it was handed** rather than building a second one, so
  a screen that already owns a panel keeps its formulas when the window
  opens;
* hover help lands on the **name** of a setting, not on the box the user is
  about to type into -- and the dialog runs that pass over everything inside
  it, including a widget its embedded panel never saw.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget,
)

from spacr.qt.widgets.formula_editor import FormulaDialog, FormulaPanel  # noqa: E402


class _PanelWithAnExtraRow(FormulaPanel):
    """A panel carrying one hand-built row whose help is still on the field.

    ``FormulaPanel`` retargets its own tooltips when it is constructed, so a
    row added afterwards is the only way to show that the *dialog* runs the
    pass too rather than relying on the panel having already done it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.extra_label = QLabel("Rounding", self)
        self.extra_field = QLineEdit(self)
        self.extra_field.setToolTip("Decimal places kept in the new column.")
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.addWidget(self.extra_label)
        layout.addWidget(self.extra_field)
        self.layout().addWidget(row)


def _close_button(dialog: FormulaDialog) -> QPushButton:
    for button in dialog.findChildren(QPushButton):
        if button.text() == "Close":
            return button
    raise AssertionError("the dialog has no Close button")


def test_the_window_is_not_modal_so_the_chart_behind_it_stays_live(qtbot):
    dialog = FormulaDialog()
    qtbot.addWidget(dialog)

    assert isinstance(dialog, QDialog)
    assert dialog.isModal() is False
    assert dialog.windowTitle() == "Computed columns"
    assert dialog.objectName() == "FormulaDialog"


def test_the_window_builds_its_own_panel_when_none_is_handed_to_it(qtbot):
    dialog = FormulaDialog()
    qtbot.addWidget(dialog)

    assert isinstance(dialog.panel, FormulaPanel)
    assert dialog.panel.parent() is dialog


def test_a_panel_handed_in_is_the_one_the_window_shows(qtbot):
    panel = FormulaPanel()
    dialog = FormulaDialog(panel=panel)
    qtbot.addWidget(dialog)

    assert dialog.panel is panel
    assert panel.parent() is dialog
    # The panel is really in the layout, not merely reparented.
    assert dialog.layout().indexOf(panel) >= 0


def test_the_close_button_accepts_the_dialog(qtbot):
    dialog = FormulaDialog()
    qtbot.addWidget(dialog)

    _close_button(dialog).click()

    assert dialog.result() == QDialog.Accepted
    assert dialog.isVisible() is False


def test_hover_help_moves_off_the_field_and_onto_its_label(qtbot):
    panel = _PanelWithAnExtraRow()
    assert panel.extra_field.toolTip(), "the row starts with help on the field"
    assert panel.extra_label.toolTip() == ""

    dialog = FormulaDialog(panel=panel)
    qtbot.addWidget(dialog)

    assert panel.extra_field.toolTip() == ""
    assert panel.extra_label.toolTip() == (
        "Decimal places kept in the new column.")


def test_the_panel_block_takes_its_colours_from_the_running_theme():
    """The status line is not a fixed red: it follows the theme's palette."""
    from spacr.qt import theme as qt_theme

    light = qt_theme.stylesheet(theme="light")
    dark = qt_theme.stylesheet(theme="dark")

    for name, sheet in (("light", light), ("dark", dark)):
        palette = qt_theme.palette_for(name)
        assert (f'QLabel#FormulaStatus[state="error"] {{ '
                f'color: {palette["error"]}; }}') in sheet
        assert (f'QLabel#FormulaStatus[state="ok"] {{ '
                f'color: {palette["success"]}; }}') in sheet
        assert "QListWidget#FormulaList" in sheet
