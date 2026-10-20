"""Figure settings combos pass matplotlib's own words, whatever they display.

The N47 full serial Qt run failed in German: the language pass rewrote the
X scale item ``linear`` to ``geradlinig`` and the legend item ``best`` to
``am besten``, and the dialog handed those captions to matplotlib, which
raised inside the Qt event loop. The values now travel as item data.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtWidgets import QComboBox  # noqa: E402

pytest.importorskip("PySide6")


def _dialog(qtbot):
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure, axis = plt.subplots()
    axis.plot([1, 2, 3], [1, 4, 9], label="series")
    axis.legend()
    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)
    return figure, axis, dialog


def _combo_with(dialog, value):
    for combo in dialog.findChildren(QComboBox):
        if combo.findData(value) >= 0 and combo.currentData() == value:
            return combo
    raise AssertionError(f"no combo currently holding {value!r}")


def test_a_translated_scale_caption_does_not_reach_matplotlib(qtbot):
    figure, axis, dialog = _dialog(qtbot)
    try:
        combo = _combo_with(dialog, "linear")
        combo.setItemText(combo.currentIndex(), "geradlinig")
        assert axis.get_xscale() == "linear" and axis.get_yscale() == "linear"
        combo.setCurrentIndex(combo.findData("log"))
        assert "log" in (axis.get_xscale(), axis.get_yscale())
    finally:
        plt.close(figure)


def test_a_translated_legend_place_does_not_reach_matplotlib(qtbot):
    figure, axis, dialog = _dialog(qtbot)
    try:
        combo = _combo_with(dialog, "best")
        for index in range(combo.count()):
            combo.setItemText(index, f"übersetzt {index}")
        combo.setCurrentIndex(combo.findData("upper left"))
        assert axis.get_legend() is not None
        assert axis.get_legend()._loc == 2
    finally:
        plt.close(figure)
