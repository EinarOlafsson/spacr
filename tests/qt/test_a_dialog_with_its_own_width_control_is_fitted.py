"""A dialog that names a control ``width`` is still fitted to its wrapped text.

The Save figure dialog keeps its figure-width spin box as ``self.width``. Once
the application theme's show filter was installed (by any earlier test in the
same process, or by the real app), fitting the dialog called
``window.width()`` on it and raised ``TypeError: 'QDoubleSpinBox' object is
not callable`` inside the event loop, so the dialog was never fitted.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QDialog, QDoubleSpinBox, QLabel,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt import theme  # noqa: E402


class _Shadowing(QDialog):
    """A dialog whose own controls hide ``width`` and ``height``."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        label = QLabel(" ".join(["word"] * 400))
        label.setWordWrap(True)
        layout.addWidget(label)
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        layout.addWidget(self.width)
        layout.addWidget(self.height)


def test_a_dialog_with_width_and_height_controls_is_made_tall_enough(qtbot):
    dialog = _Shadowing()
    qtbot.addWidget(dialog)
    QWidget.resize(dialog, 260, 60)
    assert theme._fit_a_dialog_to_its_wrapped_text(dialog) is True
    needed = dialog.layout().totalHeightForWidth(QWidget.width(dialog))
    screen = QWidget.screen(dialog).availableGeometry().height()
    assert QWidget.height(dialog) == min(needed, int(screen * 0.9))
    assert QWidget.width(dialog) == 260


def test_the_save_figure_dialog_can_be_fitted(qtbot):
    from matplotlib.figure import Figure

    from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog

    figure = Figure()
    figure.add_subplot().plot([0, 1], [1, 0])
    dialog = SaveFigureDialog(figure)
    qtbot.addWidget(dialog)
    assert isinstance(dialog.width, QDoubleSpinBox)
    theme._fit_a_dialog_to_its_wrapped_text(dialog)
    assert QWidget.height(dialog) > 0
