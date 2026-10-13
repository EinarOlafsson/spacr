"""Item 521: the backend install dialog opened from Plaque Assay.

The maintainer, 2026-09-25: "the reinstall and cancel buttons are to tall
they look square, they should be rectangles like all other buttons", and
"the bar should be bellow all text right above the buttons so it isnt
jumping up and down depending on how much text is being shown".

The dialog was parented to the Plaque preview, and a preview scaled by its
own slider re-states the application sheet at its scale on itself; the
dialog inherited that sheet, so at 150 % its buttons were 59 px tall, not
40 px. It is now parented to the window.
"""
import threading

import pytest
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QVBoxLayout

WHY = "It was installed before PDFs."


def _settle():
    for _ in range(20):
        QApplication.processEvents()


@pytest.fixture
def scaled_plaque(qtbot, qt_theme_applied):
    """Plaque Assay with its preview at 150 %, as the maintainer had it."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("analyze_plaques")
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _settle()
    panel = screen._live_preview
    panel.preview_scaler.set_scale(1.5, persist=False)
    _settle()
    assert panel.styleSheet()
    yield screen
    panel.preview_scaler.set_scale(1.0, persist=False)


def _ordinary_button_height(qtbot, parent) -> int:
    """The height of an OK button in a plain dialog of the same window."""
    dialog = QDialog(parent)
    qtbot.addWidget(dialog)
    layout = QVBoxLayout(dialog)
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    layout.addWidget(box)
    dialog.show()
    _settle()
    height = box.button(QDialogButtonBox.Ok).height()
    dialog.close()
    return height


def _dialog(qtbot, parent, **kwargs):
    from spacr.qt.widgets import model_zoo_picker as mzp

    dialog = mzp.BackendInstallDialog("papers", parent, **kwargs)
    qtbot.addWidget(dialog)
    dialog.show()
    _settle()
    return dialog


@pytest.mark.parametrize("kwargs", [{}, {"reinstall": True, "why": WHY}],
                         ids=["install", "reinstall"])
def test_the_buttons_have_the_ordinary_height_from_a_scaled_preview(
        qtbot, scaled_plaque, kwargs):
    panel = scaled_plaque._live_preview
    normal = _ordinary_button_height(qtbot, scaled_plaque)
    dialog = _dialog(qtbot, panel, **kwargs)
    for button in (dialog.start_button, dialog.cancel_button):
        assert abs(button.height() - normal) <= 2, (button.text(), button.height(), normal)
        assert button.width() > button.height(), button.text()
    assert dialog.parentWidget() is panel.window()


@pytest.mark.parametrize("kwargs", [{}, {"reinstall": True, "why": WHY}],
                         ids=["install", "reinstall"])
def test_the_bar_sits_on_the_buttons_and_stays_as_the_status_grows(
        qtbot, qt_theme_applied, kwargs):
    release = threading.Event()

    def job(progress=None, cancel=None):
        release.wait(10)

    dialog = _dialog(qtbot, None, job=job, **kwargs)
    try:
        dialog.start()
        dialog._on_progress(0, 4, "Create the environment")
        _settle()
        box = dialog.start_button.parentWidget()
        spacing = dialog.layout().spacing()
        top = dialog.progress.y()
        assert dialog.progress.isVisible()
        assert dialog.status.y() < top
        assert 0 <= box.y() - dialog.progress.geometry().bottom() <= spacing + 2
        line = dialog.status.fontMetrics().lineSpacing()
        long_text = "Install PyTorch: " + " ".join(
            ["downloading torch-2.14.0+cpu.whl"] * 5)
        dialog._on_progress(1, 4, long_text)
        _settle()
        assert dialog.status.heightForWidth(dialog.status.width()) >= 3 * line
        assert dialog.progress.y() == top
        dialog._on_progress(2, 4, "Check it loads")
        _settle()
        assert dialog.progress.y() == top
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
