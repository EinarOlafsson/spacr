"""The QC overview folder picker without a screen and image-view scrolls without an overlay."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt


def test_the_overview_folder_picker_opens_at_its_own_minimum_without_a_screen(qtbot, monkeypatch):
    monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: None))
    parent = QWidget()
    qtbot.addWidget(parent)
    monkeypatch.setattr(parent, "screen", lambda: None)
    dialog = browser._overview_folders_dialog(parent, "Choose plates")
    qtbot.addWidget(dialog)
    assert dialog.windowTitle() == "Choose plates"
    assert dialog.size().width() >= dialog.minimumSizeHint().width()
    assert dialog.size().height() >= dialog.minimumSizeHint().height()
    assert dialog.minimumSize() == dialog.minimumSizeHint()


def test_the_image_view_scrolls_without_an_overlay_or_with_one_already_deleted(qtbot):
    view = browser._OverviewImageView()
    qtbot.addWidget(view)
    view.show()
    assert view._metadata_overlay is None
    view.scrollContentsBy(3, 4)                 # no overlay: nothing to keep fixed
    overlay = QLabel("metadata", view)
    overlay.move(50, 50)
    view._metadata_overlay = overlay
    view.scrollContentsBy(3, 4)
    assert (overlay.x(), overlay.y()) == (8, 8)
    shiboken6.delete(overlay)
    view.scrollContentsBy(3, 4)                 # a deleted overlay is left alone
    assert not shiboken6.isValid(overlay)
