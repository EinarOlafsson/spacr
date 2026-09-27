"""The QC dashboard links the image-quality card to its review gallery.

Pinned behaviour of :class:`spacr.qt.screens.qc_dashboard.QCDashboardScreen`:

* an ``image_quality`` card read from a file gets a "Review image quality"
  button, and clicking it opens the ``.html`` gallery written next to that
  file;
* an ``image_quality`` card with no source file gets no such button;
* a read that produces no dashboard leaves the screen as it was, without
  cards and without claiming it read anything.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QPushButton  # noqa: E402

from spacr.qt.screens import qc_dashboard as qc_module  # noqa: E402
from spacr.qt.widgets.qc_summary import Dashboard, QCCard  # noqa: E402

pytestmark = pytest.mark.qt


def _screen(qtbot, tmp_path, reader):
    screen = qc_module.QCDashboardScreen(
        threaded=False, src=str(tmp_path), reader=reader)
    qtbot.addWidget(screen)
    return screen


def _review_buttons(screen):
    return [b for b in screen.findChildren(QPushButton)
            if b.text() == "Review image quality"]


def test_the_image_quality_card_opens_its_gallery(
        qtbot, tmp_path, monkeypatch):
    source = tmp_path / "qc" / "image_quality.csv"

    def _reader(folder):
        return Dashboard(
            root=folder, verdict="pass", headline="all good",
            cards=[QCCard(key="image_quality", title="Image quality",
                          verdict="pass", headline="sharp",
                          source=str(source))])

    opened = []
    from PySide6.QtGui import QDesktopServices
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toLocalFile()) or True)

    screen = _screen(qtbot, tmp_path, _reader)
    buttons = _review_buttons(screen)

    assert len(buttons) == 1
    qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    assert opened == [str(tmp_path / "qc" / "image_quality.html")]


def test_an_image_quality_card_without_a_file_has_no_gallery_button(
        qtbot, tmp_path):
    def _reader(folder):
        return Dashboard(
            root=folder, verdict="missing", headline="nothing yet",
            cards=[QCCard(key="image_quality", title="Image quality")])

    screen = _screen(qtbot, tmp_path, _reader)

    assert _review_buttons(screen) == []
    assert "Image quality" in screen.visible_text()


def test_a_read_that_produces_no_dashboard_draws_nothing(qtbot, tmp_path):
    screen = _screen(qtbot, tmp_path, lambda folder: None)

    assert screen.dashboard() is None
    assert screen.visible_text() == ""
    assert screen.as_text() == "No folder read yet."
    assert not screen.status_text().startswith("Read from disk")
