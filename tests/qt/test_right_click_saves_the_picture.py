"""Item 505: right-click a picture in the live view and save it.

    "also here the user should be able to right click on the images to save
     them as png or pdf" -- the maintainer, 2026-09-24

What is saved is THE PICTURE, not a grab of the widget: the user may be
zoomed into a corner of a 2048-pixel field inside a 300-pixel panel, and the
thing they want in a figure is the field. So the tests below check the size
of what lands on disk, not merely that a file appeared.

The PDF is checked as a PDF -- its MediaBox, in points -- because "save as
PDF" that writes a raster with a .pdf on the end satisfies the file name and
nothing else.
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                            # noqa: E402
from PySide6.QtGui import QColor, QImage, QPixmap            # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402

from spacr.qt.widgets import picture_export                  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def a_picture(width=320, height=200) -> QPixmap:
    """A solid pixmap of a known size, as a view would be holding."""
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(40, 90, 160))
    return pixmap


def test_a_png_keeps_the_pictures_own_size(app, tmp_path):
    """Not the size of the panel it happened to be shown in."""
    path = tmp_path / "field.png"
    assert picture_export.save_picture(a_picture(640, 401), str(path)) is True

    written = QImage(str(path))
    assert (written.width(), written.height()) == (640, 401)


def test_a_png_carries_the_resolution_it_was_saved_at(app, tmp_path):
    """A PNG with no resolution in it is imported at 72 DPI by everything,
    and the figure then arrives four times too big."""
    path = tmp_path / "field.png"
    picture_export.save_picture(a_picture(), str(path), dpi=300)

    written = QImage(str(path))
    expected = int(round(300 / picture_export.MM_PER_INCH * 1000.0))
    assert written.dotsPerMeterX() == expected
    assert written.dotsPerMeterY() == expected


def test_a_pdf_is_a_page_the_pictures_own_shape(app, tmp_path):
    """A tall field is a tall page: nothing stretched, nothing letterboxed."""
    path = tmp_path / "field.pdf"
    assert picture_export.save_picture(a_picture(400, 800), str(path),
                                       dpi=100) is True

    raw = path.read_bytes()
    assert raw.startswith(b"%PDF")
    box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
                    rb"([\d.]+)", raw)
    assert box is not None, "the PDF has no page box"
    width = float(box.group(3)) - float(box.group(1))
    height = float(box.group(4)) - float(box.group(2))
    assert width > 0 and height > 0
    assert height / width == pytest.approx(2.0, rel=0.02)


def test_an_empty_view_saves_nothing_rather_than_an_empty_file(app, tmp_path):
    """There is nothing to write before a run has drawn anything."""
    path = tmp_path / "nothing.png"
    assert picture_export.save_picture(None, str(path)) is False
    assert picture_export.save_picture(QPixmap(), str(path)) is False
    assert not path.exists()


def test_the_offered_name_follows_the_figures_preference(app):
    """A user who has already said "PDF" is not asked again in other words."""
    from spacr.qt.preferences import get_figure_format, set_figure_format

    before = get_figure_format()
    try:
        set_figure_format("pdf")
        assert picture_export.preferred_suffix() == ".pdf"
        assert picture_export.suggested_name("cell probability") == \
            "cell_probability.pdf"
        set_figure_format("png")
        assert picture_export.preferred_suffix() == ".png"
        assert picture_export.suggested_name("overlay") == "overlay.png"
    finally:
        set_figure_format(before)


def test_the_live_view_hands_over_the_full_resolution_picture(app):
    """The view is 120 px tall; the field it is showing is 512."""
    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        assert view.picture() is None, "an empty view has no picture"
        view.resize(120, 120)
        view.set_pixmap(a_picture(512, 512))
        picture = view.picture()
        assert picture is not None
        assert (picture.width(), picture.height()) == (512, 512)
    finally:
        view.deleteLater()


def test_the_menu_offers_both_formats_and_says_when_there_is_nothing(app):
    """Greyed and explained, not missing: an empty menu reads as a bug."""
    full = picture_export.build_menu(None, True)
    assert [action.text() for action in full.actions()] == \
        ["Save as PNG…", "Save as PDF…"]
    assert [action.data() for action in full.actions()] == [".png", ".pdf"]
    assert all(action.isEnabled() for action in full.actions())

    empty = picture_export.build_menu(None, False)
    texts = [action.text() for action in empty.actions() if action.text()]
    assert "There is no picture here yet" in texts
    assert not any(action.isEnabled() for action in empty.actions()
                   if action.text())


def test_the_live_views_menu_writes_the_file_it_offered(app, tmp_path,
                                                        monkeypatch):
    """The whole gesture, driven: right-click, pick PNG, choose a path.

    `choose_format` is substituted rather than `QMenu.exec`. The modal call
    spins an event loop of its own, and a Shiboken method cannot reliably
    be replaced from Python -- patching it looked like it worked and the
    real menu opened anyway, which hung the run instead of failing it.
    """
    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        view.set_pixmap(a_picture(256, 128))
        view.set_picture_name("flows")

        chosen = tmp_path / "flows.png"
        asked = {}

        def _choose(widget, point, enabled):
            asked["enabled"] = enabled
            return ".png"

        def _where(parent, stem):
            asked["stem"] = stem
            return str(chosen)

        monkeypatch.setattr(picture_export, "choose_format", _choose)
        monkeypatch.setattr(picture_export, "ask_where_to_save", _where)
        view._spacr_picture_menu(QPoint(4, 4))

        assert asked["enabled"] is True, "there is a picture to save"
        assert asked["stem"] == "flows", (
            "the name offered must say which picture this is")
        assert chosen.is_file()
        assert (QImage(str(chosen)).width(),
                QImage(str(chosen)).height()) == (256, 128)
    finally:
        view.deleteLater()


def test_a_dropped_suffix_still_writes_the_format_that_was_picked(
        app, tmp_path, monkeypatch):
    """"Save as PDF" then a name with no dot must not write a PNG."""
    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        view.set_pixmap(a_picture(200, 100))
        bare = tmp_path / "picture"
        monkeypatch.setattr(picture_export, "choose_format",
                            lambda *args: ".pdf")
        monkeypatch.setattr(picture_export, "ask_where_to_save",
                            lambda *args: str(bare))
        view._spacr_picture_menu(QPoint(4, 4))

        assert (tmp_path / "picture.pdf").is_file()
        assert (tmp_path / "picture.pdf").read_bytes().startswith(b"%PDF")
    finally:
        view.deleteLater()


def test_an_empty_view_greys_the_menu_rather_than_writing_nothing(
        app, monkeypatch):
    """Right-clicking before a run has drawn anything offers a grey menu."""
    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        asked = {}

        def _choose(widget, point, enabled):
            asked["enabled"] = enabled
            return ".png"

        monkeypatch.setattr(picture_export, "choose_format", _choose)
        monkeypatch.setattr(picture_export, "ask_where_to_save",
                            lambda *args: pytest.fail(
                                "asked for a path with no picture"))
        assert view._spacr_picture_menu(QPoint(4, 4)) == ""
        assert asked["enabled"] is False
    finally:
        view.deleteLater()


def test_the_ruler_keeps_the_right_button_while_it_is_out(app, monkeypatch):
    """Right-click clears the ruler; a menu there would take a tool away."""
    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        view.set_pixmap(a_picture())
        opened = []

        def _choose(widget, point, enabled):
            opened.append(True)
            return ""

        monkeypatch.setattr(picture_export, "choose_format", _choose)

        view.ruler.set_active(True)
        view._spacr_picture_menu(QPoint(4, 4))
        assert not opened, "the menu appeared over the ruler's right-click"

        view.ruler.set_active(False)
        view._spacr_picture_menu(QPoint(4, 4))
        assert opened, "with the ruler away the menu is the right-click"
    finally:
        view.deleteLater()


def test_the_menu_is_a_right_click_on_the_view_itself(app):
    """Installed as a policy, so Qt raises it rather than each caller."""
    from PySide6.QtCore import Qt

    from spacr.qt.widgets.live_preview import _ZoomView

    view = _ZoomView()
    try:
        assert view.contextMenuPolicy() == \
            Qt.ContextMenuPolicy.CustomContextMenu
    finally:
        view.deleteLater()


def test_saving_is_refused_quietly_when_the_path_cannot_be_written(app,
                                                                   tmp_path):
    """A folder that does not exist is a failure, not a crash."""
    missing = os.path.join(str(tmp_path), "no-such-folder", "x.png")
    assert picture_export.save_picture(a_picture(), missing) is False
