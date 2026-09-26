"""Item 288: saving a picture when the preferences, Qt or the disk say no.

`spacr.qt.widgets.picture_export` writes the picture on screen to a PNG or a
PDF from a right-click. The ordinary saves are pinned in
``test_right_click_saves_the_picture.py``; what is pinned here is every
place the save can be refused -- an unreadable preference, a writer that
raises, a dialog the user walks away from, a view that cannot carry a menu
-- and what the user is left with in each: a fallback that still writes the
right file, or nothing written and ``""`` / ``False`` said plainly.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                            # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget          # noqa: E402

import spacr.qt.preferences as preferences                   # noqa: E402
from spacr.qt.widgets import picture_export                  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def a_picture(width=64, height=40) -> QPixmap:
    """A solid pixmap of a known size."""
    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(200, 60, 30))
    return pixmap


def _raise(*_a, **_k):
    raise RuntimeError("refused")


def test_an_unreadable_dpi_preference_falls_back_to_300(monkeypatch):
    monkeypatch.setattr(preferences, "get_figure_png_dpi", _raise)
    assert picture_export.picture_dpi() == picture_export.FALLBACK_DPI == 300


def test_a_zero_dpi_preference_falls_back_to_300(monkeypatch):
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 0)
    assert picture_export.picture_dpi() == 300


def test_an_unreadable_format_preference_offers_a_png(monkeypatch):
    monkeypatch.setattr(preferences, "get_figure_format", _raise)
    assert picture_export.preferred_suffix() == ".png"
    assert picture_export.suggested_name("my  field") == "my_field.png"


def test_something_that_is_not_a_picture_is_not_saved(app, tmp_path):
    path = tmp_path / "text.png"
    assert picture_export.as_image("not a picture") is None
    assert picture_export.save_picture("not a picture", str(path)) is False
    assert not path.exists()


def test_a_negative_resolution_is_written_at_the_fallback(app, tmp_path):
    path = tmp_path / "field.png"
    assert picture_export.save_picture(a_picture(), str(path), dpi=-5)
    expected = int(round(300 / picture_export.MM_PER_INCH * 1000.0))
    assert QImage(str(path)).dotsPerMeterX() == expected


def test_a_resolution_that_cannot_be_stamped_still_writes_the_png(
        app, tmp_path, monkeypatch):
    """The header is a nicety; the pixels are the file."""

    class Unstampable(QImage):
        def setDotsPerMeterX(self, _value):
            raise RuntimeError("read-only header")

    monkeypatch.setattr(picture_export, "QImage", Unstampable)
    path = tmp_path / "field.png"
    image = a_picture(33, 21).toImage()
    assert picture_export._save_png(image, str(path), 300) is True
    written = QImage(str(path))
    assert (written.width(), written.height()) == (33, 21)


def test_a_png_the_writer_refuses_is_reported_as_not_written(
        app, tmp_path, monkeypatch):

    class Unwritable(QImage):
        def save(self, *_a, **_k):
            raise OSError("disk full")

    monkeypatch.setattr(picture_export, "QImage", Unwritable)
    path = tmp_path / "field.png"
    assert picture_export._save_png(a_picture().toImage(), str(path),
                                    300) is False
    assert not path.exists()


def test_a_pdf_writer_that_cannot_start_writes_nothing(
        app, tmp_path, monkeypatch):
    import PySide6.QtGui as QtGui

    monkeypatch.setattr(QtGui, "QPdfWriter", _raise)
    path = tmp_path / "field.pdf"
    assert picture_export.save_picture(a_picture(), str(path)) is False
    assert not path.exists()


def test_a_pdf_painter_that_will_not_close_quietly_still_leaves_the_page(
        app, tmp_path, monkeypatch, caplog):
    """The page is on disk; the complaint at closing is logged, not shown."""

    class Grumbling(QPainter):
        def end(self):
            super().end()
            raise RuntimeError("already closed")

    monkeypatch.setattr(picture_export, "QPainter", Grumbling)
    path = tmp_path / "field.pdf"
    with caplog.at_level("DEBUG", logger=picture_export.LOG.name):
        assert picture_export.save_picture(a_picture(), str(path),
                                           dpi=150) is True
    assert path.read_bytes().startswith(b"%PDF")
    assert "would not close" in caplog.text


def test_walking_away_from_the_save_dialog_gives_no_path(app, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    offered = []

    def walk_away(parent, title, start, filters):
        offered.append((title, start, filters))
        return "", ""

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(walk_away))
    assert picture_export.ask_where_to_save(None, "cell 7") == ""
    assert offered == [("Save picture", "cell_7.png",
                        picture_export.FILTERS)]


@pytest.mark.parametrize("typed, selected, expected", [
    ("/tmp/out", "PDF document (*.pdf)", "/tmp/out.pdf"),
    ("/tmp/out", "PNG image (*.png)", "/tmp/out.png"),
    ("/tmp/out.tif", "PDF document (*.pdf)", "/tmp/out.tif"),
])
def test_a_name_typed_without_a_suffix_gets_the_filters_one(
        app, monkeypatch, typed, selected, expected):
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *_a: (typed, selected)))
    assert picture_export.ask_where_to_save(None, "x") == expected


class _Action:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class _Menu:
    def __init__(self, chosen):
        self._chosen = chosen
        self.at = None

    def exec(self, where):
        self.at = where
        return self._chosen


@pytest.mark.parametrize("chosen, expected", [
    (None, ""),
    (_Action(".pdf"), ".pdf"),
    (_Action(None), ""),
])
def test_the_menus_answer_is_the_suffix_it_carries(
        app, monkeypatch, chosen, expected):
    menu = _Menu(chosen)
    built = []

    def build(view, enabled):
        built.append(enabled)
        return menu

    monkeypatch.setattr(picture_export, "build_menu", build)
    view = QWidget()
    try:
        assert picture_export.choose_format(view, QPoint(4, 5),
                                            True) == expected
        assert built == [True]
        assert menu.at == view.mapToGlobal(QPoint(4, 5))
    finally:
        view.deleteLater()


def test_save_as_writes_nothing_when_no_path_was_chosen(
        app, tmp_path, monkeypatch):
    monkeypatch.setattr(picture_export, "ask_where_to_save",
                        lambda *_a: "")
    assert picture_export.save_as(None, a_picture(), "x", ".png") == ""
    assert list(tmp_path.iterdir()) == []


def test_a_save_that_fails_warns_and_returns_nothing(
        app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    warned = []
    target = str(tmp_path / "field.png")
    monkeypatch.setattr(picture_export, "ask_where_to_save",
                        lambda *_a: target)
    monkeypatch.setattr(picture_export, "save_picture", lambda *_a: False)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda parent, title, text: warned.append((title, text))))
    assert picture_export.save_as(None, a_picture(), "x", ".png") == ""
    assert warned == [("Not saved", "field.png could not be written.")]


def test_a_guard_that_raises_does_not_take_the_menu_away(
        app, tmp_path, monkeypatch):
    """A broken ``unless`` is treated as "no objection"."""
    target = str(tmp_path / "field.png")
    monkeypatch.setattr(picture_export, "choose_format",
                        lambda *_a: ".png")
    monkeypatch.setattr(picture_export, "ask_where_to_save",
                        lambda *_a: target)
    view = QWidget()
    try:
        assert picture_export.install_picture_save(
            view, a_picture, stem=lambda: "live", unless=_raise) is True
        assert view._spacr_picture_menu(QPoint(1, 1)) == target
        assert os.path.isfile(target)
    finally:
        view.deleteLater()


def test_a_view_that_cannot_carry_a_menu_is_not_given_one():
    class NotAView:
        def setContextMenuPolicy(self, _policy):
            raise AttributeError("no context menu here")

    view = NotAView()
    assert picture_export.install_picture_save(view, a_picture) is False
    assert not hasattr(view, "_spacr_picture_menu")
