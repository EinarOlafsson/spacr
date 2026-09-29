"""Item 597: two sliders under Home's right-hand column size its widgets and text.

The maintainer, 2026-09-29: "add to new list, a slider on the home screen that
lets the user change the size of the rwidgets on the right on the bottom of
those widgest should be a slider that allows the user to change the size of
the text."
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, QSettings                        # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QSlider, QWidget  # noqa: E402

from spacr.qt.widgets import home as home_mod                        # noqa: E402

APPS = [("mask", "Mask", "Segment cells", "Core"),
        ("measure", "Measure", "Measure objects", "Core")]


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def store(tmp_path, monkeypatch, qapp):
    """A throwaway preference store and the app's real stylesheet."""
    from spacr.qt import preferences as prefs
    from spacr.qt import theme

    path = tmp_path / "spacr-597.ini"
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    before = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet("dark"))
    yield prefs
    qapp.setStyleSheet(before)


def _home():
    page = home_mod.HomePage(APPS, lambda key: None)
    page.resize(1400, 900)
    page.show()
    _pump()
    return page


def _sliders(page):
    size = page.findChild(QSlider, "HomeAsideSizeSlider")
    text = page.findChild(QSlider, "HomeAsideTextSlider")
    assert size is not None and text is not None
    return size, text


def _row_label(page) -> QLabel:
    """A label the Totals panel builds with its own font-size sheet."""
    labels = [w for w in page._totals.findChildren(QLabel)
              if "font-size" in (w.styleSheet() or "") and w.text()]
    assert labels
    return labels[0]


def _px(label: QLabel) -> int:
    label.ensurePolished()
    return label.fontInfo().pixelSize()


def test_the_sliders_sit_at_the_foot_of_the_column(store):
    page = _home()
    size, text = _sliders(page)
    controls = page.findChild(QWidget, "HomeAsideScaleControls")
    scroll = page.findChild(QWidget, "HomeAsideScroll")
    assert controls.parentWidget() is page._aside
    assert controls.geometry().top() >= scroll.geometry().bottom()
    assert (size.value(), text.value()) == (100, 100)
    page.close()


def test_widget_size_changes_width_persists_and_restores(store):
    page = _home()
    size, _text = _sliders(page)
    before = page._aside.width()
    size.setValue(150)
    _pump()
    wider = page._aside.width()
    assert wider > before
    size.setValue(size.minimum())
    _pump()
    assert page._aside.width() < wider
    size.setValue(140)
    _pump()
    assert float(store._settings().value("prefs/home_aside_size")) == 1.4
    page.close()

    again = _home()
    assert _sliders(again)[0].value() == 140
    assert again._aside.width() == wider or again._aside.width() > before
    again.close()


def test_text_size_changes_text_persists_and_restores(store):
    page = _home()
    _size, text = _sliders(page)
    label = _row_label(page)
    base = _px(label)
    text.setValue(160)
    _pump()
    big = _px(label)
    assert big > base
    text.setValue(text.minimum())
    _pump()
    assert _px(label) < base
    text.setValue(130)
    _pump()
    assert float(store._settings().value("prefs/home_aside_text")) == 1.3
    page.close()

    again = _home()
    assert _sliders(again)[1].value() == 130
    assert _px(_row_label(again)) > base
    again.close()


def test_a_rebuilt_row_takes_the_size_without_compounding(store):
    page = _home()
    _size, text = _sliders(page)
    text.setValue(150)
    _pump()
    first = _px(_row_label(page))
    page._totals.refresh({})
    page._apply_aside_text()
    page._apply_aside_text()
    _pump()
    assert _px(_row_label(page)) == first
    page.close()


def _check_no_overlap(page) -> None:
    aside = page._aside
    panels = page._aside_panels
    inside = QRect(0, 0, panels.width(), panels.height())
    boxes = [w for w in (page._queued, page._recent, page._news,
                         page._totals, page._system) if w.isVisible()]
    for box in boxes:
        assert inside.contains(box.geometry()), box
        for label in box.findChildren(QLabel):
            if label.isVisible() and label.text() and not label.wordWrap():
                assert label.width() >= label.minimumSizeHint().width(), (
                    label.text())
    for a, b in zip(boxes, boxes[1:]):
        assert not a.geometry().intersects(b.geometry())
    controls = page.findChild(QWidget, "HomeAsideScaleControls")
    scroll = page.findChild(QWidget, "HomeAsideScroll")
    assert not controls.geometry().intersects(scroll.geometry())
    assert QRect(0, 0, aside.width(), aside.height()).contains(
        controls.geometry())
    assert aside.geometry().left() >= page._tabs.geometry().right()


@pytest.mark.parametrize("size_end, text_end", [
    ("minimum", "minimum"), ("minimum", "maximum"),
    ("maximum", "minimum"), ("maximum", "maximum")])
def test_nothing_overlaps_or_clips_at_the_extremes(store, size_end, text_end):
    page = _home()
    size, text = _sliders(page)
    size.setValue(getattr(size, size_end)())
    text.setValue(getattr(text, text_end)())
    _pump(10)
    _check_no_overlap(page)
    page.close()


def test_junk_in_the_store_reads_back_the_default(store):
    store._settings().setValue("prefs/home_aside_size", "junk")
    store._settings().setValue("prefs/home_aside_text", "nan")
    page = _home()
    size, text = _sliders(page)
    assert (size.value(), text.value()) == (100, 100)
    page.close()
