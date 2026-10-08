"""Item 597: Home's right column resizes by the settings handle; one slider sizes text.

The maintainer, 2026-09-29: "add to new list, a slider on the home screen that
lets the user change the size of the rwidgets on the right on the bottom of
those widgest should be a slider that allows the user to change the size of
the text."
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRect, QSettings, Qt   # noqa: E402
from PySide6.QtGui import QMouseEvent                                # noqa: E402
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


def _home(qtbot):
    page = home_mod.HomePage(APPS, lambda key: None)
    qtbot.addWidget(page)
    page.resize(1400, 900)
    page.show()
    _pump()
    return page


def _text_slider(page) -> QSlider:
    sliders = page.findChildren(QSlider)
    assert [w.objectName() for w in sliders] == ["HomeAsideTextSlider"]
    return sliders[0]


def _handle(page):
    split = page._aside_split
    return split.handle(split.indexOf(page._aside))


def _drag_handle(page, dx: int) -> None:
    split = page._aside_split
    handle = _handle(page)
    split.moveSplitter(handle.geometry().x() + dx, split.indexOf(page._aside))
    _pump()


def _click_handle(page) -> None:
    handle = _handle(page)
    centre = QPointF(handle.rect().center())
    for kind, button, buttons in (
            (QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
            (QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton)):
        QApplication.sendEvent(handle, QMouseEvent(
            kind, centre, QPointF(handle.mapToGlobal(centre.toPoint())),
            button, buttons, Qt.NoModifier))
    _pump()


def _row_label(page) -> QLabel:
    """A label the Totals panel builds with its own font-size sheet."""
    labels = [w for w in page._totals.findChildren(QLabel)
              if "font-size" in (w.styleSheet() or "") and w.text()]
    assert labels
    return labels[0]


def _px(label: QLabel) -> int:
    label.ensurePolished()
    return label.fontInfo().pixelSize()


def test_one_slider_sits_right_below_the_lowest_widget(store, qtbot):
    page = _home(qtbot)
    _text_slider(page)
    controls = page.findChild(QWidget, "HomeAsideScaleControls")
    assert controls.parentWidget() is page._aside_panels
    lowest = page._system.geometry()
    assert 0 <= controls.geometry().top() - lowest.bottom() <= 40
    page.close()


def test_the_column_resizes_by_the_settings_columns_handle(store, qtbot):
    from spacr.qt.widgets.collapsible_splitter import (EDGE,
                                                       CollapsibleSplitter)

    page = _home(qtbot)
    split = page._aside_split
    assert isinstance(split, CollapsibleSplitter)
    assert split.pane("Widgets").mode == EDGE
    assert _handle(page).edge_pane() is split.pane("Widgets")
    before = page._aside.width()
    _drag_handle(page, -120)
    wider = page._aside.width()
    assert wider > before
    page.close()

    again = _home(qtbot)
    assert abs(again._aside.width() - wider) <= 2
    again.close()


def test_the_handle_arrow_folds_the_column_and_it_stays_folded(store, qtbot):
    page = _home(qtbot)
    assert page._aside.width() > 0
    _click_handle(page)
    assert page._aside_split.is_collapsed("Widgets")
    assert page._aside_split.sizes()[-1] == 0
    page.close()

    again = _home(qtbot)
    assert again._aside_split.is_collapsed("Widgets")
    _click_handle(again)
    assert not again._aside_split.is_collapsed("Widgets")
    assert again._aside.width() > 0
    again.close()


def test_text_size_changes_text_persists_and_restores(store, qtbot):
    page = _home(qtbot)
    text = _text_slider(page)
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

    again = _home(qtbot)
    assert _text_slider(again).value() == 130
    assert _px(_row_label(again)) > base
    again.close()


def test_a_rebuilt_row_takes_the_size_without_compounding(store, qtbot):
    page = _home(qtbot)
    text = _text_slider(page)
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
    for box in boxes:
        assert not controls.geometry().intersects(box.geometry())
    assert inside.contains(controls.geometry())
    assert aside.width() >= aside.minimumWidth() > 0
    assert panels.width() <= aside.width()


@pytest.mark.parametrize("drag, text_end", [
    (400, "minimum"), (400, "maximum"),
    (-500, "minimum"), (-500, "maximum")])
def test_nothing_overlaps_or_clips_at_the_extremes(store, qtbot, drag, text_end):
    page = _home(qtbot)
    text = _text_slider(page)
    text.setValue(getattr(text, text_end)())
    _pump()
    _drag_handle(page, drag)
    _pump(10)
    if page._aside_split.is_collapsed("Widgets"):
        _click_handle(page)
        _pump(10)
    _check_no_overlap(page)
    page.close()


def test_junk_in_the_store_reads_back_the_default(store, qtbot):
    store._settings().setValue("prefs/home_aside_text", "nan")
    page = _home(qtbot)
    assert _text_slider(page).value() == 100
    page.close()
