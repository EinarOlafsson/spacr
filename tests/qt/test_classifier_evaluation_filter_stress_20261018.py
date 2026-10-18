"""The prediction filter survives typing at full speed (lesson 39 segfault).

On the tutorial capture host, typing ``r5_c2`` into Classifier Evaluation's
"Filter predictions" box killed the application with SIGSEGV inside
``QTableWidget.clear()`` (``_render_frame`` < ``_render_predictions``), with
the field backdrop shading on its own thread at the same moment. The same
Python stack was reproduced offscreen on 2026-10-18, with the backdrop on
and with it off, once accessibility was active and Python still held
wrappers of accessible table cells that Qt deletes on every model reset. The
native frame is ``Shiboken::BindingManager::releaseWrapper`` in a table
item's destructor: every keystroke tore down and rebuilt every table item,
and a new item allocated where a deleted, still-wrapped object lived is torn
down through that stale wrapper. The backdrop thread shares no buffer,
image, palette or font with the table.

The screen now re-filters once typing pauses, applies Enter, focus-out and
an emptied box at once, never nests one render inside another, skips a
render whose filter is already on screen, and leaves a destroyed table
alone. These tests pin that, plus the frame hand-over the backdrop thread
was suspected of: a shaded frame owns its pixels.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from spacr.classifier_evaluation import (
    evaluate_predictions,
    write_evaluation_bundle,
)
from spacr.qt.screens import classifier_evaluation as CE
from spacr.qt.screens.classifier_evaluation import ClassifierEvaluationScreen

ROWS = 600


@pytest.fixture
def bundle_root(tmp_path):
    """A 600-prediction bundle whose file names carry plate, row and column."""
    paths = [
        f"plate{1 + i % 3}_r{1 + i % 8}_c{1 + (i // 8) % 12}_f{i % 4}_{i}.png"
        for i in range(ROWS)
    ]
    positive = np.linspace(0.02, 0.98, ROWS)
    evaluation = evaluate_predictions(
        np.arange(ROWS) % 2,
        np.stack([1.0 - positive, positive], axis=1),
        paths,
        classes=["infected_1", "infected_2"],
        fold_ids=[1 + i % 3 for i in range(ROWS)],
        calibration_method="none",
        calibration_bins=5,
    )
    return write_evaluation_bundle(
        tmp_path / "model" / "evaluation", evaluation).parents[1]


def _loaded_screen(bundle_root, parent=None):
    widget = ClassifierEvaluationScreen(parent, threaded=False)
    widget._source.setText(str(bundle_root))
    widget.scan()
    assert widget.bundle is not None
    widget._tabs.setCurrentIndex(4)
    return widget


@pytest.fixture
def screen(qtbot, qt_theme_applied, bundle_root):
    widget = _loaded_screen(bundle_root)
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    widget.show()
    return widget


def _count_prediction_renders(screen, monkeypatch):
    """Wrap ``_render_frame`` and count the renders of the predictions table."""
    calls = []
    original = screen._render_frame

    def counting(table, frame, **kwargs):
        if table is screen._predictions:
            calls.append(len(frame))
        return original(table, frame, **kwargs)

    monkeypatch.setattr(screen, "_render_frame", counting)
    return calls


def _expected_basenames(screen, text):
    """The rows a filter should keep, worked out without the screen's code."""
    frame = screen.bundle["predictions"]
    terms = [term.casefold() for term in text.split() if term]
    keep = [
        all(term in " ".join(str(v) for v in row).casefold() for term in terms)
        for row in frame.itertuples(index=False)
    ]
    return list(frame.loc[keep, "basename"].astype(str).head(2000))


def _shown_basenames(screen):
    table = screen._predictions
    headers = [table.horizontalHeaderItem(c).text()
               for c in range(table.columnCount())]
    column = headers.index("basename")
    return [table.item(row, column).text() for row in range(table.rowCount())]


def _read_cells_like_an_assistive_client(table):
    """Walk a table's accessible cells the way a screen reader would.

    Nothing is held afterwards: a Python reference kept to an interface Qt
    deletes on the next model reset is a dangling wrapper of the test's own
    making, not the screen's.
    """
    interface = QAccessible.queryAccessibleInterface(table)
    if interface is None:
        return
    count = interface.childCount()
    for index in range(0, count, max(1, count // 40)):
        child = interface.child(index)
        if child is not None:
            child.text(QAccessible.Name)


def test_typing_rebuilds_the_table_once_after_the_pause(
        qtbot, screen, monkeypatch):
    renders = _count_prediction_renders(screen, monkeypatch)
    QTest.keyClicks(screen._prediction_filter, "r5_c2")
    assert renders == [], "a keystroke rebuilt the table before the pause"
    expected = _expected_basenames(screen, "r5_c2")
    assert 0 < len(expected) < ROWS
    qtbot.waitUntil(lambda: _shown_basenames(screen) == expected,
                    timeout=3000)
    assert len(renders) == 1, renders


def test_enter_applies_the_filter_at_once_and_only_once(
        qtbot, screen, monkeypatch):
    renders = _count_prediction_renders(screen, monkeypatch)
    QTest.keyClicks(screen._prediction_filter, "plate2 r3")
    QTest.keyClick(screen._prediction_filter, Qt.Key_Return)
    assert _shown_basenames(screen) == _expected_basenames(screen, "plate2 r3")
    assert len(renders) == 1
    qtbot.wait(CE._FILTER_DELAY_MS * 2)
    screen._prediction_filter.editingFinished.emit()
    assert len(renders) == 1, "the same filter was rendered twice"


def test_an_emptied_box_restores_every_row_at_once(qtbot, screen):
    QTest.keyClicks(screen._prediction_filter, "plate3")
    QTest.keyClick(screen._prediction_filter, Qt.Key_Return)
    assert screen._predictions.rowCount() == len(
        _expected_basenames(screen, "plate3"))
    QTest.keyClick(screen._prediction_filter, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClick(screen._prediction_filter, Qt.Key_Backspace)
    assert screen._predictions.rowCount() == ROWS


def test_a_request_during_a_render_runs_after_it_with_the_newest_filter(
        screen, monkeypatch):
    depth, peak, renders = [0], [0], []
    original = screen._render_frame

    def interrupting(table, frame, **kwargs):
        if table is not screen._predictions:
            return original(table, frame, **kwargs)
        depth[0] += 1
        peak[0] = max(peak[0], depth[0])
        try:
            renders.append(len(frame))
            if len(renders) == 1:
                screen._prediction_filter.setText("plate1 c3")
                screen._render_predictions()
            return original(table, frame, **kwargs)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(screen, "_render_frame", interrupting)
    screen._prediction_filter.setText("plate2")
    screen._apply_prediction_filter()
    assert peak[0] == 1, "a render ran inside another render"
    assert len(renders) == 2
    assert _shown_basenames(screen) == _expected_basenames(screen, "plate1 c3")


def test_a_filter_pending_on_a_destroyed_screen_is_dropped(
        qt_theme_applied, bundle_root):
    widget = _loaded_screen(bundle_root)
    table = widget._predictions
    QTest.keyClicks(widget._prediction_filter, "r1")
    assert widget._filter_timer.isActive()
    render = widget._render_predictions
    apply = widget._apply_prediction_filter
    schedule = widget._schedule_prediction_filter
    draw = widget._render_frame
    shiboken6.delete(widget)
    assert not shiboken6.isValid(table)
    render()
    apply()
    schedule("r2")
    schedule("")
    draw(table, widget.bundle["predictions"])
    for _ in range(5):
        QApplication.processEvents()


def test_two_hundred_rapid_filter_changes_leave_the_view_on_the_last_filter(
        qtbot, qt_theme_applied, bundle_root, monkeypatch):
    host = QWidget()
    qtbot.addWidget(host)
    QVBoxLayout(host)
    screen = _loaded_screen(bundle_root, host)
    host.layout().addWidget(screen)
    host.resize(1500, 950)
    host.show()
    renders = _count_prediction_renders(screen, monkeypatch)
    edit = screen._prediction_filter
    words = ["r5_c2", "plate2", "infected", "r1", "c1", "f3", "xyz",
             "plate1 r2", "_1", "PLATE3 R8"]
    was_active = QAccessible.isActive()
    QAccessible.setActive(True)
    changes = 0
    try:
        for step in range(240):
            word = words[step % len(words)]
            mode = step % 6
            if mode == 0:
                edit.setText(word)
            elif mode == 1:
                QTest.keyClicks(edit, word[:3])
            elif mode == 2:
                QTest.keyClick(edit, Qt.Key_Backspace)
            elif mode == 3:
                QTimer.singleShot(0, lambda w=word: edit.setText(w))
                QTimer.singleShot(1, lambda w=word: edit.setText(w[::-1]))
            elif mode == 4:
                edit.setText(word + " " + words[(step + 3) % len(words)])
            elif step % 30 == 5:
                QTest.keyClick(edit, Qt.Key_Return)
            else:
                edit.setText(word.upper())
            changes += 1
            if step % 7 == 0:
                host.resize(1200 + (step * 37) % 600, 800 + (step * 53) % 300)
            if step % 11 == 0:
                _read_cells_like_an_assistive_client(screen._predictions)
            QApplication.processEvents()
        for _ in range(5):
            QApplication.processEvents()
        qtbot.wait(20)
        QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(edit, "plate1 r5_c2")
        expected = _expected_basenames(screen, "plate1 r5_c2")
        qtbot.waitUntil(lambda: _shown_basenames(screen) == expected,
                        timeout=5000)
        _read_cells_like_an_assistive_client(screen._predictions)
    finally:
        QAccessible.setActive(was_active)
    assert edit.text() == "plate1 r5_c2"
    assert 0 < len(expected) < ROWS
    assert len(renders) <= changes // 2, (
        f"{len(renders)} rebuilds for {changes} filter changes: typing is "
        "not being coalesced")


def test_a_shaded_field_frame_keeps_its_pixels_after_the_engine_moves_on(qapp):
    """The frame the shading thread hands to the GUI owns its memory.

    The backdrop's grains are written through NumPy views of the QImage's
    own buffer, so the image the GUI paints must not depend on any array
    the engine later frees, resizes or reuses for the next frame.
    """
    from spacr.qt.widgets.ambient import make_engine, palettes_for

    theme = "data_art_impulse_lens"
    engine = make_engine(theme, palettes_for(theme)[0], "#0b1622", seed=7)
    first = engine.shade(240, 135)
    assert first is not None and not first.isNull()
    pixels = bytes(first.constBits())
    assert len(pixels) == first.sizeInBytes()
    for size in ((240, 135), (320, 180), (160, 90)):
        engine.advance(0.75)
        later = engine.shade(*size)
        assert later is not first
    del engine, later
    gc.collect()
    assert (first.width(), first.height()) == (240, 135)
    assert bytes(first.constBits()) == pixels
