"""Lesson 07 clip org_organize_popup: "Sort by regex" must not freeze the app.

Pressing "Sort by regex" on a 3,136-image plate folder held the GUI thread
for 581 ms: the folder listing, the mask lookups, the regex and the layout
ran in the click, then the table was cleared and rebuilt one item (with
two translated tooltips) per cell. The click now hands the reading and the
layout to a worker and the table is a model swapped whole. These tests pin
that the result is the very one the synchronous sort makes, that the heavy
steps run off the GUI thread, and, self-calibrated against the synchronous
sort on this machine, that the click and the swap are a small part of it.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from spacr import channel_sorting as cs
from spacr.qt.widgets import organize_for_measure as ofm

pytestmark = pytest.mark.qt

REGEX = r"W(?P<wellID>\d+)F(?P<fieldID>\d+)T\d+Z\d+C(?P<chanID>\d+)\.tif"


def _plate(root: Path, wells: int = 12, fields: int = 25, channels: int = 4,
           masks: bool = True) -> Path:
    """A flat plate folder of empty Yokogawa-named images, some masked.

    The sort reads names only, so the files can be empty.

    :param root: the parent.
    :param wells: wells.
    :param fields: fields per well.
    :param channels: channels per field.
    :param masks: give channel 1 of every other field a saved mask.
    """
    folder = root / "plate"
    (folder / "masks").mkdir(parents=True)
    for well in range(1, wells + 1):
        for field in range(1, fields + 1):
            for channel in range(1, channels + 1):
                name = f"W{well:04d}F{field:04d}T0001Z000C{channel}.tif"
                (folder / name).write_bytes(b"")
                if masks and channel == 1 and field % 2:
                    (folder / "masks" / name).write_bytes(b"")
    return folder


def _state(dialog) -> tuple:
    """Everything a sort decides, as plain values."""
    return (copy.deepcopy(dialog.rows), copy.deepcopy(dialog.row_keys),
            [(c.kind, c.role, c.of_channel) for c in dialog.columns],
            dialog.status.text(), dialog.output_edit.text(),
            dialog.table.rowCount(), dialog.table.columnCount(),
            [dialog.table.model().headerData(i, Qt.Horizontal)
             for i in range(dialog.table.columnCount())])


def _popup(qtbot, source: str = ""):
    """A text-view popup on ``source`` with the test regex typed in."""
    dialog = ofm._dialog_for("images")()
    qtbot.addWidget(dialog)
    dialog._set_view("text", remember=False)
    dialog.source_edit.setText(source)
    dialog.custom_edit.setText(REGEX)
    return dialog


def _click_and_wait(qtbot, dialog) -> None:
    """Press "Sort by regex" and wait for its worker's result."""
    qtbot.mouseClick(dialog.sort_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: not dialog._sort_busy(), timeout=30000)
    QApplication.processEvents()


def test_the_button_sort_is_the_synchronous_sort(qtbot, qt_theme_applied,
                                                 tmp_path):
    folder = _plate(tmp_path)
    synchronous = _popup(qtbot, str(folder))
    report = synchronous.sort_by_regex()
    assert report is not None and len(synchronous.rows) == 12 * 25
    assert any(c.kind == "mask" for c in synchronous.columns)
    clicked = _popup(qtbot, str(folder))
    _click_and_wait(qtbot, clicked)
    assert _state(clicked) == _state(synchronous)
    assert clicked.sort_button.isEnabled()
    assert clicked.cursor().shape() != Qt.BusyCursor
    one, two = clicked.prepare_plan(), synchronous.prepare_plan()
    assert one is not None and two is not None
    moves = [[(r.source_image, r.target_image, r.source_mask, r.target_mask)
              for r in plan.rows] for plan in (one, two)]
    assert moves[0] == moves[1] and moves[0]


def test_the_button_sort_of_dropped_columns_is_the_synchronous_one(
        qtbot, qt_theme_applied, tmp_path):
    folder = _plate(tmp_path, wells=2, fields=5, channels=2, masks=False)
    states = []
    for clicked in (False, True):
        dialog = _popup(qtbot)
        for channel in (1, 2):
            dialog.add_files(dialog.add_column("channel"), [
                str(p) for p in sorted(folder.glob(f"*C{channel}.tif"))])
        if clicked:
            _click_and_wait(qtbot, dialog)
        else:
            dialog.sort_by_regex()
        states.append(_state(dialog))
    assert states[0] == states[1]
    assert states[0][5] == 10


def test_nothing_to_sort_says_so_after_the_worker(qtbot, qt_theme_applied):
    dialog = _popup(qtbot)
    _click_and_wait(qtbot, dialog)
    assert dialog.status.text().startswith(ofm.tr("Nothing to sort"))
    assert dialog.sort_button.isEnabled()


def test_the_reading_and_the_regex_run_off_the_gui_thread(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = _plate(tmp_path, wells=2, fields=4)
    threads = {}
    for name in ("list_folder_images", "mask_for", "parse_names",
                 "check_sets", "_original_names"):
        original = getattr(cs, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            threads.setdefault(_name, set()).add(
                threading.current_thread() is threading.main_thread())
            return _original(*args, **kwargs)

        monkeypatch.setattr(cs, name, spy)
    dialog = _popup(qtbot, str(folder))
    _click_and_wait(qtbot, dialog)
    assert len(dialog.rows) == 8
    assert {"list_folder_images", "parse_names", "check_sets",
            "_original_names"} <= set(threads)
    assert all(seen == {False} for seen in threads.values()), threads


def test_a_newer_sort_drops_the_older_result(qtbot, qt_theme_applied,
                                             tmp_path):
    first = _plate(tmp_path / "a", wells=1, fields=3, masks=False)
    second = _plate(tmp_path / "b", wells=2, fields=3, masks=False)
    dialog = _popup(qtbot, str(first))
    assert dialog._start_sort()
    assert not dialog.sort_button.isEnabled()
    dialog.source_edit.setText(str(second))
    report = dialog.sort_by_regex()
    assert report is not None and len(dialog.rows) == 6
    qtbot.wait(300)
    QApplication.processEvents()
    assert len(dialog.rows) == 6
    assert dialog.sort_button.isEnabled()


def test_an_edit_while_the_columns_sort_is_sorted_again(
        qtbot, qt_theme_applied, tmp_path):
    folder = _plate(tmp_path, wells=1, fields=4, channels=2, masks=False)
    dialog = _popup(qtbot)
    for channel in (1, 2):
        dialog.add_files(dialog.add_column("channel"), [
            str(p) for p in sorted(folder.glob(f"*C{channel}.tif"))
            if "F0004" not in p.name])
    assert dialog._start_sort()
    for channel in (1, 2):
        dialog.add_files(channel - 1, [
            str(folder / f"W0001F0004T0001Z000C{channel}.tif")])
    qtbot.waitUntil(lambda: not dialog._sort_busy(), timeout=30000)
    assert len(dialog.rows) == 4
    assert all(all(row) for row in dialog.rows)


def test_closing_mid_sort_touches_nothing(qtbot, qt_theme_applied, tmp_path):
    folder = _plate(tmp_path, wells=2, fields=4)
    dialog = _popup(qtbot, str(folder))
    assert dialog._start_sort()
    dialog.reject()
    qtbot.wait(300)
    QApplication.processEvents()
    assert dialog.rows == []


def _gui_blocks(qtbot, dialog, action) -> float:
    """The longest the GUI thread was held while ``action`` ran and settled.

    A 2 ms timer beats on the GUI thread; the longest gap between beats is
    the longest the thread could not answer.

    :param dialog: the popup.
    :param action: what to run.
    :returns: seconds.
    """
    from PySide6.QtCore import QElapsedTimer, QTimer

    gaps = []
    clock = QElapsedTimer()
    clock.start()
    last = [clock.elapsed()]

    def beat():
        now = clock.elapsed()
        gaps.append(now - last[0])
        last[0] = now

    timer = QTimer()
    timer.setInterval(2)
    timer.timeout.connect(beat)
    timer.start()
    try:
        qtbot.wait(50)
        gaps.clear()
        last[0] = clock.elapsed()
        action()
        qtbot.waitUntil(lambda: not dialog._sort_busy(), timeout=30000)
        qtbot.wait(100)
    finally:
        timer.stop()
    return max(gaps or [0]) / 1000.0


def test_the_click_holds_the_gui_thread_for_a_small_part_of_the_sort(
        qtbot, qt_theme_applied, tmp_path):
    """Self-calibrated: the clicked sort against the same sort inline.

    The click is measured three times and the shortest of the three longest
    holds is kept, as the inline sort keeps the shortest of its runs: a
    sort back on the GUI thread holds it every time, while one descheduled
    beat on a shared runner does not repeat. One click measured 52 ms
    against a 124 ms inline sort on CI on 2026-10-13, in one of the two
    jobs that ran it; here three clicks hold it 14-25 ms against 71-82.
    """
    folder = _plate(tmp_path, wells=24, fields=25)
    timings = []
    for _round in range(2):
        inline = _popup(qtbot, str(folder))
        inline.resize(2560, 1440)
        inline.show()
        qtbot.waitExposed(inline)
        started = time.perf_counter()
        inline.sort_by_regex()
        QApplication.processEvents()
        timings.append(time.perf_counter() - started)
        inline.reject()
    synchronous = min(timings)
    if synchronous < 0.08:
        pytest.skip(f"this machine sorts 2,400 names inline in "
                    f"{synchronous * 1000:.0f} ms; nothing to tell apart")
    holds = []
    for _round in range(3):
        clicked = _popup(qtbot, str(folder))
        clicked.resize(2560, 1440)
        clicked.show()
        qtbot.waitExposed(clicked)
        holds.append(_gui_blocks(qtbot, clicked, lambda: qtbot.mouseClick(
            clicked.sort_button, Qt.LeftButton)))
        assert len(clicked.rows) == 24 * 25
        clicked.reject()
    blocked = min(holds)
    assert blocked < max(0.035, synchronous / 3), (
        f"the GUI thread was held {blocked * 1000:.0f} ms; the inline sort "
        f"takes {synchronous * 1000:.0f} ms")


def test_a_large_table_is_one_model_not_an_item_per_cell(
        qtbot, qt_theme_applied, tmp_path):
    folder = _plate(tmp_path, wells=4, fields=10, masks=False)
    dialog = _popup(qtbot, str(folder))
    dialog.sort_by_regex()
    model = dialog.table.model()
    assert isinstance(model, ofm._CellModel)
    dialog.sort_by_regex()
    assert dialog.table.model() is not model
    item = dialog.table.item(0, 0)
    assert item.text() == os.path.basename(dialog.rows[0][0])
    assert item.data(Qt.UserRole) == dialog.rows[0][0]
    assert dialog.rows[0][0] in item.toolTip()
    assert dialog.table.item(len(dialog.rows), 0) is None
    item.setSelected(True)
    assert [(i.row(), i.column()) for i in dialog.table.selectedItems()] == [
        (0, 0)]


@pytest.mark.parametrize("pages", [1, 3])
def test_a_tiff_thumbnail_is_the_pillow_one(tmp_path, pages):
    """tifffile reads only a plain plane, the one Pillow gave, unchanged."""
    from PIL import Image

    rng = np.random.default_rng(pages)
    stack = rng.integers(0, 60000, (pages, 300, 260)).astype(np.uint16)
    path = tmp_path / "plane.tif"
    tifffile.imwrite(str(path), stack if pages > 1 else stack[0])
    plane = cs._plain_tiff_plane(str(path))
    if pages == 1:
        with Image.open(path) as image:
            assert np.array_equal(plane, np.asarray(image))
    else:
        assert plane is None
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a tiff")
    assert cs._plain_tiff_plane(str(broken)) is None
    assert cs.thumbnail(str(broken)) is None
    thumb = cs.thumbnail(str(path), 96)
    assert thumb.dtype == np.uint8 and max(thumb.shape) <= 100


def test_the_text_view_is_sized_as_qt_would_size_it(qtbot, qt_theme_applied,
                                                    tmp_path):
    folder = _plate(tmp_path, wells=3, fields=12)
    (folder / "W0001F0001T0001Z000C1_extra_long_name.tif").write_bytes(b"")
    dialog = _popup(qtbot, str(folder))
    dialog.custom_edit.setText(
        r"W(?P<wellID>\d+)F(?P<fieldID>\d+)T\d+Z\d+C(?P<chanID>\d+)"
        r"(?:_extra_long_name)?\.tif")
    dialog.show()
    _click_and_wait(qtbot, dialog)
    table = dialog.table
    fitted = ([table.columnWidth(c) for c in range(table.columnCount())],
              sorted({table.rowHeight(r) for r in range(table.rowCount())}))
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    assert fitted == (
        [table.columnWidth(c) for c in range(table.columnCount())],
        sorted({table.rowHeight(r) for r in range(table.rowCount())}))
