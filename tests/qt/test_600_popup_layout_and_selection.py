"""Item 600c: Make Masks' navigation row and the Organize popup's layout,
colour control, size slider, multi-selection and empty-table hint, offscreen.

The maintainer, 2026-09-29: "in make masks the skip button should be on the
same row as the other buttons and the keep and discard buttons should be
farther away from the save, next and previous buttons ... there needs to be a
slider for controlling the size of the images in the popup ... drag and
select many images or click the column to select all in the column and same
for row to move them or delete them."
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent

from spacr.qt.widgets import organize_for_measure as ofm

pytestmark = pytest.mark.qt


def _tree(root: Path) -> Path:
    """``exp/DAPI`` and ``exp/GFP``, three masked fields each.

    :param root: the parent.
    """
    exp = root / "exp"
    for channel in ("DAPI", "GFP"):
        for i in range(1, 4):
            for sub in ("", "masks"):
                path = exp / channel / sub / f"field{i}.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                array = np.zeros((12, 12), np.uint16)
                array[2:6, 2:6] = i if sub else 0
                array[0, 0] = 100 + i
                tifffile.imwrite(str(path), array)
    return exp


@pytest.fixture
def dialog(qtbot, qt_theme_applied):
    """An empty popup."""
    made = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(made)
    return made


@pytest.fixture
def filled(dialog, tmp_path):
    """The popup with DAPI, its mask, GFP and its mask, three rows."""
    exp = _tree(tmp_path)
    dialog.add_files(dialog.add_column("channel"), [exp / "DAPI"])
    dialog.add_files(dialog.add_column("channel"), [exp / "GFP"])
    assert [c.kind for c in dialog.columns] == [
        "channel", "mask", "channel", "mask"]
    dialog.resize(1000, 760)
    dialog.show()
    return dialog


def _mouse(kind, pos, button, buttons, modifiers=Qt.NoModifier):
    """A mouse event at ``pos`` in the viewport.

    :param kind: the event type.
    :param pos: a QPoint.
    :param button: the button that changed.
    :param buttons: the buttons held.
    :param modifiers: the keyboard modifiers.
    """
    return QMouseEvent(kind, QPointF(pos), QPointF(pos), button, buttons,
                       modifiers)


# -- Make Masks' navigation row --------------------------------------------

def test_the_maintainers_button_layout(qtbot, qt_theme_applied, monkeypatch,
                                      tmp_path):
    """The Make Masks button layout the maintainer asked for (2026-09-30).

    Bottom row, left: Open folder, Organize for Measure, Load test data,
    Uncertainty. Editor action row: Save mask, Prev, Next (item 247, 2026-10-01).
    Bottom row, right, under the console: Discard, Keep, Skip, Blind, ROIs,
    Upload data -- Keep and Discard no longer on a row of their own.
    """
    from spacr.qt import preferences
    from spacr.qt.screens import make_masks as mm

    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: True)
    screen = mm.MakeMasksScreen()
    qtbot.addWidget(screen)
    curate = screen._nav_curate_group
    steps = [screen._btn_save, screen._btn_prev, screen._btn_next]

    def widgets(group):
        layout = group.layout()
        return [layout.itemAt(i).widget() for i in range(layout.count())
                if layout.itemAt(i).widget() is not None]

    for button in steps:
        assert screen._tool_row_layout.indexOf(button) >= 0
    assert widgets(curate) == [screen._btn_discard, screen._btn_keep,
                               screen._btn_skip, screen._btn_blind,
                               screen._btn_rois, screen._btn_contribute]
    # One widget each, so neither group can wrap apart.
    for group in (curate,):
        for button in widgets(group):
            assert button.parent() is group
    assert screen._btn_training_datasets is screen._btn_test_data
    import imageio.v2 as imageio
    import numpy as np

    (tmp_path / "masks").mkdir()
    imageio.imwrite(tmp_path / "field.tif", np.zeros((48, 48), np.uint16))
    screen._open_folder(str(tmp_path))
    screen.resize(1600, 900)
    screen.show()
    qtbot.waitExposed(screen)

    def top_left(button):
        return button.mapTo(screen, button.rect().topLeft())

    toolbar = [screen._btn_open, screen._btn_organize,
               screen._btn_test_data, screen._btn_uncertainty]
    xs = [top_left(button).x() for button in toolbar]
    assert xs == sorted(xs)
    row_y = {top_left(button).y() for button in toolbar + widgets(curate)}
    assert len(row_y) == 1, "the toolbar and the curation buttons share a row"
    assert top_left(screen._btn_open).x() < top_left(screen._btn_discard).x()
    # Item 247: saving/navigation share the top editor action row.
    centers = {button.mapTo(screen, button.rect().center()).y()
               for button in steps + [screen._btn_cellpose, screen._btn_undo]}
    assert max(centers) - min(centers) <= 1
    xs = [top_left(button).x() for button in steps]
    assert xs == sorted(xs)


# -- the popup's header ----------------------------------------------------

def test_fields_above_and_every_action_button_on_one_line(filled):
    dialog = filled
    row = dialog.action_row
    buttons = [row.itemAt(i).widget() for i in range(row.count())
               if row.itemAt(i).widget() is not None]
    assert buttons == [dialog.sort_button, dialog.auto_button,
                       dialog.detect_button, dialog.teach_button,
                       dialog.add_channel_button, dialog.add_mask_button,
                       dialog.remove_button]
    tops = {b.mapTo(dialog, b.rect().topLeft()).y() for b in buttons}
    assert len(tops) == 1
    line = tops.pop()
    for field in (dialog.source_edit, dialog.consolidate_check,
                  dialog.metadata_field, dialog.custom_edit,
                  dialog.output_edit):
        assert field.mapTo(dialog, field.rect().topLeft()).y() < line
    table_top = dialog.table.mapTo(dialog, dialog.table.rect().topLeft()).y()
    assert line < table_top
    # The source, custom regex and output fields share one left edge.
    lefts = {w.mapTo(dialog, w.rect().topLeft()).x()
             for w in (dialog.source_edit, dialog.custom_edit,
                       dialog.output_edit)}
    assert len(lefts) == 1


def test_text_colour_is_plain_text_left_of_show_and_opens_a_picker(
        filled, monkeypatch):
    dialog = filled
    dialog._set_view("both")
    assert dialog.color_button.isFlat()
    assert (dialog.color_button.mapTo(dialog, dialog.color_button.rect()
                                      .topLeft()).x()
            < dialog.view_box.mapTo(dialog, dialog.view_box.rect()
                                    .topLeft()).x())
    asked = []

    # 2026-09-30 (item 43): the popup asks through the shared
    # ``pick_colour`` helper, not ``QColorDialog.getColor``, so the helper
    # is what is stood in for.
    def pick(parent, initial, title):
        asked.append(QColor(initial).name())
        return QColor("#00ff00")

    monkeypatch.setattr(ofm, "_headless", lambda: False)
    monkeypatch.setattr(ofm, "pick_colour", pick)
    dialog.color_button.click()
    assert asked
    assert dialog.delegate.text_color.name() == "#00ff00"
    sheet = dialog.color_button.styleSheet()
    assert "color: #00ff00" in sheet and "border: none" in sheet
    assert "border-left" not in sheet
    assert ofm._load_view_prefs()[1] == "#00ff00"


def test_the_size_slider_resizes_live_and_is_remembered(qtbot, filled):
    dialog = filled
    dialog._set_view("image")
    assert dialog.size_slider.isEnabled()
    dialog.size_slider.setValue(200)
    assert dialog.table.columnWidth(0) == 224
    assert dialog.table.rowHeight(0) == 208
    assert ofm._load_thumb_pref() == 200
    again = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(again)
    assert again.size_slider.value() == 200 and again.thumb_size == 200
    dialog._set_view("text")
    assert not dialog.size_slider.isEnabled()
    assert ofm._clamp_thumb("junk") == ofm._CELL_THUMB
    assert ofm._clamp_thumb(10_000) == ofm._THUMB_RANGE[1]
    dialog._stop_thumbs()
    again._stop_thumbs()


# -- selection -------------------------------------------------------------

def test_header_clicks_select_a_column_or_a_row(qtbot, filled):
    table = filled.table
    header = table.horizontalHeader()
    x = header.sectionViewportPosition(2) + header.sectionSize(2) // 2
    qtbot.mouseClick(header.viewport(), Qt.LeftButton,
                     pos=header.viewport().rect().topLeft().__class__(x, 5))
    assert sorted({i.column() for i in table.selectedIndexes()}) == [2]
    assert len(table.selectedIndexes()) == 3
    rows = table.verticalHeader()
    y = rows.sectionViewportPosition(1) + rows.sectionSize(1) // 2
    qtbot.mouseClick(rows.viewport(), Qt.LeftButton,
                     pos=rows.viewport().rect().topLeft().__class__(5, y))
    assert sorted({i.row() for i in table.selectedIndexes()}) == [1]
    # Ctrl extends to a second row.
    y2 = rows.sectionViewportPosition(2) + rows.sectionSize(2) // 2
    qtbot.mouseClick(rows.viewport(), Qt.LeftButton, Qt.ControlModifier,
                     rows.viewport().rect().topLeft().__class__(5, y2))
    assert sorted({i.row() for i in table.selectedIndexes()}) == [1, 2]


def test_a_rubber_band_selects_many_and_delete_clears_them(filled):
    dialog = filled
    table = dialog.table
    start = table.visualRect(table.model().index(0, 0)).center()
    end = table.visualRect(table.model().index(1, 2)).center()
    # Ctrl held: a band even though the press is on a file.
    table.mousePressEvent(_mouse(QEvent.MouseButtonPress, start,
                                 Qt.LeftButton, Qt.LeftButton,
                                 Qt.ControlModifier))
    table.mouseMoveEvent(_mouse(QEvent.MouseMove, end, Qt.NoButton,
                                Qt.LeftButton, Qt.ControlModifier))
    assert table._band is not None and table._band.isVisible()
    table.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, end,
                                   Qt.LeftButton, Qt.NoButton,
                                   Qt.ControlModifier))
    assert not table._band.isVisible()
    picked = ofm._selection_cells(table.selectedIndexes())
    assert [0, 0] in picked and [1, 2] in picked and [0, 1] in picked
    gone = {dialog.rows[0][0], dialog.rows[1][2]}
    table.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete,
                                  Qt.NoModifier))
    left = {p for row in dialog.rows for p in row}
    assert not gone & left
    assert len(dialog.rows) == 1


def test_the_close_mark_on_a_selected_cell_clears_the_whole_selection(filled):
    dialog = filled
    table = dialog.table
    table.selectColumn(0)
    rect = ofm._close_rect(table.visualRect(table.model().index(1, 0)))
    for kind, button, buttons in (
            (QEvent.MouseButtonPress, Qt.LeftButton, Qt.LeftButton),
            (QEvent.MouseButtonRelease, Qt.LeftButton, Qt.NoButton)):
        event = _mouse(kind, rect.center(), button, buttons)
        (table.mousePressEvent if kind == QEvent.MouseButtonPress
         else table.mouseReleaseEvent)(event)
    assert dialog._column_files(0) == [] and dialog._column_files(1) == []
    assert len(dialog._column_files(2)) == 3


def test_block_moves_are_pure_and_swap_pairwise():
    rows = [["a", "am", "x", None],
            ["b", "bm", "y", None],
            [None, None, "z", None]]
    mask_of = {0: 1, 2: 3}
    moves = ofm._block_moves([(0, 0), (1, 0)], (0, 0), (1, 2), 4, mask_of)
    assert moves[(0, 0)] == (1, 2) and moves[(0, 1)] == (1, 3)
    ofm._move_block(rows, moves)
    # a, b land on rows 1-2 of column 2 with their masks; y and z, which
    # were there, swap back into the slots a and b left.
    assert [r[2] for r in rows] == ["x", "a", "b"]
    assert [r[3] for r in rows] == [None, "am", "bm"]
    assert [r[0] for r in rows] == ["y", "z", None]
    # Off the side: nothing moves.
    assert ofm._block_moves([(0, 3)], (0, 3), (0, 4), 4, {}) == {}
    # Shifted onto itself by one row: nothing is lost, a new row appears.
    rows = [["p"], ["q"]]
    ofm._move_block(rows, ofm._block_moves([(0, 0), (1, 0)], (0, 0), (1, 0),
                                           1, {}))
    assert rows == [[None], ["p"], ["q"]]


def test_a_selection_dragged_together_moves_as_a_block(filled):
    dialog = filled
    table = dialog.table
    first, second, gfp = dialog.rows[0][0], dialog.rows[1][0], dialog.rows[0][2]
    table.item(0, 0).setSelected(True)
    table.item(1, 0).setSelected(True)
    table._drag_anchor = [0, 0]
    mime = table.mimeData([table.item(0, 0), table.item(1, 0)])
    assert ofm._mime_anchor(mime) == [0, 0]
    assert len(ofm._mime_slots(mime)) == 2
    seen = []
    table.block_moved.connect(lambda *a: seen.append(a))
    x = table.columnViewportPosition(2) + 5
    y = table.rowViewportPosition(0) + 5
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QDragEnterEvent, QDropEvent

    enter = QDragEnterEvent(QPoint(x, y), Qt.MoveAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    table.dragEnterEvent(enter)
    assert enter.isAccepted()
    table.dropEvent(QDropEvent(QPointF(x, y), Qt.MoveAction, mime,
                               Qt.LeftButton, Qt.NoModifier))
    assert seen and seen[0][1] == [0, 0] and seen[0][2] == [0, 2]
    assert dialog.rows[0][2] == first and dialog.rows[1][2] == second
    assert dialog.rows[0][0] == gfp
    # The masks went with the images, both ways.
    assert os.path.basename(os.path.dirname(dialog.rows[0][3])) == "masks"
    assert "DAPI" in dialog.rows[0][3] and "GFP" in dialog.rows[0][1]
    assert ofm._mime_anchor(None) is None


# -- the drop hint ---------------------------------------------------------

def test_the_drop_hint_is_in_the_table_until_something_is_in_it(
        dialog, tmp_path):
    table = dialog.table
    assert table._is_empty() and not dialog.new_zone.isVisibleTo(dialog)
    assert table.placeholder
    table.viewport().grab()  # paints the hint without error
    exp = _tree(tmp_path)
    dialog.add_files(dialog.add_column("channel"), [exp / "DAPI"])
    assert not table._is_empty() and dialog.new_zone.isVisibleTo(dialog)
    table.viewport().grab()
    dialog._clear_slots([[r, 0] for r in range(len(dialog.rows))])
    assert table._is_empty() and not dialog.new_zone.isVisibleTo(dialog)


def test_the_text_colour_caption_stays_legible_on_any_window():
    light, dark = QColor("#f0f0f0"), QColor("#202020")
    assert ofm._legible_backdrop(QColor("white"), light) != "transparent"
    assert ofm._legible_backdrop(QColor("black"), dark) != "transparent"
    assert ofm._legible_backdrop(QColor("white"), dark) == "transparent"
    assert ofm._legible_backdrop(QColor("black"), light) == "transparent"
    assert ofm._legible_backdrop(QColor("#ffff00"), light).startswith("rgba(0")


def test_rebuilt_column_editors_leave_no_stale_widget_on_screen(filled):
    from PySide6.QtWidgets import QApplication, QPushButton

    dialog = filled
    dialog.show()
    for _ in range(3):
        dialog._refresh()
        QApplication.processEvents()
    removes = [b for b in dialog.findChildren(QPushButton)
               if b.text() == ofm.tr("Remove") and b.isVisibleTo(dialog)]
    assert len(removes) == len(dialog.columns)


def test_image_view_columns_are_never_narrower_than_their_heading(filled):
    dialog = filled
    dialog._set_view("image")
    dialog.size_slider.setValue(ofm._THUMB_RANGE[0])
    header = dialog.table.horizontalHeader()
    for column in range(dialog.table.columnCount()):
        assert dialog.table.columnWidth(column) >= header.sectionSizeHint(column)
    assert dialog.table.columnWidth(1) > ofm._THUMB_RANGE[0] + 24
