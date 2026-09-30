"""Item 600: the "Organize for Measure" popup, offscreen.

Drops are simulated with real ``QDragEnterEvent``/``QDropEvent`` objects
carrying file URLs (a drag from the file manager) or the table's own cell
format (a drag between columns, built by the table's ``mimeData``). Modal
questions go through the dialog's ``ask``/``confirm_examples`` hooks, which
the tests replace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from spacr import channel_sorting as cs
from spacr.qt.widgets import organize_for_measure as ofm

pytestmark = pytest.mark.qt


def _tif(path: Path, array=None, shape=(12, 12)) -> Path:
    """Write a TIFF, making the folder.

    :param path: where.
    :param array: the pixels; default a small noisy image.
    :param shape: the default image's shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.random.default_rng(len(str(path))).integers(
            100, 4000, shape).astype(np.uint16)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _mask(i: int, shape=(12, 12)) -> np.ndarray:
    """A label mask with one object.

    :param i: its label.
    :param shape: its shape.
    """
    array = np.zeros(shape, np.uint16)
    array[2:6, 2:6] = i
    return array


def _tree(root: Path, masks=True) -> Path:
    """``exp/DAPI`` and ``exp/GFP``, three fields each, masked.

    :param root: the parent.
    :param masks: write each folder's ``masks/``.
    """
    exp = root / "exp"
    for channel in ("DAPI", "GFP"):
        for i in range(1, 4):
            _tif(exp / channel / f"field{i}.tif")
            if masks:
                _tif(exp / channel / "masks" / f"field{i}.tif", _mask(i))
    return exp


def _drawn(root: Path) -> Path:
    """The 593 case: nucleus and cell images of three fields in one folder.

    :param root: the parent.
    """
    folder = root / "drawn"
    for i in range(1, 4):
        for stain in ("nuc", "cell"):
            name = f"{stain}_img{i:02d}.tif"
            _tif(folder / name)
            _tif(folder / "masks" / name, _mask(i))
    return folder


def _urls(*paths) -> QMimeData:
    """A drag from the file manager.

    :param paths: the dragged files and folders.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _drop(widget, mime: QMimeData, x: int = 5) -> bool:
    """Drag ``mime`` over ``widget`` and drop it at ``x``.

    :param widget: the drop target.
    :param mime: what is dragged.
    :param x: the horizontal drop position, in the widget.
    :returns: whether the drag was accepted on entering.
    """
    enter = QDragEnterEvent(QPoint(x, 5), Qt.CopyAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    widget.dragEnterEvent(enter)
    if not enter.isAccepted():
        return False
    widget.dropEvent(QDropEvent(QPointF(x, 5), Qt.CopyAction, mime,
                                Qt.LeftButton, Qt.NoModifier))
    return True


def _x_of(table, column: int) -> int:
    """The middle of a column, in the table viewport's coordinates.

    :param table: the table.
    :param column: the column.
    """
    return table.columnViewportPosition(column) + table.columnWidth(column) // 2


@pytest.fixture
def dialog(qtbot, qt_theme_applied):
    """An empty popup."""
    made = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(made)
    return made


def test_the_regex_box_is_mask_generations_own(dialog):
    from spacr.qt.screens.settings_model import _MetadataTypeField
    from spacr.regex_infer import _metadata_convention_keys

    assert isinstance(dialog.metadata_field, _MetadataTypeField)
    combo = dialog.metadata_field.combo
    keys = {combo.itemData(i) for i in range(combo.count())} - {None}
    assert keys == set(_metadata_convention_keys())
    assert dialog.metadata_field.get_value() == "auto"


def test_mode_2_folders_dropped_into_columns_match_and_apply(
        dialog, tmp_path):
    exp = _tree(tmp_path)
    dialog.add_column("channel")
    dialog.add_column("channel")
    assert _drop(dialog.table, _urls(exp / "DAPI"), _x_of(dialog.table, 0))
    # The DAPI masks came along in a nucleus mask column of channel 1.
    kinds = [(c.kind, c.role, c.of_channel) for c in dialog.columns]
    assert kinds == [("channel", "", 1), ("channel", "", 1),
                     ("mask", "nucleus", 1)]
    assert _drop(dialog.table, _urls(exp / "GFP"), _x_of(dialog.table, 1))
    assert len(dialog.columns) == 4
    assert dialog.columns[3].kind == "mask" and dialog.columns[3].of_channel == 2
    assert len(dialog.rows) == 3 and not dialog._incomplete_rows()
    for row in dialog.rows:
        stems = {os.path.basename(p) for p in row}
        assert len(stems) == 1, row
    assert dialog.output_edit.text() == str(exp / "sorted_channels")
    plan = dialog.prepare_plan()
    assert plan.ok, plan.problems
    assert plan.mask_roles == {1: "nucleus", 2: "cell"}
    assert sum(1 for r in plan.rows if r.source_mask) == 6
    asked = []
    dialog.ask = lambda title, text: asked.append(title) or True
    dialog._on_apply()
    assert dialog.plan is not None and asked
    # Nothing moved before the screen applies the plan.
    assert (exp / "DAPI" / "field1.tif").is_file()
    result = cs.apply_plan(dialog.plan, log=lambda _t: None)
    dest = Path(result.dest)
    assert sorted(os.listdir(dest / "C02")) == [
        "masks", "plate1_A01_T0001F001L01A01Z01C02.tif",
        "plate1_A02_T0001F001L01A01Z01C02.tif",
        "plate1_A03_T0001F001L01A01Z01C02.tif"]
    assert len(result.merged) == 3
    assert np.load(result.merged[0]).shape == (12, 12, 4)


def test_files_sort_themselves_and_unmatched_cells_are_empty(dialog, tmp_path):
    exp = _tree(tmp_path, masks=False)
    _tif(exp / "GFP" / "field9.tif", shape=(20, 20))
    dialog.add_column("channel")
    dialog.add_column("channel")
    dialog.add_files(0, [exp / "DAPI" / f"field{i}.tif" for i in (3, 1, 2)])
    dialog.add_files(1, [exp / "GFP"])
    assert len(dialog.rows) == 4
    assert dialog._incomplete_rows() == [3]
    assert dialog.rows[3][0] is None
    assert dialog.table.item(3, 0).text() == "—"
    assert dialog.prepare_plan() is None
    dialog._on_apply()
    assert "1 row(s) lack" in dialog.status.text()
    assert dialog.plan is None
    # Removing the odd file makes the table complete.
    dialog.table.item(3, 1).setSelected(True)
    assert dialog._remove_selected() == 1
    assert not dialog._incomplete_rows()


def test_images_of_different_size_are_never_matched(dialog, tmp_path):
    _tif(tmp_path / "a" / "f1.tif", shape=(12, 12))
    _tif(tmp_path / "b" / "f1.tif", shape=(16, 16))
    dialog.add_column("channel")
    dialog.add_column("channel")
    dialog.add_files(0, [tmp_path / "a"])
    dialog.add_files(1, [tmp_path / "b"])
    assert len(dialog.rows) == 2 and len(dialog._incomplete_rows()) == 2


def test_cells_dragged_between_columns_move(dialog, tmp_path):
    folder = _drawn(tmp_path)
    dialog.add_column("channel")
    dialog.add_column("channel")
    dialog.add_files(0, [folder])
    assert len(dialog._column_files(0)) == 6
    cells = [dialog.table.item(r, 0) for r in range(dialog.table.rowCount())
             if dialog.table.item(r, 0).text().startswith("nuc")]
    mime = dialog.table.mimeData(cells)
    assert dialog.table.mimeTypes() == [ofm._CELLS_MIME]
    payload = json.loads(bytes(mime.data(ofm._CELLS_MIME)).decode())
    assert payload["column"] == 0 and len(payload["paths"]) == 3
    assert _drop(dialog.table, mime, _x_of(dialog.table, 1))
    assert sorted(os.path.basename(p) for p in dialog._column_files(1)) == [
        "nuc_img01.tif", "nuc_img02.tif", "nuc_img03.tif"]
    # 600c: several cells move as a block, keeping their shape.
    assert [os.path.basename(r[1]) for r in dialog.rows[:3]] == [
        "nuc_img01.tif", "nuc_img02.tif", "nuc_img03.tif"]
    assert [os.path.basename(r[0]) for r in dialog.rows[:3]] == [
        "cell_img01.tif", "cell_img02.tif", "cell_img03.tif"]


def test_the_new_channel_zone_and_column_buttons(dialog, tmp_path):
    exp = _tree(tmp_path, masks=False)
    assert _drop(dialog.new_zone, _urls(exp / "DAPI"))
    assert _drop(dialog.new_zone, _urls(exp / "GFP"))
    assert [c.kind for c in dialog.columns] == ["channel", "channel"]
    assert len(dialog.rows) == 3
    mask = dialog.add_column("mask")
    assert dialog.columns[mask].role == "cell"
    assert dialog.columns[mask].of_channel == 1
    dialog.remove_column(0)
    assert [c.kind for c in dialog.columns] == ["channel", "mask"]
    assert len(dialog.rows) == 3
    text = QMimeData()
    text.setText("hello")
    assert not _drop(dialog.new_zone, text)
    assert not _drop(dialog.table, text)


def test_a_drop_on_the_empty_table_makes_a_channel(dialog, tmp_path):
    # 600c: the drop hint is the table itself, so it takes the drop.
    exp = _tree(tmp_path)
    assert _drop(dialog.table, _urls(exp / "DAPI"))
    assert dialog.columns[0].kind == "channel"
    assert len(dialog._column_files(0)) == 3


def test_non_images_are_left_out_and_named(dialog, tmp_path):
    note = tmp_path / "notes.txt"
    note.write_text("x")
    dialog.add_column("channel")
    assert dialog.add_files(0, [note]) == []
    assert "notes.txt" in dialog.status.text()


def test_mode_1_source_and_regex_fill_the_table(dialog, tmp_path):
    folder = _drawn(tmp_path)
    dialog.source_edit.setText(str(folder))
    report = dialog.sort_by_regex()
    assert report is not None and report.ok
    assert [c.kind for c in dialog.columns] == ["channel", "channel",
                                                "mask", "mask"]
    assert {c.role for c in dialog.columns if c.kind == "mask"} == {
        "cell", "nucleus"}
    assert len(dialog.rows) == 3 and not dialog._incomplete_rows()
    assert all(key is not None for key in dialog.row_keys)
    plan = dialog.prepare_plan()
    assert plan.ok, plan.problems
    assert plan.dest == str(folder / "sorted_channels")
    assert sum(1 for r in plan.rows if r.source_mask) == 6


def test_mode_1_with_a_convention_from_mask_generation(dialog, tmp_path):
    folder = tmp_path / "plate"
    for well in ("A01", "B02"):
        for channel in (1, 2):
            _tif(folder / f"plate1_{well}_T0001F001L01A01Z01C0{channel}.tif")
    dialog.source_edit.setText(str(folder))
    dialog.metadata_field.set_value("cellvoyager")
    assert "chanID" in dialog._regex_for(cs.list_folder_images(str(folder)))
    report = dialog.sort_by_regex()
    assert report.ok and len(dialog.rows) == 2
    assert dialog.prepare_plan().ok


def test_a_regex_that_leaves_sets_incomplete_blocks_apply(dialog, tmp_path):
    folder = _drawn(tmp_path)
    (folder / "nuc_img04.tif").write_bytes((folder / "nuc_img01.tif").read_bytes())
    dialog.source_edit.setText(str(folder))
    dialog.sort_by_regex(r"(?P<chanID>[a-z]+)_img(?P<fieldID>\d+)\.tif")
    assert dialog._incomplete_rows() == [3]
    dialog._on_apply()
    assert dialog.plan is None
    assert "Change the regex" in dialog.status.text()


def test_auto_regex_goes_into_custom_regex(dialog, tmp_path):
    folder = _drawn(tmp_path)
    dialog.source_edit.setText(str(folder))
    pattern = dialog.auto_regex()
    assert pattern and dialog.custom_edit.text() == pattern
    assert dialog.metadata_field.get_value() == "custom"
    assert len(dialog.rows) == 3


def test_auto_regex_on_dropped_files(dialog, tmp_path):
    exp = _tree(tmp_path, masks=False)
    dialog.add_column("channel")
    dialog.add_column("channel")
    dialog.add_files(0, [exp / "DAPI"])
    dialog.add_files(1, [exp / "GFP"])
    assert dialog.auto_regex()
    assert len(dialog.rows) == 3 and not dialog._incomplete_rows()


def test_auto_regex_says_when_nothing_fits(dialog):
    assert dialog.auto_regex() is None
    assert "no regex" in dialog.status.text()
    assert dialog.sort_by_regex() is None


def test_detect_sets_asks_with_three_examples(dialog, tmp_path):
    folder = _drawn(tmp_path)
    dialog.source_edit.setText(str(folder))
    shown = []
    dialog.confirm_examples = lambda sets: shown.append(sets) or True
    assert dialog.detect_sets()
    assert len(shown[0]) == 3
    assert all(dict(key).keys() == {cs.DETECTED_GROUP}
               for key in dialog.row_keys)
    assert dialog.prepare_plan().ok
    dialog.confirm_examples = lambda sets: False
    before = [list(r) for r in dialog.rows]
    assert not dialog.detect_sets()
    assert dialog.rows == before


def test_detect_sets_needs_two_filled_channels(dialog):
    assert not dialog.detect_sets()
    assert "two channel columns" in dialog.status.text()


def test_the_popup_prefills_channel_folders(qtbot, qt_theme_applied, tmp_path):
    exp = _tree(tmp_path)
    made = ofm.OrganizeForMeasureDialog(
        str(exp), channel_folders=[str(exp / "DAPI"), str(exp / "GFP")])
    qtbot.addWidget(made)
    assert [c.kind for c in made.columns] == ["channel", "mask", "channel",
                                              "mask"]
    assert len(made.rows) == 3 and not made._incomplete_rows()
    assert made.prepare_plan().ok


def test_consolidation_runs_before_the_regex_when_asked(dialog, tmp_path):
    src = tmp_path / "exp"
    for folder in ("nuc", "cell"):
        for i in (1, 2):
            _tif(src / folder / f"f{i}.tif")
    dialog.source_edit.setText(str(src))
    dialog.consolidate_check.setChecked(True)
    dialog.sort_by_regex()
    assert dialog._source() == str(tmp_path / "exp_renamed")
    assert not dialog.consolidate_check.isChecked()
    assert len(dialog.rows) == 2 and not dialog._incomplete_rows()


def test_rgb_images_are_offered_for_conversion(dialog, tmp_path):
    for i in (1, 2):
        _tif(tmp_path / "a" / f"f{i}.tif",
             np.full((12, 12, 3), 50 * i, np.uint8))
        _tif(tmp_path / "b" / f"f{i}.tif")
    dialog.add_column("channel")
    dialog.add_column("channel")
    dialog.add_files(0, [tmp_path / "a"])
    dialog.add_files(1, [tmp_path / "b"])
    assert dialog.prepare_plan().convertible
    asked = []
    dialog.ask = lambda title, text: asked.append(title) or True
    dialog._on_apply()
    assert asked[0] == "Convert RGB images and z-stacks?"
    assert dialog.plan is not None
    assert all(r.convert == "rgb" for r in dialog.plan.rows if r.channel == 1)


def test_two_mask_columns_on_one_channel_are_refused(dialog, tmp_path):
    exp = _tree(tmp_path)
    dialog.add_files(dialog.add_column("channel"), [exp / "DAPI"])
    dialog.add_files(dialog.add_column("channel"), [exp / "GFP"])
    extra = dialog.add_column("mask", "pathogen", 1)
    plan = dialog.prepare_plan()
    assert any("two mask columns" in p for p in plan.problems)
    dialog.remove_column(extra)
    assert dialog.prepare_plan().ok


def test_the_source_field_takes_a_dropped_folder(dialog, tmp_path):
    exp = _tree(tmp_path)
    assert _drop(dialog.source_edit, _urls(exp))
    assert dialog._source() == str(exp)
    assert not _drop(dialog.source_edit, _urls(exp / "DAPI" / "field1.tif"))


def test_the_output_is_kept_once_typed(dialog, tmp_path):
    dialog.source_edit.setText(str(tmp_path))
    assert dialog.output_edit.text() == str(tmp_path / "sorted_channels")
    dialog.output_edit.textEdited.emit("mine")
    dialog.output_edit.setText(str(tmp_path / "mine"))
    dialog.source_edit.setText(str(tmp_path / "x"))
    assert dialog.output_edit.text() == str(tmp_path / "mine")


def test_mime_helpers_ignore_bad_payloads():
    mime = QMimeData()
    assert ofm._mime_cells(mime) == (None, [])
    assert ofm._mime_cells(None) == (None, [])
    mime.setData(ofm._CELLS_MIME, b"not json")
    assert ofm._mime_cells(mime) == (None, [])
    assert ofm._mime_paths(None) == []
    web = QMimeData()
    web.setUrls([QUrl("https://example.org/a.tif")])
    assert ofm._mime_paths(web) == []


# -- "Teach me", on a folder consolidated the way the maintainer's was --------

def _consolidated(root: Path) -> Path:
    """A consolidated copy of ``KO/1.tif, KO/1c.tif, KO/2.tif, KO/2c.tif``.

    The copies are named ``exp_KO.tif``, ``exp_KO_2.tif``... (the ``c`` is
    gone), with Make Masks' masks under the copies' names and the
    consolidation manifest that still has the original names.

    :param root: the parent.
    """
    folder = root / "exp_renamed"
    rows = ["original_path,new_filename,status,error"]
    copies = ["exp_KO.tif", "exp_KO_2.tif", "exp_KO_3.tif", "exp_KO_4.tif"]
    originals = ["1.tif", "1c.tif", "2.tif", "2c.tif"]
    for i, (copy, original) in enumerate(zip(copies, originals)):
        _tif(folder / copy)
        _tif(folder / "masks" / copy, _mask(i + 1))
        rows.append(f"/data/exp/KO/{original},{copy},copied,")
    (folder / "rename_manifest.csv").write_text("\n".join(rows) + "\n",
                                                encoding="utf-8")
    return folder


def _answer_by_original(dialog, asked):
    """Answer "Which channel is this?" from the original name's ``c``.

    :param dialog: the popup.
    :param asked: collects the names asked about.
    """
    def ask(path, _known):
        """One answer.

        :param path: the image shown.
        :param _known: labels so far.
        """
        alias = dialog._alias(path)
        asked.append(alias)
        return ("channel", 2) if cs.split_extension(alias)[0].endswith("c") \
            else ("channel", 1)
    return ask


def test_teach_me_learns_the_channels_and_fills_the_table(dialog, tmp_path):
    folder = _consolidated(tmp_path)
    dialog.source_edit.setText(str(folder))
    asked, markers = [], []
    dialog.ask_label = _answer_by_original(dialog, asked)
    dialog.ask_marker = lambda marker, label: markers.append(marker) or label
    pattern = dialog.teach()
    assert pattern and dialog.custom_edit.text() == pattern
    assert len(asked) == 2 and sorted(markers) == ["", "c"]
    assert len(dialog.rows) == 2 and not dialog._incomplete_rows()
    row = dialog.rows[0]
    assert [os.path.basename(p) for p in row[:2]] == ["exp_KO.tif",
                                                      "exp_KO_2.tif"]
    # Masks came from masks/, one column per channel.
    assert [c.kind for c in dialog.columns] == ["channel", "channel",
                                                "mask", "mask"]
    dialog.columns[3].role = "organelle"
    plan = dialog.prepare_plan()
    assert plan.ok, plan.problems
    assert plan.mask_roles == {1: "cell", 2: "organelle"}
    result = cs.apply_plan(plan, log=lambda _t: None)
    assert len(result.merged) == 2
    assert np.load(result.merged[0]).shape == (12, 12, 4)
    assert os.path.isdir(os.path.join(result.dest, "masks",
                                      "organelle_mask_stack"))


def test_teach_me_regex_is_reused_by_sort_by_regex(dialog, tmp_path):
    folder = _consolidated(tmp_path)
    dialog.source_edit.setText(str(folder))
    dialog.ask_label = _answer_by_original(dialog, [])
    dialog.teach()
    before = [list(r) for r in dialog.rows]
    assert dialog.sort_by_regex().ok
    assert dialog.rows == before


def test_teach_me_can_stop_skip_and_hear_that_a_part_names_the_field(
        dialog, tmp_path):
    folder = _consolidated(tmp_path)
    dialog.source_edit.setText(str(folder))
    answers = iter(["skip", ("channel", 1), "stop"])
    dialog.ask_label = lambda path, known: next(answers)
    assert dialog.teach() is None
    assert len(dialog._skipped) == 1
    # Now "the c only names the field": no channel marker is kept.
    fresh = ofm.OrganizeForMeasureDialog(str(folder))
    fresh.ask_label = _answer_by_original(fresh, [])
    fresh.ask_marker = lambda marker, label: "field"
    assert fresh.teach() is None
    assert fresh.ask_label is not None


def test_teach_me_with_nothing_to_teach(dialog):
    assert dialog.teach() is None
    assert "Nothing to sort" in dialog.status.text()
    assert dialog._ask_label("x.tif", []) == "stop"
    assert dialog._ask_marker("c", ("channel", 2)) == ("channel", 2)
    assert ofm.OrganizeForMeasureDialog._known_labels(
        {"a": ("mask", 1, "cell"), "b": ("channel", 2), "c": "skip"}) == [
        ("channel", 2), ("mask", 1, "cell")]


def test_the_question_box_answers(qtbot, qt_theme_applied, tmp_path):
    image = _tif(tmp_path / "a.tif")
    box = ofm._TeachQuestion(str(image), [("channel", 1)])
    qtbot.addWidget(box)
    box._choose(("channel", 2))
    assert box.answer == ("channel", 2)
    assert box.mask_channel.count() == 1


# -- 600b: the table's views, slots, swaps and × ------------------------------

def test_swap_and_clear_are_pure_and_move_masks():
    rows = [["a1", "b1", "ma1"], ["a2", None, "ma2"]]
    mask_of = {0: 2}
    ofm._swap_cells(rows, mask_of, (0, 0), (1, 0))
    assert rows == [["a2", "b1", "ma2"], ["a1", None, "ma1"]]
    ofm._swap_cells(rows, mask_of, (0, 1), (1, 1))
    assert rows == [["a2", None, "ma2"], ["a1", "b1", "ma1"]]
    ofm._swap_cells(rows, mask_of, (0, 1), (3, 1))
    assert len(rows) == 4 and rows[3] == [None, None, None]
    assert ofm._clear_cell(rows, mask_of, (1, 0)) == "a1"
    assert rows[1] == [None, "b1", None]
    assert ofm._clear_cell(rows, mask_of, (9, 0)) is None


def _two_channels(dialog, tmp_path):
    """Fill two channel columns from ``_tree``.

    :param dialog: the popup.
    :param tmp_path: where the tree goes.
    """
    exp = _tree(tmp_path)
    dialog.add_files(dialog.add_column("channel"), [exp / "DAPI"])
    dialog.add_files(dialog.add_column("channel"), [exp / "GFP"])
    return exp


def _slot_drop(table, source_item, row, column):
    """Drag one table cell onto a slot.

    :param table: the table.
    :param source_item: the dragged item.
    :param row: the slot's row (the row count for a new row).
    :param column: the slot's column.
    """
    mime = table.mimeData([source_item])
    x = _x_of(table, column)
    if row < table.rowCount():
        y = table.rowViewportPosition(row) + table.rowHeight(row) // 2
    else:
        y = table.viewport().height() - 2
    enter = QDragEnterEvent(QPoint(x, y), Qt.MoveAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    table.dragEnterEvent(enter)
    assert enter.isAccepted()
    table.dropEvent(QDropEvent(QPointF(x, y), Qt.MoveAction, mime,
                               Qt.LeftButton, Qt.NoModifier))


def test_dragging_a_cell_onto_an_occupied_slot_swaps_with_masks(dialog,
                                                                tmp_path):
    _two_channels(dialog, tmp_path)
    table = dialog.table
    first, second = dialog.rows[0][0], dialog.rows[1][0]
    mask_first = dialog.rows[0][1]
    _slot_drop(table, table.item(0, 0), 1, 0)
    assert dialog.rows[1][0] == first and dialog.rows[0][0] == second
    assert dialog.rows[1][1] == mask_first


def test_dragging_a_cell_onto_an_empty_slot_moves_it(dialog, tmp_path):
    _two_channels(dialog, tmp_path)
    table = dialog.table
    moved, its_mask = dialog.rows[0][2], dialog.rows[0][3]
    assert dialog._clear_slots([[1, 2]]) == 1
    assert dialog.rows[1][2] is None and dialog.rows[1][3] is None
    _slot_drop(table, table.item(0, 2), 1, 2)
    assert dialog.rows[1][2] == moved and dialog.rows[1][3] == its_mask
    assert dialog.rows[0][2] is None and dialog.rows[0][3] is None


def test_the_close_mark_turns_red_on_hover_and_clears_on_click(
        qtbot, dialog, tmp_path):
    from PySide6.QtCore import QPointF as P
    from PySide6.QtGui import QMouseEvent

    _two_channels(dialog, tmp_path)
    dialog.resize(1000, 700)
    table = dialog.table
    rect = ofm._close_rect(table.visualRect(table.model().index(0, 0)))
    centre = P(rect.center())
    move = QMouseEvent(QEvent.MouseMove, centre, Qt.NoButton, Qt.NoButton,
                       Qt.NoModifier)
    table.mouseMoveEvent(move)
    assert dialog.delegate.hover == (0, 0)
    elsewhere = QMouseEvent(QEvent.MouseMove, P(2, 2), Qt.NoButton,
                            Qt.NoButton, Qt.NoModifier)
    table.mouseMoveEvent(elsewhere)
    assert dialog.delegate.hover is None
    gone = dialog.rows[0][0]
    table.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, centre,
                                      Qt.LeftButton, Qt.LeftButton,
                                      Qt.NoModifier))
    table.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, centre,
                                        Qt.LeftButton, Qt.NoButton,
                                        Qt.NoModifier))
    assert gone not in [p for row in dialog.rows for p in row]
    # Its mask went with it.
    assert len(dialog._column_files(1)) == 2


def test_delete_key_clears_the_selected_cells(dialog, tmp_path):
    from PySide6.QtGui import QKeyEvent

    _two_channels(dialog, tmp_path)
    dialog.table.item(0, 2).setSelected(True)
    gone = dialog.rows[0][2]
    dialog.table.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Delete,
                                         Qt.NoModifier))
    assert gone not in dialog._column_files(2)


def test_views_thumbnails_colour_and_preferences(qtbot, dialog, tmp_path):
    _two_channels(dialog, tmp_path)
    dialog.view_box.setCurrentIndex(dialog.view_box.findData("both"))
    assert dialog.delegate.view == "both"
    assert dialog.color_button.isEnabled()
    qtbot.waitUntil(lambda: len(dialog.delegate.pixmaps) == 12, timeout=10000)
    assert dialog.table.rowHeight(0) == ofm._CELL_THUMB + 8
    dialog._set_text_color("#ffff00", remember=True)
    assert dialog.delegate.text_color.name() == "#ffff00"
    dialog.table.viewport().grab()  # paints every view path without error
    dialog._set_view("image")
    dialog.table.viewport().grab()
    assert ofm._load_view_prefs() == ("image", "#ffff00")
    again = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(again)
    assert again.delegate.view == "image"
    assert again.view_box.currentData() == "image"
    again._set_view("nonsense")
    assert again.delegate.view == "text" and not again.color_button.isEnabled()
    dialog._stop_thumbs()


def test_many_files_and_folders_fill_one_column_sorted(dialog, tmp_path):
    exp = _tree(tmp_path, masks=False)
    dialog.add_column("channel")
    dialog.add_column("channel")
    files = [exp / "GFP" / f"field{i}.tif" for i in (3, 1, 2)]
    assert _drop(dialog.table, _urls(*files), _x_of(dialog.table, 1))
    assert _drop(dialog.table, _urls(exp / "DAPI"), _x_of(dialog.table, 0))
    names = [os.path.basename(r[1]) for r in dialog.rows]
    assert names == ["field1.tif", "field2.tif", "field3.tif"]
    assert not dialog._incomplete_rows()


def test_slot_helpers_ignore_other_drags():
    assert ofm._mime_slots(None) == []
    mime = QMimeData()
    mime.setData(ofm._CELLS_MIME, b"{bad")
    assert ofm._mime_slots(mime) == []
