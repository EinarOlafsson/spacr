"""A folder drop's GUI-thread half is small, counted pieces.

The drop of a 100,000-file plate stalled the GUI thread for 796 ms on a
release runner. These tests hold the pieces that removed it to their
contracts by counting work instead of timing it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt import folder_metadata as fm
from spacr.qt import ingest_preview as ip

pytestmark = pytest.mark.qt


def _old_iter_image_files(root, cap=None):
    seen = 0
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in fm.IMAGE_EXTS and p.is_file():
            yield p
            seen += 1
            if cap is not None and seen >= cap:
                return


@pytest.fixture()
def tree(tmp_path):
    for well in ("B02", "A01", "A10"):
        for field in ("f2", "f1"):
            leaf = tmp_path / "plate" / well / field
            leaf.mkdir(parents=True)
            for name in ("c2.tif", "c1.TIF", ".hidden.tif", "..odd.png",
                         "note.txt", "x.", "noext"):
                (leaf / name).touch()
    (tmp_path / "top.png").touch()
    (tmp_path / "plate" / "dir.tif").mkdir()
    (tmp_path / "plate" / "dir.tif" / "inner.tif").touch()
    try:
        os.symlink(tmp_path / "plate" / "A01", tmp_path / "link_to_well")
        os.symlink(tmp_path / "top.png", tmp_path / "linked.png")
    except (OSError, NotImplementedError):
        pass
    return tmp_path


def test_the_walk_yields_what_rglob_yielded_in_the_same_order(tree):
    walked = list(fm.iter_image_files(tree))
    assert sorted(walked) == sorted(_old_iter_image_files(tree))
    if sys.version_info[:2] != (3, 12):
        # rglob's own order is the interpreter's: 3.10 descends each folder
        # depth-first and 3.14 lists sibling folders the other way round.
        # The walk keeps 3.12's, so only 3.12 can compare the order.
        pytest.skip("rglob visits folders in a different order on this Python")
    assert walked == list(_old_iter_image_files(tree))
    for cap in (1, 5, 13):
        assert (list(fm.iter_image_files(tree, cap=cap))
                == list(_old_iter_image_files(tree, cap=cap)))


def test_the_walk_asks_no_file_for_its_stat(tree, monkeypatch):
    stats = []
    real = os.stat

    def counting(*args, **kwargs):
        stats.append(args[0] if args else None)
        return real(*args, **kwargs)
    monkeypatch.setattr(os, "stat", counting)
    found = list(fm.iter_image_files(tree))
    monkeypatch.undo()
    regular = [p for p in found if not p.is_symlink()]
    assert regular
    assert not {os.fspath(p) for p in regular} & {os.fspath(s) for s in stats}


def test_the_plan_orders_paths_as_path_objects_do(tree):
    files = list(_old_iter_image_files(tree))
    assert sorted(files, key=ip._path_order) == sorted(files)
    names = ["/r/a-b", "/r/a/b", "/r/a b", "/r/A", "/r/a.b", "/r/é", "/r/a/B"]
    paths = [Path(n) for n in names]
    assert sorted(paths, key=ip._path_order) == sorted(paths)
    rows = ip.plan_folder_extraction(tree, limit=4)
    full = ip.plan_folder_extraction(tree, limit=None)
    assert rows == full[:4]


def test_a_late_caption_pass_renders_only_new_setting_help(qtbot, monkeypatch):
    from PySide6.QtWidgets import QLabel, QWidget
    from spacr.qt import i18n
    from spacr.qt.screens import settings_model as sm

    host = QWidget()
    qtbot.addWidget(host)
    for index in range(30):
        label = QLabel(f"setting {index}", host)
        label.setProperty("settingsAppKey", "mask")
        label.setProperty("settingKey", f"key_{index}")
    rendered = []
    real = sm.format_tooltip

    def counting(source, app_key, key, code):
        rendered.append(key)
        return real(source, app_key, key, code)
    monkeypatch.setattr(sm, "format_tooltip", counting)
    i18n.retranslate_widget_tree(host, only_new=True)
    assert len(rendered) == 30
    rendered.clear()
    i18n.retranslate_widget_tree(host, only_new=True)
    assert rendered == []
    late = QLabel("late", host)
    late.setProperty("settingsAppKey", "mask")
    late.setProperty("settingKey", "late_key")
    i18n.retranslate_widget_tree(host, only_new=True)
    assert rendered == ["late_key"]
    rendered.clear()
    i18n.retranslate_widget_tree(host)
    assert len(rendered) == 31


def _rows(n):
    return [{"original_path": f"/d/f{i:03d}.tif", "canonical": f"c{i}.tif",
             "plate": "plate1", "well": "A01", "field": i + 1,
             "channel": 1, "time": 1} for i in range(n)]


def test_a_long_plan_fills_the_table_a_chunk_per_turn(qtbot, monkeypatch):
    from spacr.qt.widgets import metadata_table as mt

    panel = mt.MetadataTablePanel(_rows(10))
    qtbot.addWidget(panel)
    added = []
    real = mt.MetadataTablePanel._append_row

    def counting(self, row):
        added.append(row["field"])
        return real(self, row)
    monkeypatch.setattr(mt.MetadataTablePanel, "_append_row", counting)
    panel._append_later(_rows(130)[10:])
    assert added == []
    turns = []
    while panel._pending:
        before = len(added)
        panel._drain_pending()
        turns.append(len(added) - before)
    assert sum(turns) == 120
    assert max(turns) == mt._ROWS_PER_TURN
    assert len(turns) == -(-120 // mt._ROWS_PER_TURN)
    assert panel._table.rowCount() == 130


def test_reading_the_table_takes_the_rows_still_to_come(qtbot, tmp_path):
    from spacr.qt.widgets import metadata_table as mt

    panel = mt.MetadataTablePanel(_rows(5))
    qtbot.addWidget(panel)
    panel._append_later(_rows(90)[5:])
    assert panel._table.rowCount() == 5
    rows = panel.rows()
    assert len(rows) == 90 and panel._pending == []
    assert [r["field"] for r in rows] == list(range(1, 91))
    full = mt.MetadataTablePanel(_rows(90))
    qtbot.addWidget(full)
    assert full.rows() == rows


def test_the_folder_report_plans_the_rows_the_full_walk_planned(tree):
    from itertools import chain, islice

    from spacr.qt import dnd_handlers as dh

    files = _old_iter_image_files(tree)
    probe = list(islice(files, dh._FOLDER_PROBE))
    template = fm.detect_folder_metadata(tree, files=probe)
    expected = ip.plan_folder_extraction(tree, files=chain(probe, files),
                                         template=template)
    report = dh.scan_folder_structure(tree)
    assert report["labels"] == tuple(template.depth_labels)
    assert report["rows"] == expected and expected
    assert [os.fspath(p) for p in fm.iter_image_files(tree)] == list(
        fm._iter_image_paths(tree))
