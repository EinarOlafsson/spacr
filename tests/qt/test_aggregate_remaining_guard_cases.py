"""Targeted viewer, repaint and cache failures from the combined CI report."""
from types import SimpleNamespace
import weakref

import pandas as pd
import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QWidget


def test_binary_editor_values_are_described_without_decoding_them(qapp):
    from spacr.qt.screens.db_browser import PreviewModel

    model = PreviewModel()
    model.set_page(["payload"], [[b"\xff\x00\x80"]])
    assert model.data(model.index(0, 0), Qt.EditRole) == "<3 bytes>"


def test_dead_accessibility_roots_have_no_live_tree():
    from spacr.qt.i18n import _a11y_live_tree

    assert _a11y_live_tree(None) == []


def test_geometry_replay_can_observe_a_show_missed_during_paint(qapp):
    from spacr.qt import gil_priority as gil

    widget = QWidget()
    before = widget.geometry()
    assert widget.isHidden()
    widget.show()
    gil._hub_geometry_after(SimpleNamespace(_object_kinds=frozenset()),
                            [(widget, True, before)])
    assert not widget.isHidden()
    widget.close()


@pytest.mark.parametrize("missing", ["folder", "kernels"])
def test_an_unavailable_kernel_cache_is_never_claimed_ready(missing, tmp_path, monkeypatch):
    from spacr.qt.widgets import ambient

    folder = None if missing == "folder" else tmp_path
    monkeypatch.setattr(ambient, "_numba_cache_directory", lambda: folder)
    monkeypatch.setattr(ambient, "_PACKED_SCATTER", None)
    monkeypatch.setattr(ambient._SATIN_COMPILER, "kernel", None)
    monkeypatch.setattr(ambient, "_warm_packed_scatter", lambda: None)
    monkeypatch.setattr(ambient._SATIN_COMPILER, "_warm", lambda: None)
    assert ambient._prime_kernel_cache() == 1
    assert list(tmp_path.iterdir()) == []


def test_a_late_render_doorbell_cannot_repaint_a_destroyed_owner(qapp):
    from spacr.qt.widgets.ambient import _repaint_if_alive

    calls = []

    class Field(QWidget):
        def repaint(self, *_args):
            calls.append("painted")

    widget = Field()
    owner = weakref.ref(widget)
    shiboken6.delete(widget)
    _repaint_if_alive(owner)
    assert calls == []


def test_shrinking_a_sparse_result_table_keeps_its_plain_item(qapp):
    from spacr.qt.screens.outliers import _fill_table

    table = QTableWidget(2, 2)
    kept = QTableWidgetItem("before")
    table.setItem(0, 0, kept)
    _fill_table(table, pd.DataFrame({"kept": [5]}))
    assert (table.rowCount(), table.columnCount()) == (1, 1)
    assert table.item(0, 0) is kept and kept.text() == "5"
