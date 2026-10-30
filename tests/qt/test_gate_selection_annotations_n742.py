"""Displayed gate populations retain source positions and persistent labels."""
import json
import sqlite3

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QDialog, QFileDialog, QInputDialog, QPushButton

from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate
from spacr.qt.widgets import condition_annotation_dialog as annotations


def _frame(n=100):
    return pd.DataFrame({'object_label': np.arange(n), 'plateID': 'p1', 'rowID': 'A',
                         'columnID': '1', 'fieldID': '1', 'area': np.arange(n),
                         'intensity': np.arange(n) * 2}, index=[5] * n)


def _screen(qtbot, frame=None):
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    if frame is not None:
        screen.set_frame(frame)
    return screen


def test_slider_random_prefix_preserves_identity_and_backing_rows(qtbot):
    source = _frame()
    screen = _screen(qtbot, source)
    screen._point_limit.setValue(7)
    first = screen._display_positions.copy()
    assert len(set(first)) == 7
    assert not np.array_equal(first, np.arange(7))
    screen._point_limit.setValue(19)
    assert np.array_equal(screen._display_positions[:7], first)
    assert list(screen.gates._frame.object_label) == list(source.iloc[screen._display_positions].object_label)
    screen._point_limit.setValue(3)
    assert np.array_equal(screen._display_positions, first[:3])
    pd.testing.assert_frame_equal(screen._frame, source)
    assert screen.gates.canvas.spec.point_budget == 3
    assert '3' in screen._point_count.text() and '100' in screen._point_count.text()
    assert not any(button.text() == 'Annotate…' for button in screen.findChildren(QPushButton))
    assert 'Annotate selection' in [row[0] for row in screen.graph_menu_items()]


def test_database_load_preserves_all_rows_above_display_cap(qtbot, tmp_path):
    path = tmp_path / 'measurements.db'
    with sqlite3.connect(path) as conn:
        _frame(10017).to_sql('cell', conn, index=False)
    screen = _screen(qtbot)
    screen.load_path(str(path), table='cell')
    assert len(screen._frame) == 10017
    assert len(screen.gates._frame) == 10000
    order = screen._display_positions.copy()
    screen._point_limit.setValue(10017)
    assert len(screen.gates._frame) == 10017
    assert np.array_equal(screen._display_positions[:10000], order)


def test_displayed_gate_annotation_only_labels_selected_rows_and_roundtrips(qtbot, monkeypatch, tmp_path):
    path = tmp_path / 'measurements.db'
    original = _frame()
    with sqlite3.connect(path) as conn:
        original.to_sql('cell', conn, index=False)
    screen = _screen(qtbot)
    screen.load_path(str(path), table='cell')
    screen._point_limit.setValue(20)
    screen.gates.set_gates(GateSet().add(ThresholdGate(name='large', column='area', low=40)))
    screen.gates.tree.select('large')
    screen._x.setCurrentText('area')
    screen.set_axis_cutoffs('x', 60, None)
    shown = screen.gates.canvas.population()
    expected = set(shown.loc[shown.area >= 60].index)
    assert screen._point_count.text() == f'Displayed {len(expected)} of 100 points (maximum 20).'
    assert expected and len(expected) < 60

    def accept(dialog):
        dialog.output_column.setEditText('phenotype')
        dialog.label.setEditText('selected')
        dialog._apply()
        return dialog.result()

    monkeypatch.setattr(annotations._SelectionAnnotationDialog, 'exec', accept)
    screen._annotate_selection()
    assert set(np.flatnonzero(screen._frame.phenotype.eq('selected').fillna(False))) == expected
    assert set(screen._display_positions) == set(shown.index)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args, **kwargs: ('cell_labeled', True))
    screen._save_selected_annotation()
    assert screen._table == 'cell_labeled'
    assert screen._condition_definition
    with sqlite3.connect(path) as conn:
        saved = pd.read_sql_query('SELECT * FROM cell_labeled', conn)
        unchanged = pd.read_sql_query('SELECT * FROM cell', conn)
    assert set(np.flatnonzero(saved.phenotype.eq('selected').fillna(False))) == expected
    assert 'phenotype' not in unchanged
    export = tmp_path / 'labels.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(export), ''))
    screen._export_selected_annotation()
    assert len(pd.read_csv(export)) == len(original)
    assert json.loads(export.with_suffix('.csv.conditions.json').read_text())['condition_annotation']


def test_cancel_leaves_annotations_and_source_untouched(qtbot, monkeypatch):
    frame = _frame()
    screen = _screen(qtbot, frame)
    screen.gates.set_gates(GateSet().add(ThresholdGate(name='all', column='area', low=0)))
    screen.gates.tree.select('all')
    monkeypatch.setattr(annotations._SelectionAnnotationDialog, 'exec', lambda self: QDialog.Rejected)
    screen._annotate_selection()
    assert screen._condition_definition is None
    pd.testing.assert_frame_equal(screen._frame, frame)
