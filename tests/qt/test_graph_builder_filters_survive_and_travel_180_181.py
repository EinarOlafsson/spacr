"""Graph Builder filters survive an annotation (#180) and travel between tables (#181)."""
import json
import sqlite3

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QMessageBox

from spacr.condition_annotations import new_definition, table_identity
from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.graph_spec import GraphSpec


def _frame(n=60, *, with_gene=True):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        'x': np.arange(n, dtype=float),
        'y': rng.normal(size=n),
        'plate': np.where(np.arange(n) % 2, 'p1', 'p2'),
        'row_name': np.array(list('ABC'))[np.arange(n) % 3],
    })
    if with_gene:
        frame['gene'] = np.array(['g1', 'g2', 'g3', 'g4'])[np.arange(n) % 4]
    return frame


def _db(path, frame):
    with sqlite3.connect(path) as conn:
        frame.to_sql('cell', conn, index=False)
    return str(path)


def _screen(qtbot, tmp_path, frame, name='plate1.db'):
    screen = GraphBuilderScreen(threaded=False, link=LinkedSelection())
    qtbot.addWidget(screen)
    screen.load_path(_db(tmp_path / name, frame), table='cell')
    assert screen._frame is not None, screen._source.text()
    screen.builder.set_spec(GraphSpec(x='x', y='y', point_budget=1000))
    return screen


def _set_filters(screen):
    panel = screen.filters
    for column in ('x', 'plate', 'gene'):
        panel.add_column(column)
    panel._rows['x']._low.setValue(10)
    panel._rows['x']._high.setValue(40)
    for box in panel._rows['plate']._boxes:
        box.setChecked(box.text() == 'p1')
    for box in panel._rows['gene']._boxes:
        box.setChecked(box.text() != 'g3')
    panel.flush()
    return panel.state()


def _comparable(state):
    return [{k: v for k, v in entry.items() if k != 'keep_new_values'}
            for entry in state['filters']]


def test_annotate_selection_keeps_every_existing_filter(qtbot, tmp_path, monkeypatch):
    screen = _screen(qtbot, tmp_path, _frame())
    before = _set_filters(screen)
    described = screen.filters.current_filter().describe()
    canvas = screen.builder.canvas
    canvas.brush(-1, -10, 1000, 10)
    assert canvas.selected_count() > 0
    from spacr.qt.widgets import condition_annotation_dialog as dialogs

    def choose(parent, base, source, definition, positions):
        result = new_definition(base, source)
        tokens = table_identity(base)[2]
        result['conditions'] = [dict(name='picked', metadata_column='x', include='', exclude='',
                                     manual_rows=[tokens[int(i)] for i in positions])]
        return result

    monkeypatch.setattr(dialogs, '_annotate_selection', choose)
    screen._annotate_selected()
    assert 'condition' in screen._frame.columns
    after = screen.filters.state()
    assert _comparable(after) == _comparable(before)
    assert set(screen.filters._rows) == {'x', 'plate', 'gene'}
    assert screen.filters.current_filter().describe() == described
    # The new column is offered, not applied.
    assert 'condition' in screen.filters.available_columns()
    assert 'condition' not in screen.filters._rows


def test_a_filter_on_the_annotation_column_keeps_labels_added_later(qtbot, tmp_path):
    screen = _screen(qtbot, tmp_path, _frame())
    base = screen._annotation_base_frame
    tokens = table_identity(base)[2]
    first = new_definition(base, screen._condition_source)
    first['conditions'] = [dict(name='a', metadata_column='x', include='', exclude='',
                                manual_rows=tokens[:5])]
    screen.apply_condition_definition(first)
    screen.filters.add_column('condition')
    screen.filters.flush()
    second = new_definition(base, screen._condition_source)
    second['conditions'] = first['conditions'] + [
        dict(name='b', metadata_column='x', include='', exclude='', manual_rows=tokens[10:15])]
    screen.apply_condition_definition(second)
    ticked = {b.text(): b.isChecked() for b in screen.filters._rows['condition']._boxes}
    assert ticked == {'a': True, 'b': True}


def test_save_and_load_filters_round_trip_onto_another_database(qtbot, tmp_path):
    screen = _screen(qtbot, tmp_path, _frame())
    before = _set_filters(screen)
    path = tmp_path / 'filters.json'
    screen._save_filters(str(path))
    saved = json.loads(path.read_text())
    assert saved['version'] == 1
    assert saved['saved_from']['table'] == 'cell'
    assert {e['column'] for e in saved['filters']} == {'x', 'plate', 'gene'}
    assert all(e['enabled'] is True for e in saved['filters'])

    other = _screen(qtbot, tmp_path, _frame(), name='plate2.db')
    assert other.filters.state()['filters'] == []
    assert other._load_filters(str(path), warn=False) == []
    assert _comparable(other.filters.state()) == _comparable(before)
    assert '3 filters loaded' in other._source.text()


def test_loading_onto_a_database_without_a_column_skips_and_lists_it(qtbot, tmp_path, monkeypatch):
    screen = _screen(qtbot, tmp_path, _frame())
    _set_filters(screen)
    path = tmp_path / 'filters.json'
    screen._save_filters(str(path))
    state = json.loads(path.read_text())
    state['filters'].append({'kind': 'range', 'column': 'y', 'enabled': False,
                             'low': 0.0, 'high': 1.0})
    path.write_text(json.dumps(state))

    other = _screen(qtbot, tmp_path, _frame(with_gene=False), name='plate2.db')
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning',
                        lambda parent, title, text, *a, **k: warnings.append(text))
    missing = other._load_filters(str(path))
    assert missing == ['gene']
    assert set(other.filters._rows) == {'x', 'plate'}
    assert len(warnings) == 1 and 'gene' in warnings[0]
    assert 'gene' in other._source.text()


def test_load_filters_reports_bad_files_and_no_table(qtbot, tmp_path):
    screen = GraphBuilderScreen(threaded=False, link=LinkedSelection())
    qtbot.addWidget(screen)
    path = tmp_path / 'filters.json'
    path.write_text(json.dumps({'version': 1, 'filters': []}))
    assert screen._load_filters(str(path), warn=False) is None
    assert 'Load a table' in screen._source.text()
    screen = _screen(qtbot, tmp_path, _frame())
    path.write_text(json.dumps({'version': 99, 'filters': []}))
    assert screen._load_filters(str(path), warn=False) is None
    assert 'Could not load' in screen._source.text()
    path.write_text('[1, 2]')
    assert screen._load_filters(str(path), warn=False) is None


def test_a_different_table_still_starts_without_filters(qtbot, tmp_path):
    screen = _screen(qtbot, tmp_path, _frame())
    _set_filters(screen)
    screen.load_path(_db(tmp_path / 'other.db', _frame()), table='cell')
    assert screen.filters.state()['filters'] == []
