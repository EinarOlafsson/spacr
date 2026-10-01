"""Extracted metadata and composition remain editable through Graph persistence."""
import copy
import json
import sqlite3

import pandas as pd
import pytest

from spacr.condition_annotations import new_definition
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.graph_spec import GraphSpec

PATTERN = r'^(?P<cell_type>[^_]+)_(?P<replicate>rep\d+)_(?P<timepoint>\d+h)_(?P<drug>.+)\.tif$'
OUTPUTS = ['cell_type', 'replicate', 'timepoint', 'drug', 'condition']


@pytest.fixture
def metadata_screen(qtbot, tmp_path):
    path = tmp_path / 'measurements.db'
    frame = pd.DataFrame({
        'original_filename': ['HeLa_rep2_24h_DMSO.tif', 'U2OS_rep12_48h_Drug-A.tif',
                              'HeLa_rep2_24h_Drug.B.tif', 'unmatched.tif', None],
        'measurement': [3, 8, 5, 1, 2],
    })
    with sqlite3.connect(path) as db:
        frame.to_sql('cells', db, index=False)
    widget = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_path(str(path), 'cells')
    return widget, frame, path


def _definition(screen):
    definition = new_definition(screen._annotation_base_frame, screen._condition_source)
    definition.pop('column')
    definition.pop('conditions')
    definition.update(version=3, columns=[
        dict(column=name, kind='extract', metadata_column='original_filename',
             pattern=PATTERN, group=name) for name in OUTPUTS[:-1]
    ] + [dict(column='condition', kind='template', parts=[
        {'kind': 'column', 'column': 'cell_type'}, {'kind': 'text', 'text': '/'},
        {'kind': 'column', 'column': 'drug'}, {'kind': 'text', 'text': '_'},
        {'kind': 'column', 'column': 'replicate'}, {'kind': 'text', 'text': '_'},
        {'kind': 'column', 'column': 'timepoint'},
    ])])
    return definition


def _assert_values(frame):
    assert frame.cell_type.fillna('').tolist() == ['HeLa', 'U2OS', 'HeLa', '', '']
    assert frame.replicate.fillna('').tolist() == ['rep2', 'rep12', 'rep2', '', '']
    assert frame.timepoint.fillna('').tolist() == ['24h', '48h', '24h', '', '']
    assert frame.drug.fillna('').tolist() == ['DMSO', 'Drug-A', 'Drug.B', '', '']
    assert frame.condition.fillna('').tolist() == [
        'HeLa/DMSO_rep2_24h', 'U2OS/Drug-A_rep12_48h', 'HeLa/Drug.B_rep2_24h', '', '']


def test_extracted_columns_chart_and_csv_keep_the_complete_recipe(metadata_screen, tmp_path):
    screen, original, source = metadata_screen
    definition = _definition(screen)
    screen.apply_condition_definition(definition)
    _assert_values(screen._frame)
    screen.builder.set_spec(GraphSpec(kind='box', x='condition', y='measurement', colour='cell_type'))
    assert screen.builder.canvas.figure().axes
    chart = tmp_path / 'chart.json'
    screen.save_chart(chart)
    exported = tmp_path / 'working.csv'
    screen.export_table(exported)
    _assert_values(pd.read_csv(exported))
    assert json.loads(exported.with_suffix('.csv.conditions.json').read_text())['condition_annotation'] == definition
    screen.load_chart(chart)
    assert screen._condition_definition == definition
    _assert_values(screen._frame)
    with sqlite3.connect(source) as db:
        pd.testing.assert_frame_equal(pd.read_sql_query('SELECT * FROM cells', db), original)


def test_sqlite_reopen_keeps_extraction_and_composition_editable(metadata_screen, qtbot, tmp_path):
    screen, original, source = metadata_screen
    screen.apply_condition_definition(_definition(screen))
    screen.save_annotated_table('annotated')
    reopened = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(reopened)
    reopened.load_path(str(source), 'annotated')
    assert reopened._condition_definition['version'] == 3
    assert list(reopened._annotation_base_frame) == list(original)
    _assert_values(reopened._frame)
    edited = copy.deepcopy(reopened._condition_definition)
    edited['columns'][-1]['parts'] = [
        {'kind': 'text', 'text': 'treatment='}, {'kind': 'column', 'column': 'drug'},
        {'kind': 'text', 'text': ';cell='}, {'kind': 'column', 'column': 'cell_type'},
    ]
    reopened.apply_condition_definition(edited)
    assert reopened._frame.condition.iloc[0] == 'treatment=DMSO;cell=HeLa'
    assert all(name in reopened._frame for name in OUTPUTS)
    reopened.save_annotated_table('revised')
    with sqlite3.connect(source) as db:
        pd.testing.assert_frame_equal(pd.read_sql_query('SELECT * FROM cells', db), original)
        _assert_values(pd.read_sql_query('SELECT * FROM annotated', db))
        assert pd.read_sql_query('SELECT condition FROM revised', db).condition.iloc[0] == 'treatment=DMSO;cell=HeLa'


@pytest.mark.parametrize('problem', ['missing_capture', 'future_dependency', 'invalid_regex'])
def test_invalid_extraction_or_composition_preserves_working_table(metadata_screen, problem):
    screen, _, _ = metadata_screen
    good = _definition(screen)
    screen.apply_condition_definition(good)
    previous = screen._frame
    bad = copy.deepcopy(good)
    if problem == 'missing_capture':
        bad['columns'][0]['group'] = 'absent'
    elif problem == 'future_dependency':
        bad['columns'].insert(0, bad['columns'].pop())
    else:
        bad['columns'][0]['pattern'] = '['
    with pytest.raises(ValueError):
        screen.apply_condition_definition(bad)
    assert screen._frame is previous
    assert screen._condition_definition == good
    _assert_values(screen._frame)
