"""Graph recipes remain editable across sources, exports and physical saves."""
import copy
import json
import sqlite3

import pandas as pd
import pytest
from PySide6.QtWidgets import QDialog

from spacr.condition_annotations import apply_conditions, new_definition
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.graph_spec import GraphSpec

OUTPUTS = ['genotype', 'replicate', 'condition']


def _recipe(screen):
    definition = new_definition(screen._annotation_base_frame, screen._condition_source)
    definition.pop('column')
    definition.pop('conditions')
    definition['version'] = 2

    def exact(name, column, values):
        return dict(name=name, metadata_column=column, match_mode='values',
                    include_values=values, exclude_values=[], manual_rows=[])

    definition['columns'] = [
        dict(column='genotype', kind='rules', conditions=[
            exact('WildType', 'columnID', ['c1', 'c2', 'c3']),
            exact('mutant', 'columnID', ['c4', 'c5', 'c6', 'c7'])]),
        dict(column='replicate', kind='rules', conditions=[
            exact('replicate1', 'rowID', ['r1', 'r4', 'r5', 'r6', 'r7', 'r9', 'r10'])]),
        dict(column='condition', kind='combine', columns=['genotype', 'replicate'], separator='_'),
    ]
    return definition


@pytest.fixture
def screen(qtbot, tmp_path):
    frame = pd.DataFrame({
        'columnID': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c10', None, 'c1'],
        'rowID': ['r1', 'r4', 'r5', 'r6', 'r7', 'r9', 'r10', 'r1', 'r1', None],
        'value': list(range(10)),
    })
    path = tmp_path / 'measurements.db'
    with sqlite3.connect(path) as db:
        frame.to_sql('samples', db, index=False)
        frame.to_sql('other', db, index=False)
    widget = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_path(str(path), 'samples')
    return widget


def _assert_example(frame):
    assert frame.genotype.fillna('').tolist() == ['WildType'] * 3 + ['mutant'] * 4 + ['', '', 'WildType']
    assert frame.replicate.fillna('').tolist() == ['replicate1'] * 9 + ['']
    assert frame.condition.fillna('').tolist() == ['WildType_replicate1'] * 3 + ['mutant_replicate1'] * 4 + ['', '', '']


def test_recipe_chart_export_and_source_cache_roundtrip(screen, tmp_path):
    definition = _recipe(screen)
    screen.apply_condition_definition(definition)
    _assert_example(screen._frame)
    assert ', '.join(OUTPUTS) in screen._source.text()
    screen.builder.set_spec(GraphSpec(kind='box', x='condition', y='value', colour='genotype'))
    assert screen.builder.canvas.figure().axes
    saved = tmp_path / 'recipe-chart.json'
    screen.save_chart(saved)
    assert json.loads(saved.read_text())['condition_annotation'] == definition
    exported = tmp_path / 'working.csv'
    screen.export_table(exported)
    _assert_example(pd.read_csv(exported))
    sidecar = json.loads(exported.with_suffix('.csv.conditions.json').read_text())
    assert sidecar['condition_annotation'] == definition
    source = screen._condition_source['path']
    screen.load_path(source, 'other')
    assert all(column not in screen._frame for column in OUTPUTS)
    screen.load_path(source, 'samples')
    _assert_example(screen._frame)
    screen.load_chart(saved)
    assert screen._condition_definition == definition
    assert screen.builder.spec.x == 'condition'
    _assert_example(screen._frame)
    assert all(column not in screen._annotation_base_frame for column in OUTPUTS)
    with sqlite3.connect(source) as db:
        assert all(column not in pd.read_sql_query('SELECT * FROM samples', db) for column in OUTPUTS)


def test_saved_sqlite_recipe_chart_reloads_editable_base(screen, tmp_path):
    screen.apply_condition_definition(_recipe(screen))
    screen.save_annotated_table('annotated')
    assert screen._table_picker.currentText() == 'annotated'
    assert screen._condition_definition['source']['table'] == 'annotated'
    _assert_example(screen._frame)
    assert list(screen._annotation_base_frame) == ['columnID', 'rowID', 'value']
    saved = tmp_path / 'saved-recipe-chart.json'
    screen.save_chart(saved)
    source = screen._condition_source['path']
    screen.load_path(source, 'other')
    screen.load_chart(saved)
    _assert_example(screen._frame)
    assert screen._condition_definition['version'] == 2
    edited = copy.deepcopy(screen._condition_definition)
    edited['columns'][0]['conditions'][0]['name'] = 'WT'
    edited['columns'][2]['column'] = 'combined'
    screen.apply_condition_definition(edited)
    assert 'condition' not in screen._frame
    assert screen._frame.combined.iloc[0] == 'WT_replicate1'
    screen.save_annotated_table('revised')
    assert screen._frame.combined.iloc[0] == 'WT_replicate1'
    assert screen._condition_definition['columns'][2]['column'] == 'combined'
    with sqlite3.connect(source) as db:
        assert pd.read_sql_query('SELECT condition FROM annotated', db).condition.iloc[0] == 'WildType_replicate1'


def test_failed_recipe_keeps_entire_previous_frame_and_definition(screen):
    definition = _recipe(screen)
    screen.apply_condition_definition(definition)
    previous = screen._frame
    invalid = copy.deepcopy(definition)
    invalid['columns'][1]['conditions'][0].update(match_mode='regex', include='[', exclude='')
    with pytest.raises(ValueError):
        screen.apply_condition_definition(invalid)
    assert screen._frame is previous
    assert screen._condition_definition == definition
    _assert_example(screen._frame)


@pytest.mark.parametrize('switch', ['none', 'frame', 'context'])
def test_cached_recipe_preview_is_installed_only_for_current_source(screen, monkeypatch, switch):
    import spacr.qt.widgets.condition_annotation_dialog as module

    definition = _recipe(screen)
    result = apply_conditions(screen._annotation_base_frame, definition, screen._condition_source)
    disposed = []

    class AcceptedDialog:
        def __init__(self, *args, **kwargs):
            self.definition = definition
            self.result_frame = result

        def exec(self):
            if switch == 'frame':
                screen.set_frame(pd.DataFrame({'different': [1]}))
            elif switch == 'context':
                screen._condition_source = dict(screen._condition_source, table='different')
            return QDialog.Accepted

        def deleteLater(self):
            disposed.append(True)

    monkeypatch.setattr(module, 'ConditionAnnotationDialog', AcceptedDialog)
    screen.open_condition_dialog()
    assert disposed == [True]
    if switch == 'none':
        assert screen._frame is result
        assert screen._condition_definition == definition
    else:
        assert screen._frame is not result
        assert all(column not in screen._frame for column in OUTPUTS)
        assert 'changed while conditions' in screen._source.text()


def test_saved_recipe_with_changed_source_does_not_replace_current_table(screen, tmp_path):
    screen.apply_condition_definition(_recipe(screen))
    saved = tmp_path / 'chart.json'
    screen.save_chart(saved)
    source = screen._condition_source['path']
    screen.load_path(source, 'other')
    previous = screen._frame
    with sqlite3.connect(source) as db:
        db.execute("UPDATE samples SET columnID='c10' WHERE columnID='c1'")
    screen.load_chart(saved)
    assert screen._frame is previous
    assert screen._condition_definition is None
    assert 'changed' in screen._source.text()


@pytest.mark.parametrize('kind', ['csv', 'derived'])
def test_recipe_on_imported_or_merged_source_preserves_context(screen, tmp_path, kind):
    source = screen._condition_source['path']
    if kind == 'csv':
        path = tmp_path / 'imported.csv'
        screen._annotation_base_frame.to_csv(path, index=False)
        screen.load_path(str(path))
    else:
        from spacr.derived_tables import default_definition
        with sqlite3.connect(source) as db:
            pd.DataFrame({'parent': list(range(10)), 'extra': list(range(10))}).to_sql('extra', db, index=False)
        merge = default_definition(source, ['samples', 'extra'], base='samples', name='Combined')
        merge.update(mode='custom', acknowledged=True, base_keys=['value'])
        merge['joins'][0].update(left_keys=['value'], right_keys=['parent'], how='left')
        screen.use_derived_table(merge)
    definition = _recipe(screen)
    for output in definition['columns'][:2]:
        for condition in output['conditions']:
            condition['metadata_column'] = next(
                column for column in screen._annotation_base_frame
                if column.endswith(condition['metadata_column']))
    screen.apply_condition_definition(definition)
    context = copy.deepcopy(screen._condition_source)
    _assert_example(screen._frame)
    saved = tmp_path / f'{kind}-chart.json'
    screen.save_chart(saved)
    screen.load_path(source, 'other')
    screen.load_chart(saved)
    assert screen._condition_source == context
    assert screen._condition_definition == definition
    _assert_example(screen._frame)
