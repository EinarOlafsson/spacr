"""Ordered annotation recipes preserve rows, provenance, and legacy definitions."""
import copy
import json
import sqlite3

import pandas as pd
import pytest

from spacr.condition_annotations import (
    AnnotationError,
    annotation_columns,
    apply_conditions,
    new_definition,
    preview,
    save_annotated_table,
    saved_table_annotation,
    source_context,
    table_identity,
)


def values(name, column, include, **kwargs):
    return dict(name=name, metadata_column=column, match_mode='values',
                include_values=include, exclude_values=[], manual_rows=[], **kwargs)


def recipe(frame, source):
    definition = new_definition(frame, source)
    definition.pop('column')
    definition.pop('conditions')
    definition.update(version=2, columns=[
        dict(column='genotype', kind='rules', conditions=[
            values('WildType', 'columnID', ['c1', 'c2', 'c3', 'c1']),
            values('mutant', 'columnID', ['c7', 'c4', 'c5', 'c6'])]),
        dict(column='replicate', kind='rules', conditions=[
            values('replicate1', 'rowID', ['r1', 'r4', 'r5', 'r6']),
            values('replicate1', 'rowID', ['r7', 'r9', 'r10'])]),
        dict(column='condition', kind='combine', columns=['genotype', 'replicate'], separator='_')])
    return definition


@pytest.fixture
def frame():
    return pd.DataFrame({'columnID': ['c1', 'c3', 'c4', 'c7', 'c10', 'c1'],
                         'rowID': ['r1', 'r7', 'r4', 'r10', 'r9', 'r99'],
                         'measurement': [1., 2., 3., 4., 5., 6.]}, index=[9, 9, 4, 4, 0, 0])


def test_user_example_exact_values_repeated_labels_and_combination(frame):
    source = source_context()
    definition = recipe(frame, source)
    original = frame.copy(deep=True)
    report = preview(frame, definition, source)
    assert annotation_columns(definition) == ['genotype', 'replicate', 'condition']
    assert report.column_previews['replicate'].counts == {'replicate1': 5}
    assert report.column_previews['replicate'].rule_counts == [2, 3]
    assert report.column_previews['genotype'].rule_counts == [3, 2]
    assert report.unmatched == 2 and report.overlaps.size == 0
    output = apply_conditions(frame, definition, source)
    assert output.condition.fillna('').tolist() == [
        'WildType_replicate1', 'WildType_replicate1', 'mutant_replicate1',
        'mutant_replicate1', '', '']
    assert output.index.equals(frame.index)
    assert output.attrs['condition_annotation'] == definition
    pd.testing.assert_frame_equal(frame, original)
    definition['columns'][0]['column'] = 'mutated'
    assert output.attrs['condition_annotation']['columns'][0]['column'] == 'genotype'


def test_same_label_overlap_unions_but_different_labels_conflict(frame):
    source = source_context()
    definition = recipe(frame, source)
    rules = definition['columns'][1]['conditions']
    rules[1]['include_values'] += ['r1']
    report = preview(frame, definition, source)
    assert report.column_previews['replicate'].rule_counts == [2, 4]
    assert report.column_previews['replicate'].counts == {'replicate1': 5}
    assert report.overlaps.size == 0
    rules.append(values('replicate2', 'rowID', ['r7']))
    definition['columns'][0]['conditions'].append(values('other', 'columnID', ['c1']))
    report = preview(frame, definition, source)
    assert report.overlaps.tolist() == [0, 1, 5]
    assert report.column_previews['replicate'].overlaps.tolist() == [1]
    with pytest.raises(AnnotationError, match='multiple conditions'):
        apply_conditions(frame, definition, source)


def test_legacy_exact_values_and_manual_exclusion(frame):
    source = source_context()
    definition = new_definition(frame, source)
    rule = values('yes', 'columnID', ['c1'])
    rule['manual_rows'] = [table_identity(frame)[2][1]]
    rule['exclude_values'] = ['c3']
    definition['conditions'] = [rule]
    report = preview(frame, definition, source)
    assert report.rule_counts == [2]
    assert list(report.column_values) == ['condition']
    assert report.values.fillna('').tolist() == ['yes', '', '', '', '', 'yes']
    definition['conditions'].append(copy.deepcopy(rule))
    with pytest.raises(AnnotationError, match='distinct'):
        preview(frame, definition, source)


def test_combination_missing_and_false_values_are_positional():
    frame = pd.DataFrame({'a': [None, pd.NA, '', 0, False, 'nan', 'None', ' '],
                          'b': ['x'] * 8}, index=[1] * 8)
    source = source_context()
    definition = new_definition(frame, source)
    definition.update(version=2, columns=[dict(column='combined', kind='combine',
                                              columns=['a', 'b'], separator='|')])
    output = apply_conditions(frame, definition, source)
    assert output.combined.isna().tolist() == [True, True, True, False, False, False, False, False]
    assert output.combined.iloc[3:].tolist() == ['0|x', 'False|x', 'nan|x', 'None|x', ' |x']


@pytest.mark.parametrize('change', ['duplicate', 'empty', 'overwrite', 'forward', 'self', 'unknown',
                                  'no_components', 'unknown_kind', 'bad_values', 'stale_count'])
def test_invalid_recipes_fail_without_mutation(frame, change):
    source = source_context()
    definition = recipe(frame, source)
    before = frame.copy(deep=True)
    entries = definition['columns']
    if change == 'duplicate':
        entries[1]['column'] = 'genotype'
    elif change == 'empty':
        entries[1]['column'] = ' '
    elif change == 'overwrite':
        entries[1]['column'] = 'rowID'
    elif change in ('forward', 'self', 'unknown'):
        entries.insert(0, dict(column='first', kind='combine', columns=[
            {'forward': 'genotype', 'self': 'first', 'unknown': 'absent'}[change]]))
    elif change == 'no_components':
        entries[-1]['columns'] = []
    elif change == 'unknown_kind':
        entries[-1]['kind'] = 'magic'
    elif change == 'bad_values':
        entries[0]['conditions'][0]['include_values'] = 'c1'
    else:
        definition['row_count'] += 1
    with pytest.raises(AnnotationError):
        apply_conditions(frame, definition, source)
    pd.testing.assert_frame_equal(frame, before)


def test_later_rules_use_earlier_output_but_manual_tokens_stay_base_bound(frame):
    source = source_context()
    definition = recipe(frame, source)
    definition['columns'][1]['conditions'] = [dict(
        name='R', metadata_column='genotype', include='^Wild', exclude='',
        manual_rows=[table_identity(frame)[2][2]])]
    output = apply_conditions(frame, definition, source)
    assert output.replicate.fillna('').tolist() == ['R', 'R', 'R', '', '', 'R']


def test_sqlite_all_columns_and_manual_tokens_rebind(tmp_path, frame):
    path = tmp_path / 'samples.db'
    with sqlite3.connect(path) as db:
        frame.to_sql('samples', db, index=False)
    source = source_context(path, 'samples')
    definition = recipe(frame, source)
    definition['columns'][0]['conditions'][0]['manual_rows'] = [table_identity(frame)[2][4]]
    definition['columns'][1]['conditions'][0]['manual_rows'] = [table_identity(frame)[2][5]]
    output = apply_conditions(frame, definition, source)
    save_annotated_table(path, 'annotated', output, definition, source)
    with sqlite3.connect(path) as db:
        materialized = pd.read_sql_query('SELECT * FROM annotated', db)
        original = pd.read_sql_query('SELECT * FROM samples', db)
        payload = json.loads(db.execute('SELECT payload_json FROM _spacr_condition_annotations').fetchone()[0])
    assert list(original.columns) == list(frame.columns)
    assert payload['original_definition'] == definition
    rebound = saved_table_annotation(path, 'annotated', materialized)
    assert rebound['version'] == 2
    base = materialized.drop(columns=annotation_columns(rebound))
    reopened = apply_conditions(base, rebound, source_context(path, 'annotated'))
    for column in annotation_columns(rebound):
        assert reopened[column].fillna('').tolist() == output[column].fillna('').tolist()
    save_annotated_table(path, 'annotated_again', reopened, rebound, source_context(path, 'annotated'))
    materialized.loc[0, 'replicate'] = 'tampered'
    with pytest.raises(AnnotationError, match='changed'):
        saved_table_annotation(path, 'annotated', materialized)


@pytest.mark.parametrize('column', ['genotype', 'replicate', 'condition'])
def test_sqlite_refuses_any_changed_output_before_writes(tmp_path, frame, column):
    path = tmp_path / 'samples.db'
    with sqlite3.connect(path) as db:
        frame.to_sql('samples', db, index=False)
    source = source_context(path, 'samples')
    definition = recipe(frame, source)
    output = apply_conditions(frame, definition, source)
    output.iloc[0, output.columns.get_loc(column)] = 'tampered'
    with pytest.raises(AnnotationError, match='changed'):
        save_annotated_table(path, 'annotated', output, definition, source)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [('samples',)]


def test_sqlite_mid_save_error_rolls_back_every_output_and_provenance(tmp_path, frame):
    path = tmp_path / 'samples.db'
    with sqlite3.connect(path) as db:
        frame.to_sql('samples', db, index=False)
        db.execute('CREATE TABLE _spacr_condition_annotations (table_name TEXT PRIMARY KEY, payload_json TEXT NOT NULL)')
        db.execute("CREATE TRIGGER fail_receipt BEFORE INSERT ON _spacr_condition_annotations BEGIN SELECT RAISE(ABORT, 'receipt failure'); END")
    source = source_context(path, 'samples')
    definition = recipe(frame, source)
    with pytest.raises(sqlite3.IntegrityError, match='receipt failure'):
        save_annotated_table(path, 'annotated', apply_conditions(frame, definition, source), definition, source)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='annotated'").fetchall() == []
        assert db.execute('SELECT * FROM _spacr_condition_annotations').fetchall() == []
