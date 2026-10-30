"""Condition labels cannot silently move to another row or overwrite measurements."""
import copy

import pandas as pd
import pytest

from spacr.condition_annotations import (
    AnnotationError,
    apply_conditions,
    new_definition,
    preview,
    source_context,
    table_identity,
)


@pytest.fixture
def frame():
    return pd.DataFrame({'wellID': ['A1', 'A2', 'B1', 'B1'], 'filename': ['a', 'b', 'c', 'c'],
                         'value': [3., 1., 2., 2.]}, index=[7, 7, 2, 2])


def rule(name='treated', column='wellID', include='', exclude='', manual=()):
    return dict(name=name, metadata_column=column, include=include, exclude=exclude, manual_rows=list(manual))


def test_regex_and_manual_exclusions_are_positional(frame):
    source = source_context('/tmp/input.csv')
    definition = new_definition(frame, source, column='treatment')
    tokens = table_identity(frame)[2]
    assert len(set(tokens)) == len(frame)
    definition['conditions'] = [rule(include='^A', exclude='A2', manual=[tokens[3]])]
    result = apply_conditions(frame, definition, source)
    assert result.treatment.fillna('').tolist() == ['treated', '', '', 'treated']
    assert result.index.tolist() == [7, 7, 2, 2]
    assert 'treatment' not in frame


def test_manual_only_overlap_preview_and_rejection(frame):
    source = source_context()
    definition = new_definition(frame, source)
    tokens = table_identity(frame)[2]
    definition['conditions'] = [rule(manual=tokens[:2]), rule('control', manual=tokens[1:3])]
    report = preview(frame, definition, source)
    assert report.counts == {'treated': 2, 'control': 2}
    assert report.overlaps.tolist() == [1]
    assert report.unmatched == 1
    with pytest.raises(AnnotationError, match='multiple conditions'):
        apply_conditions(frame, definition, source)


@pytest.mark.parametrize('change', ['values', 'order', 'schema', 'source', 'column', 'regex', 'token', 'names'])
def test_refuses_stale_or_invalid_definition_without_mutation(frame, change):
    source = source_context('/tmp/input.csv', 'table')
    definition = new_definition(frame, source)
    definition['conditions'] = [rule(include='^A')]
    before = frame.copy(deep=True)
    altered = frame.copy(deep=True)
    current_source = copy.deepcopy(source)
    if change == 'values':
        altered.iloc[0, 0] = 'Z9'
    if change == 'order':
        altered = altered.iloc[::-1]
    if change == 'schema':
        altered['new'] = 1
    if change == 'source':
        current_source['table'] = 'other'
    if change == 'column':
        definition['column'] = 'wellID'
    if change == 'regex':
        definition['conditions'][0]['include'] = '['
    if change == 'token':
        definition['conditions'][0]['manual_rows'] = ['missing']
    if change == 'names':
        definition['conditions'].append(rule())
    with pytest.raises(AnnotationError):
        apply_conditions(altered, definition, current_source)
    pd.testing.assert_frame_equal(frame, before)


def test_many_arbitrary_condition_names_and_metadata_columns():
    frame = pd.DataFrame({'external metadata': [str(i) for i in range(75)]})
    source = source_context(merge_definition={'tables': ['one', 'two']})
    definition = new_definition(frame, source)
    definition['conditions'] = [rule(f'dose {i}', 'external metadata', f'^{i}$') for i in range(75)]
    assert apply_conditions(frame, definition, source).condition.tolist() == [f'dose {i}' for i in range(75)]
    with pytest.raises(AnnotationError, match='another source'):
        apply_conditions(frame, definition, source_context(merge_definition={'tables': ['other']}))


def test_sqlite_annotations_rebind_nullable_dates_and_preserve_source_and_views(tmp_path):
    """Saving normalized SQLite values keeps editable rules and existing tables safe."""
    import sqlite3
    import numpy as np
    from spacr.condition_annotations import save_annotated_table, saved_table_annotation

    path = tmp_path / "measurements.db"
    source_frame = pd.DataFrame({"wellID": ["A1", "A2", "B1"],
                                 "acquired": pd.to_datetime(["2026-01-02", None, "2026-01-04"]),
                                 "count": pd.Series([1, pd.NA, 3], dtype="Int64"),
                                 "value": [1.0, np.nan, 2.5]})
    with sqlite3.connect(path) as connection:
        source_frame.to_sql("input", connection, index=False)
        connection.execute('CREATE VIEW existing_view AS SELECT * FROM input')
        source_before = pd.read_sql_query('SELECT * FROM input', connection)
    source = source_context(path, "input")
    definition = new_definition(source_frame, source, column="treatment")
    definition["conditions"] = [rule(name="treated", include="^A")]
    working = apply_conditions(source_frame, definition, source)
    working_before = working.copy(deep=True)

    assert save_annotated_table(path, "reviewed", working, definition, source) == "reviewed"

    with sqlite3.connect(path) as connection:
        stored = pd.read_sql_query('SELECT * FROM reviewed', connection)
        pd.testing.assert_frame_equal(pd.read_sql_query('SELECT * FROM input', connection), source_before)
        assert pd.read_sql_query('SELECT * FROM existing_view', connection).shape == source_before.shape
    assert stored.treatment.fillna("").tolist() == ["treated", "treated", ""]
    assert stored.acquired.iloc[0] == "2026-01-02T00:00:00"
    assert pd.isna(stored.acquired.iloc[1]) and pd.isna(stored["count"].iloc[1])
    rebound = saved_table_annotation(path, "reviewed", stored)
    restored = apply_conditions(stored.drop(columns="treatment"), rebound, source_context(path, "reviewed"))
    assert restored.treatment.fillna("").tolist() == stored.treatment.fillna("").tolist()
    before_refusal = path.read_bytes()
    with pytest.raises(AnnotationError, match="table or view already exists"):
        save_annotated_table(path, "existing_view", working, definition, source)
    assert path.read_bytes() == before_refusal
    changed = working.copy(deep=True)
    changed.loc[0, "treatment"] = "different"
    with pytest.raises(AnnotationError, match="working annotations changed"):
        save_annotated_table(path, "changed", changed, definition, source)
    assert path.read_bytes() == before_refusal
    pd.testing.assert_frame_equal(working, working_before)
