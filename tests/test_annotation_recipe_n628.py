"""Filename captures, literal predicates and ordered composition retain source rows."""
import copy
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


def recipe(frame, source=None):
    definition = new_definition(frame, source or source_context())
    definition.pop('column')
    definition.pop('conditions')
    definition.update(version=3, columns=[])
    return definition


def criterion(column, operator, value):
    return dict(metadata_column=column, operator=operator, value=value)


def rule(label, *criteria, **kwargs):
    return dict(name=label, criteria=list(criteria), **kwargs)


def extract(column, source='filename', group=1, pattern=r'day(?P<day>\d+)'):
    return dict(column=column, kind='extract', metadata_column=source, pattern=pattern, group=group)


def template(*parts, column='condition'):
    return dict(column=column, kind='template', parts=list(parts))


def col(name):
    return dict(kind='column', column=name)


def text(value):
    return dict(kind='text', text=value)


def example(frame, source=None):
    definition = recipe(frame, source)
    pattern = r'(?P<genotype>WildType|mutant)_day(?P<day>\d+)_r(?P<replicate>\d+)'
    definition['columns'] = [extract(name, pattern=pattern, group=name)
                             for name in ('genotype', 'day', 'replicate')]
    definition['columns'] += [dict(column='treatment', kind='rules', conditions=[
        rule('control', criterion('filename', 'contains', 'control'),
             criterion('filename', 'not_contains', 'excluded')),
        rule('drug', criterion('filename', 'contains', 'drug'))]),
        template(col('genotype'), text('_replicate '), col('replicate'),
                 text('_day'), col('day'), text('_'), col('treatment'))]
    return definition


@pytest.fixture
def frame():
    return pd.DataFrame({'filename': ['WildType_day7_r1_control.tif', 'mutant_day10_r2_drug.tif',
                                      'WildType_day7_r3_control_excluded.tif', None, 'no_match.tif'],
                         'value': [1, 2, 3, 4, 5]}, index=[8, 8, 1, 1, 0])


def test_user_filename_recipe_keeps_intermediates_and_fixed_spaces(frame):
    before = frame.copy(deep=True)
    definition = example(frame)
    report = preview(frame, definition, source_context())
    assert annotation_columns(definition) == ['genotype', 'day', 'replicate', 'treatment', 'condition']
    assert report.column_values['day'].fillna('').tolist() == ['7', '10', '7', '', '']
    assert report.column_previews['treatment'].rule_counts == [1, 1]
    assert report.unmatched == 3 and not len(report.overlaps)
    result = apply_conditions(frame, definition, source_context())
    assert result.condition.fillna('').tolist() == ['WildType_replicate 1_day7_control',
        'mutant_replicate 2_day10_drug', '', '', '']
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result[frame.columns], before)
    assert result.attrs['condition_annotation'] == definition


@pytest.mark.parametrize(('operator', 'operand', 'wanted'), [
    ('contains', '.', [True, False, False, False, False]),
    ('not_contains', '.', [False, True, True, False, True]),
    ('equals', 'c1', [False, True, False, False, False]),
    ('not_equals', 'c1', [True, False, True, False, True]),
    ('starts_with', 'c1', [False, True, True, False, False]),
    ('ends_with', '0', [False, False, True, False, False]),
    ('regex', r'^c\d+$', [False, True, True, False, False]),
    ('not_regex', r'^c\d+$', [True, False, False, False, True]),
    ('equals', '', [False, False, False, False, True]),
])
def test_predicates_are_literal_unless_regex_and_missing_never_matches(operator, operand, wanted):
    base = pd.DataFrame({'filename': ['a.b', 'c1', 'c10', pd.NA, '']})
    definition = recipe(base)
    definition['columns'] = [dict(column='label', kind='rules', conditions=[
        rule('yes', criterion('filename', operator, operand))])]
    assert apply_conditions(base, definition, source_context()).label.notna().tolist() == wanted


@pytest.mark.parametrize('mode', ['contains', 'not_contains', 'equals'])
def test_single_literal_mode_and_case_sensitive_default(mode):
    base = pd.DataFrame({'filename': ['WT', 'wt', pd.NA]})
    definition = recipe(base)
    definition['columns'] = [dict(column='label', kind='rules', conditions=[
        dict(name='yes', metadata_column='filename', match_mode=mode, match_text='WT')])]
    assert apply_conditions(base, definition, source_context()).label.notna().tolist() == (
        [False, True, False] if mode == 'not_contains' else [True, False, False])


def test_any_criteria_manual_union_exclusion_and_same_label_counts(frame):
    definition = recipe(frame)
    first = rule('selected', criterion('filename', 'contains', 'drug'),
                 criterion('value', 'equals', '1'), match='any', metadata_column='filename',
                 exclude='drug', manual_rows=[table_identity(frame)[2][2]])
    second = rule('selected', criterion('filename', 'contains', 'WildType'))
    definition['columns'] = [dict(column='label', kind='rules', conditions=[first, second])]
    report = preview(frame, definition, source_context())
    assert report.rule_counts == [2, 2]
    assert report.counts == {'selected': 2} and not len(report.overlaps)
    second['name'] = 'different'
    assert preview(frame, definition, source_context()).overlaps.tolist() == [0, 2]
    with pytest.raises(AnnotationError, match='multiple conditions'):
        apply_conditions(frame, definition, source_context())


@pytest.mark.parametrize(('group', 'wanted'), [(0, 'day7'), (1, '7'), ('day', '7')])
def test_capture_groups_take_first_match(group, wanted):
    base = pd.DataFrame({'filename': ['day7_day12', 'unmatched', None]})
    definition = recipe(base)
    definition['columns'] = [extract('day', group=group)]
    assert apply_conditions(base, definition, source_context()).day.fillna('').tolist() == [wanted, '', '']


def test_optional_and_empty_capture_groups_are_missing():
    base = pd.DataFrame({'filename': ['sample7', 'sample', 'other', None]})
    definition = recipe(base)
    definition['columns'] = [extract('digits', pattern=r'sample(\d*)')]
    assert apply_conditions(base, definition, source_context()).digits.fillna('').tolist() == ['7', '', '', '']
    definition['columns'][0]['pattern'] = r'sample(\d+)?'
    assert apply_conditions(base, definition, source_context()).digits.fillna('').tolist() == ['7', '', '', '']


def test_template_preserves_literal_text_order_zero_false_and_na():
    base = pd.DataFrame({'a': [0, False, 'nan', 'None', '', pd.NA], 'b': ['x'] * 6}, index=[2] * 6)
    definition = recipe(base)
    definition['columns'] = [template(text('['), col('b'), text(''), text(' / '), col('a'), text(']'))]
    assert apply_conditions(base, definition, source_context()).condition.fillna('').tolist() == [
        '[x / 0]', '[x / False]', '[x / nan]', '[x / None]', '', '']
    definition['columns'] = [template(text(''))]
    assert apply_conditions(base, definition, source_context()).condition.isna().all()


@pytest.mark.parametrize('bad', [
    extract('out', group='absent'), extract('out', group=2), extract('out', group=True),
    extract('out', pattern='['), extract('out', pattern=''), extract('out', source='future'),
    extract('out', source=['filename']),
    template(col('condition')), template(col('future')), template(dict(kind='other', text='x')),
    template(dict(kind='text', text=5)), dict(column='out', kind='template', parts=[]),
    dict(column='label', kind='rules', conditions=[rule('x')]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', 'bad', 'x'))]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion(['filename'], 'equals', 'x'))]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', [], 'x'))]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', 'equals', 'x'), match=[])]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', 'regex', '['))]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', 'equals', 1))]),
    dict(column='label', kind='rules', conditions=[rule('x', criterion('filename', 'equals', 'x'), match='bad')]),
])
def test_invalid_recipe_refuses_without_mutating_source(frame, bad):
    before = frame.copy(deep=True)
    definition = recipe(frame)
    definition['columns'] = [bad]
    with pytest.raises(AnnotationError):
        apply_conditions(frame, definition, source_context())
    pd.testing.assert_frame_equal(frame, before)


def test_source_names_and_forward_dependencies_remain_protected(frame):
    definition = recipe(frame)
    definition['columns'] = [extract('filename')]
    with pytest.raises(AnnotationError, match='source already'):
        apply_conditions(frame, definition, source_context())
    definition['columns'] = [template(col('later')), extract('later')]
    with pytest.raises(AnnotationError, match='earlier'):
        apply_conditions(frame, definition, source_context())
    definition['columns'] = [extract('day'), extract('day')]
    with pytest.raises(AnnotationError, match='distinct'):
        apply_conditions(frame, definition, source_context())


def test_legacy_versions_do_not_silently_activate_new_kinds(frame):
    definition = recipe(frame)
    definition.update(version=2, columns=[extract('day')])
    with pytest.raises(AnnotationError):
        apply_conditions(frame, definition, source_context())
    definition['columns'] = [dict(column='label', kind='rules', conditions=[
        dict(name='manual', metadata_column='filename', include='', criteria=[criterion('filename', 'contains', 'WildType')])])]
    assert apply_conditions(frame, definition, source_context()).label.isna().all()
    assert new_definition(frame, source_context())['version'] == 1


def test_sqlite_roundtrip_rebinds_all_outputs_and_manual_rows(frame, tmp_path):
    database = tmp_path / 'measurements.db'
    with sqlite3.connect(database) as connection:
        frame.to_sql('source', connection, index=False)
        base = pd.read_sql_query('SELECT * FROM source', connection)
    source = source_context(database, 'source')
    definition = example(base, source)
    definition['columns'][3]['conditions'][0]['manual_rows'] = [table_identity(base)[2][2]]
    output = apply_conditions(base, definition, source)
    save_annotated_table(database, 'annotated', output, definition, source)
    with sqlite3.connect(database) as connection:
        saved = pd.read_sql_query('SELECT * FROM annotated', connection)
        original = pd.read_sql_query('SELECT * FROM source', connection)
    pd.testing.assert_frame_equal(original, base)
    reopened = saved_table_annotation(database, 'annotated', saved)
    assert reopened['version'] == 3
    outputs = annotation_columns(reopened)
    assert outputs == ['genotype', 'day', 'replicate', 'treatment', 'condition']
    restored = apply_conditions(saved.drop(columns=outputs), reopened, source_context(database, 'annotated'))
    for column in outputs:
        assert restored[column].fillna('').tolist() == saved[column].fillna('').tolist()
    reopened['columns'][-1]['parts'] = [col('day'), text(': '), col('genotype')]
    edited = apply_conditions(saved.drop(columns=outputs), reopened, source_context(database, 'annotated'))
    assert edited.condition.iloc[0] == '7: WildType'
    broken = output.copy()
    broken.loc[0, 'day'] = 'wrong'
    with pytest.raises(AnnotationError, match='changed'):
        save_annotated_table(database, 'rejected', broken, definition, source)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name='rejected'").fetchall() == []


def test_recipe_source_fingerprint_and_attrs_remain_bound(frame):
    definition = example(frame)
    result = apply_conditions(frame, definition, source_context())
    saved = copy.deepcopy(result.attrs['condition_annotation'])
    definition['columns'][0]['pattern'] = 'changed'
    assert result.attrs['condition_annotation'] == saved
    with pytest.raises(AnnotationError, match='source rows'):
        apply_conditions(frame.iloc[::-1], saved, source_context())


@pytest.mark.parametrize('operator', ['contains', 'not_contains', 'starts_with', 'ends_with', 'regex', 'not_regex'])
def test_empty_search_operand_refuses_instead_of_assigning_every_row(frame, operator):
    definition = recipe(frame)
    definition['columns'] = [dict(column='label', kind='rules', conditions=[
        rule('unintentional', criterion('filename', operator, ''))])]
    with pytest.raises(AnnotationError, match='nonempty text'):
        apply_conditions(frame, definition, source_context())


def test_explicit_regex_match_all_and_empty_equality_are_distinct():
    base = pd.DataFrame({'filename': ['WT', '', None]})
    definition = recipe(base)
    definition['columns'] = [dict(column='label', kind='rules', conditions=[
        rule('yes', criterion('filename', 'regex', '.*'))])]
    assert apply_conditions(base, definition, source_context()).label.notna().tolist() == [True, True, False]
    definition['columns'][0]['conditions'][0]['criteria'][0].update(operator='not_equals', value='')
    assert apply_conditions(base, definition, source_context()).label.notna().tolist() == [True, False, False]
