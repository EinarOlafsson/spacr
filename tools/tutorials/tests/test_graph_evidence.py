"""Synthetic fixtures verify the evidence checks, not application results."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_evidence import check_brush, check_histogram, check_points


def test_points_preserve_duplicates_and_actual_values():
    expected = [(1, 2), (1, 2), (3, 4)]
    assert check_points(expected, list(reversed(expected))) == 3
    with pytest.raises(ValueError, match='multiplicities'):
        check_points(expected, [(1, 2), (3, 4)])
    with pytest.raises(ValueError, match='coordinates'):
        check_points(expected, [(1, 2), (1, 2), (3, 5)])


def test_histogram_counts_include_rightmost_endpoint():
    assert check_histogram([0, 1, 2, 2], [0, 1, 2], [1, 3]) == [1, 3]
    with pytest.raises(ValueError, match='bar counts'):
        check_histogram([0, 1, 2, 2], [0, 1, 2], [2, 2])
    with pytest.raises(ValueError, match='omit'):
        check_histogram([0, 3], [0, 1, 2], [1, 0])


def test_publication_does_not_prove_handoff():
    assert not check_brush(['a', 'b'], ['b', 'a'], 0, False)['annotation_handoff_works']
    with pytest.raises(ValueError, match='identities'):
        check_brush(['a', 'b'], ['a', 'c'], 0, False)
    with pytest.raises(ValueError, match='changed'):
        check_brush(['a', 'b'], ['a', 'b'], 2, True)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_missing_values_are_not_silently_dropped(bad):
    with pytest.raises(ValueError, match='Nonfinite'):
        check_points([(1, bad)], [(1, bad)])


def _annotation_fixture():
    import pandas as pd
    base = pd.DataFrame({'columnID': ['c1', 'c10', 'c7', None], 'rowID': ['r5', 'r5', 'r12', None]}, index=[4] * 4)
    annotated = base.assign(genotype=['WildType', None, 'mutant', None],
                            replicate=['replicate 1', 'replicate 1', None, None],
                            condition=['WildType_replicate 1', None, None, None])
    def rule(label, metadata, values):
        return dict(name=label, metadata_column=metadata, match_mode='values', include_values=values)
    definition = dict(version=3, columns=[
        dict(column='genotype', kind='rules', conditions=[rule('WildType', 'columnID', ['c1', 'c2', 'c3']),
            rule('mutant', 'columnID', ['c7', 'c4', 'c5', 'c6'])]),
        dict(column='replicate', kind='rules', conditions=[rule('replicate 1', 'rowID', ['r1', 'r4', 'r5', 'r6']),
            rule('replicate 1', 'rowID', ['r7', 'r9', 'r10'])]),
        dict(column='condition', kind='template', parts=[
            dict(kind='column', column='genotype'), dict(kind='text', text='_'),
            dict(kind='column', column='replicate')])])
    return base, annotated, definition


def test_annotation_oracle_preserves_duplicate_index_and_literal_matching():
    from graph_evidence import check_annotation_recipe
    proof = check_annotation_recipe(*_annotation_fixture())
    assert proof['rows_checked'] == 4
    assert proof['combined_missing_rows'] == 3


@pytest.mark.parametrize('corruption', ['output', 'source', 'row_order', 'recipe', 'combine', 'separator', 'legacy_version', 'missing_box'])
def test_annotation_oracle_rejects_incorrect_evidence(corruption):
    from graph_evidence import check_annotation_recipe
    base, annotated, definition = _annotation_fixture()
    if corruption == 'output':
        annotated.iloc[1, annotated.columns.get_loc('genotype')] = 'WildType'
    elif corruption == 'source':
        annotated.iloc[0, 0] = 'c9'
    elif corruption == 'row_order':
        annotated = annotated.iloc[::-1]
    elif corruption == 'recipe':
        definition['columns'][0]['conditions'][0]['include_values'].append('c10')
    elif corruption == 'combine':
        definition['columns'][2]['parts'].reverse()
    elif corruption == 'separator':
        definition['columns'][2]['parts'][1]['text'] = '-'
    elif corruption == 'legacy_version':
        definition['version'] = 2
    else:
        definition['columns'][1]['conditions'].pop()
    with pytest.raises(ValueError):
        check_annotation_recipe(base, annotated, definition)
