"""An isolated tutorial merge must not overwrite any other lesson's work."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge_model_media import merge_catalog


def fixture():
    return {'header': 'retain me', 'lessons': [
        {'id': '76_ops', 'text': 'verified OPS'},
        {'id': '21_model_compare', 'text': 'old compare'},
        {'id': '22_model_zoo', 'text': 'old zoo'},
        {'id': 'unrelated', 'text': 'another session'}]}


def test_only_the_two_assigned_entries_change_and_neither_input_is_mutated():
    target = fixture()
    source = deepcopy(target)
    source['header'] = 'must not copy'
    for item in source['lessons']:
        item['text'] = 'changed ' + item['id']
    before = deepcopy((target, source))
    merged = merge_catalog(target, source)
    assert (target, source) == before
    assert merged['header'] == target['header']
    assert [item['id'] for item in merged['lessons']] == [item['id'] for item in target['lessons']]
    assert merged['lessons'][0] == target['lessons'][0]
    assert merged['lessons'][3] == target['lessons'][3]
    assert merged['lessons'][1:3] == source['lessons'][1:3]


@pytest.mark.parametrize('defect', ['missing', 'duplicate'])
def test_merge_refuses_ambiguous_model_identities(defect):
    target, source = fixture(), fixture()
    merge_catalog(target, source)
    if defect == 'missing':
        source['lessons'].pop(1)
    else:
        source['lessons'].append(deepcopy(source['lessons'][1]))
    with pytest.raises(ValueError):
        merge_catalog(target, source)
