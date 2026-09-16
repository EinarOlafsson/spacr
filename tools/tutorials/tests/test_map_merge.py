"""A Map refresh cannot sweep up unrelated or incomplete lesson drafts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from merge_map_media import merge_catalog


def fixture():
    return {'heading': 'target metadata', 'lessons': [
        {'id': '76_ops', 'number': 76, 'scenes': ['verified OPS']},
        {'id': '12_map_barcodes', 'number': 12, 'scenes': ['old map']},
        {'id': '71_investigate_hit', 'number': 71, 'status': 'coming_soon'}]}


def test_only_map_changes_without_modifying_inputs_or_losing_metadata():
    target, source = fixture(), fixture()
    source['heading'] = 'must not overwrite'
    source['lessons'][0]['scenes'] = ['unreviewed unrelated draft']
    source['lessons'][1]['scenes'] = ['new real Map']
    source['lessons'].reverse()
    original = deepcopy((target, source))
    merged = merge_catalog(target, source)
    assert merged == {'heading': 'target metadata', 'lessons': [
        target['lessons'][0], source['lessons'][1], target['lessons'][2]]}
    assert (target, source) == original


def test_legacy_caption_metadata_can_gain_only_the_existing_map_number():
    target, source = fixture(), fixture()
    target['lessons'][1].pop('number')
    merged = merge_catalog(target, source)
    assert merged['lessons'][1]['number'] == 12
    assert 'number' not in target['lessons'][1]
    source['lessons'][1]['number'] = 78
    with pytest.raises(ValueError):
        merge_catalog(target, source)


@pytest.mark.parametrize('defect', ['missing', 'duplicate_map', 'duplicate_other', 'renumbered', 'held'])
def test_a_successful_merge_then_an_invalid_source_is_rejected(defect):
    target, source = fixture(), fixture()
    assert merge_catalog(target, source) == target
    if defect == 'missing':
        source['lessons'].pop(1)
    elif defect == 'duplicate_map':
        source['lessons'].append(deepcopy(source['lessons'][1]))
    elif defect == 'duplicate_other':
        source['lessons'].append(deepcopy(source['lessons'][0]))
    elif defect == 'renumbered':
        source['lessons'][1]['number'] = 78
    else:
        source['lessons'][1]['status'] = 'coming_soon'
    with pytest.raises(ValueError):
        merge_catalog(target, source)
