"""Observe both the intended Map promotion and rejection of unrelated changes."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_map_preservation import compare_catalogs


def fixture():
    before = {'lessons': [
        {'id': '12_map_barcodes', 'status': 'coming_soon', 'scenes': []},
        {'id': '07_mask', 'section': 'Core', 'scenes': [{'narration': 'retain exactly'}]},
        {'id': '71_investigate_hit', 'status': 'coming_soon', 'scenes': []}]}
    after = deepcopy(before)
    after['lessons'][0] = {'id': '12_map_barcodes', 'scenes': [{'narration': 'new verified map'}]}
    after['lessons'][1]['section'] = 'Módulos principales'
    assert compare_catalogs(before, after) == [
        {'lesson': '07_mask', 'before': 'Core', 'after': 'Módulos principales'}]
    return before, after


@pytest.mark.parametrize('defect', ['unrelated_scene', 'another_hold', 'missing', 'map_still_held'])
def test_a_verified_promotion_then_an_unrelated_change_is_rejected(defect):
    before, after = fixture()
    if defect == 'unrelated_scene':
        after['lessons'][1]['scenes'][0]['narration'] = 'accidentally replaced'
    elif defect == 'another_hold':
        after['lessons'][2].pop('status')
    elif defect == 'missing':
        after['lessons'].pop()
    else:
        after['lessons'][0]['status'] = 'coming_soon'
    with pytest.raises(ValueError):
        compare_catalogs(before, after)
