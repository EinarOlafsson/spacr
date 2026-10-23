"""The scoped producer restores metadata without changing retained content."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from project_experiment_design_catalog import project
from pronunciation import spoken_form


def fixture():
    lesson = {'id': '67_experiment_design', 'scenes': [
        {'visual': str(i), 'narration': 'Keep this sentence.'} for i in range(14)]}
    lesson['scenes'][0]['speech_text'] = 'Preserve this other authored override.'
    lesson['scenes'][13] = {'visual': '12_real_export_complete',
                            'narration': 'Use spacr.qt.widgets.plate_layout.'}
    canonical, published = deepcopy(lesson), deepcopy(lesson)
    lesson['scenes'][13]['speech_text'] = spoken_form(lesson['scenes'][13]['narration'], 'en')
    catalog = {'schema': 1, 'lessons': [{'id': 'other', 'speech_text': 'Explicit unrelated.'}, lesson]}
    return catalog, canonical, published


def test_restores_only_redundant_field_without_mutating_inputs():
    catalog, canonical, published = fixture()
    before = deepcopy((catalog, canonical, published))
    result = project(catalog, canonical, published)
    assert result['lessons'][1] == published
    assert result['lessons'][0] == catalog['lessons'][0]
    assert (catalog, canonical, published) == before


def test_explicit_authored_override_is_preserved():
    catalog, canonical, published = fixture()
    for lesson in (catalog['lessons'][1], canonical, published):
        lesson['scenes'][13]['speech_text'] = 'Reviewed explicit pronunciation.'
    assert project(catalog, canonical, published) == catalog


@pytest.mark.parametrize('change', ['speech', 'visual', 'narration', 'duplicate', 'baseline', 'authored'])
def test_rejects_unreviewed_or_ambiguous_changes(change):
    catalog, canonical, published = fixture()
    scene = catalog['lessons'][1]['scenes'][13]
    if change == 'speech':
        scene['speech_text'] = 'Different spoken words.'
    elif change == 'visual':
        scene['visual'] = 'another_scene'
    elif change == 'narration':
        scene['narration'] += ' New words.'
    elif change == 'duplicate':
        catalog['lessons'].append(deepcopy(catalog['lessons'][1]))
    elif change == 'baseline':
        published['scenes'][0]['narration'] += ' Changed.'
    else:
        canonical['scenes'][13]['speech_text'] = 'Intentional override.'
    with pytest.raises(ValueError):
        project(catalog, canonical, published)
