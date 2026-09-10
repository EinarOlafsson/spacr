"""An intentional refresh must not overwrite a moved-only specialist lesson."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS, audit


@pytest.fixture
def catalogs(tmp_path):
    baseline, stage = tmp_path / 'baseline', tmp_path / 'stage'
    baseline.mkdir()
    (stage / 'catalog').mkdir(parents=True)
    old = {'lessons': [{'id': 'kept', 'number': 1, 'scenes': [{'narration': 'Original'}]},
                       {'id': 'changed', 'number': 2, 'scenes': [{'narration': 'Old'}]}]}
    new = copy.deepcopy(old)
    new['lessons'][1]['scenes'] = [{'narration': 'Refreshed'}, {'narration': 'Second'}]
    for name in CATALOGS:
        (baseline / name).write_text(json.dumps(old))
        (stage / 'catalog' / name).write_text(json.dumps(new))
    return baseline, stage


def change_spanish(stage, edit):
    path = stage / 'catalog/lessons_es.json'
    data = json.loads(path.read_text())
    edit(data['lessons'])
    path.write_text(json.dumps(data))


def test_selected_refresh_and_unchanged_specialist_pass(catalogs):
    folder = catalogs[1] / 'production/changed/audio/en'
    folder.mkdir(parents=True)
    (folder / 'voice.m4a').write_bytes(b'selected replacement is allowed')
    proof = audit(*catalogs, {'changed'})
    assert proof['passed'] and proof['retained_count'] == 1
    assert proof['catalog_count'] == 14 and proof['scene_count'] == 3


def test_changed_unselected_narration_is_rejected(catalogs):
    change_spanish(catalogs[1], lambda lessons: lessons[0]['scenes'][0].update(narration='Lost original'))
    with pytest.raises(ValueError, match='Unselected lesson was modified'):
        audit(*catalogs, {'changed'})


def test_shorter_translation_of_selected_lesson_is_rejected(catalogs):
    change_spanish(catalogs[1], lambda lessons: lessons[1]['scenes'].pop())
    with pytest.raises(ValueError, match='Scene count disagrees'):
        audit(*catalogs, {'changed'})


def test_missing_existing_lesson_is_rejected(catalogs):
    change_spanish(catalogs[1], lambda lessons: lessons.pop(0))
    with pytest.raises(ValueError, match='identities/order disagree'):
        audit(*catalogs, {'changed'})


def test_route_drift_is_rejected(catalogs):
    change_spanish(catalogs[1], lambda lessons: lessons[1].update(app_key='wrong_parent'))
    with pytest.raises(ValueError, match='route disagrees'):
        audit(*catalogs, {'changed'})


def test_replacing_moved_only_media_is_rejected(catalogs):
    folder = catalogs[1] / 'production/kept/audio/en'
    folder.mkdir(parents=True)
    (folder / 'voice.m4a').write_bytes(b'existing replacement, not an absent path')
    with pytest.raises(ValueError, match='replacement media'):
        audit(*catalogs, {'changed'})


def test_visual_only_allows_actual_legacy_caption_shape_without_rewriting(catalogs):
    baseline, stage = catalogs
    for name in CATALOGS:
        if name.startswith('captions_'):
            for path in [baseline / name, stage / 'catalog' / name]:
                value = json.loads(path.read_text())
                value['lessons'][0].pop('number')
                path.write_text(json.dumps(value))
    proof = audit(*catalogs, {'changed', 'kept'}, visual_only={'kept'})
    assert proof['visual_only_lesson_ids'] == ['kept']
    assert proof['retained_count'] == 0
    with pytest.raises(ValueError, match='route disagrees'):
        audit(*catalogs, {'changed', 'kept'})


@pytest.mark.parametrize('field,value', [('narration', 'Changed'), ('app_key', 'wrong')])
def test_visual_only_requires_identical_original_content(catalogs, field, value):
    def edit(lessons):
        target = lessons[0]['scenes'][0] if field == 'narration' else lessons[0]
        target[field] = value
    change_spanish(catalogs[1], edit)
    with pytest.raises(ValueError, match='Visual-only refresh changed'):
        audit(*catalogs, {'changed', 'kept'}, visual_only={'kept'})


def test_visual_only_cannot_name_a_new_or_unselected_lesson(catalogs):
    for identities in [{'missing'}, {'kept'}]:
        with pytest.raises(ValueError, match='must select existing'):
            audit(*catalogs, {'changed'}, visual_only=identities)
