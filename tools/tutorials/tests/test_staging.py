"""Small, offline tests for the isolated tutorial refresh staging path."""
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / 'stage_lesson.py'
spec = importlib.util.spec_from_file_location('tutorial_stage', MODULE)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'refresh'
    capture = root / 'captures/home'
    capture.mkdir(parents=True)
    image = capture / 'home.png'
    Image.new('RGB', (3840, 2160)).save(image)
    import hashlib
    frames = {'home': {'image': 'home.png',
              'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
              'buttons': [
                  {'name': 'sidebar', 'nav_key': 'mask', 'rect': [0, 0, 10, 10]},
                  {'name': 'tile', 'module_key': 'mask', 'rect': [50, 20, 100, 80]},
              ]}}
    stage.write(capture / 'frames.json', frames)
    stage.write(capture / 'provenance.json', {'completed_capture': True, 'commit': 'fixture'})
    original = {'id': 'home', 'number': 1, 'scenes': [{'narration': 'Old.'}]}
    retained = {'id': 'mask', 'number': 2, 'scenes': [{'narration': 'Retained.'}]}
    stage.write(root / 'catalog/lessons_en.json', {'lessons': [original, retained]})
    changed = {'id': 'home', 'number': 1, 'scenes': [
        {'narration': 'New.', 'visual': 'home', 'focus_modules': ['mask'],
         'related_lessons': ['mask']}]}
    lesson = tmp_path / 'home.json'
    stage.write(lesson, changed)
    return root, capture, lesson, changed, retained


def test_stages_only_selected_lesson_with_measured_tile_geometry(project):
    root, capture, lesson, changed, retained = project
    stage.stage_lesson(lesson, 'home', root)
    result = stage.read(root / 'catalog/lessons_en.json')
    assert result['lessons'] == [changed, retained]
    visual = stage.read(root / 'production/home/visual.json')
    assert visual['scenes'][0]['focus'] == [50, 20, 100, 80]
    assert visual['scenes'][0]['pointer'] is False
    status = stage.read(root / 'production/home/refresh-status.json')
    assert status['status'] == 'english_staged_not_publishable'
    assert status['translation_review_complete'] is False


@pytest.mark.parametrize('defect', ['failed_capture', 'changed_image', 'missing_link',
                                   'missing_control', 'changed_number', 'bad_focus'])
def test_rejects_invalid_evidence_before_writing_any_catalog(project, defect):
    root, capture, lesson, changed, retained = project
    before = (root / 'catalog/lessons_en.json').read_bytes()
    if defect == 'failed_capture':
        stage.write(capture / 'provenance.json', {'completed_capture': False})
    elif defect == 'changed_image':
        Image.new('RGB', (3840, 2160), 'red').save(capture / 'home.png')
    elif defect == 'missing_link':
        changed['scenes'][0]['related_lessons'] = ['not-a-lesson']
    elif defect == 'missing_control':
        changed['scenes'][0]['focus_modules'] = ['not-a-module']
    elif defect == 'changed_number':
        changed['number'] = 77
    elif defect == 'bad_focus':
        changed['scenes'][0].pop('focus_modules')
        changed['scenes'][0]['focus'] = [3839, 0, 20, 20]
    stage.write(lesson, changed)
    with pytest.raises(ValueError):
        stage.stage_lesson(lesson, 'home', root)
    assert (root / 'catalog/lessons_en.json').read_bytes() == before
    assert not (root / 'production').exists()
