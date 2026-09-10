"""Editorial promotion is source-pinned and cannot overwrite other lessons."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('translation_review', TOOLS / 'apply_translation_review.py')
reviewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reviewer)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'refresh'
    english = {'id': 'home', 'number': 5, 'scenes': [
        {'narration': 'Open Home.', 'visual': 'home', 'related_lessons': ['mask']}]}
    retained = {'id': 'mask', 'number': 7, 'scenes': [{'narration': 'Preserved translation.'}]}
    reviewer.write(root / 'catalog/lessons_en.json', {'lessons': [english, retained]})
    for filename in ('lessons_es.json', 'captions_de.json'):
        reviewer.write(root / 'catalog' / filename,
                       {'language': filename, 'lessons': [dict(english, title='Old'), retained]})
    review = {'lesson': 'home', 'language': 'es', 'title': 'Inicio',
              'description': 'Descripción', 'objectives': ['Objetivo'], 'prerequisite': 'Instale spaCR.',
              'scenes': ['Abra Home.'], 'review': {'native_speaker_signoff': False},
              'english_sha256': hashlib.sha256(json.dumps(
                  english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    return root, english, retained, review


@pytest.mark.parametrize('language,filename', [('es', 'lessons_es.json'), ('de', 'captions_de.json')])
def test_promotes_only_reviewed_lesson_and_preserves_scene_links(project, language, filename):
    root, english, retained, review = project
    review['language'] = language
    reviewer.promote(review, root)
    result = reviewer.read(root / 'catalog' / filename)
    assert result['lessons'][0]['title'] == 'Inicio'
    assert result['lessons'][0]['scenes'][0]['narration'] == 'Abra Home.'
    assert result['lessons'][0]['scenes'][0]['related_lessons'] == ['mask']
    assert ('speech_text' in result['lessons'][0]['scenes'][0]) is (language == 'es')
    assert result['lessons'][1] == retained
    assert reviewer.read(root / 'production/home' / f'review.{language}.json') == review
    assert reviewer.read(root / 'catalog/lessons_en.json')['lessons'][0] == english


@pytest.mark.parametrize('defect', ['changed_source', 'missing_scene', 'blank_scene', 'invalid_language'])
def test_rejects_invalid_review_before_overwriting_an_existing_catalog(project, defect):
    root, english, retained, review = project
    before = (root / 'catalog/lessons_es.json').read_bytes()
    if defect == 'changed_source':
        review['english_sha256'] = '0' * 64
    elif defect == 'missing_scene':
        review['scenes'] = []
    elif defect == 'blank_scene':
        review['scenes'] = ['  ']
    elif defect == 'invalid_language':
        review['language'] = '../unsupported'
    with pytest.raises(ValueError):
        reviewer.promote(review, root)
    assert (root / 'catalog/lessons_es.json').read_bytes() == before
    assert not (root / 'production').exists()
