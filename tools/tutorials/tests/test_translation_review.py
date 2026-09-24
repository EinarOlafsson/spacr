"""Editorial promotion is source-pinned and cannot overwrite other lessons."""
import hashlib
import copy
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


@pytest.mark.parametrize('explicit', [False, True])
def test_section_heading_remains_translated(project, explicit):
    root, english, retained, review = project
    target = reviewer.read(root / 'catalog/lessons_es.json')
    target['lessons'][0]['section'] = 'Módulos principales'
    reviewer.write(root / 'catalog/lessons_es.json', target)
    if explicit:
        review['section'] = 'Datos'
    reviewer.promote(review, root)
    actual = reviewer.read(root / 'catalog/lessons_es.json')['lessons'][0]
    assert actual['section'] == ('Datos' if explicit else 'Módulos principales')


def new_lesson_reviews(root, english, review):
    source = reviewer.read(root / 'catalog/lessons_en.json')
    reviews = []
    for number in (8, 9):
        lesson = dict(copy.deepcopy(english), id=f'new_{number}', number=number)
        source['lessons'].append(lesson)
        reviews.append(dict(copy.deepcopy(review), lesson=lesson['id'],
                            english_sha256=hashlib.sha256(json.dumps(
                                lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()))
    reviewer.write(root / 'catalog/lessons_en.json', source)
    return reviews


def test_batch_adds_new_lessons_together_without_english_fallback(project):
    root, english, retained, review = project
    before = reviewer.read(root / 'catalog/lessons_es.json')
    reviews = new_lesson_reviews(root, english, review)
    reviewer.promote_many(reviews, root)
    actual = reviewer.read(root / 'catalog/lessons_es.json')['lessons']
    assert actual[:2] == before['lessons']
    assert [lesson['id'] for lesson in actual[2:]] == ['new_8', 'new_9']
    for lesson in actual[2:]:
        assert lesson['scenes'][0]['narration'] == 'Abra Home.'
        assert lesson['scenes'][0]['related_lessons'] == ['mask']


@pytest.mark.parametrize('fault', ['late_stale_source', 'missing_sibling', 'mixed_language', 'duplicate'])
def test_bad_batch_leaves_all_catalog_and_review_files_unchanged(project, fault):
    root, english, retained, review = project
    reviews = new_lesson_reviews(root, english, review)
    if fault == 'late_stale_source':
        reviews[1]['english_sha256'] = 'stale'
    elif fault == 'missing_sibling':
        reviews.pop()
    elif fault == 'mixed_language':
        reviews[1]['language'] = 'de'
    else:
        reviews.append(reviews[0])
    before = {p: p.read_bytes() for p in (root / 'catalog').glob('*.json')}
    with pytest.raises(ValueError):
        reviewer.promote_many(reviews, root)
    assert {p: p.read_bytes() for p in (root / 'catalog').glob('*.json')} == before
    assert not (root / 'production').exists()
