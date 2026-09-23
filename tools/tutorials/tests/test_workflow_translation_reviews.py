"""Shared workflow translation cannot change data routes or hide stale phrases."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_workflow_translation_reviews as translation


@pytest.fixture
def inputs():
    data = translation.workflow.load()
    bank = dict(language='de', workflow_map_sha256='test-map', records=[
        dict(source=source, source_sha256=translation.text_hash(source),
             translation='Übersetzung: ' + source)
        for source in sorted(translation.required_sources(data))])
    return data, bank


def test_default_generator_keeps_all_current_english_scripts_unchanged():
    data = translation.workflow.load()
    for key in data['tutorials']:
        actual = translation.workflow.lesson_document(data, key)
        path = translation.ROOT / f'tools/tutorials/lessons/{key}.json'
        assert actual == json.loads(path.read_text())


def test_shared_generator_preserves_every_scene_route_and_literal_identifier(inputs):
    data, bank = inputs
    reviews = translation.compose_reviews(bank, data, 'test-map')
    assert [len(review['scenes']) for review in reviews.values()] == [79, 18]
    reference = reviews['79_module_inputs_outputs']
    prose = ' '.join(reference['scenes'])
    for module in data['modules'].values():
        assert module['name'] in prose
        if module.get('api_entry'):
            assert module['api_entry'] in prose
    for artifact in data['artifacts'].values():
        for literal in artifact['tables'] + artifact['columns']:
            assert literal in prose
    assert all(text.startswith('Übersetzung: ') for text in reference['scenes'])
    english = translation.workflow.lesson_document(data, reference['lesson'])
    assert reference['english_sha256'] == translation.canonical_hash(english)


def test_file_format_plural_can_follow_target_grammar_without_changing_format(inputs):
    data, bank = inputs
    record = next(row for row in bank['records'] if 'label TIFFs' in row['source'])
    record['translation'] = record['translation'].replace('TIFFs', 'TIFF')
    translation.reviewed_phrases(bank, data, 'test-map')
    record['translation'] = record['translation'].replace('TIFF', 'PNG')
    with pytest.raises(ValueError, match='technical token'):
        translation.reviewed_phrases(bank, data, 'test-map')


@pytest.mark.parametrize('fault,match', [
    ('map', 'map changed'), ('missing', 'Missing'), ('hash', 'source hash'),
    ('duplicate', 'Duplicate'), ('empty', 'Empty'), ('placeholder', 'placeholders'),
    ('token', 'technical token'), ('language', 'language')])
def test_stale_incomplete_or_identifier_changing_review_is_rejected(inputs, fault, match):
    data, original = inputs
    bank = copy.deepcopy(original)
    if fault == 'map':
        bank['workflow_map_sha256'] = 'old-map'
    elif fault == 'missing':
        bank['records'].pop()
    elif fault == 'hash':
        bank['records'][0]['source_sha256'] = 'old-hash'
    elif fault == 'duplicate':
        bank['records'].append(bank['records'][0])
    elif fault == 'empty':
        bank['records'][0]['translation'] = ''
    elif fault == 'placeholder':
        record = next(r for r in bank['records'] if '{api}' in r['source'])
        record['translation'] = record['translation'].replace('{api}', '{function}')
    elif fault == 'token':
        record = next(r for r in bank['records'] if 'max_workers' in r['source'])
        record['translation'] = record['translation'].replace('max_workers', 'max_processes')
    else:
        bank['language'] = 'unknown'
    with pytest.raises(ValueError, match=match):
        translation.compose_reviews(bank, data, 'test-map')
