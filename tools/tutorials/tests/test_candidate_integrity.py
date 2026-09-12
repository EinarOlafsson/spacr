"""A positive package plus one changed fact per negative integrity check."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from check_completed_matrix import digest
from validate_candidate import validate


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def candidate(tmp_path):
    lessons = [{'id': 'ready', 'scenes': [{'narration': 'A real example.'}]},
               {'id': 'held', 'scenes': [], 'status': 'coming_soon'}]
    for filename in CATALOGS:
        save(tmp_path / 'web/catalog' / filename, {'lessons': lessons})
    player = tmp_path / 'web/app_v2.js'
    player.write_text('/* unit fixture, not a production tutorial */')
    media = tmp_path / 'media_host/ready/audio/en/test.m4a'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'unit fixture, not playable media')
    files = [{'path': p.relative_to(tmp_path).as_posix(), 'sha256': digest(p),
              'bytes': p.stat().st_size} for p in sorted(tmp_path.rglob('*')) if p.is_file()]
    save(tmp_path / 'release-manifest.json', {'routes': 2, 'ready_lessons': 1,
         'coming_soon': ['held'], 'catalog_languages': len(CATALOGS), 'files': files})
    save(tmp_path / 'checks/candidate-browser-checks.json', {
        'manifest_sha256': digest(tmp_path / 'release-manifest.json'), 'passed': True,
        'ready_playback_cases': [{'lesson': 'ready', 'passed': True}],
        'placeholder_cases': [{'lesson': 'held', 'language': name.split('_', 1)[1][:-5],
                               'passed': True} for name in CATALOGS]})
    save(tmp_path / 'checks/placeholder-mutation-checks.json', {
        'passed': True, 'baseline_before_and_after_passed': True,
        'player_sha256': digest(player), 'mutations': [
            {'guard': 'availability guard', 'observed_red': True},
            {'guard': 'completion guard', 'observed_red': True}]})
    assert validate(tmp_path, include_hosted_media=True, require_browser=True)['routes'] == 2
    return tmp_path


def change_json(path, operation):
    value = json.loads(path.read_text())
    operation(value)
    save(path, value)


@pytest.mark.parametrize('operation', [
    lambda m: m.update(routes=1),
    lambda m: m.update(ready_lessons=2),
    lambda m: m.update(coming_soon=[]),
    lambda m: m.update(catalog_languages=13),
    lambda m: m['files'].append(deepcopy(m['files'][0])),
    lambda m: m['files'][0].update(path='../outside'),
    lambda m: m['files'][0].update(path='/outside'),
    lambda m: m['files'][0].update(path=''),
    lambda m: m['files'][0].update(path='web//app_v2.js'),
    lambda m: m['files'][0].update(sha256='stale'),
    lambda m: m['files'][0].update(bytes=0),
    lambda m: m['files'].pop(),
])
def test_rejects_wrong_inventory_or_record(candidate, operation):
    change_json(candidate / 'release-manifest.json', operation)
    with pytest.raises(ValueError):
        validate(candidate, include_hosted_media=True)


@pytest.mark.parametrize('relative', ['web/app_v2.js', 'media_host/ready/audio/en/test.m4a'])
def test_actual_changed_bytes_rejected(candidate, relative):
    (candidate / relative).write_bytes(b'changed')
    with pytest.raises(ValueError, match='recorded bytes'):
        validate(candidate, include_hosted_media=True)


@pytest.mark.parametrize('relative', ['web/extra.js', 'media_host/extra.m4a'])
def test_unrecorded_file_rejected(candidate, relative):
    (candidate / relative).write_bytes(b'extra')
    with pytest.raises(ValueError, match='Unrecorded'):
        validate(candidate, include_hosted_media=True)


def test_translated_structure_must_match_even_after_rehashing(candidate):
    translated = candidate / 'web/catalog/lessons_fr.json'
    change_json(translated, lambda c: c['lessons'][0]['scenes'].append({'narration': 'Extra.'}))
    def rehash(manifest):
        record = next(r for r in manifest['files'] if r['path'] == 'web/catalog/lessons_fr.json')
        record.update(sha256=digest(translated), bytes=translated.stat().st_size)
    change_json(candidate / 'release-manifest.json', rehash)
    with pytest.raises(ValueError, match='Catalog structure'):
        validate(candidate)


@pytest.mark.parametrize('operation', [
    lambda r: r.update(passed=False),
    lambda r: r.update(manifest_sha256='old'),
    lambda r: r['ready_playback_cases'].clear(),
    lambda r: r['ready_playback_cases'][0].update(passed=False),
    lambda r: r['ready_playback_cases'].append(deepcopy(r['ready_playback_cases'][0])),
    lambda r: r['placeholder_cases'].pop(),
    lambda r: r['placeholder_cases'].__setitem__(-1, deepcopy(r['placeholder_cases'][0])),
    lambda r: r['placeholder_cases'][0].update(passed=False),
])
def test_browser_evidence_must_cover_actual_package(candidate, operation):
    change_json(candidate / 'checks/candidate-browser-checks.json', operation)
    with pytest.raises(ValueError, match='Browser evidence'):
        validate(candidate, require_browser=True)


@pytest.mark.parametrize('operation', [
    lambda r: r.update(passed=False),
    lambda r: r.update(player_sha256='old'),
    lambda r: r.update(baseline_before_and_after_passed=False),
    lambda r: r['mutations'].pop(),
    lambda r: r['mutations'][0].update(observed_red=False),
])
def test_mutation_evidence_must_describe_current_player(candidate, operation):
    change_json(candidate / 'checks/placeholder-mutation-checks.json', operation)
    with pytest.raises(ValueError, match='Mutation evidence'):
        validate(candidate, require_browser=True)
