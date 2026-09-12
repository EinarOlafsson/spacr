"""Every live route is either a real tutorial or an explicit Coming soon screen."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from build_navigation import build
from coming_soon import COPY, PLACEHOLDERS
from check_completed_matrix import digest
from validate_candidate import validate

ROOT = Path(__file__).resolve().parents[1] / 'release_candidate'


def test_candidate_manifest_and_browser_evidence_match_the_actual_package():
    result = validate(ROOT, require_browser=True)
    assert result['routes'] == 77
    assert result['ready'] == 71
    assert result['coming_soon'] == 6


def test_checkpoint_records_match_actual_committed_files():
    checkpoint = json.loads((ROOT / 'checkpoint.json').read_text())
    assert checkpoint['manifest_sha256'] == digest(ROOT / 'release-manifest.json')
    assert checkpoint['browser_checks_complete'] is True
    records = checkpoint['files']
    assert len(records) == len({record['path'] for record in records})
    for record in records:
        path = ROOT / record['path']
        assert path.stat().st_size == record['bytes'], record['path']
        assert digest(path) == record['sha256'], record['path']


def test_candidate_has_all_routes_without_claiming_placeholders_are_recorded():
    english = json.loads((ROOT / 'web/catalog/lessons_en.json').read_text())
    lessons = english['lessons']
    unavailable = [x for x in lessons if x.get('status') == 'coming_soon']
    ready = [x for x in lessons if x.get('status') != 'coming_soon']
    # 76 -> 77 on 2026-09-11: Embeddings shipped a Home tile after this
    # candidate was verified, so it is carried as an explicit Coming soon
    # route rather than excluded. Playable stays 71 -- a placeholder is
    # not a lesson and must never be counted as one.
    assert len(ready) == 71 and len(lessons) == 77
    assert [x['id'] for x in unavailable] == list(PLACEHOLDERS)
    nav = build(english)
    assert nav['missing_tutorials'] == []
    assert nav['routes']['76_ops']['host_app_key'] == 'mask'
    for lesson in lessons:
        route = nav['routes'].get(lesson['id'], {})
        if route.get('kind') == 'submodule':
            assert lesson['host_app_key'] == route['host_app_key']
    expected = [(x['id'], x.get('app_key'), x.get('host_app_key'), x.get('status'), len(x['scenes'])) for x in lessons]
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        localized = json.loads((ROOT / 'web/catalog' / filename).read_text())['lessons']
        assert [(x['id'], x.get('app_key'), x.get('host_app_key'), x.get('status'), len(x['scenes'])) for x in localized] == expected
        for lesson in localized:
            if lesson['id'] in PLACEHOLDERS:
                assert lesson['availability_title'] == COPY[language][0]
                assert lesson['scenes'] == []
            else:
                assert lesson['scenes'] and all(x['narration'].strip() for x in lesson['scenes'])
    manifest = json.loads((ROOT / 'release-manifest.json').read_text())
    assert manifest['published'] is False and manifest['release_hold'] is True
    videos = {Path(r['path']).parts[2] for r in manifest['files']
              if r['path'].startswith('web/production/') and r['path'].endswith('.mp4')}
    assert videos == {x['id'] for x in ready}
    assert not videos & set(PLACEHOLDERS)
