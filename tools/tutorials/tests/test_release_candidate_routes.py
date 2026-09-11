"""Every live route is either a real tutorial or an explicit Coming soon screen."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from build_navigation import build
from coming_soon import COPY, PLACEHOLDERS

ROOT = Path(__file__).resolve().parents[1] / 'release_candidate'


def test_candidate_has_all_routes_without_claiming_placeholders_are_recorded():
    english = json.loads((ROOT / 'web/catalog/lessons_en.json').read_text())
    lessons = english['lessons']
    unavailable = [x for x in lessons if x.get('status') == 'coming_soon']
    ready = [x for x in lessons if x.get('status') != 'coming_soon']
    assert len(ready) == 71 and len(lessons) == 76
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
