"""The unavailable screens belong to the actual website, not only staging."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from coming_soon import COPY, HELD, PLACEHOLDERS, release_catalog
from integrate_public_coming_soon import check_preserved

PUBLIC = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials'


@pytest.mark.parametrize('filename', CATALOGS)
def test_public_catalog_has_playable_lessons_and_translated_unavailable_screens(filename):
    language = filename.split('_', 1)[1].removesuffix('.json')
    catalog = json.loads((PUBLIC / 'catalog' / filename).read_text())
    held = [lesson for lesson in catalog['lessons'] if lesson.get('status') == 'coming_soon']
    ready = [lesson for lesson in catalog['lessons'] if lesson.get('status') != 'coming_soon']
    assert [lesson['id'] for lesson in held] == list(PLACEHOLDERS)
    assert len(ready) == 71 and all(lesson['scenes'] for lesson in ready)
    for lesson in held:
        assert (lesson['availability_title'], lesson['description']) == COPY[language]
        assert lesson['scenes'] == []
        assert not {'poster', 'silent', 'example_files'} & lesson.keys()


@pytest.mark.parametrize('field,value', [
    ('scenes', [{'narration': 'Wrong soundtrack'}]),
    ('title', 'Different lesson'), ('example_files', ['changed.zip']),
])
def test_preservation_guard_rejects_changed_ready_content(field, value):
    before = {'lessons': [dict(id=identity, scenes=[{'narration': 'Original'}], title=identity)
                           for identity in ('01_ready', *HELD)]}
    after = release_catalog(before, 'en')
    assert check_preserved(before, after) == 1
    changed = deepcopy(after)
    changed['lessons'][0][field] = value
    with pytest.raises(ValueError, match='Playable content changed'):
        check_preserved(before, changed)


def test_public_player_keeps_public_media_roots_and_exposes_coming_soon():
    index = (PUBLIC / 'index.html').read_text()
    assert 'data-production-root="production"' in index
    assert index.count('https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/main') == 2
    assert '../media_host' not in index
    assert '<h3 id="planned-title">Coming soon</h3>' in index
    for name in ('app_v2.js', 'lesson_catalog.js', 'module_navigation.js', 'styles.css'):
        assert name + '?v=20260911-coming-soon' in index
