"""The website source is the published candidate: its screens, catalogs and media revision."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from coming_soon import COPY, EMBEDDINGS, HELD, OPS, PLACEHOLDERS, release_catalog
from integrate_public_coming_soon import check_preserved

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import docs_media_budget  # noqa: E402

PUBLIC = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials'
CHECKPOINT = Path(__file__).resolve().parents[1] / 'release_candidate'
PROMOTED = {'12_map_barcodes', '21_model_compare', '22_model_zoo', OPS, EMBEDDINGS}
REMAINING = [identity for identity in PLACEHOLDERS if identity not in PROMOTED]
HOST = 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'
# Changed with the catalogs it versions, 2026-09-15 (candidate 8738b_pd).
# app_v2.js and styles.css did not change, so their keys did not either.
CATALOG_KEY = '20260915-8738b_pd'


@pytest.mark.parametrize('filename', CATALOGS)
def test_public_catalog_has_playable_lessons_and_translated_unavailable_screens(filename):
    language = filename.split('_', 1)[1].removesuffix('.json')
    catalog = json.loads((PUBLIC / 'catalog' / filename).read_text())
    held = [lesson for lesson in catalog['lessons'] if lesson.get('status') == 'coming_soon']
    ready = [lesson for lesson in catalog['lessons'] if lesson.get('status') != 'coming_soon']
    assert [lesson['id'] for lesson in held] == REMAINING == ['71_investigate_hit']
    assert len(ready) == 76 and all(lesson['scenes'] for lesson in ready)
    for lesson in held:
        assert (lesson['availability_title'], lesson['description']) == COPY[language]
        assert lesson['scenes'] == []
        assert not {'poster', 'silent', 'example_files'} & lesson.keys()
    # Publication copies the verified candidate; the public tree may not drift from it.
    assert (PUBLIC / 'catalog' / filename).read_bytes() == (CHECKPOINT / 'web/catalog' / filename).read_bytes()


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


def test_public_player_pins_the_verified_media_revision_and_exposes_coming_soon():
    """Narration and 4K come from one immutable, read-back commit, never from main."""
    index = (PUBLIC / 'index.html').read_text()
    receipt = json.loads((CHECKPOINT / 'publication-receipt.json').read_text())
    root = receipt['media_root']
    assert re.fullmatch(re.escape(HOST) + r'[0-9a-f]{40}', root) and root == HOST + receipt['commit']
    assert receipt['tag'] and receipt['readback']['passed'] is True
    assert index.count(f'data-audio-root="{root}"') == 1
    assert index.count(f'data-video4k-root="{root}"') == 1
    assert 'resolve/main' not in index and '../media_host' not in index
    # The build must not ship a second copy of what the page fetches from the host.
    assert docs_media_budget.NARRATION_HOST == root
    assert 'data-production-root="production"' in index
    assert '<h3 id="planned-title">Coming soon</h3>' in index
    assert 'app_v2.js?v=20260911-narration-captions' in index
    assert 'styles.css?v=20260911-coming-soon' in index
    for name in ('lesson_catalog.js', 'module_navigation.js'):
        assert name + '?v=' + CATALOG_KEY in index
    for name in ('lesson_catalog.js', 'module_navigation.js', 'app_v2.js', 'styles.css'):
        assert (PUBLIC / name).read_bytes() == (CHECKPOINT / 'web' / name).read_bytes()
