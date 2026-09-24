"""Every live route is either a real tutorial or an explicit Coming soon screen."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_staged_catalogs import CATALOGS
from append_staged_lessons import NAV_PREFIX, JS_SUFFIX
from build_navigation import build
from coming_soon import COPY, EMBEDDINGS, OPS, PLACEHOLDERS
from check_completed_matrix import digest
from validate_candidate import validate

ROOT = Path(__file__).resolve().parents[1] / 'release_candidate'
REMAINING_HOLDS = [identity for identity in PLACEHOLDERS
                   if identity not in {EMBEDDINGS, OPS, '21_model_compare', '22_model_zoo', '12_map_barcodes'}]


def test_candidate_manifest_and_browser_evidence_match_the_actual_package():
    result = validate(ROOT, require_browser=True)
    assert result['routes'] == 85
    assert result['ready'] == 84
    assert result['coming_soon'] == 1


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
    # Embeddings and OPS are measured API examples, not fictitious GUI runs.
    # Model Compare/Zoo are likewise their explicitly recorded subsets.
    # Map now includes real search, mapped counts and its explicit API subset.
    # Investigate Hit still must not be counted as a completed tutorial.
    assert len(ready) == 84 and len(lessons) == 85
    assert [x['id'] for x in unavailable] == REMAINING_HOLDS
    embeddings = next(x for x in ready if x['id'] == EMBEDDINGS)
    assert embeddings['app_key'] == 'embeddings'
    assert len(embeddings['scenes']) == 11
    nav = build(english)
    manifest = json.loads((ROOT / 'release-manifest.json').read_text())
    assert nav['missing_tutorials'] == manifest['outstanding_module_tutorials']
    assert nav['missing_tutorials'] == []
    assert nav['routes']['76_ops']['host_app_key'] == 'mask'
    navigation_text = (ROOT / 'web/module_navigation.js').read_text()
    assert navigation_text.startswith(NAV_PREFIX) and navigation_text.endswith(JS_SUFFIX)
    published_nav = json.loads(navigation_text[len(NAV_PREFIX):-len(JS_SUFFIX)])
    assert published_nav['routes'] == nav['routes']
    moved_assays = {'26_invasion', '27_replication'}
    for lesson in lessons:
        route = nav['routes'].get(lesson['id'], {})
        if lesson['id'] in moved_assays:
            # Preserved lesson metadata predates item 495. The player's
            # current breadcrumb follows the shared generated navigation.
            assert lesson.get('host_app_key') is None
            assert route['host_app_key'] == 'toxoplasma'
        elif route.get('kind') == 'submodule':
            assert lesson['host_app_key'] == route['host_app_key']
    expected = [(x['id'], x.get('app_key'), x.get('host_app_key'), x.get('status'), len(x['scenes'])) for x in lessons]
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        localized = json.loads((ROOT / 'web/catalog' / filename).read_text())['lessons']
        assert [(x['id'], x.get('app_key'), x.get('host_app_key'), x.get('status'), len(x['scenes'])) for x in localized] == expected
        for lesson in localized:
            if lesson['id'] in REMAINING_HOLDS:
                assert lesson['availability_title'] == COPY[language][0]
                assert lesson['scenes'] == []
            else:
                assert lesson['scenes'] and all(x['narration'].strip() for x in lesson['scenes'])
    manifest = json.loads((ROOT / 'release-manifest.json').read_text())
    assert manifest['published'] is False and manifest['release_hold'] is True
    assert manifest['narration_tracks'] == 1211
    videos = {Path(r['path']).parts[2] for r in manifest['files']
              if r['path'].startswith('web/production/') and r['path'].endswith('.mp4')}
    assert videos == {x['id'] for x in ready}
    assert not videos & set(REMAINING_HOLDS)


def test_the_hold_is_lifted_only_beside_a_read_back_media_revision():
    """The manifest stays as built (held); publication is recorded next to it.

    Browser evidence hashes release-manifest.json, so flipping its flags would
    orphan that evidence. The lift lives in checkpoint.json and the receipt,
    and it may only name a commit whose every media byte was read back.
    """
    checkpoint = json.loads((ROOT / 'checkpoint.json').read_text())
    receipt = json.loads((ROOT / 'publication-receipt.json').read_text())
    published = json.loads((ROOT / 'published-media-browser-checks.json').read_text())
    manifest_sha = digest(ROOT / 'release-manifest.json')
    assert checkpoint['release_hold'] is False and checkpoint['media_uploaded'] is True
    assert checkpoint['media_revision']['commit'] == receipt['commit']
    assert receipt['manifest_sha256'] == manifest_sha == published['manifest_sha256']
    assert receipt['branch'] != 'main' and receipt['tag']
    readback = receipt['readback']
    assert readback['passed'] is True and not readback['download_failures'] and not readback['metadata_failures']
    assert readback['downloaded_sha256_matched'] == readback['files_expected'] == receipt['media_files']
    assert published['passed'] is True and published['media_root'] == receipt['media_root']
    assert len(published['ready_playback_cases']) == 84
