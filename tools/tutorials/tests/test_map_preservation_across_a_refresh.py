"""Map survives a pair of candidates in which another lesson was refreshed.

The Map-only proof (``check_map_preservation.check``) compares a pair that
differs by the Map promotion alone. That pair no longer exists: Map's
evidence names ``release-candidate-zveusu2j`` and the approved candidate is
``release-candidate-8738b_pd``, with a reviewed 07_mask refresh between them.

``check_refresh`` makes the narrower claim that is true of that pair, and
these tests are about the line between the two: it must still refuse a
candidate in which Map moved, in which a lesson nobody named moved, or in
which a lesson that WAS named did not move. The last one is the way a proof
like this rots -- the allow-list outlives the change it was written for --
so it is asserted rather than assumed.

The candidate trees here are synthetic and tiny. ``validate`` is stubbed:
it reads every byte of both candidates, it has its own tests, and what is
under test is the comparison, not the inventory.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_map_preservation as preservation
from audit_staged_catalogs import CATALOGS
from barcode_promotion import IDENTITY

OTHER = '07_mask'
UNNAMED = '13_regression'


def lesson(identity, narration):
    return {'id': identity, 'app_key': identity.split('_', 1)[1],
            'scenes': [{'narration': narration}]}


def frozen(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('"use strict";\nwindow.X = Object.freeze('
                    + json.dumps(payload, ensure_ascii=False) + ');\n',
                    encoding='utf-8')


def build(root, *, narrations, media, commit='c0ffee'):
    """Write a candidate tree: fourteen catalogs, both indexes, a manifest."""
    lessons = [lesson(identity, text) for identity, text in narrations.items()]
    catalog = {'schema': 1, 'lessons': lessons}
    for name in CATALOGS:
        path = root / 'web/catalog' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(catalog, ensure_ascii=False), encoding='utf-8')
    frozen(root / 'web/lesson_catalog.js', catalog)
    frozen(root / 'web/module_navigation.js', {'schema': 1, 'source_commit': commit,
                                               'labels': {'en': ['Main modules']}})
    files = [{'path': 'web/catalog/' + name, 'sha256': 'catalog-' + name}
             for name in CATALOGS]
    files.append({'path': 'web/lesson_catalog.js', 'sha256': 'index'})
    files.append({'path': 'web/module_navigation.js', 'sha256': 'navigation-' + commit})
    for identity, digest in media.items():
        files.append({'path': 'media_host/' + identity + '/audio/en/af_heart.m4a',
                      'sha256': digest})
        files.append({'path': 'web/production/' + identity + '/scenes.json',
                      'sha256': digest + '-scenes'})
    root.mkdir(parents=True, exist_ok=True)
    (root / 'release-manifest.json').write_text(
        json.dumps({'ready_lessons': len(lessons), 'routes': len(lessons),
                    'files': files}), encoding='utf-8')
    return root


@pytest.fixture(autouse=True)
def no_inventory_pass(monkeypatch):
    monkeypatch.setattr(preservation, 'validate', lambda *a, **k: None)


@pytest.fixture
def pair(tmp_path):
    """A prior and a candidate that differ by 07_mask and by nothing else."""
    prior = build(tmp_path / 'prior',
                  narrations={IDENTITY: 'Map', OTHER: 'Old mask', UNNAMED: 'Regression'},
                  media={IDENTITY: 'map', OTHER: 'mask-old', UNNAMED: 'regression'})
    candidate = build(tmp_path / 'candidate',
                      narrations={IDENTITY: 'Map', OTHER: 'New mask', UNNAMED: 'Regression'},
                      media={IDENTITY: 'map', OTHER: 'mask-new', UNNAMED: 'regression'},
                      commit='deadbee')
    return prior, candidate


def test_the_refresh_proof_passes_and_counts_what_it_kept(pair):
    prior, candidate = pair
    result = preservation.check_refresh(prior, candidate, [OTHER])

    assert result['passed'] is True
    assert result['mode'] == 'refresh'
    assert result['refreshed_lessons'] == [OTHER]
    assert result['map_media_files'] == 2
    assert result['retained_media_files'] == 4
    assert result['unchanged_lessons_per_catalog'] == 2
    assert result['catalogs_checked'] == len(CATALOGS)
    assert result['published'] is False
    written = json.loads((candidate / 'checks/map-preservation-checks.json').read_text())
    assert written == result


def test_the_commit_the_navigation_was_built_from_is_excused_by_name(pair):
    prior, candidate = pair
    result = preservation.check_refresh(prior, candidate, [OTHER])
    assert result['navigation_keys_excused'] == {
        'source_commit': {'before': 'c0ffee', 'after': 'deadbee'}}


def test_a_changed_map_lesson_is_refused(tmp_path):
    prior = build(tmp_path / 'prior', narrations={IDENTITY: 'Map', OTHER: 'Mask'},
                  media={IDENTITY: 'map', OTHER: 'mask'})
    candidate = build(tmp_path / 'candidate',
                      narrations={IDENTITY: 'Map, reworded', OTHER: 'Mask'},
                      media={IDENTITY: 'map', OTHER: 'mask'})
    with pytest.raises(ValueError, match='Map itself changed'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_changed_map_media_is_refused_before_any_catalog_is_read(tmp_path):
    prior = build(tmp_path / 'prior', narrations={IDENTITY: 'Map', OTHER: 'Mask'},
                  media={IDENTITY: 'map', OTHER: 'mask'})
    candidate = build(tmp_path / 'candidate', narrations={IDENTITY: 'Map', OTHER: 'Mask'},
                      media={IDENTITY: 'map-rerendered', OTHER: 'mask'})
    with pytest.raises(ValueError, match='A Map media file changed'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_a_lesson_that_changed_without_being_named_is_refused(pair, tmp_path):
    prior, _ = pair
    candidate = build(tmp_path / 'wider',
                      narrations={IDENTITY: 'Map', OTHER: 'New mask',
                                  UNNAMED: 'Regression, reworded'},
                      media={IDENTITY: 'map', OTHER: 'mask-new', UNNAMED: 'regression'})
    with pytest.raises(ValueError, match='did not name'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_a_named_lesson_that_did_not_move_at_all_is_refused(pair):
    prior, candidate = pair
    with pytest.raises(ValueError, match='named lessons that did not move'):
        preservation.check_refresh(prior, candidate, [OTHER, UNNAMED])


def test_a_lesson_refreshed_in_media_only_is_accepted_and_reported(tmp_path):
    """13_regression's shape: new frames, narration untouched."""
    prior = build(tmp_path / 'prior',
                  narrations={IDENTITY: 'Map', OTHER: 'Mask', UNNAMED: 'Regression'},
                  media={IDENTITY: 'map', OTHER: 'mask', UNNAMED: 'regression-old'})
    candidate = build(tmp_path / 'candidate',
                      narrations={IDENTITY: 'Map', OTHER: 'Mask', UNNAMED: 'Regression'},
                      media={IDENTITY: 'map', OTHER: 'mask', UNNAMED: 'regression-new'})
    result = preservation.check_refresh(prior, candidate, [UNNAMED])
    assert result['refreshed_in_prose'] == []
    assert result['refreshed_in_media_only'] == [UNNAMED]
    assert result['map_media_files'] == 2


def test_media_that_belongs_to_no_named_lesson_is_refused(pair, tmp_path):
    prior, candidate = pair
    manifest_path = candidate / 'release-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for record in manifest['files']:
        if record['path'] == 'media_host/13_regression/audio/en/af_heart.m4a':
            record['sha256'] = 'regression-rerendered'
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='An unexplained file changed'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_map_cannot_be_declared_refreshed(pair):
    prior, candidate = pair
    with pytest.raises(ValueError, match='both preserved and refreshed'):
        preservation.check_refresh(prior, candidate, [IDENTITY])


def test_an_empty_declaration_is_sent_back_to_the_promotion_proof(pair):
    prior, candidate = pair
    with pytest.raises(ValueError, match='use the promotion check'):
        preservation.check_refresh(prior, candidate, [])


def test_a_promotion_is_not_a_refresh(pair, tmp_path):
    prior, _ = pair
    candidate = build(tmp_path / 'promoted',
                      narrations={IDENTITY: 'Map', OTHER: 'New mask',
                                  UNNAMED: 'Regression', '77_embeddings': 'New'},
                      media={IDENTITY: 'map', OTHER: 'mask-new', UNNAMED: 'regression'})
    with pytest.raises(ValueError, match='add or remove a lesson'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_a_navigation_change_outside_the_commit_is_refused(pair):
    prior, candidate = pair
    frozen(candidate / 'web/module_navigation.js',
           {'schema': 1, 'source_commit': 'deadbee',
            'labels': {'en': ['Main modules', 'Renamed']}})
    with pytest.raises(ValueError, match='navigation changed at labels'):
        preservation.check_refresh(prior, candidate, [OTHER])


def test_a_file_that_is_not_a_frozen_payload_fails_loudly(pair):
    prior, candidate = pair
    (candidate / 'web/lesson_catalog.js').write_text('window.X = {"lessons": []};')
    with pytest.raises(ValueError, match='Not a frozen catalog payload'):
        preservation.check_refresh(prior, candidate, [OTHER])
