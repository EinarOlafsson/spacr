"""An unavailable route has no media promises; ready scripts stay identical."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coming_soon import COPY, HELD, OPS, EMBEDDINGS, PLACEHOLDERS, release_catalog


def test_browser_guard_targets_an_actual_hold_after_ops_is_recorded():
    from coming_soon import first_placeholder
    lessons = [{'id': OPS, 'scenes': [{'narration': 'Ready'}]},
               {'id': HELD[0], 'status': 'coming_soon', 'scenes': []}]
    assert first_placeholder(lessons) == HELD[0]
    with pytest.raises(ValueError, match='actually unavailable'):
        first_placeholder(lessons[:1])


@pytest.fixture
def catalog():
    return {'lessons': [dict(id=identity, number=i, title=identity,
            app_key=identity.split('_', 1)[-1], silent='existing.mp4',
            poster='existing.jpg', example_files=['existing.zip'],
            objectives=['Old promise'], prerequisite='Old input',
            scenes=[{'narration': 'Existing narration'}])
            for i, identity in enumerate(('01_ready', *HELD), 1)]}


@pytest.mark.parametrize('language', COPY)
def test_all_languages_preserve_ready_content_and_explicitly_hold_only_six(catalog, language):
    before = deepcopy(catalog)
    result = release_catalog(catalog, language)
    assert catalog == before
    assert result['lessons'][0] == before['lessons'][0]
    held = [x for x in result['lessons'] if x.get('status') == 'coming_soon']
    assert [x['id'] for x in held] == list(PLACEHOLDERS)
    for lesson in held:
        assert lesson['scenes'] == [] and lesson['objectives'] == []
        assert lesson['availability_title'] == COPY[language][0]
        assert lesson['description'] == COPY[language][1]
        assert not {'silent', 'poster', 'example_files'} & lesson.keys()
    by_id = {lesson['id']: lesson for lesson in held}
    assert by_id[OPS]['app_key'] == 'ops' and by_id[OPS]['host_app_key'] == 'mask'
    assert by_id[EMBEDDINGS]['app_key'] == 'embeddings'
    assert 'host_app_key' not in by_id[EMBEDDINGS]


@pytest.mark.parametrize('change', [
    lambda c: c['lessons'].append(deepcopy(c['lessons'][0])),
    lambda c: c['lessons'].pop(),
    lambda c: c['lessons'].append({'id': OPS}),
    lambda c: c['lessons'].append({'id': EMBEDDINGS}),
])
def test_unexpected_input_cannot_silently_drop_or_duplicate_a_route(catalog, change):
    release_catalog(catalog, 'en')
    change(catalog)
    with pytest.raises(ValueError, match='distinct original lessons'):
        release_catalog(catalog, 'en')


@pytest.mark.parametrize('language', COPY)
def test_recorded_embeddings_promotes_only_its_own_route(catalog, language):
    lesson = {'id': EMBEDDINGS, 'number': 77, 'app_key': 'embeddings',
              'title': 'Recorded API, not GUI completion', 'scenes': [{'narration': 'Real recording'}]}
    catalog['lessons'].append(deepcopy(lesson))
    before = deepcopy(catalog)
    result = release_catalog(catalog, language)
    assert catalog == before
    assert next(x for x in result['lessons'] if x['id'] == EMBEDDINGS) == lesson
    assert [x['id'] for x in result['lessons'] if x.get('status') == 'coming_soon'] == [*HELD, OPS]
    assert len({x['id'] for x in result['lessons']}) == len(result['lessons'])


@pytest.mark.parametrize('change', [{'app_key': 'wrong'}, {'number': 78},
                                  {'host_app_key': 'mask'}, {'scenes': []}, {'status': 'coming_soon'}])
def test_promotion_must_keep_the_actual_home_identity_and_recorded_scenes(catalog, change):
    lesson = {'id': EMBEDDINGS, 'number': 77, 'app_key': 'embeddings', 'scenes': [{'narration': 'Real'}]}
    catalog['lessons'].append(lesson)
    release_catalog(catalog, 'en')
    lesson.update(change)
    with pytest.raises(ValueError, match='recorded Embeddings promotion'):
        release_catalog(catalog, 'en')


@pytest.mark.parametrize('language', ('da', 'de', 'is', 'ko', 'nb', 'sv'))
def test_retained_caption_entries_without_numbers_keep_their_content_and_order(catalog, language):
    # Real retained caption catalogs (notably Plate Viewer and Database)
    # predate the numeric route metadata added by write_catalogs later.
    for lesson in catalog['lessons']:
        lesson.pop('number')
    catalog['lessons'].append({'id': EMBEDDINGS, 'number': 77, 'app_key': 'embeddings',
                              'scenes': [{'narration': 'Recorded API example'}]})
    before = deepcopy(catalog)
    result = release_catalog(catalog, language)
    assert catalog == before
    assert result['lessons'][0] == before['lessons'][0]
    assert result['lessons'][-1] == before['lessons'][-1]
    assert [x['id'] for x in result['lessons']] == [
        *[x['id'] for x in before['lessons'][:-1]], OPS, EMBEDDINGS]


@pytest.mark.parametrize('language', COPY)
def test_recorded_ops_keeps_exact_mask_route_and_other_holds(catalog, language, monkeypatch):
    import ops_promotion
    calls = []
    monkeypatch.setattr(ops_promotion, 'require_recorded_ops',
                        lambda stage, lang, row: calls.append((stage, lang, row)))
    lesson = {'id': OPS, 'number': 76, 'app_key': 'ops', 'host_app_key': 'mask',
              'title': 'Four real tiles, not a full pipeline', 'scenes': [{'narration': 'Actual geometry.'}]}
    # Existing caption records may predate numeric metadata.
    for item in catalog['lessons']:
        item.pop('number')
    catalog['lessons'].append(lesson)
    before = deepcopy(catalog)
    result = release_catalog(catalog, language, recording_stage='unit-test-stage')
    assert calls == [('unit-test-stage', language, lesson)]
    assert catalog == before
    assert result['lessons'][:-1] == [
        *release_catalog({'lessons': before['lessons'][:-1]}, language)['lessons'][:-2], lesson]
    assert [item['id'] for item in result['lessons'] if item.get('status') == 'coming_soon'] == [*HELD, EMBEDDINGS]
    assert len({item['id'] for item in result['lessons']}) == len(result['lessons'])


@pytest.mark.parametrize('change', [{'app_key': 'wrong'}, {'number': 78},
                                  {'host_app_key': 'stitch'}, {'scenes': []}, {'status': 'coming_soon'}])
def test_ops_promotion_rejects_wrong_route_or_unrecorded_placeholder(catalog, change, monkeypatch):
    import ops_promotion
    monkeypatch.setattr(ops_promotion, 'require_recorded_ops', lambda *args: None)
    lesson = {'id': OPS, 'number': 76, 'app_key': 'ops', 'host_app_key': 'mask',
              'scenes': [{'narration': 'Actual geometry.'}]}
    catalog['lessons'].append(lesson)
    release_catalog(catalog, 'en')
    lesson.update(change)
    with pytest.raises(ValueError, match='recorded OPS'):
        release_catalog(catalog, 'en')


def test_ops_catalog_metadata_alone_is_never_a_recording(catalog):
    catalog['lessons'].append({'id': OPS, 'number': 76, 'app_key': 'ops',
                              'host_app_key': 'mask', 'scenes': [{'narration': 'A promise.'}]})
    with pytest.raises(ValueError, match='requires verified recording and final media'):
        release_catalog(catalog, 'en')
