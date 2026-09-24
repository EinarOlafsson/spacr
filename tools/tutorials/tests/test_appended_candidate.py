"""New tutorials can ship available media without disturbing published lessons."""
from copy import deepcopy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_appended_candidate import (append_catalogs, append_javascript_catalog,
                                     copy_preserved_web,
                                     complete_translation_compatibility,
                                     require_baseline_receipt, require_no_new_route_gaps,
                                     update_catalogs, synchronize_links)
from audit_staged_catalogs import CATALOGS


@pytest.mark.parametrize('replace', [False, True])
@pytest.mark.parametrize('local_missing', [False, True])
def test_preserved_web_uses_verified_media_and_excludes_leftover_clips(tmp_path, replace, local_missing):
    published, baseline, candidate = [tmp_path / name for name in ('pages', 'baseline', 'candidate')]
    relative = Path('production/01_existing/video/01_existing_silent.mp4')
    for base, content in ((published, b'unverified local change'), (baseline / 'web', b'verified video')):
        (base / relative).parent.mkdir(parents=True)
        (base / relative).write_bytes(content)
    leftover = published / 'production/02_held/video/02_held_silent.mp4'
    leftover.parent.mkdir(parents=True)
    leftover.write_bytes(b'retired clip')
    (published / 'app_v2.js').write_text('current player')
    if local_missing:
        (published / relative).unlink()
    manifest = {'files': [{'path': 'web/' + relative.as_posix(), 'bytes': 14,
                           'sha256': hashlib.sha256(b'verified video').hexdigest()}]}
    copied = copy_preserved_web(published, baseline, candidate, manifest,
                                replacements=['01_existing'] if replace else [])
    assert (candidate / 'web/app_v2.js').read_text() == 'current player'
    assert not (candidate / 'web/production/02_held').exists()
    assert (candidate / 'web' / relative).exists() is not replace
    if not replace:
        assert (candidate / 'web' / relative).read_bytes() == b'verified video'
    assert len(copied) == (1 if replace else 2)


@pytest.fixture
def sources():
    old = dict(id='01_existing', number=1, status='coming_soon', scenes=[{'narration': 'Existing'}])
    published = {name: {'lessons': [deepcopy(old)]} for name in CATALOGS}
    new = dict(id='02_new', number=2, title='Title', description='Description',
               objectives=['Objective'], prerequisite='Prerequisite', scenes=[{'narration': 'Current text'}])
    voices = {'02_new': {'en': ['af_heart']}}
    return published, new, voices


def test_link_sync_preserves_translated_prose_and_media():
    source = dict(id='01_workflow', scenes=[dict(narration='Open Home', related_lessons=['82_new'])])
    old = dict(id='01_workflow', scenes=[dict(narration='Open Home', related_lessons=['05_home'])],
               narration_voices={'en': ['af_heart']})
    localized = deepcopy(old)
    localized['scenes'][0]['narration'] = 'Öffnen Sie Home'
    catalogs = {'lessons_en.json': {'lessons': [old]}, 'captions_de.json': {'lessons': [localized]}}
    before = deepcopy(catalogs)
    result = synchronize_links(catalogs, [source])
    assert catalogs == before
    assert result['captions_de.json']['lessons'][0]['scenes'][0] == dict(
        narration='Öffnen Sie Home', related_lessons=['82_new'])
    assert result['lessons_en.json']['lessons'][0]['narration_voices'] == old['narration_voices']


@pytest.mark.parametrize('change', ['narration', 'hold_after', 'visual', 'chapter_count'])
def test_link_sync_rejects_content_or_timing_changes(change):
    source = dict(id='01_workflow', scenes=[dict(narration='Open Home', hold_after=0.7, visual='home')])
    catalogs = {'lessons_en.json': {'lessons': [deepcopy(source)]}}
    if change == 'chapter_count':
        source['scenes'].append(deepcopy(source['scenes'][0]))
    else:
        source['scenes'][0][change] = 'changed'
    with pytest.raises(ValueError, match='changes lesson content'):
        synchronize_links(catalogs, [source])


def test_missing_translations_do_not_block_but_remain_registered(sources):
    published, new, voices = sources
    before = deepcopy(published)
    result, compatibility = append_catalogs(published, [new], voices, {})
    assert published == before
    assert len(compatibility) == 13
    assert all(row['status'] == 'english_fallback' and row['reason'] for row in compatibility)
    for name in CATALOGS:
        assert result[name]['lessons'][:-1] == before[name]['lessons']
        assert result[name]['lessons'][-1]['scenes'] == new['scenes']
        assert result[name]['lessons'][-1]['narration_voices'] == voices[new['id']]


def test_later_batch_keeps_earlier_fallbacks_registered(sources):
    published, first, voices = sources
    intermediate, first_report = append_catalogs(published, [first], voices, {})
    second = {**first, 'id': '03_second', 'number': 3}
    catalogs, second_report = append_catalogs(
        intermediate, [second], {second['id']: {'en': ['af_heart']}}, {})
    complete = complete_translation_compatibility(catalogs, second_report, first_report)
    assert len(complete) == 26
    assert all(row in complete for row in first_report + second_report)
    assert not any(row['lesson'] == '01_existing' for row in complete)
    recovered = complete_translation_compatibility(catalogs, second_report)
    assert {(row['lesson'], row['language']) for row in recovered} == {
        (row['lesson'], row['language']) for row in complete}
    assert all(row['status'] == 'english_fallback' and row['reason'] for row in recovered)


def test_mixed_release_preserves_unselected_lesson_and_updates_both_catalog_formats(sources):
    published, new, _ = sources
    for catalog in published.values():
        catalog['lessons'].append(deepcopy(new))
    before = deepcopy(published)
    fresh = {**new, 'scenes': [{'narration': 'Practical current steps'}]}
    extra = {**new, 'id': '03_extra', 'number': 3}
    voices = {row['id']: {'en': ['af_heart']} for row in (fresh, extra)}
    result, compatibility = update_catalogs(published, [extra, fresh], voices, {}, [fresh['id']])
    assert published == before
    assert len(compatibility) == 26
    for catalog in result.values():
        assert [row['id'] for row in catalog['lessons']] == ['01_existing', '02_new', '03_extra']
        assert catalog['lessons'][0] == before['lessons_en.json']['lessons'][0]
        assert catalog['lessons'][1]['scenes'] == fresh['scenes']
        assert catalog['lessons'][2]['scenes'] == extra['scenes']
    javascript = deepcopy(before['lessons_en.json'])
    javascript['lessons'][0]['historical_only'] = 'Keep this field'
    javascript = append_javascript_catalog(javascript, result['lessons_en.json'], 1)
    javascript = append_javascript_catalog(javascript, result['lessons_en.json'], 1,
                                           replacements=[fresh['id']])
    assert javascript['lessons'][0]['historical_only'] == 'Keep this field'
    assert javascript['lessons'][1]['scenes'] == fresh['scenes']
    assert javascript['lessons'][2]['scenes'] == extra['scenes']


@pytest.mark.parametrize('refresh', [['02_new'], ['01_existing', '01_existing'], ['99_unknown']])
def test_mixed_release_rejects_refreshes_outside_baseline_or_duplicate_selection(sources, refresh):
    published, new, voices = sources
    before = deepcopy(published)
    with pytest.raises(ValueError):
        update_catalogs(published, [new], voices, {}, refresh)
    assert published == before


def test_source_bound_review_wins_and_stale_review_falls_back(sources):
    published, new, voices = sources
    review = dict(lesson=new['id'], language='de', title='Titel', description='Beschreibung',
                  objectives=['Ziel'], prerequisite='Voraussetzung', scenes=['Aktueller Text'],
                  english_sha256=hashlib.sha256(json.dumps(new, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    result, records = append_catalogs(published, [new], voices, {(new['id'], 'de'): review})
    assert result['captions_de.json']['lessons'][-1]['scenes'][0]['narration'] == 'Aktueller Text'
    assert next(row for row in records if row['language'] == 'de')['status'] == 'source_bound_review'
    review['english_sha256'] = '0' * 64
    result, records = append_catalogs(published, [new], voices, {(new['id'], 'de'): review})
    assert result['captions_de.json']['lessons'][-1]['scenes'] == new['scenes']
    assert next(row for row in records if row['language'] == 'de')['status'] == 'english_fallback'


@pytest.mark.parametrize('change', ['duplicate', 'gap', 'replacement', 'no_english', 'held', 'unsafe'])
def test_reject_invalid_appends_without_altering_baseline(sources, change):
    published, new, voices = sources
    before = deepcopy(published)
    lessons = [new]
    if change == 'duplicate':
        lessons.append(new)
    elif change == 'gap':
        new['number'] = 3
    elif change == 'replacement':
        new['id'] = '01_existing'
    elif change == 'no_english':
        voices[new['id']] = {'es': ['ef_dora']}
    elif change == 'held':
        new['status'] = 'coming_soon'
    elif change == 'unsafe':
        new['id'] = '../02_new'
    with pytest.raises(ValueError):
        append_catalogs(published, lessons, voices, {})
    assert published == before


def test_javascript_preserves_its_own_historical_objects(sources):
    published, new, voices = sources
    english, _ = append_catalogs(published, [new], voices, {})
    javascript = deepcopy(published['lessons_en.json'])
    javascript['lessons'][0]['extra_historical_field'] = 'Preserve independently'
    before = deepcopy(javascript)
    result = append_javascript_catalog(javascript, english['lessons_en.json'], 1)
    assert javascript == before
    assert result['lessons'][:-1] == before['lessons']
    assert result['lessons'][-1]['silent'] == '02_new/video/02_new_silent.mp4'


@pytest.mark.parametrize('fault', [None, 'partial_download', 'wrong_commit', 'mutable_root', 'wrong_manifest', 'missing_file'])
def test_baseline_requires_complete_immutable_readback(fault):
    manifest = {'files': [{'path': 'media_host/01_example/audio/en/af_heart.m4a'}]}
    commit = 'a' * 40
    receipt = dict(manifest_sha256='hash', commit=commit, tag='published', media_files=1,
        media_root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + commit,
        readback=dict(commit=commit, passed=True, files_expected=1, metadata_matched=1,
                      downloaded_sha256_matched=1, metadata_failures=[], download_failures=[]))
    if fault == 'partial_download':
        receipt['readback']['downloaded_sha256_matched'] = 0
    elif fault == 'wrong_commit':
        receipt['readback']['commit'] = 'b' * 40
    elif fault == 'mutable_root':
        receipt['media_root'] = receipt['media_root'].replace(commit, 'main')
    elif fault == 'wrong_manifest':
        receipt['manifest_sha256'] = 'other'
    elif fault == 'missing_file':
        manifest['files'].append({'path': 'media_host/01_example/audio/en/af_heart.json'})
    if fault is None:
        require_baseline_receipt(manifest, receipt, 'hash')
    else:
        with pytest.raises(ValueError):
            require_baseline_receipt(manifest, receipt, 'hash')


def test_preexisting_route_debt_is_not_a_gate_but_lost_routes_fail():
    before = {'missing_tutorials': [{'app_key': 'new_module'}]}
    require_no_new_route_gaps(before, deepcopy(before))
    require_no_new_route_gaps(before, {'missing_tutorials': []})
    with pytest.raises(ValueError):
        require_no_new_route_gaps(before, {'missing_tutorials': [{'app_key': 'previously_covered'}]})


def test_refresh_replaces_only_selected_prose_and_registers_translation_fallback(sources):
    published, new, _ = sources
    for catalog in published.values():
        catalog['lessons'].append(deepcopy(new))
        catalog['lessons'][1]['scenes'] = [{'narration': 'Old translated narration'}]
        catalog['lessons'][1]['narration_voices'] = {'en': ['af_heart'], 'es': ['ef_dora']}
    before = deepcopy(published)
    voices = {new['id']: {'en': ['af_heart']}}
    result, compatibility = append_catalogs(published, [new], voices, {}, replace=True)
    assert published == before
    assert len(compatibility) == 13
    for name in CATALOGS:
        assert [row['id'] for row in result[name]['lessons']] == ['01_existing', '02_new']
        assert result[name]['lessons'][0] == before[name]['lessons'][0]
        updated = result[name]['lessons'][1]
        assert updated['scenes'] == new['scenes']
        assert updated['narration_voices'] == {'en': ['af_heart']}


@pytest.mark.parametrize('change', ['unknown', 'number', 'module', 'parent', 'duplicate', 'no_english'])
def test_refresh_rejects_identity_changes_and_missing_english(sources, change):
    published, new, _ = sources
    fresh = {**new, 'id': '01_existing', 'number': 1}
    voices = {fresh['id']: {'en': ['af_heart']}}
    if change == 'unknown':
        fresh['id'] = '99_missing'
    elif change == 'number':
        fresh['number'] = 9
    elif change == 'module':
        fresh['app_key'] = 'mask'
    elif change == 'parent':
        fresh['host_app_key'] = 'measure'
    elif change == 'no_english':
        voices[fresh['id']] = {'es': ['ef_dora']}
    selection = [fresh, fresh] if change == 'duplicate' else [fresh]
    before = deepcopy(published)
    with pytest.raises(ValueError):
        append_catalogs(published, selection, voices, {}, replace=True)
    assert published == before


def test_refresh_can_repair_host_metadata_only_against_current_gui(sources):
    published, new, _ = sources
    fresh = {**new, 'id': '01_existing', 'number': 1, 'host_app_key': 'toxoplasma'}
    voices = {fresh['id']: {'en': ['af_heart']}}
    result, _ = append_catalogs(published, [fresh], voices, {}, replace=True,
                               current_hosts={'01_existing': 'toxoplasma'})
    assert result['lessons_en.json']['lessons'][0]['host_app_key'] == 'toxoplasma'
    with pytest.raises(ValueError, match='current GUI'):
        append_catalogs(published, [fresh], voices, {}, replace=True,
                        current_hosts={'01_existing': 'measure'})


def test_javascript_refresh_preserves_unselected_historical_objects(sources):
    published, new, voices = sources
    english, _ = append_catalogs(published, [new], voices, {})
    javascript = deepcopy(english['lessons_en.json'])
    javascript['lessons'][0]['extra_historical_field'] = 'Keep this'
    javascript['lessons'][1]['scenes'] = [{'narration': 'Old narration'}]
    before = deepcopy(javascript)
    result = append_javascript_catalog(javascript, english['lessons_en.json'], 1,
                                       replacements=['02_new'])
    assert javascript == before
    assert result['lessons'][0] == before['lessons'][0]
    assert result['lessons'][1]['scenes'] == new['scenes']
    assert result['lessons'][1]['silent'] == '02_new/video/02_new_silent.mp4'


def test_javascript_refresh_rejects_an_unknown_identity(sources):
    published, new, voices = sources
    english, _ = append_catalogs(published, [new], voices, {})
    with pytest.raises(ValueError):
        append_javascript_catalog(english['lessons_en.json'], english['lessons_en.json'], 1,
                                   replacements=['99_missing'])
