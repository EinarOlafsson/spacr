"""Selective source-only append guards; no spaCR imports, media or real writes."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import append_staged_lessons as appender


def lesson(number, identity=None):
    identity = identity or f'{number:02d}_kept_{number}'
    return {'id': identity, 'number': number, 'slug': identity.split('_', 1)[1],
            'app_key': f'route_{number}', 'host_app_key': 'host', 'series': 1,
            'section': 'Tools', 'title': f'Title {number}', 'description': 'Description',
            'objectives': ['First objective', 'Second objective'], 'prerequisite': 'Prerequisite',
            'scenes': [{'visual': 'step_one', 'narration': 'First narration', 'hold_after': .7,
                        'focus': [1, 2, 3, 4], 'related_lessons': ['01_kept_1']}]}


@pytest.fixture
def catalogs():
    # Small rows, but the real 73 -> 74/75 numbering boundary is retained.
    old = {'schema': 1, 'title': 'Original catalog metadata',
           'series': [{'number': 1, 'title': 'Original series'}],
           'custom': {'preserve': [1, 2]}, 'lessons': [lesson(i) for i in range(1, 74)]}
    english = copy.deepcopy(old)
    english['title'] = 'Staged metadata must not replace old metadata'
    english['lessons'][0]['title'] = 'An unselected staged refresh'
    english['lessons'][0]['scenes'].append({'visual': 'refreshed', 'narration': 'Not selected'})
    for number, identity in zip((74, 75), appender.ALLOWED_IDS):
        value = lesson(number, identity)
        value['scenes'].append({'visual': 'step_two', 'narration': 'Second narration',
                                'hold_after': .8})
        english['lessons'].append(value)
    existing, staged = {}, {}
    for name in appender.CATALOGS:
        before = copy.deepcopy(old)
        after = copy.deepcopy(english)
        before['locale_metadata'] = {'filename': name, 'source': 'original'}
        after['locale_metadata'] = {'filename': name, 'source': 'staged'}
        for value in before['lessons'] + after['lessons']:
            value['title'] = name + ': ' + value['title']
            for scene in value['scenes']:
                scene['narration'] = name + ': ' + scene['narration']
        if name.startswith('captions_'):
            before = {'schema': 1, 'language': name[9:-5], 'source_language': 'en',
                      'lessons': [{'id': value['id'], 'scenes': [
                          {'narration': s['narration']} for s in value['scenes']]}
                          for value in before['lessons']]}
        existing[name], staged[name] = before, after
    javascript = copy.deepcopy(existing['lessons_en.json'])
    javascript['custom_js_metadata'] = 'Keep me'
    javascript['lessons'][0]['title'] = 'Original JS is deliberately different'
    javascript['lessons'][0]['scenes'].append({'visual': 'legacy_js', 'narration': 'Keep this too'})
    for value in javascript['lessons']:
        value['poster'] = value['id'] + '/poster.jpg'
        value['silent'] = value['id'] + '/video/' + value['id'] + '_silent.mp4'
    return existing, staged, javascript


def merge(catalogs, selected=appender.ALLOWED_IDS):
    return appender.merge_catalogs(catalogs[0], catalogs[1], selected, catalogs[2])


def test_append_preserves_all_original_objects_and_top_level_metadata(catalogs):
    snapshot = copy.deepcopy(catalogs)
    result = merge(catalogs)
    assert catalogs == snapshot
    for name in appender.CATALOGS:
        original, source = catalogs[0][name], catalogs[1][name]
        actual = result['catalogs'][name]
        assert actual['lessons'][:73] == original['lessons']
        assert actual['lessons'][73:] == source['lessons'][73:]
        assert {k: v for k, v in actual.items() if k != 'lessons'} == {
            k: v for k, v in original.items() if k != 'lessons'}
    result['catalogs']['lessons_en.json']['lessons'][0]['title'] = 'Changed result'
    assert catalogs == snapshot


def test_javascript_originals_and_count_discrepancy_are_preserved(catalogs):
    result = merge(catalogs)
    actual = result['javascript_catalog']
    assert actual['lessons'][:73] == catalogs[2]['lessons']
    assert actual['custom_js_metadata'] == 'Keep me'
    for value in actual['lessons'][73:]:
        assert value['poster'] == value['id'] + '/poster.jpg'
        assert value['silent'] == value['id'] + '/video/' + value['id'] + '_silent.mp4'
    report = result['report']
    assert (report['existing_json_scene_count'], report['existing_javascript_scene_count']) == (73, 74)
    assert (report['merged_json_scene_count'], report['merged_javascript_scene_count']) == (77, 78)
    assert report['preserved_javascript_scene_count_differences'] == [
        {'lesson_id': '01_kept_1', 'json': 1, 'javascript': 2}]


def test_append_only_first_selected_lesson_does_not_copy_other_staged_addition(catalogs):
    result = merge(catalogs, appender.ALLOWED_IDS[:1])
    assert len(result['catalogs']['lessons_en.json']['lessons']) == 74
    assert result['catalogs']['lessons_en.json']['lessons'][-1]['id'] == appender.ALLOWED_IDS[0]


def test_sparse_existing_caption_objects_stay_sparse(catalogs):
    result = merge(catalogs)
    values = result['catalogs']['captions_ko.json']['lessons']
    assert set(values[0]) == {'id', 'scenes'}
    assert values[73]['number'] == 74
    assert values[73]['host_app_key'] == 'host'


@pytest.mark.parametrize('selection', [[], [appender.ALLOWED_IDS[0]] * 2, ['01_kept_1']])
def test_invalid_or_unapproved_selection_is_rejected(catalogs, selection):
    with pytest.raises(ValueError, match='Explicit, unique'):
        merge(catalogs, selection)


def test_replacement_of_previously_appended_lesson_is_rejected(catalogs):
    first = merge(catalogs, appender.ALLOWED_IDS[:1])
    with pytest.raises(ValueError, match='Refusing replacement'):
        appender.merge_catalogs(first['catalogs'], catalogs[1], appender.ALLOWED_IDS,
                                first['javascript_catalog'])


def test_second_append_can_follow_first_without_replacing_it(catalogs):
    first = merge(catalogs, appender.ALLOWED_IDS[:1])
    result = appender.merge_catalogs(first['catalogs'], catalogs[1], appender.ALLOWED_IDS[1:],
                                    first['javascript_catalog'])
    assert result['catalogs']['lessons_en.json']['lessons'][:74] == first['catalogs']['lessons_en.json']['lessons']
    assert result['report']['merged_lesson_count'] == 75


@pytest.mark.parametrize('selection', [appender.ALLOWED_IDS[::-1], appender.ALLOWED_IDS[1:]])
def test_wrong_append_order_or_number_gap_is_rejected(catalogs, selection):
    with pytest.raises(ValueError, match='order|immediately continue'):
        merge(catalogs, selection)


@pytest.mark.parametrize('side', [0, 1])
def test_missing_locale_catalog_is_rejected(catalogs, side):
    catalogs[side].pop('captions_ko.json')
    with pytest.raises(ValueError, match='Missing locale'):
        merge(catalogs)


@pytest.mark.parametrize('side', [0, 1])
def test_missing_locale_lesson_is_rejected(catalogs, side):
    catalogs[side]['captions_ko.json']['lessons'].pop()
    with pytest.raises(ValueError, match='identities/order'):
        merge(catalogs)


@pytest.mark.parametrize('side', [0, 1, 2])
def test_duplicate_lesson_identity_is_rejected(catalogs, side):
    values = (catalogs[side]['lessons_en.json'] if side < 2 else catalogs[2])['lessons']
    values[-1]['id'] = values[-2]['id']
    with pytest.raises(ValueError, match='duplicate lesson ID'):
        merge(catalogs)


@pytest.mark.parametrize('field', ['app_key', 'slug'])
def test_new_duplicate_route_is_rejected(catalogs, field):
    catalogs[1]['lessons_en.json']['lessons'][-1][field] = catalogs[1]['lessons_en.json']['lessons'][0][field]
    with pytest.raises(ValueError, match='duplicate .* route'):
        merge(catalogs)


def test_new_noncontiguous_number_is_rejected(catalogs):
    catalogs[1]['lessons_en.json']['lessons'][-1]['number'] = 76
    with pytest.raises(ValueError, match='contiguous'):
        merge(catalogs)


def test_old_caption_order_drift_is_rejected(catalogs):
    values = catalogs[0]['captions_ko.json']['lessons']
    values[0], values[1] = values[1], values[0]
    with pytest.raises(ValueError, match='identities/order'):
        merge(catalogs)


def test_new_scene_focus_mismatch_is_rejected(catalogs):
    catalogs[1]['lessons_ja.json']['lessons'][-1]['scenes'][0]['focus'][0] = 999
    with pytest.raises(ValueError, match='structure/scenes/order'):
        merge(catalogs)


def test_new_scene_order_mismatch_is_rejected(catalogs):
    catalogs[1]['captions_ko.json']['lessons'][-1]['scenes'].reverse()
    with pytest.raises(ValueError, match='structure/scenes/order'):
        merge(catalogs)


@pytest.mark.parametrize('change', ['missing_scene', 'empty_scenes', 'missing_narration',
                                   'missing_title', 'missing_objective', 'wrong_host'])
def test_incomplete_or_wrong_new_translation_is_rejected(catalogs, change):
    value = catalogs[1]['lessons_ja.json']['lessons'][-1]
    if change == 'missing_scene': value['scenes'].pop()
    elif change == 'empty_scenes': value['scenes'] = []
    elif change == 'missing_narration': value['scenes'][0].pop('narration')
    elif change == 'missing_title': value.pop('title')
    elif change == 'missing_objective': value['objectives'].pop()
    elif change == 'wrong_host': value['host_app_key'] = 'wrong'
    with pytest.raises(ValueError):
        merge(catalogs)


def test_optional_local_speech_text_may_differ(catalogs):
    catalogs[1]['lessons_ja.json']['lessons'][-1]['scenes'][0]['speech_text'] = 'Local pronunciation'
    assert merge(catalogs)['report']['merged_lesson_count'] == 75


def navigation(catalog):
    return {'schema': 1, 'preserved_lesson_ids': [v['id'] for v in catalog['lessons']],
            'routes': {v['id']: {'app_key': v['app_key'], 'host_app_key': v['host_app_key']}
                       for v in catalog['lessons']},
            'missing_tutorials': [{'app_key': 'ops', 'status': 'deferred_unvalidated_workflow'}]}


@pytest.fixture
def folders(tmp_path, catalogs):
    root, stage = tmp_path / 'web', tmp_path / 'stage'
    for folder, values in ((root, catalogs[0]), (stage, catalogs[1])):
        (folder / 'catalog').mkdir(parents=True)
        for name, value in values.items():
            (folder / 'catalog' / name).write_text(json.dumps(value), encoding='utf-8')
    (root / 'lesson_catalog.js').write_text(appender.JS_PREFIX + json.dumps(catalogs[2]) + appender.JS_SUFFIX)
    (root / 'module_navigation.js').write_text('Original navigation bytes\n')
    return root, stage


def snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_plan_is_read_only_and_local_apply_writes_only_sixteen_catalog_files(folders, catalogs):
    root, stage = folders
    before, source = snapshot(root), snapshot(stage)
    plan = appender.plan_append(root, stage, appender.ALLOWED_IDS, navigation_builder=navigation)
    assert not plan.report['written'] and not plan.report['published']
    assert snapshot(root) == before and snapshot(stage) == source
    report = appender.write_plan(plan)
    assert report['written'] and not report['published']
    assert set(snapshot(root)) == set(before) and len(plan.outputs) == 16
    assert snapshot(stage) == source
    for name in appender.CATALOGS:
        actual = json.loads((root / 'catalog' / name).read_text())
        assert actual['lessons'][:73] == catalogs[0][name]['lessons']
    js = appender.parse_javascript((root / 'lesson_catalog.js').read_text())
    assert js['lessons'][:73] == catalogs[2]['lessons']


@pytest.mark.parametrize('relative,source_side', [('catalog/lessons_en.json', False),
                                                ('lesson_catalog.js', False),
                                                ('module_navigation.js', False),
                                                ('catalog/captions_ko.json', True)])
def test_stale_input_guard_refuses_before_any_write(folders, relative, source_side):
    root, stage = folders
    plan = appender.plan_append(root, stage, appender.ALLOWED_IDS, navigation_builder=navigation)
    changed = (stage if source_side else root) / relative
    changed.write_bytes(changed.read_bytes() + b' ')
    before, source = snapshot(root), snapshot(stage)
    with pytest.raises(ValueError, match='Input changed since planning'):
        appender.write_plan(plan)
    assert snapshot(root) == before and snapshot(stage) == source


@pytest.mark.parametrize('kind', ['extra_missing', 'no_ops', 'wrong_route', 'missing_lesson'])
def test_navigation_must_cover_selected_routes_and_leave_only_ops(folders, kind):
    def bad(catalog):
        value = navigation(catalog)
        if kind == 'extra_missing': value['missing_tutorials'].append({'app_key': 'unexpected'})
        elif kind == 'no_ops': value['missing_tutorials'] = []
        elif kind == 'wrong_route': value['routes'][appender.ALLOWED_IDS[0]]['host_app_key'] = 'wrong'
        else: value['preserved_lesson_ids'].pop()
        return value
    before = snapshot(folders[0])
    with pytest.raises(ValueError, match='Navigation'):
        appender.plan_append(*folders, appender.ALLOWED_IDS, navigation_builder=bad)
    assert snapshot(folders[0]) == before


def test_malformed_javascript_wrapper_is_rejected_without_execution(folders):
    (folders[0] / 'lesson_catalog.js').write_text('throw new Error("Do not execute");')
    with pytest.raises(ValueError, match='wrapper'):
        appender.plan_append(*folders, appender.ALLOWED_IDS, navigation_builder=navigation)


@pytest.mark.parametrize('payload', ['{"lessons":[],"lessons":[]}', '{"lessons":[],"value":NaN}'])
def test_ambiguous_or_nonfinite_json_is_rejected(folders, payload):
    (folders[1] / 'catalog/lessons_en.json').write_text(payload)
    with pytest.raises(ValueError, match='Duplicate JSON member|Nonfinite JSON'):
        appender.plan_append(*folders, appender.ALLOWED_IDS, navigation_builder=navigation)


def test_symlink_target_is_rejected(folders, tmp_path):
    path = folders[0] / 'lesson_catalog.js'
    other = tmp_path / 'other.js'
    other.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match='symlink/out-of-root'):
        appender.plan_append(*folders, appender.ALLOWED_IDS, navigation_builder=navigation)


def test_cli_without_write_only_reports_plan(folders, monkeypatch, capsys):
    original = appender.plan_append
    monkeypatch.setattr(appender, 'plan_append', lambda root, stage, ids: original(
        root, stage, ids, navigation_builder=navigation))
    before = snapshot(folders[0])
    assert appender.main(['--root', str(folders[0]), '--stage', str(folders[1]),
                          '--lesson-ids', *appender.ALLOWED_IDS]) == 0
    assert not json.loads(capsys.readouterr().out)['written']
    assert snapshot(folders[0]) == before
