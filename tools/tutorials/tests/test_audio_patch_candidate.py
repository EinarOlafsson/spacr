"""Audio correction must not change scripts or existing offered voice sets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_audio_patch_candidate as patch
from audit_staged_catalogs import CATALOGS
from stage_lesson import write


@pytest.fixture
def added_offerings(tmp_path, monkeypatch):
    """Synthetic bytes exercise scope only; no audio is synthesized or decoded."""
    from append_staged_lessons import JS_PREFIX, JS_SUFFIX
    from apply_translation_review import translated_lesson
    from render_all_voices import LANGUAGES
    baseline, stage, repo = [tmp_path / name for name in ('baseline', 'stage', 'repo')]
    identity = '06_api'
    canonical = dict(id=identity, number=6, title='API', description='Use the API.',
                     objectives=['Inspect inputs.'], prerequisite='Open Python.',
                     scenes=[dict(visual='reference', narration='Inspect the inputs.')])
    inventory = {language: list(pair[1]) for language, pair in LANGUAGES.items()}
    old_voices = {**inventory, 'en': ['af_heart']}
    english = dict(canonical, narration_voices=old_voices)
    other = dict(id='01_intro', scenes=[dict(narration='Unchanged.')])
    for name in CATALOGS:
        language = name.split('_', 1)[1].removesuffix('.json')
        if language == 'en':
            localized = deepcopy(english)
        else:
            review = dict(lesson=identity, language=language,
                          english_sha256=hashlib.sha256(json.dumps(canonical, sort_keys=True,
                                                       ensure_ascii=False).encode()).hexdigest(),
                          scenes=[language + ': Inspect the inputs.'],
                          **{key: canonical[key] for key in
                             ('title', 'description', 'objectives', 'prerequisite')})
            localized = translated_lesson(review, english, {})
            write(repo / 'tools/tutorials/lessons/reviews' / f'{identity}.{language}.json', review)
        for root in (baseline / 'web', stage):
            write(root / 'catalog' / name, dict(schema=1, lessons=[other, localized]))
    write(repo / 'tools/tutorials/lessons' / f'{identity}.json', canonical)
    write(stage / 'production' / identity / 'lesson.en.json', canonical)
    runtime = dict(lessons=[other, dict(english, web=f'{identity}/web/{identity}_silent.mp4')])
    (baseline / 'web/lesson_catalog.js').write_text(JS_PREFIX + json.dumps(runtime) + JS_SUFFIX)
    records = []
    for language, voices in inventory.items():
        localized = next(row for row in patch.read(stage / 'catalog' / f'lessons_{language}.json')['lessons']
                         if row['id'] == identity)
        for voice in voices:
            audio = stage / 'production' / identity / 'audio' / language / (voice + '.m4a')
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes((language + '/' + voice).encode())
            timing = dict(schema=1, language=language, voice=voice, media_sha256=patch.digest(audio),
                          total_duration=2., render_inputs=dict(lesson=identity, language=language,
                          voice=voice, scenes=localized['scenes']),
                          scenes=[dict(scene=1, text=localized['scenes'][0]['narration'],
                                       speech_start=0., speech_end=1., scene_end=2., duration=2.)])
            write(audio.with_suffix('.json'), timing)
            if voice in old_voices[language]:
                for path in (audio, audio.with_suffix('.json')):
                    relative = f'media_host/{identity}/audio/{language}/{path.name}'
                    destination = baseline / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(path.read_bytes())
                    records.append(dict(path=relative, sha256=patch.digest(path), bytes=path.stat().st_size))
    for relative in (f'media_host/{identity}/video/{identity}_silent.mp4',
                     f'media_host/{identity}/web/{identity}_silent.mp4',
                     f'web/production/{identity}/poster.jpg', 'web/app_v2.js'):
        path = baseline / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        records.append(dict(path=relative, sha256=patch.digest(path), bytes=path.stat().st_size))
    for path in [*(baseline / 'web/catalog').glob('*.json'), baseline / 'web/lesson_catalog.js']:
        records.append(dict(path=path.relative_to(baseline).as_posix(), sha256=patch.digest(path),
                            bytes=path.stat().st_size))
    write(baseline / 'release-manifest.json', dict(files=records, routes=2, ready_lessons=2,
                                                 ceiling_bytes=10**7, narration_tracks=27))
    write(baseline / 'checks/candidate-browser-checks.json', dict(passed=True,
          manifest_sha256=patch.digest(baseline / 'release-manifest.json'),
          ready_playback_cases=[dict(passed=True), dict(passed=True)]))
    monkeypatch.setattr(patch, 'REPO', repo)
    return baseline, stage, repo, identity


def test_add_offerings_preserves_old_tracks_visuals_clock_and_other_objects(added_offerings, monkeypatch):
    baseline, stage, _, identity = added_offerings
    before = {row['path']: row for row in patch.read(baseline / 'release-manifest.json')['files']}
    checked = []
    monkeypatch.setattr(patch, 'validate', lambda *args, **kwargs: None)
    monkeypatch.setattr(patch, 'require_published_script_baseline', lambda *args: None)
    monkeypatch.setattr(patch, 'verify_current_tracks', lambda base, rows: checked.extend(rows))
    candidate = patch.build(baseline, stage, [identity], add_english_offerings=True)
    assert len(checked) == 23 and all(row['voice'] != 'af_heart' for row in checked)
    after = {row['path']: row for row in patch.read(candidate / 'release-manifest.json')['files']}
    changed = {'web/lesson_catalog.js', *('web/catalog/' + name for name in CATALOGS)}
    for relative, row in before.items():
        if relative not in changed:
            assert after[relative] == row
            assert (candidate / relative).read_bytes() == (baseline / relative).read_bytes()
    assert len(set(after) - set(before)) == 46
    for relative in changed:
        old = (patch.parse_javascript((baseline / relative).read_text()) if relative.endswith('.js')
               else patch.read(baseline / relative))
        new = (patch.parse_javascript((candidate / relative).read_text()) if relative.endswith('.js')
               else patch.read(candidate / relative))
        selected = new['lessons'][1]
        assert sum(map(len, selected['narration_voices'].values())) == 50
        selected['narration_voices'] = old['lessons'][1]['narration_voices']
        assert new == old


@pytest.fixture
def browser_offering_candidate(added_offerings, monkeypatch):
    baseline, stage, _, identity = added_offerings
    monkeypatch.setattr(patch, 'validate', lambda *args, **kwargs: None)
    monkeypatch.setattr(patch, 'require_published_script_baseline', lambda *args: None)
    monkeypatch.setattr(patch, 'verify_current_tracks', lambda *args: None)
    return patch.build(baseline, stage, [identity], add_english_offerings=True)


def test_added_offering_browser_preserves_all_heart_cases_and_adds_exact23(browser_offering_candidate):
    from verify_release_candidate import added_offering_cases, playback_cases
    candidate = browser_offering_candidate
    manifest = patch.read(candidate / 'release-manifest.json')
    added = added_offering_cases(candidate, manifest)
    full = [dict(id=f'lesson{i}') for i in range(83)] + [dict(id='06_api')]
    for ready in (full, full[25:]):
        cases = playback_cases(ready, added)
        assert cases[:len(ready)] == [(row, 'af_heart', False) for row in ready]
        assert len(cases) == len(ready) + 23
        assert all(row['id'] == '06_api' and voice != 'af_heart' and extra
                   for row, voice, extra in cases[len(ready):])
    with pytest.raises(ValueError, match='active playable'):
        playback_cases(full[:-1], added)
    assert added_offering_cases(candidate, {}) == []


@pytest.mark.parametrize('change', ['lesson', 'dropped', 'duplicate', 'heart', 'language',
                                   'hash', 'audio_bytes', 'timing_bytes', 'catalog', 'runtime'])
def test_added_offering_browser_rejects_incomplete_or_unbound_scope(browser_offering_candidate, change):
    from verify_release_candidate import added_offering_cases
    candidate = browser_offering_candidate
    manifest = patch.read(candidate / 'release-manifest.json')
    tracks = manifest['audio_patch_tracks']
    if change == 'lesson':
        manifest['added_english_offerings'] = ['05_home']
    elif change == 'dropped':
        tracks.pop()
    elif change == 'duplicate':
        tracks[-1] = tracks[0]
    elif change == 'heart':
        tracks[0]['voice'] = 'af_heart'
    elif change == 'language':
        tracks[0]['language'] = 'es'
    elif change == 'hash':
        tracks[0]['audio_sha256'] = '0' * 64
    elif change in {'audio_bytes', 'timing_bytes'}:
        suffix = '.m4a' if change == 'audio_bytes' else '.json'
        (candidate / f"media_host/06_api/audio/en/{tracks[0]['voice']}{suffix}").write_bytes(b'drift')
    elif change == 'catalog':
        path = candidate / 'web/catalog/captions_de.json'
        value = patch.read(path)
        value['lessons'][1]['narration_voices']['en'].pop()
        write(path, value)
    else:
        from append_staged_lessons import JS_PREFIX, JS_SUFFIX
        path = candidate / 'web/lesson_catalog.js'
        value = patch.parse_javascript(path.read_text())
        value['lessons'][1]['narration_voices']['en'].pop()
        path.write_text(JS_PREFIX + json.dumps(value) + JS_SUFFIX)
    with pytest.raises(ValueError):
        added_offering_cases(candidate, manifest)


def test_added_offering_navigation_avoids_appended_case_collision():
    from verify_release_candidate import next_playable_lesson
    ready = [dict(id='01_intro'), dict(id='06_api'), dict(id='07_mask')]
    assert next_playable_lesson(ready, '06_api', 1) == '07_mask'
    # The first appended case has position3; the old formula selected06 again.
    assert next_playable_lesson(ready, '06_api', 3) == '01_intro'
    with pytest.raises(ValueError, match='different ready'):
        next_playable_lesson([dict(id='06_api')], '06_api', 1)


@pytest.mark.parametrize('change', ['old_audio', 'old_timing', 'missing_old', 'metadata',
                                    'review', 'canonical', 'extra_voice', 'clock'])
def test_add_offerings_rejects_drift_before_final_audio_check(added_offerings, change):
    baseline, stage, repo, identity = added_offerings
    if change in {'old_audio', 'old_timing', 'missing_old'}:
        path = stage / 'production' / identity / 'audio/en' / (
            'af_heart.json' if change == 'old_timing' else 'af_heart.m4a')
        if change == 'missing_old':
            path.rename(path.with_suffix('.held'))
        else:
            path.write_bytes(b'mutated')
    elif change == 'extra_voice':
        (stage / 'production' / identity / 'audio/en/unapproved.m4a').write_bytes(b'extra')
    elif change == 'clock':
        path = baseline / 'media_host' / identity / 'audio/en/af_heart.json'
        data = patch.read(path)
        data['scenes'][0]['scene'] = 2
        write(path, data)
    else:
        path = (stage / 'catalog/captions_de.json' if change == 'metadata' else
                repo / 'tools/tutorials/lessons/reviews' / f'{identity}.fr.json' if change == 'review' else
                repo / 'tools/tutorials/lessons' / f'{identity}.json')
        data = patch.read(path)
        if change == 'metadata':
            data['lessons'][0]['title'] = 'Unrelated change'
        elif change == 'review':
            data['scenes'][0] = 'Different narration'
        else:
            data['title'] = 'Changed'
        write(path, data)
    with pytest.raises(ValueError):
        patch.offering_patch_plan(baseline, stage, [identity])
    assert not list(stage.glob('release-candidate-*'))


@pytest.mark.parametrize('ids', [[], ['06_api', '48_hit_list'], ['07_mask']])
def test_add_offerings_is_explicit_single_lesson_scope(ids):
    with pytest.raises(ValueError, match='exactly one'):
        patch.offering_patch_plan(Path('unused'), Path('unused'), ids)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    baseline, stage, repo = [tmp_path / name for name in ('baseline', 'stage', 'repo')]
    canonical = {'id': '07_mask', 'number': 7, 'title': 'Masks',
                 'scenes': [{'narration': 'Inspect the live preview.', 'visual': 'preview'}]}
    lesson = deepcopy(canonical)
    lesson['narration_voices'] = {'en': ['af_heart', 'bf_alice'], 'fr': ['ff_siwis']}
    for name in CATALOGS:
        write(baseline / 'web/catalog' / name, {'lessons': [lesson]})
        write(stage / 'catalog' / name, {'lessons': [lesson]})
    write(repo / 'tools/tutorials/lessons/07_mask.json', canonical)
    write(stage / 'production/07_mask/lesson.en.json', canonical)
    records = []
    for voice in lesson['narration_voices']['en']:
        audio = stage / 'production/07_mask/audio/en' / (voice + '.m4a')
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(b'fixture corrected audio ' + voice.encode())
        timing = {'language': 'en', 'voice': voice,
                  'media_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
                  'render_inputs': {'lesson': '07_mask', 'language': 'en', 'voice': voice,
                                    'scenes': canonical['scenes']}}
        write(audio.with_suffix('.json'), timing)
        for suffix in ('.m4a', '.json'):
            records.append({'path': f'media_host/07_mask/audio/en/{voice}{suffix}',
                            'sha256': 'original-' + voice, 'bytes': 1})
    write(baseline / 'release-manifest.json', {'files': records})
    monkeypatch.setattr(patch, 'REPO', repo)
    return baseline, stage, repo


def test_uses_existing_english_offerings_and_leaves_other_language_unstaged(staged):
    baseline, stage, _ = staged
    plan = patch.patch_plan(baseline, stage, ['07_mask'])
    assert [row['voice'] for row in plan] == ['af_heart', 'bf_alice']
    assert not (stage / 'production/07_mask/audio/fr').exists()
    assert all(row['audio_sha256'] != row['original_audio_sha256'] for row in plan)


@pytest.mark.parametrize('mutation', ['translation', 'canonical', 'extra_voice', 'missing_voice',
                                    'audio_identity', 'timing_narration', 'missing_original'])
def test_rejects_drift_before_any_candidate_is_created(staged, mutation):
    baseline, stage, repo = staged
    if mutation == 'translation':
        path = stage / 'catalog/lessons_fr.json'
        data = json.loads(path.read_text(encoding="utf-8"))
        data['lessons'][0]['title'] = 'Modified translation'
        write(path, data)
    elif mutation == 'canonical':
        path = repo / 'tools/tutorials/lessons/07_mask.json'
        data = json.loads(path.read_text(encoding="utf-8"))
        data['scenes'][0]['narration'] = 'Different workflow instructions.'
        write(path, data)
    elif mutation == 'extra_voice':
        (stage / 'production/07_mask/audio/en/am_new.m4a').write_bytes(b'extra')
    elif mutation == 'missing_voice':
        (stage / 'production/07_mask/audio/en/bf_alice.json').unlink()
    elif mutation in {'audio_identity', 'timing_narration'}:
        path = stage / 'production/07_mask/audio/en/af_heart.json'
        data = json.loads(path.read_text(encoding="utf-8"))
        if mutation == 'audio_identity':
            data['media_sha256'] = 'wrong'
        else:
            data['render_inputs']['scenes'][0]['narration'] = 'Altered speech.'
        write(path, data)
    elif mutation == 'missing_original':
        path = baseline / 'release-manifest.json'
        data = json.loads(path.read_text(encoding="utf-8"))
        data['files'] = data['files'][1:]
        write(path, data)
    with pytest.raises(ValueError):
        patch.patch_plan(baseline, stage, ['07_mask'])
    assert not list(stage.glob('release-candidate-*'))


@pytest.mark.parametrize('identities', [[], ['07_mask', '07_mask'], ['../07_mask'], ['88_absent']])
def test_rejects_invalid_selection(staged, identities):
    baseline, stage, _ = staged
    with pytest.raises(ValueError):
        patch.patch_plan(baseline, stage, identities)


@pytest.fixture
def refreshed_scripts(tmp_path, monkeypatch):
    """Metadata-only unit fixture: synthetic bytes are never decoded or accepted."""
    baseline, stage, repo = [tmp_path / name for name in ('published', 'fresh', 'source')]
    identity = '79_module_inputs_outputs'
    voices = {'en': ['af_heart'], 'es': ['ef_dora', 'em_alex', 'em_santa'],
              'fr': ['ff_siwis'], 'hi': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
              'it': ['if_sara', 'im_nicola'],
              'ja': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
              'pt-BR': ['pf_dora', 'pm_alex', 'pm_santa'],
              'zh-CN': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi',
                        'zm_yunjian', 'zm_yunxi', 'zm_yunxia', 'zm_yunyang']}
    assert sum(map(len, voices.values())) == 27
    old = dict(id=identity, number=79, title='Module map', section='Workflows',
               description='Choose a module.', objectives=['Follow the files.'], prerequisite='Open Home.',
               scenes=[dict(visual='home', narration='Old opening.', hold_after=.7),
                       dict(visual='module_mask', narration='Old mask instruction.', hold_after=.7)],
               narration_voices=voices)
    new = deepcopy(old)
    new['scenes'][0]['narration'] = 'Choose a module from your input files.'
    new['scenes'][1]['narration'] = 'Inspect the masks before measurement.'
    canonical = {k: v for k, v in new.items() if k != 'narration_voices'}
    other = dict(id='01_intro', number=1, title='Other', scenes=[dict(visual='intro', narration='Unchanged.')])
    from pronunciation import spoken_form
    for name in CATALOGS:
        language = name.split('_', 1)[1].removesuffix('.json')
        before, after = deepcopy(old), deepcopy(new)
        if language != 'en':
            for item in (before, after):
                for scene in item['scenes']:
                    scene['narration'] = language + ': ' + scene['narration']
                    if name.startswith('lessons_'):
                        scene['speech_text'] = spoken_form(scene['narration'], language)
            review = dict(lesson=identity, language=language,
                english_sha256=hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                review=dict(kind='Unit fixture editorial review', native_speaker_signoff=False, listening_review=False),
                scenes=[s['narration'] for s in after['scenes']],
                **{key: after[key] for key in ('title', 'section', 'description', 'objectives', 'prerequisite')})
            write(stage / 'production' / identity / f'review.{language}.json', review)
        write(baseline / 'web/catalog' / name, dict(schema=1, lessons=[other, before]))
        write(stage / 'catalog' / name, dict(schema=1, lessons=[other, after]))
    write(repo / 'tools/tutorials/lessons' / f'{identity}.json', canonical)
    write(stage / 'production' / identity / 'lesson.en.json', canonical)
    for language, offered in voices.items():
        localized = next(l for l in json.loads((stage / 'catalog' / f'lessons_{language}.json').read_text())['lessons'] if l['id'] == identity)
        original = next(l for l in json.loads((baseline / 'web/catalog' / f'lessons_{language}.json').read_text())['lessons'] if l['id'] == identity)
        for voice in offered:
            for root, lesson, payload in [(baseline / 'media_host' / identity, original, b'old'),
                                          (stage / 'production' / identity, localized, b'new')]:
                audio = root / 'audio' / language / (voice + '.m4a')
                audio.parent.mkdir(parents=True, exist_ok=True)
                audio.write_bytes(payload + language.encode() + voice.encode())
                timing = dict(schema=1, language=language, voice=voice, total_duration=2.,
                    media_sha256=patch.digest(audio),
                    render_inputs=dict(lesson=identity, language=language, voice=voice, scenes=lesson['scenes']),
                    scenes=[dict(scene=i+1, speech_start=float(i), speech_end=i+.8, scene_end=i+1.,
                                 duration=1., text=s['narration']) for i, s in enumerate(lesson['scenes'])])
                write(audio.with_suffix('.json'), timing)
    for relative in [f'media_host/{identity}/video/{identity}_silent.mp4',
                     f'media_host/{identity}/web/{identity}_silent.mp4', f'web/production/{identity}/poster.jpg']:
        path = baseline / relative;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(b'original visual fixture')
    javascript = dict(schema=1, lessons=[other, dict(old, silent=f'{identity}/video/{identity}_silent.mp4')])
    (baseline / 'web/lesson_catalog.js').write_text('"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze(' + json.dumps(javascript) + ');\n')
    records = [dict(path=p.relative_to(baseline).as_posix(), sha256=patch.digest(p), bytes=p.stat().st_size)
               for p in baseline.rglob('*') if p.is_file()]
    manifest = dict(files=records, ready_lessons=2, routes=2, hosted_web_lessons=[identity], ceiling_bytes=10000000)
    write(baseline / 'release-manifest.json', manifest)
    cases = [dict(lesson=l, passed=True) for l in ['01_intro', identity]]
    manifest_hash=patch.digest(baseline / 'release-manifest.json')
    write(baseline / 'checks/candidate-browser-checks.json', dict(passed=True, manifest_sha256=manifest_hash, ready_playback_cases=cases))
    commit='a'*40;count=sum(r['path'].startswith('media_host/') for r in records)
    receipt=dict(commit=commit, manifest_sha256=manifest_hash, tag='fixture',
        media_root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/'+commit,media_files=count,
        readback=dict(commit=commit, passed=True,files_expected=count,metadata_matched=count,
                      downloaded_sha256_matched=count,metadata_failures=[],download_failures=[]),
        pages=dict(ready=2,routes=2,hosted_web_copies=1,index_sha256='index'))
    write(baseline / 'publication-receipt.json', receipt)
    write(baseline / 'checks/published-media-browser-checks.json',dict(passed=True,
        manifest_sha256=manifest_hash,media_root=receipt['media_root'],index_sha256='index',
        placeholder_cases=[],ready_playback_cases=cases))
    monkeypatch.setattr(patch, 'REPO', repo)
    return baseline, stage, repo, identity


def test_script_refresh_preserves_exact_original_27_offerings(refreshed_scripts):
    baseline, stage, _, identity = refreshed_scripts
    catalogs, tracks = patch.script_patch_plan(baseline, stage, [identity])
    assert len(tracks) == 27
    assert [t['voice'] for t in tracks if t['language'] == 'en'] == ['af_heart']
    assert len(catalogs) == 14
    with pytest.raises(ValueError, match='preserve every catalog byte'):
        patch.patch_plan(baseline, stage, [identity])


@pytest.mark.parametrize('change', ['metadata','visual','order','extra_scene','other_lesson',
    'global_metadata','extra_voice','missing_voice','review_missing','review_stale','review_text',
    'audio_identity','native_clock','clock_order','clock_source_text','clock_hash'])
def test_script_refresh_rejects_non_script_or_unreviewed_drift(refreshed_scripts, change):
    baseline, stage, repo, identity = refreshed_scripts
    catalog=stage / 'catalog/lessons_en.json'
    data=json.loads(catalog.read_text());lesson=data['lessons'][1]
    if change in {'metadata','visual','order','extra_scene','other_lesson','global_metadata','extra_voice'}:
        if change=='metadata':lesson['title']='Different metadata'
        elif change=='visual':lesson['scenes'][0]['visual']='invented'
        elif change=='order':lesson['scenes'].reverse()
        elif change=='extra_scene':lesson['scenes'].append(deepcopy(lesson['scenes'][0]))
        elif change=='other_lesson':data['lessons'][0]['scenes'][0]['narration']='Unrelated change'
        elif change=='global_metadata':data['schema']=2
        else:lesson['narration_voices']['en'].append('bf_alice')
        write(catalog,data)
    elif change=='missing_voice':
        (stage / 'production' / identity / 'audio/ja/jf_alpha.json').unlink()
    elif change.startswith('review_'):
        path=stage / 'production' / identity / 'review.ko.json'
        if change=='review_missing':path.unlink()
        else:
            review=json.loads(path.read_text())
            if change=='review_stale':review['english_sha256']='stale'
            else:review['scenes'][0]='Not the staged caption.'
            write(path,review)
    elif change=='audio_identity':
        path=stage / 'production' / identity / 'audio/es/ef_dora.json'
        timing=json.loads(path.read_text());timing['media_sha256']='wrong';write(path,timing)
    else:
        path=baseline / 'media_host' / identity / 'audio/en/af_heart.json'
        timing=json.loads(path.read_text())
        if change=='native_clock':timing['native_live_video']=True
        elif change=='clock_order':timing['scenes'][1]['speech_start']=.5
        elif change=='clock_source_text':timing['scenes'][0]['text']='Unbound source scene'
        else:timing['speed']=123
        write(path,timing)
    with pytest.raises((ValueError,FileNotFoundError)):
        patch.script_patch_plan(baseline,stage,[identity])
    assert not list(stage.glob('release-candidate-*'))


def test_script_fresh_validation_calls_normal_all_language_decoder(refreshed_scripts,monkeypatch):
    import build_appended_candidate
    baseline,stage,_,identity=refreshed_scripts
    catalogs,planned=patch.script_patch_plan(baseline,stage,[identity]);calls=[]
    def verify(actual_stage,lesson,actual_catalogs):
        calls.append((actual_stage,lesson['id'],actual_catalogs))
        return lesson['narration_voices'],[{k:t[k] for k in ['language','voice','audio_sha256','timing_sha256']} for t in planned]
    monkeypatch.setattr(build_appended_candidate,'verify_tracks',verify)
    patch.verify_script_tracks(stage,[identity],catalogs,planned)
    assert calls==[(stage,identity,catalogs)]
    planned[0]['audio_sha256']='not verified'
    with pytest.raises(ValueError,match='Normal fresh verification'):
        # Return the original manifest identities, not the changed expectation.
        def wrong(*args):return catalogs['lessons_en.json']['lessons'][1]['narration_voices'],[]
        monkeypatch.setattr(build_appended_candidate,'verify_tracks',wrong)
        patch.verify_script_tracks(stage,[identity],catalogs,planned)


def test_script_candidate_preserves_visuals_and_independent_old_clock(refreshed_scripts,monkeypatch):
    baseline,stage,_,identity=refreshed_scripts
    monkeypatch.setattr(patch,'validate',lambda *args,**kwargs:None)
    monkeypatch.setattr(patch,'verify_script_tracks',lambda *args:None)
    candidate=patch.build(baseline,stage,[identity],refresh_scripts=True)
    for relative in [f'media_host/{identity}/video/{identity}_silent.mp4',
                     f'media_host/{identity}/web/{identity}_silent.mp4', f'web/production/{identity}/poster.jpg']:
        assert (candidate/relative).read_bytes()==(baseline/relative).read_bytes()
    original=baseline / 'media_host' / identity / 'audio/en/af_heart.json'
    independent=candidate / 'media_host' / identity / 'video/original-visual-timings.json'
    assert independent.read_bytes()==original.read_bytes()
    assert (candidate/'media_host'/identity/'audio/en/af_heart.json').read_bytes()!=original.read_bytes()
    runtime=patch.parse_javascript((candidate/'web/lesson_catalog.js').read_text())
    assert runtime['lessons'][1]['visual_timings']==f'{identity}/video/original-visual-timings.json'
    assert runtime['lessons'][0]==patch.parse_javascript((baseline/'web/lesson_catalog.js').read_text())['lessons'][0]
    report=json.loads((candidate/'release-manifest.json').read_text())
    assert report['script_refresh'] is True and report['fresh_capture'] is False
    assert report['raw_capture_revalidated'] is False and len(report['audio_patch_tracks'])==27


@pytest.mark.parametrize('change',['readback','published_case','published_root','pages'])
def test_script_visual_reuse_requires_original_complete_publication(refreshed_scripts,change):
    baseline,_,_,_=refreshed_scripts
    path=baseline/('checks/published-media-browser-checks.json' if change.startswith('published') else 'publication-receipt.json')
    data=json.loads(path.read_text())
    if change=='readback':data['readback']['downloaded_sha256_matched']-=1
    elif change=='published_case':data['ready_playback_cases'][0]['passed']=False
    elif change=='published_root':data['media_root']='another revision'
    else:data['pages']['hosted_web_copies']=0
    write(path,data)
    with pytest.raises(ValueError):
        patch.require_published_script_baseline(baseline,json.loads((baseline/'release-manifest.json').read_text()))


@pytest.mark.parametrize('change', [None, 'drop_active', 'ready', 'routes', 'hosted_web_copies'])
def test_script_publication_requires_exact_archive_projection(refreshed_scripts, monkeypatch, change):
    import archive_lessons
    baseline, _, _, identity = refreshed_scripts
    monkeypatch.setattr(archive_lessons.archived, 'ARCHIVED_IDS', (identity,))
    receipt = patch.read(baseline / 'publication-receipt.json')
    checks = patch.read(baseline / 'checks/published-media-browser-checks.json')
    checks['ready_playback_cases'] = [row for row in checks['ready_playback_cases']
                                     if row['lesson'] != identity]
    assert len(checks['ready_playback_cases']) == 1
    receipt['pages'].update(ready=1, routes=1, hosted_web_copies=0)
    if change == 'drop_active':
        checks['ready_playback_cases'] = []
        receipt['pages'].update(ready=0, routes=0, hosted_web_copies=0)
    elif change is not None:
        receipt['pages'][change] += 1
    write(baseline / 'publication-receipt.json', receipt)
    write(baseline / 'checks/published-media-browser-checks.json', checks)
    manifest = patch.read(baseline / 'release-manifest.json')
    if change is None:
        patch.require_published_script_baseline(baseline, manifest)
    else:
        with pytest.raises(ValueError, match='exact complete published'):
            patch.require_published_script_baseline(baseline, manifest)
