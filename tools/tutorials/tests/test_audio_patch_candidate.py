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
