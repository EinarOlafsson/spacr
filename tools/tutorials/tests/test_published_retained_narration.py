"""Published retention requires immutable proof, never generic stale-cache acceptance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import types

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import published_retained_narration as retained
from check_completed_matrix import voice_matrix
from stage_lesson import write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_release(base):
    files = [dict(path=p.relative_to(base).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
             for prefix in ('media_host', 'web') for p in sorted((base / prefix).rglob('*')) if p.is_file()]
    write(base / 'release-manifest.json', dict(files=files))
    count = sum(r['path'].startswith('media_host/') for r in files)
    commit = 'a' * 40
    write(base / 'publication-receipt.json', dict(manifest_sha256=sha(base / 'release-manifest.json'),
        commit=commit, tag='fixture-published', media_root='https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + commit,
        media_files=count, readback=dict(commit=commit, passed=True, files_expected=count,
        metadata_matched=count, downloaded_sha256_matched=count, metadata_failures=[], download_failures=[])))


@pytest.fixture
def published(tmp_path, monkeypatch):
    from audit_staged_catalogs import CATALOGS
    from stage_lesson import REPO
    inventory = voice_matrix(REPO / 'tools/tutorials/authoring/tools/render_all_voices.py')
    repo, base, stage = tmp_path / 'repo', tmp_path / 'published', tmp_path / 'fresh-stage'
    renderer = repo / 'tools/tutorials/authoring/tools/render_all_voices.py'
    renderer.parent.mkdir(parents=True)
    shutil.copyfile(REPO / 'tools/tutorials/authoring/tools/render_all_voices.py', renderer)
    identity = '49_fixture'
    english = dict(id=identity, number=49, scenes=[dict(visual='actual_result', narration='Inspect the result.')])
    write(repo / 'tools/tutorials/lessons' / (identity + '.json'), english)
    localized = {}
    for name in CATALOGS:
        language = name.split('_', 1)[1][:-5]
        lesson = {**deepcopy(english), 'narration_voices': inventory}
        if language != 'en': lesson['scenes'][0]['narration'] = 'Reviewed ' + language
        catalog = dict(lessons=[lesson])
        write(base / 'web/catalog' / name, catalog)
        write(repo / 'tools/tutorials/authoring/catalog' / name, catalog)
        localized[language] = lesson
    proofs = []
    for language, voices in inventory.items():
        for voice in voices:
            audio = base / f'media_host/{identity}/audio/{language}/{voice}.m4a'
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(('published PCM fixture ' + language + voice).encode())
            speed = .96 if voice == 'af_aoede' else 1.0
            inputs = dict(lesson=identity, language=language, voice=voice, speed=speed,
                          scenes=[dict(narration=localized[language]['scenes'][0]['narration'])],
                          pronunciation={'sha256': 'old-reviewed-pronunciation'}, synthesis_runtime={'state': 'published-runtime'})
            fingerprint = hashlib.sha256(json.dumps(inputs, ensure_ascii=False, sort_keys=True,
                                                   separators=(',', ':')).encode()).hexdigest()
            timing = dict(language=language, voice=voice, speed=speed, media_sha256=sha(audio),
                          media_bytes=audio.stat().st_size, render_inputs=inputs, render_fingerprint=fingerprint,
                          scenes=[dict(scene=1, text=localized[language]['scenes'][0]['narration'])])
            write(audio.with_suffix('.json'), timing)
            proofs.append(dict(language=language, voice=voice, audio_sha256=sha(audio),
                               timing_sha256=sha(audio.with_suffix('.json')), render_fingerprint=fingerprint))
    master = base / f'media_host/{identity}/video/{identity}_silent.mp4'
    master.parent.mkdir(parents=True)
    master.write_bytes(b'unit fixture: no media decode requested')
    native = dict(kind='native_visual_timing', lesson=identity, native_live_video=True,
                  fps=30, playback_speed=1, looped_or_stretched=False, total_duration=10,
                  coverage=dict(accepted=True), source_tracks=proofs,
                  scenes=[dict(scene=1, visual='actual_result', speech_start=0, scene_end=10, duration=10, native_live_video=True)])
    write(master.with_name('native-live-timings.json'), native)
    bind_release(base)
    monkeypatch.setattr(retained, 'REPO', repo)
    return base, stage, identity, repo


def test_exact_fifty_published_tracks_keep_bytes_speed_and_fingerprint(published):
    base, stage, identity, _ = published
    retained.prepare(stage, identity, base)
    tracks, proofs = retained.checked_published_narration(stage, identity)
    assert len(tracks) == len(proofs) == 50
    aoede = next(t for t in tracks if t['voice'] == 'af_aoede')
    assert aoede['speed'] == .96 and aoede['render_inputs']['pronunciation']['sha256'] == 'old-reviewed-pronunciation'
    for source in (base / f'media_host/{identity}/audio').rglob('*'):
        if source.is_file():
            target = stage / 'production' / identity / 'audio' / source.relative_to(base / f'media_host/{identity}/audio')
            assert target.read_bytes() == source.read_bytes()
    record = json.loads((stage / 'production' / identity / retained.RECORD).read_text())
    for flag in ['resynthesized', 'fresh_capture', 'current_runtime_freshness_claimed', 'new_translation_review_claimed', 'human_listening_review']:
        assert record[flag] is False


@pytest.mark.parametrize('change', ['audio', 'timing', 'catalog', 'source_audio', 'source_catalog', 'canonical', 'current_translation', 'master', 'native', 'receipt', 'record', 'extra_track'])
def test_retained_binding_rejects_any_staged_or_published_drift(published, change):
    base, stage, identity, repo = published
    retained.prepare(stage, identity, base)
    root = stage / 'production' / identity
    targets = dict(audio=root / 'audio/en/af_heart.m4a', timing=root / 'audio/en/af_heart.json',
                   catalog=stage / 'catalog/lessons_en.json', source_audio=base / f'media_host/{identity}/audio/en/af_heart.m4a',
                   source_catalog=base / 'web/catalog/lessons_en.json', canonical=repo / 'tools/tutorials/lessons' / (identity + '.json'),
                   current_translation=repo / 'tools/tutorials/authoring/catalog/lessons_es.json',
                   master=base / f'media_host/{identity}/video/{identity}_silent.mp4',
                   native=base / f'media_host/{identity}/video/native-live-timings.json', receipt=base / 'publication-receipt.json',
                   record=root / retained.RECORD, extra_track=root / 'audio/en/unpublished.m4a')
    path = targets[change]
    if change in {'audio', 'source_audio', 'master', 'extra_track'}:
        path.write_bytes(b'changed')
    else:
        value = json.loads(path.read_text())
        if change == 'current_translation':value['lessons'][0]['scenes'][0]['narration'] = 'Changed'
        elif change == 'canonical':value['scenes'][0]['narration'] = 'Changed'
        elif change == 'receipt':value['readback']['passed'] = False
        else:value['unexpected'] = True
        write(path, value)
    with pytest.raises((ValueError, KeyError)):
        retained.checked_published_narration(stage, identity)


@pytest.mark.parametrize('change', ['duplicate_proof', 'wrong_fingerprint', 'missing_track', 'invalid_speed'])
def test_fully_rebound_candidate_still_requires_genuine_native_source_proofs(published, change):
    base, stage, identity, _ = published
    native_path = base / f'media_host/{identity}/video/native-live-timings.json'
    native = json.loads(native_path.read_text())
    if change == 'duplicate_proof': native['source_tracks'][-1] = native['source_tracks'][0]
    elif change == 'wrong_fingerprint': native['source_tracks'][0]['render_fingerprint'] = 'wrong'
    elif change == 'missing_track':
        native['source_tracks'].pop()
    else:
        path = base / f'media_host/{identity}/audio/en/af_heart.json'
        timing = json.loads(path.read_text());timing['speed'] = .5;write(path, timing)
    write(native_path, native);bind_release(base)
    with pytest.raises(ValueError):retained.prepare(stage, identity, base)
    assert not stage.exists()


def test_fresh_stage_is_exclusive_and_no_partial_stage_is_overwritten(published):
    base, stage, identity, _ = published
    stage.mkdir();(stage / 'partial').write_bytes(b'preserve')
    with pytest.raises(FileExistsError):retained.prepare(stage, identity, base)
    assert (stage / 'partial').read_bytes() == b'preserve'


def test_native_dispatch_requires_exact_retained_record(published, monkeypatch):
    import native_live_timing as native
    base, stage, identity, _ = published
    retained.prepare(stage, identity, base)
    english = json.loads((stage / 'production' / identity / 'lesson.en.json').read_text())
    tracks, proofs = native.verified_narration_tracks(stage, identity, english)
    assert len(tracks) == len(proofs) == 50
    wrong = deepcopy(english);wrong['scenes'][0]['narration'] = 'Changed'
    with pytest.raises(ValueError):native.verified_narration_tracks(stage, identity, wrong)
    write(stage / 'production' / identity / retained.RECORD, {})
    with pytest.raises(ValueError):native.verified_narration_tracks(stage, identity, english)


def test_record_absence_keeps_original_fresh_fingerprint_validation(tmp_path, monkeypatch):
    import native_live_timing as native
    identity = '49_fresh';stage = tmp_path;root = stage / 'production' / identity
    lesson = dict(id=identity, scenes=[dict(visual='view', narration='Fresh text')])
    write(stage / 'catalog/lessons_en.json', dict(lessons=[lesson]))
    audio = root / 'audio/en/voice.m4a';audio.parent.mkdir(parents=True);audio.write_bytes(b'fresh')
    inputs = dict(synthesis_runtime={'current': True}, speed=1)
    metadata = dict(language='en', voice='voice', media_sha256=sha(audio), media_bytes=audio.stat().st_size,
                    scenes=[dict(text='Fresh text')], render_fingerprint='fresh-bound', render_inputs=inputs)
    write(audio.with_suffix('.json'), metadata)
    module = types.ModuleType('render_all_voices');module.LANGUAGES={'en': ('a', ['voice'])}
    module.narration_dialect=lambda *a:'dialect'
    module.resolve_voice_speed=lambda voice:1
    module.prepare_scene_plans=lambda *a,**k:[]
    module.track_fingerprint=lambda *a,**k:('fresh-bound', inputs)
    monkeypatch.setitem(sys.modules, 'render_all_voices', module)
    assert native.verified_narration_tracks(stage, identity, lesson)[0] == [metadata]
    metadata['render_fingerprint'] = 'stale';write(audio.with_suffix('.json'), metadata)
    with pytest.raises(ValueError, match='current render inputs'):
        native.verified_narration_tracks(stage, identity, lesson)
