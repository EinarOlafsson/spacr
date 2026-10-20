"""Retention uses explicit originals and ignores only generated voice inventory."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_completed_matrix as matrix
import verify_staged_audio as audio
from retain_narration import digest
from stage_lesson import read, write


@pytest.fixture
def source_files(tmp_path, monkeypatch):
    stage, original, baseline, repo = [tmp_path / name for name in ('stage', 'original', 'published', 'repo')]
    identity = '02_install_spacr'
    lesson = dict(id=identity, title='Title', scenes=[dict(narration='Unchanged words')])
    renderer = repo / 'tools/render_all_voices.py'
    owning_renderer = Path(__file__).resolve().parents[1] / 'authoring/tools/render_all_voices.py'
    inventory = matrix.voice_matrix(owning_renderer)
    monkeypatch.setattr(matrix, 'voice_matrix', lambda path: inventory)
    monkeypatch.setattr(matrix, 'REPO', repo)
    monkeypatch.setattr(audio, 'REPO', repo)
    monkeypatch.setattr(audio, 'DEFAULT_STAGE', original / 'old-default-stage')
    expected_tracks = []
    for language in [*inventory, *matrix.CAPTION_LANGUAGES]:
        prefix = 'captions' if language in matrix.CAPTION_LANGUAGES else 'lessons'
        name = f'{prefix}_{language}.json'
        for directory in (stage / 'catalog', original / 'catalog', baseline,
                          repo / 'docs/source/_extra/tutorials/catalog'):
            item = {**lesson, 'narration_voices': {**inventory, 'en': sorted(inventory['en'])}}
            if directory == original / 'catalog':
                item = dict(lesson) if language == 'en' else {**lesson, 'narration_voices': {'en': ['af_heart']}}
            write(directory / name, dict(lessons=[item]))
        for voice in inventory.get(language, []):
            for directory in (original, stage):
                path = directory / 'production' / identity / 'audio' / language / f'{voice}.m4a'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f'{language}/{voice} original'.encode())
                write(path.with_suffix('.json'), dict(language=language, voice=voice,
                    media_sha256=digest(path), render_inputs=dict(lesson=identity, language=language,
                        voice=voice, scenes=deepcopy(lesson['scenes']))))
            expected_tracks.append(dict(language=language, voice=voice, errors=[], audio_sha256=digest(path)))
    folder = stage / 'production' / identity
    write(folder / 'lesson.en.json', lesson)
    write(folder / 'audio-checks.all.json', dict(lesson=identity, passed=True,
        tracks_checked=50, tracks=expected_tracks))
    cases = [(language, voices[0], None) for language, voices in inventory.items()]
    cases += [('en', 'af_heart', language) for language in matrix.CAPTION_LANGUAGES]
    for language, voice, caption in cases:
        tag = f'{language}-{voice}' + (f'-captions-{caption}' if caption else '')
        path = stage / 'browser' / identity / tag / 'playback-checks.json'
        write(path, dict(lesson=identity, passed=True,
            loaded_audio_sha256=digest(folder / 'audio' / language / f'{voice}.m4a'), scene_count=1,
            scope=f'{language}/{voice} playback and scene links only', navigation_contains_staged_lesson=True,
            caption_language=caption, caption_scenes_match_staging=True, caption_webvtt_sha256='a' * 64,
            seek_playback_clocks=dict(mediaError=None, audio=1, video=1, expectedVideo=1,
                                      audioDuration=2, videoDuration=2)))
    image = folder / 'image.png';image.write_bytes(b'fixture pixels')
    write(folder / 'visual.json', dict(scenes=[dict(image=image.name, capture_sha256=digest(image))]))
    (folder / 'video').mkdir()
    (folder / 'video' / f'{identity}_silent.mp4').write_bytes(b'fixture master')
    (folder / 'poster.jpg').write_bytes(b'fixture poster')
    monkeypatch.setattr(matrix.subprocess, 'check_output', lambda *args, **kwargs: json.dumps(dict(
        streams=[dict(codec_type='video', width=3840, height=2160, r_frame_rate='30/1')])))
    verify = SimpleNamespace(LANGUAGES={language: ('code', voices) for language, voices in inventory.items()},
                             check_track=lambda path: [])
    monkeypatch.setitem(sys.modules, 'verify_audio_release', verify)
    monkeypatch.setenv('SPACR_TUTORIAL_WORKSPACE', str(stage))
    return SimpleNamespace(stage=stage, original=original, baseline=baseline, repo=repo,
                            identity=identity, renderer=renderer, inventory=inventory)


def test_matrix_current_renderer_inventory_and_explicit_actual_original(source_files):
    r = source_files
    assert not r.renderer.exists()
    result = matrix.check(r.stage, r.identity, r.renderer, retained_narration=True,
                          retained_from=r.original, baseline=r.baseline)
    assert result['passed'] is True and result['unique_final_tracks'] == 50
    assert len(result['retained_narration']['tracks']) == 50
    assert result['retained_narration']['current_runtime_freshness_claimed'] is False


@pytest.mark.parametrize('change', ['english_extra', 'canonical_extra', 'caption', 'audio'])
def test_retained_generated_field_exception_does_not_allow_other_drift(source_files, change):
    r = source_files
    if change == 'english_extra':
        for directory in (r.stage / 'catalog', r.original / 'catalog', r.baseline):
            path = directory / 'lessons_en.json';value = read(path)
            value['lessons'][0]['other_generated_field'] = True;write(path, value)
    elif change == 'canonical_extra':
        path = r.stage / 'production' / r.identity / 'lesson.en.json';value = read(path)
        value['extra_canonical_field'] = True;write(path, value)
    elif change == 'caption':
        path = r.stage / 'catalog/captions_sv.json';value = read(path)
        value['lessons'][0]['scenes'][0]['narration'] = 'Different caption';write(path, value)
    else:
        (r.stage / 'production' / r.identity / 'audio/en/af_heart.m4a').write_bytes(b'changed')
    with pytest.raises(ValueError):
        matrix.check(r.stage, r.identity, r.renderer, retained_narration=True,
                     retained_from=r.original, baseline=r.baseline)


def test_fresh_matrix_keeps_exact_english_comparison(source_files):
    r = source_files
    with pytest.raises(ValueError, match='Staged English differs'):
        matrix.check(r.stage, r.identity, r.renderer)
    with pytest.raises(ValueError, match='only with retained-narration'):
        matrix.check(r.stage, r.identity, r.renderer, retained_from=r.original)


@pytest.mark.parametrize('explicit', [False, True])
def test_audio_cli_retained_original_and_legacy_default_both_preserve_bytes(source_files, explicit):
    r = source_files
    args = ['--lesson', r.identity, '--languages', 'en', '--stage', str(r.stage), '--retained-narration']
    if explicit:
        args.extend(['--retained-from', str(r.original)])
    assert audio.main(args) == 0
    report = read(r.stage / 'production' / r.identity / 'audio-checks.en.json')
    assert report['tracks_checked'] == 24 and report['passed'] is True
    assert report['retained_narration'] is True and report['current_runtime_freshness_claimed'] is False


def test_audio_cli_rejects_original_flag_in_fresh_mode_before_source_reads(tmp_path):
    with pytest.raises(ValueError, match='only with retained-narration'):
        audio.main(['--lesson', 'missing', '--languages', 'en', '--stage', str(tmp_path),
                    '--retained-from', str(tmp_path / 'original')])


def test_audio_cli_does_not_ignore_wrong_original(source_files):
    r = source_files
    with pytest.raises(FileNotFoundError):
        audio.main(['--lesson', r.identity, '--languages', 'en', '--stage', str(r.stage),
                    '--retained-narration', '--retained-from', str(r.stage / 'missing')])


@pytest.mark.parametrize('retained', [False, True])
def test_browser_cli_selects_protected_retained_or_normal_validator_before_browser(source_files, monkeypatch, retained):
    import build_navigation
    import native_live_timing
    import retained_visual_refresh
    pytest.importorskip('playwright', reason='Playwright is installed only in the tutorial render Python')
    import verify_staged_lesson as browser
    r = source_files
    monkeypatch.setattr(browser, 'REPO', r.repo)
    monkeypatch.setattr(browser, 'DEFAULT_STAGE', browser.DEFAULT_STAGE)
    monkeypatch.setattr(browser, 'WORKSPACE', browser.WORKSPACE)
    player = r.repo / 'tools/tutorials/authoring/web/app_v2.js'
    player.parent.mkdir(parents=True);player.write_text('fixture player')
    monkeypatch.setattr(build_navigation, 'build', lambda *args: {})
    calls = []

    class CheckedStop(Exception): pass

    def check(*args):
        calls.append(args)
        raise CheckedStop('Protected checker reached before browser launch')

    monkeypatch.setattr(retained_visual_refresh, 'checked_retained_visual_timing', check)
    monkeypatch.setattr(native_live_timing, 'checked_native_timing', check)
    monkeypatch.setattr(browser, 'sync_playwright', lambda: pytest.fail('No browser should launch'))
    args = ['--stage', str(r.stage), '--lesson', r.identity, '--candidate-player']
    if retained: args.extend(['--retain-visual-from', str(r.baseline)])
    with pytest.raises(CheckedStop): browser.main(args)
    assert len(calls) == 1
    assert calls[0] == ((r.stage, r.identity, r.baseline) if retained else (r.stage, r.identity))


def test_browser_rejects_whole_retention_and_visual_refresh_together(tmp_path):
    pytest.importorskip('playwright', reason='Playwright is installed only in the tutorial render Python')
    import verify_staged_lesson as browser
    with pytest.raises(SystemExit):
        browser.main(['--retained-media', '--retain-visual-from', str(tmp_path)])
