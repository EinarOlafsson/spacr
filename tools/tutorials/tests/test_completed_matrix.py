"""Final reports must describe the actual current files, not an earlier render."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_completed_matrix import reconcile_audio, reconcile_browser, voice_matrix


@pytest.fixture
def audio():
    hashes = {('en', 'voice_a'): 'a' * 64, ('fr', 'voice_b'): 'b' * 64}
    report = {'lesson': 'example', 'passed': True, 'tracks_checked': 2,
              'tracks': [{'language': lang, 'voice': voice, 'errors': [], 'audio_sha256': value}
                         for (lang, voice), value in hashes.items()]}
    return report, hashes


def test_all_final_tracks_and_identical_recheck_pass(audio):
    report, hashes = audio
    assert reconcile_audio([report, deepcopy(report)], 'example', set(hashes), hashes) == hashes


@pytest.mark.parametrize('change', [
    lambda r: r.update(passed=False), lambda r: r.update(lesson='another'),
    lambda r: r.update(tracks_checked=50), lambda r: r['tracks'][0].update(errors=['decode failed']),
    lambda r: r['tracks'][0].update(audio_sha256='old'),
    lambda r: r['tracks'][0].update(voice='unexpected'),
])
def test_failed_wrong_or_stale_audio_rejected(audio, change):
    report, hashes = audio
    change(report)
    with pytest.raises(ValueError):
        reconcile_audio([report], 'example', set(hashes), hashes)


def test_one_missing_voice_cannot_be_replaced_by_duplicate(audio):
    report, hashes = audio
    report['tracks'][1] = deepcopy(report['tracks'][0])
    with pytest.raises(ValueError, match='every preserved voice'):
        reconcile_audio([report], 'example', set(hashes), hashes)


@pytest.fixture
def browser():
    return {'lesson': 'example', 'passed': True, 'loaded_audio_sha256': 'a' * 64,
            'scope': 'en/voice_a playback and scene links only', 'scene_count': 8,
            'navigation_contains_staged_lesson': True,
            'seek_playback_clocks': {'audio': 20, 'video': 19, 'expectedVideo': 19.1,
                                     'audioDuration': 40, 'videoDuration': 38, 'mediaError': None}}


def test_actual_synced_browser_passes(browser):
    reconcile_browser(browser, 'example', 'en', 'voice_a', None, 'a' * 64, 8)


def test_caption_requires_independent_evidence(browser):
    with pytest.raises(ValueError, match='caption evidence'):
        reconcile_browser(browser, 'example', 'en', 'voice_a', 'de', 'a' * 64, 8)
    browser.update(caption_language='de', caption_scenes_match_staging=True,
                   caption_webvtt_sha256='c' * 64)
    reconcile_browser(browser, 'example', 'en', 'voice_a', 'de', 'a' * 64, 8)


@pytest.mark.parametrize('field,value', [('passed', False), ('loaded_audio_sha256', 'stale'),
    ('lesson', 'other'), ('scope', 'fr/voice_a playback and scene links only'),
    ('scene_count', 7), ('navigation_contains_staged_lesson', False)])
def test_wrong_browser_identity_rejected(browser, field, value):
    browser[field] = value
    with pytest.raises(ValueError, match='browser report'):
        reconcile_browser(browser, 'example', 'en', 'voice_a', None, 'a' * 64, 8)


@pytest.mark.parametrize('field,value', [('video', 20), ('expectedVideo', float('nan')),
    ('audio', float('inf')), ('audioDuration', 0), ('videoDuration', -1), ('mediaError', 'decode')])
def test_invalid_browser_clocks_rejected(browser, field, value):
    browser['seek_playback_clocks'][field] = value
    with pytest.raises(ValueError, match='browser clocks'):
        reconcile_browser(browser, 'example', 'en', 'voice_a', None, 'a' * 64, 8)


def test_real_preserved_voice_inventory_is_read_without_import():
    path = Path(__file__).resolve().parents[1] / 'authoring/tools/render_all_voices.py'
    matrix = voice_matrix(path)
    assert len(matrix) == 8 and sum(map(len, matrix.values())) == 50
    assert matrix['en'][0] == 'af_heart' and len(matrix['en']) == 24


def test_missing_inventory_is_rejected(tmp_path):
    path = tmp_path / 'renderer.py'
    path.write_text('LANGUAGES = {}')
    with pytest.raises(ValueError, match='matrix changed'):
        voice_matrix(path)
