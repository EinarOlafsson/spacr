"""Both model lessons have actual scoped recordings and fully checked media."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_promotion import validate_scope

LESSONS = [('21_model_compare', 8), ('22_model_zoo', 7)]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('identity,scenes', LESSONS)
def test_actual_scope_and_all_fifty_tracks_are_bound_to_the_final_candidate(identity, scenes):
    label = identity.split('_', 1)[1]
    report = read(ROOT / 'evidence' / f'2026-09-12_{label}_final_checks.json')
    assert report['lesson'] == identity and report['published'] is False
    assert report['gui_inference_or_benchmark_completed'] is False
    assert report['native_speaker_signoff'] is False and report['human_listening_signoff'] is False
    validate_scope(identity, report['capture'])
    matrix = report['matrix']
    assert matrix['passed'] is True and matrix['scene_count'] == scenes
    assert matrix['unique_final_tracks'] == len(matrix['tracks']) == 50
    assert len(matrix['browser_reports']) == 14
    candidate = ROOT / 'release_candidate'
    assert report['candidate_manifest_sha256'] == sha(candidate / 'release-manifest.json')
    records = {item['path']: item for item in read(candidate / 'release-manifest.json')['files']}
    for track in matrix['tracks']:
        key = f"media_host/{identity}/audio/{track['language']}/{track['voice']}.m4a"
        assert records[key]['sha256'] == track['sha256']
    assert records[f'media_host/{identity}/video/{identity}_silent.mp4']['sha256'] == matrix['master_sha256']
    english = read(ROOT / 'lessons' / (identity + '.json'))
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert matrix['canonical_english_sha256'] == canonical


@pytest.mark.parametrize('identity,scenes', LESSONS)
def test_every_model_heart_caption_is_observed_at_its_actual_narrated_time(identity, scenes):
    label = identity.split('_', 1)[1]
    path = ROOT / 'evidence' / f'2026-09-12_{label}_heart_timing.json'
    timing = read(path)
    report = read(ROOT / 'evidence' / f'2026-09-12_{label}_final_checks.json')
    assert report['heart_timing_sha256'] == sha(path)
    browser = read(ROOT / 'release_candidate/candidate-browser-checks.json')
    case = next(item for item in browser['ready_playback_cases'] if item['lesson'] == identity)
    assert case == report['candidate_browser_case'] and case['audio_sha256'] == timing['media_sha256']
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    assert len(case['sentence_cue_checks']) == len(sentences) > scenes
    for expected, actual in zip(sentences, case['sentence_cue_checks']):
        midpoint = (expected['speech_start'] + expected['speech_end']) / 2
        assert actual['requested_audio_time'] == midpoint and abs(actual['audio'] - midpoint) < 1
        assert actual['text'] == expected['text'] and expected['text'] in actual['cues']
