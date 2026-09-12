"""OPS is the verified four-tile geometry lesson, not a claimed full pipeline."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ops_capture_keeps_the_real_geometry_and_unfinished_workflow_boundary():
    report = read(ROOT / 'evidence/2026-09-12_ops_final_checks.json')
    assert report['published'] is False and report['full_ops_pipeline_completed'] is False
    assert report['gui_workflow_completed'] is False
    capture = report['capture']
    assert capture['accepted'] is True and capture['gui']['run_clicked'] is False
    run = capture['terminal']['run']
    assert run['accepted'] is True
    assert run['placed'] == run['accepted_edges'] == 4
    assert run['canvas'] == [2756, 2756]
    assert run['full_pipeline_completed'] is False
    assert run['segmentation_or_decoding_performed'] is False


def test_ops_all_fifty_tracks_and_fourteen_language_cases_match_the_package():
    report = read(ROOT / 'evidence/2026-09-12_ops_final_checks.json')
    matrix = report['matrix']
    candidate = ROOT / 'release_candidate'
    assert report['candidate_manifest_sha256'] == sha(candidate / 'release-manifest.json')
    assert matrix['passed'] is True and matrix['scene_count'] == 9
    assert matrix['unique_final_tracks'] == len(matrix['tracks']) == 50
    assert len(matrix['browser_reports']) == 14
    assert report['native_speaker_signoff'] is False and report['human_listening_signoff'] is False
    records = {item['path']: item for item in read(candidate / 'release-manifest.json')['files']}
    for track in matrix['tracks']:
        key = f"media_host/76_ops/audio/{track['language']}/{track['voice']}.m4a"
        assert records[key]['sha256'] == track['sha256']
    assert records['media_host/76_ops/video/76_ops_silent.mp4']['sha256'] == matrix['master_sha256']
    english = read(ROOT / 'lessons/76_ops.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert matrix['canonical_english_sha256'] == canonical


def test_ops_heart_native_captions_follow_every_actual_narrated_sentence():
    path = ROOT / 'evidence/2026-09-12_ops_heart_timing.json'
    timing = read(path)
    report = read(ROOT / 'evidence/2026-09-12_ops_final_checks.json')
    assert report['heart_timing_sha256'] == sha(path)
    browser = read(ROOT / 'release_candidate/candidate-browser-checks.json')
    case = next(item for item in browser['ready_playback_cases'] if item['lesson'] == '76_ops')
    assert case == report['candidate_browser_case'] and case['audio_sha256'] == timing['media_sha256']
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    assert len(case['sentence_cue_checks']) == len(sentences) > 9
    for expected, actual in zip(sentences, case['sentence_cue_checks']):
        midpoint = (expected['speech_start'] + expected['speech_end']) / 2
        assert actual['requested_audio_time'] == midpoint and abs(actual['audio'] - midpoint) < 1
        assert actual['text'] == expected['text'] and expected['text'] in actual['cues']
