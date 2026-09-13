"""The new lesson is a measured API example, with real synchronized media."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_capture_proves_the_api_and_discloses_the_unfinished_gui():
    report = read(ROOT / 'evidence/2026-09-12_embeddings_final_checks.json')
    assert report['published'] is False
    assert report['gui_workflow_completed'] is False
    capture = report['capture']
    assert capture['accepted'] is True
    assert capture['gui']['crops_injected'] is False
    assert capture['gui']['embed_enabled'] is False
    actual = capture['terminal']['runs']
    assert [run['shape'] for run in actual] == [[16, 1536], [16, 512]]
    assert actual[0]['sources'] == actual[1]['sources']
    for run in actual:
        assert run['helper_sha256'] == sha(ROOT / 'embeddings_example.py')
        assert run['source_unchanged'] is True
        assert run['spec']['device'] == 'cpu'
        assert run['biology_validated'] is False
        assert run['stain_mapping_verified'] is False
        assert len(run['weights_sha256']) == 64
        proof = run['verification']
        assert proof['matrix_cells_checked'] == run['shape'][0] * run['shape'][1]
        assert all(proof[key] is True for key in
                   ('ordered_identities_match', 'npy_exact', 'csv_float32_exact'))


def test_every_final_voice_and_shared_video_are_the_ones_in_the_candidate():
    report = read(ROOT / 'evidence/2026-09-12_embeddings_final_checks.json')
    matrix = report['matrix']
    assert matrix['passed'] is True
    assert matrix['scene_count'] == 11
    assert matrix['unique_final_tracks'] == len(matrix['tracks']) == 50
    assert len(matrix['browser_reports']) == 14
    assert report['native_speaker_signoff'] is False
    assert report['human_listening_signoff'] is False
    candidate = ROOT / 'release_candidate'
    assert report['candidate_manifest_sha256'] == sha(candidate / 'release-manifest.json')
    records = {r['path']: r for r in read(candidate / 'release-manifest.json')['files']}
    for track in matrix['tracks']:
        path = f"media_host/77_embeddings/audio/{track['language']}/{track['voice']}.m4a"
        assert records[path]['sha256'] == track['sha256']
    master = 'media_host/77_embeddings/video/77_embeddings_silent.mp4'
    assert records[master]['sha256'] == matrix['master_sha256']
    english = read(ROOT / 'lessons/77_embeddings.json')
    canonical = hashlib.sha256(json.dumps(english, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert matrix['canonical_english_sha256'] == canonical


def test_heart_native_captions_match_each_actual_sentence_not_estimated_scene_slots():
    timing_path = ROOT / 'evidence/2026-09-12_embeddings_heart_timing.json'
    timing = read(timing_path)
    report = read(ROOT / 'evidence/2026-09-12_embeddings_final_checks.json')
    assert report['heart_timing_sha256'] == sha(timing_path)
    manifest = read(ROOT / 'release_candidate/release-manifest.json')
    record = next(r for r in manifest['files']
                  if r['path'] == 'media_host/77_embeddings/audio/en/af_heart.json')
    assert record['sha256'] == sha(timing_path)
    browser = read(ROOT / 'release_candidate/candidate-browser-checks.json')
    case = next(c for c in browser['ready_playback_cases'] if c['lesson'] == '77_embeddings')
    assert report['candidate_browser_case'] == case
    assert case['audio_sha256'] == timing['media_sha256']
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    assert len(case['sentence_cue_checks']) == len(sentences) > 11
    for actual, observed in zip(sentences, case['sentence_cue_checks']):
        midpoint = (actual['speech_start'] + actual['speech_end']) / 2
        assert observed['requested_audio_time'] == pytest.approx(midpoint)
        assert abs(observed['audio'] - midpoint) < 1
        assert observed['text'] == actual['text']
        assert actual['text'] in observed['cues']
