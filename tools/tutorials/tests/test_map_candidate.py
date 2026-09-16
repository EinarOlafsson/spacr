"""Acceptance for the real recorded Map lesson, not a metadata-only promotion."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def test_map_candidate_has_actual_data_complete_media_and_native_caption_evidence():
    evidence = read(ROOT / 'evidence/2026-09-13_map_barcodes_final_checks.json')
    manifest = read(ROOT / 'release_candidate/release-manifest.json')
    checkpoint = read(ROOT / 'release_candidate/checkpoint.json')
    assert evidence['candidate_manifest_sha256'] == checkpoint['manifest_sha256']
    assert evidence['capture']['gui']['mapped_reads'] == 7657
    assert evidence['capture']['api']['api_two_barcodes']['mapped_reads'] == 793
    matrix = evidence['matrix']
    assert matrix['passed'] is True and matrix['unique_final_tracks'] == 50
    assert matrix['scene_count'] == 13 and len(matrix['browser_reports']) == 14
    records = {r['path']: r for r in manifest['files']}
    for track in matrix['tracks']:
        path = f"media_host/12_map_barcodes/audio/{track['language']}/{track['voice']}.m4a"
        assert records[path]['sha256'] == track['sha256']
    timing = read(ROOT / 'evidence/2026-09-13_map_barcodes_heart_timing.json')
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    actual = evidence['candidate_browser_case']
    assert actual['passed'] is True and actual['audio_sha256'] == timing['media_sha256']
    assert len(actual['sentence_cue_checks']) == len(sentences) > 30
    for sentence, cue in zip(sentences, actual['sentence_cue_checks']):
        assert cue['text'] == sentence['text'] and sentence['text'] in cue['cues']
    english = [r for r in matrix['tracks'] if r['language'] == 'en']
    phonemes = evidence['english_read_depth_phoneme_checks']
    assert len(phonemes) == len(english) == 24
    assert {r['audio_sha256'] for r in phonemes} == {r['sha256'] for r in english}
    assert all('ɹˈid dˈɛpθ' in r['phonemes'] for r in phonemes)
    assert evidence['published'] is False
    assert evidence['human_listening_signoff'] is False
    assert evidence['native_speaker_signoff'] is False


def test_map_is_available_in_every_catalog_and_other_media_was_preserved():
    for path in (ROOT / 'release_candidate/web/catalog').glob('*.json'):
        records = [r for r in read(path)['lessons'] if r['id'] == '12_map_barcodes']
        assert len(records) == 1
        lesson = records[0]
        assert lesson.get('status') != 'coming_soon' and lesson['app_key'] == 'map_barcodes'
        assert len(lesson['scenes']) == 13
        assert 'primers_3' in lesson['prerequisite'] and 'SRR33531217' in lesson['prerequisite']
        links = {link for scene in lesson['scenes'] for link in scene.get('related_lessons', [])}
        assert links == {'06_api', '13_regression', '47_barcode_qc'}
    preservation = read(ROOT / 'evidence/2026-09-13_map_barcodes_preservation.json')
    assert preservation['passed'] is True and preservation['catalogs_checked'] == 14
    assert preservation['retained_media_files'] == 75 * 103
