"""Acceptance for the real recorded Map lesson, not a metadata-only promotion.

The 2026-09-13 evidence describes the first (1.5.0.8, thirteen-scene)
recording and the candidate it was promoted in; the lesson has since been
rewritten as a twelve-scene walkthrough and republished, so its published
media is checked against the current candidate.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from published_lesson import check_published_lesson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def test_first_map_recording_has_actual_data_complete_media_and_native_caption_evidence():
    evidence = read(ROOT / 'evidence/2026-09-13_map_barcodes_final_checks.json')
    assert evidence['capture']['gui']['mapped_reads'] == 7657
    assert evidence['capture']['api']['api_two_barcodes']['mapped_reads'] == 793
    matrix = evidence['matrix']
    assert matrix['passed'] is True and matrix['unique_final_tracks'] == 50
    assert matrix['scene_count'] == 13 and len(matrix['browser_reports']) == 14
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


def test_map_is_available_in_every_catalog_and_its_first_promotion_preserved_other_media():
    source = read(ROOT / 'lessons/12_map_barcodes.json')
    for path in (ROOT / 'release_candidate/web/catalog').glob('*.json'):
        records = [r for r in read(path)['lessons'] if r['id'] == '12_map_barcodes']
        assert len(records) == 1
        lesson = records[0]
        assert lesson.get('status') != 'coming_soon' and lesson['app_key'] == 'map_barcodes'
        assert len(lesson['scenes']) == len(source['scenes']) == 12
        assert 'primers_3' in lesson['prerequisite'] and 'SRR33531217' in lesson['prerequisite']
        links = {link for scene in lesson['scenes'] for link in scene.get('related_lessons', [])}
        assert links == {link for scene in source['scenes'] for link in scene.get('related_lessons', [])}
    preservation = read(ROOT / 'evidence/2026-09-13_map_barcodes_preservation.json')
    assert preservation['passed'] is True and preservation['catalogs_checked'] == 14
    assert preservation['map_media_files'] == 103
    assert preservation['refreshed_lessons'] == ['07_mask', '13_regression']
    assert '12_map_barcodes' not in preservation['refreshed_lessons']
    changed_media = [path for path in preservation['changed_files']
                     if path.startswith(('media_host/', 'web/production/'))]
    # Counts of the candidate that promotion was checked against (7828 media files).
    assert len(changed_media) == 105 and preservation['retained_media_files'] == 7723
    owners = {path.split('/')[1] if path.startswith('media_host/') else path.split('/')[2]
              for path in changed_media}
    assert owners == {'07_mask', '13_regression'}
    assert not [path for path in changed_media if '12_map_barcodes' in path]


def test_map_published_voices_and_heart_captions_are_the_candidates():
    check_published_lesson('12_map_barcodes', 12)
