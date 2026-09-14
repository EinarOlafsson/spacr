"""Checkpoint Map's actual data, every voice and every native Heart caption."""
import argparse
from pathlib import Path

from barcode_promotion import IDENTITY, require_recorded_map
from build_release_candidate import copy_checked
from check_completed_matrix import digest
from stage_lesson import DEFAULT_STAGE, read, write
from validate_candidate import validate


def checkpoint(candidate):
    candidate = Path(candidate)
    validate(candidate, include_hosted_media=True, require_browser=True)
    root = Path(__file__).resolve().parent
    checked = root / 'release_candidate'
    if digest(candidate / 'release-manifest.json') != digest(checked / 'release-manifest.json'):
        raise ValueError('Checkpoint the complete candidate before the Map evidence')
    folder = DEFAULT_STAGE / 'production' / IDENTITY
    matrix = require_recorded_map(DEFAULT_STAGE, 'en', read(folder / 'lesson.en.json'))
    timing_path = folder / 'audio/en/af_heart.json'
    timing = read(timing_path)
    browser = read(checked / 'candidate-browser-checks.json')
    case = next(row for row in browser['ready_playback_cases'] if row['lesson'] == IDENTITY)
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    if (case['audio_sha256'] != timing['media_sha256']
            or len(case['sentence_cue_checks']) != len(sentences)):
        raise ValueError('The candidate must check every actual narrated Heart sentence')
    for sentence, actual in zip(sentences, case['sentence_cue_checks']):
        midpoint = (sentence['speech_start'] + sentence['speech_end']) / 2
        if (actual['requested_audio_time'] != midpoint or abs(actual['audio'] - midpoint) >= 1
                or actual['text'] != sentence['text'] or sentence['text'] not in actual['cues']):
            raise ValueError('Native captions disagree with the actual narrated sentence')
    preservation = read(candidate / 'checks/map-preservation-checks.json')
    if preservation.get('passed') is not True:
        raise ValueError('Verify preservation of the other tutorials first')
    evidence = root / 'evidence'
    for source, name in [(timing_path, '2026-09-13_map_barcodes_heart_timing.json'),
                         (candidate / 'checks/map-preservation-checks.json',
                          '2026-09-13_map_barcodes_preservation.json')]:
        copy_checked(source, evidence / name, [], evidence)
    write(evidence / '2026-09-13_map_barcodes_final_checks.json', {
        'lesson': IDENTITY, 'scope': 'Actual GUI search/mapping and separate supported two-barcode API',
        'capture': read(DEFAULT_STAGE / 'captures/map_verified/scientific_acceptance.json'),
        'matrix': matrix, 'candidate_manifest_sha256': digest(checked / 'release-manifest.json'),
        'heart_timing_sha256': digest(timing_path), 'candidate_browser_case': case,
        'native_speaker_signoff': False, 'human_listening_signoff': False, 'published': False})
    print('Checkpointed Map: real counts, fifty voices, fourteen cases, every native Heart caption', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    checkpoint(parser.parse_args().candidate)
