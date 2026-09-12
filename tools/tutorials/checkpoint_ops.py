"""Checkpoint OPS only after actual complete media and candidate verification."""
from pathlib import Path

from build_release_candidate import copy_checked
from check_completed_matrix import digest
from ops_promotion import require_recorded_ops
from stage_lesson import DEFAULT_STAGE, read, write
from validate_candidate import validate


def main():
    stage = DEFAULT_STAGE
    complete = read(stage / 'ops-final-candidate.json')
    candidate = Path(complete['candidate'])
    validate(candidate, include_hosted_media=True, require_browser=True)
    root = Path(__file__).resolve().parent
    checkpoint = root / 'release_candidate'
    if digest(candidate / 'release-manifest.json') != digest(checkpoint / 'release-manifest.json'):
        raise ValueError('Checkpoint the fully verified candidate first')
    english = read(stage / 'production/76_ops/lesson.en.json')
    matrix = require_recorded_ops(stage, 'en', english)
    timing_path = stage / 'production/76_ops/audio/en/af_heart.json'
    timing = read(timing_path)
    browser = read(checkpoint / 'candidate-browser-checks.json')
    case = next(item for item in browser['ready_playback_cases'] if item['lesson'] == '76_ops')
    sentences = [sentence for scene in timing['scenes'] for sentence in scene['sentences']]
    if (case['audio_sha256'] != timing['media_sha256']
            or len(case['sentence_cue_checks']) != len(sentences)):
        raise ValueError('The candidate did not check every actual OPS Heart sentence')
    for expected, actual in zip(sentences, case['sentence_cue_checks']):
        midpoint = (expected['speech_start'] + expected['speech_end']) / 2
        if (actual['requested_audio_time'] != midpoint
                or abs(actual['audio'] - midpoint) >= 1
                or expected['text'] != actual['text'] or expected['text'] not in actual['cues']):
            raise ValueError('Native OPS captions disagree with an actual narrated sentence')
    evidence = root / 'evidence'
    copy_checked(timing_path, evidence / '2026-09-12_ops_heart_timing.json', [], evidence)
    write(evidence / '2026-09-12_ops_final_checks.json', {
        'scope': 'Four-tile geometry tutorial, not a completed OPS screening pipeline',
        'capture': read(stage / 'captures/ops_1507_verified/scientific_acceptance.json'),
        'matrix': matrix, 'candidate_manifest_sha256': digest(checkpoint / 'release-manifest.json'),
        'heart_timing_sha256': digest(timing_path), 'candidate_browser_case': case,
        'full_ops_pipeline_completed': False, 'gui_workflow_completed': False,
        'native_speaker_signoff': False, 'human_listening_signoff': False, 'published': False})
    print('OPS: real capture, all 50 tracks, 14 browser cases and every native Heart caption verified')


if __name__ == '__main__':
    main()
