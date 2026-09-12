#!/usr/bin/env python3
"""Save verified Embeddings evidence without copying narration or publishing.

Run after complete_embeddings_media.py and the full candidate checkpoint.
Records distinguish the successful real API example from the incomplete GUI.
"""
import argparse
from pathlib import Path

from build_release_candidate import copy_checked
from check_completed_matrix import check, digest
from stage_lesson import DEFAULT_STAGE, read, write
from validate_candidate import validate


def main(candidate=None):
    stage = DEFAULT_STAGE
    if candidate is None:
        complete = read(stage / 'embeddings-final-candidate.json')
        candidate = complete['candidate']
    candidate = Path(candidate)
    validate(candidate, include_hosted_media=True, require_browser=True)
    root = Path(__file__).resolve().parent
    checked = root / 'release_candidate'
    if digest(candidate / 'release-manifest.json') != digest(checked / 'release-manifest.json'):
        raise ValueError('Checkpoint the completed candidate before its lesson evidence')
    matrix = check(stage, '77_embeddings', stage.parent / 'tools/render_all_voices.py')
    capture = read(stage / 'captures/embeddings_1507_verified_readable/scientific_acceptance.json')
    if (capture.get('accepted') is not True or capture.get('gui_workflow_completed') is not False
            or capture['terminal'].get('source_unchanged') is not True):
        raise ValueError('Require actual API success, not an invented GUI completion')
    for path, sha in capture['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError('Recorded GUI/API source changed: ' + path)
    for run in capture['terminal']['runs']:
        if run['helper_sha256'] != digest(root / 'embeddings_example.py'):
            raise ValueError('Recorded example differs from the committed helper')
    timing = stage / 'production/77_embeddings/audio/en/af_heart.json'
    browser = read(checked / 'candidate-browser-checks.json')
    case = next(x for x in browser['ready_playback_cases'] if x['lesson'] == '77_embeddings')
    if case['audio_sha256'] != read(timing)['media_sha256']:
        raise ValueError('Candidate browser loaded a different Heart track')
    evidence = root / 'evidence'
    copy_checked(timing, evidence / '2026-09-12_embeddings_heart_timing.json', [], evidence)
    write(evidence / '2026-09-12_embeddings_final_checks.json', {
        'scope': 'Real GUI introduction and supported Python API; technical media checks',
        'capture': capture, 'matrix': matrix,
        'candidate_manifest_sha256': digest(checked / 'release-manifest.json'),
        'heart_timing_sha256': digest(timing), 'candidate_browser_case': case,
        'native_speaker_signoff': False, 'human_listening_signoff': False,
        'gui_workflow_completed': False, 'published': False})
    print('Checkpointed real capture, all 50 tracks, 14 browser cases and native Heart captions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path,
                        help='Reverify unchanged Embeddings media in a newly checked candidate')
    main(parser.parse_args().candidate)
