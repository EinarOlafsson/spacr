"""Verify the OPS render and assemble a new private, unpublished candidate.

Waits only for the already-running disjoint narration workers. No synthesis,
source edits, checkpoint overwrite, upload or deployment is performed here.
"""
import argparse
from pathlib import Path
import time

from check_completed_matrix import voice_matrix
from complete_embeddings_media import BASELINE, command
from stage_lesson import DEFAULT_STAGE, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 7200:
        parser.error('Bound the existing-render wait to at most two hours')
    lesson = '76_ops'
    stage = DEFAULT_STAGE
    matrix = voice_matrix(stage.parent / 'tools/render_all_voices.py')
    pairs = [(lang, voice) for lang, voices in matrix.items() for voice in voices]
    deadline, previous = time.monotonic() + args.wait_seconds, None
    while True:
        present = [pair for pair in pairs if all(
            (stage / 'production' / lesson / 'audio' / pair[0] / (pair[1] + suffix)).is_file()
            for suffix in ('.m4a', '.json'))]
        if len(present) != previous:
            print(f'OPS render files {len(present)}/{len(pairs)}; verification still required', flush=True)
            previous = len(present)
        if len(present) == len(pairs):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('The existing narration workers have not finished')
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    command('verify_staged_audio.py', '--lesson', lesson, '--languages', *matrix, timeout=1200)
    for language, voices in matrix.items():
        command('verify_staged_lesson.py', '--lesson', lesson, '--language', language, '--voice', voices[0])
    for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
        command('verify_staged_lesson.py', '--lesson', lesson, '--caption-language', language)
    command('check_completed_matrix.py', '--lesson', lesson)
    from ops_promotion import require_recorded_ops
    english = next(item for item in read(stage / 'catalog/lessons_en.json')['lessons'] if item['id'] == lesson)
    require_recorded_ops(stage, 'en', english)
    from coming_soon import HELD
    command('verify_library_checkpoint.py', '--held', *HELD, '--baseline', BASELINE,
            '--output', stage / 'library-checkpoint-2026-09-11.json', timeout=1200)
    command('stage_web_renditions.py', '--lesson', lesson, timeout=1200)
    command('verify_staged_lesson.py', '--lesson', lesson, '--web-rendition')
    from build_release_candidate import build
    candidate, _ = build(stage, baseline=BASELINE)
    command('verify_release_candidate.py', candidate, timeout=1800)
    command('check_placeholder_mutations.py', candidate)
    from validate_candidate import validate
    result = validate(candidate, include_hosted_media=True, require_browser=True)
    write(stage / 'ops-final-candidate.json', {
        'candidate': str(candidate), 'validated': result, 'ready_lesson': lesson,
        'scope': 'Current GUI introduction and four-tile geometry API example',
        'full_ops_pipeline_completed': False, 'native_speaker_signoff': False,
        'human_listening_signoff': False, 'published': False})
    print('VERIFIED PRIVATE OPS CANDIDATE', candidate, flush=True)


if __name__ == '__main__':
    main()
