#!/usr/bin/env python3
"""Verify a finished Embeddings render and assemble a NEW private candidate.

Does not synthesize, upload, deploy, checkpoint into Git, or edit an older
candidate. A bounded optional wait lets existing disjoint renderers finish.
Run with the speech/browser environment and the 100-GiB memory guard.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from check_completed_matrix import voice_matrix
from stage_lesson import DEFAULT_STAGE, REPO, read, write

ROOT = Path(__file__).resolve().parent
LESSON = '77_embeddings'
BASELINE = DEFAULT_STAGE / 'baseline-d2d4c189b-hmx3Us/docs/source/_extra/tutorials/catalog'


def command(script, *args, timeout=600):
    print('VERIFY', script, *args, flush=True)
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                   cwd=REPO, check=True, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 7200:
        parser.error('Wait must be between zero and two hours')
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    matrix = voice_matrix(DEFAULT_STAGE.parent / 'tools/render_all_voices.py')
    pairs = [(language, voice) for language, voices in matrix.items() for voice in voices]
    deadline = time.monotonic() + args.wait_seconds
    previous = None
    while True:
        present = [pair for pair in pairs if all(
            (DEFAULT_STAGE / 'production' / LESSON / 'audio' / pair[0] / (pair[1] + suffix)).is_file()
            for suffix in ('.m4a', '.json'))]
        if len(present) != previous:
            print(f'RENDER FILES {len(present)}/{len(pairs)} (not yet technical acceptance)', flush=True)
            previous = len(present)
        if len(present) == len(pairs):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Existing renderers have not produced all required pairs')
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    # File presence does NOT grant acceptance: fully decode every track and
    # require its actual current script, pronunciation and voice fingerprint.
    command('verify_staged_audio.py', '--lesson', LESSON, '--languages', *matrix, timeout=1200)
    for language, voices in matrix.items():
        command('verify_staged_lesson.py', '--lesson', LESSON, '--language', language,
                '--voice', voices[0])
    for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
        command('verify_staged_lesson.py', '--lesson', LESSON, '--caption-language', language)
    command('check_completed_matrix.py', '--lesson', LESSON)
    from coming_soon import HELD
    command('verify_library_checkpoint.py', '--held', *HELD, '--baseline', BASELINE,
            '--output', DEFAULT_STAGE / 'library-checkpoint-2026-09-11.json', timeout=1200)
    command('stage_web_renditions.py', '--lesson', LESSON, timeout=1200)
    command('verify_staged_lesson.py', '--lesson', LESSON, '--web-rendition')
    from build_release_candidate import build
    candidate, _ = build(DEFAULT_STAGE, baseline=BASELINE)
    command('verify_release_candidate.py', candidate, timeout=1800)
    command('check_placeholder_mutations.py', candidate)
    from validate_candidate import validate
    result = validate(candidate, include_hosted_media=True, require_browser=True)
    write(DEFAULT_STAGE / 'embeddings-final-candidate.json', {
        'candidate': str(candidate), 'validated': result,
        'ready_lesson': LESSON, 'native_speaker_signoff': False,
        'human_listening_signoff': False, 'published': False})
    print('VERIFIED PRIVATE CANDIDATE', candidate, flush=True)


if __name__ == '__main__':
    main()
