"""Verify both isolated model lessons after their existing narration jobs finish.

No synthesis, shared-stage changes, publication or app modifications. Every
recording keeps one visual master across the preserved fifty voices.
"""
import argparse
import importlib.util
from pathlib import Path
import time

from check_completed_matrix import check, voice_matrix
from complete_embeddings_media import command
from stage_lesson import DEFAULT_STAGE, read, write
from stage_web_renditions import stage_one

LESSONS = ('21_model_compare', '22_model_zoo')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 7200:
        parser.error('Wait for existing workers for no more than two hours')
    stage = Path(read(DEFAULT_STAGE / 'models-next-stage.json')['stage']).resolve()
    if stage == DEFAULT_STAGE or stage.parent != DEFAULT_STAGE.parent:
        raise ValueError('Expected the separate model-tutorial stage')
    inventory = voice_matrix(stage.parent / 'tools/render_all_voices.py')
    expected = [(identity, lang, voice) for identity in LESSONS
                for lang, voices in inventory.items() for voice in voices]
    deadline, previous = time.monotonic() + args.wait_seconds, None
    while True:
        count = sum(all((stage / 'production' / identity / 'audio' / lang / (voice + suffix)).is_file()
                        for suffix in ('.m4a', '.json')) for identity, lang, voice in expected)
        if count != previous:
            print(f'Model render files {count}/100; verification still required', flush=True)
            previous = count
        if count == len(expected):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('The already-running model narration jobs are not finished')
        time.sleep(min(20, max(0, deadline - time.monotonic())))
    specification = importlib.util.spec_from_file_location('model_tutorial_encoder',
                         stage.parent / 'tools/publish_tutorials.py')
    publisher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(publisher)
    publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
    results = []
    for identity in LESSONS:
        command('verify_staged_audio.py', '--stage', stage, '--lesson', identity,
                '--languages', *inventory, timeout=1200)
        for language, voices in inventory.items():
            command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity,
                    '--language', language, '--voice', voices[0])
        for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
            command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity,
                    '--caption-language', language)
        matrix = check(stage, identity, stage.parent / 'tools/render_all_voices.py')
        write(stage / 'production' / identity / 'final-artifact-checks.json', matrix)
        rendition = stage_one(stage, {'lesson': identity, 'scope': 'staged final bytes',
                                      'reconciliation': matrix}, publisher)
        command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity, '--web-rendition')
        result = {'lesson': identity, 'matrix': matrix, 'web_rendition': rendition,
                  'scope': 'Current recorded subset only; GUI inference/benchmark not completed'}
        results.append(result)
        write(stage / 'production' / identity / 'isolated-final-checks.json', result)
    write(stage / 'models-final-artifacts.json', {'lessons': results, 'passed': True,
          'scope': 'Two isolated recordings, not a whole-library or publication sign-off',
          'native_speaker_signoff': False, 'human_listening_signoff': False, 'published': False})
    print('BOTH ISOLATED MODEL LESSONS VERIFIED', stage, flush=True)


if __name__ == '__main__':
    main()
