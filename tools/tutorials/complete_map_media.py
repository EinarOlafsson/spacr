"""Verify the isolated Map Barcodes package after its disjoint renderers finish."""
import argparse
import importlib.util
from pathlib import Path
import time

from barcode_promotion import require_recorded_map
from check_completed_matrix import check, voice_matrix
from complete_embeddings_media import command
from stage_lesson import read, write
from stage_web_renditions import stage_one


def complete(stage, wait_seconds):
    matrix = voice_matrix(stage.parent / 'tools/render_all_voices.py')
    identity = '12_map_barcodes'
    pairs = [(language, voice) for language, voices in matrix.items() for voice in voices]
    deadline, previous = time.monotonic() + wait_seconds, None
    folder = stage / 'production' / identity
    while True:
        count = sum(all((folder / 'audio' / language / (voice + suffix)).is_file()
                        for suffix in ('.m4a', '.json')) for language, voice in pairs)
        if count != previous:
            print(f'Map Barcodes render files: {count}/50; verification still pending', flush=True)
            previous = count
        if count == 50:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('Narration workers have not finished all fifty tracks')
        time.sleep(20)
    command('verify_staged_audio.py', '--stage', stage, '--lesson', identity,
            '--languages', *matrix, timeout=1800)
    for language, voices in matrix.items():
        command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity,
                '--language', language, '--voice', voices[0])
    for language in ('da', 'de', 'is', 'ko', 'nb', 'sv'):
        command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity,
                '--caption-language', language)
    proof = check(stage, identity, stage.parent / 'tools/render_all_voices.py')
    write(folder / 'final-artifact-checks.json', proof)
    require_recorded_map(stage, 'en', read(folder / 'lesson.en.json'))
    specification = importlib.util.spec_from_file_location('map_tutorial_publisher',
                                stage.parent / 'tools/publish_tutorials.py')
    publisher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(publisher)
    publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
    rendition = stage_one(stage, {'lesson': identity, 'scope': 'staged final bytes',
                                  'reconciliation': proof}, publisher)
    command('verify_staged_lesson.py', '--stage', stage, '--lesson', identity, '--web-rendition')
    write(stage / 'map-final-artifacts.json', {'lesson': identity, 'passed': True,
          'matrix': proof, 'rendition': rendition, 'native_speaker_signoff': False,
          'human_listening_signoff': False, 'published': False})
    print('ISOLATED MAP BARCODES MEDIA VERIFIED', stage, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_seconds <= 7200:
        parser.error('Wait at most two hours for existing renderers')
    complete(args.stage.resolve(), args.wait_seconds)
