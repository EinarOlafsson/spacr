"""Stage only Map Barcodes from its actual GUI and API recordings."""
import argparse
from pathlib import Path
import shutil
import sys

from stage_lesson import DEFAULT_STAGE, REPO, read, write, stage_lesson
from map_barcodes_data import digest


def prepare(stage):
    source = Path(__file__).parent / 'lessons/12_map_barcodes.json'
    target = stage / 'captures/map_verified'
    if target.exists():
        raise FileExistsError('Preserve the previous verified recording')
    parts = [('gui', 'map_search_v2'), ('api', 'map_barcodes_api')]
    frames, proofs, sources = {}, {}, {}
    target.mkdir(parents=True)
    for prefix, name in parts:
        original = stage / 'captures' / name
        proof = read(original / 'scientific_acceptance.json')
        provenance = read(original / 'provenance.json')
        if proof.get('accepted') is not True or provenance.get('completed_capture') is not True:
            raise ValueError('A real GUI/API recording did not pass')
        proofs[prefix] = proof
        sources[prefix] = provenance
        for name, record in read(original / 'frames.json').items():
            path = original / record['image']
            if digest(path) != record['sha256']:
                raise ValueError('A recorded frame changed')
            filename = prefix + '_' + path.name
            shutil.copy2(path, target / filename)
            frames[prefix + '_' + name] = dict(record, image=filename)
    write(target / 'frames.json', frames)
    write(target / 'scientific_acceptance.json', {'accepted': True, 'gui': proofs['gui'],
          'api': proofs['api'], 'app_source_modified': False, 'published': False})
    write(target / 'provenance.json', {'completed_capture': True, 'module': 'map_barcodes',
          'version': sources['gui']['version'], 'commit': sources['gui']['commit'],
          'sources': sources, 'app_source_modified': False})
    # Keep existing identities/media outside this isolated stage untouched.
    shutil.copytree(DEFAULT_STAGE / 'catalog', stage / 'catalog')
    lesson = read(source)
    sys.path.insert(0, str(DEFAULT_STAGE.parent / 'tools'))
    from pronunciation import spoken_form
    for scene in lesson['scenes']:
        scene['speech_text'] = spoken_form(scene['narration'], 'en')
    authored = stage / 'map_lesson.en.json'
    write(authored, lesson)
    stage_lesson(authored, 'map_verified', stage)
    write(stage / 'map-recording-checkpoint.json', {
        'stage': str(stage), 'lesson': '12_map_barcodes', 'scene_count': len(lesson['scenes']),
        'source_lesson_sha256': digest(source), 'recorded_source': sources,
        'gui_result': proofs['gui'], 'api_result': proofs['api'],
        'native_speaker_signoff': False, 'human_listening_signoff': False,
        'published': False, 'media_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    prepare(parser.parse_args().stage.resolve())
