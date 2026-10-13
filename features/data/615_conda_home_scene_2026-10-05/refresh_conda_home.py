"""Replace only the accepted Conda lesson's outdated Home scene privately."""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005')
SCRATCH = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
BASELINE = SCRATCH / 'tutorial-final-wave/release-candidate-append-co2wgfyt'
STAGE = SCRATCH / 'tutorial-conda-home-current-r2'
IDENTITY = '02_conda_install'
sys.path[:0] = [str(ROOT / 'tools/tutorials'), str(ROOT / 'tools/tutorials/authoring/tools')]
from stage_lesson import read, write
from retain_narration import digest
from render_visual_master import concat, encode_still, frame_aligned_durations
from stage_web_renditions import probe, require_timestamps, require_video, timestamps


def run(command):
    subprocess.run(command, check=True)


def frame_hashes(path, output):
    run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2', '-i', str(path),
         '-map', '0:v:0', '-an', '-f', 'framemd5', str(output)])
    return [line.split(',')[-1].strip() for line in output.read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def splice(source, replacement, destination, work, begin, finish, total):
    packets = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time,flags', '-of', 'json', str(source)]))['packets']
    for boundary in (begin, finish):
        if not any('K' in packet['flags'] and abs(float(packet['pts_time']) - boundary / 30) < 1e-6
                   for packet in packets):
            raise ValueError('The unchanged source must have an independently decodable scene boundary')
    before, after = work / 'before.mp4', work / 'after.mp4'
    common = ['ffmpeg', '-nostdin', '-v', 'error']
    run([*common, '-i', str(source), '-frames:v', str(begin),
         '-map', '0:v:0', '-an', '-c', 'copy', str(before)])
    run([*common, '-ss', f'{finish / 30:.9f}', '-i', str(source),
         '-map', '0:v:0', '-an', '-c', 'copy', str(after)])
    for path, expected in ((before, begin), (replacement, finish - begin), (after, total - finish)):
        if int(probe(path)['streams'][0]['nb_frames']) != expected:
            raise ValueError(f'Stream-copy segment has a different frame count: {path}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    concat([before, replacement, after], destination, work / 'concat.txt')
    if int(probe(destination)['streams'][0]['nb_frames']) != total:
        raise ValueError('The replacement changed the complete video frame count')
    require_timestamps(timestamps(source), timestamps(destination), total)
    old_hashes = frame_hashes(source, work / 'before.framemd5')
    new_hashes = frame_hashes(destination, work / 'after.framemd5')
    if (len(old_hashes) != total or len(new_hashes) != total
            or old_hashes[:begin] != new_hashes[:begin]
            or old_hashes[finish:] != new_hashes[finish:]):
        raise ValueError('An unchanged video frame differs after the single-scene replacement')
    if any(a == b for a, b in zip(old_hashes[begin:finish], new_hashes[begin:finish])):
        raise ValueError('The outdated Home frame remains inside the replacement scene')
    return {'all_unchanged_decoded_frames_identical': True,
            'unchanged_frames_verified': total - (finish - begin),
            'replacement_frames': finish - begin, 'frame_presentation_times_preserved': True,
            'full_decode_passed': True, 'source_sha256': digest(source),
            'replacement_sha256': digest(destination), 'frames': total}


def main():
    if STAGE.exists():
        raise ValueError('Preserve previous work; use a fresh stage for a new attempt')
    manifest = read(BASELINE / 'release-manifest.json')
    publication = read(BASELINE / 'publication-receipt.json')
    if (publication['commit'] != 'd8d275cc932a01d78185ba5cb6e41d586876aa75'
            or publication['manifest_sha256'] != digest(BASELINE / 'release-manifest.json')
            or publication['readback']['passed'] is not True):
        raise ValueError('The existing lesson must come from the accepted immutable publication')
    records = {row['path']: row for row in manifest['files']}
    lesson_path = ROOT / 'tools/tutorials/lessons' / (IDENTITY + '.json')
    lesson = read(lesson_path)
    selected = [i for i, row in enumerate(lesson['scenes']) if row['visual'] == '06_installed_home']
    if selected != [7] or len(lesson['scenes']) != 9:
        raise ValueError('The single requested Home scene has changed its identity')
    timing = read(BASELINE / f'media_host/{IDENTITY}/audio/en/af_heart.json')
    if [row['text'] for row in timing['scenes']] != [row['narration'] for row in lesson['scenes']]:
        raise ValueError('Published timing and canonical narration differ')
    counts = [round(seconds * 30) for seconds in frame_aligned_durations(timing['scenes'], 30)]
    begin, finish, total = sum(counts[:7]), sum(counts[:8]), sum(counts)
    if (begin, finish, total) != (2532, 2832, 3402):
        raise ValueError('Do not change the accepted scene timing')
    native = read(ROOT / 'features/data/662_yolo_native_capture_2026-10-05.json')
    capture = Path(native['capture'])
    frame = read(capture / 'frames.json')['00_home']
    image = capture / frame['image']
    archived_frames = 'features/data/662_yolo_native_capture_2026-10-05/frames.json'
    if (not native['accepted_native_capture'] or not native['provenance']['completed_capture']
            or native['provenance']['app_source_modified'] is not False
            or digest(image) != frame['sha256']
            or digest(capture / 'frames.json') != native['artifacts'][archived_frames]['sha256']):
        raise ValueError('The current Home scene needs exact accepted native pixels')
    folder = STAGE / 'production' / IDENTITY
    folder.mkdir(parents=True)
    catalogs = {}
    for source in sorted((BASELINE / 'web/catalog').glob('*.json')):
        if not source.name.startswith(('lessons_', 'captions_')):
            continue
        if source.read_bytes() != (ROOT / 'docs/source/_extra/tutorials/catalog' / source.name).read_bytes():
            raise ValueError('Preserve every currently published catalog byte')
        target = STAGE / 'catalog' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        catalogs[source.name] = digest(target)
    if len(catalogs) != 14:
        raise ValueError('Preserve the full fourteen-catalog language matrix')
    retained = []
    for relative, record in records.items():
        if not (relative.startswith(f'media_host/{IDENTITY}/audio/')
                or relative == f'web/production/{IDENTITY}/poster.jpg'):
            continue
        source = BASELINE / relative
        if digest(source) != record['sha256']:
            raise ValueError('Published narration or poster changed')
        target = folder / Path(*Path(relative).parts[2 if relative.startswith('media_host/') else 3:])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(target) == record['sha256']
        retained.append({'path': str(target.relative_to(STAGE)), 'sha256': record['sha256']})
    if len(retained) != 55:
        raise ValueError('Preserve twenty-seven audio tracks, all timings and the existing poster')
    write(folder / 'lesson.en.json', lesson)
    work = STAGE / 'scene-replacement'
    work.mkdir()
    part = work / 'current-home-4k.mp4'
    encode_still(image, (finish - begin) / 30, part, 30)
    specification = importlib.util.spec_from_file_location('normal_publisher', ROOT / 'tools/tutorials/authoring/tools/publish_tutorials.py')
    publisher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(publisher)
    publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
    web_part = work / 'current-home-1440p.mp4'
    ok, note = publisher.encode_1440p(part, web_part)
    if not ok:
        raise RuntimeError(note)
    evidence = {}
    for kind, replacement in (('video', part), ('web', web_part)):
        source = BASELINE / f'media_host/{IDENTITY}/{kind}/{IDENTITY}_silent.mp4'
        if digest(source) != records[str(source.relative_to(BASELINE))]['sha256']:
            raise ValueError('The baseline video differs from its immutable manifest')
        destination = (folder / 'video' if kind == 'video' else STAGE / 'web-renditions' / IDENTITY / 'video') / source.name
        segment_work = work / kind
        segment_work.mkdir()
        evidence[kind] = splice(source, replacement, destination, segment_work, begin, finish, total)
        print(kind, 'single scene replaced; every other decoded frame and presentation time identical', flush=True)
    master = folder / 'video' / f'{IDENTITY}_silent.mp4'
    web = STAGE / 'web-renditions' / IDENTITY / 'video' / master.name
    require_video(probe(master), probe(web))
    write(web.parent.parent / 'rendition-checks.json', {
        'lesson': IDENTITY, 'accepted': True, 'master_sha256': digest(master),
        'rendition_sha256': digest(web), 'bytes': web.stat().st_size, 'frames': total,
        'dimensions': [2560, 1440], 'all_frame_presentation_times_match': True,
        'full_decode_passed': True, 'publisher_sha256': digest(Path(publisher.__file__)),
        'encoder_arguments': publisher.ENCODE_ARGS, 'single_scene_stream_copy': evidence['web'],
        'visual_review_complete': False, 'narration_regenerated': False, 'published': False})
    result = {'accepted': True, 'lesson': IDENTITY, 'scope': 'Only scene 8 Home image; original Conda installation evidence, narration, captions, timings and all other video pixels preserved.',
              'baseline_media_commit': publication['commit'], 'baseline_manifest_sha256': digest(BASELINE / 'release-manifest.json'),
              'scene_index': 7, 'first_frame': begin, 'end_frame_exclusive': finish,
              'native_frame': str(image), 'native_frame_sha256': digest(image),
              'native_provenance': native['provenance'], 'native_frame_metadata': frame,
              'fresh_conda_installation_claimed': False,
              'catalogs_preserved': catalogs, 'retained_audio_timings_poster': retained,
              'video': evidence, 'new_narration_generated': False, 'published': False,
              'browser_verified': False}
    write(folder / 'single-scene-refresh.json', result)
    print('Private Conda Home-scene replacement accepted; browser checks and publication remain.', flush=True)


if __name__ == '__main__':
    main()
