"""Decode, measure and downscale a completed private tutorial visual master.

Normal poster and web-rendition tools produce the derived assets. Native scene
timing requires all fifty source-bound tracks; browser acceptance remains open.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from stage_lesson import REPO, read, write
from native_live_timing import checked_native_timing
from stage_web_renditions import probe, stage_one, timestamps


def finish(stage, identity):
    root = stage / 'production' / identity
    spec_path = root / 'scenes.json'
    spec, english = read(spec_path), read(root / 'lesson.en.json')
    canonical = read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
    if english != canonical or [row['narration'] for row in spec['scenes']] != [
            row['narration'] for row in english['scenes']]:
        raise ValueError('Visual specification and current authored narration differ')
    native = checked_native_timing(stage, identity)
    timing_path = native or root / 'audio/en/af_heart.json'
    timing = read(timing_path)
    for scene in spec['scenes']:
        source = root / scene['image']
        if hashlib.sha256(source.read_bytes()).hexdigest() != scene['capture_sha256']:
            raise ValueError('Original source pixels changed after staging')
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    from render_visual_master import frame_aligned_durations
    import prepare_web_media
    master = root / 'video' / (identity + '_silent.mp4')
    metadata = probe(master)
    if len(metadata['streams']) != 1:
        raise ValueError('Expected one silent master video stream')
    stream = metadata['streams'][0]
    expected_frames = sum(round(value * 30) for value in frame_aligned_durations(timing['scenes'], 30))
    if (stream.get('codec_type') != 'video' or (stream.get('width'), stream.get('height')) != (3840, 2160)
            or stream.get('r_frame_rate') != '30/1' or int(stream.get('nb_frames', 0)) != expected_frames):
        raise ValueError('Master dimensions, frame rate or scene duration differ')
    points = timestamps(master)
    if len(points) != expected_frames or any(abs(point - index / 30) > 1 / 45000
                                             for index, point in enumerate(points)):
        raise ValueError('Master presentation timestamps do not preserve normal elapsed playback')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
                    '-i', str(master), '-f', 'null', '-'], check=True)
    prepare_web_media.PRODUCTION = stage / 'production'
    previous = sys.argv
    try:
        sys.argv = ['prepare_web_media.py', '--lessons', identity]
        prepare_web_media.main()
    finally:
        sys.argv = previous
    publisher_path = REPO / 'tools/tutorials/authoring/tools/publish_tutorials.py'
    module = importlib.util.spec_from_file_location('normal_tutorial_encoder', publisher_path)
    publisher = importlib.util.module_from_spec(module)
    module.loader.exec_module(publisher)
    publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
    master_hash = hashlib.sha256(master.read_bytes()).hexdigest()
    rendition = stage_one(stage, {'lesson': identity, 'scope': 'current actual staged bytes',
        'reconciliation': {'master_sha256': master_hash}}, publisher)
    receipt = {'lesson': identity, 'canonical_english_sha256': hashlib.sha256(json.dumps(
        english, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        'scene_specification_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'timing_sha256': hashlib.sha256(timing_path.read_bytes()).hexdigest(),
        'native_fifty_voice_timeline': native is not None,
        'master_sha256': master_hash, 'master_probe': metadata, 'master_full_decode_passed': True,
        'master_frame_count_matches_scene_boundaries': True, 'master_all_presentation_times_checked': True,
        'poster_sha256': hashlib.sha256((root / 'poster.jpg').read_bytes()).hexdigest(),
        'web_rendition': rendition, 'browser_verified': False, 'published': False}
    write(root / 'current-video-acceptance.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--lesson', required=True)
    args = parser.parse_args()
    receipt = finish(args.stage.resolve(), args.lesson)
    print(f"{args.lesson}: complete master decode and PTS checks, normal poster and web copy; browser acceptance remains open")


if __name__ == '__main__':
    main()
