#!/usr/bin/env python3
"""Prepare checked 1440p copies privately, without publishing or resynthesis.

Uses the existing publisher's encoder, with two encoder/filter threads. Every
source must appear in the final library checkpoint. Retained Plate Viewer uses
its existing published 1440p bytes, not a new encode. Checkpoints make resuming
safe; no incomplete output is reported as accepted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

from check_completed_matrix import digest
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)]))


def require_video(source, target):
    if len(source['streams']) != 1 or len(target['streams']) != 1:
        raise ValueError('Expected one silent video stream in each file')
    before, after = source['streams'][0], target['streams'][0]
    if (before.get('codec_type') != 'video' or after.get('codec_type') != 'video'
            or [before.get('width'), before.get('height')] != [3840, 2160]
            or [after.get('width'), after.get('height')] != [2560, 1440]
            or before.get('r_frame_rate') != '30/1' or after.get('r_frame_rate') != '30/1'
            or int(before.get('nb_frames', 0)) <= 0
            or before.get('nb_frames') != after.get('nb_frames')):
        raise ValueError('Rendition dimensions, rate or frame count differ')


def timestamps(path):
    payload = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts_time', '-of', 'json', str(path)]))
    return sorted(float(packet['pts_time']) for packet in payload['packets'])


def require_timestamps(before, after, frames):
    import math
    if (len(before) != frames or len(after) != frames
            or any(not math.isfinite(x) for x in before + after)
            or any(abs(a - b) > 2 / 90000 for a, b in zip(before, after))):
        raise ValueError('Frame presentation times changed during downscaling')


def stage_one(stage, item, publisher):
    lesson = item['lesson']
    if Path(lesson).name != lesson or lesson in {'.', '..'}:
        raise ValueError('Expected a single lesson identity')
    retained = item['scope'] == 'unchanged original media'
    master = (stage.parent if retained else stage) / 'production' / lesson / 'video' / f'{lesson}_silent.mp4'
    expected = (item['retained_sources']['media']['video']['sha256'] if retained
                else item['reconciliation']['master_sha256'])
    if digest(master) != expected:
        raise ValueError('Master changed since the library checkpoint')
    folder = stage / 'web-renditions' / lesson
    target = folder / 'video' / master.name
    receipt = folder / 'rendition-checks.json'
    encoder_hash = digest(Path(publisher.__file__))
    published_source = REPO / 'docs/source/_extra/tutorials/production' / lesson / 'video' / master.name
    retained_hash = digest(published_source) if retained else None
    if receipt.exists() and target.exists():
        old = read(receipt)
        if (old.get('accepted') is True and old.get('master_sha256') == expected
                and old.get('rendition_sha256') == digest(target)
                and old.get('publisher_sha256') == encoder_hash
                and old.get('encoder_arguments') == publisher.ENCODE_ARGS
                and old.get('retained_published_sha256') == retained_hash):
            print(lesson, 'already checked', flush=True)
            return old
        raise ValueError('Existing rendition differs; preserve it and investigate before replacing')
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if retained:
        shutil.copy2(published_source, target)
        if digest(target) != retained_hash:
            raise ValueError('Retained web media copy differs')
    else:
        ok, note = publisher.encode_1440p(master, target)
        if not ok:
            raise RuntimeError(note)
    original_probe, web_probe = probe(master), probe(target)
    require_video(original_probe, web_probe)
    frames = int(web_probe['streams'][0]['nb_frames'])
    before, after = timestamps(master), timestamps(target)
    require_timestamps(before, after, frames)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '2',
                    '-i', str(target), '-f', 'null', '-'], check=True)
    if digest(master) != expected:
        raise ValueError('Master changed while preparing its rendition')
    result = {'lesson': lesson, 'accepted': True, 'master_sha256': expected,
              'rendition_sha256': digest(target), 'bytes': target.stat().st_size,
              'frames': frames, 'dimensions': [2560, 1440],
              'all_frame_presentation_times_match': True, 'full_decode_passed': True,
              'publisher_sha256': encoder_hash, 'encoder_arguments': publisher.ENCODE_ARGS,
              'retained_published_sha256': retained_hash,
              'elapsed_seconds': round(time.monotonic() - started, 2),
              'visual_review_complete': False, 'narration_regenerated': False,
              'original_master_changed': False, 'published': False}
    write(receipt, result)
    print(lesson, 'PASS', result['bytes'], 'bytes', result['elapsed_seconds'], 'seconds', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--lesson', action='append', default=[])
    parser.add_argument('--all-verified', action='store_true')
    args = parser.parse_args()
    if bool(args.lesson) == args.all_verified:
        parser.error('Choose --lesson (repeatable) or --all-verified')
    library = read(args.stage / 'library-checkpoint-2026-09-11.json')
    if library.get('checked_subset_passed') is not True:
        parser.error('A passing library checkpoint is required')
    by_id = {item['lesson']: item for item in library['lessons']}
    chosen = list(by_id) if args.all_verified else args.lesson
    if len(chosen) != len(set(chosen)) or not set(chosen) <= set(by_id):
        parser.error('Selected lessons must each have a verified matrix')
    source = args.stage.parent / 'tools/publish_tutorials.py'
    spec = importlib.util.spec_from_file_location('existing_tutorial_publisher', source)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    publisher.ENCODE_ARGS = ['-threads', '2', '-filter_threads', '2', *publisher.ENCODE_ARGS]
    for identity in chosen:
        stage_one(args.stage.resolve(), by_id[identity], publisher)
