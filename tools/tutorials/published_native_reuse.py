"""Refresh narration while reusing an exact verified published native master.

The source is a published immutable video and its accepted scene clock, not a
new capture. Every new narration track must fit those unchanged intervals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import errno
import math
import os
from pathlib import Path
import shutil

from check_completed_matrix import digest
from stage_lesson import REPO, read, write

RECORD = 'published-native-reuse.json'


def _published_inputs(baseline, identity):
    from build_appended_candidate import require_baseline_receipt
    baseline = Path(baseline).resolve()
    manifest_path = baseline / 'release-manifest.json'
    publication_path = baseline / 'publication-receipt.json'
    manifest, publication = read(manifest_path), read(publication_path)
    require_baseline_receipt(manifest, publication, digest(manifest_path))
    records = {row['path']: row for row in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate baseline manifest paths')
    paths = {'master': f'media_host/{identity}/video/{identity}_silent.mp4',
             'timing': f'media_host/{identity}/video/native-live-timings.json'}
    bindings = {}
    for name, relative in paths.items():
        record = records.get(relative)
        source = baseline / relative
        if (record is None or not source.is_file()
                or source.stat().st_size != record['bytes'] or digest(source) != record['sha256']):
            raise ValueError('Published native source differs from its verified manifest: ' + relative)
        bindings[name] = {'path': relative, 'bytes': record['bytes'], 'sha256': record['sha256']}
    timing = read(baseline / paths['timing'])
    if (timing.get('kind') != 'native_visual_timing' or timing.get('lesson') != identity
            or timing.get('native_live_video') is not True or timing.get('fps') != 30
            or timing.get('playback_speed') != 1 or timing.get('looped_or_stretched') is not False
            or timing.get('coverage', {}).get('accepted') is not True):
        raise ValueError('Expected an accepted unchanged published native scene clock')
    cursor = 0.0
    for index, scene in enumerate(timing.get('scenes', []), 1):
        start, end, duration = (float(scene[key]) for key in ('speech_start', 'scene_end', 'duration'))
        if (scene.get('scene') != index or not all(math.isfinite(x) for x in (start, end, duration))
                or abs(start - cursor) > 1e-8 or duration <= 0 or abs(end - start - duration) > 1e-8
                or abs(end * 30 - round(end * 30)) > 1e-5):
            raise ValueError('Published scene clock is not contiguous and frame aligned')
        cursor = end
    if not cursor or abs(float(timing['total_duration']) - cursor) > 1e-8:
        raise ValueError('Published total duration differs from its scene clock')
    return baseline, timing, bindings, {'baseline_root': str(baseline),
        'baseline_manifest_sha256': digest(manifest_path),
        'baseline_publication_receipt_sha256': digest(publication_path),
        'baseline_media_commit': publication['commit'],
        'media_root': publication['media_root']}


def interval_spec(english, timing):
    if (len(english['scenes']) != len(timing['scenes']) or
            [scene['visual'] for scene in english['scenes']] !=
            [scene.get('visual') for scene in timing['scenes']]):
        raise ValueError('Current visual scene identities or order differ from published footage')
    return {'lesson': english['id'], 'kind': 'published_native_intervals',
        'size': [3840, 2160], 'fps': 30,
        'scenes': [{'visual': authored['visual'], 'narration': authored['narration'],
                    'published_native_interval': {'start': source['speech_start'], 'end': source['scene_end']}}
                   for authored, source in zip(english['scenes'], timing['scenes'])]}


def current_reuse_timing(stage, identity):
    from native_live_timing import verified_narration_tracks
    stage = Path(stage).resolve()
    root = stage / 'production' / identity
    record = read(root / RECORD)
    if (record.get('schema') != 1 or record.get('kind') != 'published_native_narration_reuse'
            or record.get('lesson') != identity or record.get('fresh_capture') is not False
            or record.get('raw_capture_revalidated') is not False):
        raise ValueError('Expected explicit published-native reuse provenance')
    baseline, original, bindings, source = _published_inputs(record['baseline_root'], identity)
    expected_record = dict(schema=1, kind='published_native_narration_reuse', lesson=identity,
                           fresh_capture=False, raw_capture_revalidated=False, **source,
                           published_master=bindings['master'], published_timing=bindings['timing'])
    if record != expected_record:
        raise ValueError('Published-native source bindings changed')
    canonical = read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
    english = read(root / 'lesson.en.json')
    if english != canonical:
        raise ValueError('Reused native narration differs from current canonical source')
    spec = interval_spec(english, original)
    if read(root / 'scenes.json') != spec:
        raise ValueError('Published-native interval specification changed')
    master = root / 'video' / f'{identity}_silent.mp4'
    if (master.stat().st_size != bindings['master']['bytes'] or
            digest(master) != bindings['master']['sha256']):
        raise ValueError('Staged native master differs from verified published bytes')
    tracks, proofs = verified_narration_tracks(stage, identity, english)
    result = deepcopy(original)
    gaps = []
    for index, (scene, authored) in enumerate(zip(result['scenes'], english['scenes'])):
        durations = [float(track['scenes'][index]['duration']) for track in tracks]
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError('Invalid new narration duration')
        longest = max(durations)
        if longest > float(scene['duration']) + 1e-9:
            gaps.append({'scene': index + 1, 'visual': authored['visual'],
                         'available_seconds': scene['duration'], 'required_seconds': longest})
        scene['text'] = authored['narration']
        scene['longest_narration_duration'] = longest
    if gaps:
        raise ValueError('Current narration exceeds unchanged published native intervals: ' + str(gaps))
    result['english_sha256'] = hashlib.sha256(json.dumps(english, sort_keys=True,
        ensure_ascii=False).encode()).hexdigest()
    result['visual_spec_sha256'] = digest(root / 'scenes.json')
    result['source_tracks'] = proofs
    result['published_native_reuse'] = {**{key: value for key, value in source.items()
        if key != 'baseline_root'}, 'master_sha256': bindings['master']['sha256'],
        'original_timing_sha256': bindings['timing']['sha256'],
        'fresh_capture': False, 'raw_capture_revalidated': False}
    result['coverage'] = {'accepted': True, 'gaps': [], 'track_count': len(tracks),
        'native_scene_count': sum(scene.get('native_live_video') is True for scene in original['scenes']),
        'published_scene_clocks_preserved': True}
    return result


def checked_published_native_timing(stage, identity):
    path = Path(stage) / 'production' / identity / 'video/native-live-timings.json'
    if read(path) != current_reuse_timing(stage, identity):
        raise ValueError('Published-native current narration timing changed after validation')
    return path


def prepare(stage, identity, baseline):
    stage = Path(stage).resolve()
    root = stage / 'production' / identity
    baseline, original, bindings, source = _published_inputs(baseline, identity)
    english = read(root / 'lesson.en.json')
    canonical = read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
    if english != canonical:
        raise ValueError('Prepared narration is not current canonical English')
    spec = interval_spec(english, original)
    target = root / 'video' / f'{identity}_silent.mp4'
    timing_path = target.with_name('native-live-timings.json')
    destinations = [root / RECORD, root / 'scenes.json', target, timing_path]
    if any(os.path.lexists(path) for path in destinations):
        raise FileExistsError('Preserve existing published-native reuse outputs')
    target.parent.mkdir(parents=True, exist_ok=True)
    source_master = baseline / bindings['master']['path']
    try:
        os.link(source_master, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        with source_master.open('rb') as before, target.open('xb') as after:
            shutil.copyfileobj(before, after, 1024 * 1024)
        shutil.copystat(source_master, target)
    write(root / 'scenes.json', spec)
    write(root / RECORD, dict(schema=1, kind='published_native_narration_reuse', lesson=identity,
        fresh_capture=False, raw_capture_revalidated=False, **source,
        published_master=bindings['master'], published_timing=bindings['timing']))
    result = current_reuse_timing(stage, identity)
    write(timing_path, result)
    return timing_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.stage, args.lesson, args.baseline))


if __name__ == '__main__':
    main()
