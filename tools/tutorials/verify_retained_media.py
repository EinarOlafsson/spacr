"""Verify an unchanged original lesson without copying or regenerating media.

Evidence goes into the private refresh directory. Original narration, timing,
captions, poster and video stay in place; current-runtime or human review is not
implied by successful decoding. The current GUI workflow is verified separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from check_completed_matrix import voice_matrix
from retain_narration import digest, retained_sources
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def retained_media_sources(stage, original, baseline, lesson, inventory):
    """Reject overrides: the browser must serve exactly the audited old media."""
    destination = Path(stage) / 'production' / lesson
    if destination.exists():
        raise ValueError('Whole-media retention cannot have a staged production override')
    proof = retained_sources(stage, original, baseline, lesson, inventory, require_staged=False)
    base = Path(original) / 'production' / lesson
    proof['media'] = {key: {'path': str(path), 'sha256': digest(path),
                            'bytes': path.stat().st_size}
                      for key, path in {'video': base / 'video' / f'{lesson}_silent.mp4',
                                        'poster': base / 'poster.jpg'}.items()}
    proof['audio_root'] = str(base / 'audio')
    proof['media_regenerated'] = False
    proof['media_copied'] = False
    return proof


def require_master(probe, duration):
    streams = probe['streams']
    if len(streams) != 1:
        raise ValueError('The retained master must have one silent video stream')
    stream = streams[0]
    if (stream.get('codec_type') != 'video' or stream.get('width') != 3840
            or stream.get('height') != 2160 or stream.get('r_frame_rate') != '30/1'
            or not math.isfinite(float(stream['duration']))
            or abs(float(stream['duration']) - float(duration)) > .1):
        raise ValueError('The retained master is not 4K/30 or does not match English timing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    original = DEFAULT_STAGE.parent
    baseline = REPO / 'docs/source/_extra/tutorials/catalog'
    inventory = voice_matrix(original / 'tools/render_all_voices.py')
    before = retained_media_sources(args.stage, original, baseline, args.lesson, inventory)
    video = before['media']['video']['path']
    probe = json.loads(subprocess.check_output(
        ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', video]))
    timing = read(Path(before['audio_root']) / 'en/af_heart.json')
    require_master(probe, timing['total_duration'])
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '2',
                    '-i', video, '-f', 'null', '-'], check=True)
    # Import only after the cheap source checks; this path never calls synthesis.
    os.environ.setdefault('USE_TF', '0')
    sys.path.insert(0, str(original / 'tools'))
    from verify_audio_release import check_track
    tracks = []
    for record in before['tracks']:
        path = Path(before['audio_root']) / record['language'] / (record['voice'] + '.m4a')
        errors = check_track(path)
        tracks.append(dict(record, errors=errors))
        print(f"{record['language']}/{record['voice']}: {len(errors)} errors", flush=True)
    after = retained_media_sources(args.stage, original, baseline, args.lesson, inventory)
    if before != after:
        raise ValueError('Original sources or media changed during the retention verification')
    report = dict(before, passed=all(not r['errors'] for r in tracks), tracks=tracks,
                  tracks_checked=len(tracks), video_probe=probe, video_fully_decoded=True,
                  original_sources_unchanged=True)
    write(args.stage / 'retention' / args.lesson / 'media-checks.json', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
