"""Plan a shared native video timeline from every source-bound narration track.

Real clips play at their recorded speed. Each reference scene covers the
longest accepted voice; shorter narration advances to the next recorded scene
instead of changing animation speed. This file describes visuals, not speech.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from stage_lesson import read, write


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_scene_timing(lesson, tracks, native_scenes, *, fps=30):
    """Keep scene identities and cover each voice with whole native frames."""
    if fps <= 0 or not tracks:
        raise ValueError('A native timeline needs tracks and a positive frame rate')
    count = len(lesson['scenes'])
    if any(len(track['scenes']) != count for track in tracks):
        raise ValueError('Narration and visual scene counts differ')
    scenes, elapsed = [], 0.0
    for index, authored in enumerate(lesson['scenes']):
        durations = [float(track['scenes'][index]['duration']) for track in tracks]
        if any(not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError('Invalid narration scene duration')
        longest = max(durations)
        duration = math.ceil(longest * fps - 1e-9) / fps
        scenes.append({'scene': index + 1, 'speech_start': elapsed,
                       'speech_end': elapsed + duration, 'scene_end': elapsed + duration,
                       'duration': duration, 'text': authored['narration'],
                       'visual': authored['visual'], 'longest_narration_duration': longest,
                       'native_live_video': index in native_scenes})
        elapsed += duration
    return {'schema': 1, 'kind': 'native_visual_timing', 'native_live_video': True,
            'lesson': lesson['id'], 'fps': fps, 'total_duration': elapsed,
            'scenes': scenes, 'playback_speed': 1, 'looped_or_stretched': False}


def plan(stage, identity):
    """Validate all fifty current track inputs before measuring footage gaps."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'authoring/tools'))
    from render_all_voices import (LANGUAGES, narration_dialect, prepare_scene_plans,
                                  resolve_voice_speed, track_fingerprint)
    root = stage / 'production' / identity
    english = read(root / 'lesson.en.json')
    spec = read(root / 'scenes.json')
    if len(spec['scenes']) != len(english['scenes']):
        raise ValueError('Visual specification differs from the English scene structure')
    native = {index for index, scene in enumerate(spec['scenes']) if scene.get('clip')}
    if not native:
        raise ValueError('This lesson has no genuine native clips')
    tracks, proofs = [], []
    for language, (code, voices) in LANGUAGES.items():
        localized = next(row for row in read(stage / 'catalog' / f'lessons_{language}.json')['lessons']
                         if row['id'] == identity)
        for voice in voices:
            path = root / 'audio' / language / (voice + '.json')
            timing = read(path)
            media = path.with_suffix('.m4a')
            if (timing.get('media_sha256') != sha256(media)
                    or timing.get('media_bytes') != media.stat().st_size
                    or timing.get('language') != language or timing.get('voice') != voice
                    or [row['text'] for row in timing['scenes']]
                       != [row['narration'] for row in localized['scenes']]):
                raise ValueError(f'Narration media or source changed: {path}')
            actual_code = 'b' if language == 'en' and voice.startswith('b') else code
            dialect = narration_dialect(language, actual_code, voice)
            speed = resolve_voice_speed(voice)
            plans = prepare_scene_plans(localized, language, dialect, speed, voice=voice)
            fingerprint, inputs = track_fingerprint(localized, language, actual_code, dialect,
                voice, speed, plans, runtime_identity=timing['render_inputs']['synthesis_runtime'])
            if fingerprint != timing['render_fingerprint'] or inputs != timing['render_inputs']:
                raise ValueError(f'Narration is not bound to current render inputs: {path}')
            tracks.append(timing)
            proofs.append({'language': language, 'voice': voice,
                           'timing_sha256': sha256(path), 'audio_sha256': timing['media_sha256'],
                           'render_fingerprint': fingerprint})
    result = longest_scene_timing(english, tracks, native, fps=int(spec.get('fps', 30)))
    result['english_sha256'] = hashlib.sha256(json.dumps(english, sort_keys=True,
                                                       ensure_ascii=False).encode()).hexdigest()
    result['source_tracks'] = proofs
    result['visual_spec_sha256'] = sha256(root / 'scenes.json')
    result['native_clip_sources'] = []
    gaps, capture_plan = [], {}
    for index, scene in enumerate(result['scenes']):
        if index not in native:
            continue
        clip = spec['scenes'][index]['clip']
        receipt_path = root / clip['receipt']
        receipt = read(receipt_path)
        if (sha256(root / clip['video']) != clip['sha256']
                or sha256(receipt_path) != clip['receipt_sha256']
                or receipt.get('sha256') != clip['sha256']
                or not receipt.get('full_decode_passed') or receipt.get('audio_streams') != 0):
            raise ValueError('Native capture evidence changed')
        available = float(receipt['duration'])
        result['native_clip_sources'].append({'scene': index + 1,
            'clip_sha256': clip['sha256'], 'receipt_sha256': clip['receipt_sha256']})
        if available + 1 / result['fps'] < scene['duration']:
            gaps.append({'scene': index + 1, 'visual': scene['visual'],
                         'available_seconds': available, 'required_seconds': scene['duration']})
        capture_plan[scene['visual']] = max(capture_plan.get(scene['visual'], 0),
                                           round(scene['duration'] + 2, 3))
    result['coverage'] = {'accepted': not gaps, 'gaps': gaps,
                          'track_count': len(tracks), 'native_scene_count': len(native),
                          'capture_plan_seconds': capture_plan}
    return result


def checked_native_timing(stage, identity):
    """Return an accepted visual sidecar only when every input still matches."""
    path = stage / 'production' / identity / 'video' / 'native-live-timings.json'
    if not path.exists():
        spec = stage / 'production' / identity / 'scenes.json'
        if spec.exists() and any(scene.get('clip') for scene in read(spec)['scenes']):
            raise ValueError('Native clips require a verified fifty-voice visual timeline')
        return None
    stored = read(path)
    if not stored.get('coverage', {}).get('accepted') or stored != plan(stage, identity):
        raise ValueError('Native visual timeline is incomplete or changed after verification')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.stage.resolve(), args.lesson)
    write(args.output, result)
    print(f"{len(result['source_tracks'])} source-bound tracks; {len(result['coverage']['gaps'])} genuine footage duration gaps")
    return 0 if result['coverage']['accepted'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
