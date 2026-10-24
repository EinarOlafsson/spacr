"""Plan a shared native video timeline from every source-bound narration track.

Real clips play completely at their recorded speed. Each reference scene covers
both its complete capture and the longest accepted voice. This file describes
visuals, not speech.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import sys
from stage_lesson import read, write


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def longest_scene_timing(lesson, tracks, native_scenes, *, fps=30, native_durations=None):
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
        native_duration = (native_durations or {}).get(index, 0)
        duration = math.ceil(max(longest, native_duration) * fps - 1e-9) / fps
        scenes.append({'scene': index + 1, 'speech_start': elapsed,
                       'speech_end': elapsed + duration, 'scene_end': elapsed + duration,
                       'duration': duration, 'text': authored['narration'],
                       'visual': authored['visual'], 'longest_narration_duration': longest,
                       'complete_native_duration': native_duration,
                       'native_live_video': index in native_scenes})
        elapsed += duration
    return {'schema': 1, 'kind': 'native_visual_timing', 'native_live_video': True,
            'lesson': lesson['id'], 'fps': fps, 'total_duration': elapsed,
            'scenes': scenes, 'playback_speed': 1, 'looped_or_stretched': False}


def complete_native_duration(receipt, fps):
    """Use decoded frames, avoiding the container's rounded millisecond duration."""
    frames = receipt.get('frames')
    if (fps <= 0 or Fraction(receipt.get('encoded_fps', '0')) != fps
            or not isinstance(frames, list) or not frames
            or any(frame.get('pts') != index for index, frame in enumerate(frames))):
        raise ValueError('Native capture requires contiguous decoded frames at the visual frame rate')
    duration = len(frames) / fps
    recorded = float(receipt['duration'])
    if not math.isfinite(recorded) or abs(recorded - duration) > .002:
        raise ValueError('Native capture duration differs from its decoded frame clock')
    return duration


def _mixed_prose_native_tracks(stage, identity, english):
    """Join explicit mixed speech to the same immutable native-reuse source."""
    if identity not in {'20_cellpose_masks', '78_spacr_screens', '80_image_analysis_pathways'}:
        raise ValueError('Mixed prose/native is limited to the three reviewed lessons')
    from published_scene_audio_refresh import regular
    from published_native_reuse import RECORD as REUSE_RECORD, _published_inputs
    from published_prose_audio_refresh import KIND, RECORD, checked_tracks
    from check_completed_matrix import voice_matrix
    from stage_lesson import REPO
    root = Path(stage) / 'production' / identity
    if any((root / name).exists() or (root / name).is_symlink() for name in (
            'published-retained-narration.json', 'published-native-edit.json',
            'published-native-mix.json')):
        raise ValueError('Choose explicit mixed prose with unchanged native reuse only')
    mixed_path = regular(root, RECORD)
    mixed = read(mixed_path)
    reuse = read(regular(root, REUSE_RECORD))
    baseline, original, bindings, source = _published_inputs(reuse['baseline_root'], identity)
    expected = dict(schema=1, kind='published_native_narration_reuse', lesson=identity,
                    fresh_capture=False, raw_capture_revalidated=False, **source,
                    published_master=bindings['master'], published_timing=bindings['timing'])
    if reuse != expected or english != read(regular(root, 'lesson.en.json')):
        raise ValueError('Exact native reuse record/current staged English required')
    provenance = mixed.get('source', {})
    visual = dict(kind='native_visual_timing', path=bindings['timing']['path'],
                  sha256=bindings['timing']['sha256'], published_bindings=bindings)
    if (mixed.get('schema') != 1 or mixed.get('kind') != KIND
            or provenance.get('lesson') != identity
            or provenance.get('native_adapter_required') is not True
            or provenance.get('baseline_root') != str(baseline)
            or provenance.get('baseline_manifest_sha256') != source['baseline_manifest_sha256']
            or provenance.get('publication_receipt_sha256') != source['baseline_publication_receipt_sha256']
            or provenance.get('original_visual_clock') != visual):
        raise ValueError('Mixed speech/native baseline, master or clock differs')
    inventory = voice_matrix(REPO / 'tools/tutorials/authoring/tools/render_all_voices.py')
    wanted = {(language, voice) for language, voices in inventory.items() for voice in voices}
    original_proofs = original.get('source_tracks', [])
    if (len(original_proofs) != 50 or len(wanted) != 50
            or {(p.get('language'), p.get('voice')) for p in original_proofs} != wanted):
        raise ValueError('Require all fifty original native source proofs')
    declared, proofs = checked_tracks(Path(stage), baseline, identity, decode=False)
    if (set(declared) != set(inventory)
            or any(len(declared[language]) != len(inventory[language])
                   or set(declared[language]) != set(inventory[language]) for language in inventory)
            or len(proofs) != 50
            or {(p.get('language'), p.get('voice')) for p in proofs} != wanted):
        raise ValueError('Mixed native narration must preserve every original offering')
    original_audio = {(p['language'], p['voice']): p['audio_sha256']
                      for p in provenance.get('tracks', [])}
    if len(provenance.get('tracks', [])) != 50 or set(original_audio) != wanted:
        raise ValueError('Mixed original track provenance is incomplete')
    tracks = []
    fields = {'language', 'voice', 'audio_sha256', 'timing_sha256', 'provenance_kind',
              'mixed_record_sha256', 'original_audio_sha256'}
    for proof in proofs:
        language, voice = proof['language'], proof['voice']
        path = regular(root, f'audio/{language}/{voice}.json')
        timing = read(path)
        media = regular(root, f'audio/{language}/{voice}.m4a')
        if (set(proof) != fields or proof['provenance_kind'] != KIND
                or proof['mixed_record_sha256'] != sha256(mixed_path)
                or proof['original_audio_sha256'] != original_audio[language, voice]
                or proof['timing_sha256'] != sha256(path)
                or timing.get('media_sha256') != proof['audio_sha256']
                or timing.get('media_bytes') != media.stat().st_size
                or timing.get('language') != language or timing.get('voice') != voice):
            raise ValueError('Mixed native track/proof binding changed')
        tracks.append(timing)
    return tracks, [dict(proof) for proof in proofs]


def verified_narration_tracks(stage, identity, english):
    """Validate fresh tracks, or explicitly bound immutable published narration."""
    mixed = Path(stage) / 'production' / identity / 'published-prose-audio-refresh.json'
    if mixed.exists() or mixed.is_symlink():
        return _mixed_prose_native_tracks(stage, identity, english)
    from published_retained_narration import RECORD, checked_published_narration
    record = Path(stage) / 'production' / identity / RECORD
    if record.exists() or record.is_symlink():
        tracks, proofs = checked_published_narration(stage, identity)
        if english != read(Path(stage) / 'production' / identity / 'lesson.en.json'):
            raise ValueError('Retained narration canonical English changed')
        return tracks, proofs
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'authoring/tools'))
    from render_all_voices import (LANGUAGES, narration_dialect, prepare_scene_plans,
                                  resolve_voice_speed, track_fingerprint)
    root = stage / 'production' / identity
    tracks, proofs = [], []
    for language, (code, voices) in LANGUAGES.items():
        localized = next(row for row in read(stage / 'catalog' / f'lessons_{language}.json')['lessons']
                         if row['id'] == identity)
        if ([row['visual'] for row in localized['scenes']] !=
                [row['visual'] for row in english['scenes']]
                or (language == 'en' and [row['narration'] for row in localized['scenes']] !=
                    [row['narration'] for row in english['scenes']])):
            raise ValueError('Narration catalog scene order or canonical English changed')
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
    return tracks, proofs


def plan(stage, identity):
    """Validate all fifty current track inputs before measuring footage gaps."""
    root = stage / 'production' / identity
    english = read(root / 'lesson.en.json')
    spec = read(root / 'scenes.json')
    if len(spec['scenes']) != len(english['scenes']):
        raise ValueError('Visual specification differs from the English scene structure')
    native = {index for index, scene in enumerate(spec['scenes']) if scene.get('clip')}
    if not native:
        raise ValueError('This lesson has no genuine native clips')
    tracks, proofs = verified_narration_tracks(stage, identity, english)
    fps = int(spec.get('fps', 30))
    native_durations, native_sources = {}, []
    for index in sorted(native):
        clip = spec['scenes'][index]['clip']
        receipt_path = root / clip['receipt']
        receipt = read(receipt_path)
        if (sha256(root / clip['video']) != clip['sha256']
                or sha256(receipt_path) != clip['receipt_sha256']
                or receipt.get('sha256') != clip['sha256']
                or not receipt.get('full_decode_passed') or receipt.get('audio_streams') != 0):
            raise ValueError('Native capture evidence changed')
        native_durations[index] = complete_native_duration(receipt, fps)
        native_sources.append({'scene': index + 1,
            'clip_sha256': clip['sha256'], 'receipt_sha256': clip['receipt_sha256']})
    result = longest_scene_timing(english, tracks, native, fps=fps,
                                 native_durations=native_durations)
    result['complete_native_clips'] = True
    result['english_sha256'] = hashlib.sha256(json.dumps(english, sort_keys=True,
                                                       ensure_ascii=False).encode()).hexdigest()
    result['source_tracks'] = proofs
    result['visual_spec_sha256'] = sha256(root / 'scenes.json')
    result['native_clip_sources'] = native_sources
    gaps, capture_plan = [], {}
    for index, scene in enumerate(result['scenes']):
        if index not in native:
            continue
        available = native_durations[index]
        if available + 1e-9 < scene['duration']:
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
    if (stage / 'production' / identity / 'published-native-mix.json').exists():
        from published_native_mix import checked_mixed_native_timing
        return checked_mixed_native_timing(stage, identity)
    if (stage / 'production' / identity / 'published-native-edit.json').exists():
        from published_native_edit import checked_edited_native_timing
        return checked_edited_native_timing(stage, identity)
    if (stage / 'production' / identity / 'published-native-reuse.json').exists():
        from published_native_reuse import checked_published_native_timing
        return checked_published_native_timing(stage, identity)
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
