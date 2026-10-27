#!/usr/bin/env python3
"""Stage a changed lesson and verified capture geometry without publishing it.

Unselected lessons come from the current published catalog, not a potentially
older authoring copy. Translations and media remain isolated until the complete
language/voice matrix has passed review. Existing identifiers never change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
DEFAULT_STAGE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/refresh_2026-09-09')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def union(rectangles):
    if not rectangles:
        raise ValueError('A named focus must identify visible controls')
    x = min(r[0] for r in rectangles)
    y = min(r[1] for r in rectangles)
    right = max(r[0] + r[2] for r in rectangles)
    bottom = max(r[1] + r[3] for r in rectangles)
    return [x, y, right - x, bottom - y]


def require_append_identity(lesson, current, released):
    """Allow a numbered public placeholder to become real without renumbering it."""
    expected = max(item['number'] for item in current) + 1
    if lesson['number'] == expected:
        return
    known = [item for item in released if item['id'] == lesson['id']]
    if (any(item['number'] == lesson['number'] for item in current) or len(known) != 1
            or known[0].get('status') != 'coming_soon'
            or any(known[0].get(key) != lesson.get(key)
                   for key in ('number', 'app_key', 'host_app_key'))):
        raise ValueError(f'New lesson must append with number {expected} or its exact published placeholder identity')


def require_cellpose_acceptance(lesson, capture, frames):
    """Keep lesson 20 bound to its verified science and selected live clips."""
    if lesson['id'] != '20_cellpose_masks':
        return
    path = capture / 'scientific_acceptance.json'
    if not path.is_file():
        raise ValueError('Lesson 20 requires source-bound scientific acceptance')
    proof = read(path)
    canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True,
                                         ensure_ascii=False).encode()).hexdigest()
    if (proof.get('schema') != 1 or proof.get('accepted') is not True
            or proof.get('lesson') != lesson['id']
            or proof.get('canonical_english_sha256') != canonical
            or proof.get('accuracy_validated') is not False
            or proof.get('ground_truth_used') is not False):
        raise ValueError('Lesson 20 scientific acceptance differs from the lesson or its verified scope')
    hashes = proof.get('source_hashes')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Lesson 20 scientific acceptance has no bound source evidence')
    for source, expected in hashes.items():
        source = Path(source)
        if not source.is_absolute() or not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('Lesson 20 scientific acceptance has invalid source hashes')
        digest = hashlib.sha256()
        try:
            with source.open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
        except OSError as error:
            raise ValueError(f'Scientific acceptance evidence is unavailable: {source}') from error
        if digest.hexdigest() != expected:
            raise ValueError(f'Scientific acceptance evidence changed: {source}')
    selected = proof.get('selected_clips')
    expected_scenes = {scene['visual'] for scene in lesson['scenes']}
    if (not isinstance(selected, list) or len(selected) != len(expected_scenes)
            or any(not isinstance(row, dict) for row in selected)
            or {row.get('scene') for row in selected} != expected_scenes
            or set(frames) != expected_scenes):
        raise ValueError('Lesson 20 scientific acceptance must cover every scene exactly once')
    for row in selected:
        frame = frames[row['scene']]
        clip = frame.get('clip')
        origin = frame.get('composition_source', {})
        if (not isinstance(clip, dict) or not clip
                or any(not isinstance(row.get(key), str)
                       or not re.fullmatch(r'[0-9a-f]{64}', row[key])
                       for key in ('video_sha256', 'native_sha256', 'gate_sha256'))
                or origin.get('capture') != row.get('capture')
                or origin.get('visual') != row['scene']
                or clip.get('sha256') != row.get('video_sha256')
                or clip.get('native_sha256') != row.get('native_sha256')
                or row.get('decision') not in ('strict_pass', 'approved_capture_loss_near_miss')):
            raise ValueError('Lesson 20 scientific acceptance names different selected clips')
        for key, expected in (('video', row['video_sha256']),
                              ('native_video', row['native_sha256']),
                              ('receipt', clip.get('receipt_sha256'))):
            relative = Path(clip.get(key, ''))
            if (not relative.parts or relative.is_absolute() or '..' in relative.parts
                    or hashes.get(str(Path(row['capture']) / relative)) != expected
                    or not isinstance(expected, str)
                    or not re.fullmatch(r'[0-9a-f]{64}', expected)):
                raise ValueError('Lesson 20 scientific acceptance does not bind selected clip files')
        gate_path = Path(row['capture']) / f'{row["scene"]}.gate.json'
        if hashes.get(str(gate_path)) != row['gate_sha256']:
            raise ValueError('Lesson 20 scientific acceptance does not bind the selected gate')
        gate = read(gate_path)
        if (gate.get('app_freeze_seconds') or gate.get('app_slow_seconds')
                or float(gate.get('limits', {}).get('min_fps', 0)) < 27.9):
            raise ValueError('Lesson 20 scientific acceptance contains an app stall or weakened gate')
        if row['decision'] == 'strict_pass':
            valid = (gate.get('passed') is True and gate.get('reasons') == []
                     and float(gate.get('distinct_fps', 0)) >= 27.9)
        else:
            reasons = gate.get('reasons')
            valid = (gate.get('passed') is False
                     and gate.get('accepted_near_miss') == 'capture loss'
                     and float(gate.get('distinct_fps', 0)) >= 27.0
                     and not gate.get('app_unknown_seconds')
                     and bool(gate.get('capture_loss_seconds'))
                     and isinstance(reasons, list) and len(reasons) == 1
                     and reasons[0].startswith('choppy (capture loss):'))
        if not valid:
            raise ValueError('Lesson 20 scientific acceptance does not match the selected gate decision')


def stage_lesson(lesson_path, capture_module, stage, *, check_only=False, focus_map=None):
    from PIL import Image
    lesson = read(lesson_path)
    overrides = read(focus_map) if focus_map is not None else None
    if overrides is not None:
        canonical = hashlib.sha256(json.dumps(lesson, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if (overrides.get('english_sha256') != canonical
                or set(overrides.get('scenes', {})) != {s['visual'] for s in lesson['scenes']}):
            raise ValueError('Visual-only focus map must match the exact lesson and every scene')
    # A recapture changes layout, not prose. Its spotlights may replace the
    # authored ones only when pinned to this capture's exact frame bytes, so
    # the English lesson (and every translation review bound to it) is unchanged.
    recapture = overrides.get('recapture') if overrides is not None else None
    if recapture is not None and recapture.get('capture_module') != capture_module:
        raise ValueError('Recapture focus map names a different capture')
    catalog_path = stage / 'catalog/lessons_en.json'
    baseline = REPO / 'docs/source/_extra/tutorials/catalog'
    catalog = read(catalog_path if catalog_path.exists() else baseline / 'lessons_en.json')
    from lesson_redirects import INSTALL_ID, installation_placeholder
    if lesson['id'] == INSTALL_ID:
        catalog = installation_placeholder(catalog, catalog)
    identities = {item['id'] for item in catalog['lessons']}
    capture = stage / 'captures' / capture_module
    provenance = read(capture / 'provenance.json')
    if not provenance.get('completed_capture'):
        raise ValueError('Cannot stage an incomplete or failed capture')
    if (capture / 'batch_acceptance.json').exists():
        if not read(capture / 'batch_acceptance.json').get('accepted'):
            raise ValueError('Cannot stage a successful-example lesson from partial pipeline output')
    if (capture / 'scientific_acceptance.json').exists():
        if not read(capture / 'scientific_acceptance.json').get('accepted'):
            raise ValueError('Cannot stage an example that failed its real-data scientific checks')
    frames = read(capture / 'frames.json')
    require_cellpose_acceptance(lesson, capture, frames)
    destination = stage / 'production' / lesson['id']
    scenes = []
    for number, scene in enumerate(lesson['scenes'], 1):
        missing_links = set(scene.get('related_lessons', [])) - identities
        if missing_links:
            raise ValueError(f'Scene {number} links missing lessons: {missing_links}')
        frame = frames[scene['visual']]
        path = capture / frame['image']
        if hashlib.sha256(path.read_bytes()).hexdigest() != frame['sha256']:
            raise ValueError(f'Capture changed after measurement: {path}')
        with Image.open(path) as image:
            if image.size != (3840, 2160):
                raise ValueError(f'Not a native 4K capture: {path}')
        visual = {'image': os.path.relpath(path, destination), 'pointer': False,
                  'capture_sha256': frame['sha256']}
        if frame.get('clip'):
            clip = frame['clip']
            video = capture / clip['video']
            receipt_path = capture / clip['receipt']
            receipt = read(receipt_path)
            if (hashlib.sha256(video.read_bytes()).hexdigest() != clip['sha256']
                    or hashlib.sha256(receipt_path.read_bytes()).hexdigest() != clip['receipt_sha256']
                    or receipt.get('sha256') != clip['sha256']
                    or not receipt.get('full_decode_passed')
                    or receipt.get('audio_streams') != 0
                    or receipt.get('size') != [3840, 2160]
                    or not receipt.get('frames')):
                raise ValueError(f'Unverified actual application clip: {video}')
            visual['clip'] = {'video': os.path.relpath(video, destination),
                              'sha256': clip['sha256'],
                              'receipt': os.path.relpath(receipt_path, destination),
                              'receipt_sha256': clip['receipt_sha256'],
                              'duration': receipt['duration']}
        if recapture is not None and recapture.get('frames', {}).get(scene['visual']) != frame['sha256']:
            raise ValueError(f'Recapture focus is bound to different frame bytes: {scene["visual"]}')
        replaces_authored = (recapture is not None
                             and overrides['scenes'][scene['visual']] is not None)
        if not replaces_authored:
            if 'focus' in scene:
                visual['focus'] = scene['focus']
            if 'focus_modules' in scene:
                selected = []
                for key in scene['focus_modules']:
                    matches = [b for b in frame['buttons'] if
                               b.get('module_key') == key or b.get('nav_key') == key]
                    # A Home tile and the left navigation can share a key. Use the
                    # tile, which is the larger recorded control, not both regions.
                    if not matches:
                        raise ValueError(f'No visible control for module {key}')
                    selected.append(max(matches, key=lambda b: b['rect'][2] * b['rect'][3])['rect'])
                visual['focus'] = union(selected)
            if 'focus_buttons' in scene:
                visual['focus'] = union([b['rect'] for b in frame['buttons']
                                         if b['name'] == scene['focus_buttons']])
        if overrides is not None:
            if recapture is None and 'focus' in visual:
                raise ValueError('A visual-only focus map cannot override an authored focus')
            region = overrides['scenes'][scene['visual']]
            if region is not None:
                visual['focus'] = region
        if 'focus' in visual:
            x, y, width, height = visual['focus']
            if min(x, y) < 0 or min(width, height) <= 0 or x + width > 3840 or y + height > 2160:
                raise ValueError(f'Scene {number} focus leaves the frame')
        scenes.append(visual)
    replaced = False
    for index, old in enumerate(catalog['lessons']):
        if old['id'] == lesson['id']:
            if old['number'] != lesson['number']:
                raise ValueError('A refreshed lesson must retain its existing number')
            catalog['lessons'][index] = lesson
            replaced = True
            break
    if not replaced:
        require_append_identity(lesson, catalog['lessons'],
                                read(baseline / 'lessons_en.json')['lessons'])
        catalog['lessons'].append(lesson)
    if check_only:
        print(f"Validated {lesson['id']}: {len(scenes)} real capture scenes; catalogs unchanged.")
        return
    write(catalog_path, catalog)
    write(destination / 'visual.json', {'size': [3840, 2160], 'fps': 30,
          'english_sha256': hashlib.sha256(json.dumps(lesson, sort_keys=True,
                                                     ensure_ascii=False).encode()).hexdigest(),
          'capture_source': provenance, 'scenes': scenes})
    write(destination / 'lesson.en.json', lesson)
    write(destination / 'refresh-status.json', {
        'status': 'english_staged_not_publishable', 'lesson': lesson['id'],
        'english_sha256': hashlib.sha256(lesson_path.read_bytes()).hexdigest(),
        'scene_count': len(scenes), 'capture_source': provenance,
        'translation_review_complete': False, 'all_50_voices_verified': False,
        'shared_master_verified': False,
    })
    print(f"Staged {lesson['id']}: {len(scenes)} real capture scenes; not published.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lesson', type=Path)
    parser.add_argument('--capture-module', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--check-only', action='store_true', help='Validate all captures and links without changing any catalog')
    parser.add_argument('--focus-map', type=Path,
                        help='Source-pinned visual-only focus geometry for retained narration')
    args = parser.parse_args()
    stage_lesson(args.lesson, args.capture_module, args.stage.resolve(),
                 check_only=args.check_only, focus_map=args.focus_map)


if __name__ == '__main__':
    main()
