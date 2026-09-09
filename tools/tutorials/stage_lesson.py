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

REPO = Path(__file__).resolve().parents[2]
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


def stage_lesson(lesson_path, capture_module, stage, *, check_only=False):
    from PIL import Image
    lesson = read(lesson_path)
    catalog_path = stage / 'catalog/lessons_en.json'
    baseline = REPO / 'docs/source/_extra/tutorials/catalog'
    catalog = read(catalog_path if catalog_path.exists() else baseline / 'lessons_en.json')
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
        expected = max(item['number'] for item in catalog['lessons']) + 1
        if lesson['number'] != expected:
            raise ValueError(f'New lesson must append with number {expected}')
        catalog['lessons'].append(lesson)
    if check_only:
        print(f"Validated {lesson['id']}: {len(scenes)} real capture scenes; catalogs unchanged.")
        return
    write(catalog_path, catalog)
    write(destination / 'visual.json', {'size': [3840, 2160], 'fps': 30,
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
    args = parser.parse_args()
    stage_lesson(args.lesson, args.capture_module, args.stage.resolve(), check_only=args.check_only)


if __name__ == '__main__':
    main()
