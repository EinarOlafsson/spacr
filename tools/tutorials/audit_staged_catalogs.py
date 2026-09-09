#!/usr/bin/env python3
"""Prove that a partial tutorial refresh preserves every unselected lesson.

This is a local structural audit, not semantic approval of an old tutorial or
proof that remote media is unchanged. It never writes catalogs or media.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stage_lesson import DEFAULT_STAGE, REPO, read, write

CATALOGS = ['lessons_' + lang + '.json' for lang in
            ('en', 'es', 'fr', 'hi', 'it', 'ja', 'pt-BR', 'zh-CN')] + [
            'captions_' + lang + '.json' for lang in ('da', 'de', 'is', 'ko', 'nb', 'sv')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


def indexed(path):
    lessons = read(path)['lessons']
    result = {lesson['id']: lesson for lesson in lessons}
    if len(result) != len(lessons):
        raise ValueError(f'Duplicate lesson identity: {path.name}')
    return result


def audit(baseline, stage, refreshed):
    original = indexed(baseline / 'lessons_en.json')
    english = indexed(stage / 'catalog/lessons_en.json')
    if [key for key in english if key in original] != list(original):
        raise ValueError('Existing lesson identities and their catalog order must survive')
    if not refreshed <= set(english) or set(english) - set(original) - refreshed:
        raise ValueError('Every new or refreshed lesson must be explicitly selected')
    retained = [key for key in original if key not in refreshed]
    reports = []
    for filename in CATALOGS:
        before = indexed(baseline / filename)
        after = indexed(stage / 'catalog' / filename)
        if list(after) != list(english):
            raise ValueError(f'Catalog identities/order disagree: {filename}')
        for key, lesson in after.items():
            if len(lesson['scenes']) != len(english[key]['scenes']):
                raise ValueError(f'Scene count disagrees: {filename}/{key}')
            if key in retained and lesson != before[key]:
                raise ValueError(f'Unselected lesson was modified: {filename}/{key}')
            if key in refreshed:
                for field in ('number', 'app_key', 'host_app_key'):
                    if lesson.get(field) != english[key].get(field):
                        raise ValueError(f'Refreshed route disagrees: {filename}/{key}/{field}')
        reports.append({'catalog': filename, 'retained_lessons': len(retained),
                        'retained_content_sha256': digest([after[key] for key in retained])})
    for key in retained:
        folder = stage / 'production' / key
        replacements = [p for p in folder.rglob('*') if p.is_file() and
                        (p.suffix in {'.m4a', '.mp4'} or p.name == 'poster.jpg')]
        if replacements:
            raise ValueError(f'Unselected lesson has replacement media staged: {key}')
    return {'passed': True, 'catalog_count': len(CATALOGS),
            'lesson_count': len(english),
            'scene_count': sum(len(l['scenes']) for l in english.values()),
            'refreshed_lesson_ids': sorted(refreshed), 'retained_lesson_ids': retained,
            'retained_count': len(retained), 'catalog_checks': reports,
            'unselected_replacement_media_staged': False,
            'old_lesson_semantic_or_live_media_approval': False, 'published': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--refreshed', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof = audit(REPO / 'docs/source/_extra/tutorials/catalog', args.stage,
                  set(args.refreshed))
    write(args.output, proof)
    print(f"{proof['catalog_count']} catalogs agree on {proof['lesson_count']} lessons / "
          f"{proof['scene_count']} scenes; {proof['retained_count']} unselected lessons preserved.")


if __name__ == '__main__':
    main()
