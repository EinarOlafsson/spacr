#!/usr/bin/env python3
"""Reuse unchanged narration only after exact catalog, timing and byte checks.

No synthesis, translation, source overwrite or listening claim. This permits a
visual refresh to preserve approved voices despite a newer renderer runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from stage_lesson import DEFAULT_STAGE, REPO, read, write

CAPTION_LANGUAGES = ('da', 'de', 'is', 'ko', 'nb', 'sv')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def catalog_lesson(path, lesson_id):
    matches = [item for item in read(path)['lessons'] if item['id'] == lesson_id]
    if len(matches) != 1:
        raise ValueError('Expected exactly one retained lesson in each catalog')
    return matches[0]


def require_unchanged_lesson(original, published, staged):
    if original != published or original != staged:
        raise ValueError('Retained catalog differs from original or published lesson')


def require_timing(timing, lesson, language, voice, audio_hash):
    inputs = timing.get('render_inputs', {})
    if (inputs.get('lesson') != lesson['id'] or inputs.get('language') != language
            or inputs.get('voice') != voice or timing.get('language') != language
            or timing.get('voice') != voice or timing.get('media_sha256') != audio_hash
            or [scene.get('narration') for scene in inputs.get('scenes', [])] !=
               [scene['narration'] for scene in lesson['scenes']]):
        raise ValueError('Retained timing identity, narration or media hash differs')


def retained_sources(stage, original, baseline, lesson_id, inventory, *, require_staged):
    """Check all sources before copying; optionally recheck every staged byte."""
    stage, original, baseline = map(lambda p: Path(p).resolve(), (stage, original, baseline))
    if (Path(lesson_id).name != lesson_id or lesson_id in {'.', '..'}
            or stage == original):
        raise ValueError('Expected a private stage and one lesson identity')
    tracks, catalogs = [], []
    for language in [*inventory, *CAPTION_LANGUAGES]:
        filename = f"{'captions' if language in CAPTION_LANGUAGES else 'lessons'}_{language}.json"
        source = catalog_lesson(original / 'catalog' / filename, lesson_id)
        published = catalog_lesson(baseline / filename, lesson_id)
        staged = catalog_lesson(stage / 'catalog' / filename, lesson_id)
        require_unchanged_lesson(source, published, staged)
        catalogs.append({'language': language, 'lesson_sha256': hashlib.sha256(
            json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
        for voice in inventory.get(language, []):
            path = original / 'production' / lesson_id / 'audio' / language / (voice + '.m4a')
            hashes = {suffix: digest(path.with_suffix(suffix)) for suffix in ('.m4a', '.json')}
            require_timing(read(path.with_suffix('.json')), source, language, voice, hashes['.m4a'])
            for suffix, expected in hashes.items():
                dest = stage / 'production' / lesson_id / 'audio' / language / (voice + suffix)
                if (require_staged or dest.exists()) and digest(dest) != expected:
                    raise ValueError('Retained audio or timing changed in staging')
            tracks.append({'language': language, 'voice': voice,
                           'audio_sha256': hashes['.m4a'], 'timing_sha256': hashes['.json']})
    return {'lesson': lesson_id, 'catalogs': catalogs, 'tracks': tracks,
            'resynthesized': False, 'current_runtime_freshness_claimed': False,
            'new_translation_review_claimed': False, 'human_listening_review': False,
            'published': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lesson', required=True)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    args = parser.parse_args()
    from check_completed_matrix import voice_matrix
    original = DEFAULT_STAGE.parent
    inventory = voice_matrix(original / 'tools/render_all_voices.py')
    baseline = REPO / 'docs/source/_extra/tutorials/catalog'
    retained_sources(args.stage, original, baseline, args.lesson, inventory, require_staged=False)
    for language, voices in inventory.items():
        for voice in voices:
            for suffix in ('.m4a', '.json'):
                relative = Path('production') / args.lesson / 'audio' / language / (voice + suffix)
                destination = args.stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    with (original / relative).open('rb') as source, destination.open('xb') as output:
                        shutil.copyfileobj(source, output)
    report = retained_sources(args.stage, original, baseline, args.lesson, inventory, require_staged=True)
    write(args.stage / 'production' / args.lesson / 'retained-narration.json', report)
    print(f"Retained {len(report['tracks'])} byte-identical tracks; no synthesis or publication.")


if __name__ == '__main__':
    main()
