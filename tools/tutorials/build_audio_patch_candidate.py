#!/usr/bin/env python3
"""Replace existing English narration while preserving scripts and visual media.

The original visual clock remains pinned independently of corrected narration.
The candidate is private and requires the normal browser and publication gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tempfile

from append_staged_lessons import parse_javascript
from audit_staged_catalogs import CATALOGS
from build_release_candidate import copy_checked
from check_completed_matrix import digest
from retain_narration import require_timing
from stage_lesson import REPO, read, write
from validate_candidate import validate


def patch_plan(baseline, stage, identities):
    """Require unchanged scripts, exact old offerings and genuine new timing."""
    baseline, stage = Path(baseline), Path(stage)
    if (not identities or len(identities) != len(set(identities))
            or any(not re.fullmatch(r'[0-9]+_[a-z0-9_]+', key) for key in identities)):
        raise ValueError('Select unique existing lesson identities')
    manifest = read(baseline / 'release-manifest.json')
    records = {row['path']: row for row in manifest['files']}
    english = {row['id']: row for row in read(baseline / 'web/catalog/lessons_en.json')['lessons']}
    for name in CATALOGS:
        if digest(stage / 'catalog' / name) != digest(baseline / 'web/catalog' / name):
            raise ValueError('Audio correction must preserve every catalog byte')
    planned = []
    for identity in identities:
        lesson = english.get(identity)
        if not lesson or lesson.get('status') == 'coming_soon':
            raise ValueError('Select an existing ready lesson')
        canonical = deepcopy(lesson)
        declared = canonical.pop('narration_voices', {})
        if (canonical != read(REPO / 'tools/tutorials/lessons' / (identity + '.json'))
                or canonical != read(stage / 'production' / identity / 'lesson.en.json')):
            raise ValueError('Audio correction cannot change canonical narration')
        voices = declared.get('en', [])
        if not voices or 'af_heart' not in voices or len(set(voices)) != len(voices):
            raise ValueError('Expected the original English voice offerings')
        directory = stage / 'production' / identity / 'audio/en'
        if {path.name for path in directory.iterdir() if path.is_file()} != {
                voice + suffix for voice in voices for suffix in ('.m4a', '.json')}:
            raise ValueError('English output inventory differs from original offerings')
        for voice in voices:
            relative = f'media_host/{identity}/audio/en/{voice}.m4a'
            audio = directory / (voice + '.m4a')
            timing = audio.with_suffix('.json')
            old = records.get(relative)
            if not old or relative.removesuffix('.m4a') + '.json' not in records:
                raise ValueError('Original manifest omits a declared English track')
            require_timing(read(timing), lesson, 'en', voice, digest(audio))
            planned.append(dict(lesson=identity, voice=voice, audio=str(audio),
                                timing=str(timing), audio_sha256=digest(audio),
                                timing_sha256=digest(timing), original_audio_sha256=old['sha256']))
    return planned


def verify_current_tracks(baseline, planned):
    """Use the normal current-source fingerprint and complete AAC decoder."""
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import verify_audio_release as audio
    lessons = {row['id']: row for row in read(baseline / 'web/catalog/lessons_en.json')['lessons']}
    for track in planned:
        voice, lesson = track['voice'], lessons[track['lesson']]
        code = 'b' if voice.startswith('b') else 'a'
        dialect = audio.narration_dialect('en', code, voice)
        speed = audio.resolve_voice_speed(voice)
        plans = audio.prepare_scene_plans(lesson, 'en', dialect, speed, voice=voice)
        errors = audio.check_track(audio.TrackSpec(Path(track['audio']), lesson, 'en',
                                                  code, voice, dialect, speed, plans))
        if errors:
            raise ValueError(f"{track['lesson']}/{voice}: {errors}")


def build(baseline, stage, identities, *, refresh_player=False):
    """Create a separate candidate without altering any baseline file."""
    baseline, stage = Path(baseline).resolve(), Path(stage).resolve()
    validate(baseline, include_hosted_media=True)
    manifest = read(baseline / 'release-manifest.json')
    browser = read(baseline / 'checks/candidate-browser-checks.json')
    if (not browser.get('passed') or browser.get('manifest_sha256') != digest(baseline / 'release-manifest.json')
            or len(browser['ready_playback_cases']) != manifest['ready_lessons']
            or not all(row.get('passed') for row in browser['ready_playback_cases'])):
        raise ValueError('The complete baseline must have passing browser evidence')
    planned = patch_plan(baseline, stage, identities)
    verify_current_tracks(baseline, planned)
    target = Path(tempfile.mkdtemp(prefix='release-candidate-audio-', dir=stage))
    replacements = {}
    player_patch = None
    if refresh_player:
        source = REPO / 'tools/tutorials/authoring/web/app_v2.js'
        original = next(row for row in manifest['files'] if row['path'] == 'web/app_v2.js')
        player_patch = dict(source=str(source), original_sha256=original['sha256'],
                            sha256=digest(source))
        replacements['web/app_v2.js'] = (source, player_patch['sha256'])
    for track in planned:
        prefix = f"media_host/{track['lesson']}/audio/en/{track['voice']}"
        replacements[prefix + '.m4a'] = (Path(track['audio']), track['audio_sha256'])
        replacements[prefix + '.json'] = (Path(track['timing']), track['timing_sha256'])
    records = []
    for record in manifest['files']:
        source, expected = replacements.get(record['path'], (baseline / record['path'], record['sha256']))
        copy_checked(source, target / record['path'], records, target, expected)
    javascript = parse_javascript((baseline / 'web/lesson_catalog.js').read_text())
    timing_references = []
    for lesson in javascript['lessons']:
        if lesson['id'] not in identities:
            continue
        identity = lesson['id']
        if lesson.get('visual_timings'):
            source = baseline / 'media_host' / lesson['visual_timings']
            if not source.is_file():
                raise ValueError('Original explicit visual timing is missing')
            if 'media_host/' + lesson['visual_timings'] in replacements:
                lesson['visual_timings'] = f'{identity}/video/original-visual-timings.json'
                copy_checked(source, target / 'media_host' / lesson['visual_timings'], records, target, digest(source))
        else:
            source = baseline / 'media_host' / identity / 'audio/en/af_heart.json'
            lesson['visual_timings'] = f'{identity}/video/original-visual-timings.json'
            copy_checked(source, target / 'media_host' / lesson['visual_timings'], records, target, digest(source))
        timing_references.append(dict(lesson=identity, path=lesson['visual_timings'],
                                     original_visual_timing_sha256=digest(source)))
    catalog_path = target / 'web/lesson_catalog.js'
    catalog_path.write_text('"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze('
                            + json.dumps(javascript, ensure_ascii=False) + ');\n')
    records = [row for row in records if row['path'] != 'web/lesson_catalog.js']
    records.append(dict(path='web/lesson_catalog.js', sha256=digest(catalog_path), bytes=catalog_path.stat().st_size))
    report = deepcopy(manifest)
    report.update(scope='Private English audio correction; every lesson catalog, other narration and visual master preserved',
                  files=sorted(records, key=lambda row: row['path']),
                  baseline_manifest_sha256=digest(baseline / 'release-manifest.json'),
                  audio_patch_tracks=planned, original_visual_timings=timing_references,
                  audio_patch_player=player_patch,
                  refreshed_lessons=list(identities), preserved_lessons=manifest['routes']-len(identities),
                  appended_lessons=[], withdrawn_lessons=[], retained_narration={},
                  new_lesson_tracks={}, release_hold=True, uploaded=False, published=False)
    report['web_bytes'] = sum(row['bytes'] for row in records if row['path'].startswith('web/'))
    report['media_host_bytes'] = sum(row['bytes'] for row in records if row['path'].startswith('media_host/'))
    if report['web_bytes'] > report['ceiling_bytes']:
        raise ValueError('Audio correction exceeds the existing web budget')
    write(target / 'release-manifest.json', report)
    validate(target, include_hosted_media=True)
    print('CANDIDATE', target, flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--lesson', action='append', required=True)
    parser.add_argument('--refresh-player', action='store_true',
                        help='Use the current normal authoring player; requires new browser acceptance')
    args = parser.parse_args()
    build(args.baseline, args.stage, args.lesson, refresh_player=args.refresh_player)


if __name__ == '__main__':
    main()
