#!/usr/bin/env python3
"""Replace existing English narration while preserving scripts and visual media.

The original visual clock remains pinned independently of corrected narration.
The candidate is private and requires the normal browser and publication gates.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
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


def offering_patch_plan(baseline, stage, identities):
    """Add only the missing English matrix; retain every published track byte."""
    if len(identities) != 1 or identities[0] not in {'06_api', '48_hit_list'}:
        raise ValueError('Add English offerings for exactly one of Tutorial06 or Tutorial48')
    from apply_translation_review import translated_lesson
    from published_retained_narration import _authoring_without_generated_aliases
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    from render_all_voices import LANGUAGES

    identity = identities[0]
    baseline, stage = Path(baseline), Path(stage)
    manifest = read(baseline / 'release-manifest.json')
    records = {row['path']: row for row in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate baseline manifest paths')
    catalogs = {name: read(baseline / 'web/catalog' / name) for name in CATALOGS}
    if any(read(stage / 'catalog' / name) != value for name, value in catalogs.items()):
        raise ValueError('Offering addition must preserve all staged catalog objects')

    def selected(catalog):
        matches = [row for row in catalog['lessons'] if row['id'] == identity]
        if len(matches) != 1:
            raise ValueError('Expected one selected lesson per catalog')
        return matches[0]

    lesson = selected(catalogs['lessons_en.json'])
    canonical = {key: value for key, value in lesson.items() if key != 'narration_voices'}
    if (read(REPO / 'tools/tutorials/lessons' / f'{identity}.json') != canonical
            or read(stage / 'production' / identity / 'lesson.en.json') != canonical):
        raise ValueError('Offering addition cannot change canonical narration or metadata')
    inventory = {language: list(pair[1]) for language, pair in LANGUAGES.items()}
    previous = {**inventory, 'en': ['af_heart']}
    if sum(map(len, inventory.values())) != 50 or sum(map(len, previous.values())) != 27:
        raise ValueError('Expected normal50 inventory with23 missing English offerings')
    if lesson.get('narration_voices') != previous:
        raise ValueError('Expected exactly the original27 voice offerings')
    legacy_visual_source(baseline, identity, lesson, retain_explicit_picture_clock=True)
    for name, catalog in catalogs.items():
        localized = selected(catalog)
        if localized.get('narration_voices') != previous:
            raise ValueError('Published language voice inventories disagree')
        language = name.split('_', 1)[1].removesuffix('.json')
        if language != 'en':
            review = read(REPO / 'tools/tutorials/lessons/reviews' / f'{identity}.{language}.json')
            if review.get('lesson') != identity or review.get('language') != language:
                raise ValueError('Wrong source-bound translation review')
            reviewed = translated_lesson(review, lesson, localized)
            if _authoring_without_generated_aliases(reviewed, localized, language) != localized:
                raise ValueError('Reviewed translation differs from published prose')

    directory = stage / 'production' / identity / 'audio'
    expected_old = {f'{language}/{voice}{suffix}' for language, voices in previous.items()
                    for voice in voices for suffix in ('.m4a', '.json')}
    expected_new = {f'en/{voice}{suffix}' for voice in inventory['en'] if voice != 'af_heart'
                    for suffix in ('.m4a', '.json')}
    entries = list(directory.rglob('*'))
    if (directory.is_symlink() or any(path.is_symlink() for path in entries)
            or {path.relative_to(directory).as_posix() for path in entries if path.is_file()}
               != expected_old | expected_new):
        raise ValueError('Offering addition requires exact50-track staged inventory')
    prefix = f'media_host/{identity}/audio/'
    if {path[len(prefix):] for path in records if path.startswith(prefix)} != expected_old:
        raise ValueError('Published audio inventory differs from original27 offerings')
    for relative in sorted(expected_old):
        source, target = baseline / (prefix + relative), directory / relative
        row = records[prefix + relative]
        if (source.is_symlink() or (digest(source), source.stat().st_size) != (row['sha256'], row['bytes'])
                or (digest(target), target.stat().st_size) != (row['sha256'], row['bytes'])):
            raise ValueError('Original published narration changed or was dropped')
        if relative.endswith('.json'):
            language, filename = relative.split('/')
            voice = filename.removesuffix('.json')
            require_timing(read(target), selected(catalogs[f'lessons_{language}.json']),
                           language, voice, records[prefix + relative.removesuffix('.json') + '.m4a']['sha256'])
    planned = []
    for voice in inventory['en']:
        if voice == 'af_heart':
            continue
        audio = directory / 'en' / (voice + '.m4a')
        timing = audio.with_suffix('.json')
        require_timing(read(timing), lesson, 'en', voice, digest(audio))
        planned.append(dict(lesson=identity, language='en', voice=voice, audio=str(audio),
                            timing=str(timing), audio_sha256=digest(audio),
                            timing_sha256=digest(timing)))
    for catalog in catalogs.values():
        selected(catalog)['narration_voices'] = deepcopy(inventory)
    return catalogs, planned


def _scripts_only(original, current):
    """Allow spoken/display text only; preserve every indexed visual property."""
    before, after = deepcopy(original), deepcopy(current)
    for lesson in (before, after):
        scenes = lesson.get('scenes')
        if not isinstance(scenes, list) or not scenes:
            raise ValueError('Expected nonempty ordered scenes')
        for scene in scenes:
            if (not isinstance(scene.get('visual'), str) or not scene['visual']
                    or not isinstance(scene.get('narration'), str) or not scene['narration'].strip()):
                raise ValueError('Expected indexed visual identities and narration')
            scene.pop('narration', None)
            scene.pop('speech_text', None)
    if before != after:
        raise ValueError('Script refresh cannot change metadata, voices, order or visual structure')


def legacy_visual_source(baseline, identity, lesson, *, retain_explicit_picture_clock=False):
    """Bind the already-published legacy clock, never fabricate native timing."""
    runtime = parse_javascript((baseline / 'web/lesson_catalog.js').read_text())
    matches = [row for row in runtime['lessons'] if row['id'] == identity]
    if len(matches) != 1:
        raise ValueError('Expected exactly one published runtime lesson')
    relative = matches[0].get('visual_timings', f'{identity}/audio/en/af_heart.json')
    # Offering addition never changes Heart or the picture timeline. A prior
    # reviewed script refresh may deliberately pin its older picture clock.
    # Ordinary script-refresh callers still require exact narration below.
    independent_picture = (retain_explicit_picture_clock
        and matches[0].get('visual_timings') == f'{identity}/video/original-visual-timings.json')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path.parts[0] != identity:
        raise ValueError('Legacy visual clock must belong to the selected lesson')
    source = baseline / 'media_host' / path
    if independent_picture and (source.is_symlink() or not source.resolve().is_relative_to(Path(baseline).resolve())):
        raise ValueError('Explicit published picture clock must remain inside its baseline')
    timing = read(source)
    if (timing.get('schema') != 1 or timing.get('kind') or timing.get('native_live_video')
            or len(timing.get('scenes', [])) != len(lesson['scenes'])):
        raise ValueError('Script refresh supports only the existing non-native legacy clock')
    end = 0.0
    for index, scene in enumerate(timing['scenes'], 1):
        values = [scene.get(key) for key in ('speech_start', 'speech_end', 'scene_end', 'duration')]
        if (scene.get('scene') != index or scene.get('native_live_video')
                or (not independent_picture and scene.get('text') != lesson['scenes'][index - 1]['narration'])
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
            raise ValueError('Invalid indexed legacy visual clock')
        start, speech_end, stop, duration = values
        if (abs(start - end) > 1e-6 or not start < speech_end <= stop or duration <= 0
                or abs(stop - start - duration) > 1e-6):
            raise ValueError('Legacy visual clock has gaps or invalid scene bounds')
        end = stop
    total = timing.get('total_duration')
    if type(total) not in (int, float) or not math.isfinite(total) or abs(total - end) > 1e-6:
        raise ValueError('Legacy visual clock does not cover the published master')
    records = {row['path']: row for row in read(baseline / 'release-manifest.json')['files']}
    row = records.get('media_host/' + path.as_posix())
    if not row or (digest(source), source.stat().st_size) != (row['sha256'], row['bytes']):
        raise ValueError('Legacy visual clock differs from its published manifest')
    for relative in (f'media_host/{identity}/video/{identity}_silent.mp4',
                     f'media_host/{identity}/web/{identity}_silent.mp4',
                     f'web/production/{identity}/poster.jpg'):
        media = baseline / relative
        original = records.get(relative)
        if not original or not media.is_file() or media.stat().st_size != original['bytes']:
            raise ValueError('Missing original master, web copy or poster')
        # The owning build validates all manifest hashes before planning, and
        # copy_checked revalidates them before writing the fresh candidate.
    return source


def script_patch_plan(baseline, stage, identities):
    """Plan reviewed narration changes with exact published voice offerings."""
    from apply_translation_review import translated_lesson
    from audit_staged_catalogs import CATALOGS
    if (not identities or len(set(identities)) != len(identities)
            or any(not re.fullmatch(r'[0-9]+_[a-z0-9_]+', key) for key in identities)):
        raise ValueError('Select unique existing lesson identities')
    catalogs = {}
    old_catalogs = {}
    for name in CATALOGS:
        before, after = read(baseline / 'web/catalog' / name), read(stage / 'catalog' / name)
        if ([row['id'] for row in before['lessons']] != [row['id'] for row in after['lessons']]
                or len({row['id'] for row in before['lessons']}) != len(before['lessons'])):
            raise ValueError('Script refresh preserves every catalog identity and order')
        expected = deepcopy(before)
        for old, new, target in zip(before['lessons'], after['lessons'], expected['lessons']):
            if old['id'] in identities:
                _scripts_only(old, new)
                target['scenes'] = deepcopy(new['scenes'])
        if expected != after:
            raise ValueError('Unselected objects or catalog metadata changed')
        catalogs[name], old_catalogs[name] = after, before
    english = {row['id']: row for row in catalogs['lessons_en.json']['lessons']}
    previous = {row['id']: row for row in old_catalogs['lessons_en.json']['lessons']}
    manifest = read(baseline / 'release-manifest.json')
    records = {row['path']: row for row in manifest['files']}
    if len(records) != len(manifest['files']):
        raise ValueError('Duplicate baseline manifest paths')
    planned = []
    for identity in identities:
        lesson = english.get(identity)
        if not lesson or lesson.get('status') == 'coming_soon':
            raise ValueError('Select an existing ready lesson')
        canonical = read(REPO / 'tools/tutorials/lessons' / f'{identity}.json')
        if ({k: v for k, v in lesson.items() if k != 'narration_voices'} != canonical
                or read(stage / 'production' / identity / 'lesson.en.json') != canonical):
            raise ValueError('Refreshed English must match the current canonical and staged source')
        for name in CATALOGS:
            if name == 'lessons_en.json':
                continue
            language = name.split('_', 1)[1].removesuffix('.json')
            old = next(row for row in old_catalogs[name]['lessons'] if row['id'] == identity)
            current = next(row for row in catalogs[name]['lessons'] if row['id'] == identity)
            review = read(stage / 'production' / identity / f'review.{language}.json')
            if (review.get('lesson') != identity or review.get('language') != language
                    or not isinstance(review.get('review'), dict) or not review['review']
                    or translated_lesson(review, lesson, old) != current):
                raise ValueError('Every localized script needs its exact full source-bound review')
        declared = previous[identity].get('narration_voices', {})
        if (not isinstance(declared, dict) or not declared.get('en') or 'af_heart' not in declared['en']
                or any(not isinstance(voices, list) or not voices
                       or any(not isinstance(v, str) or not v for v in voices)
                       or len(voices) != len(set(voices)) for voices in declared.values())):
            raise ValueError('Expected exact original voice offerings')
        expected_files = {f'{language}/{voice}{suffix}' for language, voices in declared.items()
                          for voice in voices for suffix in ('.m4a', '.json')}
        root = stage / 'production' / identity / 'audio'
        actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        prefix = f'media_host/{identity}/audio/'
        if (actual_files != expected_files
                or {name[len(prefix):] for name in records if name.startswith(prefix)} != expected_files):
            raise ValueError('Staged or original tracks differ from the exact original offerings')
        legacy_visual_source(baseline, identity, previous[identity])
        for language, voices in declared.items():
            localized = next(row for row in catalogs[f'lessons_{language}.json']['lessons'] if row['id'] == identity)
            for voice in voices:
                relative = f'{language}/{voice}.m4a'
                audio, timing = root / relative, (root / relative).with_suffix('.json')
                require_timing(read(timing), localized, language, voice, digest(audio))
                planned.append(dict(lesson=identity, language=language, voice=voice, audio=str(audio),
                    timing=str(timing), audio_sha256=digest(audio), timing_sha256=digest(timing),
                    original_audio_sha256=records[prefix + relative]['sha256']))
    return catalogs, planned


def require_published_script_baseline(baseline, manifest):
    """Reuse visual bytes only from a complete normally verified publication."""
    from build_appended_candidate import require_baseline_receipt
    receipt = read(baseline / 'publication-receipt.json')
    require_baseline_receipt(manifest, receipt, digest(baseline / 'release-manifest.json'))
    checks = read(baseline / 'checks/published-media-browser-checks.json')
    pages = receipt.get('pages', {})
    cases = checks.get('ready_playback_cases', [])
    from archive_lessons import archived_view
    catalog = read(baseline / 'web/catalog/lessons_en.json')
    shown = archived_view({'catalog/lessons_en.json': catalog})['catalog/lessons_en.json']
    full_ready = {row['id'] for row in catalog['lessons'] if row.get('status') != 'coming_soon'}
    shown_ready = {row['id'] for row in shown['lessons'] if row.get('status') != 'coming_soon'}
    shown_ids = {row['id'] for row in shown['lessons']}
    # Accept the original complete publication or exactly the normal archive
    # projection, never an arbitrary passing subset or mixed count domains.
    domains = [(full_ready, manifest['ready_lessons'], manifest['routes'],
                len(manifest['hosted_web_lessons'])),
               (shown_ready, len(shown_ready), len(shown['lessons']),
                len(set(manifest['hosted_web_lessons']) & shown_ids))]
    complete = any(len(cases) == len(expected)
                   and {row.get('lesson') for row in cases} == expected
                   and (pages.get('ready'), pages.get('routes'), pages.get('hosted_web_copies'))
                       == (ready, routes, hosted)
                   for expected, ready, routes, hosted in domains)
    if (checks.get('passed') is not True or checks.get('manifest_sha256') != receipt['manifest_sha256']
            or checks.get('media_root') != receipt['media_root']
            or checks.get('index_sha256') != pages.get('index_sha256')
            or checks.get('placeholder_cases') != []
            or not complete
            or any(row.get('passed') is not True for row in cases)):
        raise ValueError('Legacy visual reuse requires the exact complete published browser/Pages evidence')


def verify_script_tracks(stage, identities, catalogs, planned, *, baseline=None):
    """Use the existing complete current-source decoder for every old offering."""
    from build_appended_candidate import verify_tracks
    for identity in identities:
        lesson = next(row for row in catalogs['lessons_en.json']['lessons'] if row['id'] == identity)
        mixed_record = stage / 'production' / identity / 'published-scene-audio-refresh.json'
        if mixed_record.exists() or mixed_record.is_symlink():
            if identity != '19_train_cellpose':
                raise ValueError('Published-scene audio refresh is reviewed only for Tutorial19')
            from published_scene_audio_refresh import checked_tracks
            declared, tracks = checked_tracks(stage, baseline, lesson, catalogs)
        else:
            declared, tracks = verify_tracks(stage, lesson, catalogs)
        expected = {(t['language'], t['voice'], t['audio_sha256'], t['timing_sha256'])
                    for t in planned if t['lesson'] == identity}
        actual = {(t['language'], t['voice'], t['audio_sha256'], t['timing_sha256']) for t in tracks}
        if (actual != expected or {k: set(v) for k, v in declared.items()} !=
                {k: set(v) for k, v in lesson['narration_voices'].items()}):
            raise ValueError('Normal fresh verification differs from original voice inventory')


def build(baseline, stage, identities, *, refresh_player=False, refresh_scripts=False,
          add_english_offerings=False):
    """Create a separate candidate without altering any baseline file."""
    baseline, stage = Path(baseline).resolve(), Path(stage).resolve()
    if add_english_offerings and (refresh_scripts or refresh_player):
        raise ValueError('Offering addition cannot refresh scripts or the player')
    validate(baseline, include_hosted_media=True)
    manifest = read(baseline / 'release-manifest.json')
    browser = read(baseline / 'checks/candidate-browser-checks.json')
    if (not browser.get('passed') or browser.get('manifest_sha256') != digest(baseline / 'release-manifest.json')
            or len(browser['ready_playback_cases']) != manifest['ready_lessons']
            or not all(row.get('passed') for row in browser['ready_playback_cases'])):
        raise ValueError('The complete baseline must have passing browser evidence')
    if add_english_offerings:
        require_published_script_baseline(baseline, manifest)
        catalogs, planned = offering_patch_plan(baseline, stage, identities)
        verify_current_tracks(baseline, planned)
    elif refresh_scripts:
        require_published_script_baseline(baseline, manifest)
        catalogs, planned = script_patch_plan(baseline, stage, identities)
        if any((stage / 'production' / identity / 'published-scene-audio-refresh.json').exists() or (stage / 'production' / identity / 'published-scene-audio-refresh.json').is_symlink() for identity in identities):
            verify_script_tracks(stage, identities, catalogs, planned, baseline=baseline)
        else:
            verify_script_tracks(stage, identities, catalogs, planned)
    else:
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
        prefix = f"media_host/{track['lesson']}/audio/{track.get('language', 'en')}/{track['voice']}"
        replacements[prefix + '.m4a'] = (Path(track['audio']), track['audio_sha256'])
        replacements[prefix + '.json'] = (Path(track['timing']), track['timing_sha256'])
    if refresh_scripts:
        for name in CATALOGS:
            replacements['web/catalog/' + name] = (stage / 'catalog' / name, digest(stage / 'catalog' / name))
    records = []
    for record in manifest['files']:
        source, expected = replacements.get(record['path'], (baseline / record['path'], record['sha256']))
        copy_checked(source, target / record['path'], records, target, expected)
    if add_english_offerings:
        for relative, (source, expected) in replacements.items():
            if relative not in {row['path'] for row in manifest['files']}:
                copy_checked(source, target / relative, records, target, expected)
        for name, catalog in catalogs.items():
            path = target / 'web/catalog' / name
            write(path, catalog)
            records = [row for row in records if row['path'] != 'web/catalog/' + name]
            records.append(dict(path='web/catalog/' + name, sha256=digest(path), bytes=path.stat().st_size))
    javascript = parse_javascript((baseline / 'web/lesson_catalog.js').read_text(encoding="utf-8"))
    timing_references = []
    for lesson in javascript['lessons']:
        if lesson['id'] not in identities:
            continue
        identity = lesson['id']
        if add_english_offerings:
            lesson['narration_voices'] = deepcopy(next(row for row in catalogs['lessons_en.json']['lessons']
                                                       if row['id'] == identity)['narration_voices'])
            # Heart is retained byte-exact, so even an implicit visual clock is unchanged.
            continue
        if refresh_scripts:
            replacement = next(row for row in catalogs['lessons_en.json']['lessons'] if row['id'] == identity)
            for key in ('title', 'description', 'objectives', 'prerequisite', 'scenes'):
                if key in replacement:
                    lesson[key] = deepcopy(replacement[key])
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
                            + json.dumps(javascript, ensure_ascii=False) + ');\n', encoding="utf-8")
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
    if refresh_scripts:
        report.update(scope='Private reviewed script refresh; exact published visual media and voice offerings preserved',
            script_refresh=True, fresh_capture=False, raw_capture_revalidated=False)
    if add_english_offerings:
        report.update(scope='Add23 English offerings; original27 narration tracks and all visual media preserved',
                      added_english_offerings=list(identities),
                      narration_tracks=sum(row['path'].startswith('media_host/') and row['path'].endswith('.m4a')
                                           for row in records))
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
    parser.add_argument('--refresh-scripts', action='store_true',
                        help='Refresh reviewed narration only; retain exact existing non-native visuals and voice offerings')
    parser.add_argument('--add-english-offerings', action='store_true',
                        help='Add the missing23 English voices to unchanged Tutorial06 or48; retain all original tracks')
    args = parser.parse_args()
    build(args.baseline, args.stage, args.lesson, refresh_player=args.refresh_player,
          refresh_scripts=args.refresh_scripts, add_english_offerings=args.add_english_offerings)


if __name__ == '__main__':
    main()
