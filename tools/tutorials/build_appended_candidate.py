"""Publish individual verified lessons without waiting for every voice.

Existing lesson objects and hosted media are preserved from a verified release.
New lessons expose only audio that passes current-source checks. Missing or
incompatible translations are registered and use English. Upload, readback,
candidate playback and publication still use the regular release tools.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import tempfile

from apply_translation_review import translated_lesson
from append_staged_lessons import parse_javascript
from audit_staged_catalogs import CATALOGS
from build_navigation import build as navigation
from build_release_candidate import copy_checked, require_web_receipt
from check_completed_matrix import digest
from stage_lesson import REPO, read, write
from validate_candidate import validate


def append_catalogs(published, lessons, voices, reviews, *, replace=False):
    """Preserve published objects and bind each added translation to its source."""
    existing = published['lessons_en.json']['lessons']
    identities = {item['id'] for item in existing}
    numbers = list(range(len(existing) + 1, len(existing) + len(lessons) + 1))
    if not lessons or len({item['id'] for item in lessons}) != len(lessons):
        raise ValueError('Select at least one unique lesson')
    positions = {item['id']: index for index, item in enumerate(existing)}
    if replace:
        for item in lessons:
            if item['id'] not in positions or any(
                    item.get(key) != existing[positions[item['id']]].get(key)
                    for key in ('number', 'app_key', 'host_app_key')):
                raise ValueError('A refresh must preserve the existing lesson identity and route')
    elif ([item['number'] for item in lessons] != numbers
            or any(item['id'] in identities for item in lessons)):
        raise ValueError('Append new, unique lessons in contiguous number order')
    for item in lessons:
        identity = item['id']
        if (not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity)
                or int(identity.split('_', 1)[0]) != item['number']
                or not item.get('scenes') or item.get('status') == 'coming_soon'
                or 'af_heart' not in voices.get(identity, {}).get('en', [])):
            raise ValueError('Every appended lesson needs scenes and verified English narration')
    catalogs, compatibility = deepcopy(published), []
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        if [row['id'] for row in published[filename]['lessons']] != [row['id'] for row in existing]:
            raise ValueError('Published language catalogs disagree on lesson order')
        for english in lessons:
            identity = english['id']
            translated = deepcopy(english)
            if language != 'en':
                review = reviews.get((identity, language))
                try:
                    if not review or review.get('lesson') != identity or review.get('language') != language:
                        raise ValueError('Missing or mismatched translation review')
                    translated = translated_lesson(review, english, {})
                    status, reason = 'source_bound_review', None
                except (ValueError, KeyError, TypeError) as error:
                    status, reason = 'english_fallback', str(error)
                compatibility.append(dict(lesson=identity, language=language, status=status, reason=reason))
            translated['narration_voices'] = deepcopy(voices[identity])
            if replace:
                catalogs[filename]['lessons'][positions[identity]] = translated
            else:
                catalogs[filename]['lessons'].append(translated)
    return catalogs, compatibility


def verify_tracks(stage, lesson, catalogs):
    """Decode and check current synthesis inputs for every offered audio track."""
    sys.path.insert(0, str(REPO / 'tools/tutorials/authoring/tools'))
    import verify_audio_release as audio

    identity = lesson['id']
    declared, records = {}, []
    for path in sorted((stage / 'production' / identity / 'audio').glob('*/*.m4a')):
        language, voice = path.parent.name, path.stem
        if language not in audio.LANGUAGES or voice not in audio.LANGUAGES[language][1]:
            raise ValueError(f'Unsupported narration track: {path}')
        localized = next(row for row in catalogs[f'lessons_{language}.json']['lessons'] if row['id'] == identity)
        code = 'b' if language == 'en' and voice.startswith('b') else audio.LANGUAGES[language][0]
        dialect = audio.narration_dialect(language, code, voice)
        speed = audio.resolve_voice_speed(voice)
        plans = audio.prepare_scene_plans(localized, language, dialect, speed, voice=voice)
        errors = audio.check_track(audio.TrackSpec(path, localized, language, code, voice, dialect, speed, plans))
        if errors:
            raise ValueError(f'{identity}/{language}/{voice}: {errors}')
        declared.setdefault(language, []).append(voice)
        records.append(dict(language=language, voice=voice, audio_sha256=digest(path),
                            timing_sha256=digest(path.with_suffix('.json'))))
    if 'af_heart' not in declared.get('en', []):
        raise ValueError('New lessons require verified English Heart narration')
    return declared, records


def append_javascript_catalog(published, english, count, *, replacements=()):
    """Retain historical JavaScript objects independently of JSON catalogs."""
    previous = published['lessons']
    baseline = english['lessons'] if replacements else english['lessons'][:-count]
    if [row['id'] for row in previous] != [row['id'] for row in baseline]:
        raise ValueError('JavaScript and JSON baseline lesson identities differ')
    if replacements and (len(replacements) != len(set(replacements))
                         or set(replacements) - {row['id'] for row in previous}):
        raise ValueError('Refresh only unique existing JavaScript lesson identities')
    result = deepcopy(published)
    selected = [row for row in english['lessons'] if row['id'] in replacements] if replacements else english['lessons'][-count:]
    for original in selected:
        lesson = deepcopy(original)
        lesson['poster'] = f'{lesson["id"]}/poster.jpg'
        lesson['silent'] = f'{lesson["id"]}/video/{lesson["id"]}_silent.mp4'
        if replacements:
            position = next(i for i, row in enumerate(previous) if row['id'] == lesson['id'])
            result['lessons'][position] = lesson
        else:
            result['lessons'].append(lesson)
    return result


def require_baseline_receipt(manifest, receipt, manifest_hash):
    """Require readback coverage of the exact immutable baseline media set."""
    count = sum(record['path'].startswith('media_host/') for record in manifest['files'])
    checked = receipt.get('readback', {})
    commit = receipt.get('commit', '')
    if (not count or receipt.get('manifest_sha256') != manifest_hash
            or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or receipt.get('media_root') != 'https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/' + commit
            or not receipt.get('tag') or receipt.get('media_files') != count
            or checked.get('commit') != commit or checked.get('passed') is not True
            or any(checked.get(key) != count for key in
                   ('files_expected', 'metadata_matched', 'downloaded_sha256_matched'))
            or checked.get('metadata_failures') != [] or checked.get('download_failures') != []):
        raise ValueError('Baseline readback does not cover its exact immutable media set')


def require_no_new_route_gaps(before, after):
    """Keep existing tutorial debt visible while rejecting newly lost routes."""
    old = {item['app_key'] for item in before['missing_tutorials']}
    new = {item['app_key'] for item in after['missing_tutorials']}
    if not new <= old:
        raise ValueError('Appending lessons introduced missing module routes')


def copy_preserved_web(published, baseline, root, manifest, *, replacements=()):
    """Retain verified media; ignore leftover production files in the Pages tree."""
    records = []
    baseline_web = {record['path'][len('web/'):]: record for record in manifest['files']
                    if record['path'].startswith('web/')}
    for path in sorted(published.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(published)
        if relative.parts[0] != 'production':
            copy_checked(path, root / 'web' / relative, records, root)
    for name, record in sorted(baseline_web.items()):
        relative = Path(name)
        if relative.parts[0] != 'production' or relative.parts[1] in replacements:
            continue
        copy_checked(baseline / 'web' / relative, root / 'web' / relative,
                     records, root, record['sha256'])
    return records


def build(stage, baseline, identities, *, replace=False):
    """Create a new private candidate; never upload or modify the published tree."""
    stage, baseline = Path(stage).resolve(), Path(baseline).resolve()
    validate(baseline, include_hosted_media=True, require_browser=True)
    previous = read(baseline / 'release-manifest.json')
    receipt = read(baseline / 'publication-receipt.json')
    require_baseline_receipt(previous, receipt, digest(baseline / 'release-manifest.json'))
    published = REPO / 'docs/source/_extra/tutorials'
    index = (published / 'index.html').read_text()
    roots = re.findall(r'data-(?:audio|video4k)-root="([^"]+)"', index)
    if len(roots) != 2 or set(roots) != {receipt['media_root']}:
        raise ValueError('Baseline media revision is not the currently published revision')
    catalogs = {name: read(published / 'catalog' / name) for name in CATALOGS}
    previous_navigation = navigation(catalogs['lessons_en.json'])
    for name in CATALOGS:
        if catalogs[name] != read(baseline / 'web/catalog' / name):
            raise ValueError('Published lesson sources differ from the verified media baseline')
    if (len(identities) != len(set(identities))
            or any(not re.fullmatch(r'[0-9]+_[a-z0-9_]+', identity) for identity in identities)):
        raise ValueError('Duplicate or invalid appended lesson identity')
    lessons = [read(REPO / 'tools/tutorials/lessons' / (identity + '.json')) for identity in identities]
    reviews = {}
    for lesson in lessons:
        if read(stage / 'production' / lesson['id'] / 'lesson.en.json') != lesson:
            raise ValueError('Staged English differs from the current canonical lesson')
        for filename in CATALOGS:
            language = filename.split('_', 1)[1].removesuffix('.json')
            path = REPO / 'tools/tutorials/lessons/reviews' / f'{lesson["id"]}.{language}.json'
            if path.exists():
                reviews[lesson['id'], language] = read(path)
    planned = {lesson['id']: {'en': ['af_heart']} for lesson in lessons}
    provisional, _ = append_catalogs(catalogs, lessons, planned, reviews, replace=replace)
    checks, voices = {}, {}
    for lesson in lessons:
        identity = lesson['id']
        voices[identity], checks[identity] = verify_tracks(stage, lesson, provisional)
        print(identity, len(checks[identity]), 'current audio tracks verified', flush=True)
    catalogs, compatibility = append_catalogs(catalogs, lessons, voices, reviews, replace=replace)
    root = Path(tempfile.mkdtemp(prefix='release-candidate-append-', dir=stage))
    records = copy_preserved_web(published, baseline, root, previous,
                                 replacements=identities if replace else ())
    web_checks = []
    for record in previous['files']:
        if record['path'].startswith('media_host/'):
            if replace and Path(record['path']).parts[1] in identities:
                continue
            copy_checked(baseline / record['path'], root / record['path'], records, root, record['sha256'])
    for lesson in lessons:
        identity = lesson['id']
        source = stage / 'production' / identity
        rendition = stage / 'web-renditions' / identity
        video = rendition / 'video' / f'{identity}_silent.mp4'
        proof = read(rendition / 'rendition-checks.json')
        browser_path = stage / 'browser-web' / identity / 'en-af_heart/playback-checks.json'
        require_web_receipt(identity, proof, read(browser_path), digest(video))
        copy_checked(source / 'video' / video.name, root / 'media_host' / identity / 'video' / video.name,
                     records, root, proof['master_sha256'])
        copy_checked(video, root / 'web/production' / identity / 'video' / video.name,
                     records, root, proof['rendition_sha256'])
        copy_checked(source / 'poster.jpg', root / 'web/production' / identity / 'poster.jpg', records, root)
        for track in checks[identity]:
            for suffix, field in (('.m4a', 'audio_sha256'), ('.json', 'timing_sha256')):
                relative = Path('audio') / track['language'] / (track['voice'] + suffix)
                copy_checked(source / relative, root / 'media_host' / identity / relative,
                             records, root, track[field])
        web_checks.append(dict(lesson=identity, rendition_sha256=proof['rendition_sha256'],
                               browser_report_sha256=digest(browser_path)))
    for name, catalog in catalogs.items():
        write(root / 'web/catalog' / name, catalog)
    js_catalog = append_javascript_catalog(
        parse_javascript((published / 'lesson_catalog.js').read_text()),
        catalogs['lessons_en.json'], len(lessons), replacements=identities if replace else ())
    nav = navigation(catalogs['lessons_en.json'])
    require_no_new_route_gaps(previous_navigation, nav)
    for name, variable, data in [('lesson_catalog.js', 'SPACR_LESSON_CATALOG', js_catalog),
                                 ('module_navigation.js', 'SPACR_TUTORIAL_NAVIGATION', nav)]:
        (root / 'web' / name).write_text('"use strict";\nwindow.' + variable + ' = Object.freeze('
                                       + json.dumps(data, ensure_ascii=False) + ');\n')
    (root / 'web/app_v2.js').write_bytes((REPO / 'tools/tutorials/authoring/web/app_v2.js').read_bytes())
    for attribute in ('audio', 'video4k'):
        index, count = re.subn(rf'data-{attribute}-root="[^"]+"', f'data-{attribute}-root="../media_host"', index)
        if count != 1:
            raise ValueError('Expected one media root per type')
    (root / 'web/index.html').write_text(index)
    write(root / 'web/translation-compatibility.json', dict(policy='report-only; English fallback', entries=compatibility))
    records = [record for record in records if record['path'].startswith('media_host/')]
    records.extend(dict(path=path.relative_to(root).as_posix(), sha256=digest(path), bytes=path.stat().st_size)
                   for path in sorted((root / 'web').rglob('*')) if path.is_file())
    web_bytes = sum(record['bytes'] for record in records if record['path'].startswith('web/'))
    if web_bytes > previous['ceiling_bytes']:
        raise ValueError('Candidate exceeds the existing web media budget')
    english = catalogs['lessons_en.json']['lessons']
    held = [lesson['id'] for lesson in english if lesson.get('status') == 'coming_soon']
    report = dict(scope='Private incremental release; incomplete voices and translations remain tracked',
                  ready_lessons=len(english)-len(held), coming_soon=held, routes=len(english),
                  catalog_languages=len(CATALOGS), narration_tracks=sum(
                      r['path'].startswith('media_host/') and r['path'].endswith('.m4a') for r in records),
                  web_bytes=web_bytes, ceiling_bytes=previous['ceiling_bytes'],
                  media_host_bytes=sum(r['bytes'] for r in records if r['path'].startswith('media_host/')),
                  files=sorted(records, key=lambda record: record['path']), web_checks=web_checks,
                  baseline_manifest_sha256=digest(baseline / 'release-manifest.json'),
                  preserved_lessons=len(english)-len(lessons),
                  appended_lessons=[] if replace else identities,
                  refreshed_lessons=identities if replace else [],
                  outstanding_module_tutorials=nav['missing_tutorials'],
                  new_lesson_tracks=checks, translation_incompatibilities=compatibility,
                  all_workflows_demonstrated=False, native_speaker_signoff=False,
                  human_listening_signoff=False, release_hold=True, uploaded=False, published=False)
    write(root / 'release-manifest.json', report)
    validate(root, include_hosted_media=True)
    print('CANDIDATE', root, flush=True)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--lesson', action='append', required=True)
    parser.add_argument('--replace-existing', action='store_true',
                        help='Refresh only the selected existing lessons; keep all other lesson media and prose')
    args = parser.parse_args()
    build(args.stage, args.baseline, args.lesson, replace=args.replace_existing)
