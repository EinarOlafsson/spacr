#!/usr/bin/env python3
"""Verify candidate inventory and bytes, not merely that a report exists."""
from pathlib import Path

from append_staged_lessons import parse_javascript
from audit_staged_catalogs import CATALOGS
from check_completed_matrix import digest
from stage_lesson import read


def hosted_web_path(identity):
    """Where a hosted web copy lives on the media revision, beside the 4K master."""
    return f'{identity}/web/{identity}_silent.mp4'


def require_consistent_web_hosting(records, lessons):
    """Each hosted lesson has exactly one web copy: on the media host, not on Pages.

    ``lessons`` are the lesson_catalog.js objects; a hosted lesson names its
    copy in ``web``. Returns the hosted lesson identities.
    """
    paths = {record['path'] for record in records}
    hosted_files = {path for path in paths if path.startswith('media_host/')
                    and len(Path(path).parts) == 4 and Path(path).parts[2] == 'web'}
    flagged = set()
    for lesson in lessons:
        if 'web' not in lesson:
            continue
        identity = lesson['id']
        expected = hosted_web_path(identity)
        if (lesson['web'] != expected or 'media_host/' + expected not in paths
                or f'web/production/{identity}/video/{identity}_silent.mp4' in paths):
            raise ValueError(f'Hosted web copy is missing, misnamed or duplicated on Pages: {identity}')
        flagged.add('media_host/' + expected)
    if hosted_files != flagged:
        raise ValueError(f'Hosted web copies without a catalog entry: {sorted(hosted_files - flagged)}')
    return sorted(Path(path).parts[1] for path in flagged)


def validate(root, *, include_hosted_media=False, require_browser=False):
    root = Path(root)
    manifest_path = root / 'release-manifest.json'
    manifest = read(manifest_path)
    lessons = read(root / 'web/catalog/lessons_en.json')['lessons']
    ready = [x['id'] for x in lessons if x.get('status') != 'coming_soon']
    held = [x['id'] for x in lessons if x.get('status') == 'coming_soon']
    if (len({x['id'] for x in lessons}) != len(lessons)
            or manifest['routes'] != len(lessons)
            or manifest['ready_lessons'] != len(ready)
            or manifest['coming_soon'] != held
            or manifest['catalog_languages'] != len(CATALOGS)):
        raise ValueError('Manifest inventory differs from the actual lesson catalogs')
    def structure(items):
        return [(x['id'], x.get('app_key'), x.get('host_app_key'), x.get('status'),
                 len(x['scenes'])) for x in items]

    for filename in CATALOGS:
        if structure(read(root / 'web/catalog' / filename)['lessons']) != structure(lessons):
            raise ValueError(f'Catalog structure differs from English: {filename}')
    seen, checked = set(), 0
    for record in manifest['files']:
        relative = Path(record['path'])
        if (relative.is_absolute() or '..' in relative.parts or not relative.parts
                or relative.parts[0] not in {'web', 'media_host'}
                or relative.as_posix() != record['path'] or record['path'] in seen):
            raise ValueError('Unsafe or duplicate manifest path')
        seen.add(record['path'])
        if relative.parts[0] != 'web' and not include_hosted_media:
            continue
        path = root / relative
        if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError(f'Candidate file differs from its recorded bytes: {relative}')
        checked += 1
    # Without a lesson_catalog.js nothing may claim a hosted web copy.
    player_catalog = root / 'web/lesson_catalog.js'
    require_consistent_web_hosting(manifest['files'], parse_javascript(
        player_catalog.read_text())['lessons'] if player_catalog.is_file() else [])
    actual_web = {p.relative_to(root).as_posix() for p in (root / 'web').rglob('*') if p.is_file()}
    if actual_web != {p for p in seen if p.startswith('web/')}:
        raise ValueError('Unrecorded or missing website files')
    if include_hosted_media:
        actual_media = {p.relative_to(root).as_posix() for p in (root / 'media_host').rglob('*') if p.is_file()}
        if actual_media != {p for p in seen if p.startswith('media_host/')}:
            raise ValueError('Unrecorded or missing hosted media files')
    if require_browser:
        report_path = root / 'checks/candidate-browser-checks.json'
        if not report_path.exists():
            report_path = root / 'candidate-browser-checks.json'
        report = read(report_path)
        if report.get('passed') is not True or report.get('manifest_sha256') != digest(manifest_path):
            raise ValueError('Browser evidence does not describe this exact manifest')
        playback = report['ready_playback_cases']
        screens = report['placeholder_cases']
        languages = [f.split('_', 1)[1].removesuffix('.json') for f in CATALOGS]
        if (len(playback) != len(ready) or {x['lesson'] for x in playback} != set(ready)
                or not all(x.get('passed') is True for x in playback)
                or len(screens) != len(held) * len(languages)
                or {(x['lesson'], x['language']) for x in screens} != {(h, l) for h in held for l in languages}
                or not all(x.get('passed') is True for x in screens)):
            raise ValueError('Browser evidence omits or duplicates current lessons/languages')
        mutations_path = root / 'checks/placeholder-mutation-checks.json'
        if not mutations_path.exists():
            mutations_path = root / 'placeholder-mutation-checks.json'
        mutations = read(mutations_path)
        if (mutations.get('passed') is not True
                or mutations.get('baseline_before_and_after_passed') is not True
                or mutations.get('player_sha256') != digest(root / 'web/app_v2.js')
                or len(mutations.get('mutations', [])) != 2
                or {x['guard'] for x in mutations['mutations']} != {'availability guard', 'completion guard'}
                or not all(x.get('observed_red') is True for x in mutations['mutations'])):
            raise ValueError('Mutation evidence does not verify this exact player')
    return {'routes': len(lessons), 'ready': len(ready), 'coming_soon': len(held),
            'checked_files': checked, 'manifest_sha256': digest(manifest_path)}
