#!/usr/bin/env python3
"""Assemble a private, byte-checked tutorial candidate; never deploy or upload.

Ready lessons keep their exact scripts/media. Explicitly unavailable routes
are text screens, not successful workflow demonstrations.
All copies live in a new directory. Existing catalogs/recordings stay intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import tempfile

from audit_staged_catalogs import CATALOGS
from build_navigation import build as navigation
from check_completed_matrix import digest, voice_matrix
from coming_soon import EMBEDDINGS, OPS, HELD, release_catalog
from stage_lesson import DEFAULT_STAGE, REPO, read, write
from verify_library_checkpoint import verify


def require_web_receipt(identity, receipt, browser, actual_hash):
    """A passing report must describe this exact lesson and current web bytes."""
    if (receipt.get('lesson') != identity or receipt.get('accepted') is not True
            or receipt.get('rendition_sha256') != actual_hash
            or receipt.get('all_frame_presentation_times_match') is not True
            or receipt.get('full_decode_passed') is not True
            or browser.get('lesson') != identity or browser.get('passed') is not True
            or browser.get('checked_web_rendition', {}).get('sha256') != actual_hash):
        raise ValueError(f'Stale or failed web-copy evidence: {identity}')


def copy_checked(source, target, records, root, expected=None):
    actual = digest(source)
    if expected is not None and actual != expected:
        raise ValueError(f'Source changed after verification: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if digest(target) != actual:
        raise ValueError(f'Copy differs from source: {target}')
    records.append({'path': target.relative_to(root).as_posix(),
                    'sha256': actual, 'bytes': target.stat().st_size})


def write_catalogs(stage, web):
    """Derive parent metadata from today's GUI without altering lesson scenes."""
    catalogs = {}
    for filename in CATALOGS:
        language = filename.split('_', 1)[1].removesuffix('.json')
        catalogs[filename] = release_catalog(read(stage / 'catalog' / filename), language,
                                             recording_stage=stage)
    nav = navigation(catalogs['lessons_en.json'])
    if nav['missing_tutorials']:
        raise ValueError(f"Unaccounted tutorial routes: {nav['missing_tutorials']}")
    english_by_id = {lesson['id']: deepcopy(lesson) for lesson in catalogs['lessons_en.json']['lessons']}
    for filename, catalog in catalogs.items():
        for lesson in catalog['lessons']:
            # Historical held translations predate app_key metadata. Routing
            # is language-independent; do not copy their old omissions.
            canonical = english_by_id[lesson['id']]
            for field in ('number', 'app_key', 'host_app_key', 'series', 'section', 'slug'):
                if field in canonical:
                    lesson[field] = canonical[field]
                else:
                    lesson.pop(field, None)
            route = nav['routes'].get(lesson['id'], {})
            if route.get('kind') == 'submodule':
                lesson['host_app_key'] = route['host_app_key']
            elif route.get('kind') == 'main':
                lesson.pop('host_app_key', None)
        write(web / 'catalog' / filename, catalog)
    js_catalog = deepcopy(catalogs['lessons_en.json'])
    for lesson in js_catalog['lessons']:
        if lesson.get('status') != 'coming_soon':
            identity = lesson['id']
            lesson['poster'] = f'{identity}/poster.jpg'
            lesson['silent'] = f'{identity}/video/{identity}_silent.mp4'
    for name, variable, data in [('lesson_catalog.js', 'SPACR_LESSON_CATALOG', js_catalog),
                                 ('module_navigation.js', 'SPACR_TUTORIAL_NAVIGATION', nav)]:
        (web / name).write_text('"use strict";\nwindow.' + variable + ' = Object.freeze('
                               + json.dumps(data, ensure_ascii=False) + ');\n')


def refresh_candidate_player(root):
    """Update candidate player/routing before checks, preserving all media."""
    root = Path(root).resolve()
    report = read(root / 'release-manifest.json')
    if report.get('release_hold') is not True or report.get('published') is not False:
        raise ValueError('Only an unpublished, held candidate may be refreshed')
    existing = read(root / 'web/catalog/lessons_en.json')['lessons']
    proposed = release_catalog(read(root.parent / 'catalog/lessons_en.json'), 'en')['lessons']
    def disposition(lessons):
        return [(lesson['id'], lesson.get('status') == 'coming_soon') for lesson in lessons]
    if disposition(existing) != disposition(proposed):
        raise ValueError('A newly recorded route needs a new candidate, not a player-only refresh')
    names = ('app_v2.js', 'styles.css')
    records = report['files']
    for name in names:
        previous = next(r for r in records if r['path'] == 'web/' + name)
        if digest(root / 'web' / name) != previous['sha256']:
            raise ValueError(f'Unrecorded candidate modification: {name}')
        records.remove(previous)
        copy_checked(root.parent.parent / 'web' / name, root / 'web' / name, records, root)
    catalog_paths = {'web/catalog/' + filename for filename in CATALOGS}
    catalog_paths.update({'web/lesson_catalog.js', 'web/module_navigation.js'})
    for record in records:
        if record['path'] in catalog_paths and digest(root / record['path']) != record['sha256']:
            raise ValueError('Unrecorded candidate catalog modification')
    write_catalogs(root.parent, root / 'web')
    records[:] = [r for r in records if r['path'] not in catalog_paths]
    records.extend({'path': path, 'sha256': digest(root / path), 'bytes': (root / path).stat().st_size}
                   for path in sorted(catalog_paths))
    report['files'] = sorted(records, key=lambda r: r['path'])
    report['web_bytes'] = sum(r['bytes'] for r in records if r['path'].startswith('web/'))
    write(root / 'release-manifest.json', report)


def build(stage=DEFAULT_STAGE, *, baseline=None):
    stage = Path(stage).resolve()
    # Revalidate current sources before creating any release copies.
    proof = verify(stage, set(HELD), baseline=baseline)
    ready_ids = {item['lesson'] for item in proof['lessons']}
    expected_ready = 71 + (EMBEDDINGS in ready_ids) + (OPS in ready_ids)
    if proof['checked_lessons'] != expected_ready or proof['checked_tracks'] != expected_ready * 50:
        raise ValueError('The approved ready/tutorial partition changed')
    root = Path(tempfile.mkdtemp(prefix='release-candidate-', dir=stage))
    web, media = root / 'web', root / 'media_host'
    records = []
    source_web = stage.parent / 'web'
    published = REPO / 'docs/source/_extra/tutorials'
    web.mkdir()
    for name in ('index.html', 'app_v2.js', 'styles.css'):
        copy_checked(source_web / name, web / name, records, root)
    for name in ('voice_catalog.js', 'logo_spacr.png', 'favicon.svg', 'TUTORIAL_MEDIA_NOTICE.md'):
        copy_checked(published / name, web / name, records, root)
    for name in ('fonts', 'examples'):
        for source in sorted((published / name).rglob('*')):
            if source.is_file():
                copy_checked(source, web / source.relative_to(published), records, root)
    # Relative roots make this an offline preview. Deployment must explicitly
    # replace them with an immutable uploaded media revision after the hold.
    index = (web / 'index.html').read_text()
    for attr, value in [('production-root', 'production'), ('audio-root', '../media_host'),
                        ('video4k-root', '../media_host')]:
        index = re.sub(rf'data-{attr}="[^"]*"', f'data-{attr}="{value}"', index)
    (web / 'index.html').write_text(index)
    records[:] = [r for r in records if r['path'] != 'web/index.html']

    write_catalogs(stage, web)
    voices = voice_matrix(stage.parent / 'tools/render_all_voices.py')
    web_checks = []
    for result in proof['lessons']:
        identity = result['lesson']
        source = (stage.parent if identity == '33_plate_viewer' else stage) / 'production' / identity
        rendition = stage / 'web-renditions' / identity
        video = rendition / 'video' / f'{identity}_silent.mp4'
        receipt = read(rendition / 'rendition-checks.json')
        browser_path = stage / 'browser-web' / identity / 'en-af_heart/playback-checks.json'
        require_web_receipt(identity, receipt, read(browser_path), digest(video))
        copy_checked(video, web / 'production' / identity / 'video' / video.name, records, root,
                     receipt['rendition_sha256'])
        copy_checked(source / 'poster.jpg', web / 'production' / identity / 'poster.jpg', records, root)
        copy_checked(source / 'video' / video.name, media / identity / 'video' / video.name,
                     records, root, receipt['master_sha256'])
        for language, names in voices.items():
            for voice in names:
                for suffix in ('.m4a', '.json'):
                    relative = Path('audio') / language / (voice + suffix)
                    copy_checked(source / relative, media / identity / relative, records, root)
        web_checks.append({'lesson': identity, 'rendition_sha256': receipt['rendition_sha256'],
                           'browser_report_sha256': digest(browser_path)})
        print(identity, 'COPIED', flush=True)
    write(root / 'checks/library-source-checks.json', proof)
    recorded = {r['path'] for r in records}
    for path in sorted(web.rglob('*')):
        if path.is_file() and path.relative_to(root).as_posix() not in recorded:
            records.append({'path': path.relative_to(root).as_posix(), 'sha256': digest(path),
                            'bytes': path.stat().st_size})
    web_bytes = sum(r['bytes'] for r in records if r['path'].startswith('web/'))
    if web_bytes > 700 * 1024**2:
        raise ValueError('Candidate exceeds the tutorial media budget')
    unavailable = [item['id'] for item in read(web / 'catalog/lessons_en.json')['lessons']
                   if item.get('status') == 'coming_soon']
    report = {'scope': 'Private release candidate, not a live deployment',
              'ready_lessons': len(proof['lessons']), 'coming_soon': unavailable,
              'routes': len(read(web / 'catalog/lessons_en.json')['lessons']),
              'catalog_languages': len(CATALOGS), 'narration_tracks': proof['checked_tracks'],
              'web_bytes': web_bytes, 'ceiling_bytes': 700 * 1024**2,
              'media_host_bytes': sum(r['bytes'] for r in records if r['path'].startswith('media_host/')),
              'files': sorted(records, key=lambda r: r['path']), 'web_checks': web_checks,
              'all_workflows_demonstrated': False, 'native_speaker_signoff': False,
              'human_listening_signoff': False, 'release_hold': True,
              'uploaded': False, 'published': False}
    write(root / 'release-manifest.json', report)
    print('CANDIDATE', root, flush=True)
    return root, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--baseline', type=Path,
                        help='Frozen pre-refresh catalogs, before public Coming soon conversion')
    args = parser.parse_args()
    build(args.stage, baseline=args.baseline)
