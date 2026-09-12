#!/usr/bin/env python3
"""Checkpoint candidate text and byte identities, not gigabytes of media.

This does not modify the published docs tree. The release candidate and its
original authoring inputs/media remain on disk; Git protects the exact player,
all localized scripts, route inventory, checks and every media fingerprint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from build_release_candidate import copy_checked
from check_completed_matrix import digest
from stage_lesson import REPO, read, write
from validate_candidate import validate


def checkpoint(root, *, include_web_media=False):
    root = Path(root).resolve()
    manifest = read(root / 'release-manifest.json')
    if manifest.get('release_hold') is not True or manifest.get('published') is not False:
        raise ValueError('Expected an unpublished release candidate')
    # Validate everything before overwriting any checkpoint file. Report
    # existence alone is not evidence that all current routes were exercised.
    validate(root, include_hosted_media=True, require_browser=True)
    target = REPO / 'tools/tutorials/release_candidate'
    previous = target / 'checkpoint.json'
    include_web_media = include_web_media or (previous.exists() and read(previous).get('web_media_in_git', False))
    selected = {'index.html', 'app_v2.js', 'styles.css', 'voice_catalog.js',
                'module_navigation.js', 'lesson_catalog.js', 'TUTORIAL_MEDIA_NOTICE.md', 'favicon.svg'}
    copied = []
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.parts[0] != 'web':
            continue
        if include_web_media or relative.parts[1] == 'catalog' or relative.name in selected:
            copy_checked(root / relative, target / relative, copied, target, record['sha256'])
    copy_checked(root / 'release-manifest.json', target / 'release-manifest.json', copied, target)
    checks = root / 'checks/candidate-browser-checks.json'
    if checks.exists():
        report = read(checks)
        if report['manifest_sha256'] != digest(root / 'release-manifest.json'):
            raise ValueError('Browser checks describe a different candidate')
        copy_checked(checks, target / 'candidate-browser-checks.json', copied, target)
    source_checks = root / 'checks/library-source-checks.json'
    if source_checks.exists():
        # Keep the immutable pre-refresh baseline identity alongside the
        # final browser checks; today's public catalogs are not that baseline.
        proof = read(source_checks)
        write(target / 'source-verification-summary.json', {
            key: proof[key] for key in ('scope', 'checked_lessons', 'checked_tracks',
                'browser_cases', 'held_lesson_ids', 'checked_subset_passed',
                'preservation_baseline', 'whole_library_complete')})
        summary = target / 'source-verification-summary.json'
        copied.append({'path': summary.name, 'sha256': digest(summary), 'bytes': summary.stat().st_size})
    mutations = root / 'checks/placeholder-mutation-checks.json'
    if mutations.exists():
        report = read(mutations)
        if report.get('passed') is not True or report.get('player_sha256') != digest(root / 'web/app_v2.js'):
            raise ValueError('Mutation checks describe a different player')
        copy_checked(mutations, target / 'placeholder-mutation-checks.json', copied, target)
    write(target / 'checkpoint.json', {'private_candidate': str(root),
          'manifest_sha256': digest(root / 'release-manifest.json'), 'files': copied,
          'web_media_in_git': bool(include_web_media), 'narration_and_4k_in_git': False,
          'browser_checks_complete': checks.exists(), 'published': False, 'release_hold': True})
    print('Checkpointed', len(copied), 'candidate files at', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--include-web-media', action='store_true',
                        help='Also checkpoint the web videos, posters, fonts and examples; not narration/4K')
    args = parser.parse_args()
    checkpoint(args.candidate, include_web_media=args.include_web_media)
