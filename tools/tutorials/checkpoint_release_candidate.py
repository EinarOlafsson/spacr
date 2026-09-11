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


def checkpoint(root):
    root = Path(root).resolve()
    manifest = read(root / 'release-manifest.json')
    if manifest.get('release_hold') is not True or manifest.get('published') is not False:
        raise ValueError('Expected an unpublished release candidate')
    target = REPO / 'tools/tutorials/release_candidate'
    selected = {'index.html', 'app_v2.js', 'styles.css', 'voice_catalog.js',
                'module_navigation.js', 'lesson_catalog.js', 'TUTORIAL_MEDIA_NOTICE.md', 'favicon.svg'}
    copied = []
    for record in manifest['files']:
        relative = Path(record['path'])
        if relative.parts[0] != 'web':
            continue
        if relative.parts[1] == 'catalog' or relative.name in selected:
            copy_checked(root / relative, target / relative, copied, target, record['sha256'])
    copy_checked(root / 'release-manifest.json', target / 'release-manifest.json', copied, target)
    checks = root / 'checks/candidate-browser-checks.json'
    if checks.exists():
        report = read(checks)
        if report['manifest_sha256'] != digest(root / 'release-manifest.json'):
            raise ValueError('Browser checks describe a different candidate')
        copy_checked(checks, target / 'candidate-browser-checks.json', copied, target)
    mutations = root / 'checks/placeholder-mutation-checks.json'
    if mutations.exists():
        report = read(mutations)
        if report.get('passed') is not True or report.get('player_sha256') != digest(root / 'web/app_v2.js'):
            raise ValueError('Mutation checks describe a different player')
        copy_checked(mutations, target / 'placeholder-mutation-checks.json', copied, target)
    write(target / 'checkpoint.json', {'private_candidate': str(root),
          'manifest_sha256': digest(root / 'release-manifest.json'), 'files': copied,
          'browser_checks_complete': checks.exists(), 'published': False, 'release_hold': True})
    print('Checkpointed', len(copied), 'candidate text/evidence files at', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    checkpoint(parser.parse_args().candidate)
