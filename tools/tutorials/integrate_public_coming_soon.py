#!/usr/bin/env python3
"""Add unavailable screens to the website source, without publishing media.

Use a preserved pre-integration website directory as --baseline. This does
not install the refreshed candidate catalogs: their new narration is still
private. Existing playable scripts and public media roots must stay paired.
"""
import argparse
from pathlib import Path
import re
import shutil
import tempfile

from audit_staged_catalogs import CATALOGS, digest
from build_release_candidate import write_catalogs
from coming_soon import HELD, PLACEHOLDERS
from stage_lesson import read, write

ROUTING = {'number', 'app_key', 'host_app_key', 'series', 'section', 'slug'}
KEY = '20260911-coming-soon'


def check_preserved(before, after):
    """Only routing may change on an existing playable lesson."""
    old = {lesson['id']: lesson for lesson in before['lessons']}
    new = {lesson['id']: lesson for lesson in after['lessons']}
    ready = set(old) - set(HELD)
    assert set(new) == ready | set(PLACEHOLDERS)
    for identity in ready:
        original = {k: v for k, v in old[identity].items() if k not in ROUTING}
        updated = {k: v for k, v in new[identity].items() if k not in ROUTING}
        if updated != original:
            raise ValueError(f'Playable content changed: {identity}')
    return len(ready)


def integrate(baseline, destination, player):
    baseline, destination, player = map(Path, (baseline, destination, player))
    if baseline.resolve() == destination.resolve():
        raise ValueError('Preserve a separate baseline before integration')
    # Validate all generated files before changing any website source.
    with tempfile.TemporaryDirectory(prefix='tutorial-public-') as temporary:
        generated = Path(temporary)
        write_catalogs(baseline, generated)
        checks = []
        for filename in CATALOGS:
            before = read(baseline / 'catalog' / filename)
            after = read(generated / 'catalog' / filename)
            count = check_preserved(before, after)
            checks.append({'catalog': filename, 'ready_content_preserved': count,
                           'before_sha256': digest(before), 'after_sha256': digest(after)})
        index = (baseline / 'index.html').read_text()
        template = (player / 'index.html').read_text()
        pattern = r'<section\b[^>]*id="planned-card"[\s\S]*?</section>'
        panel = re.search(pattern, template)
        if not panel:
            raise ValueError('Missing coming-soon panel in verified player')
        index, count = re.subn(pattern, lambda _: panel.group(), index)
        if count != 1:
            raise ValueError('Expected exactly one website placeholder panel')
        for name in ('app_v2.js', 'lesson_catalog.js', 'module_navigation.js', 'styles.css'):
            index, count = re.subn(re.escape(name) + r'(?:\?v=[^"\s]+)?(?=")', name + '?v=' + KEY, index)
            if count != 1:
                raise ValueError(f'Missing cache key: {name}')
        index = re.sub(r'0 of \d+ complete', '0 of 71 complete', index)
        index = re.sub(r'Lesson 1 of \d+', 'Lesson 1 of 77', index)
        index = re.sub(r'(id="available-count">)\d+', r'\g<1>71', index)
        index = re.sub(r'(id="total-count">)\d+', r'\g<1>77', index)
        original_roots = re.findall(r'data-(?:production|audio|video4k)-root="[^"]+"',
                                    (baseline / 'index.html').read_text())
        if original_roots != re.findall(r'data-(?:production|audio|video4k)-root="[^"]+"', index):
            raise ValueError('Public media roots changed')
        (generated / 'index.html').write_text(index)
        for name in ('app_v2.js', 'styles.css'):
            shutil.copy2(player / name, generated / name)
        for path in generated.rglob('*'):
            if path.is_file():
                target = destination / path.relative_to(generated)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        report = {'published': False, 'scope': 'Website source placeholders only; not refreshed media',
                  'placeholder_ids': list(PLACEHOLDERS), 'catalogs': checks,
                  'media_roots_preserved': original_roots}
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--player', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    write(args.report, integrate(args.baseline, args.destination, args.player))
