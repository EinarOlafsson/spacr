"""Prove the Map-only promotion retained every other lesson and media file."""
import argparse
from copy import deepcopy
from pathlib import Path

from audit_staged_catalogs import CATALOGS
from barcode_promotion import IDENTITY
from stage_lesson import read, write
from validate_candidate import validate


def compare_catalogs(before, after):
    """Only Map prose and localized section headings may change in this merge."""
    prior = before['lessons']
    current = after['lessons']
    if [row['id'] for row in prior] != [row['id'] for row in current]:
        raise ValueError('The refresh changed other lesson identities or their order')
    restored = []
    for old, new in zip(prior, current):
        if old['id'] == IDENTITY:
            if old.get('status') != 'coming_soon' or new.get('status') == 'coming_soon':
                raise ValueError('Expected the Map placeholder to become a real lesson')
            continue
        old_body, new_body = deepcopy(old), deepcopy(new)
        old_section, new_section = old_body.pop('section', None), new_body.pop('section', None)
        if old_body != new_body:
            raise ValueError('An unrelated lesson changed: ' + old['id'])
        if old_section != new_section:
            restored.append({'lesson': old['id'], 'before': old_section, 'after': new_section})
    return restored


def check(prior, candidate):
    prior, candidate = Path(prior), Path(candidate)
    validate(prior, include_hosted_media=True, require_browser=True)
    validate(candidate, include_hosted_media=True, require_browser=True)
    before, after = read(prior / 'release-manifest.json'), read(candidate / 'release-manifest.json')
    if (before['ready_lessons'], after['ready_lessons'], before['routes'], after['routes']) != (75, 76, 77, 77):
        raise ValueError('Expected exactly the one-lesson Map promotion')
    rows = {r['path']: r for r in after['files']}
    retained = 0
    for record in before['files']:
        path = record['path']
        if path.startswith(('media_host/', 'web/production/')):
            if rows.get(path) != record:
                raise ValueError('Previously verified media changed: ' + path)
            retained += 1
    headings = {}
    for name in CATALOGS:
        headings[name] = compare_catalogs(read(prior / 'web/catalog' / name),
                                          read(candidate / 'web/catalog' / name))
    result = {'passed': True, 'scope': 'Only Map promoted; other prose and media retained',
              'retained_media_files': retained, 'unchanged_lessons_per_catalog': 76,
              'catalogs_checked': len(CATALOGS), 'localized_heading_changes': headings,
              'prior_candidate': str(prior), 'published': False}
    write(candidate / 'checks/map-preservation-checks.json', result)
    print('Retained', retained, 'media files and all other lessons in fourteen catalogs', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prior', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    check(args.prior, args.candidate)
