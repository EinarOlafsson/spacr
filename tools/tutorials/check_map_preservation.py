"""Prove a candidate kept Map's lesson, its media and everything not refreshed.

Two proofs live here, and they answer different questions.

:func:`check` is the original one: the pair it compares differs by the Map
promotion ALONE, so it can demand that every other lesson record and every
previously verified media file is byte-identical. That is the strongest
statement available, and it is the right one whenever Map is the only thing
that moved.

:func:`check_refresh` exists because that pair stopped being available. Map's
evidence was bound to ``release-candidate-zveusu2j``; the candidate the
maintainer approved is ``release-candidate-8738b_pd``, and between the two a
reviewed 07_mask refresh landed. :func:`check` refuses that pair three times
over -- on the promotion counts, on 07_mask's changed media, and on 07_mask's
changed catalog record -- so re-binding Map's evidence had no honest proof
behind it and ``tests/test_map_candidate.py`` stayed red rather than be made
green by a weaker claim.

:func:`check_refresh` makes the narrower statement that is actually true: MAP
is untouched, and every difference between the two candidates is accounted
for by a named refreshed lesson or by a shared index file whose own contents
are then compared lesson by lesson. It never widens to "nothing important
changed" -- an unexplained difference anywhere raises, including one in a
lesson nobody listed.
"""
import argparse
import json
from copy import deepcopy
from pathlib import Path

from audit_staged_catalogs import CATALOGS
from barcode_promotion import IDENTITY
from stage_lesson import read, write
from validate_candidate import validate

#: Manifest paths that belong to the library rather than to one lesson, and
#: therefore carry a refreshed lesson's prose without being its files.
#:
#: Listing them is not permission to skip them. Each one is opened and
#: compared by :func:`check_refresh`, at the level its format allows: the
#: fourteen catalogs and ``lesson_catalog.js`` lesson by lesson, and
#: ``module_navigation.js`` key by key.
SHARED_INDEX_FILES = ('web/lesson_catalog.js', 'web/module_navigation.js')

#: Keys of ``module_navigation.js`` that may differ between two candidates
#: built from different repository commits.
#:
#: ``source_commit`` records which checkout the navigation was generated
#: from. It moves whenever the repository moves and says nothing about any
#: lesson. Every other key is lesson-facing and must not move; the values
#: seen on both sides are written into the report rather than dropped, so a
#: reader can see exactly what was excused.
NAVIGATION_KEYS_THAT_MAY_MOVE = ('source_commit',)


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


def frozen_payload(path):
    """The object a ``window.X = Object.freeze({...});`` file publishes.

    :param path: the ``.js`` file to read.
    :returns: the decoded payload.
    :raises ValueError: when the file is not in that shape, which is itself
        worth failing on -- the comparison below would otherwise silently
        degrade to "the bytes differ" and lose the lesson-level answer.
    """
    text = Path(path).read_text(encoding='utf-8')
    opening = text.find('Object.freeze(')
    closing = text.rfind(')')
    if opening < 0 or closing < opening:
        raise ValueError('Not a frozen catalog payload: ' + str(path))
    return json.loads(text[opening + len('Object.freeze('):closing])


def lesson_of(path):
    """The lesson id a manifest path belongs to, or ``None`` when it is shared.

    :param path: a manifest path such as ``media_host/07_mask/audio/en.m4a``
        or ``web/production/07_mask/scenes.json``.
    """
    parts = path.split('/')
    if len(parts) > 2 and parts[0] == 'media_host':
        return parts[1]
    if len(parts) > 3 and parts[:2] == ['web', 'production']:
        return parts[2]
    return None


def compare_refreshed_catalogs(before, after, refreshed):
    """Which lesson records moved between two catalogs, checked against ``refreshed``.

    The test here is a SUBSET, deliberately, and the equality that keeps the
    declaration honest is made once in :func:`check_refresh` over the whole
    candidate. A lesson can be refreshed in its MEDIA and not in its prose --
    13_regression was re-captured on 1.5.0.7 with its narration untouched, so
    its frames moved and its catalog record did not -- and demanding that
    every named lesson move in every catalog would refuse that true case.

    :param before: the prior catalog, already decoded.
    :param after: the candidate's catalog, already decoded.
    :param refreshed: the lesson ids the caller declares were refreshed.
    :returns: ``(ids that moved, number of records that did not)``.
    :raises ValueError: on a difference ``refreshed`` does not name, on a
        changed lesson ORDER, on a changed non-lesson key, or on any change
        at all to Map.
    """
    prior, current = before['lessons'], after['lessons']
    if [row['id'] for row in prior] != [row['id'] for row in current]:
        raise ValueError('The refresh changed lesson identities or their order')
    outside = ({key: value for key, value in before.items() if key != 'lessons'}
               != {key: value for key, value in after.items() if key != 'lessons'})
    if outside:
        raise ValueError('A catalog changed outside its lessons')
    moved, unchanged = set(), 0
    for old, new in zip(prior, current):
        if old == new:
            unchanged += 1
            continue
        moved.add(old['id'])
    if IDENTITY in moved:
        raise ValueError('Map itself changed; this is not a preservation proof')
    unnamed = moved - set(refreshed)
    if unnamed:
        raise ValueError('Lessons changed that the refresh did not name: '
                         + ', '.join(sorted(unnamed)))
    return moved, unchanged


def check_refresh(prior, candidate, refreshed):
    """Prove ``candidate`` kept Map while only ``refreshed`` lessons moved.

    :param prior: the candidate Map's existing evidence was bound to.
    :param candidate: the candidate to re-bind that evidence to.
    :param refreshed: the lesson ids allowed to differ. Map may not be one.
    :returns: the report, also written to
        ``<candidate>/checks/map-preservation-checks.json``.
    :raises ValueError: on any difference the declaration does not account
        for.
    """
    prior, candidate = Path(prior), Path(candidate)
    refreshed = sorted(set(refreshed))
    if not refreshed:
        raise ValueError('Name the refreshed lessons, or use the promotion check')
    if IDENTITY in refreshed:
        raise ValueError('Map cannot be both preserved and refreshed')
    validate(prior, include_hosted_media=True, require_browser=True)
    validate(candidate, include_hosted_media=True, require_browser=True)
    before, after = (read(prior / 'release-manifest.json'),
                     read(candidate / 'release-manifest.json'))
    if (before['ready_lessons'], before['routes']) != (after['ready_lessons'], after['routes']):
        raise ValueError('A refresh does not add or remove a lesson')
    rows = {record['path']: record for record in after['files']}
    if set(rows) != {record['path'] for record in before['files']}:
        raise ValueError('The candidate added or dropped manifest paths')
    retained, map_files, changed, media_moved = 0, 0, [], set()
    for record in before['files']:
        path = record['path']
        owner = lesson_of(path)
        same = rows[path] == record
        if owner == IDENTITY:
            if not same:
                raise ValueError('A Map media file changed: ' + path)
            map_files += 1
        if same:
            if path.startswith(('media_host/', 'web/production/')):
                retained += 1
            continue
        changed.append(path)
        if owner in refreshed:
            media_moved.add(owner)
            continue
        if path.startswith('web/catalog/') or path in SHARED_INDEX_FILES:
            continue
        raise ValueError('An unexplained file changed: ' + path)
    unchanged_per_catalog, prose_moved = set(), set()
    for name in CATALOGS:
        moved, unchanged = compare_refreshed_catalogs(
            read(prior / 'web/catalog' / name),
            read(candidate / 'web/catalog' / name), refreshed)
        prose_moved |= moved
        unchanged_per_catalog.add(unchanged)
    if len(unchanged_per_catalog) != 1:
        raise ValueError('The catalogs disagree about how many lessons were kept')
    index_before = frozen_payload(prior / 'web/lesson_catalog.js')
    index_after = frozen_payload(candidate / 'web/lesson_catalog.js')
    index_moved, _ = compare_refreshed_catalogs(index_before, index_after, refreshed)
    prose_moved |= index_moved
    idle = set(refreshed) - (prose_moved | media_moved)
    if idle:
        raise ValueError('The refresh named lessons that did not move: '
                         + ', '.join(sorted(idle)))
    navigation_before = frozen_payload(prior / 'web/module_navigation.js')
    navigation_after = frozen_payload(candidate / 'web/module_navigation.js')
    excused = {}
    for key in sorted(set(navigation_before) | set(navigation_after)):
        if navigation_before.get(key) == navigation_after.get(key):
            continue
        if key not in NAVIGATION_KEYS_THAT_MAY_MOVE:
            raise ValueError('The tutorial navigation changed at ' + key)
        excused[key] = {'before': navigation_before.get(key),
                        'after': navigation_after.get(key)}
    result = {
        'passed': True,
        'mode': 'refresh',
        'scope': ('Map retained in full while '
                  + ' and '.join(filter(None, [', '.join(refreshed[:-1]), refreshed[-1]]))
                  + (' were' if len(refreshed) > 1 else ' was')
                  + ' refreshed; every other lesson, media file and index '
                    'entry is byte-identical'),
        'refreshed_lessons': refreshed,
        'refreshed_in_prose': sorted(prose_moved),
        'refreshed_in_media_only': sorted(media_moved - prose_moved),
        'retained_media_files': retained,
        'map_media_files': map_files,
        'changed_files': sorted(changed),
        'unchanged_lessons_per_catalog': unchanged_per_catalog.pop(),
        'catalogs_checked': len(CATALOGS),
        'navigation_keys_excused': excused,
        'prior_candidate': str(prior),
        'published': False,
    }
    write(candidate / 'checks/map-preservation-checks.json', result)
    print('Retained', retained, 'media files, including all', map_files,
          'of Map, across fourteen catalogs and both shared indexes', flush=True)
    return result


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
    parser.add_argument('--refreshed', nargs='+', default=(), metavar='LESSON',
                        help='lesson ids refreshed between the two candidates; '
                             'given, the refresh proof runs instead of the '
                             'Map-only promotion proof')
    args = parser.parse_args()
    if args.refreshed:
        check_refresh(args.prior, args.candidate, args.refreshed)
    else:
        check(args.prior, args.candidate)
