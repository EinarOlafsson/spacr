"""Merge only verified Map Barcodes media, with recoverable local backups.

The isolated recording, existing candidate and other lessons are preserved.
An interrupted merge leaves an explicit receipt: inspect it, never rerun blindly.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

from audit_staged_catalogs import CATALOGS
from barcode_promotion import IDENTITY, require_recorded_map
from check_completed_matrix import digest
from stage_lesson import DEFAULT_STAGE, REPO, read, write
from validate_candidate import validate


def merge_catalog(target, source):
    """Replace only Map, retaining all other lesson values and catalog metadata."""
    for catalog in (target, source):
        ids = [row['id'] for row in catalog['lessons']]
        if ids.count(IDENTITY) != 1 or len(ids) != len(set(ids)):
            raise ValueError('Map must occur exactly once, without duplicate lesson identities')
    replacement = next(row for row in source['lessons'] if row['id'] == IDENTITY)
    original = next(row for row in target['lessons'] if row['id'] == IDENTITY)
    if (replacement.get('status') == 'coming_soon'
            or replacement.get('number') != 12 or original.get('number', 12) != 12):
        raise ValueError('The completed lesson must preserve its numbered identity')
    result = deepcopy(target)
    result['lessons'] = [deepcopy(replacement if row['id'] == IDENTITY else row)
                         for row in target['lessons']]
    return result


def merge(source, prior, stage=DEFAULT_STAGE):
    source, prior, stage = (Path(path).resolve() for path in (source, prior, stage))
    if source == stage or source.parent != stage.parent:
        raise ValueError('Expected a distinct isolated recording beside the shared stage')
    receipt = stage / 'map-merge.json'
    if receipt.exists():
        raise FileExistsError('Inspect the existing map-merge recovery receipt; do not repeat the merge')
    report = read(source / 'map-final-artifacts.json')
    if report.get('passed') is not True or report.get('lesson') != IDENTITY:
        raise ValueError('Verify the complete isolated lesson first')
    english = read(source / 'production' / IDENTITY / 'lesson.en.json')
    matrix = require_recorded_map(source, 'en', english)
    if matrix != report['matrix']:
        raise ValueError('Isolated media changed since its final verification')
    validate(prior, include_hosted_media=True, require_browser=True)
    if digest(prior / 'release-manifest.json') != digest(
            REPO / 'tools/tutorials/release_candidate/release-manifest.json'):
        raise ValueError('Checkpoint the preceding candidate before merging Map')
    catalogs = {name: merge_catalog(read(stage / 'catalog' / name),
                                    read(source / 'catalog' / name)) for name in CATALOGS}
    folders = [(Path(kind) / IDENTITY, Path(kind) / IDENTITY)
               for kind in ('production', 'browser', 'browser-web', 'web-renditions')]
    folders += [(Path('captures') / name, Path('captures') / name)
                for name in ('map_verified', 'map_search_v2', 'map_barcodes_api')]
    folders += [(Path('map_references/map_search_v2'), Path('map_references/map_search_v2')),
                (Path('example_data/sequencing'), Path('map_dataset')),
                (Path('map_api_recorded_output'), Path('map_api_recorded_output'))]
    for relative, _ in folders:
        if not (source / relative).is_dir():
            raise FileNotFoundError(source / relative)
    backup = Path(tempfile.mkdtemp(prefix='map-before-refresh-', dir=stage))
    operations = {'source': str(source), 'backup': str(backup), 'copied': [],
                  'scope': 'Only lesson 12 and its actual recording/data evidence',
                  'prior_manifest_sha256': digest(prior / 'release-manifest.json'),
                  'complete': False, 'published': False}
    write(receipt, operations)
    (backup / 'catalog').mkdir()
    for name in CATALOGS:
        shutil.copy2(stage / 'catalog' / name, backup / 'catalog' / name)
    for relative, target in folders:
        destination = stage / target
        if destination.exists():
            saved = backup / target
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(saved))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source / relative, destination)
        for path in (source / relative).rglob('*'):
            if path.is_file() and digest(path) != digest(destination / path.relative_to(source / relative)):
                raise ValueError('A copied Map artifact differs from its verified source')
        operations['copied'].append({'source': str(relative), 'destination': str(target)})
        write(receipt, operations)
    for name, catalog in catalogs.items():
        write(stage / 'catalog' / name, catalog)
    require_recorded_map(stage, 'en', english)
    operations['complete'] = True
    write(receipt, operations)
    print('Merged only verified Map Barcodes; previous drafts preserved at', backup, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--stage', default=DEFAULT_STAGE, type=Path)
    args = parser.parse_args()
    merge(args.source, args.prior, args.stage)
