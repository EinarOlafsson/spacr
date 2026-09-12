"""Merge only the two complete isolated lessons; archive previous local drafts.

The preceding OPS package must already be fully checked and checkpointed.
All unrelated catalog entries remain byte-equivalent as JSON values, and old
production folders/catalog files remain recoverable in a named backup folder.
"""
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile

from audit_staged_catalogs import CATALOGS
from check_completed_matrix import check, digest
from complete_model_media import LESSONS
from model_promotion import require_recorded_model
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def merge_catalog(target, source):
    """Replace exactly the assigned identities, preserving order and all others."""
    for catalog in (target, source):
        ids = [item['id'] for item in catalog['lessons']]
        if len(ids) != len(set(ids)) or not set(LESSONS) <= set(ids):
            raise ValueError('Both model lessons must occur exactly once')
    replacements = {item['id']: item for item in source['lessons'] if item['id'] in LESSONS}
    result = deepcopy(target)
    result['lessons'] = [deepcopy(replacements.get(item['id'], item)) for item in target['lessons']]
    return result


def main():
    stage = DEFAULT_STAGE.resolve()
    source = Path(read(stage / 'models-next-stage.json')['stage']).resolve()
    if source == stage or source.parent != stage.parent:
        raise ValueError('Expected the isolated model source stage')
    report = read(source / 'models-final-artifacts.json')
    if report.get('passed') is not True or [item['lesson'] for item in report['lessons']] != list(LESSONS):
        raise ValueError('Complete both isolated model media verifications first')
    prior = Path(read(stage / 'ops-final-candidate.json')['candidate'])
    if digest(prior / 'release-manifest.json') != digest(
            REPO / 'tools/tutorials/release_candidate/release-manifest.json'):
        raise ValueError('Finish and checkpoint OPS before changing the shared stage')
    for item in report['lessons']:
        current = check(source, item['lesson'], source.parent / 'tools/render_all_voices.py')
        if current != item['matrix']:
            raise ValueError('Isolated model media changed since final verification')
    receipt = stage / 'models-merge.json'
    if receipt.exists():
        raise FileExistsError('A model merge already exists; inspect its recovery receipt instead of repeating it')
    catalogs = {name: merge_catalog(read(stage / 'catalog' / name), read(source / 'catalog' / name))
                for name in CATALOGS}
    folders = [Path(kind) / identity for identity in LESSONS
               for kind in ('production', 'browser', 'browser-web', 'web-renditions')]
    for relative in folders:
        if not (source / relative).is_dir():
            raise FileNotFoundError(source / relative)
    backup = Path(tempfile.mkdtemp(prefix='models-before-refresh-', dir=stage))
    operations = {'source': str(source), 'backup': str(backup), 'copied': [],
                  'scope': 'Only lessons 21 and 22', 'complete': False, 'published': False}
    write(receipt, operations)
    # Source recordings are already present in the shared stage. Re-check every
    # captured file against its isolated copy before moving any production data.
    for capture in ('model_zoo_1507_inventory_v2', 'model_compare_1507_verified',
                    'model_compare_1507_gui_v2', 'model_compare_1507_api'):
        for path in (source / 'captures' / capture).rglob('*'):
            if path.is_file() and digest(path) != digest(stage / path.relative_to(source)):
                raise ValueError('Original and isolated capture differ')
    (backup / 'catalog').mkdir()
    for name in CATALOGS:
        shutil.copy2(stage / 'catalog' / name, backup / 'catalog' / name)
    for relative in folders:
        destination = stage / relative
        if destination.exists():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(saved))
        shutil.copytree(source / relative, destination)
        for path in (source / relative).rglob('*'):
            if path.is_file() and digest(path) != digest(stage / path.relative_to(source)):
                raise ValueError('A copied model artifact differs from its source')
        operations['copied'].append(str(relative))
        write(receipt, operations)
    for name, catalog in catalogs.items():
        write(stage / 'catalog' / name, catalog)
    for identity in LESSONS:
        require_recorded_model(stage, 'en', read(stage / 'production' / identity / 'lesson.en.json'))
    operations['complete'] = True
    write(receipt, operations)
    print('Merged only the verified model lessons; previous local drafts preserved at', backup, flush=True)


if __name__ == '__main__':
    main()
