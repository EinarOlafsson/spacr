"""Isolate the next two recordings while the OPS candidate finishes verification.

Copies verified captures and catalog inputs only; never alters the current
candidate, shared narration, app sources or another session's working tree.
"""
import hashlib
from pathlib import Path
import shutil
import tempfile

from stage_lesson import DEFAULT_STAGE, read, write, stage_lesson


def main():
    source = DEFAULT_STAGE
    pointer = source / 'models-next-stage.json'
    if pointer.exists():
        raise FileExistsError('An isolated model-tutorial stage already exists; reuse it explicitly')
    destination = Path(tempfile.mkdtemp(prefix='models-refresh-', dir=source.parent))
    records = {}
    folders = ['catalog', *['captures/' + name for name in (
        'model_zoo_1507_inventory_v2', 'model_compare_1507_gui_v2',
        'model_compare_1507_api', 'model_compare_1507_verified')]]
    for name in folders:
        before, after = source / name, destination / name
        shutil.copytree(before, after)
        for original in before.rglob('*'):
            if original.is_file():
                relative = original.relative_to(source)
                expected = hashlib.sha256(original.read_bytes()).hexdigest()
                if hashlib.sha256((destination / relative).read_bytes()).hexdigest() != expected:
                    raise ValueError('A staging copy differs from its verified input')
                records[str(relative)] = expected
    root = Path(__file__).resolve().parent
    for identity, capture in [('21_model_compare', 'model_compare_1507_verified'),
                              ('22_model_zoo', 'model_zoo_1507_inventory_v2')]:
        stage_lesson(root / 'lessons' / (identity + '.json'), capture, destination)
    write(pointer, {'stage': str(destination), 'lessons': ['21_model_compare', '22_model_zoo'],
                    'copied_source_hashes': records,
                    'scope': 'English staging only; translation and complete media checks pending',
                    'published': False})
    print(destination)


if __name__ == '__main__':
    main()
