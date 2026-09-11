"""Join original GUI-route and terminal frames without changing their pixels.

Acceptance is explicitly the API workaround, never a successful GUI export.
The failed GUI capture and partial file remain separate and untouched.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from compose_report_capture import _frame, _read
from stage_lesson import DEFAULT_STAGE, write


def check_sources(gui, terminal):
    if (gui.get('accepted') is not True or gui.get('gui_exports_completed') is not False
            or gui.get('original_unchanged') is not True
            or gui.get('private_measurements_unchanged') is not True
            or gui.get('remaining_workers') is not False or gui.get('exports') != []):
        raise ValueError('Expected preserved-source navigation-only GUI evidence')
    if (terminal.get('accepted') is not True or terminal.get('gui_defect_fixed') is not False
            or terminal.get('source_unchanged') is not True or len(terminal.get('exports', [])) != 6):
        raise ValueError('Expected six verified API exports, not a GUI success')
    for record in terminal['exports']:
        shape = record.get('shape', [])
        if (record.get('accepted') is not True or len(shape) != 2 or min(shape) <= 0
                or record.get('matrix_cells_checked') != shape[0] * shape[1]
                or record.get('maximum_float32_discrepancy') != 0):
            raise ValueError('Every export needs a complete independent matrix comparison')


def compose(stage=DEFAULT_STAGE):
    root = Path(stage).resolve() / 'captures'
    gui = root / 'anndata_api_introduction'
    terminal = root / 'anndata_api_terminal'
    destination = root / 'anndata_api_verified'
    if destination.exists():
        raise FileExistsError('Preserve the existing composed capture')
    hashes = {}
    ga = _read(gui / 'scientific_acceptance.json', hashes)
    ta = _read(terminal / 'scientific_acceptance.json', hashes)
    check_sources(ga, ta)
    frames, sources = {}, []
    for prefix, source in [('gui', gui), ('api', terminal)]:
        provenance = _read(source / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'anndata_export':
            raise ValueError('Expected completed native AnnData capture')
        sources.append(provenance)
        for key, original in _read(source / 'frames.json', hashes).items():
            frame = deepcopy(original)
            path = _frame(source / frame['image'], frame['sha256'], source, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(source)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed during composition')
    destination.mkdir()
    provenance = dict(sources[-1], sources=sources, composition_only=True,
                      gui_exports_completed=False, gui_defect_fixed=False)
    acceptance = dict(accepted=True, scope='GUI navigation plus explicit verified API workaround',
                      gui=ga, terminal=ta, source_hashes=hashes,
                      gui_defect_fixed=False, published=False)
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', provenance)
    write(destination / 'scientific_acceptance.json', acceptance)
    print(destination)


if __name__ == '__main__':
    compose()
