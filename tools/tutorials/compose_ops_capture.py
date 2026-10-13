"""Combine actual OPS GUI, terminal and figure captures without repainting pixels."""
from copy import deepcopy
import os
from pathlib import Path

from compose_report_capture import _frame, _read
from stage_lesson import DEFAULT_STAGE, write
from ops_geometry_example import digest


def compose(stage=DEFAULT_STAGE):
    root = Path(stage) / 'captures'
    gui, terminal = root / 'ops_1507_gui_v2', root / 'ops_1507_geometry_api_v2'
    destination = root / 'ops_1507_verified'
    if destination.exists():
        raise FileExistsError('Preserve the existing composed recording')
    hashes = {}
    ga, ta = (_read(path / 'scientific_acceptance.json', hashes) for path in (gui, terminal))
    if (any(p.get('accepted') is not True or p.get('gui_workflow_completed') is not False
            or p.get('inputs_injected') is not False for p in (ga, ta))
            or ta.get('source_unchanged') is not True or ga.get('run_clicked') is not False):
        raise ValueError('Require real API evidence distinct from GUI navigation only')
    run = ta['run']
    if (run['canvas'] != [2756, 2756] or run['accepted_edges'] != 4
            or run['placed'] != 4 or run['full_pipeline_completed'] is not False
            or run['segmentation_or_decoding_performed'] is not False):
        raise ValueError('Recorded geometry no longer supports the authored scope')
    frames, sources = {}, []
    for prefix, source in [('gui', gui), ('api', terminal)]:
        provenance = _read(source / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'ops':
            raise ValueError('Expected completed native OPS recording')
        sources.append(provenance)
        for name, original in _read(source / 'frames.json', hashes).items():
            frame = deepcopy(original)
            path = _frame(source / frame['image'], frame['sha256'], source, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frames[prefix + '_' + name] = frame
    if any(digest(path) != sha for path, sha in hashes.items()):
        raise ValueError('Source changed while composing')
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[-1], sources=sources,
          composition_only=True, gui_workflow_completed=False))
    write(destination / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Current GUI route and four-tile geometry API example',
        'gui': ga, 'terminal': ta, 'source_hashes': hashes,
        'gui_workflow_completed': False, 'published': False})
    print(destination)


if __name__ == '__main__':
    compose()
