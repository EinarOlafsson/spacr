"""Join real GUI and pure-API recordings without inventing GUI comparison results."""
from copy import deepcopy
import os
from pathlib import Path

from compose_report_capture import _frame, _read
from ops_geometry_example import digest
from stage_lesson import DEFAULT_STAGE, write


def compose(stage=DEFAULT_STAGE):
    root = Path(stage) / 'captures'
    gui, terminal = root / 'model_compare_1507_gui_v2', root / 'model_compare_1507_api'
    destination = root / 'model_compare_1507_verified'
    if destination.exists():
        raise FileExistsError('Preserve the existing verified recording')
    hashes = {}
    ga, ta = (_read(path / 'scientific_acceptance.json', hashes) for path in (gui, terminal))
    if (any(p.get('accepted') is not True or p.get('gui_workflow_completed') is not False
            or p.get('results_injected') is not False for p in (ga, ta))
            or ga.get('compare_clicked') is not False or ga.get('loaded_pixels_equal_originals') is not True):
        raise ValueError('Require genuine field loading distinct from the pure API example')
    run = ta['run']
    comparison = run['comparison']
    if (run.get('accepted') is not True or run.get('source_unchanged') is not True
            or (comparison['n_objects_a'], comparison['n_objects_b'], comparison['n_matched']) != (94, 95, 91)
            or run['accuracy_validated'] is not False or run['inference_performed'] is not False
            or run['helper_sha256'] != digest(Path(__file__).with_name('model_compare_example.py'))):
        raise ValueError('Recorded API output does not support the authored comparison')
    frames, sources = {}, []
    for prefix, source in [('gui', gui), ('api', terminal)]:
        provenance = _read(source / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'model_compare':
            raise ValueError('Expected completed native Model Compare recording')
        sources.append(provenance)
        for name, original in _read(source / 'frames.json', hashes).items():
            frame = deepcopy(original)
            path = _frame(source / frame['image'], frame['sha256'], source, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frames[prefix + '_' + name] = frame
    if any(digest(path) != value for path, value in hashes.items()):
        raise ValueError('A recording changed during composition')
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[-1], sources=sources,
          composition_only=True, gui_workflow_completed=False))
    write(destination / 'scientific_acceptance.json', {
        'accepted': True, 'scope': 'Real GUI introduction and pure saved-mask comparison API',
        'gui': ga, 'terminal': ta, 'source_hashes': hashes,
        'gui_workflow_completed': False, 'accuracy_validated': False, 'published': False})
    print(destination)


if __name__ == '__main__':
    compose()
