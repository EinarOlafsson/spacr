"""Compose genuine preview, disclosed failed batch and verified API-workaround clips.

Do not convert the native batch's rejection into a successful native run.
Only the separately recorded per-file overlay export is complete.
"""
from copy import deepcopy
import csv
import hashlib
import os
from pathlib import Path
import subprocess

from compose_report_capture import _frame, _read
from mask_preview_evidence import inspect
from stage_lesson import DEFAULT_STAGE, REPO, read, write


def compose(stage=DEFAULT_STAGE, preview_capture='mask', destination_name='mask_preview_and_explicit_overlay'):
    stage = Path(stage)
    destination = stage / 'captures' / destination_name
    if destination.exists():
        raise FileExistsError('Preserve earlier compositions')
    native = stage / 'mask_fresh_v1/captures/mask_bounded_native'
    command = stage / 'captures/mask_explicit_overlay_command_opaque'
    viewer = stage / 'captures/mask_explicit_overlay_viewer_opaque'
    old_preview = stage / 'captures' / preview_capture
    preview = inspect(old_preview)
    # The reuse rule: the live-preview code that produced these frames must be
    # byte-identical to the checkout composing them. The 09-09 `captures/mask`
    # keeps its original pin (04b23431f; its provenance names a757b8672). A new
    # recording is pinned to the commit it recorded itself.
    if preview_capture == 'mask':
        recorded_commit = '04b23431ff2eecff4d5ee444a8486188ec05152c'
    else:
        recorded_commit = read(old_preview / 'provenance.json')['commit']
    unchanged_implementation = {'preview_capture_commit': recorded_commit}
    for name in ['spacr/qt/widgets/live_preview.py', 'spacr/qt/widgets/preview_contract.py']:
        prior = subprocess.check_output(['git', 'show', recorded_commit + ':' + name], cwd=REPO)
        actual = (REPO / name).read_bytes()
        if prior != actual:
            raise ValueError('The live-preview implementation changed since the preview was recorded; re-record its controls')
        unchanged_implementation[name] = hashlib.sha256(actual).hexdigest()
    rejected = read(native / 'batch_acceptance.json')
    if rejected.get('accepted') is not False:
        raise ValueError('This composition explicitly describes the recorded rejected overlay run')
    outcome = read(native / 'batch_outcome.json')
    if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
        raise ValueError('The mask worker itself did not complete')
    api = read(command / 'scientific_acceptance.json')
    external = read(viewer / 'scientific_acceptance.json')
    if api.get('accepted') is not True or external.get('accepted') is not True:
        raise ValueError('Both the actual API command and real external viewer must finish')
    journal = stage / 'mask_fresh_v1/runs/2026-09-11_071752_9f9ee05b__mask/manifest.json'
    recorded = read(journal)
    project = stage / 'mask_fresh_v1/example_data/plate1'
    for path, digest in api['overlay']['source_hashes'].items():
        p = Path(path)
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError('A merged source changed after plotting')
        if p.suffix == '.npy':
            key = '/home/olafsson/.cache/spacr/example_data/plate1/' + p.relative_to(project).as_posix()
            if recorded['output_hashes'][key]['sha256'] != digest:
                raise ValueError('The plotted array is not the native journaled output')
    fields = {Path(row['source']).stem for row in api['overlay']['reports']}
    qc = []
    for role in ['cell', 'nucleus', 'pathogen']:
        with (project / f'test/qc/segmentation_qc_{role}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if {r['field'] for r in rows} != fields or any(r['object_type'] != role for r in rows):
            raise ValueError('QC rows do not describe the two plotted fields')
        qc.extend(rows)
    locations = [
        ('native', native, ['00_home', '01_module', '02_download', '03_data_ready', '20_batch_settings', '23_batch_finished']),
        ('preview', old_preview, ['04_live_image', '05_live_settings', '07_preview_result',
                                 '08_filters_before', '09_filters_after', '10_filters_restored',
                                 '11_model_diameter', '12_model_result']),
        ('command', command, None),
        ('viewer', viewer, ['09_saved_plate1_E01_17_1', '10_saved_plate1_L02_18_1', '30_ai_unsent_question']),
    ]
    hashes, frames, sources = {}, {}, []
    for prefix, source, selected in locations:
        provenance = _read(source / 'provenance.json', hashes)
        if provenance['module'] != 'mask':
            raise ValueError('Every recording must be the actual Mask workflow')
        sources.append(provenance)
        available = _read(source / 'frames.json', hashes)
        for key in available if selected is None else selected:
            frame = deepcopy(available[key])
            path = _frame(source / frame['image'], frame['sha256'], source, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(source)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('An original recording changed during composition')
    proof = dict(accepted=True, scope='Verified live preview plus explicitly disclosed partial native plotting and a separate complete per-file API export',
                 preview=preview, live_preview_implementation_unchanged=unchanged_implementation,
                 native_batch_accepted=False, native_failure_disclosed=True,
                 native_batch_rejection=rejected, native_worker_outcome=outcome,
                 native_journal=str(journal), merged_arrays_match_native_journal=True,
                 qc_rows=qc, cell_qc_precedes_mask_adjustment=True,
                 command=api, external_viewer=external, source_hashes=hashes,
                 native_plotting_loop_fixed=False, segmentation_accuracy_certified=False,
                 transparent_probes_preserved_but_not_used=True, published=False)
    destination.mkdir()
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(completed_capture=True, module='mask',
          composition_only=True, sources=sources, app_source_modified=False,
          native_batch_accepted=False, native_failure_disclosed=True))
    write(destination / 'scientific_acceptance.json', proof)
    print(destination)
    return proof


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--preview-capture', default='mask',
                        help='Capture directory under <stage>/captures holding the live-preview frames and arrays')
    parser.add_argument('--destination', default='mask_preview_and_explicit_overlay',
                        help='New composed capture directory under <stage>/captures; never overwritten')
    args = parser.parse_args()
    compose(args.stage, args.preview_capture, args.destination)
