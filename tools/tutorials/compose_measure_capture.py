"""Join usable preview controls and verified full-run footage, without edits.

Only explicitly listed frames are included. The old offscreen Crop settings
scenes remain preserved but cannot enter this composition. Both captures use
the same byte-identical published arrays; the batch is not rerun or fabricated.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

from compose_report_capture import _frame, _read, _same_hash
from measure_controls_evidence import check_controls
from measure_evidence import inspect_project
from stage_lesson import DEFAULT_STAGE, REPO, write


PREVIEW_FRAMES = ('01_module', '03_data_ready', '04_preview_normalization_setup',
                  '05_live_all_channels', '06_live_channel_zero',
                  '07_live_second_field', '08_saved_normalization_restored')
BATCH_FRAMES = ('20_batch_settings', '26_measure_console_complete',
                '25_measure_figure_00', '25_measure_figure_02',
                '24_batch_figure', '30_ai_unsent_question')


def compose(stage=DEFAULT_STAGE):
    stage = Path(stage).resolve()
    preview = stage / 'measure_controls/captures/measure_visible_controls_v2'
    batch = stage / 'captures/measure_readable_native'
    destination = stage / 'captures/measure_usable_preview_verified_batch'
    if destination.exists():
        raise FileExistsError('Preserve the existing accepted composition')
    hashes = {}
    proof = _read(preview / 'scientific_acceptance.json', hashes)
    check_controls(proof)
    receipt = _read(REPO / 'tools/tutorials/evidence/2026-09-11_measure_readable_native_checks.json', hashes)
    if not receipt['batch_acceptance']['accepted']:
        raise ValueError('The native batch must have completed successfully')
    project = stage / 'measure_production/example_data/plate1'
    output = inspect_project(project, batch)
    if output != receipt['independent_output_checks']:
        raise ValueError('The previously checked batch outputs changed')
    for published, digest in proof['source_hashes'].items():
        stem = Path(published).stem
        if output['source_file_sha256'].get(stem) != digest:
            raise ValueError('The new preview does not use the same original batch array')
        _same_hash(stage / 'measure_controls/example_data/plate1/merged' / Path(published).name,
                   digest, hashes)
    for row in receipt['saved_pdf_files']:
        _same_hash(project / row['path'], row['sha256'], hashes)
    frames, sources = {}, []
    for prefix, root, names in [('preview', preview, PREVIEW_FRAMES), ('batch', batch, BATCH_FRAMES)]:
        provenance = _read(root / 'provenance.json', hashes)
        if provenance.get('completed_capture') is not True or provenance.get('module') != 'measure':
            raise ValueError('Both native captures must be complete Measure captures')
        sources.append(provenance)
        available = _read(root / 'frames.json', hashes)
        for key in names:
            frame = deepcopy(available[key])
            path = _frame(root / frame['image'], frame['sha256'], root, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(root)
            frames[prefix + '_' + key] = frame
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed while being read')
    destination.mkdir()
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[0], sources=sources,
                                              composition_only=True, app_source_modified=False))
    acceptance = {'accepted': True, 'scope': 'Visible preview controls plus verified full batch',
                  'preview': proof, 'independent_output_checks': output,
                  'batch_acceptance': receipt['batch_acceptance'], 'source_hashes': hashes,
                  'offscreen_crop_dialog_scenes_included': False,
                  'application_layout_fixed': False, 'biological_validation': False,
                  'published': False}
    write(destination / 'scientific_acceptance.json', acceptance)
    print(destination)


if __name__ == '__main__':
    compose()
