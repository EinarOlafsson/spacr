"""Join usable preview controls and verified full-run footage, without edits.

Only explicitly listed frames are included. The old offscreen Crop settings
scenes remain preserved but cannot enter this composition. Both captures use
the same byte-identical published arrays; the batch is not rerun or fabricated.
"""
from copy import deepcopy
import argparse
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


def compose(stage=DEFAULT_STAGE, *, preview=None, batch=None, project=None,
            preview_project=None, receipt_path=None, destination=None,
            recorded_project=None, templates=None, ram=None):
    stage = Path(stage).resolve()
    preview = Path(preview or stage / 'measure_controls/captures/measure_visible_controls_v2').resolve()
    batch = Path(batch or stage / 'captures/measure_readable_native').resolve()
    destination = Path(destination or stage / 'captures/measure_usable_preview_verified_batch').resolve()
    project = Path(project or stage / 'measure_production/example_data/plate1').resolve()
    preview_project = Path(preview_project or stage / 'measure_controls/example_data/plate1').resolve()
    if destination.exists():
        raise FileExistsError('Preserve the existing accepted composition')
    hashes = {}
    proof = _read(preview / 'scientific_acceptance.json', hashes)
    check_controls(proof)
    receipt = _read(Path(receipt_path or REPO / 'tools/tutorials/evidence/2026-09-11_measure_readable_native_checks.json'), hashes)
    if not receipt['batch_acceptance']['accepted']:
        raise ValueError('The native batch must have completed successfully')
    output = inspect_project(project, batch, recorded_project=recorded_project,
                             preview_capture=preview if recorded_project is not None else None)
    recorded_output = receipt['independent_output_checks']
    batch_output = {key: value for key, value in output.items() if key != 'live_grid_counts'}
    recorded_batch_output = {key: value for key, value in recorded_output.items()
                             if key != 'live_grid_counts'}
    if batch_output != recorded_batch_output:
        raise ValueError('The previously checked batch outputs changed')
    for published, digest in proof['source_hashes'].items():
        stem = Path(published).stem
        if output['source_file_sha256'].get(stem) != digest:
            raise ValueError('The new preview does not use the same original batch array')
        _same_hash(preview_project / 'merged' / Path(published).name,
                   digest, hashes)
    for row in receipt['saved_pdf_files']:
        _same_hash(project / row['path'], row['sha256'], hashes)
    current = templates is not None or ram is not None
    if current and (templates is None or ram is None):
        raise ValueError('Current complete composition requires both Templates and RAM captures')
    plans = [('preview', preview, {key: 'preview_' + key for key in PREVIEW_FRAMES}, 'measure'),
             ('batch', batch, {key: 'batch_' + key for key in BATCH_FRAMES}, 'measure')]
    if current:
        templates, ram = Path(templates).resolve(), Path(ram).resolve()
        template_proof = _read(templates / 'templates_shortcuts.json', hashes)
        if (template_proof.get('accepted') is not True or template_proof.get('keys_changed') is not False
                or template_proof.get('measure_settings_unchanged') is not True
                or template_proof.get('template_applied') is not False):
            raise ValueError('The native Templates tour must preserve shortcuts and measurement settings')
        _same_hash(project / 'settings/measure_crop_settings.csv', template_proof['csv_sha256'], hashes)
        ram_proof = _read(ram / 'ram_guard.json', hashes)
        if ram_proof.get('accepted') is not True or ram_proof.get('run_started') is not False:
            raise ValueError('The memory demonstration must show the real warning without starting analysis')
        preview_names = {key: 'preview_' + key for key in PREVIEW_FRAMES}
        preview_names['07c_checked_two_fields'] = 'preview_07c_checked_two_fields'
        preview_names.update({key: key for key in ('09a_qc_popup', '09b_image_preprocessing')})
        plans = [('preview', preview, preview_names, 'measure'),
                 ('batch', batch, {key: 'batch_' + key for key in ('20_batch_settings', '25_measure_figure_02')}, 'measure'),
                 ('templates', templates, {key: 'tpl_' + key for key in
                    ('31_template_save_name', '33_templates_imported', '35_templates_renamed', '37_change_shortcuts')}, 'measure'),
                 ('ram', ram, {'batch_21_ram_guard': 'batch_21_ram_guard'}, 'home')]
        lesson = _read(REPO / 'tools/tutorials/lessons/08_measure.json', hashes)
        required = {scene['visual'] for scene in lesson['scenes']}
        plans = [(prefix, root, {key: name for key, name in names.items()
                                if name in required}, module)
                 for prefix, root, names, module in plans]
    frames, sources = {}, []
    for prefix, root, names, module in plans:
        provenance = _read(root / 'provenance.json', hashes)
        if (provenance.get('completed_capture') is not True or provenance.get('module') != module
                or provenance.get('app_source_modified') is not False):
            raise ValueError('Every native capture must be complete and preserve application source')
        sources.append(provenance)
        available = _read(root / 'frames.json', hashes)
        for key, destination_key in names.items():
            frame = deepcopy(available[key])
            path = _frame(root / frame['image'], frame['sha256'], root, hashes)
            frame['image'] = os.path.relpath(path, destination)
            frame['source_capture'] = str(root)
            frames[destination_key] = frame
    if current:
        if set(frames) != {scene['visual'] for scene in lesson['scenes']}:
            raise ValueError('Current composition must provide every exact Measure lesson visual')
    for path, digest in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError('A composition input changed while being read')
    destination.mkdir()
    write(destination / 'frames.json', frames)
    write(destination / 'provenance.json', dict(sources[0], sources=sources,
                                              composition_only=True, app_source_modified=False))
    acceptance = {'accepted': True, 'scope': 'Visible preview controls plus verified full batch',
                  'preview': proof, 'independent_output_checks': output,
                  'recorded_preview_grid_counts': recorded_output['live_grid_counts'],
                  'current_preview_grid_counts': output['live_grid_counts'],
                  'complete_batch_output_preserved': True,
                  'batch_acceptance': receipt['batch_acceptance'], 'source_hashes': hashes,
                  'offscreen_crop_dialog_scenes_included': False,
                  'application_layout_fixed': False, 'biological_validation': False,
                  'published': False}
    if current:
        acceptance.update(templates=template_proof, ram_warning=ram_proof,
                          scope='Complete current Measure controls, verified sixteen-field run, Templates and disclosed RAM warning')
    write(destination / 'scientific_acceptance.json', acceptance)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    for name in ('preview', 'batch', 'project', 'preview-project', 'receipt-path',
                 'destination', 'recorded-project', 'templates', 'ram'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    compose(**vars(args))
