"""Bounded artifact fixtures test acceptance; no Cellpose, recording or GPU."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/tutorials')]
import build_module_stage_with_acceptance as builder  # noqa: E402
import validate_cellpose_host as validator  # noqa: E402
from cellpose_apply_evidence import minimum_area_labels  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def get(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@pytest.fixture
def artifacts(tmp_path):
    stage = tmp_path / 'recording'
    source = stage / 'captures/cellpose_masks'
    source.mkdir(parents=True)
    recorded = tmp_path / 'recorded-checkout'
    (recorded / 'spacr/qt').mkdir(parents=True)
    (recorded / 'spacr/qt/app.py').write_bytes(b'recorded application source\n')
    (recorded / 'tools/tutorials').mkdir(parents=True)
    for name in ('verify_cellpose_apply.py', 'compose_cellpose_apply_capture.py', 'cellpose_apply_evidence.py'):
        (recorded / 'tools/tutorials' / name).write_bytes(b'recorded fixture source\n')
    put(stage / 'exact-source.json', {'fresh': True, 'private_profile': True,
                                     'sha256': {'spacr/qt/app.py': digest(recorded / 'spacr/qt/app.py')}})
    english = tmp_path / 'lesson.en.json'
    shutil.copyfile(ROOT / 'tools/tutorials/lessons/20_cellpose_masks.json', english)
    lesson = get(english)
    names = [scene['visual'] for scene in lesson['scenes']]
    work = stage / 'cellpose_apply_runs/fixture'
    (work / 'masks').mkdir(parents=True)
    originals = tmp_path / 'originals'
    originals.mkdir()
    raw = np.zeros((512, 512), np.uint16)
    raw[10:60, 10:60] = 1
    raw[70:90, 70:120] = 2
    raw[100:110, 100:130] = 3
    np.save(source / 'preview_cell.npy', raw, allow_pickle=False)
    np.save(source / 'preview_cell_filtered.npy', minimum_area_labels(raw, 1001), allow_pickle=False)
    np.save(source / 'preview_cell_restored.npy', raw, allow_pickle=False)
    output_rows, reference_rows, original_hashes, figures = [], [], {}, []
    for index in range(1, 4):
        name = f'cell_pair_{index:02}.tif'
        image = (np.arange(512 * 512).reshape(512, 512) % 4096 + index).astype(np.uint16)
        tifffile.imwrite(originals / name, image)
        shutil.copyfile(originals / name, work / name)
        original_hashes[str(originals / name)] = digest(originals / name)
        mask = raw.copy()
        mask[10, 10] = 0
        tifffile.imwrite(work / 'masks' / name, mask)
        output_rows.append({'image': name, 'sha256': digest(work / 'masks' / name)})
        reference_rows.append({'name': name, 'source_sha256': digest(originals / name),
                               'saved_mask_sha256': digest(work / 'masks' / name),
                               'checked_pixels': 512 * 512, 'exact_mask_equal': True})
        np.savez_compressed(source / f'batch_figure_{index - 1:02}_arrays.npz',
                            original=image, overlay=np.zeros((512, 512, 3), np.uint8),
                            flow=np.zeros((512, 512), np.uint8))
        Image.new('RGB', (8, 8)).save(source / f'batch_figure_{index - 1:02}.png')
        figures.append({'file': f'batch_figure_{index - 1:02}_arrays.npz', 'source_image': name,
                        'exact_values_checked': {'original': 262144, 'overlay': 786432, 'flow': 262144}})
    checkpoint = tmp_path / 'cpsam'
    checkpoint.write_bytes(b'fixture checkpoint identity, not a model\n')
    proof = {'lesson': lesson['id'], 'accepted': False, 'synthetic': False,
             'private_folder': '/tmp/spacr-tutorial-current/cellpose_apply_runs/fixture',
             'original_inputs': original_hashes, 'accuracy_validated': False,
             'ground_truth_used': False, 'pipeline': {'accepted': True, 'reasons': [],
                 'worker_finished': True, 'worker_ok': True, 'figure_count': 3},
             'outputs': output_rows, 'preview': {'batch_mask_equal': False},
             'filter': {'cutoff': 1001, 'raw_unchanged': True, 'restored': True},
             'original_inputs_preserved': True, 'private_inputs_preserved': True,
             'figures': [{'path': f'batch_figure_{i:02}.png',
                          'sha256': digest(source / f'batch_figure_{i:02}.png')} for i in range(3)]}
    put(source / 'scientific_acceptance.json', proof)
    put(source / 'configured_settings.json', {'model_name': 'cpsam'})
    put(source / 'batch_outcome.json', {'finished': True, 'ok': True, 'errors': []})
    put(source / 'batch_console.json', ['Completed bounded fixture batch.'])
    put(source / 'provenance.json', {'module': 'cellpose_masks', 'completed_capture': True})
    reference = {'accepted': True, 'scope': 'input/output attribution, not segmentation accuracy',
                 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                 'files': reference_rows, 'figures': figures,
                 'preview_default_reference': {'exact_mask_equal': True, 'checked_pixels': 262144,
                                               'differs_from_batch': True}}
    put(source / 'independent_reference.json', reference)
    Image.new('RGB', (3840, 2160), (12, 24, 36)).save(source / 'poster.png')
    frames, gates = {}, {}
    for name in names:
        video, native = source / f'{name}.mp4', source / f'{name}.native.mp4'
        video.write_bytes((name + ': handoff fixture').encode())
        native.write_bytes((name + ': native fixture').encode())
        gate = {'video': f'/tmp/spacr-tutorial-current/captures/cellpose_masks/{native.name}',
                'size': [2560, 1440], 'fps': 30, 'passed': True, 'reasons': [],
                'limits': {'min_fps': 27.9}, 'distinct_fps': 29.9,
                'cursor': {'jump_count': 0, 'jumps': [], 'found_fraction': 1.0},
                'top_bar': {'hidden': True}, 'adapter_bypasses': [], 'app_freeze_seconds': [],
                'app_slow_seconds': [], 'capture_loss_seconds': []}
        put(source / f'{name}.gate.json', gate)
        row = {'file': f'{name}.gate.json', 'passed': True, 'reasons': [], 'distinct_fps': 29.9}
        gates[name] = row
        receipt = {'video': video.name, 'sha256': digest(video), 'full_decode_passed': True,
                   'audio_streams': 0, 'size': [3840, 2160], 'frames': [{'decoded_md5': 'fixture'}],
                   'duration': 1.0, 'target_fps': 30, 'encoded_fps': '30/1',
                   'native': {'video': native.name, 'sha256': digest(native), 'size': [2560, 1440]},
                   'gate': {'passed': True, 'reasons': []}, 'adapter_bypasses': []}
        put(source / f'{name}.capture.json', receipt)
        put(source / f'{name}.host.json', {'adapter_bypasses': [],
            'app_paint_samples': [[i / 10, i * 3] for i in range(11)]})
        put(source / f'{name}.pointer.json', [])
        frames[name] = {'image': 'poster.png', 'sha256': digest(source / 'poster.png'), 'dialogs': [],
                        'clip': {'video': video.name, 'sha256': digest(video),
                                 'native_video': native.name, 'native_sha256': digest(native),
                                 'receipt': f'{name}.capture.json',
                                 'receipt_sha256': digest(source / f'{name}.capture.json'), 'gate': row}}
    put(source / 'frames.json', frames)
    put(source / 'host_gate_summary.json', {'clips': gates, 'failed': [], 'passed': True})
    picks = tmp_path / 'selected-scenes.json'
    put(picks, {str(source): names})
    verifier_path = source / 'verification_run_retry.json'
    arguments = SimpleNamespace(sources=[source], selected_scenes=picks, english=english,
                                verifier_receipts=[verifier_path], output=tmp_path / 'accepted.json',
                                checkout=ROOT, private_bind=[], terminal_review=None)
    result = SimpleNamespace(stage=stage, source=source, work=work, names=names, english=english,
                             recorded=recorded, picks=picks, verifier_path=verifier_path,
                             args=arguments, tmp_path=tmp_path)
    refresh_verifier(result)
    return result


def refresh_verifier(artifacts):
    source = artifacts.source
    reference = source / 'independent_reference.json'
    stamp = datetime.fromtimestamp(reference.stat().st_mtime, timezone.utc)
    dependencies = [source / name for name in ('scientific_acceptance.json', 'configured_settings.json',
                    'frames.json', 'provenance.json', *validator.PREVIEWS)]
    dependencies += [artifacts.stage / 'exact-source.json', *source.glob('batch_figure_*_arrays.npz')]
    put(artifacts.verifier_path, {'capture': str(source), 'source_run': str(artifacts.stage),
        'reference_absent_at_start': True, 'dependencies_unchanged': True, 'exit_code': 0,
        'started_at': (stamp - timedelta(seconds=0.1)).isoformat(),
        'finished_at': (datetime.now(timezone.utc) + timedelta(seconds=0.1)).isoformat(),
        'command': ['bwrap', '--chdir', str(artifacts.recorded), '/usr/bin/python',
                    'tools/tutorials/verify_cellpose_apply.py',
                    '/tmp/spacr-tutorial-current/captures/cellpose_masks', '--device', 'cuda'],
        'reference_sha256': digest(reference),
        'dependencies': {str(path): digest(path) for path in dependencies}})


def alter_gate(artifacts, *, name=None, passed=False, approved=False, freeze=False, unknown=False):
    name = name or artifacts.names[0]
    source = artifacts.source
    path = source / f'{name}.gate.json'
    gate = get(path)
    gate.update(passed=passed, reasons=[] if passed else ['choppy (capture loss): 27.5 distinct fps < 27.9'],
                distinct_fps=29.9 if passed else 27.5,
                app_freeze_seconds=[0] if freeze else [], app_unknown_seconds=[0] if unknown else [],
                capture_loss_seconds=[] if passed else [0])
    if approved:
        gate['accepted_near_miss'] = 'capture loss'
    put(path, gate)
    summary = get(source / 'host_gate_summary.json')
    row = summary['clips'][name]
    row.update(passed=passed, reasons=gate['reasons'], distinct_fps=gate['distinct_fps'])
    if approved:
        row['accepted_near_miss'] = 'capture loss'
    summary.update(passed=passed, failed=[] if passed else [name])
    put(source / 'host_gate_summary.json', summary)
    frames = get(source / 'frames.json')
    frames[name]['clip']['gate'] = row
    receipt_path = source / frames[name]['clip']['receipt']
    receipt = get(receipt_path)
    receipt['gate'] = {'passed': passed, 'reasons': gate['reasons']}
    put(receipt_path, receipt)
    frames[name]['clip']['receipt_sha256'] = digest(receipt_path)
    put(source / 'frames.json', frames)
    refresh_verifier(artifacts)


def accepted(artifacts):
    return validator.validate(artifacts.args)


def catalogs(artifacts):
    folder = artifacts.tmp_path / 'catalog-baseline'
    (folder / 'catalog').mkdir(parents=True)
    lesson = get(artifacts.english)
    related = {item for scene in lesson['scenes'] for item in scene.get('related_lessons', [])}
    value = {'lessons': [lesson, *[{'id': item} for item in sorted(related)]]}
    for name in builder.CATALOGS:
        put(folder / 'catalog' / name, value)
    return folder


def test_scoped_acceptance_binds_all_bytes_and_preserves_unaccepted_raw_proof(artifacts):
    before = (artifacts.source / 'scientific_acceptance.json').read_bytes()
    output = accepted(artifacts)
    record = get(output)
    assert record['accepted'] is True and record['scope'] == validator.ACCEPTANCE_SCOPE
    assert all(record[name] is False for name in validator.SCOPE_FLAGS)
    assert record['native_observations'][0]['raw_observation']['accepted'] is False
    assert (artifacts.source / 'scientific_acceptance.json').read_bytes() == before
    assert {row['scene'] for row in record['selected_clips']} == set(artifacts.names)
    for row in record['selected_clips']:
        for suffix in ('mp4', 'native.mp4', 'capture.json', 'host.json', 'gate.json', 'pointer.json'):
            path = artifacts.source / f'{row["scene"]}.{suffix}'
            assert record['source_hashes'][str(path)] == digest(path)
        assert row['native_sha256'] == digest(artifacts.source / f'{row["scene"]}.native.mp4')


def test_relative_verifier_uses_recorded_checkout_from_another_cwd(artifacts, monkeypatch):
    other = artifacts.tmp_path / 'other-cwd'
    other.mkdir()
    monkeypatch.chdir(other)
    assert accepted(artifacts).is_file()


@pytest.mark.parametrize('kind', ['missing', 'pending', 'failed', 'freeze', 'unknown', 'gui_delay'])
def test_invalid_selected_clips_cannot_produce_acceptance(artifacts, kind):
    source, name = artifacts.source, artifacts.names[0]
    if kind in ('missing', 'pending'):
        frames = get(source / 'frames.json')
        if kind == 'missing':
            del frames[name]['clip']
        else:
            frames[name]['clip'] = {'pending_post_processing': True}
        put(source / 'frames.json', frames)
        refresh_verifier(artifacts)
    elif kind == 'freeze':
        alter_gate(artifacts, passed=True, freeze=True)
    else:
        alter_gate(artifacts, approved=kind != 'failed', unknown=kind == 'unknown')
        if kind == 'gui_delay':
            put(source / f'{name}.host.json', {'adapter_bypasses': [],
                'app_paint_samples': [[0, 0], [0.1, 3], [0.2, 6], [1.3, 39], [1.4, 42], [1.5, 45]]})
    with pytest.raises(ValueError):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


def test_explicit_near_miss_preserves_original_failure_in_scoped_record(artifacts):
    alter_gate(artifacts, approved=True)
    put(artifacts.source / 'provenance.json', {'module': 'cellpose_masks', 'completed_capture': False,
                                            'status': 'gate_failed'})
    refresh_verifier(artifacts)
    record = get(accepted(artifacts))
    row = record['selected_clips'][0]
    assert row['decision'] == 'approved_capture_loss_near_miss'
    assert row['original_gate_passed'] is False
    assert record['native_observations'][0]['original_provenance']['completed_capture'] is False
    assert record['native_observations'][0]['host_gate_summary']['failed'] == [artifacts.names[0]]


@pytest.mark.parametrize('kind', ['omitted', 'duplicate', 'absent'])
def test_selection_must_cover_every_unique_english_scene(artifacts, kind):
    names = artifacts.names.copy()
    if kind == 'omitted':
        names.pop()
    elif kind == 'duplicate':
        names.append(names[0])
    else:
        names[0] = 'not_a_recorded_scene'
    put(artifacts.picks, {str(artifacts.source): names})
    with pytest.raises(ValueError):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


@pytest.mark.parametrize('target', ['preview_cell.npy', 'batch_figure_00_arrays.npz', 'mask', 'original', 'source'])
def test_changed_bound_inputs_are_rejected(artifacts, target):
    path = {'mask': artifacts.work / 'masks/cell_pair_01.tif',
            'original': artifacts.tmp_path / 'originals/cell_pair_01.tif',
            'source': artifacts.recorded / 'spacr/qt/app.py'}.get(target, artifacts.source / target)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='digest changed'):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


@pytest.mark.parametrize('kind', ['failed', 'wrong_capture', 'no_prior_check', 'no_dependency_check',
                                 'missing_dependency', 'wrong_script', 'wrong_device', 'stale_reference'])
def test_verifier_receipt_paths_and_freshness_are_required(artifacts, kind):
    receipt = get(artifacts.verifier_path)
    if kind == 'failed':
        receipt['exit_code'] = 1
    elif kind == 'wrong_capture':
        receipt['capture'] = str(artifacts.tmp_path / 'different_capture')
    elif kind == 'no_prior_check':
        receipt['reference_absent_at_start'] = False
    elif kind == 'no_dependency_check':
        receipt['dependencies_unchanged'] = False
    elif kind == 'missing_dependency':
        del receipt['dependencies'][str(artifacts.source / 'preview_cell.npy')]
    elif kind == 'wrong_script':
        receipt['command'][-4] = 'elsewhere/verify_cellpose_apply.py'
    elif kind == 'wrong_device':
        receipt['command'][-1] = 'cpu'
    else:
        stamp = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
        os.utime(artifacts.source / 'independent_reference.json', (stamp, stamp))
    put(artifacts.verifier_path, receipt)
    with pytest.raises(ValueError):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


def test_raw_accuracy_claim_cannot_be_relabelled_as_scoped_acceptance(artifacts):
    proof_path = artifacts.source / 'scientific_acceptance.json'
    proof = get(proof_path)
    proof['accuracy_validated'] = True
    put(proof_path, proof)
    refresh_verifier(artifacts)
    with pytest.raises(ValueError, match='lack of accuracy'):
        accepted(artifacts)


def test_mutation_during_validation_is_detected_before_output(artifacts, monkeypatch):
    original = validator.check_clip

    def changing(evidence, source, name, frame, summary, reviews):
        value = original(evidence, source, name, frame, summary, reviews)
        if name == artifacts.names[-1]:
            (artifacts.work / 'masks/cell_pair_01.tif').write_bytes(b'changed after verification')
        return value

    monkeypatch.setattr(validator, 'check_clip', changing)
    with pytest.raises(ValueError, match='during validation'):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


def terminal_review(artifacts):
    name = '22b_actual_zoomed_filtered_preview'
    frames = get(artifacts.source / 'frames.json')
    frames[name]['dialogs'] = [{'title': 'Live settings', 'rect': [100, 100, 500, 500]}]
    put(artifacts.source / 'frames.json', frames)
    refresh_verifier(artifacts)
    image = artifacts.tmp_path / 'terminal-result.png'
    shutil.copyfile(artifacts.source / 'poster.png', image)
    value = {'capture': str(artifacts.source), 'scene': name,
             'reviewer': 'Codex AI visual review', 'reviewed_at': datetime.now(timezone.utc).isoformat(),
             'dialogs_covering_result': False, 'image': str(image), 'image_sha256': digest(image),
             'video_sha256': frames[name]['clip']['sha256'], 'time_seconds': 0.5,
             'command': ['ffmpeg', '-ss', '0.5', '-i', str(artifacts.source / f'{name}.mp4'),
                         '-frames:v', '1', str(image)]}
    path = artifacts.tmp_path / 'terminal-review.json'
    put(path, {'schema': 1, 'reviews': [value]})
    artifacts.args.terminal_review = path
    return path, value


def test_terminal_review_does_not_claim_the_whole_clip_is_unobstructed(artifacts):
    terminal_review(artifacts)
    record = get(accepted(artifacts))
    row = next(row for row in record['selected_clips']
               if row['scene'] == '22b_actual_zoomed_filtered_preview')
    assert row['original_start_frame_dialogs'][0]['title'] == 'Live settings'
    assert 'terminal frame only' in row['result_dialog_evidence']['scope']
    assert row['result_dialog_evidence']['review']['reviewer'] == 'Codex AI visual review'
    assert record['human_listening_review'] is False
    builder.preflight(artifacts.args.output, artifacts.english, [artifacts.source])


@pytest.mark.parametrize('kind', ['missing', 'covered', 'wrong_image_hash', 'wrong_video_hash',
                                 'wrong_input_command', 'wrong_time'])
def test_terminal_review_must_bind_the_actual_selected_result(artifacts, kind):
    path, review = terminal_review(artifacts)
    if kind == 'missing':
        artifacts.args.terminal_review = None
    elif kind == 'covered':
        review['dialogs_covering_result'] = True
    elif kind == 'wrong_image_hash':
        review['image_sha256'] = '0' * 64
    elif kind == 'wrong_video_hash':
        review['video_sha256'] = '0' * 64
    elif kind == 'wrong_input_command':
        review['command'][4] = str(artifacts.source / f'{artifacts.names[0]}.mp4')
    else:
        review['time_seconds'] = 2.0
    put(path, {'schema': 1, 'reviews': [review]})
    with pytest.raises(ValueError):
        accepted(artifacts)
    assert not artifacts.args.output.exists()


def test_staging_rechecks_and_attaches_only_scoped_acceptance(artifacts):
    acceptance = accepted(artifacts)
    stage = artifacts.tmp_path / 'new-stage'
    original = (artifacts.source / 'scientific_acceptance.json').read_bytes()
    builder.build_stage(stage, lesson_path=artifacts.english, acceptance_path=acceptance,
                        sources=[artifacts.stage], catalogs=catalogs(artifacts))
    record = get(stage / 'captures/cellpose_masks/scientific_acceptance.json')
    assert record['scope'] == validator.ACCEPTANCE_SCOPE and record['published'] is False
    assert (artifacts.source / 'scientific_acceptance.json').read_bytes() == original
    assert (stage / 'production/20_cellpose_masks/visual.json').is_file()


@pytest.mark.parametrize('kind', ['duplicate', 'omitted_scene', 'native_binding', 'native_declared',
                                 'scope', 'accuracy', 'changed_mask', 'missing_clip'])
def test_staging_refuses_invalid_acceptance_before_creating_destination(artifacts, kind):
    acceptance_path = accepted(artifacts)
    acceptance = get(acceptance_path)
    if kind == 'duplicate':
        acceptance['selected_clips'].append(acceptance['selected_clips'][0].copy())
    elif kind == 'omitted_scene':
        acceptance['selected_clips'].pop()
    elif kind == 'native_binding':
        del acceptance['source_hashes'][str(artifacts.source / f'{artifacts.names[0]}.native.mp4')]
    elif kind == 'native_declared':
        acceptance['selected_clips'][0]['native_sha256'] = '0' * 64
    elif kind == 'scope':
        acceptance['scope'] = 'Mask accuracy validated'
    elif kind == 'accuracy':
        acceptance['accuracy_validated'] = True
    elif kind == 'changed_mask':
        (artifacts.work / 'masks/cell_pair_01.tif').write_bytes(b'changed after acceptance')
    else:
        (artifacts.source / f'{artifacts.names[0]}.mp4').unlink()
    put(acceptance_path, acceptance)
    stage = artifacts.tmp_path / 'must-not-exist'
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.build_stage(stage, lesson_path=artifacts.english, acceptance_path=acceptance_path,
                            sources=[artifacts.stage], catalogs=catalogs(artifacts))
    assert not stage.exists()


def test_staging_cannot_use_a_different_explicit_selection(artifacts):
    acceptance = accepted(artifacts)
    stage = artifacts.tmp_path / 'new-stage'
    with pytest.raises(ValueError, match='selection differs'):
        builder.build_stage(stage, lesson_path=artifacts.english, acceptance_path=acceptance,
                            sources=[artifacts.stage], catalogs=catalogs(artifacts),
                            picks={str(artifacts.source): artifacts.names[:-1]})
    assert not stage.exists()


def test_staging_does_not_delete_a_preexisting_destination(artifacts):
    stage = artifacts.tmp_path / 'existing-stage'
    stage.mkdir()
    marker = stage / 'prior-artifact.txt'
    marker.write_bytes(b'preserve existing user work')
    with pytest.raises(ValueError, match='new destination'):
        builder.build_stage(stage, lesson_path=artifacts.english,
                            acceptance_path=artifacts.args.output,
                            sources=[artifacts.stage], catalogs=artifacts.tmp_path / 'unused')
    assert marker.read_bytes() == b'preserve existing user work'


def test_staging_failure_removes_only_its_new_destination(artifacts, monkeypatch):
    acceptance = accepted(artifacts)
    stage = artifacts.tmp_path / 'new-stage'
    original = (artifacts.source / 'frames.json').read_bytes()

    def fail(*_args, **_kwargs):
        raise ValueError('demonstrated downstream stage rejection')

    monkeypatch.setattr(builder, 'stage_lesson', fail)
    with pytest.raises(ValueError, match='downstream stage rejection'):
        builder.build_stage(stage, lesson_path=artifacts.english, acceptance_path=acceptance,
                            sources=[artifacts.stage], catalogs=catalogs(artifacts))
    assert not stage.exists()
    assert (artifacts.source / 'frames.json').read_bytes() == original
