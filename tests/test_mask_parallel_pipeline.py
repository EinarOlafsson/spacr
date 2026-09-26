"""Item 493: Mask generation dispatches prepared batches to GPU worker processes.

CPU stand-ins replace Cellpose inside real spawned workers, so these checks
cover partitioning, exclusive writes, resume and finalization without a GPU.
The GPU acceptance run is recorded in the item file.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from spacr import _mask_workers as mw
from spacr.checkpoint import fingerprint

DEVICES = [{'index': 0, 'name': 'Card A', 'memory_bytes': 1 << 30, 'backend': 'cuda'},
           {'index': 1, 'name': 'Card B', 'memory_bytes': 1 << 30, 'backend': 'cuda'}]


def _segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
    """Deterministic CPU segmenter that refuses to overwrite any output."""
    from spacr.cancellation import checkpoint

    assert run_qc is False
    root = Path(src) / f'{role}_mask_stack'
    root.mkdir(exist_ok=True)
    for path in batch_paths:
        checkpoint()
        time.sleep(0.05)
        if Path(path).name == settings.get('fail_on'):
            raise RuntimeError('inference failed')
        with open(Path(src).parent / 'calls.log', 'a') as log:
            log.write(f'{os.getpid()} {Path(path).name}\n')
        with np.load(path, allow_pickle=False) as data:
            for plane, name in zip(data['data'], data['filenames']):
                target = root / str(name)
                if target.exists():
                    raise RuntimeError(f'duplicate write {target}')
                np.save(target, (plane[..., 0] > 0).astype(np.uint16) * (1 + len(str(name))))
        on_batch_done(path)


def _plate(root, archives=4, fields=2):
    masks = root / 'masks'
    masks.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for index in range(archives):
        names = [f'plate1_A0{index}_{field}.npy' for field in range(fields)]
        np.savez(masks / f'batch{index}.npz',
                 data=rng.integers(0, 2, (fields, 8, 9, 1)).astype(np.float32),
                 filenames=np.array(names))
    return masks


def _settings(**extra):
    settings = {'cell_channel': 0, 'channels': [0], 'nucleus_channel': None,
                'pathogen_channel': None, 'verbose': False, 'plot': False}
    settings.update(extra)
    return settings


@pytest.fixture
def cpu_workers(monkeypatch):
    """Sign a stand-in model and count the shared QC pass in this process."""
    qc = []
    monkeypatch.setattr(mw, '_prepare_mask_model', lambda settings, role: (
        dict(settings), fingerprint({'model': 'cpu', 'role': role}),
        {'path': 'cpu-stand-in', 'sha256': '0' * 64, 'bytes': 1}))
    import spacr.object as sobj
    monkeypatch.setattr(sobj, '_run_seg_qc', lambda src, settings, role, **kw: qc.append((src, role, kw)))
    return qc


def _plan(settings):
    env = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
    import spacr.accelerator as acc
    original = acc._mask_worker_environment
    plan = mw._parallel_mask_plan(settings, devices=DEVICES)
    plan['environments'] = {index: original(index, backend='cuda', count=2, environment=env)
                            for index in plan['devices']}
    return plan


def _calls(root):
    path = root / 'calls.log'
    return [line.split() for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('value, expected', [
    (None, None), ('', None), (' all ', None), ('0,1', [0, 1]), ('1; 0', [1, 0]),
    ('0 2', [0, 2]), ([3, 1], [3, 1]), (2, [2])])
def test_gpu_selection_is_parsed_without_widening(value, expected):
    assert mw._selected_gpu_indices(value) == expected


@pytest.mark.parametrize('value', ['0,0', '-1', 'gpu0', [True], '1.5'])
def test_invalid_gpu_selection_is_refused(value):
    with pytest.raises(ValueError, match='mask_gpu_indices'):
        mw._selected_gpu_indices(value)


def test_the_feature_off_never_queries_devices(monkeypatch):
    monkeypatch.setattr(mw, '_compatible_mask_gpus', lambda: pytest.fail('queried GPUs'))
    for settings in ({}, {'mask_parallel': False}, {'mask_parallel': 'false'}):
        assert mw._parallel_mask_plan(settings) is None


def test_fewer_than_two_visible_gpus_keeps_the_ordinary_path(capsys):
    assert mw._parallel_mask_plan({'mask_parallel': True}, devices=DEVICES[:1]) is None
    assert mw._parallel_mask_plan({'mask_parallel': 'True'}, devices=[]) is None
    assert 'ordinary single-device' in capsys.readouterr().out


@pytest.mark.parametrize('settings, match', [
    ({'mask_gpu_indices': '0'}, 'two or more'),
    ({'mask_gpu_indices': '0,2'}, 'two or more'),
    ({'timelapse': True}, 'timelapse'),
    ({'t_stack': True}, 'timelapse or t_stack')])
def test_an_impossible_selection_is_refused_before_work(settings, match):
    with pytest.raises(ValueError, match=match):
        mw._parallel_mask_plan(dict(settings, mask_parallel=True), devices=DEVICES)


def test_each_selected_gpu_gets_its_own_isolated_environment(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-aaa,GPU-bbb,GPU-ccc')
    three = DEVICES + [dict(DEVICES[0], index=2, name='Card C')]
    plan = mw._parallel_mask_plan({'mask_parallel': True, 'mask_gpu_indices': '2,0'},
                                  devices=three)
    assert plan['devices'] == [2, 0]
    assert plan['environments'][2]['CUDA_VISIBLE_DEVICES'] == 'GPU-ccc'
    assert plan['environments'][0]['CUDA_VISIBLE_DEVICES'] == 'GPU-aaa'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-aaa,GPU-bbb,GPU-ccc'
    everything = mw._parallel_mask_plan({'mask_parallel': True}, devices=three)
    assert everything['devices'] == [0, 1, 2]


def test_parallel_batches_match_the_serial_run_and_write_each_field_once(
        tmp_path, cpu_workers, capsys):
    serial = _plate(tmp_path / 'serial')
    _segment(str(serial), {}, 'cell', batch_paths=sorted(serial.glob('*.npz')),
             on_batch_done=lambda path: None, run_qc=False)
    masks = _plate(tmp_path / 'parallel')
    settings = _settings(mask_parallel=True)
    plan = _plan(settings)

    mw._generate_masks_in_parallel(str(masks), settings, 'cell', plan, segmenter=_segment)

    serial_out = sorted((serial / 'cell_mask_stack').glob('*.npy'))
    parallel_out = sorted((masks / 'cell_mask_stack').glob('*.npy'))
    assert [path.name for path in serial_out] == [path.name for path in parallel_out]
    assert len(parallel_out) == 8
    for left, right in zip(serial_out, parallel_out):
        assert left.read_bytes() == right.read_bytes()
    calls = _calls(tmp_path / 'parallel')
    assert sorted(name for _, name in calls) == [f'batch{index}.npz' for index in range(4)]
    assert len({pid for pid, _ in calls}) == 2
    assert cpu_workers == [(str(masks), 'cell', {})]
    ledger = json.loads((masks / '.mask-workers-cell.json').read_text())
    assert ledger['status'] == 'finalized' and len(ledger['completed']) == 4
    assert settings['src'] == str(masks) and settings['cellpose_cell_channel'] == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith('[mask GPUs]')]
    from spacr.qt.bridge import _PROGRESS_RE, mask_gpu_progress
    final = mask_gpu_progress(lines[-1])
    assert final['done'] == final['total'] == 4 and final['failed'] == 0
    assert sorted(device for device, *_ in final['workers']) == ['0', '1']
    assert all(state == 'success' and done == total == 2
               for _, state, done, total in final['workers'])
    assert _PROGRESS_RE.search(lines[-1]).groups() == ('4', '4')


def test_an_interrupted_worker_resumes_only_unfinished_batches(tmp_path, cpu_workers, capsys):
    masks = _plate(tmp_path, archives=4)
    settings = _settings(mask_parallel=True, fail_on='batch3.npz')
    plan = _plan(settings)
    with pytest.raises(RuntimeError, match='inference failed'):
        mw._generate_masks_in_parallel(str(masks), settings, 'cell', plan, segmenter=_segment)
    out = capsys.readouterr().out
    assert 'failed 1' in out and 'unfinished of 4 archives' in out
    assert 'every prepared input are kept' in out
    assert len(list(masks.glob('*.npz'))) == 4
    assert cpu_workers == []
    done_before = {path.name: path.stat().st_mtime_ns
                   for path in (masks / 'cell_mask_stack').glob('*.npy')}
    first = [name for _, name in _calls(tmp_path)]
    assert 'batch3.npz' not in first
    assert json.loads((masks / '.mask-workers-cell.json').read_text())['status'] == 'failed'

    settings.pop('fail_on')
    mw._generate_masks_in_parallel(str(masks), settings, 'cell', plan, segmenter=_segment)
    second = [name for _, name in _calls(tmp_path)][len(first):]
    assert sorted(first + second) == [f'batch{index}.npz' for index in range(4)]
    assert 'batch3.npz' in second and not set(first) & set(second)
    for path in (masks / 'cell_mask_stack').glob('*.npy'):
        if path.name in done_before:
            assert path.stat().st_mtime_ns == done_before[path.name]
    assert len(list((masks / 'cell_mask_stack').glob('*.npy'))) == 8
    assert cpu_workers == [(str(masks), 'cell', {})]


def test_changed_settings_cannot_claim_a_previous_runs_masks(tmp_path, cpu_workers, monkeypatch):
    masks = _plate(tmp_path, archives=2)
    settings = _settings(mask_parallel=True)
    mw._generate_masks_in_parallel(str(masks), settings, 'cell', _plan(settings), segmenter=_segment)
    monkeypatch.setattr(mw, '_prepare_mask_model', lambda settings, role: (
        dict(settings), fingerprint({'model': 'other'}), {'path': 'other'}))
    from spacr.checkpoint import CheckpointMismatch
    with pytest.raises(CheckpointMismatch):
        mw._generate_masks_in_parallel(str(masks), settings, 'cell', _plan(settings),
                                       segmenter=_segment)


def _roles(root):
    masks = root / 'masks'
    cell = np.zeros((20, 20), np.uint16)
    cell[2:9, 2:9] = 7
    cell[9:16, 2:9] = 300
    nucleus = np.zeros_like(cell)
    nucleus[4:7, 4:7] = 1
    parasite = np.zeros_like(cell)
    parasite[7:11, 4:7] = 1
    for role, array in (('cell', cell), ('nucleus', nucleus), ('pathogen', parasite)):
        (masks / f'{role}_mask_stack').mkdir(parents=True, exist_ok=True)
        for name in ('f0.npy', 'f1.npy'):
            np.save(masks / f'{role}_mask_stack' / name, array)
    return masks


def test_adjusted_cells_are_finalized_beside_raw_masks_with_provenance(tmp_path, monkeypatch, capsys):
    import spacr.utils as su
    masks = _roles(tmp_path)
    raw = {path: path.read_bytes() for path in masks.rglob('*.npy')}
    calls = []
    original = su.adjust_cell_masks
    monkeypatch.setattr(su, 'adjust_cell_masks',
                        lambda *args, **kwargs: calls.append(kwargs) or original(*args, **kwargs))

    output = Path(mw._finalize_adjusted_cells(str(masks), n_jobs=1))
    assert output == masks / 'adjusted_cell_mask_stack'
    assert calls[0]['output_folder'] == str(output)
    assert all(path.read_bytes() == data for path, data in raw.items())
    adjusted = np.load(output / 'f0.npy')
    assert len(np.unique(adjusted)) == 2, 'the pathogen bridging both cells merged them'
    record = json.loads((masks / '.adjusted-cell-mask-provenance.json').read_text())
    assert set(record['outputs']) == {'f0.npy', 'f1.npy'}
    assert set(record['raw_masks']['raw']) == {'cell', 'nucleus', 'pathogen'}

    mw._finalize_adjusted_cells(str(masks), n_jobs=1)
    assert len(calls) == 1 and 'verified' in capsys.readouterr().out

    (output / 'f1.npy').write_bytes(b'truncated')
    mw._finalize_adjusted_cells(str(masks), n_jobs=1)
    assert len(calls) == 2
    np.testing.assert_array_equal(np.load(output / 'f1.npy'), adjusted)

    np.save(masks / 'nucleus_mask_stack' / 'f1.npy', np.zeros((20, 20), np.uint16))
    mw._finalize_adjusted_cells(str(masks), n_jobs=1)
    assert len(calls) == 3


def test_an_interrupted_adjustment_leaves_no_provenance_claiming_it(tmp_path, monkeypatch):
    import spacr.utils as su
    masks = _roles(tmp_path)
    mw._finalize_adjusted_cells(str(masks), n_jobs=1)
    np.save(masks / 'cell_mask_stack' / 'f0.npy', np.zeros((20, 20), np.uint16))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(su, 'adjust_cell_masks', interrupted)
    with pytest.raises(KeyboardInterrupt):
        mw._finalize_adjusted_cells(str(masks), n_jobs=1)
    assert not (masks / '.adjusted-cell-mask-provenance.json').exists()


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.fixture
def core_stubs(monkeypatch, tmp_path):
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.utils as su
    spies = {name: _Spy() for name in ('cellpose', 'concat', 'adjust', 'qc', 'parallel', 'finalize')}
    monkeypatch.setattr(sobj, 'generate_cellpose_masks_sam', spies['cellpose'])
    monkeypatch.setattr(sobj, '_run_seg_qc', spies['qc'])
    monkeypatch.setattr(sio, '_load_and_concatenate_arrays', spies['concat'])
    monkeypatch.setattr(su, 'adjust_cell_masks', spies['adjust'])
    monkeypatch.setattr(su, '_pivot_counts_table', _Spy())
    monkeypatch.setattr(su, 'cleanup_pipeline_folders', lambda *a, **k: [])
    monkeypatch.setattr(mw, '_generate_masks_in_parallel', spies['parallel'])
    monkeypatch.setattr(mw, '_finalize_adjusted_cells',
                        lambda *args, **kwargs: spies['finalize'](*args, **kwargs) or 'ADJUSTED')
    src = tmp_path / 'plate1'
    (src / 'stack').mkdir(parents=True)
    for index in range(2):
        np.save(src / 'stack' / f'f{index}.npy', np.zeros((3, 8, 8), np.uint16))
    return spies, src


def _core_settings(src, **extra):
    settings = {'src': str(src), 'metadata_type': 'cellvoyager', 'channels': [0, 1, 2],
                'cell_channel': 1, 'nucleus_channel': 0, 'pathogen_channel': 2,
                'organelle_channel': None, 'preprocess': False, 'masks': True,
                'plot': False, 'verbose': False, 'test_mode': False, 'timelapse': False,
                'n_jobs': 1, 'adjust_cells': True, 'consolidate': False, 'batch_size': 10,
                'save': True, 'custom_regex': None, 'randomize': True, 'examples_to_plot': 2}
    settings.update(extra)
    return settings


@pytest.mark.parametrize('extra', [{}, {'mask_parallel': False, 'mask_gpu_indices': '0,1'}])
def test_with_the_feature_off_core_calls_are_unchanged(core_stubs, monkeypatch, extra):
    from spacr.core import preprocess_generate_masks
    spies, src = core_stubs
    monkeypatch.setattr(mw, '_compatible_mask_gpus', lambda: pytest.fail('queried GPUs'))
    preprocess_generate_masks(_core_settings(src, **extra))
    mask_src = os.path.join(str(src), 'masks')
    assert [(call[0][0], call[0][2], call[1]) for call in spies['cellpose'].calls] == [
        (mask_src, 'cell', {}), (mask_src, 'nucleus', {}), (mask_src, 'pathogen', {})]
    assert [len(args) for args, _ in spies['cellpose'].calls] == [3, 3, 3]
    assert spies['adjust'].calls[0][0][:3] == tuple(
        os.path.join(mask_src, f'{role}_mask_stack') for role in ('pathogen', 'cell', 'nucleus'))
    assert 'output_folder' not in spies['adjust'].calls[0][1]
    assert spies['qc'].calls == [((mask_src, spies['qc'].calls[0][0][1], 'cell'), {})]
    assert 'mask_folders' not in spies['concat'].calls[0][1]
    assert not spies['parallel'].calls and not spies['finalize'].calls


def test_with_the_feature_on_core_dispatches_every_role_and_finalizes(core_stubs, monkeypatch):
    from spacr.core import preprocess_generate_masks
    spies, src = core_stubs
    monkeypatch.setattr(mw, '_compatible_mask_gpus', lambda: tuple(DEVICES))
    preprocess_generate_masks(_core_settings(src, mask_parallel=True))
    mask_src = os.path.join(str(src), 'masks')
    assert not spies['cellpose'].calls and not spies['adjust'].calls
    assert [(args[0], args[2], args[3]['devices']) for args, _ in spies['parallel'].calls] == [
        (mask_src, role, [0, 1]) for role in ('cell', 'nucleus', 'pathogen')]
    assert spies['finalize'].calls == [((mask_src, None), {'n_jobs': 1})]
    assert spies['qc'].calls[0][1] == {'mask_folder': 'ADJUSTED'}
    assert spies['concat'].calls[0][1]['mask_folders'] == {'cell': 'ADJUSTED'}
