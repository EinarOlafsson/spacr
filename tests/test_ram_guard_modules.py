"""Every module that starts workers sizes them against free RAM.

RAM comes from a fake psutil and pools are replaced by fakes, so no test
depends on this machine's memory, starts a real worker or signals a process.
"""
from __future__ import annotations

import inspect
import types

import numpy as np
import pytest

from spacr import resource_log

GIB = 1024 ** 3


class _FakePsutil:
    def __init__(self, available, total):
        self._memory = types.SimpleNamespace(available=available, total=total)

    def virtual_memory(self):
        return self._memory


def test_the_plan_clamps_to_what_fits_and_keeps_the_reserve():
    fake = _FakePsutil(20 * GIB, 32 * GIB)
    plan = resource_log._ram_plan(GIB, 30, module='sweep', psutil_module=fake)
    assert plan['per_worker'] == GIB
    assert plan['max_safe'] == 16
    assert plan['exceeds'] and plan['module'] == 'sweep'
    assert resource_log._ram_plan(0, 30, psutil_module=fake) is None


def test_each_module_uses_its_own_multiplier():
    fake = _FakePsutil(20 * GIB, 32 * GIB)
    adjust = resource_log._ram_plan(GIB // 16, 4, module='adjust_masks',
                                    psutil_module=fake)
    assert adjust['per_worker'] == int(GIB // 16 * 14.0)


def test_headless_clamps_with_a_warning_unless_the_guard_is_off(capsys):
    fake = _FakePsutil(10 * GIB, 16 * GIB)
    assert resource_log._guard_workers('sweep', 32, 2 * GIB,
                                       psutil_module=fake) == 4
    assert 'sweep: 32 workers' in capsys.readouterr().out
    assert resource_log._guard_workers('sweep', 32, 2 * GIB,
                                       settings={'ram_guard': False},
                                       psutil_module=fake) == 32
    assert resource_log._guard_workers('sweep', 3, 2 * GIB,
                                       psutil_module=fake) == 3
    assert resource_log._guard_workers('sweep', -1, 0,
                                       psutil_module=fake) == -1


def test_the_run_scope_carries_the_setting_to_helpers_without_settings():
    fake = _FakePsutil(10 * GIB, 16 * GIB)
    with resource_log._ram_guard_scope({'ram_guard': False}):
        assert not resource_log._ram_guard_enabled()
        assert resource_log._guard_workers('sweep', 32, 2 * GIB,
                                           psutil_module=fake) == 32
    assert resource_log._ram_guard_enabled()
    assert resource_log._ram_guard_enabled({'ram_guard': True})


def test_minus_one_means_every_core(monkeypatch):
    monkeypatch.setattr(resource_log.os, 'cpu_count', lambda: 8)
    assert resource_log._requested_workers(-1) == 8
    assert resource_log._requested_workers(-2) == 7
    assert resource_log._requested_workers(None) == 8
    assert resource_log._requested_workers(3) == 3


def test_input_units_are_sized_from_their_headers(tmp_path):
    np.save(tmp_path / 'a.npy', np.zeros((64, 64, 3), np.float32))
    np.savez(tmp_path / 'b.npz', x=np.zeros((10, 10), np.uint8),
             y=np.zeros(5, np.int64))
    assert resource_log._array_file_nbytes(tmp_path / 'a.npy') == 64 * 64 * 12
    assert resource_log._array_file_nbytes(tmp_path / 'b.npz') == 140
    assert resource_log._array_file_nbytes(tmp_path / 'missing') == 0
    assert resource_log._array_file_nbytes(('not', 'a path')) == 0


def test_a_sample_input_is_found_under_the_source(tmp_path):
    (tmp_path / 'plate' / 'deep').mkdir(parents=True)
    np.save(tmp_path / 'plate' / 'deep' / 'f.npy', np.zeros(4))
    found = resource_log._sample_input_file(str(tmp_path / 'plate'), ('.npy',))
    assert found.endswith('f.npy')
    assert resource_log._sample_input_file(str(tmp_path), ('.png',)) is None


@pytest.mark.parametrize('app_key, module', [
    ('mask', 'mask'), ('timelapse', 'mask'), ('motility', 'motility'),
    ('train_cellpose', 'cellpose_dataset')])
def test_the_gui_estimate_reads_each_modules_input(tmp_path, app_key, module):
    merged = tmp_path / 'merged'
    merged.mkdir()
    np.save(merged / 'f.npy', np.zeros((256, 256, 4), np.float32))
    fake = _FakePsutil(4 * GIB, 8 * GIB)
    plan = resource_log._app_ram_plan(app_key, {'src': str(tmp_path)}, 2000,
                                      psutil_module=fake)
    assert plan['module'] == module
    assert plan['nbytes'] == 256 * 256 * 16
    assert plan['exceeds']


def test_map_barcodes_is_sized_from_its_chunk():
    fake = _FakePsutil(4 * GIB, 8 * GIB)
    plan = resource_log._app_ram_plan('map_barcodes', {'chunk_size': 1000},
                                      2, psutil_module=fake)
    assert plan['nbytes'] == 1000 * 2 * 1024


def test_a_module_without_workers_has_no_plan():
    assert resource_log._app_ram_plan('plate_view', {'src': '.'}, 4) is None


GUARDED = [
    ('spacr.measure', '_ram_guard_plan', "'measure'"),
    ('spacr.utils', 'adjust_cell_masks', "'adjust_masks'"),
    ('spacr.utils', 'merge_split_objects', "'merge_split'"),
    ('spacr.utils', 'augment_images', "'augment'"),
    ('spacr.utils', 'reduction_and_clustering', "'umap'"),
    ('spacr.utils', 'search_reduction_and_clustering', "'umap'"),
    ('spacr.object', '_segment_classical_parallel', "'classical_masks'"),
    ('spacr.timelapse', 'automated_motility_assay', "'motility'"),
    ('spacr.timelapse', '_btrack_track_cells', "'mask'"),
    ('spacr.io', 'prepare_cellpose_dataset', "'cellpose_dataset'"),
    ('spacr.io', 'generate_dataset', "'dataset'"),
    ('spacr.io', 'generate_loaders', "'classify'"),
    ('spacr.io', 'generate_cv_loaders', "'classify'"),
    ('spacr.deep_spacr', 'apply_model', "'classify'"),
    ('spacr.deep_spacr', 'apply_model_to_tar', "'classify'"),
    ('spacr.deep_spacr', 'generate_activation_map', "'classify'"),
    ('spacr.sequencing', 'paired_read_chunked_processing', "'map_barcodes'"),
    ('spacr.sequencing', 'single_read_chunked_processing', "'map_barcodes'"),
    ('spacr.sim', 'run_multiple_simulations', "'simulation'"),
    ('spacr.ml', 'ml_analysis', "'ml_analyze'"),
    ('spacr.hyperparam', 'run_search_for_app', '"regression"'),
    ('spacr.parameter_sweep', 'run_sweep_parallel', '"sweep"'),
    ('spacr.ops_engine', '_decode', '"ops_decode"'),
]


@pytest.mark.parametrize('module_name, function, key', GUARDED)
def test_every_pool_site_calls_the_guard(module_name, function, key):
    import importlib
    module = importlib.import_module(module_name)
    source = inspect.getsource(getattr(module, function))
    assert '_guard_workers(' in source or '_ram_plan(' in source
    assert key in source


def test_every_pool_site_is_listed():
    import ast
    import pathlib
    import re
    root = pathlib.Path(resource_log.__file__).parent
    pattern = re.compile(r'\b(Pool|ProcessPoolExecutor)\(|Parallel\(n_jobs')
    sites = {path.stem for path in root.glob('*.py')
             if pattern.search(path.read_text(encoding='utf-8'))}
    listed = {name.split('.')[1] for name, _f, _k in GUARDED}
    tree = ast.parse(inspect.getsource(resource_log))
    factories = {
        function.name
        for function in ast.walk(tree) if isinstance(function, ast.FunctionDef)
        for call in ast.walk(function) if isinstance(call, ast.Call)
        if (isinstance(call.func, ast.Name) and call.func.id in (
            'Pool', 'ProcessPoolExecutor')) or (
                isinstance(call.func, ast.Attribute) and call.func.attr == 'Pool')
    }
    assert factories == {'Pool', '_parallel_pool', '_parallel_process_executor'}
    assert sites <= listed | {'example_archives', '_mask_workers', 'resource_log'}


def test_classical_segmentation_starts_only_the_safe_count(monkeypatch):
    from spacr import object as spacr_object
    calls = {}

    def fake_guard(module, n_jobs, unit_bytes, **_kw):
        calls['guard'] = (module, n_jobs, unit_bytes)
        return 2

    class FakePool:
        def __init__(self, processes):
            calls['pool'] = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return [None for _ in items]

    monkeypatch.setattr(resource_log, '_guard_workers', fake_guard)
    monkeypatch.setattr(spacr_object, 'Pool', FakePool)
    batch = np.zeros((3, 8, 8), np.float32)
    spacr_object._segment_classical_parallel(batch, {}, n_jobs=3)
    assert calls['guard'][0] == 'classical_masks'
    assert calls['guard'][2] == 8 * 8 * 4
    assert calls['pool'] == 2


def test_augmentation_starts_only_the_safe_count(monkeypatch, tmp_path):
    from spacr import utils
    seen = {}

    class FakePool:
        def __init__(self, processes):
            seen['pool'] = processes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, fn, items):
            return []

        def close(self):
            seen['closed'] = True

        def join(self):
            seen['joined'] = seen.get('closed', False)

    monkeypatch.setattr(resource_log, '_guard_workers',
                        lambda module, n, unit, **kw: seen.setdefault(
                            'guard', module) and 1)
    context = object()
    monkeypatch.setattr(utils, '_augment_pool_context', lambda: context)
    def make_pool(processes, *, context):
        seen['context'] = context
        return FakePool(processes)
    monkeypatch.setattr(utils, 'Pool', make_pool)
    utils.augment_images([str(tmp_path / 'a.png')], str(tmp_path / 'out'))
    assert seen['guard'] == 'augment'
    assert seen['pool'] == 1
    assert seen['joined'] is True
    assert seen['context'] is context


def test_augmentation_never_starts_more_workers_than_images(monkeypatch, tmp_path):
    from spacr import utils
    seen = {}

    class FakePool:
        def __init__(self, processes):
            seen['pool'] = processes

        def map(self, fn, items):
            raise RuntimeError('worker failed')

        def close(self):
            seen['closed'] = True

        def join(self):
            seen['joined'] = seen.get('closed', False)

    monkeypatch.setattr(resource_log, '_guard_workers',
                        lambda module, n, unit, **kw: 16)
    context = object()
    monkeypatch.setattr(utils, '_augment_pool_context', lambda: context)
    def make_pool(processes, *, context):
        seen['context'] = context
        return FakePool(processes)
    monkeypatch.setattr(utils, 'Pool', make_pool)
    with pytest.raises(RuntimeError, match='worker failed'):
        utils.augment_images([str(tmp_path / 'a.png'), str(tmp_path / 'b.png')],
                             str(tmp_path / 'out'))
    assert seen['pool'] == 2
    assert seen['joined'] is True
    assert seen['context'] is context


def test_augment_pool_context_spawns():
    from spacr import utils
    assert utils._augment_pool_context().get_start_method() == 'spawn'
