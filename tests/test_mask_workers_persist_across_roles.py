"""Item 801: mask workers persist across the roles and plates of one run.

No GPU is used. CPU stand-in segmenters run inside real spawned workers,
a fake Cellpose model stands in for Cellpose-SAM in this process, and a
stand-in reader replaces NVML for the utilisation record.
"""

import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from spacr import _mask_workers as mw
from tests.test_cov_object_masks_sam import _settings as sam_settings
from tests.test_cov_object_masks_sam import _write_npz
from tests.test_cov_object_masks_sam import fake_model, force_cpu  # noqa: F401
from tests.test_mask_parallel_pipeline import (
    _calls, _plate, _segment, _settings, cpu_workers,  # noqa: F401
)

GB = 1 << 30
CARD = {'index': 0, 'name': 'Card A', 'memory_bytes': 24 * GB, 'backend': 'cuda',
        'uuid': 'GPU-aaa'}
ROLES = ('cell', 'nucleus')


@pytest.fixture(autouse=True)
def _unrestricted_gpu_visibility(monkeypatch):
    """The stand-in card is visible: no visibility list hides it."""
    for key in ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fresh_torch(monkeypatch):
    """A torch whose CUDA runtime is not initialized, as in a new worker process."""
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: False)))


@pytest.fixture
def sampled(monkeypatch):
    """A stand-in NVML reader at 90 % busy, sampled every 10 ms."""
    monkeypatch.setattr(mw, '_SAMPLE_SECONDS', 0.01)
    monkeypatch.setattr(mw, '_nvml_reader', lambda gpus: (lambda: {0: (90.0, 6 * GB)}))


class _Gate:
    """Start gate that starts at once and records which starts were paced."""

    def __init__(self):
        self.paced = 0
        self.unpaced = 0

    def start(self, call):
        self.paced += 1
        return call()

    def start_now(self, call):
        self.unpaced += 1
        return call()


def _timed_segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
    """The item-493 CPU segmenter, logging each job's role, start and end."""
    names = [Path(path).name for path in batch_paths]
    start = time.time()
    marker = Path(src).parent / f'overloaded-{os.getpid()}'
    if role == 'cell' and settings.get('slow') in names:
        time.sleep(1.5)
    if role == settings.get('crash_role') and settings.get('crash_on') in names:
        os._exit(7)
    if (role == 'cell' and settings.get('overload_on') in names
            and not (Path(src).parent / 'overloaded').exists()):
        (Path(src).parent / 'overloaded').write_text('once', encoding='utf-8')
        marker.write_text('primary', encoding='utf-8')
        raise MemoryError('primary buffer exhausted')
    _segment(src, settings, role, batch_paths=batch_paths, on_batch_done=on_batch_done,
             run_qc=run_qc)
    with open(Path(src).parent / 'jobs.log', 'a', encoding='utf-8') as log:
        log.write(f'{os.getpid()} {role} {start} {time.time()}\n')


def _jobs(plate):
    """``(pid, role, start, end)`` of every job the workers finished."""
    path = plate / 'jobs.log'
    rows = [line.split() for line in path.read_text(encoding='utf-8').splitlines()]
    return [(int(pid), role, float(start), float(end)) for pid, role, start, end in rows]


def _shared_plan(settings):
    """Two workers sharing the one stand-in card, with CPU-only environments."""
    plan = mw._parallel_mask_plan(settings, devices=[CARD])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    plan['environments'] = {worker: dict(env) for worker in plan['devices']}
    return plan


def _serial(root):
    """Both roles segmented in one process, the reference outputs."""
    serial = _plate(root)
    for role in ROLES:
        _segment(str(serial), {}, role, batch_paths=sorted(serial.glob('*.npz')),
                 on_batch_done=lambda path: None, run_qc=False)
    return serial


def _same_masks(left, right, role):
    """Assert two mask folders hold the same files with the same bytes."""
    left_out = sorted((left / f'{role}_mask_stack').glob('*.npy'))
    right_out = sorted((right / f'{role}_mask_stack').glob('*.npy'))
    assert [path.name for path in left_out] == [path.name for path in right_out]
    assert len(left_out) == 8
    for one, other in zip(left_out, right_out):
        assert one.read_bytes() == other.read_bytes()


def _no_workers_left():
    return not [child for child in multiprocessing.active_children()
                if child.name.startswith('spacr-mask-gpu-')]


def test_workers_spawn_once_for_every_role_and_plate(tmp_path, cpu_workers, sampled, capsys):
    serial = _serial(tmp_path / 'serial' / 'plate1')
    settings = _settings(mask_workers_per_gpu=2)
    plan = _shared_plan(settings)
    plates = [_plate(tmp_path / 'pool' / f'plate{index}') for index in (1, 2)]
    gate = _Gate()
    with mw._MaskWorkerPool(segmenter=_timed_segment, gate=gate) as pool:
        for masks in plates:
            seconds = mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                                     plan, pool=pool)
            assert set(seconds) == set(ROLES) and all(value > 0 for value in seconds.values())
        assert pool.spawns == 2 and gate.paced == 2 and gate.unpaced == 0
        pids = {pid for masks in plates for pid, *_ in _jobs(masks.parent)}
        assert len(pids) == 2
        assert all(process.is_alive() for process in pool._processes.values())
    assert _no_workers_left()
    for masks in plates:
        for role in ROLES:
            _same_masks(serial, masks, role)
            ledger = json.loads((masks / f'.mask-workers-{role}.json').read_text(encoding='utf-8'))
            assert ledger['status'] == 'finalized' and len(ledger['completed']) == 4
        assert sorted(name for _, name in _calls(masks.parent)) == sorted(
            f'batch{index}.npz' for index in range(4) for _ in ROLES)
        record = json.loads((masks.parent / 'gpu_utilisation.json').read_text(encoding='utf-8'))
        for role in ROLES:
            row = record[role]['0']
            assert row['workers'] == 2 and isinstance(row['started_at'], float)
            assert row['started_at_iso'] and row['startup_seconds'] is not None
            assert [label for _, label in row['events']].count('archive finished') == 4
        assert record['nucleus']['0']['started_at'] >= record['cell']['0']['started_at']
    assert cpu_workers == [(str(masks), role, {}) for masks in plates for role in ROLES]
    out = capsys.readouterr().out
    assert 'cell masks: GPU 0 with 2 workers busy 90% on average' in out
    assert 'after the first archive finished at' in out


def test_a_worker_starts_the_next_role_while_its_peer_finishes(tmp_path, cpu_workers):
    masks = _plate(tmp_path / 'plate1')
    settings = _settings(mask_workers_per_gpu=2, slow='batch0.npz')
    with mw._MaskWorkerPool(segmenter=_timed_segment, gate=_Gate()) as pool:
        mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                       _shared_plan(settings), pool=pool)
    jobs = _jobs(masks.parent)
    last_cell = max(end for _, role, _, end in jobs if role == 'cell')
    first_nucleus = min(start for _, role, start, _ in jobs if role == 'nucleus')
    assert first_nucleus < last_cell
    assert cpu_workers == [(str(masks), role, {}) for role in ROLES]
    assert _no_workers_left()


def test_flagging_qc_runs_the_roles_one_after_another_in_the_same_workers(
        tmp_path, cpu_workers):
    masks = _plate(tmp_path / 'plate1')
    settings = _settings(mask_workers_per_gpu=2, slow='batch0.npz', seg_qc='flag')
    gate = _Gate()
    with mw._MaskWorkerPool(segmenter=_timed_segment, gate=gate) as pool:
        mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                       _shared_plan(settings), pool=pool)
        assert pool.spawns == 2
    jobs = _jobs(masks.parent)
    last_cell = max(end for _, role, _, end in jobs if role == 'cell')
    assert min(start for _, role, start, _ in jobs if role == 'nucleus') >= last_cell
    assert len({pid for pid, *_ in jobs}) == 2


def test_overlapped_roles_are_signed_with_the_settings_of_a_serial_run(tmp_path, monkeypatch):
    """The next role's contract sees what finalizing the previous role left."""
    import spacr.object as sobj

    signed = []
    monkeypatch.setattr(mw, '_mask_model_contract', lambda settings, role: (
        signed.append((role, dict(settings))) or (dict(settings), '0' * 64,
                                                  {'path': 'cpu'})))
    monkeypatch.setattr(sobj, '_run_seg_qc', lambda *args, **kwargs: None)

    class _Pool:
        def run_roles(self, roles, environments, *, on_role_done=None):
            for index, _ in enumerate(roles):
                on_role_done(index, {})
            return [{} for _ in roles]

    masks = _plate(tmp_path / 'plate1')
    settings = _settings(mask_workers_per_gpu=2, src=str(tmp_path / 'plate1'))
    serial_settings = dict(settings)
    mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                   _shared_plan(settings), pool=_Pool())
    expected = mw._settings_after_role(serial_settings, str(masks))
    assert signed[0][1]['src'] == str(tmp_path / 'plate1')
    assert signed[1][1] == expected
    assert settings == expected


def test_a_worker_dying_mid_role_fails_the_run_and_still_reports(
        tmp_path, cpu_workers, sampled, capsys):
    masks = _plate(tmp_path / 'plate1')
    settings = _settings(mask_workers_per_gpu=2, crash_role='cell', crash_on='batch1.npz')
    with pytest.raises(RuntimeError, match=r'cell GPU 0\.\d: Worker exited with code 7'):
        mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                       _shared_plan(settings), segmenter=_timed_segment)
    assert _no_workers_left()
    assert len(list(masks.glob('*.npz'))) == 4
    assert cpu_workers == []
    for role in ROLES:
        ledger = json.loads((masks / f'.mask-workers-{role}.json').read_text(encoding='utf-8'))
        assert ledger['status'] == 'failed'
    out = capsys.readouterr().out
    assert 'Parallel cell masks stopped:' in out and 'Parallel nucleus masks stopped:' in out
    record = json.loads((masks.parent / 'gpu_utilisation.json').read_text(encoding='utf-8'))
    labels = [label for _, label in record['cell']['0']['events']]
    assert any(label.endswith(' failed') for label in labels)


def test_an_overloaded_worker_restarts_alone_then_takes_its_next_role(tmp_path, cpu_workers):
    serial = _serial(tmp_path / 'serial' / 'plate1')
    masks = _plate(tmp_path / 'pool' / 'plate1')
    settings = _settings(mask_workers_per_gpu=2, overload_on='batch0.npz')
    gate = _Gate()
    with mw._MaskWorkerPool(segmenter=_timed_segment, gate=gate) as pool:
        mw._generate_roles_in_parallel(str(masks), settings, list(ROLES),
                                       _shared_plan(settings), pool=pool)
        assert pool.spawns == 3 and gate.paced == 2 and gate.unpaced == 1
    for role in ROLES:
        _same_masks(serial, masks, role)
    jobs = _jobs(masks.parent)
    overloaded = int(next(masks.parent.glob('overloaded-*')).name.split('-')[1])
    cell_jobs = sorted((job for job in jobs if job[1] == 'cell'), key=lambda job: job[2])
    retry = cell_jobs[-1]
    assert len(cell_jobs) == 2 and retry[0] not in (overloaded, cell_jobs[0][0])
    others = [job for job in jobs if job is not retry]
    assert all(end <= retry[2] or start >= retry[3] for _, _, start, end in others)
    assert cpu_workers == [(str(masks), role, {}) for role in ROLES]


def test_bad_assignments_fail_before_any_worker_starts(tmp_path):
    pool = mw._MaskWorkerPool(segmenter=_timed_segment, gate=_Gate())
    role = {'src': str(tmp_path), 'settings': {}, 'object_type': 'cell',
            'assignments': {'0.1': [str(tmp_path / 'a.npz')]}}
    with pool, pytest.raises(ValueError, match='No isolated environment'):
        pool.run_roles([role], {})
    assert pool.spawns == 0
    pool.close()


def test_a_different_plan_replaces_the_workers(tmp_path):
    pool = mw._MaskWorkerPool(segmenter=_timed_segment, gate=_Gate())
    with pool:
        pool._open({0: {'A': '1'}}, False)
        first = pool._messages
        pool._open({0: {'A': '1'}}, False)
        assert pool._messages is first
        pool._open({0: {'A': '2'}}, True)
        assert pool._messages is not first and pool._figures is not None
    assert pool._messages is None


def test_one_worker_serves_jobs_in_order_and_keeps_its_model(monkeypatch, tmp_path, fresh_torch):
    import queue

    monkeypatch.setattr(mw, '_MODEL_CACHE', None)
    built = []
    seen = []

    def segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
        model = mw._cached_model(('cellpose', 'cpsam', {'gpu': False}),
                                 lambda: built.append(role) or object())
        seen.append((role, id(model)))
        if settings.get('fail'):
            raise ValueError('simulated inference failure')
        for path in batch_paths:
            on_batch_done(path)

    jobs = iter([(str(tmp_path), {}, 'cell', ['a.npz']),
                 (str(tmp_path), {}, 'nucleus', ['a.npz']),
                 (str(tmp_path), {'fail': True}, 'pathogen', ['a.npz']),
                 (str(tmp_path), {}, 'cell', ['b.npz'])])
    messages = queue.Queue()
    stop = threading.Event()
    mw._serve_mask_jobs('0.1', {}, messages, stop, lambda: next(jobs, None), segment,
                        cache_models=True)
    events = []
    while not messages.empty():
        events.append(messages.get())
    kinds = [(kind, value[0] if kind == 'finished' else None) for kind, _, value in events]
    assert kinds == [('started', None), ('batch', None), ('finished', 'success'),
                     ('batch', None), ('finished', 'success'), ('finished', 'failed')]
    assert built == ['cell'] and len({identity for _, identity in seen}) == 1
    assert mw._MODEL_CACHE['hits'] == 2 and mw._MODEL_CACHE['loads'] == 1
    assert stop.is_set() and next(jobs) is not None


def test_a_cancelled_job_ends_the_worker(tmp_path, fresh_torch):
    import queue

    def segment(*args, **kwargs):
        raise AssertionError('a stopped worker segmented')

    stop = threading.Event()
    stop.set()
    jobs = iter([(str(tmp_path), {}, 'cell', ['a.npz']), (str(tmp_path), {}, 'cell', ['b.npz'])])
    messages = queue.Queue()
    mw._serve_mask_jobs('0', {}, messages, stop, lambda: next(jobs, None), segment)
    assert messages.get_nowait() == ('finished', '0', ('cancelled', ''))
    assert messages.empty() and next(jobs) is not None


def test_the_model_cache_reuses_only_an_identical_model(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(mw, '_MODEL_CACHE', None)
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'one')
    assert mw._cached_model(('cellpose', str(weights), {}), lambda: built.append(1) or 1) == 1
    assert mw._cached_model(('cellpose', str(weights), {}), lambda: built.append(1) or 2) == 2
    assert len(built) == 2
    monkeypatch.setattr(mw, '_MODEL_CACHE', {'key': None, 'model': None, 'hits': 0, 'loads': 0})
    first = mw._cached_model(('cellpose', str(weights), {'gpu': True}), lambda: 'first')
    assert mw._cached_model(('cellpose', str(weights), {'gpu': True}), lambda: 'new') == first
    assert mw._cached_model(('cellpose', str(weights), {'gpu': False}), lambda: 'cpu') == 'cpu'
    os.utime(weights, ns=(1, 1))
    assert mw._cached_model(('cellpose', str(weights), {'gpu': False}), lambda: 'edited') == 'edited'
    assert mw._cached_model(('cellpose', 'cpsam', {}), lambda: 'stock') == 'stock'
    assert mw._cached_model(('cellpose', 'cpsam', {}), lambda: 'again') == 'stock'
    assert mw._MODEL_CACHE['loads'] == 4 and mw._MODEL_CACHE['hits'] == 2


def test_a_model_file_is_hashed_once_while_a_pool_is_open(monkeypatch, tmp_path):
    from spacr import model_zoo

    reads = []
    original = model_zoo.sha256_file
    monkeypatch.setattr(model_zoo, 'sha256_file', lambda path: reads.append(path) or original(path))
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'checkpoint')
    assert mw._file_sha256(weights) == mw._file_sha256(weights)
    assert len(reads) == 2
    reads.clear()
    with mw._MaskWorkerPool():
        digest = mw._file_sha256(weights)
        assert mw._file_sha256(weights) == digest
        assert len(reads) == 1
        weights.write_bytes(b'checkpoint, edited')
        assert mw._file_sha256(weights) != digest
        assert len(reads) == 2
    assert mw._DIGESTS is None


def test_the_cached_generator_writes_the_same_masks_as_the_ordinary_one(
        tmp_path, fake_model, monkeypatch):
    """Cell then nucleus with one cached model gives the bytes of two fresh loads."""
    import spacr.object as O

    built = []
    original = O.cp_models.CellposeModel
    monkeypatch.setattr(O.cp_models, 'CellposeModel',
                        lambda *args, **kwargs: built.append(1) or original(*args, **kwargs))
    announced = []
    monkeypatch.setattr(mw, '_announce_archive', announced.append)
    outputs = {}
    for cached in (False, True):
        monkeypatch.setattr(mw, '_MODEL_CACHE', {'key': None, 'model': None, 'hits': 0,
                                                 'loads': 0} if cached else None)
        built.clear()
        src = tmp_path / ('cached' if cached else 'ordinary') / 'masks'
        _write_npz(src, name='batch1.npz', n=4, seed=4)
        settings = sam_settings(src)
        for role in ROLES:
            O.generate_cellpose_masks_sam(str(src), settings, role, run_qc=False)
        assert len(built) == (1 if cached else 2)
        outputs[cached] = {(role, path.name): path.read_bytes() for role in ROLES
                           for path in sorted((src / f'{role}_mask_stack').glob('*.npy'))}
    assert outputs[True] == outputs[False] and len(outputs[True]) == 8
    assert len(announced) == 4


def test_the_ordinary_path_notes_each_finished_archive(tmp_path, monkeypatch):
    from spacr import accelerator

    monkeypatch.setattr(accelerator, 'is_cuda', lambda: True)
    monkeypatch.setattr(accelerator, 'device_string', lambda: 'cuda:0')
    monkeypatch.setattr(mw, '_device_uuid', lambda index: 'GPU-0')
    monkeypatch.setattr(mw, '_nvml_reader', lambda gpus: (lambda: {0: (70.0, GB)}))
    masks = tmp_path / 'plate1' / 'masks'
    masks.mkdir(parents=True)
    with mw._sampled_ordinary_masks(str(masks), 'cell'):
        mw._announce_archive('batch1.npz')
        mw._announce_archive('batch2.npz')
    mw._announce_archive('after.npz')
    assert mw._ARCHIVE_SAMPLERS == []
    row = json.loads((masks.parent / 'gpu_utilisation.json').read_text(encoding='utf-8'))['cell']['0']
    assert [label for _, label in row['events']] == ['archive finished'] * 2
    assert row['startup_seconds'] is not None and isinstance(row['started_at'], float)


def test_without_the_qc_module_roles_still_overlap(monkeypatch):
    monkeypatch.setitem(sys.modules, 'spacr.seg_qc', None)
    assert mw._roles_may_overlap({'seg_qc': 'flag'}) is True


def test_qc_flags_keep_the_roles_apart():
    assert mw._roles_may_overlap({'seg_qc': 'flag'}) is False
    assert mw._roles_may_overlap({'seg_qc': 'report'}) is True
    assert mw._roles_may_overlap({}) is True


class _Ended:
    """A worker process that already exited with ``exitcode``."""

    def __init__(self, exitcode):
        self.exitcode = exitcode
        self.pid = 4242
        self.closed = False

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None

    def close(self):
        self.closed = True


def _bare_pool(tmp_path):
    """A pool with one role's bookkeeping and no processes, for event checks."""
    pool = mw._MaskWorkerPool(segmenter=_timed_segment, gate=_Gate())
    path = str(tmp_path / 'a.npz')
    run = pool._role_run({'src': str(tmp_path), 'settings': {}, 'object_type': 'cell',
                          'assignments': {'0.1': [path]}}, {'0.1': {}})
    pool._runs = [run]
    pool._inflight = {'0.1': []}
    pool._waiting = {'0.1': []}
    pool._last = {}
    pool._dead_since = {}
    pool._retrying = None
    pool._requested = False
    pool._stop = threading.Event()
    return pool, run


def test_events_a_worker_could_not_have_sent_are_refused(tmp_path):
    pool, run = _bare_pool(tmp_path)
    with pytest.raises(RuntimeError, match='Unknown mask-worker event'):
        pool._handle(0.0, ('progress', '0.1', None))
    with pytest.raises(RuntimeError, match='never given'):
        pool._handle(0.0, ('finished', '0.1', ('success', '')))
    with pytest.raises(RuntimeError, match='Invalid or repeated batch'):
        pool._handle(0.0, ('batch', '0.1', str(tmp_path / 'a.npz')))


def test_a_worker_that_dies_between_jobs_fails_the_role_it_served(tmp_path):
    pool, run = _bare_pool(tmp_path)
    run['state']['workers']['0.1']['state'] = 'success'
    pool._last['0.1'] = run
    crashed = _Ended(9)
    pool._processes = {'0.1': crashed}
    pool._reap()
    worker = run['state']['workers']['0.1']
    assert worker['state'] == 'failed' and worker['error'] == 'Worker exited with code 9'
    assert pool._stop.is_set() and crashed.closed and pool._processes == {}
    quiet = _Ended(0)
    pool._processes = {'0.1': quiet}
    pool._reap()
    assert quiet.closed and worker['error'] == 'Worker exited with code 9'


def test_a_run_whose_roles_cannot_finish_says_so(tmp_path, monkeypatch):
    pool = mw._MaskWorkerPool(segmenter=_timed_segment, gate=_Gate())
    role = {'src': str(tmp_path), 'settings': {}, 'object_type': 'cell',
            'assignments': {'0.1': [str(tmp_path / 'a.npz')]}}
    monkeypatch.setattr(mw._MaskWorkerPool, '_dispatch', lambda self: None)
    with pool, pytest.raises(RuntimeError, match='before every archive finished'):
        pool.run_roles([role], {'0.1': dict(os.environ)})
    assert pool._messages is None and pool.spawns == 0
