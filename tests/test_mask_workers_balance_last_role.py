"""Item 801: the last role ends on every worker together, and each archive's stages are timed.

Five archives split between two workers leave one worker with three in
every role. Workers serve roles in order, so the next role fills the gap
the first one leaves, but nothing fills the last role's: one worker ran
on alone and the nucleus role's steady GPU use fell to the one-worker
figure. The split now carries each worker's running total from one role
to the next. No GPU is used here: CPU stand-in segmenters run in real
spawned workers, a fake Cellpose model stands in for Cellpose-SAM, and a
stand-in reader replaces NVML.
"""

import json
import os
from pathlib import Path

from spacr import _mask_workers as mw
from tests.test_cov_object_masks_sam import _settings as sam_settings
from tests.test_cov_object_masks_sam import _write_npz
from tests.test_cov_object_masks_sam import fake_model, force_cpu  # noqa: F401
from tests.test_mask_parallel_pipeline import _plate, _segment, _settings, cpu_workers  # noqa: F401
from tests.test_mask_workers_persist_across_roles import (
    ROLES, _Gate, _shared_plan, _unrestricted_gpu_visibility, sampled,  # noqa: F401
)


def _records(sizes):
    return [{'path': f'/plate/masks/batch{index}.npz', 'bytes': size}
            for index, size in enumerate(sizes)]


def test_one_role_alone_splits_as_before():
    """Without a running total the split is the plain byte balance."""
    records = _records([50, 50, 50, 50, 40])
    assert mw._partition_batches(records, ['0.1', '0.2']) == mw._partition_batches(
        records, ['0.1', '0.2'], {})


def test_the_next_role_gives_more_to_the_worker_that_had_less():
    """240 fields in archives of 50: each worker ends both roles with 240."""
    records = _records([50, 50, 50, 50, 40])
    load = {}
    cell = mw._partition_batches(records, ['0.1', '0.2'], load)
    nucleus = mw._partition_batches(records, ['0.1', '0.2'], load)
    size = {record['path']: record['bytes'] for record in records}

    def total(assignments, worker):
        return sum(size[path] for path in assignments[worker])

    assert (total(cell, '0.1'), total(cell, '0.2')) == (140, 100)
    assert (total(nucleus, '0.1'), total(nucleus, '0.2')) == (100, 140)
    assert load == {'0.1': 240, '0.2': 240}
    for assignments in (cell, nucleus):
        assert sorted(path for paths in assignments.values() for path in paths) == sorted(size)


def _staged_segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
    """The item-493 CPU segmenter, logging who took which archive and timing it."""
    for path in batch_paths:
        _segment(src, settings, role, batch_paths=[path], on_batch_done=on_batch_done,
                 run_qc=run_qc)
        with open(Path(src).parent / 'roles.log', 'a', encoding='utf-8') as log:
            log.write(f'{os.getpid()} {role} {Path(path).name}\n')
        mw._announce_archive(path, {'load': 0.01, 'prep': 0.02, 'segment': 0.03,
                                    'post': 0.04, 'write': 0.05, 'release': 0.0,
                                    'fields': 2})


def test_two_roles_balance_across_workers_with_the_same_masks(tmp_path, cpu_workers, sampled):
    serial = _serial_five(tmp_path / 'serial' / 'plate1')
    masks = _plate(tmp_path / 'pool' / 'plate1', archives=5)
    settings = _settings(mask_workers_per_gpu=2)
    plan = _shared_plan(settings)
    with mw._MaskWorkerPool(segmenter=_staged_segment, gate=_Gate()) as pool:
        mw._generate_roles_in_parallel(str(masks), settings, list(ROLES), plan, pool=pool)
    for role in ROLES:
        names = sorted(path.name for path in (serial / f'{role}_mask_stack').glob('*.npy'))
        assert len(names) == 10
        for name in names:
            assert ((serial / f'{role}_mask_stack' / name).read_bytes()
                    == (masks / f'{role}_mask_stack' / name).read_bytes())
    rows = [line.split() for line in
            (masks.parent / 'roles.log').read_text(encoding='utf-8').splitlines()]
    per_pid = {}
    for pid, role, _ in rows:
        per_pid.setdefault(pid, {}).setdefault(role, 0)
        per_pid[pid][role] += 1
    assert len(per_pid) == 2
    assert sorted(counts['cell'] + counts['nucleus'] for counts in per_pid.values()) == [5, 5]
    assert sorted((counts['cell'], counts['nucleus']) for counts in per_pid.values()) == [
        (2, 3), (3, 2)]
    record = json.loads((masks.parent / 'gpu_utilisation.json').read_text(encoding='utf-8'))
    for role in ROLES:
        row = record[role]['0']
        assert len(row['archives']) == 5
        assert {archive['worker'] for archive in row['archives']} == {
            str(device) for device in plan['devices']}
        for archive in row['archives']:
            assert archive['fields'] == 2 and archive['segment'] == 0.03
            assert archive['archive'].startswith('batch') and archive['at'] >= 0
        totals = row['stage_seconds']
        assert sum(worker['archives'] for worker in totals.values()) == 5
        assert sum(worker['fields'] for worker in totals.values()) == 10
        assert row['tail_seconds'] is not None and row['tail_samples'] >= 0


def _serial_five(root):
    serial = _plate(root, archives=5)
    for role in ROLES:
        _segment(str(serial), {}, role, batch_paths=sorted(serial.glob('*.npz')),
                 on_batch_done=lambda path: None, run_qc=False)
    return serial


def test_timing_the_stages_leaves_the_masks_bit_identical(tmp_path, fake_model, monkeypatch):
    """The real generator writes the same bytes with or without a stage listener.

    Three batches of one archive are timed into one record.
    """
    import spacr.object as O

    outputs = {}
    reported = []
    for listening in (False, True):
        monkeypatch.setattr(mw, '_STAGE_SINK', (lambda path, stages: reported.append(
            (Path(path).name, dict(stages)))) if listening else None)
        src = tmp_path / ('timed' if listening else 'plain') / 'masks'
        _write_npz(src, name='batch1.npz', n=5, seed=4)
        settings = sam_settings(src, batch_size=2)
        for role in ROLES:
            O.generate_cellpose_masks_sam(str(src), settings, role, run_qc=False)
        outputs[listening] = {(role, path.name): path.read_bytes() for role in ROLES
                              for path in sorted((src / f'{role}_mask_stack').glob('*.npy'))}
    assert outputs[True] == outputs[False] and len(outputs[True]) == 10
    assert [(name, stages['fields']) for name, stages in reported] == [('batch1.npz', 5)] * 2
    for _, stages in reported:
        assert set(stages) == set(O._ARCHIVE_STAGES) | {'fields'}
        assert all(stages[stage] >= 0 for stage in O._ARCHIVE_STAGES)
        assert stages['segment'] > 0


def test_the_ordinary_path_records_each_archive_s_stages(tmp_path, monkeypatch):
    from spacr import accelerator

    monkeypatch.setattr(accelerator, 'is_cuda', lambda: True)
    monkeypatch.setattr(accelerator, 'device_string', lambda: 'cuda:0')
    monkeypatch.setattr(mw, '_device_uuid', lambda index: 'GPU-0')
    monkeypatch.setattr(mw, '_nvml_reader', lambda gpus: (lambda: {0: (70.0, 1 << 30)}))
    masks = tmp_path / 'plate1' / 'masks'
    masks.mkdir(parents=True)
    with mw._sampled_ordinary_masks(str(masks), 'nucleus'):
        mw._announce_archive('batch1.npz', {'load': 1.0, 'segment': 2.0, 'fields': 50})
    row = json.loads((masks.parent / 'gpu_utilisation.json').read_text(
        encoding='utf-8'))['nucleus']['0']
    assert row['archives'][0]['archive'] == 'batch1.npz'
    assert row['archives'][0]['worker'] is None
    assert row['stage_seconds'] == {'None': {'archives': 1, 'load': 1.0, 'segment': 2.0,
                                             'fields': 50}}
    assert row['tail_seconds'] is None


def test_the_tail_starts_when_the_first_worker_finishes():
    values = [(95.0, 0, 1.0), (95.0, 0, 2.0), (60.0, 0, 3.0), (60.0, 0, 4.0)]
    events = [[1.5, 'archive finished'], [2.5, 'worker 0.2 success'],
              [4.0, 'worker 0.1 success']]
    assert mw._tail_split(values, events) == {
        'tail_seconds': 1.5, 'tail_mean_busy_percent': 60.0, 'tail_samples': 2}
    assert mw._tail_split(values, [])['tail_seconds'] is None


def test_a_stage_event_from_an_idle_worker_is_ignored():
    pool = mw._MaskWorkerPool()
    pool._inflight = {'0.1': []}
    pool._runs = [{'on_stages': lambda *args: (_ for _ in ()).throw(AssertionError)}]
    pool._last = {}
    pool._handle(0.0, ('stages', '0.1', ('/x/batch0.npz', {'fields': 1})))
