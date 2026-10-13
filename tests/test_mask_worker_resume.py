"""Interrupted spawned mask runs retain checked provenance and completed outputs."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

from spacr._mask_workers import _MaskBatchLedger, _run_checkpointed_mask_workers
from spacr.cancellation import CancellationToken, PipelineCancelled, installed_token
from spacr.checkpoint import CheckpointError, CheckpointMismatch, fingerprint
from spacr.io import _mask_batch_manifest


def _segment(src, settings, role, *, batch_paths, on_batch_done, run_qc):
    from spacr.cancellation import checkpoint

    root = Path(src) / f'{role}_mask_stack'
    root.mkdir(exist_ok=True)
    for path in batch_paths:
        for _ in range(20):
            checkpoint()
            time.sleep(0.01)
        if settings.get('fail'):
            raise RuntimeError('inference failed')
        with np.load(path, allow_pickle=False) as data:
            for name in data['filenames']:
                np.save(root / name, np.ones((8, 9), np.uint16))
        on_batch_done(path)


def _setup(root):
    for index in range(3):
        np.savez(root / f'batch{index}.npz', data=np.zeros((1, 8, 9, 1)),
                 filenames=np.array([f'field{index}.npy']))
    return _mask_batch_manifest(root)


def _ledger(root, records, **kwargs):
    return _MaskBatchLedger(root, records, 'cell', fingerprint({'model': 'cpu-stand-in'}), **kwargs)


def _save(ledger, record):
    ledger.output_root.mkdir(exist_ok=True)
    for name in record['fields']:
        np.save(ledger.output_root / name, np.ones((8, 9), np.uint16))


def test_cancelled_process_run_resumes_only_unfinished_archives(tmp_path):
    records = _setup(tmp_path)
    assignments = {0: [record['path'] for record in records]}
    environments = {0: {**os.environ, 'CUDA_VISIBLE_DEVICES': '7'}}
    ledger = _ledger(tmp_path, records)
    token = CancellationToken()

    def stop_after_one(state):
        if state['completed_batches']:
            token.cancel()

    with installed_token(token), pytest.raises(PipelineCancelled):
        _run_checkpointed_mask_workers(str(tmp_path), {}, 'cell', assignments,
            environments, ledger, segmenter=_segment, on_progress=stop_after_one)
    assert ledger.store.status == 'cancelled'
    complete = ledger.verified()
    assert len(complete) == 1
    first = ledger.output_root / 'field0.npy'
    before = (first.stat().st_mtime_ns, first.read_bytes())
    updates = []
    resumed = _ledger(tmp_path, _mask_batch_manifest(tmp_path))
    result = _run_checkpointed_mask_workers(str(tmp_path), {}, 'cell', assignments,
        environments, resumed, segmenter=_segment, on_progress=updates.append)
    assert result['total_batches'] == len(result['completed_batches']) == 3
    assert result['workers'][0]['total'] == 2
    assert updates[0]['completed_batches'] == sorted(complete)
    assert (first.stat().st_mtime_ns, first.read_bytes()) == before
    assert len(resumed.verified()) == 3
    assert resumed.store.status == 'segmented'
    assert len(list(tmp_path.glob('*.npz'))) == 3
    result = _run_checkpointed_mask_workers(str(tmp_path), {}, 'cell', assignments,
        environments, resumed, segmenter=_segment)
    assert result['workers'] == {}
    assert len(result['completed_batches']) == 3
    document = resumed.store.path.read_text()
    assert 'CUDA_VISIBLE_DEVICES' not in document
    assert 'manifest' in document


@pytest.mark.parametrize('change', ['settings', 'input', 'selection'])
def test_changed_provenance_refuses_resume_without_replacing_state(tmp_path, change):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    _save(ledger, records[0])
    ledger.mark(records[0]['path'])
    before = ledger.store.path.read_bytes()
    signature = fingerprint({'model': 'cpu-stand-in'})
    excluded = []
    if change == 'settings':
        signature = fingerprint({'model': 'new-weights'})
    elif change == 'input':
        records[0]['sha256'] = '0' * 64
    else:
        excluded = records[0]['fields']
    with pytest.raises(CheckpointMismatch):
        _MaskBatchLedger(tmp_path, records, 'cell', signature, excluded_fields=excluded)
    assert ledger.store.path.read_bytes() == before


def test_unclaimed_masks_do_not_acquire_provenance_on_second_attempt(tmp_path):
    records = _setup(tmp_path)
    masks = tmp_path / 'cell_mask_stack'
    masks.mkdir()
    np.save(masks / 'field0.npy', np.zeros((8, 9), np.uint16))
    for _ in range(2):
        with pytest.raises(ValueError, match='no parallel-run provenance'):
            _ledger(tmp_path, records)
        assert not (tmp_path / '.mask-workers-cell.json').exists()


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'changed'])
def test_completed_outputs_are_rechecked(tmp_path, damage):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    _save(ledger, records[0])
    ledger.mark(records[0]['path'])
    target = ledger.output_root / records[0]['fields'][0]
    if damage == 'missing':
        target.unlink()
    elif damage == 'truncated':
        target.write_bytes(target.read_bytes()[:50])
    else:
        np.save(target, np.zeros((8, 9), np.uint16))
    if damage == 'changed':
        with pytest.raises(ValueError, match='Completed mask changed'):
            ledger.verified()
    else:
        assert ledger.verified() == set()


def test_an_excluded_field_needs_no_fake_mask(tmp_path):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records, excluded_fields=records[0]['fields'])
    ledger.mark(records[0]['path'])
    assert ledger.verified() == {records[0]['path']}
    assert not ledger.output_root.exists()


def test_missing_or_truncated_outputs_never_get_a_completion_receipt(tmp_path):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    with pytest.raises(FileNotFoundError):
        ledger.mark(records[0]['path'])
    _save(ledger, records[0])
    (ledger.output_root / records[0]['fields'][0]).write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete mask output'):
        ledger.mark(records[0]['path'])
    assert ledger.store.completed == {}


def test_failed_atomic_receipt_preserves_old_checkpoint(tmp_path, monkeypatch):
    from spacr import checkpoint

    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    _save(ledger, records[0])
    before = ledger.store.path.read_bytes()

    def full_disk(source, target):
        raise OSError('disk full')

    monkeypatch.setattr(checkpoint.os, 'replace', full_disk)
    with pytest.raises(CheckpointError, match='disk full'):
        ledger.mark(records[0]['path'])
    assert ledger.store.completed == {}
    assert ledger.store.path.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))
    assert np.load(ledger.output_root / records[0]['fields'][0]).max() == 1


def test_worker_failure_is_persisted_without_completing_archives(tmp_path):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    assignments = {0: [record['path'] for record in records]}
    with pytest.raises(RuntimeError, match='inference failed'):
        _run_checkpointed_mask_workers(str(tmp_path), {'fail': True}, 'cell',
            assignments, {0: dict(os.environ)}, ledger, segmenter=_segment)
    assert ledger.store.status == 'failed'
    assert not ledger.store.completed
    with pytest.raises(ValueError, match='exactly once'):
        _run_checkpointed_mask_workers(str(tmp_path), {}, 'cell', {0: []}, {}, ledger)


def test_corrupt_receipt_is_refused(tmp_path):
    records = _setup(tmp_path)
    ledger = _ledger(tmp_path, records)
    ledger.store.mark(records[0]['path'], {'wrong.npy': 'digest'})
    with pytest.raises(ValueError, match='Invalid output receipt'):
        ledger.verified()
    payload = json.loads(ledger.store.path.read_text())
    payload['completed']['unknown.npz'] = {}
    ledger.store.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='unknown archive'):
        _ledger(tmp_path, records)


@pytest.mark.parametrize('role,signature,empty', [('../outside', 'a' * 64, False),
    ('cell', 'invalid', False), ('cell', 'a' * 64, True)])
def test_invalid_ledger_identity_fails_before_writing(tmp_path, role, signature, empty):
    records = [] if empty else _setup(tmp_path)
    with pytest.raises(ValueError):
        _MaskBatchLedger(tmp_path, records, role, signature)
    assert not list(tmp_path.glob('*.json'))
