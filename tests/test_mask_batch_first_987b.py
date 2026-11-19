"""Batch-first masks (item 987b): one normalised batch in RAM at most, waits for RAM, falls back from the GPU."""
import time

import numpy as np

from spacr import _mask_batches as MB


def _plan(n=4):
    """A stand-in plan; the fake builder never reads it."""
    return {'_masks_dir': '/nonexistent', 'stack_dir': 'stack', 'channels': [0], 'batches': []}


class _Fits:
    shape = (1, 1, 1, 1)
    ndim = 4
    dtype = np.float32

    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]

    def free(self):
        self.array = None


def _feeder(n, store, built, wait=None):
    def build(plan, index):
        built.append(index)
        return {'data': np.full((2, 3, 3, 1), index, dtype=np.float32), 'filenames': [f'f{index}']}
    return MB._BatchFeeder(_plan(), list(range(n)), build=build, store=store,
                           wait_for_ram=wait or (lambda estimate: None))


def test_two_normalised_batches_never_coexist_when_the_gpu_is_full(monkeypatch):
    """Without room on the GPU the batch stays in RAM; the next one is not built until it is freed."""
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    built = []
    feeder = _feeder(4, lambda array: None, built)
    seen = []
    for _ in range(4):
        index, resident = feeder.next()
        time.sleep(0.05)
        seen.append((index, len(built)))
        assert len(built) == index + 1, 'the next batch was built while this one was still in RAM'
        resident.free()
    feeder.close()
    assert feeder.peak_live == 1


def test_the_next_batch_is_built_while_this_one_sits_on_the_gpu(monkeypatch):
    """With the batch on the GPU its RAM slot is free, so batch k+1 is prefetched during batch k."""
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    built = []
    feeder = _feeder(3, lambda array: _Fits(array), built)
    index, resident = feeder.next()
    deadline = time.time() + 5
    while len(built) < 2 and time.time() < deadline:
        time.sleep(0.01)
    assert built[:2] == [0, 1]
    resident.free()
    for _ in range(2):
        feeder.next()[1].free()
    feeder.close()
    assert feeder.peak_live == 1


def test_the_feeder_waits_for_ram_before_each_build(monkeypatch):
    """The RAM check runs before every build, with that batch's estimate."""
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 100 + index)
    asked, built = [], []
    feeder = _feeder(3, lambda array: _Fits(array), built, wait=asked.append)
    for _ in range(3):
        feeder.next()[1].free()
    feeder.close()
    assert asked == [100, 101, 102]


def test_wait_for_ram_polls_until_enough_is_free():
    """Short RAM makes the loader wait; it loads as soon as MemAvailable is back."""
    levels = iter([1 * 2**30, 1 * 2**30, 9 * 2**30])
    slept = []
    waited = MB._wait_for_ram(4 * 2**30, available=lambda: next(levels), sleep=slept.append)
    assert len(slept) == 2 and waited == 2 * MB._RAM_POLL_SECONDS


def test_wait_for_ram_does_not_wait_when_unknown_or_enough():
    """Unknown or plentiful RAM never sleeps."""
    slept = []
    assert MB._wait_for_ram(1, available=lambda: None, sleep=slept.append) == 0
    assert MB._wait_for_ram(1, available=lambda: 99 * 2**30, sleep=slept.append) == 0
    assert slept == []


def test_wait_for_ram_gives_up_after_the_limit(monkeypatch):
    """A card of RAM that never frees does not block the run for ever."""
    monkeypatch.setattr(MB, '_RAM_WAIT_SECONDS', 3.0)
    slept = []
    waited = MB._wait_for_ram(10 * 2**30, available=lambda: 0, sleep=slept.append)
    assert waited == 3.0 and len(slept) == 3


def test_gpu_store_falls_back_to_ram_without_room(monkeypatch):
    """A batch that does not fit free VRAM plus the margin stays in RAM (None)."""
    array = np.zeros((4, 8, 8, 1), dtype=np.float32)
    assert MB._store_on_gpu(array, memory=lambda: (array.nbytes, 2 * array.nbytes)) is None


def test_gpu_store_is_none_without_cuda(monkeypatch):
    """No CUDA device means no GPU copy."""
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert MB._store_on_gpu(np.zeros((1, 2, 2, 1), dtype=np.float32), memory=lambda: (10**12, 10**12)) is None


def test_roles_run_on_every_batch_before_the_next_batch_is_loaded(monkeypatch):
    """Batch-first order: all roles on batch k, then batch k+1; the resident batch is visible only meanwhile."""
    monkeypatch.setattr(MB, '_read_plan', lambda masks_dir: _plan())
    monkeypatch.setattr(MB, '_batch_index_of', lambda path: int(path[-1]))
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    built = []
    feeder = _feeder(3, lambda array: _Fits(array), built)
    order = []

    def run_role(role, path):
        assert MB._RESIDENT.get(__import__('os').path.abspath(path)) is not None
        order.append((role, path))

    MB._batch_first('/x', ['0', '1', '2'], ['cell', 'nucleus'], run_role, feeder=feeder)
    assert order == [(r, p) for p in ('0', '1', '2') for r in ('cell', 'nucleus')]
    assert MB._RESIDENT == {}
    assert feeder.peak_live == 1


def test_a_role_that_fails_frees_the_batch_and_stops_the_feeder(monkeypatch):
    """An exception inside a role leaves nothing resident and the builder stopped."""
    monkeypatch.setattr(MB, '_read_plan', lambda masks_dir: _plan())
    monkeypatch.setattr(MB, '_batch_index_of', lambda path: int(path[-1]))
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    feeder = _feeder(3, lambda array: None, [])

    def run_role(role, path):
        raise RuntimeError('boom')

    try:
        MB._batch_first('/x', ['0', '1', '2'], ['cell'], run_role, feeder=feeder)
    except RuntimeError:
        pass
    assert MB._RESIDENT == {}
    feeder._thread.join(timeout=5)
    assert not feeder._thread.is_alive()
