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
    assert MB._store_on_gpu(np.zeros((1, 2, 2, 1), dtype=np.float32)) is None


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


class _Card:
    """A fake card: tracks bytes held, host-to-device copies and cache releases."""

    def __init__(self, total_gib):
        self.total = total_gib * 2**30
        self.held = 0
        self.copies = 0
        self.cache_released = 0

    def memory(self):
        return self.total - self.held, self.total

    def copy(self, array):
        self.copies += 1
        card = self
        card.held += array.nbytes

        class Tensor:
            shape = array.shape
            dtype = 'torch.float32'

            def __getitem__(self, key):
                class Slice:
                    def cpu(self_inner):
                        return self_inner

                    def numpy(self_inner):
                        return array[key]
                return Slice()

            def __del__(self):
                card.held -= array.nbytes

        return Tensor()


def _store_with(card, logs=None, tile_batch=5):
    return lambda array: MB._store_on_gpu(
        array, memory=card.memory, copy=card.copy, model_loaded=False,
        tile_batch=tile_batch, log=(logs.append if logs is not None else print))


def test_a_batch_that_does_not_fit_stays_in_ram_and_says_so():
    """Batch + network + eval working set + margin must fit in free VRAM before any byte moves."""
    array = np.zeros((10, 512, 512, 3), dtype=np.float32)          # 30 MiB
    logs = []
    tight = _Card(total_gib=1.5 + 1.3 + 5 * 0.2 + array.nbytes / 2**30 - 0.05)
    assert _store_with(tight, logs)(array) is None
    assert tight.copies == 0 and tight.held == 0, 'a copy started on a card without room'
    assert logs and 'stays in RAM' in logs[0] and 'margin' in logs[0]
    roomy = _Card(total_gib=24)
    assert _store_with(roomy)(array) is not None and roomy.copies == 1


def test_the_working_set_of_a_larger_tile_batch_is_part_of_the_fit():
    """The same batch fits at 5 tiles per pass and not at 128."""
    array = np.zeros((10, 512, 512, 3), dtype=np.float32)
    card = _Card(total_gib=6)
    assert _store_with(card, tile_batch=5)(array) is not None
    other = _Card(total_gib=6)
    assert _store_with(other, [], tile_batch=128)(array) is None


def test_device_memory_returns_to_baseline_after_the_last_role(monkeypatch):
    """The batch is deleted on the device and the cache released when the last role is done."""
    released = []
    monkeypatch.setattr(MB, '_empty_cache', lambda: released.append(1))
    monkeypatch.setattr(MB, '_read_plan', lambda masks_dir: _plan())
    monkeypatch.setattr(MB, '_batch_index_of', lambda path: int(path[-1]))
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    card = _Card(total_gib=24)
    feeder = _feeder(3, _store_with(card), [])
    held_during = []

    def run_role(role, path):
        held_during.append(card.held)

    MB._batch_first('/x', ['0', '1', '2'], ['cell', 'nucleus', 'pathogen'], run_role, feeder=feeder)
    assert all(h > 0 for h in held_during)
    assert card.held == 0, 'a batch stayed on the card after its last role'
    assert len(released) == 3, 'the allocator cache was not released after each batch'
    assert card.copies == 3, 'a batch went to the card more than once'


def test_a_batch_is_freed_when_a_role_raises(monkeypatch):
    """try/finally: a failed batch never stays resident."""
    released = []
    monkeypatch.setattr(MB, '_empty_cache', lambda: released.append(1))
    monkeypatch.setattr(MB, '_read_plan', lambda masks_dir: _plan())
    monkeypatch.setattr(MB, '_batch_index_of', lambda path: int(path[-1]))
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    card = _Card(total_gib=24)
    feeder = _feeder(3, _store_with(card), [])

    def run_role(role, path):
        if role == 'nucleus':
            raise RuntimeError('boom')

    try:
        MB._batch_first('/x', ['0', '1', '2'], ['cell', 'nucleus'], run_role, feeder=feeder)
    except RuntimeError:
        pass
    assert card.held == 0 and released, 'the failed batch stayed on the card'
    assert MB._RESIDENT == {}


def test_roles_read_slices_without_a_second_full_copy(monkeypatch):
    """Eval calls take slices of the resident batch; the full batch crosses to the device once."""
    monkeypatch.setattr(MB, '_empty_cache', lambda: None)
    monkeypatch.setattr(MB, '_read_plan', lambda masks_dir: _plan())
    monkeypatch.setattr(MB, '_batch_index_of', lambda path: int(path[-1]))
    monkeypatch.setattr(MB, '_batch_ram_estimate', lambda plan, index: 0)
    card = _Card(total_gib=24)
    feeder = _feeder(2, _store_with(card), [])
    seen = []

    def run_role(role, path):
        batch = MB._open_batch(path)
        seen.append(np.asarray(batch['data'][0:1]).shape)

    MB._batch_first('/x', ['0', '1'], ['cell', 'nucleus', 'pathogen'], run_role, feeder=feeder)
    assert card.copies == 2 and len(seen) == 6
