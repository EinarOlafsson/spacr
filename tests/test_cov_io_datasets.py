"""CPU-only coverage for the dataset / dataloader classes in ``spacr.io``.

Targets the branches the rest of the suite never reaches:

* ``CombineLoaders.__next__`` pruning the exhausted prefix of its iterator
  list (``pos`` truthy).
* ``spacrDataset`` with ``pin_memory=True`` (multiprocessing preload and the
  in-RAM ``__getitem__`` path) and ``get_plate``.
* ``spacrDataLoader`` pinning batches in the producer thread, every branch of
  ``_pin_memory_batch``, the ``queue.Empty`` timeout in ``__next__`` and the
  ``__del__`` finaliser.

Everything here stays on the CPU: ``torch.Tensor.pin_memory`` and torch's own
``_utils.pin_memory.pin_memory`` are stubbed out so no CUDA allocation ever
happens, which also isolates the calls made by spaCR's own code.
"""
from __future__ import annotations

import os
import queue

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _png(path, rng, size=8):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _class_dirs(root, rng, classes=("nc", "pc"), n=2):
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _png(d / f"plate1_{cls}_{i}.png", rng)
    return root


# ---------------------------------------------------------------------------
# CombineLoaders
# ---------------------------------------------------------------------------

def test_combine_loaders_drops_exhausted_prefix(monkeypatch):
    """An empty loader ahead of a live one is pruned, not re-polled forever.

    ``random.shuffle`` is neutralised so the empty loader is deterministically
    visited first; the successful draw therefore happens at ``pos == 1`` and
    the iterator list must be truncated to just the live loader.
    """
    from torch.utils.data import DataLoader, TensorDataset
    import spacr.io as sio
    from spacr.io import CombineLoaders

    monkeypatch.setattr(sio.random, "shuffle", lambda seq: None)

    empty = DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=2)
    full = DataLoader(TensorDataset(torch.arange(6, dtype=torch.float32).unsqueeze(1)),
                      batch_size=2)

    comb = CombineLoaders([empty, full])
    assert len(comb.loader_iters) == 2

    first_idx, first_batch = next(comb)
    # The empty loader sat at position 0 and has been dropped.
    assert first_idx == 1
    assert [i for i, _ in comb.loader_iters] == [1]
    assert first_batch[0].flatten().tolist() == [0.0, 1.0]

    seen = [first_batch[0].flatten().tolist()]
    for idx, batch in comb:
        assert idx == 1
        seen.append(batch[0].flatten().tolist())
    assert seen == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert comb.loader_iters == []


def test_combine_loaders_all_empty_raises_stopiteration(monkeypatch):
    """Only-empty loaders end the combined stream instead of looping."""
    from torch.utils.data import DataLoader, TensorDataset
    import spacr.io as sio
    from spacr.io import CombineLoaders

    monkeypatch.setattr(sio.random, "shuffle", lambda seq: None)
    loaders = [DataLoader(TensorDataset(torch.empty(0, 1)), batch_size=1)
               for _ in range(2)]
    comb = CombineLoaders(loaders)
    with pytest.raises(StopIteration):
        next(comb)
    assert comb.loader_iters == []


# ---------------------------------------------------------------------------
# spacrDataset
# ---------------------------------------------------------------------------

def test_spacr_dataset_pin_memory_preloads_all_images(tmp_path, rng, monkeypatch):
    """pin_memory=True decodes every file through a Pool and serves from RAM."""
    import spacr.io as sio
    from spacr.io import spacrDataset

    # One worker keeps the fork cheap; the real multiprocessing.Pool still runs.
    monkeypatch.setattr(sio, "cpu_count", lambda: 1)

    root = _class_dirs(tmp_path / "train", rng, n=3)
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"], shuffle=False,
                      pin_memory=True)

    assert len(ds) == 6
    assert ds.images is not None and len(ds.images) == 6
    assert all(isinstance(im, Image.Image) for im in ds.images)
    assert all(im.mode == "RGB" and im.size == (8, 8) for im in ds.images)

    img, label, filename = ds[2]
    # Served straight from the preloaded list, not re-read from disk.
    assert img is ds.images[2]
    assert label == ds.classes.index(os.path.basename(os.path.dirname(filename)))
    assert filename == ds.filenames[2]


def test_spacr_dataset_pin_memory_getitem_after_file_deleted(tmp_path, rng, monkeypatch):
    """The in-RAM path never touches disk: __getitem__ works after unlinking."""
    import spacr.io as sio
    from spacr.io import spacrDataset

    monkeypatch.setattr(sio, "cpu_count", lambda: 1)
    d = tmp_path / "flat"
    d.mkdir()
    files = [_png(d / f"plateX_{i}.png", rng) for i in range(3)]
    ds = spacrDataset(str(d), loader_classes=["a"], shuffle=False,
                      pin_memory=True, specific_files=files,
                      specific_labels=[0, 1, 0])
    for f in files:
        os.remove(f)

    img, label, filename = ds[1]
    assert isinstance(img, Image.Image)
    assert img.size == (8, 8)
    assert label == 1
    assert filename == files[1]


def test_spacr_dataset_get_plate_parses_leading_token(tmp_path, rng):
    """get_plate returns the token before the first underscore of the basename."""
    from spacr.io import spacrDataset

    d = tmp_path / "flat"
    d.mkdir()
    files = [_png(d / "plate1_A01_f1.png", rng)]
    ds = spacrDataset(str(d), loader_classes=["a"], shuffle=False,
                      specific_files=files, specific_labels=[0])

    assert ds.get_plate("/some/where/plate1_A01_f1.png") == "plate1"
    assert ds.get_plate(files[0]) == "plate1"
    # No underscore at all -> whole basename (minus nothing) is the plate.
    assert ds.get_plate("/tmp/plate42.png") == "plate42.png"


# ---------------------------------------------------------------------------
# spacrDataLoader
# ---------------------------------------------------------------------------

def _stub_pinning(monkeypatch):
    """Replace tensor pinning with a recording no-op (keeps the test CPU-only).

    Also neutralises torch's *own* pinning inside the base DataLoader iterator
    so every recorded call provably comes from ``spacrDataLoader``.
    """
    from torch.utils.data._utils import pin_memory as torch_pin

    calls = []

    def fake_pin(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", fake_pin, raising=False)
    monkeypatch.setattr(torch_pin, "pin_memory",
                        lambda data, device=None: data, raising=False)
    return calls


def test_spacr_dataloader_pins_every_batch(tmp_path, rng, monkeypatch):
    """With pin_memory set, the producer thread pins each collated batch."""
    from torchvision import transforms
    from spacr.io import spacrDataset, spacrDataLoader

    calls = _stub_pinning(monkeypatch)

    root = _class_dirs(tmp_path / "train", rng, n=2)
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"],
                      transform=transforms.ToTensor(), shuffle=False)
    dl = spacrDataLoader(ds, batch_size=2, preload_batches=1, pin_memory=True)
    try:
        assert dl.pin_memory is True
        batches = list(iter(dl))
        assert len(batches) == 2                      # 4 images / batch 2
        # _pin_memory_batch turns the collated tuple into a list.
        assert all(isinstance(b, list) for b in batches)
        images, labels, filenames = batches[0]
        assert isinstance(images, torch.Tensor)
        assert images.shape == (2, 3, 8, 8)
        assert isinstance(labels, torch.Tensor) and labels.shape == (2,)
        assert len(filenames) == 2 and all(isinstance(f, str) for f in filenames)
        # Two tensors per batch (images + labels), two batches, and nothing
        # else pinned: the filename list is passed through untouched.
        assert calls == [(2, 3, 8, 8), (2,), (2, 3, 8, 8), (2,)]
    finally:
        dl.cleanup()


def test_pin_memory_batch_handles_tensor_and_passthrough(tmp_path, rng, monkeypatch):
    """_pin_memory_batch: list, bare-tensor and unsupported-type branches."""
    from spacr.io import spacrDataset, spacrDataLoader

    calls = _stub_pinning(monkeypatch)

    d = tmp_path / "flat"
    d.mkdir()
    files = [_png(d / f"plateX_{i}.png", rng) for i in range(2)]
    ds = spacrDataset(str(d), loader_classes=["a"], shuffle=False,
                      specific_files=files, specific_labels=[0, 1])
    dl = spacrDataLoader(ds, batch_size=1, preload_batches=1)
    try:
        t = torch.zeros(2, 2)
        out = dl._pin_memory_batch([t, "name"])
        assert isinstance(out, list) and len(out) == 2
        assert out[0] is t                    # tensor pinned (stubbed) in place
        assert out[1] == "name"               # non-tensors passed through

        bare = torch.ones(3)
        assert dl._pin_memory_batch(bare) is bare

        payload = {"images": 1}
        assert dl._pin_memory_batch(payload) is payload   # unsupported -> as-is

        # tuple input also takes the list branch
        tup_out = dl._pin_memory_batch((torch.zeros(1), 7))
        assert isinstance(tup_out, list) and tup_out[1] == 7

        assert calls == [(2, 2), (3,), (1,)]
    finally:
        dl.cleanup()


def test_spacr_dataloader_next_stops_when_queue_times_out(tmp_path, rng):
    """A queue that never produces ends iteration instead of hanging forever."""
    from spacr.io import spacrDataset, spacrDataLoader

    d = tmp_path / "flat"
    d.mkdir()
    files = [_png(d / f"plateX_{i}.png", rng) for i in range(2)]
    ds = spacrDataset(str(d), loader_classes=["a"], shuffle=False,
                      specific_files=files, specific_labels=[0, 1])
    dl = spacrDataLoader(ds, batch_size=1, preload_batches=1)

    class _TimeoutQueue:
        """Stands in for a producer that never delivers within the timeout."""

        def __init__(self):
            self.timeouts = []

        def get(self, timeout=None):
            self.timeouts.append(timeout)
            raise queue.Empty

    try:
        dl.batch_queue = _TimeoutQueue()
        dl.current_batch_index = 0
        with pytest.raises(StopIteration):
            next(dl)
        assert dl.batch_queue.timeouts == [60]
        assert dl.current_batch_index == 0     # nothing was counted as a batch
    finally:
        dl.cleanup()


def test_spacr_dataloader_del_runs_cleanup(tmp_path, rng):
    """__del__ signals the producer to stop and joins the thread."""
    from torchvision import transforms
    from spacr.io import spacrDataset, spacrDataLoader

    root = _class_dirs(tmp_path / "train", rng, n=2)
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"],
                      transform=transforms.ToTensor(), shuffle=False)
    dl = spacrDataLoader(ds, batch_size=2, preload_batches=1)

    it = iter(dl)
    assert next(it)[0].shape == (2, 3, 8, 8)
    assert dl.thread is not None
    assert dl._stop_event is False

    dl.__del__()

    assert dl._stop_event is True
    assert not dl.thread.is_alive()
    # cleanup is idempotent - the real finaliser may run again later.
    dl.__del__()
    assert dl._stop_event is True


# ---------------------------------------------------------------------------
# TarImageDataset / CombinedDataset sanity around the same region
# ---------------------------------------------------------------------------

def test_combined_dataset_shuffle_permutes_index_space(tmp_path, rng):
    """A shuffled CombinedDataset still exposes exactly the union of samples."""
    from spacr.io import CombinedDataset, NoClassDataset
    import spacr.io as sio

    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    for i in range(3):
        _png(a / f"a{i}.png", rng)
    for i in range(2):
        _png(b / f"b{i}.png", rng)

    ds_a = NoClassDataset(str(a), shuffle=False)
    ds_b = NoClassDataset(str(b), shuffle=False)
    comb = CombinedDataset([ds_a, ds_b], shuffle=True)

    assert len(comb) == 5
    assert sorted(comb.indices) == list(range(5))
    paths = {comb[i][1] for i in range(len(comb))}
    assert paths == set(ds_a.filenames) | set(ds_b.filenames)
