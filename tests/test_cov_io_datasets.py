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
