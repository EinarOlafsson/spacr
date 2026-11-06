"""train_model and its per-epoch progress plots must not retain host memory."""
from __future__ import annotations

import gc

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import spacr.accelerator as accelerator
    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _batches(n=2, size=4):
    g = torch.Generator().manual_seed(0)
    return [(torch.rand(size, 4, generator=g),
             torch.tensor([i % 2 for i in range(size)]),
             [f"f{b}_{i}.png" for i in range(size)]) for b in range(n)]


def _run(tmp_path, monkeypatch, epochs, name):
    from spacr.deep_spacr import train_model
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: nn.Sequential(nn.Linear(4, 2)))
    return train_model(
        str(tmp_path / "nosrc"), str(tmp_path / name), "stub", _batches(),
        epochs=epochs, val_loaders=_batches(), tensorboard=False,
        write_card=False, intermedeate_save=False)


def test_progress_plots_are_closed_every_epoch(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    model, _ = _run(tmp_path, monkeypatch, 3, "run")
    assert model is not None
    assert plt.get_fignums() == []


def test_repeated_calls_keep_no_figure_artists(tmp_path, monkeypatch):
    import weakref
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    created = []
    original_init = Figure.__init__

    def tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(weakref.ref(self))

    monkeypatch.setattr(Figure, "__init__", tracking_init)
    for i in range(2):
        _run(tmp_path, monkeypatch, 2, f"run{i}")
    gc.collect()
    assert created, "the run drew no progress figures"
    assert plt.get_fignums() == []
    # A figure may survive as an empty shell, but never with its artists.
    assert [ref() for ref in created if ref() is not None and ref().axes] == []
