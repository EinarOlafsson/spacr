"""CPU-only checks of the TensorBoard writer and of resuming ``train_model``.

The resumed run uses a 4-feature ``nn.Linear`` stand-in model and hand-built
"loaders" (plain lists of ``(data, target, filenames)`` tuples), so it costs
milliseconds and never touches CUDA, torchvision weights or the filesystem
outside ``tmp_path``.  The training loop itself is covered table-driven in
``tests/test_deep_spacr_training_loop.py``.
"""
from __future__ import annotations


import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (after importorskip)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let live-training figures accumulate across tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Pin train_model (and evaluate_model_performance) to the CPU device.

    train_model picks its device from torch.cuda.is_available(); on a
    GPU box these tests would otherwise move the stand-in model to cuda.
    """
    import spacr.accelerator as accelerator

    # ``pick_device`` consults this cache before torch.  Reset it so a CUDA
    # result cached during collection or by an earlier test cannot defeat the
    # CPU premise this module declares.
    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _loaders(n_batches=2, batch=4, n_feat=4, n_classes=2, mode="index", seed=0):
    """Build a list of ``(data, target, filenames)`` batches.

    ``mode`` selects the target encoding: ``'index'`` (class indices),
    ``'onehot'`` (N, C float) or ``'binary'`` (float 0/1 for a 1-logit head).
    A list is a perfectly good stand-in for a DataLoader here — train_model
    only calls ``len()`` and iterates.
    """
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in range(n_batches):
        x = torch.rand(batch, n_feat, generator=g)
        idx = torch.tensor([i % n_classes for i in range(batch)], dtype=torch.long)
        if mode == "onehot":
            y = torch.nn.functional.one_hot(idx, num_classes=n_classes).float()
        elif mode == "binary":
            y = idx.float()
        else:
            y = idx
        out.append((x, y, [f"b{b}_{i}.png" for i in range(batch)]))
    return out


def _use_model(monkeypatch, model):
    """Make ``choose_model`` hand back our tiny stand-in network."""
    monkeypatch.setattr("spacr.utils.choose_model", lambda *a, **k: model)
    return model


def _metrics(epoch, acc=0.5, loss=0.7):
    """A metrics dict shaped like ``evaluate_model_performance``'s output."""
    return {
        "accuracy": float(acc), "Accuracy": float(acc),
        "neg_accuracy": float(acc), "pos_accuracy": float(acc),
        "prauc": 0.5, "optimal_threshold": 0.5, "f1_macro": float(acc),
        "loss": float(loss), "epoch": int(epoch),
    }


def _stub_eval(monkeypatch, acc=0.5, loss=0.7, calls=None):
    """Replace evaluate_model_performance with a deterministic stub."""
    def fake(model, loader, epoch, **kw):
        if calls is not None:
            calls.append((epoch, len(loader)))
        return _metrics(epoch, acc=acc, loss=loss), [None, None]
    monkeypatch.setattr("spacr.deep_spacr.evaluate_model_performance", fake)
    return calls


def _stub_progress(monkeypatch, saved):
    """Record _save_progress calls instead of writing CSVs + 6 PDF plots."""
    def fake(dst, train_df, validation_df):
        saved.append((dst, train_df, validation_df))
    monkeypatch.setattr("spacr.io._save_progress", fake)
    return saved


def test_tensorboard_can_be_disabled_without_importing_backend(tmp_path):
    from spacr.deep_spacr import _open_tensorboard_writer

    writer, log_dir = _open_tensorboard_writer(tmp_path, enabled=False)
    assert writer is None
    assert log_dir == str((tmp_path / 'tensorboard').resolve())


def test_tensorboard_writer_creates_readable_event_file(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from spacr.deep_spacr import (
        _log_tensorboard_epoch,
        _open_tensorboard_writer,
    )

    writer, log_dir = _open_tensorboard_writer(tmp_path, enabled=True)
    assert writer is not None
    _log_tensorboard_epoch(
        writer, {'loss': 0.25, 'accuracy': 0.9, 'lr': 1e-3}, None, 1)
    writer.close()

    events = EventAccumulator(log_dir)
    events.Reload()
    assert 'loss/train' in events.Tags()['scalars']
    assert events.Scalars('accuracy/train')[0].value == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# early stopping + suggestion guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("catalogue_key", [None, "published-training-checkpoint"])
def test_train_model_resumes_after_saved_epoch_with_optimizer_state(
        tmp_path, monkeypatch, catalogue_key):
    """A local file or catalogue key resumes at epoch+1 with optimizer state."""
    from spacr.deep_spacr import train_model
    from spacr.torch_artifacts import load_model_artifact, save_model_artifact

    original = nn.Sequential(nn.Linear(4, 2))
    optimizer = torch.optim.Adam(original.parameters(), lr=2e-4)
    loss = nn.functional.cross_entropy(
        original(torch.rand(4, 4)), torch.tensor([0, 1, 0, 1]))
    loss.backward()
    optimizer.step()
    resume = tmp_path / "resume.pth"
    save_model_artifact(
        original, resume, optimizer=optimizer, epoch=1, best_metric=1.0,
        epochs_without_improvement=0, artifact_role="best")
    resolved = []
    if catalogue_key is not None:
        def _resolve(requested, *, kinds=(), setting=None):
            resolved.append((requested, kinds, setting))
            return resume

        monkeypatch.setattr("spacr.model_zoo._ensure_model_file", _resolve)

    fresh = nn.Sequential(nn.Linear(4, 2))
    _use_model(monkeypatch, fresh)
    calls = []
    _stub_eval(monkeypatch, acc=0.5, calls=calls)
    saved = _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"
    dst.mkdir()

    trained, selected = train_model(
        str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=2,
        learning_rate=2e-4, optimizer_type="adam", num_classes=2,
        schedule=None, resume_checkpoint=catalogue_key or str(resume),
        intermedeate_save=False)

    assert calls == []
    assert [int(e) for _d, train_df, _v in saved for e in train_df["epoch"]] == [2]
    assert resolved == ([(catalogue_key, ("classifier",), "resume_checkpoint")]
                        if catalogue_key is not None else [])
    assert selected == str(resume)
    assert (tmp_path / "resume.card.json").is_file()
    assert not (tmp_path / "published-training-checkpoint.card.json").exists()
    last = dst / "resnet18_last_channels_rgb.pth"
    _, payload = load_model_artifact(
        last, model=nn.Sequential(nn.Linear(4, 2)))
    assert payload["training_state"]["epoch"] == 2
    assert payload["optimizer_state_dict"] is not None
    assert trained is fresh
