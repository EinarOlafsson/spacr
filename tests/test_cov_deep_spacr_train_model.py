"""CPU-only coverage for ``spacr.deep_spacr.train_model`` (the training loop).

Every branch of the loop is driven with a 4-feature ``nn.Linear`` stand-in
model and hand-built "loaders" (plain lists of ``(data, target, filenames)``
tuples), so a whole run costs milliseconds and never touches CUDA, torchvision
weights or the filesystem outside ``tmp_path``.

Covered here: the ``channels``/``classes`` defaults, every ``optimizer_type``
branch (including the unknown-optimizer ``ValueError``), all three schedulers,
gradient accumulation (loss scaling + end-of-epoch flush), one-hot targets,
the head/num_classes mismatch ``RuntimeError``, the single-logit binary path,
the live-plot hook (success and swallowed failure), early stopping, and the
``suggest_training_changes`` failure guard.
"""
from __future__ import annotations

import os

import pandas as pd
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


def _recording_optimizer(base, log):
    """Subclass an optimizer so the test can see its ctor kwargs + step count."""
    class Recording(base):
        def __init__(self, params, **kwargs):
            log["kwargs"] = dict(kwargs)
            log["steps"] = 0
            super().__init__(params, **kwargs)
            log["instance"] = self

        def step(self, *args, **kwargs):
            log["steps"] += 1
            return super().step(*args, **kwargs)

    return Recording


# ---------------------------------------------------------------------------
# happy path: defaults, no validation loader, real io/eval helpers
# ---------------------------------------------------------------------------

def test_train_model_defaults_no_val_writes_csv_and_checkpoint(tmp_path, monkeypatch, capsys):
    """Full real run (real evaluate/_save_model/_save_progress), no val split.

    Exercises the channels=None default, the missing-``train``-folder branch
    (classes stays None so no class counting happens), the test-loader print,
    ``schedule=None``, the no-validation progress print, the train-only
    _save_progress branch and the best-model fallback.
    """
    from spacr.deep_spacr import train_model
    model = _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    src = tmp_path / "data"; src.mkdir()          # deliberately has no train/ dir
    dst = tmp_path / "model"; dst.mkdir()

    out = train_model(str(src), str(dst), "resnet18", _loaders(2), epochs=1,
                      val_loaders=None, test_loaders=_loaders(1),
                      num_classes=2, schedule=None, learning_rate=1e-3)

    assert isinstance(out, tuple) and len(out) == 2
    trained, path = out
    assert trained is model
    # channels defaulted to ['r', 'g', 'b'] -> filename suffix 'channels_rgb'
    assert os.path.basename(path) == "resnet18_epoch_1_channels_rgb.pth"
    assert os.path.exists(path)

    df = pd.read_csv(dst / "train.csv")
    assert len(df) == 1
    assert {"accuracy", "loss", "epoch", "train_time"} <= set(df.columns)
    assert df["epoch"].iloc[0] == 1
    assert not (dst / "validation.csv").exists()

    cap = capsys.readouterr().out
    assert "Train batches:2, Validation batches:0" in cap
    assert "Test batches:1" in cap
    assert "Val Loss" not in cap          # the no-validation progress line


def test_train_model_counts_classes_from_train_folder(tmp_path, monkeypatch, capsys):
    """With src/train/<class>/ present the class counts come from the folders."""
    from spacr.deep_spacr import train_model
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    src = tmp_path / "data"
    for cls, n in (("nc", 3), ("pc", 5)):
        d = src / "train" / cls
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"{cls}_{i}.png").write_bytes(b"x")
    (src / "train" / ".hidden").mkdir()            # dot-dirs are ignored
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(src), str(dst), "resnet18", _loaders(1),
                                epochs=1, num_classes=2, schedule=None)

    assert os.path.exists(path)
    out = capsys.readouterr().out
    assert "Class counts (from folders): {'nc': 3, 'pc': 5}" in out


def test_train_model_invalid_model_type_bails_out(tmp_path, monkeypatch, capsys):
    """choose_model returning None must abort before any training happens."""
    from spacr.deep_spacr import train_model
    monkeypatch.setattr("spacr.utils.choose_model", lambda *a, **k: None)
    dst = tmp_path / "model"; dst.mkdir()

    out = train_model(str(tmp_path), str(dst), "not_a_real_backbone", _loaders(1),
                      epochs=1, num_classes=2)

    # nothing trained, nothing written. The bail-out must keep the 2-tuple arity
    # of the success path, or the caller's unpack raises TypeError instead.
    assert out == (None, None)
    assert list(dst.iterdir()) == []
    assert "Model not_a_real_backbone not found" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# optimizer_type branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opt_name,module,attr", [
    ("adamw", "spacr.deep_spacr", "AdamW"),
    ("Adam", "torch.optim", "Adam"),          # mixed case -> exercises .lower()
    ("adagrad", "spacr.deep_spacr", "Adagrad"),
    ("sgd", "torch.optim", "SGD"),
    ("rmsprop", "torch.optim", "RMSprop"),
    ("nadam", "torch.optim", "NAdam"),
    ("radam", "torch.optim", "RAdam"),
])
def test_train_model_optimizer_variants(tmp_path, monkeypatch, opt_name, module, attr):
    """Every optimizer branch builds the documented optimizer and steps per batch."""
    import importlib
    from spacr.deep_spacr import train_model

    mod = importlib.import_module(module)
    base = getattr(mod, attr)
    log = {}
    monkeypatch.setattr(mod, attr, _recording_optimizer(base, log))

    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(3), epochs=1,
                learning_rate=3e-4, weight_decay=0.02, optimizer_type=opt_name,
                num_classes=2, schedule=None)

    assert isinstance(log["instance"], base)
    assert log["kwargs"]["lr"] == 3e-4
    assert log["kwargs"]["weight_decay"] == 0.02
    # one optimizer step per batch when accumulation is off
    assert log["steps"] == 3
    assert log["instance"].param_groups[0]["lr"] == 3e-4
    if opt_name == "sgd":
        assert log["kwargs"]["nesterov"] is True and log["kwargs"]["momentum"] == 0.9


def test_train_model_unknown_optimizer_raises(tmp_path, monkeypatch):
    from spacr.deep_spacr import train_model
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    dst = tmp_path / "model"; dst.mkdir()

    with pytest.raises(ValueError, match=r"Unknown optimizer_type: 'lbfgs'"):
        train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=1,
                    optimizer_type="lbfgs", num_classes=2)


# ---------------------------------------------------------------------------
# scheduler branches
# ---------------------------------------------------------------------------

def test_train_model_step_lr_decays_learning_rate(tmp_path, monkeypatch):
    """schedule='step_lr' steps once per epoch with gamma=0.75."""
    from spacr.deep_spacr import train_model
    log = {}
    monkeypatch.setattr("torch.optim.SGD",
                        _recording_optimizer(torch.optim.SGD, log))
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=2,
                learning_rate=0.1, optimizer_type="sgd", schedule="step_lr",
                num_classes=2)

    # step_size = max(1, int(2/5)) = 1 -> gamma applied on both epochs
    assert log["instance"].param_groups[0]["lr"] == pytest.approx(0.1 * 0.75 ** 2)


def test_train_model_cosine_schedule_decays_learning_rate(tmp_path, monkeypatch):
    from spacr.deep_spacr import train_model
    log = {}
    monkeypatch.setattr("torch.optim.SGD",
                        _recording_optimizer(torch.optim.SGD, log))
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=2,
                learning_rate=0.1, optimizer_type="sgd", schedule="cosine",
                num_classes=2)

    lr = log["instance"].param_groups[0]["lr"]
    assert 0.0 < lr < 0.1


def test_train_model_reduce_lr_on_plateau_steps_on_val_loss(tmp_path, monkeypatch):
    """The plateau scheduler is built with mode='min' and fed the val loss."""
    from spacr.deep_spacr import train_model
    built = []

    class FakePlateau:
        def __init__(self, optimizer, **kwargs):
            self.kwargs = kwargs
            self.steps = []
            built.append(self)

        def step(self, metric):
            self.steps.append(float(metric))

    monkeypatch.setattr("torch.optim.lr_scheduler.ReduceLROnPlateau", FakePlateau)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch, acc=0.5, loss=0.42)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=2,
                val_loaders=_loaders(1), schedule="reduce_lr_on_plateau",
                num_classes=2)

    assert len(built) == 1
    assert built[0].kwargs["mode"] == "min"
    assert built[0].kwargs["factor"] == 0.1
    assert built[0].kwargs["patience"] == 10
    # `verbose` was removed in torch 2.5; this fake swallows **kwargs, so pin the
    # exact kwarg set here or a reintroduction would sail past the whole suite.
    assert set(built[0].kwargs) == {"mode", "factor", "patience"}
    # one step per epoch, always with the validation loss
    assert built[0].steps == [0.42, 0.42]


def test_train_model_reduce_lr_on_plateau_works_with_installed_torch(tmp_path, monkeypatch):
    """The real (unstubbed) scheduler must build on the installed torch.

    torch 2.5 removed the ``verbose`` kwarg that train_model used to pass, which
    made this documented schedule raise TypeError before the first batch.
    """
    from spacr.deep_spacr import train_model
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18", _loaders(1),
                                epochs=1, val_loaders=_loaders(1),
                                schedule="reduce_lr_on_plateau", num_classes=2)
    assert path is not None and os.path.exists(path)


# ---------------------------------------------------------------------------
# gradient accumulation
# ---------------------------------------------------------------------------

def _run_for_bias(tmp_path, monkeypatch, *, accumulate, n_batches, steps):
    """Train one epoch on a linear model with loss = logits.mean().

    d(loss)/d(bias_j) is exactly 0.5 per batch regardless of the data, so the
    resulting bias is a precise probe of how many gradients were accumulated
    and whether they were divided by ``gradient_accumulation_steps``.
    """
    from spacr.deep_spacr import train_model
    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    _use_model(monkeypatch, model)
    monkeypatch.setattr("spacr.utils.build_loss",
                        lambda **kw: (lambda logits, target: logits.mean()))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    monkeypatch.setattr("spacr.io._save_model", lambda *a, **k: None)
    dst = tmp_path / "model"; dst.mkdir(parents=True, exist_ok=True)

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(n_batches), epochs=1,
                learning_rate=0.1, weight_decay=0.0, optimizer_type="sgd",
                schedule=None, num_classes=2,
                gradient_accumulation=accumulate,
                gradient_accumulation_steps=steps)
    return float(model.bias[0].item())


def test_gradient_accumulation_scales_loss_and_flushes_leftovers(tmp_path, monkeypatch):
    """2 batches / 4 accumulation steps: no in-loop step, one leftover flush.

    The accumulated gradient is 2 * (0.5 / 4) = 0.25 versus 0.5 for a single
    un-accumulated batch, so the bias must move exactly half as far.
    """
    plain = _run_for_bias(tmp_path / "a", monkeypatch, accumulate=False,
                          n_batches=1, steps=4)
    accum = _run_for_bias(tmp_path / "b", monkeypatch, accumulate=True,
                          n_batches=2, steps=4)
    assert plain < 0 and accum < 0                      # both descended
    assert accum == pytest.approx(0.5 * plain, rel=1e-6)


def test_gradient_accumulation_step_count(tmp_path, monkeypatch):
    """steps=2 with 3 batches -> one in-loop step (batch 2) + one flush."""
    from spacr.deep_spacr import train_model
    log = {}
    monkeypatch.setattr("torch.optim.SGD",
                        _recording_optimizer(torch.optim.SGD, log))
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(3), epochs=1,
                optimizer_type="sgd", schedule=None, num_classes=2,
                gradient_accumulation=True, gradient_accumulation_steps=2)

    assert log["steps"] == 2


def test_gradient_accumulation_no_flush_when_divisible(tmp_path, monkeypatch):
    """steps=2 with 4 batches -> two in-loop steps and no leftover flush."""
    from spacr.deep_spacr import train_model
    log = {}
    monkeypatch.setattr("torch.optim.SGD",
                        _recording_optimizer(torch.optim.SGD, log))
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(4), epochs=1,
                optimizer_type="sgd", schedule=None, num_classes=2,
                gradient_accumulation=True, gradient_accumulation_steps=2)

    assert log["steps"] == 2


# ---------------------------------------------------------------------------
# target / head handling
# ---------------------------------------------------------------------------

def test_one_hot_targets_are_argmaxed_to_indices(tmp_path, monkeypatch):
    """(N, C) one-hot targets reach the loss as (N,) class indices."""
    from spacr.deep_spacr import train_model
    from spacr.utils import build_loss as real_build_loss

    seen = []

    def spy_build_loss(**kwargs):
        fn = real_build_loss(**kwargs)

        def wrapped(logits, target):
            seen.append(tuple(target.shape))
            return fn(logits, target)
        return wrapped

    monkeypatch.setattr("spacr.utils.build_loss", spy_build_loss)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18",
                                _loaders(2, mode="onehot"), epochs=1,
                                num_classes=2, schedule=None)

    assert len(seen) >= 2                       # at least the two train batches
    assert all(len(shape) == 1 for shape in seen), seen
    assert all(shape[0] == 4 for shape in seen)
    assert os.path.exists(path)


def test_head_size_mismatch_raises_runtime_error(tmp_path, monkeypatch):
    """num_classes=3 with a 2-logit head is caught before the loss is computed."""
    from spacr.deep_spacr import train_model
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    dst = tmp_path / "model"; dst.mkdir()

    with pytest.raises(RuntimeError, match=r"Expected logits \(N,3\) for CE, got \(4, 2\)"):
        train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=1,
                    num_classes=3, schedule=None)


def test_single_logit_binary_head_uses_bce_and_binary_metrics(tmp_path, monkeypatch):
    """num_classes=1 keeps float targets and produces binary metric columns."""
    from spacr.deep_spacr import train_model
    model = _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 1)))
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18",
                                _loaders(2, mode="binary"), epochs=1,
                                num_classes=1, schedule=None)

    assert trained is model
    assert trained(torch.rand(3, 4)).shape == (3, 1)
    assert os.path.exists(path)
    df = pd.read_csv(dst / "train.csv")
    # binary metrics (not the multiclass NaN placeholders)
    assert df["neg_accuracy"].notna().all()
    assert df["pos_accuracy"].notna().all()
    assert 0.0 <= df["accuracy"].iloc[0] <= 1.0
    assert df["optimal_threshold"].notna().all()


# ---------------------------------------------------------------------------
# live plotting hook
# ---------------------------------------------------------------------------

def test_plot_true_refreshes_one_live_figure(tmp_path, monkeypatch):
    """plot=True refreshes one monitor instead of retaining every epoch."""
    import matplotlib.pyplot as plt
    from spacr.deep_spacr import train_model
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch, acc=0.6, loss=0.3)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()
    plt.close("all")

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=2,
                val_loaders=_loaders(1), num_classes=2, schedule=None, plot=True)

    figs = plt.get_fignums()
    assert len(figs) == 1
    last = plt.figure(figs[-1])
    assert "epoch 2 / 2" in last._suptitle.get_text()
    # both curves present in the final figure (train + val)
    assert len(last.axes[0].lines) == 2


def test_plot_failure_is_swallowed(tmp_path, monkeypatch):
    """A broken plotting backend must not abort training."""
    from spacr.deep_spacr import train_model
    calls = []

    def boom(train_hist, val_hist, total_epochs, figure=None):
        calls.append((len(train_hist), len(val_hist), total_epochs))
        raise RuntimeError("no display")

    monkeypatch.setattr("spacr.deep_spacr._plot_training_curves", boom)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18", _loaders(1),
                                epochs=2, val_loaders=None, num_classes=2,
                                schedule=None, plot=True)

    assert trained is not None
    # called every epoch, with no val history (val_loaders=None) and growing train history
    assert calls == [(1, 0, 2), (2, 0, 2)]


def test_tensorboard_logs_epoch_scalars_and_flushes():
    """Training and validation metrics use grouped TensorBoard scalar names."""
    from spacr.deep_spacr import _log_tensorboard_epoch

    class Writer:
        def __init__(self):
            self.scalars = []
            self.flushes = 0

        def add_scalar(self, name, value, epoch):
            self.scalars.append((name, value, epoch))

        def flush(self):
            self.flushes += 1

    writer = Writer()
    _log_tensorboard_epoch(
        writer,
        {'loss': 0.3, 'accuracy': 0.8, 'f1_macro': 0.7, 'lr': 1e-4},
        {'loss': 0.4, 'accuracy': 0.75, 'f1_macro': 0.65},
        3,
    )

    assert writer.flushes == 1
    assert ('loss/train', 0.3, 3) in writer.scalars
    assert ('accuracy/validation', 0.75, 3) in writer.scalars
    assert ('f1_macro/validation', 0.65, 3) in writer.scalars
    assert ('learning_rate', 1e-4, 3) in writer.scalars


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

def test_early_stopping_breaks_after_patience(tmp_path, monkeypatch, capsys):
    """A flat validation accuracy trips early stopping on the 2nd epoch."""
    from spacr.deep_spacr import train_model
    calls = []
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch, acc=0.5, loss=0.7, calls=calls)
    saved = _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=6,
                val_loaders=_loaders(1), num_classes=2, schedule=None,
                early_stopping_patience=1)

    epochs_run = sorted({epoch for epoch, _ in calls})
    assert epochs_run == [1, 2]                    # stopped 4 epochs early
    assert len(saved) == 2
    train_df, val_df = saved[-1][1], saved[-1][2]
    assert len(train_df) == 1 and len(val_df) == 1
    out = capsys.readouterr().out
    assert "Early stopping at epoch 2" in out
    assert "Best val acc: 0.5000" in out


def test_suggest_training_changes_failure_is_reported_not_raised(tmp_path, monkeypatch, capsys):
    """A failing training advisor is caught and logged at the final epoch."""
    from spacr.deep_spacr import train_model

    def boom(dst, **kwargs):
        raise ValueError("no csv here")

    monkeypatch.setattr("spacr.utils.suggest_training_changes", boom)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18", _loaders(1),
                                epochs=1, num_classes=2, schedule=None)

    assert trained is not None
    assert "[suggest_training_changes] Skipped at epoch 1: no csv here" in capsys.readouterr().out


def test_suggestion_report_is_printed_at_final_epoch(tmp_path, monkeypatch, capsys):
    """The advisor's summary, flags and suggestions are all echoed."""
    from spacr.deep_spacr import train_model
    report = {
        "summary": {"best_val_acc": 0.87, "epochs": 1},
        "flags": ["overfitting", "plateau"],
        "suggestions": ["lower the learning rate", "add augmentation"],
    }
    monkeypatch.setattr("spacr.utils.suggest_training_changes",
                        lambda dst, **kw: report)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_eval(monkeypatch)
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    train_model(str(tmp_path), str(dst), "resnet18", _loaders(1), epochs=1,
                num_classes=2, schedule=None)

    out = capsys.readouterr().out
    assert "best_val_acc: 0.87" in out
    assert "epochs: 1" in out
    assert "overfitting, plateau" in out
    assert "1. lower the learning rate" in out
    assert "2. add augmentation" in out


def test_validation_improvement_resets_patience_counter(tmp_path, monkeypatch, capsys):
    """Improving val accuracy keeps training alive for all epochs."""
    from spacr.deep_spacr import train_model
    accs = {1: 0.30, 2: 0.55, 3: 0.80}

    def fake_eval(model, loader, epoch, **kw):
        return _metrics(epoch, acc=accs[epoch]), [None, None]

    monkeypatch.setattr("spacr.deep_spacr.evaluate_model_performance", fake_eval)
    _use_model(monkeypatch, nn.Sequential(nn.Linear(4, 2)))
    _stub_progress(monkeypatch, [])
    dst = tmp_path / "model"; dst.mkdir()

    trained, path = train_model(str(tmp_path), str(dst), "resnet18", _loaders(1),
                                epochs=3, val_loaders=_loaders(1), num_classes=2,
                                schedule=None, early_stopping_patience=1)

    out = capsys.readouterr().out
    assert "Early stopping" not in out
    assert "Val acc.: 0.800" in out
    # final epoch checkpoint was written and returned
    assert path == str(dst / "resnet18_epoch_3_channels_rgb.pth")
    assert os.path.exists(path)
