"""CPU coverage for the ``spacr.deep_spacr`` train/test entry point.

Targets :func:`spacr.deep_spacr.train_test_model` (settings canonicalisation,
loss-type auto-selection, the save_settings branch ladder, the train half, the
test half incl. best-checkpoint pickup, CSV emission and misclassified-image
copying) and :func:`spacr.deep_spacr._plot_training_curves`.

Everything runs on tiny constant-output torch modules and hand-built loader
iterables so no real dataset, no torchvision backbone and no GPU is involved.
``generate_loaders`` / ``train_model`` are stubbed where they would otherwise
pull in the heavy image pipeline; the pieces under test (metrics, dataframes,
CSV writing, file copying, matplotlib figure construction) all run for real.
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_leaked_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """These tests must stay CPU-only even on a CUDA box: ``test_model_core``
    picks its device from ``torch.cuda.is_available()``."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


class ConstantClassifier(nn.Module):
    """Deterministic classifier: always votes for class 0, whatever the input.

    Defined at module scope so ``torch.save``/``torch.load`` of the whole
    module object (which is what ``train_test_model`` does when it picks a
    checkpoint) can pickle it by reference.
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        # a real parameter so .eval()/.to(device) exercise real nn.Module code
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.n_classes, device=x.device)
        out[:, 0] = 4.0
        return out * self.scale


def _write_png(path: Path, value: int) -> None:
    """Write a genuine (tiny) PNG so shutil.copy has real bytes to move."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


@pytest.fixture
def split_dir(tmp_path):
    """A dataset root with a populated ``test/`` split (2 classes, 1 image each)."""
    src = tmp_path / "dataset"
    nc = src / "test" / "nc" / "a.png"
    pos = src / "test" / "pc" / "b.png"
    _write_png(nc, 10)
    _write_png(pos, 200)
    return src, nc, pos


def _batch(nc_path: Path, pos_path: Path):
    """One loader batch: (images, labels, filenames) — nc=0, pc=1."""
    return (
        torch.zeros(2, 3, 8, 8),
        torch.tensor([0, 1], dtype=torch.long),
        [str(nc_path), str(pos_path)],
    )


def _base_settings(src: Path, **over):
    s = {
        'src': str(src),
        'model_type': 'tinynet',
        'classes': ['nc', 'pc'],
        'train_channels': ['r', 'g'],
        'epochs': 1,
        'batch_size': 2,
        'image_size': 8,
        'n_jobs': 0,
        'pin_memory': False,
        'verbose': False,
        'augment': False,
        'val_split': 0.0,
    }
    s.update(over)
    return s


# ---------------------------------------------------------------------------
# settings validation / loss-type auto-selection
# ---------------------------------------------------------------------------

def test_empty_classes_raises_value_error(tmp_path):
    """classes=[] must abort before any loader is built (line 576)."""
    from spacr.deep_spacr import train_test_model

    src = tmp_path / "ds"
    src.mkdir()
    settings = _base_settings(src, classes=[], train=False, test=False)

    with pytest.raises(ValueError, match="No classes provided"):
        train_test_model(settings)

    # dst was still created before the guard fired
    assert (src / "model" / "tinynet" / "rg" / "epochs_1").is_dir()


def _stub_train_only(monkeypatch, record, model=None):
    """Stub generate_loaders + train_model so only the entry-point logic runs."""
    import spacr.io as sio
    import spacr.deep_spacr as ds

    def fake_generate_loaders(src, mode='train', **kw):
        record.setdefault('loader_modes', []).append(mode)
        return ['train_loader'], ['val_loader'], None

    def fake_train_model(**kw):
        record['train_kwargs'] = kw
        return (model if model is not None else ConstantClassifier()), '/tmp/best_model.pth'

    monkeypatch.setattr(sio, 'generate_loaders', fake_generate_loaders)
    monkeypatch.setattr(ds, 'train_model', fake_train_model)


@pytest.mark.parametrize(
    "classes,loss_in,expected",
    [
        (['nc', 'pc'], 'auto', 'cross_entropy'),
        (['only'], 'auto', 'binary_cross_entropy_with_logits'),
        (['nc', 'pc'], None, 'cross_entropy'),
    ],
)
def test_loss_type_auto_resolution(tmp_path, monkeypatch, classes, loss_in, expected):
    """'auto'/None loss_type resolves by class count and reaches train_model (579)."""
    from spacr.deep_spacr import train_test_model

    src = tmp_path / "ds"
    src.mkdir()
    record = {}
    _stub_train_only(monkeypatch, record)

    settings = _base_settings(src, classes=classes, loss_type=loss_in,
                              train=True, test=False)
    out = train_test_model(settings)

    assert out == '/tmp/best_model.pth'
    assert record['train_kwargs']['loss_type'] == expected
    assert record['train_kwargs']['num_classes'] == len(classes)
    assert record['loader_modes'] == ['train']
    # train-only => the "train_<model>_<epochs>" settings snapshot was written
    assert (src / 'settings' / 'train_tinynet_1.csv').is_file()


def test_explicit_loss_type_is_not_overwritten(tmp_path, monkeypatch):
    """A concrete loss_type skips the auto branch entirely."""
    from spacr.deep_spacr import train_test_model

    src = tmp_path / "ds"
    src.mkdir()
    record = {}
    _stub_train_only(monkeypatch, record)

    settings = _base_settings(src, loss_type='focal_loss', train=True, test=False)
    train_test_model(settings)

    assert record['train_kwargs']['loss_type'] == 'focal_loss'


def test_train_flag_truthy_but_not_true_still_saves_settings(tmp_path, monkeypatch):
    """train=1 (truthy, not ``True``) with test=False is a train-only run.

    The ladder used to compare with ``is True``, so a scripted caller passing 1
    instead of True fell through every arm and silently lost its snapshot.
    """
    import spacr.utils as sutils
    from spacr.deep_spacr import train_test_model

    src = tmp_path / "ds"
    src.mkdir()
    record = {}
    _stub_train_only(monkeypatch, record)

    saved = []
    monkeypatch.setattr(sutils, 'save_settings',
                        lambda s, name='settings', show=False: saved.append(name))

    settings = _base_settings(src, train=1, test=False, loss_type='cross_entropy')
    out = train_test_model(settings)

    assert saved == ['train_tinynet_1']     # the train-only arm fired
    assert out == '/tmp/best_model.pth'     # and training still ran
    assert record['loader_modes'] == ['train']


# ---------------------------------------------------------------------------
# the test half: loaders, metrics, CSVs, misclassified copy
# ---------------------------------------------------------------------------

def _install_test_loader(monkeypatch, batch, record):
    import spacr.io as sio

    def fake_generate_loaders(src, mode='train', **kw):
        record.setdefault('calls', []).append((mode, kw))
        if mode == 'test':
            return [batch], None, None
        return [batch], [batch], None

    monkeypatch.setattr(sio, 'generate_loaders', fake_generate_loaders)


def test_test_only_run_loads_best_checkpoint_and_writes_results(
        split_dir, monkeypatch, capsys):
    """train=False/test=True: pick_best_model + torch.load + metrics + CSVs +
    misclassified copy + the ``return result_loc`` tail (637-672, 680-681)."""
    from spacr.deep_spacr import train_test_model

    src, nc_png, pos_png = split_dir
    model_dir = src / "model"
    model_dir.mkdir(parents=True)
    # two checkpoints; the higher-accuracy one must win
    torch.save(ConstantClassifier(), model_dir / "ck_epoch_1_acc_0.10.pth")
    torch.save(ConstantClassifier(), model_dir / "ck_epoch_9_acc_0.97.pth")

    record = {}
    _install_test_loader(monkeypatch, _batch(nc_png, pos_png), record)

    settings = _base_settings(src, train=False, test=True,
                              loss_type='cross_entropy')
    result_loc = train_test_model(settings)

    # --- the test loader was requested with the hard-coded test-mode args ---
    (mode, kw), = record['calls']
    assert mode == 'test'
    assert kw['validation_split'] == 0.0
    assert kw['augment'] is False
    assert kw['classes'] == ['nc', 'pc']

    # --- best checkpoint selection ---
    out = capsys.readouterr().out
    assert 'ck_epoch_9_acc_0.97.pth' in out
    assert 'ck_epoch_1_acc_0.10.pth' not in out

    # --- returned path is the per-file result CSV inside dst ---
    today = datetime.date.today().strftime('%y%m%d')
    dst = src / 'model' / 'tinynet' / 'rg' / 'epochs_1'
    assert result_loc == f"{dst}/tinynet_time_{today}_test_result.csv"
    assert os.path.isfile(result_loc)

    summary = pd.read_csv(result_loc)
    assert len(summary) == 1
    # constant class-0 predictions against labels [0, 1] => 50% accuracy
    assert summary.loc[0, 'accuracy'] == pytest.approx(0.5)
    assert summary.loc[0, 'epoch'] == 1

    acc_loc = f"{dst}/tinynet_time_{today}_test_acc.csv"
    per_file = pd.read_csv(acc_loc)
    assert list(per_file['true_label']) == [0, 1]
    assert list(per_file['predicted_label']) == [0, 0]
    assert set(per_file['filename']) == {str(nc_png), str(pos_png)}

    # --- exactly the class-1 image was copied into the review folder ---
    assert 'Copied 1 misclassified images.' in out
    copied = src / 'test' / 'missclassified' / 'pc' / 'b.png'
    assert copied.is_file()
    assert copied.read_bytes() == pos_png.read_bytes()
    assert not (src / 'test' / 'missclassified' / 'nc').exists()


def test_train_and_test_run_reuses_trained_model_and_returns_model_path(
        split_dir, monkeypatch, capsys):
    """train=True and test=True: the combined settings snapshot is written
    (line 583), the in-memory model is reused instead of a checkpoint, and the
    model path (not the CSV path) is returned."""
    import spacr.deep_spacr as ds
    import spacr.utils as sutils
    from spacr.deep_spacr import train_test_model

    src, nc_png, pos_png = split_dir
    record = {}
    _install_test_loader(monkeypatch, _batch(nc_png, pos_png), record)

    trained = ConstantClassifier()
    monkeypatch.setattr(ds, 'train_model',
                        lambda **kw: (record.setdefault('train_kwargs', kw),
                                      (trained, str(src / 'best.pth')))[1])

    def _boom(_src):
        raise AssertionError("pick_best_model must not run when a model is in hand")

    monkeypatch.setattr(sutils, 'pick_best_model', _boom)

    settings = _base_settings(src, train=True, test=True,
                              loss_type='cross_entropy')
    out = train_test_model(settings)

    assert out == str(src / 'best.pth')
    assert [c[0] for c in record['calls']] == ['train', 'test']
    assert (src / 'settings' / 'train_test_tinynet_1.csv').is_file()

    today = datetime.date.today().strftime('%y%m%d')
    dst = src / 'model' / 'tinynet' / 'rg' / 'epochs_1'
    assert (dst / f'tinynet_time_{today}_test_result.csv').is_file()
    assert (dst / f'tinynet_time_{today}_test_acc.csv').is_file()


def test_test_only_run_snapshots_its_settings(split_dir, monkeypatch):
    """A test-only run should still write ``test_<model>_<epochs>.csv`` so the
    evaluation is reproducible — that is what the third arm of the save_settings
    ladder was written to do, and it was unreachable while the ladder sat inside
    ``if settings['train']:``."""
    from spacr.deep_spacr import train_test_model

    src, nc_png, pos_png = split_dir
    model_dir = src / "model"
    model_dir.mkdir(parents=True)
    torch.save(ConstantClassifier(), model_dir / "ck_epoch_1_acc_0.80.pth")

    _install_test_loader(monkeypatch, _batch(nc_png, pos_png), {})
    train_test_model(_base_settings(src, train=False, test=True,
                                    loss_type='cross_entropy'))

    assert (src / 'settings' / 'test_tinynet_1.csv').is_file()


def test_neither_train_nor_test_returns_none_and_still_prepares_dst(
        tmp_path, monkeypatch):
    """train=False/test=False falls off the end of the function -> None."""
    import spacr.io as sio
    from spacr.deep_spacr import train_test_model

    src = tmp_path / "ds"
    src.mkdir()

    def _boom(*a, **k):
        raise AssertionError("no loader should be built")

    monkeypatch.setattr(sio, 'generate_loaders', _boom)

    settings = _base_settings(src, train=False, test=False,
                              loss_type='cross_entropy')
    assert train_test_model(settings) is None
    assert settings['dst'] == str(src / 'model' / 'tinynet' / 'rg' / 'epochs_1')
    assert os.path.isdir(settings['dst'])


# ---------------------------------------------------------------------------
# _plot_training_curves
# ---------------------------------------------------------------------------

@pytest.fixture
def captured_figs(monkeypatch):
    """Intercept plt.show() and hand back the figure it would have displayed."""
    import matplotlib.pyplot as plt
    figs = []
    monkeypatch.setattr(plt, 'show', lambda *a, **k: figs.append(plt.gcf()))
    return figs


def test_plot_training_curves_empty_history_is_a_noop(captured_figs):
    """No train history -> early return, no figure at all (lines 690-691)."""
    import matplotlib.pyplot as plt
    from spacr.deep_spacr import _plot_training_curves

    plt.close('all')
    assert _plot_training_curves([], [{'epoch': 1, 'loss': 0.5}]) is None
    assert captured_figs == []
    assert plt.get_fignums() == []


def test_plot_training_curves_train_only_defaults_epoch_and_accuracy(captured_figs):
    """Missing 'epoch' falls back to the enumerate index; missing metrics become
    NaN; no val history means only one line per axis and no ' / N' suffix."""
    from spacr.deep_spacr import _plot_training_curves

    train_hist = [
        {'loss': 1.0, 'accuracy': 0.20},
        {'loss': 0.5, 'accuracy': 0.60},
        {'loss': 0.25},                      # no 'accuracy' -> NaN
    ]
    _plot_training_curves(train_hist, None)

    assert len(captured_figs) == 1
    fig = captured_figs[0]
    # Three panels since C10: loss, aggregate accuracy, and per-class
    # accuracy. This history carries no per-class metrics, so the third panel
    # says so rather than being left off (an absent panel and an absent class
    # breakdown would look the same).
    ax1, ax2, ax3 = fig.axes
    assert (ax1.get_title(), ax2.get_title()) == ('Loss', 'Accuracy')
    assert ax3.get_title() == 'Per-class accuracy'
    assert len(ax3.lines) == 0
    assert any('no per-class metrics' in t.get_text() for t in ax3.texts)
    assert len(ax1.lines) == 1 and len(ax2.lines) == 1

    assert list(ax1.lines[0].get_xdata()) == [1, 2, 3]      # epoch defaulted
    assert list(ax1.lines[0].get_ydata()) == [1.0, 0.5, 0.25]

    acc = np.asarray(ax2.lines[0].get_ydata(), dtype=float)
    assert acc[:2].tolist() == [0.20, 0.60]
    assert np.isnan(acc[2])

    assert ax2.get_ylim() == (0, 1.02)
    assert ax1.get_xlabel() == 'epoch'
    assert fig.texts[0].get_text() == 'Training — epoch 3'


def test_plot_training_curves_with_val_history_and_total_epochs(captured_figs):
    """A val history adds a second line to each axis and total_epochs adds the
    ' / N' suffix to the suptitle (lines 699-704, 709)."""
    from spacr.deep_spacr import _plot_training_curves

    train_hist = [{'epoch': 1, 'loss': 0.9, 'accuracy': 0.4},
                  {'epoch': 2, 'loss': 0.4, 'accuracy': 0.8}]
    val_hist = [{'epoch': 1, 'loss': 1.1, 'accuracy': 0.3},
                {'epoch': 2, 'accuracy': 0.7}]        # no 'loss' -> NaN

    fig1 = _plot_training_curves(train_hist, val_hist, total_epochs=10)

    assert len(captured_figs) == 1
    fig = captured_figs[0]
    ax1, ax2, _ax3 = fig.axes
    assert len(ax1.lines) == 2 and len(ax2.lines) == 2

    assert list(ax1.lines[1].get_xdata()) == [1, 2]
    val_loss = np.asarray(ax1.lines[1].get_ydata(), dtype=float)
    assert val_loss[0] == pytest.approx(1.1)
    assert np.isnan(val_loss[1])
    assert list(ax2.lines[1].get_ydata()) == [0.3, 0.7]

    assert ax1.lines[1].get_label() == 'val'
    assert ax1.get_legend() is not None and ax2.get_legend() is not None
    assert fig.texts[0].get_text() == 'Training — epoch 2 / 10'

    # A subsequent epoch redraws the same object so the Qt gallery can refresh
    # one live monitor rather than append another static snapshot.
    train_hist.append({'epoch': 3, 'loss': 0.2, 'accuracy': 0.9})
    fig2 = _plot_training_curves(
        train_hist, val_hist, total_epochs=10, figure=fig1)
    assert fig2 is fig1
    assert fig2._spacr_live_update is True
    assert list(fig2.axes[0].lines[0].get_xdata()) == [1, 2, 3]
