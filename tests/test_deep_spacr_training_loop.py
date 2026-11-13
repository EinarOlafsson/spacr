"""The classifier training loop of ``spacr.deep_spacr`` and what it reports.

One table-driven test per unit (three for ``train_model``), each row run as a
subtest so a failing case names itself.  Everything is CPU-only and offline:
``choose_model`` hands back a 4-feature ``nn.Linear`` and the "loaders" are
plain lists of ``(data, target, filenames)`` tuples, so a whole run costs
milliseconds.  A zero-weight head trained at learning rate 0 keeps every
logit at 0, which pins the recorded loss at ln 2 and accuracy at 0.5.

Units: train_model, train_test_model, _plot_training_curves,
_log_tensorboard_epoch and _flowview_event.
"""
from __future__ import annotations

import contextlib
import datetime
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import deep_spacr as D  # noqa: E402
# Imported at collection time so the package is in ``sys.modules`` before a
# test swaps ``spacr.flowview.trace`` for a double.
import spacr.flowview  # noqa: E402,F401

LN2 = math.log(2.0)

#: What ``pick_device`` says when a busy card sent the run to the CPU.
_NOTE = ("The GPU has 300 MiB free of 24576 and this run needs about 2000 "
         "MiB, so it is running on the CPU instead.")


@pytest.fixture(autouse=True)
def _cpu_and_no_figures(monkeypatch):
    """Pin training to the CPU and never leak live-curve figures."""
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    plt.close("all")
    yield
    plt.close("all")


# ===========================================================================
# helpers
# ===========================================================================

def _loader(n_batches=2, batch=4, n_classes=2, mode="index", seed=0):
    """``n_batches`` batches with labels cycling 0..n_classes-1."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for b in range(n_batches):
        x = torch.rand(batch, 4, generator=g)
        idx = torch.tensor([i % n_classes for i in range(batch)])
        y = idx.float() if mode == "binary" else idx
        out.append((x, y, [f"b{b}_{i}.png" for i in range(batch)]))
    return out


def _zero(out_dim=2):
    """A linear head whose logits are all 0."""
    model = nn.Linear(4, out_dim)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def _metrics(epoch, acc=0.5, loss=0.7, **extra):
    """An evaluation dict without a per-class breakdown."""
    return {"accuracy": float(acc), "loss": float(loss), "f1_macro": float(acc),
            "epoch": int(epoch), **extra}


def _stub_eval(mp, accs=None, acc=0.5, loss=0.7, calls=None):
    """Replace evaluate_model_performance; ``accs`` maps epoch -> accuracy."""
    def fake(model, loader, epoch, **kw):
        if calls is not None:
            calls.append(epoch)
        a = accs[epoch] if accs else acc
        return _metrics(epoch, a, loss), [np.array([0.5, 0.5]), [0, 1]]

    mp.setattr(D, "evaluate_model_performance", fake)


def _record_saves(mp, saves):
    """Replace _save_model with a recorder returning ``<dst>/epoch_<n>.pth``."""
    def fake(model, model_type, train_dict, dst, epoch, epochs, **kw):
        saves.append({"epoch": epoch, **kw})
        return os.path.join(dst, f"epoch_{epoch}.pth")

    mp.setattr("spacr.io._save_model", fake)


def _record_progress(mp, flushes):
    """Replace _save_progress with a recorder of the two frames."""
    mp.setattr("spacr.io._save_progress",
               lambda dst, train_df, val_df: flushes.append((train_df, val_df)))


def _train(mp, root, model, *, src=None, loader=None, **kw):
    """Run train_model on ``model`` with quiet defaults; returns (out, dst)."""
    mp.setattr("spacr.utils.choose_model", lambda *a, **k: model)
    root = Path(root)
    src = Path(src) if src is not None else root / "data"
    src.mkdir(parents=True, exist_ok=True)
    dst = root / "model"
    dst.mkdir(parents=True, exist_ok=True)
    args = {"epochs": 1, "num_classes": 2, "schedule": None,
            "tensorboard": False, "write_card": False}
    args.update(kw)
    out = D.train_model(str(src), str(dst), "resnet18",
                        loader if loader is not None else _loader(), **args)
    return out, dst


def _recording_optimizer(base, log):
    """Subclass ``base`` so a test sees its kwargs and its step count."""
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


# ===========================================================================
# train_model: progress, classes and the returned checkpoint
# ===========================================================================

def test_train_model_writes_progress_and_returns_the_best_checkpoint(
        subtests, tmp_path, capsys):
    """A real run writes one train row per epoch (with exact metrics), its
    checkpoints, card and TensorBoard log; class names come from
    ``src/train`` folders, else from the caller, and class counts from the
    folders, the dataset labels or the loader; full-pass metrics, the device
    notice and the validation frame flushed beside each training row."""
    for cid, out_dim, mode, extra in (
            ("two-logit-head", 2, "index",
             {"prauc": 0.5, "per_class_accuracy": "[1.0, 0.0]"}),
            ("single-logit-head", 1, "binary",
             {"neg_accuracy": 0.0, "pos_accuracy": 1.0, "prauc": 0.75,
              "optimal_threshold": 0.5, "per_class_accuracy": "[0.0, 1.0]"})):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            capsys.readouterr()
            model = _zero(out_dim)
            (trained, path), dst = _train(
                mp, tmp_path / cid, model, loader=_loader(2, mode=mode),
                num_classes=out_dim, learning_rate=0.0, test_loaders=_loader(1),
                tensorboard=True, write_card=True)
            assert trained is model
            assert path == str(dst / "resnet18_best_channels_rgb.pth")
            assert sorted(os.listdir(dst)) == [
                "artifacts.db", "resnet18_best_channels_rgb.card.json",
                "resnet18_best_channels_rgb.card.md",
                "resnet18_best_channels_rgb.pth",
                "resnet18_last_channels_rgb.pth", "tensorboard", "train.csv"]
            row = pd.read_csv(dst / "train.csv").iloc[0]
            assert row["epoch"] == 1 and row["num_classes"] == 2
            assert row["accuracy"] == 0.5 and row["Accuracy"] == 0.5
            assert row["loss"] == pytest.approx(LN2, rel=1e-6)
            assert row["f1_macro"] == pytest.approx(1 / 3)
            assert row["lr"] == 0.0
            assert row["class_support"] == "[4, 4]"
            assert row["class_names"] == "['class_0', 'class_1']"
            for key, value in extra.items():
                assert row[key] == value, key
            if out_dim == 2:
                assert math.isnan(row["neg_accuracy"])
            worst = "class_1" if out_dim == 2 else "class_0"
            accs = ("1.000", "0.000") if out_dim == 2 else ("0.000", "1.000")
            out = capsys.readouterr().out
            for line in (
                    "Train batches:2, Validation batches:0",
                    "Test batches:1",
                    "Using cpu for Torch",
                    "Progress: 1/1, operation_type: Training, Train Loss: 0.693, "
                    "Train acc.: 0.500, Train F1(macro): 0.333",
                    f"  Train per-class acc.: class_0 {accs[0]} (n=4), class_1 "
                    f"{accs[1]} (n=4)  <- WORST: {worst} at 0.000, 1.000 below "
                    "the best class",
                    f"Model card: {dst}/resnet18_best_channels_rgb.card.json"):
                assert line in out.splitlines(), line
            assert "Val Loss" not in out

    # The card written after training: per-split class balance from the
    # train/<class> folders, the imbalance note at a >= 70% share only, and
    # the held-out section built from the validation pass.
    for cid, (n_nc, n_pc), note in (
            ("card-balanced-enough-no-note", (3, 5), None),
            ("card-dominant-class-gets-a-note", (3, 7),
             "> Class 'pc' is 70% of the training set and 'nc' is 30%. A model "
             "that always answered 'pc' would score 70% accuracy, so read the "
             "per-class numbers, not the aggregate.")):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            src = tmp_path / cid / "data"
            for cls, n in (("nc", n_nc), ("pc", n_pc)):
                (src / "train" / cls).mkdir(parents=True)
                for i in range(n):
                    (src / "train" / cls / f"{i}.png").write_bytes(b"x")
            _, dst = _train(mp, tmp_path / cid, _zero(), src=src, learning_rate=0.0,
                            val_loaders=_loader(1), write_card=True,
                            settings={"val_split": 0.2})
            card = (dst / "resnet18_best_channels_rgb.card.md").read_text().splitlines()
            total = n_nc + n_pc
            start = card.index(f"**train** ({total} images)")
            assert card[start:start + 6] == [
                f"**train** ({total} images)", "", "| class | n | share |",
                "| --- | ---: | ---: |", f"| nc | {n_nc} | {n_nc / total:.1%} |",
                f"| pc | {n_pc} | {n_pc / total:.1%} |"]
            notes = [line for line in card if line.startswith("> Class")]
            assert notes == ([note] if note else [])
            assert "n = 4 · accuracy 0.5000 · macro-F1 0.3333" in card
            assert ("validation split held out by generate_loaders (val_split=0.2, "
                    "grouped by well)") in card
            capsys.readouterr()

    with subtests.test(case="card-write-failure-keeps-the-weights"), \
            pytest.MonkeyPatch.context() as mp:
        def boom(*a, **k):
            raise OSError("disk full")

        mp.setattr(D, "model_card", boom)
        capsys.readouterr()
        (_, path), dst = _train(mp, tmp_path / "card-fails", _zero(),
                                learning_rate=0.0, write_card=True)
        assert capsys.readouterr().out.splitlines()[-1] == (
            f"Could not write the model card for {path} (OSError: disk full). "
            "The weights are unaffected.")
        assert os.path.isfile(path)

    with subtests.test(case="class-names-and-counts-from-train-folders"), \
            pytest.MonkeyPatch.context() as mp:
        capsys.readouterr()
        src = tmp_path / "folders" / "data"
        for cls, n in (("nc", 3), ("pc", 5)):
            (src / "train" / cls).mkdir(parents=True)
            for i in range(n):
                (src / "train" / cls / f"{cls}_{i}.png").write_bytes(b"x")
        (src / "train" / ".hidden").mkdir()
        saves = []
        _record_saves(mp, saves)
        _train(mp, tmp_path / "folders", _zero(), src=src, learning_rate=0.0,
               classes=["ignored_a", "ignored_b"])
        out = capsys.readouterr().out
        assert "Class counts (from folders): {'nc': 3, 'pc': 5}" in out
        assert ("  Train per-class acc.: nc 1.000 (n=4), pc 0.000 (n=4)  <- WORST: "
                "pc at 0.000, 1.000 below the best class") in out.splitlines()
        assert [s["classes"] for s in saves] == [["nc", "pc"]]

    class _Labelled:
        """A loader whose dataset carries its labels."""

        def __init__(self, batches, labels):
            self._batches = batches
            self.dataset = type("DS", (), {"labels": labels,
                                           "__len__": lambda s: len(labels)})()

        def __iter__(self):
            return iter(self._batches)

        def __len__(self):
            return len(self._batches)

    for cid, classes, loader, want_counts, want_line in (
            ("folderless-names-kept", ["nc", "pc"], _loader(1),
             [{"head_dim": 2, "classes": ["nc", "pc"]}], None),
            ("folderless-no-names", None, _loader(1), [], None),
            ("folderless-counts-from-dataset-labels", ["nc", "pc"],
             _Labelled(_loader(1), [0, 1, 1, 1, 1, 1, 0, 1]), [],
             "Class counts (from the training labels): {'nc': 2, 'pc': 6}")):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            capsys.readouterr()
            counted, saves = [], []
            mp.setattr("spacr.utils.estimate_class_counts",
                       lambda loaders, head_dim, src=None, classes=None:
                       counted.append({"head_dim": head_dim, "classes": classes,
                                       "src": src}))
            _record_saves(mp, saves)
            _, dst = _train(mp, tmp_path / cid, _zero(), loader=loader,
                            classes=classes)
            for c in counted:
                assert c.pop("src") == str(tmp_path / cid / "data" / "train")
            assert counted == want_counts
            assert [s["classes"] for s in saves] == [classes]
            out = capsys.readouterr().out
            assert (want_line in out.splitlines()) if want_line else \
                "Class counts" not in out

    with subtests.test(case="full-pass-metrics-without-per-class-breakdown"), \
            pytest.MonkeyPatch.context() as mp:
        capsys.readouterr()
        calls = []
        _stub_eval(mp, acc=0.75, loss=0.5, calls=calls)
        (_, path), dst = _train(mp, tmp_path / "fullpass", _zero(),
                                classes=["neg", "pos"],
                                settings={"train_metrics_full_pass": True})
        out = capsys.readouterr().out
        assert calls == [1]
        assert ("Progress: 1/1, operation_type: Training, Train Loss: 0.500, "
                "Train acc.: 0.750, Train F1(macro): 0.750") in out.splitlines()
        assert "per-class acc." not in out
        assert pd.read_csv(dst / "train.csv")["accuracy"].tolist() == [0.75]
        assert path == str(dst / "resnet18_best_channels_rgb.pth")

    with subtests.test(case="device-notice-and-full-precision"), \
            pytest.MonkeyPatch.context() as mp:
        capsys.readouterr()
        mp.setattr(D, "pick_device", lambda *a, **k: (torch.device("cpu"), _NOTE))
        (_, path), _ = _train(mp, tmp_path / "notice", _zero(),
                              settings={"mixed_precision": True})
        out = capsys.readouterr().out
        assert out.index(_NOTE) < out.index("Using cpu for Torch")
        assert "training in full precision instead." in out
        assert path.endswith("resnet18_best_channels_rgb.pth")

    for cid, val, want in (("flush-train-only", None, [(1, None), (2, None)]),
                           ("flush-with-validation", _loader(1), [(1, 1), (2, 2)])):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            flushes = []
            _record_progress(mp, flushes)
            _stub_eval(mp)
            _train(mp, tmp_path / cid, _zero(), epochs=2, val_loaders=val)
            got = [(int(t["epoch"].iloc[0]),
                    None if v is None else int(v["epoch"].iloc[0]))
                   for t, v in flushes]
            assert got == want
            assert all(len(t) == 1 and (v is None or len(v) == 1) for t, v in flushes)


# ===========================================================================
# train_model: best-checkpoint selection, stopping, plots and reports
# ===========================================================================

def test_train_model_selects_the_best_epoch_stops_early_and_reports(
        subtests, tmp_path, capsys):
    """Validation accuracy (else training accuracy) picks the best epoch,
    patience counts epochs without improvement and stops the run, a resume
    restores (or, when unrecorded, resets) the bar to beat, and the live plot,
    TensorBoard and training advisor never take the run down."""
    cases = [
        # (id, val accs per epoch, epochs, patience, is_best, without, last force)
        ("flat-validation-stops-at-patience", {1: 0.5, 2: 0.5, 3: 0.5}, 6, 1,
         [True, False], [0, 1], "Early stopping at epoch 2: no val improvement "
                                "for 1 epochs. Best val acc: 0.5000"),
        ("improving-validation-resets-patience", {1: 0.3, 2: 0.55, 3: 0.8}, 3, 1,
         [True, True, True], [0, 0, 0], None),
        ("dip-then-recover-within-patience", {1: 0.6, 2: 0.4, 3: 0.7}, 3, 2,
         [True, False, True], [0, 1, 0], None),
    ]
    for cid, accs, epochs, patience, best, without, stop_line in cases:
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            capsys.readouterr()
            calls, saves = [], []
            _stub_eval(mp, accs=accs, calls=calls)
            _record_saves(mp, saves)
            _record_progress(mp, [])
            (_, path), dst = _train(mp, tmp_path / cid, _zero(), epochs=epochs,
                                    val_loaders=_loader(1),
                                    early_stopping_patience=patience)
            n = len(best)
            assert calls == list(range(1, n + 1))
            assert [s["is_best"] for s in saves] == best
            assert [s["epochs_without_improvement"] for s in saves] == without
            assert [s["force_last"] for s in saves] == \
                [False] * (n - 1) + [stop_line is not None]
            last_best = max(i + 1 for i, b in enumerate(best) if b)
            assert path == str(dst / f"epoch_{last_best}.pth")
            assert [s["best_metric"] for s in saves] == [
                max(accs[e] for e in range(1, i + 2)) for i in range(n)]
            out = capsys.readouterr().out
            assert (f"Val acc.: {accs[n]:.3f}") in out
            if stop_line:
                assert stop_line in out.splitlines()
            else:
                assert "Early stopping" not in out

    checkpoint = tmp_path / "resume.pth"
    checkpoint.write_bytes(b"stub")
    for cid, state, first_best, first_metric, want_path in (
            ("resume-keeps-the-recorded-bar", {"epoch": 1, "best_metric": 0.9},
             False, 0.9, str(checkpoint)),
            ("resume-without-a-bar-starts-from-scratch",
             {"epoch": 1, "best_metric": None}, True, 0.5, "epoch_2.pth")):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            capsys.readouterr()
            saves = []
            _stub_eval(mp, acc=0.5)
            _record_saves(mp, saves)
            _record_progress(mp, [])
            loaded = nn.Linear(4, 2)
            loaded.num_classes = 2
            mp.setattr(D, "load_model_artifact",
                       lambda *a, **k: (loaded, {"optimizer_state_dict": {"x": 1}}))
            mp.setattr(D, "restore_training_state", lambda *a, **k: state)
            (trained, path), dst = _train(
                mp, tmp_path / cid, _zero(), epochs=4, val_loaders=_loader(1),
                resume_checkpoint=str(checkpoint))
            assert trained is loaded
            assert [s["epoch"] for s in saves] == [2, 3, 4]
            assert saves[0]["is_best"] is first_best
            assert saves[0]["best_metric"] == first_metric
            assert path == (want_path if os.path.isabs(want_path)
                            else str(dst / want_path))
            assert f"Resuming training state from {checkpoint}" in \
                capsys.readouterr().out.splitlines()

    with subtests.test(case="plot-refreshes-one-live-figure"), \
            pytest.MonkeyPatch.context() as mp:
        _stub_eval(mp, acc=0.6, loss=0.3)
        _record_progress(mp, [])
        plt.close("all")
        _train(mp, tmp_path / "plot", _zero(), epochs=2, val_loaders=_loader(1),
               plot=True)
        assert len(plt.get_fignums()) == 1
        fig = plt.figure(plt.get_fignums()[0])
        assert fig._suptitle.get_text() == "Training — epoch 2 / 2"
        assert [line.get_label() for line in fig.axes[0].lines] == ["train", "val"]

    with subtests.test(case="plot-failure-is-swallowed"), \
            pytest.MonkeyPatch.context() as mp:
        calls = []

        def boom(train_hist, val_hist, total_epochs, figure=None):
            calls.append((len(train_hist), len(val_hist), total_epochs))
            raise RuntimeError("no display")

        mp.setattr(D, "_plot_training_curves", boom)
        _stub_eval(mp)
        _record_progress(mp, [])
        (trained, path), dst = _train(mp, tmp_path / "plotfail", _zero(),
                                      epochs=2, plot=True)
        assert calls == [(1, 0, 2), (2, 0, 2)]
        assert path == str(dst / "resnet18_best_channels_rgb.pth")

    with subtests.test(case="tensorboard-error-closes-the-writer"), \
            pytest.MonkeyPatch.context() as mp:
        capsys.readouterr()
        closed = []

        class Broken:
            def add_scalar(self, *a):
                raise OSError("disk full")

            def close(self):
                closed.append(True)
                raise OSError("still full")

        mp.setattr(D, "_open_tensorboard_writer", lambda dst, enabled=True:
                   (Broken(), str(dst)))
        _record_progress(mp, [])
        _train(mp, tmp_path / "tb", _zero(), epochs=2)
        out = capsys.readouterr().out
        assert out.splitlines().count(
            "TensorBoard logging disabled after an error: disk full") == 1
        assert closed == [True]

    report = {"summary": {"best_val_acc": 0.87, "epochs": 1},
              "flags": ["overfitting", "plateau"],
              "suggestions": ["lower the learning rate", "add augmentation"]}
    for cid, advisor, want in (
            ("advisor-report-printed-at-final-epoch", lambda dst, **k: report,
             ["== Summary ==", "best_val_acc: 0.87", "epochs: 1", "",
              "== Flags ==", "overfitting, plateau", "", "== Suggestions ==",
              "1. lower the learning rate", "2. add augmentation"]),
            ("advisor-without-flags-says-none",
             lambda dst, **k: {"summary": {}, "flags": [], "suggestions": []},
             ["== Summary ==", "", "== Flags ==", "none", "", "== Suggestions =="]),
            ("advisor-failure-is-reported-not-raised",
             lambda dst, **k: (_ for _ in ()).throw(ValueError("no csv here")),
             ["[suggest_training_changes] Skipped at epoch 1: no csv here"])):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            capsys.readouterr()
            mp.setattr("spacr.utils.suggest_training_changes", advisor)
            _stub_eval(mp)
            _record_progress(mp, [])
            _train(mp, tmp_path / cid, _zero())
            lines = capsys.readouterr().out.splitlines()
            start = lines.index(want[0])
            assert lines[start:start + len(want)] == want


# ===========================================================================
# train_model: optimizers, schedules, accumulation and refusals
# ===========================================================================

def test_train_model_builds_optimizers_schedules_and_refuses_bad_setups(
        subtests, tmp_path):
    """Every ``optimizer_type`` builds its optimizer with the run's learning
    rate and weight decay (case-insensitive) and steps once per batch, or once
    per ``gradient_accumulation_steps`` plus a leftover flush with the loss
    divided by the steps; every schedule moves the logged learning rate; an
    unknown optimizer, an unbuildable backbone, a mismatched head and an
    unusable resume checkpoint are refused by name."""
    optimizers = [("adamw", D, "AdamW"), ("Adam", torch.optim, "Adam"),
                  ("adamax", torch.optim, "Adamax"), ("adagrad", D, "Adagrad"),
                  ("adadelta", torch.optim, "Adadelta"),
                  ("asgd", torch.optim, "ASGD"), ("SGD", torch.optim, "SGD"),
                  ("rmsprop", torch.optim, "RMSprop"),
                  ("nadam", torch.optim, "NAdam"), ("radam", torch.optim, "RAdam")]
    for name, module, attr in optimizers:
        with subtests.test(optimizer=name), pytest.MonkeyPatch.context() as mp:
            base = getattr(module, attr)
            log = {}
            mp.setattr(module, attr, _recording_optimizer(base, log))
            _stub_eval(mp)
            _record_progress(mp, [])
            _train(mp, tmp_path / f"opt_{name}", _zero(), loader=_loader(3),
                   learning_rate=3e-4, weight_decay=0.02, optimizer_type=name)
            assert isinstance(log["instance"], base)
            assert log["kwargs"]["lr"] == 3e-4
            assert log["kwargs"]["weight_decay"] == 0.02
            assert log["steps"] == 3
            if name == "SGD":
                assert (log["kwargs"]["momentum"], log["kwargs"]["nesterov"]) == (0.9, True)
            if name in ("adamw", "Adam"):
                assert log["kwargs"]["amsgrad"] is False

    eta = 1e-7
    schedules = [
        (None, [0.1, 0.1], 0.1),
        ("step_lr", [0.1, 0.075], 0.1 * 0.75 ** 2),
        ("cosine", [0.1, eta + (0.1 - eta) / 2], eta),
        ("cosine_warm_restarts", [0.1, 0.1], 0.1),
        ("exponential", [0.1, 0.095], 0.1 * 0.95 ** 2),
        ("linear", [0.1, 0.055], 0.01),
        ("reduce_lr_on_plateau", [0.1, 0.1], 0.1),
    ]
    for schedule, logged, final in schedules:
        with subtests.test(schedule=schedule), pytest.MonkeyPatch.context() as mp:
            log, flushes = {}, []
            mp.setattr(torch.optim, "SGD", _recording_optimizer(torch.optim.SGD, log))
            _stub_eval(mp)
            _record_progress(mp, flushes)
            (_, path), _ = _train(mp, tmp_path / f"sched_{schedule}", _zero(),
                                  loader=_loader(1), epochs=2, learning_rate=0.1,
                                  optimizer_type="sgd", schedule=schedule,
                                  val_loaders=_loader(1))
            assert [float(t["lr"].iloc[0]) for t, _v in flushes] == \
                pytest.approx(logged, rel=1e-9)
            assert log["instance"].param_groups[0]["lr"] == pytest.approx(final, rel=1e-9)
            assert os.path.isfile(path)

    with subtests.test(schedule="plateau-is-fed-the-validation-loss"), \
            pytest.MonkeyPatch.context() as mp:
        built = []

        class FakePlateau:
            def __init__(self, optimizer, **kwargs):
                self.kwargs, self.steps = kwargs, []
                built.append(self)

            def step(self, metric):
                self.steps.append(float(metric))

        mp.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", FakePlateau)
        _stub_eval(mp, loss=0.42)
        _record_progress(mp, [])
        _train(mp, tmp_path / "plateau", _zero(), epochs=2,
               val_loaders=_loader(1), schedule="reduce_lr_on_plateau")
        assert len(built) == 1
        assert built[0].kwargs == {"mode": "min", "factor": 0.1, "patience": 10}
        assert built[0].steps == [0.42, 0.42]

    for cid, accumulate, n_batches, steps, want_steps in (
            ("no-accumulation-steps-every-batch", False, 3, 4, 3),
            ("three-batches-by-two-flushes-one-leftover", True, 3, 2, 2),
            ("four-batches-by-two-no-leftover", True, 4, 2, 2),
            ("two-batches-by-four-only-the-flush", True, 2, 4, 1)):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            log = {}
            mp.setattr(torch.optim, "SGD", _recording_optimizer(torch.optim.SGD, log))
            _stub_eval(mp)
            _record_progress(mp, [])
            _train(mp, tmp_path / cid, _zero(), loader=_loader(n_batches),
                   optimizer_type="sgd", gradient_accumulation=accumulate,
                   gradient_accumulation_steps=steps)
            assert log["steps"] == want_steps

    # loss = mean(logits): d/d(bias_j) is 0.5 per batch whatever the data, and
    # the first nesterov-SGD step moves a parameter by lr * 1.9 * grad.
    for cid, accumulate, n_batches, grad in (
            ("bias-probe-plain-one-batch", False, 1, 0.5),
            ("bias-probe-two-batches-divided-by-four", True, 2, 2 * 0.5 / 4)):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            model = _zero()
            mp.setattr("spacr.utils.build_loss",
                       lambda **kw: (lambda logits, target: logits.mean()))
            _stub_eval(mp)
            _record_progress(mp, [])
            mp.setattr("spacr.io._save_model", lambda *a, **k: None)
            _train(mp, tmp_path / cid, model, loader=_loader(n_batches),
                   learning_rate=0.1, weight_decay=0.0, optimizer_type="sgd",
                   gradient_accumulation=accumulate, gradient_accumulation_steps=4)
            assert model.bias.tolist() == pytest.approx([-0.1 * 1.9 * grad] * 2,
                                                        rel=1e-6)

    checkpoint = tmp_path / "ckpt.pth"
    checkpoint.write_bytes(b"stub")

    def resume(num_classes, payload, state):
        def setup(mp):
            loaded = nn.Linear(4, 2)
            loaded.num_classes = num_classes
            mp.setattr(D, "load_model_artifact", lambda *a, **k: (loaded, payload))
            mp.setattr(D, "restore_training_state", lambda *a, **k: state)
        return setup

    refusals = [
        ("unknown-optimizer", None, {"optimizer_type": "lbfgs"}, ValueError,
         "Unknown optimizer_type: 'lbfgs'. Choose from: adamw, adam, adamax, "
         "adagrad, adadelta, asgd, sgd, rmsprop, nadam, radam."),
        ("head-narrower-than-classes", None, {"num_classes": 3}, RuntimeError,
         "Expected logits (N,3) for CE, got (4, 2)"),
        ("checkpoint-class-count-mismatch",
         resume(3, {"optimizer_state_dict": {}}, {"epoch": 0}),
         {"custom_model_path": str(checkpoint)}, ValueError,
         "Checkpoint has 3 output classes but this run requests 2. Use matching "
         "classes or train a new head."),
        ("resume-without-optimizer-state", resume(2, {}, {"epoch": 0}),
         {"resume_checkpoint": str(checkpoint)}, ValueError,
         "resume_checkpoint is not a resumable spaCR training artifact "
         "(optimizer state is missing). Use custom_model_path to fine-tune its "
         "weights instead."),
        ("resume-already-complete",
         resume(2, {"optimizer_state_dict": {"x": 1}}, {"epoch": 4}),
         {"resume_checkpoint": str(checkpoint), "epochs": 4}, ValueError,
         "Checkpoint already completed epoch 4, but epochs=4. Increase epochs to "
         "continue training."),
    ]
    for cid, setup, kw, exc, message in refusals:
        with subtests.test(refusal=cid), pytest.MonkeyPatch.context() as mp:
            if setup:
                setup(mp)
            with pytest.raises(exc) as info:
                _train(mp, tmp_path / cid, _zero(), **kw)
            assert str(info.value) == message

    with subtests.test(refusal="unbuildable-backbone-writes-nothing"):
        dst = tmp_path / "badmodel"
        dst.mkdir()
        with pytest.raises(ValueError, match="not_a_real_backbone"):
            D.train_model(str(tmp_path), str(dst), "not_a_real_backbone",
                          _loader(1), epochs=1, num_classes=2)
        assert list(dst.iterdir()) == []

    with subtests.test(case="fine-tune-announces-its-source"), \
            pytest.MonkeyPatch.context() as mp:
        resume(2, {}, {})(mp)
        _stub_eval(mp)
        _record_progress(mp, [])
        printed = []
        mp.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a))))
        (trained, _), _ = _train(mp, tmp_path / "finetune", _zero(),
                                 custom_model_path=str(checkpoint))
        assert f"Fine-tuning model weights from {checkpoint}" in printed
        assert trained.num_classes == 2


# ===========================================================================
# train_test_model
# ===========================================================================

class ConstantClassifier(nn.Module):
    """Always votes class 0; module-level so a whole-module checkpoint pickles."""

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.n_classes, device=x.device)
        out[:, 0] = 4.0
        return out * self.scale


def _png(path: Path, value: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((4, 4, 3), value, dtype=np.uint8)).save(path)


def _tt_settings(src, **over):
    s = {"src": str(src), "model_type": "tinynet", "classes": ["nc", "pc"],
         "train_channels": ["r", "g"], "epochs": 1, "batch_size": 2,
         "image_size": 8, "n_jobs": 0, "pin_memory": False, "verbose": False,
         "augment": False, "val_split": 0.0}
    s.update(over)
    return s


def test_train_test_model_resolves_settings_trains_and_scores_the_test_split(
        subtests, tmp_path, capsys):
    """The run folder is ``src/model/<type>/<channels>[_shuffled_labels]/epochs_<n>``;
    no classes is refused; ``auto``/None loss picks CE or BCE by class count;
    the settings snapshot is named by the halves that run; fixed crop scaling
    loads crops raw and is refused with cross-validation; the test half scores
    the trained model (or the best checkpoint), writes both CSVs, copies the
    misclassified images and returns the result CSV only when nothing trained."""
    import spacr.io as sio
    import spacr.utils as sutils

    today = datetime.date.today().strftime("%y%m%d")

    def stub_train(mp, record, model=None, path="/tmp/best_model.pth"):
        def loaders(src, mode="train", **kw):
            record.setdefault("modes", []).append(mode)
            record.setdefault("kw", []).append(kw)
            return ["train_loader"], ["val_loader"], None

        def train(**kw):
            record["train"] = kw
            return (model or ConstantClassifier()), path

        mp.setattr(sio, "generate_loaders", loaders)
        mp.setattr(D, "train_model", train)

    for cid, over, folder in (
            ("no-classes", {}, "rg"),
            ("no-classes-shuffled-control", {"control_shuffled_labels": True},
             "rg_shuffled_labels")):
        with subtests.test(case=cid):
            src = tmp_path / cid
            src.mkdir()
            with pytest.raises(ValueError, match="^No classes provided: neither "
                                                 "class_folder_names nor classes"):
                D.train_test_model(_tt_settings(src, classes=[], train=False,
                                                test=False, **over))
            assert (src / "model" / "tinynet" / folder / "epochs_1").is_dir()

    for cid, classes, loss_in, loss_out in (
            ("auto-two-classes-is-ce", ["nc", "pc"], "auto", "cross_entropy"),
            ("auto-one-class-is-bce", ["only"], "auto",
             "binary_cross_entropy_with_logits"),
            ("none-two-classes-is-ce", ["nc", "pc"], None, "cross_entropy"),
            ("explicit-loss-is-kept", ["nc", "pc"], "focal_loss", "focal_loss")):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            src = tmp_path / cid
            src.mkdir()
            record = {}
            stub_train(mp, record)
            out = D.train_test_model(_tt_settings(src, classes=classes,
                                                  loss_type=loss_in,
                                                  train=True, test=False))
            assert out == "/tmp/best_model.pth"
            assert record["train"]["loss_type"] == loss_out
            assert record["train"]["num_classes"] == len(classes)
            assert record["modes"] == ["train"]
            assert (src / "settings" / "train_tinynet_1.csv").is_file()

    with subtests.test(case="truthy-train-flag-is-a-train-only-run"), \
            pytest.MonkeyPatch.context() as mp:
        src = tmp_path / "truthy"
        src.mkdir()
        record, saved = {}, []
        stub_train(mp, record)
        mp.setattr(sutils, "save_settings",
                   lambda s, name="settings", show=False: saved.append(name))
        out = D.train_test_model(_tt_settings(src, train=1, test=False,
                                              loss_type="cross_entropy"))
        assert (saved, out, record["modes"]) == (
            ["train_tinynet_1"], "/tmp/best_model.pth", ["train"])

    with subtests.test(case="fixed-crop-scaling-refused-with-cross-validation"):
        src = tmp_path / "fixedcv"
        src.mkdir()
        with pytest.raises(ValueError) as info:
            D.train_test_model(_tt_settings(src, train=True, test=False,
                                            crop_scaling="fixed",
                                            cross_validation_folds=3))
        assert str(info.value) == (
            "crop_scaling='fixed' is not available with cross-validation yet. "
            "Use the single validation split, or crop_scaling='legacy'.")

    with subtests.test(case="fixed-crop-scaling-loads-crops-raw"), \
            pytest.MonkeyPatch.context() as mp:
        src = tmp_path / "fixed"
        src.mkdir()
        record = {}
        stub_train(mp, record)
        D.train_test_model(_tt_settings(src, train=True, test=False,
                                        crop_scaling="fixed",
                                        loss_type="cross_entropy"))
        kw = record["kw"][0]
        assert (kw["crop_loading_policy"], kw["crop_scaling"]) == (
            D.RAW_FIXED_SCALE, "fixed")

    with subtests.test(case="neither-half-returns-none"), \
            pytest.MonkeyPatch.context() as mp:
        src = tmp_path / "neither"
        src.mkdir()
        mp.setattr(sio, "generate_loaders",
                   lambda *a, **k: pytest.fail("no loader should be built"))
        settings = _tt_settings(src, train=False, test=False,
                                loss_type="cross_entropy")
        assert D.train_test_model(settings) is None
        assert settings["dst"] == str(src / "model" / "tinynet" / "rg" / "epochs_1")
        assert os.path.isdir(settings["dst"])
        assert (src / "model" / "tinynet" / "rg" / "epochs_1" / "settings.csv").is_file()
        assert not (src / "settings").exists()

    def split(name):
        src = tmp_path / name / "dataset"
        nc, pc = src / "test" / "nc" / "a.png", src / "test" / "pc" / "b.png"
        _png(nc, 10)
        _png(pc, 200)
        batch = (torch.zeros(2, 3, 8, 8), torch.tensor([0, 1]), [str(nc), str(pc)])
        return src, nc, pc, batch

    def patch_loaders(mp, batch, record):
        def loaders(src, mode="train", **kw):
            record.append((mode, kw))
            return ([batch], None, None) if mode == "test" else ([batch], [batch], None)

        mp.setattr(sio, "generate_loaders", loaders)

    with subtests.test(case="test-only-scores-the-best-checkpoint"), \
            pytest.MonkeyPatch.context() as mp:
        capsys.readouterr()
        src, nc, pc, batch = split("testonly")
        (src / "model").mkdir()
        torch.save(ConstantClassifier(), src / "model" / "ck_epoch_1_acc_0.10.pth")
        torch.save(ConstantClassifier(), src / "model" / "ck_epoch_9_acc_0.97.pth")
        calls = []
        patch_loaders(mp, batch, calls)
        result_loc = D.train_test_model(_tt_settings(src, train=False, test=True,
                                                     loss_type="cross_entropy"))
        (mode, kw), = calls
        assert mode == "test"
        assert (kw["validation_split"], kw["augment"], kw["classes"]) == (
            0.0, False, ["nc", "pc"])
        out = capsys.readouterr().out
        assert f"Best model: {src}/model/ck_epoch_9_acc_0.97.pth" in out.splitlines()
        assert "ck_epoch_1_acc_0.10.pth" not in out
        assert "Copied 1 misclassified images." in out
        dst = src / "model" / "tinynet" / "rg" / "epochs_1"
        assert result_loc == f"{dst}/tinynet_time_{today}_test_result.csv"
        summary = pd.read_csv(result_loc)
        assert (len(summary), summary.loc[0, "accuracy"], summary.loc[0, "epoch"],
                summary.loc[0, "majority_class_accuracy"]) == (1, 0.5, 1, 0.5)
        per_file = pd.read_csv(f"{dst}/tinynet_time_{today}_test_acc.csv")
        assert per_file["filename"].tolist() == [str(nc), str(pc)]
        assert per_file["true_label"].tolist() == [0, 1]
        assert per_file["predicted_label"].tolist() == [0, 0]
        copied = src / "test" / "misclassified" / "pc" / "b.png"
        assert copied.read_bytes() == pc.read_bytes()
        assert not (src / "test" / "misclassified" / "nc").exists()
        assert (src / "settings" / "test_tinynet_1.csv").is_file()

    with subtests.test(case="train-and-test-reuses-the-trained-model"), \
            pytest.MonkeyPatch.context() as mp:
        src, nc, pc, batch = split("both")
        calls = []
        patch_loaders(mp, batch, calls)
        mp.setattr(D, "train_model",
                   lambda **kw: (ConstantClassifier(), str(src / "best.pth")))
        mp.setattr(sutils, "pick_best_model",
                   lambda _src: pytest.fail("pick_best_model must not run"))
        out = D.train_test_model(_tt_settings(src, train=True, test=True,
                                              loss_type="cross_entropy"))
        assert out == str(src / "best.pth")
        assert [c[0] for c in calls] == ["train", "test"]
        assert (src / "settings" / "train_test_tinynet_1.csv").is_file()
        dst = src / "model" / "tinynet" / "rg" / "epochs_1"
        assert sorted(p.name for p in dst.glob("*.csv")) == [
            "settings.csv", f"tinynet_time_{today}_test_acc.csv",
            f"tinynet_time_{today}_test_result.csv"]


# ===========================================================================
# _plot_training_curves
# ===========================================================================

def test_plot_training_curves_draws_loss_accuracy_and_per_class_panels(subtests):
    """No history draws nothing; otherwise three panels (loss, accuracy,
    per-class) with missing epochs/metrics defaulted, validation lines beside
    the training ones, per-class lines from the validation history naming the
    worst class, a ``/ total`` title suffix and an in-place refresh."""
    with subtests.test(case="empty-history-draws-nothing"):
        plt.close("all")
        assert D._plot_training_curves([], [{"epoch": 1, "loss": 0.5}]) is None
        assert plt.get_fignums() == []

    with subtests.test(case="train-only-defaults"):
        fig = D._plot_training_curves(
            [{"loss": 1.0, "accuracy": 0.2}, {"loss": 0.5, "accuracy": 0.6},
             {"loss": 0.25}], None)
        ax1, ax2, ax3 = fig.axes
        assert [a.get_title() for a in fig.axes] == ["Loss", "Accuracy",
                                                     "Per-class accuracy"]
        assert (len(ax1.lines), len(ax2.lines), len(ax3.lines)) == (1, 1, 0)
        assert list(ax1.lines[0].get_xdata()) == [1, 2, 3]
        assert list(ax1.lines[0].get_ydata()) == [1.0, 0.5, 0.25]
        acc = np.asarray(ax2.lines[0].get_ydata(), dtype=float)
        assert acc[:2].tolist() == [0.2, 0.6] and math.isnan(acc[2])
        assert ax2.get_ylim() == (0, 1.02)
        assert (ax1.get_xlabel(), ax2.get_xlabel(), ax3.get_xlabel()) == (
            "epoch", "epoch", "epoch")
        assert [t.get_text() for t in ax3.texts] == ["no per-class metrics"]
        assert fig._suptitle.get_text() == "Training — epoch 3"
        assert fig._spacr_live_update is True

    train = [{"epoch": 1, "loss": 0.9, "accuracy": 0.4},
             {"epoch": 2, "loss": 0.4, "accuracy": 0.8}]
    val = [{"epoch": 1, "loss": 1.1, "accuracy": 0.3,
            "per_class_accuracy": [0.2, 0.4], "class_support": [3, 3],
            "class_names": ["nc", "pc"]},
           {"epoch": 2, "accuracy": 0.7,
            "per_class_accuracy": [0.9, 0.5], "class_support": [3, 3],
            "class_names": ["nc", "pc"]}]
    with subtests.test(case="validation-and-per-class-from-validation"):
        fig = D._plot_training_curves(train, val, total_epochs=10)
        ax1, ax2, ax3 = fig.axes
        assert [line.get_label() for line in ax1.lines] == ["train", "val"]
        assert list(ax1.lines[1].get_xdata()) == [1, 2]
        val_loss = np.asarray(ax1.lines[1].get_ydata(), dtype=float)
        assert val_loss[0] == 1.1 and math.isnan(val_loss[1])
        assert list(ax2.lines[1].get_ydata()) == [0.3, 0.7]
        assert ax3.get_title() == "Per-class accuracy (val)"
        assert [line.get_label() for line in ax3.lines] == ["nc", "pc"]
        assert [list(line.get_ydata()) for line in ax3.lines] == [[0.2, 0.9],
                                                                  [0.4, 0.5]]
        assert ax3.get_xlabel() == "epoch — worst: pc at 0.500"
        assert fig._suptitle.get_text() == "Training — epoch 2 / 10"

        train.append({"epoch": 3, "loss": 0.2, "accuracy": 0.9})
        again = D._plot_training_curves(train, val, total_epochs=10, figure=fig)
        assert again is fig
        assert list(again.axes[0].lines[0].get_xdata()) == [1, 2, 3]
        assert again._suptitle.get_text() == "Training — epoch 3 / 10"

    with subtests.test(case="per-class-from-training-with-one-finite-class"):
        fig = D._plot_training_curves(
            [{"epoch": 1, "loss": 0.5, "accuracy": 0.5,
              "per_class_accuracy": [0.5, float("nan")], "class_support": [2, 0]}],
            [], classes=["neg", "pos"])
        ax3 = fig.axes[2]
        assert ax3.get_title() == "Per-class accuracy (train)"
        assert [line.get_label() for line in ax3.lines] == ["neg", "pos"]
        assert ax3.get_xlabel() == "epoch"


# ===========================================================================
# _log_tensorboard_epoch
# ===========================================================================

class _Writer:
    """The two SummaryWriter methods the epoch logger calls."""

    def __init__(self):
        self.scalars, self.flushes = [], 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flushes += 1


def test_log_tensorboard_epoch_writes_one_scalar_per_metric_and_class(subtests):
    """Loss, accuracy and macro F1 per split, one ``accuracy_<class>/<split>``
    per class with a finite accuracy, the learning rate from the training
    dict, then one flush; an absent split or metric writes nothing and no
    writer is a no-op."""
    nan = float("nan")
    cases = [
        ("train-and-validation",
         {"loss": 0.3, "accuracy": 0.8, "f1_macro": 0.7, "lr": 1e-4},
         {"loss": 0.4, "accuracy": 0.75, "f1_macro": 0.65}, None,
         [("loss/train", 0.3, 3), ("accuracy/train", 0.8, 3),
          ("f1_macro/train", 0.7, 3), ("loss/validation", 0.4, 3),
          ("accuracy/validation", 0.75, 3), ("f1_macro/validation", 0.65, 3),
          ("learning_rate", 1e-4, 3)]),
        ("class-without-support-is-skipped",
         {"loss": 0.4, "accuracy": 0.9, "f1_macro": 0.88, "lr": 1e-4,
          "per_class_accuracy": [0.9, nan], "class_support": [40, 0]},
         {}, ["neg", "pos"],
         [("loss/train", 0.4, 3), ("accuracy/train", 0.9, 3),
          ("f1_macro/train", 0.88, 3), ("accuracy_neg/train", 0.9, 3),
          ("learning_rate", 1e-4, 3)]),
        ("first-class-absent-second-present",
         {"loss": 0.3, "accuracy": 0.8, "per_class_accuracy": [nan, 0.9],
          "class_support": [0, 30]},
         None, ["absent", "present"],
         [("loss/train", 0.3, 3), ("accuracy/train", 0.8, 3),
          ("accuracy_present/train", 0.9, 3)]),
        ("validation-classes-named-by-metrics",
         {"loss": 0.25},
         {"accuracy": 0.5, "per_class_accuracy": [1.0, 0.0],
          "class_support": [1, 1], "class_names": ["a", "b"]}, None,
         [("loss/train", 0.25, 3), ("accuracy/validation", 0.5, 3),
          ("accuracy_a/validation", 1.0, 3), ("accuracy_b/validation", 0.0, 3)]),
    ]
    for cid, train, val, classes, want in cases:
        with subtests.test(case=cid):
            writer = _Writer()
            assert D._log_tensorboard_epoch(writer, train, val, 3, classes) is None
            assert writer.scalars == want
            assert all(type(v) is float for _t, v, _s in writer.scalars)
            assert writer.flushes == 1

    with subtests.test(case="no-writer-is-a-no-op"):
        assert D._log_tensorboard_epoch(None, {"loss": 1.0}, None, 1) is None


# ===========================================================================
# _flowview_event
# ===========================================================================

@contextlib.contextmanager
def _flowview_unloaded():
    """Run with ``spacr.flowview`` absent from ``sys.modules``, then restore
    both the module table and the package attribute a fresh import rebinds."""
    import spacr as spacr_package

    before = {n: m for n, m in sys.modules.items() if n.startswith("spacr.flowview")}
    attribute = getattr(spacr_package, "flowview", None)
    with pytest.MonkeyPatch.context() as mp:
        for name in before:
            mp.delitem(sys.modules, name, raising=False)
        try:
            yield mp
        finally:
            for name in [n for n in sys.modules if n.startswith("spacr.flowview")]:
                sys.modules.pop(name, None)
    sys.modules.update(before)
    if attribute is not None:
        spacr_package.flowview = attribute


def test_flowview_event_reaches_the_tracer_only_when_enabled(subtests):
    """Unset or non-true ``SPACR_FLOWVIEW`` imports nothing; a true value
    (any case, padded) imports and enables the tracer; an import that fails, a
    disabled tracer or a stage that raises answers False; an enabled tracer
    answers what the stage said."""
    for value in (None, "", "0", "off", "no", "enabled"):
        with subtests.test(env=value), _flowview_unloaded() as mp:
            if value is None:
                mp.delenv("SPACR_FLOWVIEW", raising=False)
            else:
                mp.setenv("SPACR_FLOWVIEW", value)
            assert D._flowview_event("advance", "scores") is False
            assert "spacr.flowview.trace" not in sys.modules

    for value in ("TRUE", " on ", "Yes", "1"):
        with subtests.test(env=value), _flowview_unloaded() as mp:
            mp.setenv("SPACR_FLOWVIEW", value)
            assert D._flowview_event("advance", "scores") is True
            assert sys.modules["spacr.flowview.trace"].is_enabled() is True
            # the pipeline graph is monotonic: the same boundary twice is refused
            assert D._flowview_event("advance", "scores") is False

    with subtests.test(case="opted-in-but-unimportable"), _flowview_unloaded() as mp:
        mp.setenv("SPACR_FLOWVIEW", "1")
        mp.setitem(sys.modules, "spacr.flowview", None)
        assert D._flowview_event("advance", "scores") is False

    calls = []

    class Tracer:
        def __init__(self, enabled):
            self._enabled = enabled

        def is_enabled(self):
            return self._enabled

    class Stages:
        @staticmethod
        def _advance(node_id):
            calls.append(("advance", node_id))
            return 1

        @staticmethod
        def _metric(name, value):
            calls.append(("metric", name, value))
            return 0

        @staticmethod
        def _fail(error):
            raise RuntimeError("tracer bug")

    for cid, enabled, action, args, want, want_calls in (
            ("loaded-but-disabled", False, "advance", ("scores",), False, []),
            ("enabled-truthy-stage", True, "advance", ("scores",), True,
             [("advance", "scores")]),
            ("enabled-falsy-stage", True, "metric", ("objects", 4), False,
             [("metric", "objects", 4)]),
            ("stage-that-raises", True, "fail", (ValueError("x"),), False, []),
            ("unknown-stage", True, "nope", (), False, [])):
        with subtests.test(case=cid), pytest.MonkeyPatch.context() as mp:
            calls.clear()
            mp.setitem(sys.modules, "spacr.flowview.trace", Tracer(enabled))
            mp.setattr(sys.modules["spacr.flowview"], "_classify_stages", Stages,
                       raising=False)
            assert D._flowview_event(action, *args) is want
            assert calls == want_calls
