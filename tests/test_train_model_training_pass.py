"""Training metrics come from the training batches, not a second pass.

``train_model`` used to re-evaluate the whole training loader after every
epoch, in eval mode and full precision, and the classes for the loss were
counted from folders that a custom loader does not have. These tests run the
real loop on a 4-feature linear model over tiny CPU tensors and check what
the loop logs, what it reads, and what stays the same.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """Keep the stand-in model on the CPU and close any figure afterwards."""
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


class _Labelled(Dataset):
    """Eight 4-feature samples with a ``labels`` list, like spacrDataset."""

    def __init__(self, labels, seed=0):
        """Build fixed features for the given integer labels."""
        generator = torch.Generator().manual_seed(seed)
        self.labels = list(labels)
        self.data = torch.rand(len(self.labels), 4, generator=generator)

    def __len__(self):
        """Number of samples."""
        return len(self.labels)

    def __getitem__(self, index):
        """Return ``(features, label, filename)``."""
        return self.data[index], self.labels[index], f"p1_A01_1_{index}.png"


def _loader(labels, batch_size=4, seed=0):
    """A deterministic, unshuffled loader over ``_Labelled``."""
    return DataLoader(_Labelled(labels, seed=seed), batch_size=batch_size,
                      shuffle=False)


def _model(seed=0, n_classes=2):
    """A fresh linear model with fixed initial weights."""
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, n_classes))


def _train(tmp_path, monkeypatch, model, train_loader, *, val_loader=None,
           full_pass=False, learning_rate=0.05, epochs=3, n_classes=2,
           gradient_accumulation=False, name="run", calls=None,
           loss_type="ce"):
    """Run the real ``train_model`` and return the progress it logged."""
    import spacr.deep_spacr as ds

    monkeypatch.setattr("spacr.utils.choose_model", lambda *a, **k: model)
    if calls is not None:
        original = ds.evaluate_model_performance

        def counted(model_, loader, epoch, **kw):
            calls.append(loader)
            return original(model_, loader, epoch, **kw)

        monkeypatch.setattr(ds, "evaluate_model_performance", counted)
    saved = []
    monkeypatch.setattr(
        "spacr.io._save_progress",
        lambda dst, train_df, val_df: saved.append((train_df, val_df)))
    dst = tmp_path / name
    dst.mkdir()
    ds.train_model(
        str(dst), str(dst), "stand_in", train_loader, epochs=epochs,
        learning_rate=learning_rate, weight_decay=0.0, optimizer_type="sgd",
        val_loaders=val_loader, intermedeate_save=None, loss_type=loss_type,
        label_smoothing=0.0, logit_adjust_tau=0.0, num_classes=n_classes,
        classes=[f"c{i}" for i in range(n_classes)], plot=False,
        tensorboard=False, write_card=False,
        gradient_accumulation=gradient_accumulation,
        gradient_accumulation_steps=2,
        settings={"train_metrics_full_pass": full_pass})
    train = [row for train_df, _ in saved for row in train_df.to_dict("records")]
    val = [row for _, val_df in saved if val_df is not None
           for row in val_df.to_dict("records")]
    return train, val


def test_no_second_pass_over_the_training_loader_by_default(tmp_path, monkeypatch):
    """Only the validation loader is evaluated; the training loader is not."""
    train_loader = _loader([0, 1] * 4)
    val_loader = _loader([1, 0] * 2, seed=1)
    calls = []
    _train(tmp_path, monkeypatch, _model(), train_loader,
           val_loader=val_loader, calls=calls)
    assert len(calls) == 3
    assert all(loader is val_loader for loader in calls)


def test_full_pass_setting_restores_the_extra_evaluation(tmp_path, monkeypatch):
    """With the setting on, every epoch evaluates both loaders again."""
    train_loader = _loader([0, 1] * 4)
    val_loader = _loader([1, 0] * 2, seed=1)
    calls = []
    _train(tmp_path, monkeypatch, _model(), train_loader,
           val_loader=val_loader, full_pass=True, calls=calls)
    assert [loader is train_loader for loader in calls] == [True, False] * 3


@pytest.mark.parametrize("accumulate", [False, True])
def test_in_pass_metrics_equal_the_full_pass_when_weights_stand_still(
        tmp_path, monkeypatch, accumulate):
    """With no weight change and no dropout both routes report the same numbers.

    A zero learning rate keeps the weights fixed through the epoch, so the
    batches the loop records and a separate pass afterwards see the same
    model. The recorded loss is the undivided batch loss, also when gradients
    are accumulated.
    """
    labels = [0, 1, 1, 0, 1, 1, 0, 1]
    in_pass, _ = _train(tmp_path, monkeypatch, _model(), _loader(labels),
                        learning_rate=0.0, name="in_pass",
                        gradient_accumulation=accumulate)
    full, _ = _train(tmp_path, monkeypatch, _model(), _loader(labels),
                     learning_rate=0.0, full_pass=True, name="full",
                     gradient_accumulation=accumulate)
    assert len(in_pass) == len(full) == 3
    for mine, theirs in zip(in_pass, full):
        for key in ("loss", "accuracy", "f1_macro", "prauc",
                    "acc_class_c0", "acc_class_c1", "n_c0", "n_c1"):
            assert mine[key] == pytest.approx(theirs[key], rel=1e-6, abs=1e-7), key
    assert in_pass[0]["n_c0"] == 3 and in_pass[0]["n_c1"] == 5


def test_validation_metrics_do_not_depend_on_the_setting(tmp_path, monkeypatch):
    """Weights, and so validation metrics, follow the same path either way."""
    labels = [0, 1, 1, 0, 1, 1, 0, 1]
    model_a, model_b = _model(), _model()
    _, val_a = _train(tmp_path, monkeypatch, model_a, _loader(labels),
                      val_loader=_loader([1, 0, 0, 1], seed=2), name="a")
    _, val_b = _train(tmp_path, monkeypatch, model_b, _loader(labels),
                      val_loader=_loader([1, 0, 0, 1], seed=2),
                      full_pass=True, name="b")
    assert len(val_a) == len(val_b) == 3
    for row_a, row_b in zip(val_a, val_b):
        assert row_a["loss"] == pytest.approx(row_b["loss"], rel=1e-9)
        assert row_a["accuracy"] == row_b["accuracy"]
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p_a, p_b)


def test_in_pass_metrics_track_the_training_batches(tmp_path, monkeypatch):
    """The logged training loss is the sample mean of the batch losses.

    Plain cross-entropy in spaCR is weighted by inverse class frequency, here
    3 of class 0 and 5 of class 1, so the expected value uses those weights.
    """
    labels = [0, 1, 1, 0, 1, 1, 0, 1]
    model = _model()
    seen = []
    original_forward = model.forward

    def recording_forward(x):
        out = original_forward(x)
        seen.append((out.detach().clone(), x))
        return out

    model.forward = recording_forward
    train, _ = _train(tmp_path, monkeypatch, model, _loader(labels),
                      epochs=1)
    targets = torch.tensor(labels)
    logits = torch.cat([out for out, _ in seen])
    priors = torch.tensor([3.0, 5.0]) / 8.0
    weights = (1.0 / priors) / (1.0 / priors).mean()
    per_batch = [nn.functional.cross_entropy(
        logits[i:i + 4], targets[i:i + 4], weight=weights).item()
        for i in (0, 4)]
    assert train[0]["loss"] == pytest.approx(sum(per_batch) / 2, rel=1e-6)
    accuracy = (logits.argmax(1) == targets).float().mean().item()
    assert train[0]["accuracy"] == pytest.approx(accuracy)


def test_binary_head_is_recorded_from_the_batches(tmp_path, monkeypatch):
    """A single-logit head goes through the sigmoid route and is scored."""
    labels = [0.0, 1.0] * 4
    in_pass, _ = _train(tmp_path, monkeypatch, _model(n_classes=1),
                        DataLoader(_Labelled(labels), batch_size=4),
                        learning_rate=0.0, n_classes=1, epochs=1,
                        name="binary_in", loss_type="bce")
    full, _ = _train(tmp_path, monkeypatch, _model(n_classes=1),
                     DataLoader(_Labelled(labels), batch_size=4),
                     learning_rate=0.0, n_classes=1, epochs=1, full_pass=True,
                     name="binary_full", loss_type="bce")
    assert in_pass[0]["loss"] == pytest.approx(full[0]["loss"], rel=1e-6)
    assert in_pass[0]["accuracy"] == full[0]["accuracy"]


def test_class_counts_come_from_the_dataset_labels(tmp_path, monkeypatch):
    """Without a train/ folder the loss is built from the dataset's labels.

    Neither the folder counter nor a walk over the loader runs, and the
    counts are those of the Subset actually trained on.
    """
    import spacr.utils as utils

    def refuse(*args, **kwargs):
        raise AssertionError("classes were counted by walking the loader")

    monkeypatch.setattr(utils, "estimate_class_counts", refuse)
    received = {}
    original = utils.build_loss

    def spy(**kwargs):
        received["counts"] = kwargs.get("class_counts")
        return original(**kwargs)

    monkeypatch.setattr(utils, "build_loss", spy)
    base = _Labelled([0, 0, 0, 1, 1, 2, 2, 2, 2, 2])
    subset = Subset(Subset(base, [0, 1, 3, 5, 6, 7, 8, 9]), [0, 1, 2, 3, 4])
    _train(tmp_path, monkeypatch, _model(n_classes=3),
           DataLoader(subset, batch_size=5), epochs=1, n_classes=3)
    assert received["counts"].tolist() == [2, 1, 2]


def test_folder_counts_still_win_when_the_train_folder_exists(tmp_path, monkeypatch):
    """A dataset root with train/<class>/ keeps the folder count as before."""
    import spacr.utils as utils

    root = tmp_path / "root"
    for name, n in (("c0", 3), ("c1", 1)):
        folder = root / "train" / name
        folder.mkdir(parents=True)
        for i in range(n):
            (folder / f"p1_A01_1_{i}.png").write_bytes(b"")
    received = {}
    original = utils.build_loss

    def spy(**kwargs):
        received["counts"] = kwargs.get("class_counts")
        return original(**kwargs)

    monkeypatch.setattr(utils, "build_loss", spy)
    monkeypatch.setattr("spacr.utils.choose_model", lambda *a, **k: _model())
    monkeypatch.setattr("spacr.io._save_progress", lambda *a: None)
    import spacr.deep_spacr as ds
    ds.train_model(str(root), str(tmp_path), "stand_in", _loader([0, 1] * 4),
                   epochs=1, optimizer_type="sgd", intermedeate_save=None,
                   loss_type="ce", num_classes=2, plot=False,
                   tensorboard=False, write_card=False)
    assert received["counts"].tolist() == [3, 1]
