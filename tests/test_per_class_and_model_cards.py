"""Per-class accuracy per epoch (C10) and model cards (C9).

The two failure modes these features exist to end:

* **A 96 % aggregate hiding a class at 40 %.** Every number here comes from a
  run that is *deliberately* imbalanced and a model that *deliberately*
  refuses to predict the minority class, so the aggregate looks healthy and
  the weak class does not. A test that only asserted "a per-class key exists"
  would pass on an implementation that reported the aggregate C times.
* **A card that is a claim rather than a record.** The card's held-out
  metrics are recomputed here from the confusion matrix the card itself
  carries, and independently from the raw labels and probabilities, and all
  three must agree. A card whose accuracy came from somewhere other than its
  own matrix fails.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (after importorskip)

import spacr.deep_spacr as D  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


class _AlwaysMajority(nn.Module):
    """A model that always answers class 0, with a real, trainable parameter.

    This is what a classifier trained on a 9:1 screen with no reweighting
    collapses to, and it is the exact case a mean accuracy flatters: it
    scores 90 % while being useless. Built explicitly rather than hoped for,
    because a genuinely-trained tiny model would land somewhere arbitrary.
    """

    def __init__(self, n_features=4, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.dummy = nn.Linear(n_features, n_classes)
        with torch.no_grad():
            self.dummy.weight.zero_()
            bias = torch.full((n_classes,), -5.0)
            bias[0] = 5.0
            self.dummy.bias.copy_(bias)

    def forward(self, x):
        return self.dummy(x)


def _imbalanced_loader(n_batches=2, batch=10, n_feat=4, minority=1):
    """Batches that are 9:1 class 0 : class 1."""
    out = []
    g = torch.Generator().manual_seed(7)
    for b in range(n_batches):
        x = torch.rand(batch, n_feat, generator=g)
        y = torch.zeros(batch, dtype=torch.long)
        y[:minority] = 1
        out.append((x, y, [f"b{b}_{i}.png" for i in range(batch)]))
    return out


# ---------------------------------------------------------------------------
# C10 — per-class accuracy is computed, kept, and surfaced per epoch
# ---------------------------------------------------------------------------

def test_binary_metrics_report_per_class_under_the_multiclass_key():
    """Binary reports its two class accuracies the way multiclass does.

    Before this, binary said ``neg_accuracy``/``pos_accuracy`` and multiclass
    said ``per_class_accuracy``, so every consumer branched on head shape and
    most of them only handled one.
    """
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])   # 9:1
    probs = np.zeros(10)                            # always answers class 0
    metrics = D._binary_metrics(y, probs)

    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["per_class_accuracy"] == [1.0, 0.0]
    assert metrics["class_support"] == [9, 1]
    assert metrics["num_classes"] == 2
    # the old keys still say the same thing
    assert metrics["neg_accuracy"] == pytest.approx(1.0)
    assert metrics["pos_accuracy"] == pytest.approx(0.0)


def test_multiclass_metrics_carry_support_beside_each_accuracy():
    """0.40 over 500 objects and 0.40 over 5 are different facts."""
    y = np.array([0] * 20 + [1] * 5 + [2] * 3)
    probs = np.zeros((28, 3))
    probs[:, 0] = 1.0                    # everything predicted class 0
    metrics = D._multiclass_metrics(y, probs)

    assert metrics["per_class_accuracy"] == [1.0, 0.0, 0.0]
    assert metrics["class_support"] == [20, 5, 3]
    assert metrics["accuracy"] == pytest.approx(20 / 28)


def test_per_class_accuracy_names_and_flat_columns():
    metrics = {"per_class_accuracy": [0.99, 0.40], "class_support": [900, 100],
               "num_classes": 2}
    rows = D.per_class_accuracy(metrics, ["nc", "pc"])
    assert rows == [("nc", 0.99, 900), ("pc", 0.40, 100)]

    flat = D.attach_per_class_columns(dict(metrics), ["nc", "pc"])
    assert flat["acc_class_nc"] == pytest.approx(0.99)
    assert flat["acc_class_pc"] == pytest.approx(0.40)
    assert flat["n_nc"] == 900 and flat["n_pc"] == 100
    # self-describing: the names travel with the epoch dict
    assert flat["class_names"] == ["nc", "pc"]
    # and are then found without being handed the list again
    assert D.per_class_accuracy(flat)[1][0] == "pc"


def test_format_per_class_accuracy_flags_the_weak_class():
    line = D.format_per_class_accuracy(
        {"per_class_accuracy": [0.99, 0.40], "class_support": [900, 100],
         "num_classes": 2}, ["nc", "pc"], "Val ")
    assert "nc 0.990 (n=900)" in line
    assert "pc 0.400 (n=100)" in line
    assert "WORST" in line and "pc" in line
    # a balanced pair says nothing alarming
    assert "WORST" not in D.format_per_class_accuracy(
        {"per_class_accuracy": [0.90, 0.88], "class_support": [10, 10],
         "num_classes": 2})
    # and no breakdown at all is an empty line, not a crash
    assert D.format_per_class_accuracy({"accuracy": 0.9}) == ""


def test_live_curves_get_a_per_class_panel_with_one_line_per_class():
    history = [
        D.attach_per_class_columns(
            {"epoch": e, "loss": 0.5, "accuracy": 0.9,
             "per_class_accuracy": [0.99, 0.10 * e],
             "class_support": [900, 100], "num_classes": 2}, ["nc", "pc"])
        for e in (1, 2, 3)]
    figure = D._plot_training_curves(history, history, total_epochs=3)
    assert figure is not None
    axes = figure.get_axes()
    assert len(axes) == 3, "loss, accuracy AND per-class"
    per_class_axis = axes[2]
    assert "Per-class accuracy" in per_class_axis.get_title()
    labels = [line.get_label() for line in per_class_axis.get_lines()]
    assert labels == ["nc", "pc"]
    # the weak class's own line, epoch by epoch — not an average
    weak = per_class_axis.get_lines()[1].get_ydata()
    assert list(weak) == pytest.approx([0.1, 0.2, 0.3])
    # and the panel names the worst class where it cannot be missed
    assert "pc" in per_class_axis.get_xlabel()


def test_live_curves_without_per_class_metrics_still_render():
    figure = D._plot_training_curves(
        [{"epoch": 1, "loss": 0.5, "accuracy": 0.5}], [])
    assert len(figure.get_axes()) == 3


def test_tensorboard_gets_one_scalar_per_class():
    written = []

    class _Writer:
        def add_scalar(self, tag, value, step):
            written.append((tag, float(value), step))

        def flush(self):
            pass

    metrics = D.attach_per_class_columns(
        {"epoch": 4, "loss": 0.2, "accuracy": 0.9, "f1_macro": 0.5,
         "per_class_accuracy": [0.99, 0.40], "class_support": [900, 100],
         "num_classes": 2}, ["nc", "pc"])
    D._log_tensorboard_epoch(_Writer(), metrics, metrics, 4)
    tags = {tag for tag, _, _ in written}
    assert "accuracy_nc/train" in tags
    assert "accuracy_pc/validation" in tags
    assert ("accuracy_pc/validation", 0.40, 4) in written


def test_training_run_surfaces_the_weak_class_every_epoch(tmp_path, capsys,
                                                          monkeypatch):
    """A whole (tiny) run: 9:1 classes, a model that never says class 1.

    The aggregate is 0.9 every epoch and the minority class is 0.0 every
    epoch, and BOTH must be visible per epoch — in the printed line and in
    the per-epoch CSV — not only in a summary at the end of the run.
    """
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: _AlwaysMajority())
    src = tmp_path / "dataset"
    for split in ("train",):
        for name in ("nc", "pc"):
            (src / split / name).mkdir(parents=True)
    # 9 negative crops, 1 positive — the balance the card must also record
    for i in range(9):
        (src / "train" / "nc" / f"n{i}.png").write_bytes(b"x")
    (src / "train" / "pc" / "p0.png").write_bytes(b"x")
    dst = tmp_path / "model"
    dst.mkdir()

    loaders = _imbalanced_loader()
    model, path = D.train_model(
        str(src), str(dst), "resnet18", loaders, epochs=2,
        val_loaders=loaders, num_classes=2, schedule=None, plot=False,
        tensorboard=False, learning_rate=0.0)

    out = capsys.readouterr().out
    # once per epoch, not once per run
    assert out.count("Val per-class acc.:") == 2
    assert "pc 0.000" in out and "nc 1.000" in out
    assert "WORST: pc" in out
    # the aggregate that hides it is right there beside it
    assert "Val acc.: 0.900" in out

    validation = pd.read_csv(dst / "validation.csv", index_col=0)
    assert len(validation) == 2, "one row per epoch"
    assert list(validation["accuracy"]) == pytest.approx([0.9, 0.9])
    assert list(validation["acc_class_pc"]) == pytest.approx([0.0, 0.0])
    assert list(validation["acc_class_nc"]) == pytest.approx([1.0, 1.0])
    assert list(validation["n_pc"]) == [2, 2]
    assert path and os.path.isfile(path)


# ---------------------------------------------------------------------------
# C9 — model cards
# ---------------------------------------------------------------------------

def _recompute_from_matrix(matrix):
    """Accuracy and per-class accuracy, straight off the confusion matrix."""
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1)
    per_class = np.where(row_sums > 0,
                         np.diag(matrix) / np.maximum(row_sums, 1), 0.0)
    total = matrix.sum()
    return (float(np.trace(matrix) / total) if total else float("nan"),
            [float(v) for v in per_class])


def test_held_out_report_is_recomputable_from_its_own_matrix():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=200)
    probs = rng.random((200, 3))
    probs = probs / probs.sum(axis=1, keepdims=True)

    report = D.held_out_report(y, probs, ["a", "b", "c"])
    accuracy, per_class = _recompute_from_matrix(report["confusion_matrix"])
    assert report["accuracy"] == pytest.approx(accuracy)
    assert report["per_class_accuracy"] == pytest.approx(per_class)
    assert sum(report["class_support"]) == 200
    # and the matrix agrees with an independent count
    expected = np.zeros((3, 3), dtype=int)
    for true, pred in zip(y, probs.argmax(axis=1)):
        expected[true, pred] += 1
    assert report["confusion_matrix"] == expected.tolist()


def test_held_out_report_handles_a_single_logit_head():
    report = D.held_out_report([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.1])
    assert report["num_classes"] == 2
    assert report["confusion_matrix"] == [[2, 0], [1, 1]]
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["per_class_accuracy"] == pytest.approx([1.0, 0.5])


def test_dataset_class_balance_counts_what_is_on_disk(tmp_path):
    for split, counts in (("train", {"nc": 9, "pc": 1}),
                          ("test", {"nc": 4, "pc": 2})):
        for name, n in counts.items():
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for i in range(n):
                (folder / f"{i}.png").write_bytes(b"x")
    balance = D.dataset_class_balance(str(tmp_path))
    assert balance == {"train": {"nc": 9, "pc": 1},
                       "test": {"nc": 4, "pc": 2}}


def test_build_model_card_records_provenance_and_warns_when_it_cannot(tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"weights")
    card = D.build_model_card(
        str(weights), settings={"src": str(tmp_path), "epochs": 3},
        classes=["nc", "pc"], split_rule="grouped by well",
        held_out=D.held_out_report([0, 0, 1], [0.1, 0.2, 0.9], ["nc", "pc"]),
        class_balance={"train": {"nc": 90, "pc": 10}}, epochs=3)

    assert card["spacr_version"]
    assert card["settings_hash"]
    assert card["created_utc"].endswith("+00:00")
    assert card["classes"] == ["nc", "pc"]
    assert card["split_rule"] == "grouped by well"
    assert card["training_set"]["class_balance"]["train"]["pc"] == 10
    assert "90%" in card["training_set"]["imbalance_note"]
    assert "warnings" not in card

    naked = D.build_model_card(str(weights))
    assert any("split rule" in w for w in naked["warnings"])
    assert any("held-out" in w for w in naked["warnings"])


def test_write_and_read_model_card_round_trip(tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"weights")
    card = D.build_model_card(str(weights), classes=["a", "b"],
                              split_rule="grouped by well",
                              held_out=D.held_out_report([0, 1], [0.1, 0.9]))
    path = D.write_model_card(str(weights), card)
    assert path == str(tmp_path / "m.card.json")
    assert os.path.isfile(tmp_path / "m.card.md")
    assert D.read_model_card(str(weights))["classes"] == ["a", "b"]
    assert D.read_model_card(str(tmp_path / "absent.pth")) is None

    markdown = (tmp_path / "m.card.md").read_text()
    assert "# Model card" in markdown
    assert "Confusion matrix" in markdown
    assert "grouped by well" in markdown


def test_model_card_registers_as_an_artifact(tmp_path):
    weights = tmp_path / "m.pth"
    weights.write_bytes(b"weights")
    card, card_path, artifact = D.model_card(
        str(weights), project=str(tmp_path), settings={"epochs": 3},
        classes=["a", "b"], split_rule="grouped by well",
        held_out=D.held_out_report([0, 1], [0.1, 0.9]), module="train")

    assert artifact is not None, "the registry is the point of registering"
    assert artifact.path == str(weights)
    assert artifact.spacr_version
    assert artifact.fingerprint, "content-addressed, not just a filename"
    assert artifact.extra["split_rule"] == "grouped by well"
    # the card on disk names the artifact it became
    assert json.loads(open(card_path).read())["artifact_id"] == \
        artifact.artifact_id

    from spacr import artifacts as A
    found = A.open_registry(str(tmp_path)).get(artifact.artifact_id)
    assert found is not None and found.role == D.MODEL_CARD_ROLE


def test_training_writes_a_card_whose_metrics_match_a_recomputation(
        tmp_path, monkeypatch):
    """The card beside the weights must be checkable, not merely present."""
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: _AlwaysMajority())
    src = tmp_path / "dataset"
    for name, n in (("nc", 9), ("pc", 1)):
        folder = src / "train" / name
        folder.mkdir(parents=True)
        for i in range(n):
            (folder / f"{i}.png").write_bytes(b"x")
    dst = tmp_path / "model"
    dst.mkdir()

    loaders = _imbalanced_loader()
    _model, path = D.train_model(
        str(src), str(dst), "resnet18", loaders, epochs=1,
        val_loaders=loaders, num_classes=2, schedule=None, plot=False,
        tensorboard=False, learning_rate=0.0,
        settings={"val_split": 0.2, "cv_group_by": "well", "epochs": 1},
        split_rule="20% of train/ held out, grouped by well")

    card = D.read_model_card(path)
    assert card is not None, "every trained checkpoint gets a card"
    held = card["held_out"]

    # 1. the card's own numbers are exactly its own matrix
    accuracy, per_class = _recompute_from_matrix(held["confusion_matrix"])
    assert held["accuracy"] == pytest.approx(accuracy)
    assert held["per_class_accuracy"] == pytest.approx(per_class)

    # 2. and that matrix is what the model actually does: 9:1, always class 0
    assert held["confusion_matrix"] == [[18, 0], [2, 0]]
    assert held["accuracy"] == pytest.approx(0.9)
    assert held["per_class_accuracy"] == pytest.approx([1.0, 0.0])
    assert held["classes"] == ["nc", "pc"]

    # 3. the rest of the record
    assert card["split_rule"] == "20% of train/ held out, grouped by well"
    assert card["training_set"]["class_balance"]["train"] == {"nc": 9, "pc": 1}
    assert card["spacr_version"] and card["settings_hash"]
    assert card["extra"]["model_type"] == "resnet18"
    assert card["history"], "the per-epoch curve travels with the card"
    assert card["history"][0]["per_class_accuracy"] == pytest.approx([1.0, 0.0])
    assert os.path.isfile(os.path.splitext(path)[0] + ".card.md")


def test_a_card_that_cannot_be_written_does_not_lose_the_weights(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: _AlwaysMajority())
    monkeypatch.setattr(D, "model_card",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("read-only filesystem")))
    dst = tmp_path / "model"
    dst.mkdir()
    loaders = _imbalanced_loader()
    _model, path = D.train_model(
        str(tmp_path), str(dst), "resnet18", loaders, epochs=1,
        val_loaders=loaders, num_classes=2, schedule=None, plot=False,
        tensorboard=False, learning_rate=0.0)
    assert path and os.path.isfile(path)
    assert "Could not write the model card" in capsys.readouterr().out
