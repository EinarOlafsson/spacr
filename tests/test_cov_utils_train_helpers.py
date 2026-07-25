"""Coverage fill for the spacr.utils training-helper block.

Targets the defensive / rarely-taken branches of ``classification_metrics``,
``compute_irm_penalty``, ``_list_torchvision_model_names``, ``choose_model``,
``calculate_loss``, ``pick_best_model`` and ``save_file_lists``.

Everything here is CPU-only and offline: the two real-backbone checks build
``resnet18`` with ``pretrained=False`` at 32x32, and every other
``choose_model`` branch is driven through a monkeypatched ``TorchModel``
double so no weights are ever constructed or downloaded.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest
import torch

from spacr import utils as U


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let a stray figure survive a test in this module."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# classification_metrics — the "no positive-class rows" branch
# ---------------------------------------------------------------------------

def test_classification_metrics_all_negative_labels_gives_nan_pos_accuracy():
    """With no ``label == 1`` rows, pos_accuracy is NaN and neg_accuracy is real."""
    labels = [0, 0, 0]
    probs = [0.10, 0.20, 0.60]          # last one crosses the 0.5 threshold
    out = U.classification_metrics(labels, probs, torch.tensor(0.25), epoch=7)

    assert list(out.index) == ["7"]
    assert np.isnan(out["pos_accuracy"].iloc[0])          # pc_df was empty
    assert out["neg_accuracy"].iloc[0] == pytest.approx(2 / 3)
    assert out["accuracy"].iloc[0] == pytest.approx(2 / 3)
    assert np.isnan(out["prauc"].iloc[0])                 # single-class -> no PR-AUC
    assert out["optimal_threshold"].iloc[0] == 0.5
    assert out["loss"].iloc[0] == pytest.approx(0.25)


def test_classification_metrics_all_negative_and_all_correct():
    """A perfectly-classified all-negative epoch still reports NaN pos_accuracy."""
    out = U.classification_metrics([0, 0], [0.01, 0.02], torch.tensor(0.0), epoch=0)
    assert out["accuracy"].iloc[0] == 1.0
    assert out["neg_accuracy"].iloc[0] == 1.0
    assert np.isnan(out["pos_accuracy"].iloc[0])


# ---------------------------------------------------------------------------
# compute_irm_penalty
# ---------------------------------------------------------------------------

def test_compute_irm_penalty_sums_squared_gradient_dot_products():
    """d(loss*w)/dw == loss, so the penalty is sum of squared pairwise products."""
    device = torch.device("cpu")
    losses = [torch.tensor(0.5), torch.tensor(2.0), torch.tensor(1.5)]
    dummy_w = torch.tensor([1.0], requires_grad=True)

    penalty = U.compute_irm_penalty(losses, dummy_w, device)

    expected = (0.5 * 2.0) ** 2 + (0.5 * 1.5) ** 2 + (2.0 * 1.5) ** 2
    assert float(penalty.detach()) == pytest.approx(expected)
    # create_graph=True means the penalty is still differentiable
    assert penalty.requires_grad


def test_compute_irm_penalty_single_environment_is_zero():
    """With <2 environments there are no pairs, so the penalty is a plain 0.0."""
    penalty = U.compute_irm_penalty(
        [torch.tensor(3.0)], torch.tensor([1.0], requires_grad=True), torch.device("cpu")
    )
    assert penalty == 0.0
    assert isinstance(penalty, float)


def test_compute_irm_penalty_no_environments_is_zero():
    """An empty environment list short-circuits to the 0.0 initialiser."""
    assert U.compute_irm_penalty([], torch.tensor([1.0], requires_grad=True),
                                 torch.device("cpu")) == 0.0


# ---------------------------------------------------------------------------
# _list_torchvision_model_names — the list_models() failure fallback
# ---------------------------------------------------------------------------

def test_list_torchvision_model_names_falls_back_when_list_models_raises(monkeypatch):
    """When ``tv_models.list_models`` blows up we still harvest public callables."""
    fake = types.ModuleType("fake_tv_models")

    def _boom(module=None):
        raise RuntimeError("list_models unavailable in this torchvision")

    fake.list_models = _boom
    fake.tiny_net = lambda **kw: None
    fake._private_net = lambda **kw: None
    fake.NOT_CALLABLE = 42
    monkeypatch.setattr(U, "tv_models", fake)

    names = U._list_torchvision_model_names()

    assert names == {"tiny_net", "list_models"}
    assert "_private_net" not in names       # leading underscore filtered
    assert "NOT_CALLABLE" not in names       # non-callable filtered


def test_list_torchvision_model_names_uses_list_models_when_available(monkeypatch):
    """The newer API result is unioned with the __dict__ scan."""
    fake = types.ModuleType("fake_tv_models")
    fake.list_models = lambda module=None: ["from_api"]
    fake.from_dict = lambda **kw: None
    monkeypatch.setattr(U, "tv_models", fake)

    names = U._list_torchvision_model_names()
    assert {"from_api", "from_dict", "list_models"} <= names


# ---------------------------------------------------------------------------
# choose_model — invalid names, the custom branch, sanity-check failures
# ---------------------------------------------------------------------------

def _fake_torch_model(z, record):
    """Return a TorchModel stand-in whose forward yields ``z``."""

    class _FakeTorchModel:
        def __init__(self, **kwargs):
            record.update(kwargs)

        def eval(self):
            record["eval_called"] = True
            return self

        def __call__(self, x):
            record["forward_shape"] = tuple(x.shape)
            return z

        def __repr__(self):
            return "<FakeTorchModel repr marker>"

    return _FakeTorchModel


def test_choose_model_unknown_name_returns_none_and_reports(capsys):
    """An unknown architecture is rejected before any model is built."""
    out = U.choose_model("definitely_not_a_torchvision_model", torch.device("cpu"))
    assert out is None
    printed = capsys.readouterr().out
    assert "Invalid model_type" in printed
    assert "definitely_not_a_torchvision_model" in printed


def test_choose_model_custom_raises_not_implemented():
    """'custom' is an accepted name but has no wired implementation."""
    with pytest.raises(NotImplementedError, match="CustomCellClassifier"):
        U.choose_model("custom", torch.device("cpu"))


def test_choose_model_returns_none_when_forward_yields_dict(monkeypatch, capsys):
    """A backbone that returns a dict fails the sanity check -> None."""
    record = {}
    monkeypatch.setattr(
        U, "TorchModel", _fake_torch_model({"logits": torch.zeros(1, 2)}, record)
    )

    out = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                         num_classes=2, height=16, width=16)

    assert out is None
    printed = capsys.readouterr().out
    assert "sanity-check failed" in printed
    assert "dict, not logits" in printed
    assert record["eval_called"] is True


def test_choose_model_returns_none_on_wrong_logit_width(monkeypatch, capsys):
    """Logits whose class dimension disagrees with num_classes are rejected."""
    record = {}
    monkeypatch.setattr(U, "TorchModel", _fake_torch_model(torch.zeros(1, 5), record))

    out = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                         num_classes=2, height=16, width=16)

    assert out is None
    printed = capsys.readouterr().out
    assert "Expected logits of shape (1,2)" in printed
    assert record["forward_shape"] == (1, 3, 16, 16)


def test_choose_model_returns_none_on_non_tensor_output(monkeypatch, capsys):
    """A non-Tensor, non-dict return also trips the sanity check."""
    record = {}
    monkeypatch.setattr(U, "TorchModel", _fake_torch_model([0.1, 0.9], record))

    out = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                         num_classes=2, height=16, width=16)

    assert out is None
    assert "Expected logits of shape (1,2)" in capsys.readouterr().out


def test_choose_model_verbose_prints_model_and_forwards_kwargs(monkeypatch, capsys):
    """verbose=True echoes the built model; constructor kwargs are derived correctly."""
    record = {}
    monkeypatch.setattr(U, "TorchModel", _fake_torch_model(torch.zeros(1, 3), record))

    model = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                           dropout_rate=0.0, use_checkpoint=True,
                           num_classes=3, height=64, width=64, verbose=True)

    assert model is not None
    assert "<FakeTorchModel repr marker>" in capsys.readouterr().out
    assert record["model_name"] == "resnet18"
    assert record["pretrained"] is False
    assert record["dropout_rate"] is None      # 0.0 is normalised away
    assert record["use_checkpoint"] is True
    assert record["num_classes"] == 3
    assert record["image_size"] == 64
    assert record["forward_shape"] == (1, 3, 64, 64)


def test_choose_model_falsy_height_defaults_to_224(monkeypatch):
    """height=0 falls back to the 224 default image size."""
    record = {}
    monkeypatch.setattr(U, "TorchModel", _fake_torch_model(torch.zeros(1, 1), record))

    model = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                           num_classes=0, height=0, width=0)

    assert model is not None
    assert record["image_size"] == 224
    assert record["forward_shape"] == (1, 3, 224, 224)
    assert record["num_classes"] == 1          # max(1, int(0))


def test_choose_model_passes_positive_dropout_through(monkeypatch):
    """A positive dropout_rate is forwarded verbatim."""
    record = {}
    monkeypatch.setattr(U, "TorchModel", _fake_torch_model(torch.zeros(1, 2), record))
    U.choose_model("resnet18", torch.device("cpu"), init_weights=True,
                   dropout_rate=0.25, num_classes=2, height=16, width=16)
    assert record["dropout_rate"] == 0.25
    assert record["pretrained"] is True


def test_choose_model_real_backbone_binary_head():
    """Sanity anchor: a real resnet18 with num_classes=1 yields (1,1) logits."""
    model = U.choose_model("resnet18", torch.device("cpu"), init_weights=False,
                           dropout_rate=0.0, num_classes=1, height=32, width=32)
    assert model is not None
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 32, 32))
    assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# calculate_loss — focal reductions and the 1-D multilabel one-hot path
# ---------------------------------------------------------------------------

def test_calculate_loss_focal_binary_sum_and_none_reductions():
    """focal-BCE 'sum'/'none' agree with each other and stay under plain BCE."""
    torch.manual_seed(0)
    logits = torch.tensor([[2.5], [-3.0], [0.4], [-0.2]])
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    per_sample = U.calculate_loss(logits, target, prefer_focal=True, reduction="none")
    total = U.calculate_loss(logits, target, prefer_focal=True, reduction="sum")
    mean = U.calculate_loss(logits, target, prefer_focal=True, reduction="mean")

    assert per_sample.shape == (4, 1)
    assert torch.all(per_sample >= 0)
    assert float(total) == pytest.approx(float(per_sample.sum()), rel=1e-6)
    assert float(mean) == pytest.approx(float(per_sample.mean()), rel=1e-6)

    plain = U.calculate_loss(logits, target, prefer_focal=False, reduction="sum")
    # alpha=1, gamma=2 -> focal is (1-p_t)**2 * BCE, i.e. strictly down-weighted
    assert float(total) < float(plain)


def test_calculate_loss_focal_binary_alpha_scales_linearly():
    """alpha multiplies the summed focal-BCE loss."""
    logits = torch.tensor([[1.0], [-1.0]])
    target = torch.tensor([1.0, 0.0])
    one = U.calculate_loss(logits, target, prefer_focal=True, alpha=1.0, reduction="sum")
    two = U.calculate_loss(logits, target, prefer_focal=True, alpha=2.0, reduction="sum")
    assert float(two) == pytest.approx(2 * float(one), rel=1e-6)


def test_calculate_loss_focal_multiclass_sum_and_none_reductions():
    """focal-CE 'sum'/'none' agree and remain below plain cross-entropy."""
    logits = torch.tensor([[3.0, 0.0, -1.0],
                           [0.1, 0.2, 0.3],
                           [-2.0, 4.0, 0.0]])
    target = torch.tensor([0, 2, 1])

    per_sample = U.calculate_loss(logits, target, prefer_focal=True, reduction="none")
    total = U.calculate_loss(logits, target, prefer_focal=True, reduction="sum")
    mean = U.calculate_loss(logits, target, prefer_focal=True, reduction="mean")

    assert per_sample.shape == (3,)
    assert torch.all(per_sample >= 0)
    assert float(total) == pytest.approx(float(per_sample.sum()), rel=1e-6)
    assert float(mean) == pytest.approx(float(per_sample.mean()), rel=1e-6)

    plain = U.calculate_loss(logits, target, prefer_focal=False, reduction="sum")
    assert float(total) < float(plain)


def test_calculate_loss_focal_multiclass_unknown_reduction_returns_per_sample():
    """Any reduction other than mean/sum falls through to the raw per-sample tensor."""
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([0, 1])
    weird = U.calculate_loss(logits, target, prefer_focal=True, reduction="batchmean")
    none = U.calculate_loss(logits, target, prefer_focal=True, reduction="none")
    assert weird.shape == (2,)
    assert torch.allclose(weird, none)


def test_calculate_loss_multilabel_1d_float_target_is_one_hot_encoded():
    """A 1-D non-long target against (N,C) logits is one-hot encoded first."""
    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0],
                           [0.0, 1.0, -2.0, 0.3],
                           [-1.0, 0.2, 0.4, 3.0]])
    class_idx = torch.tensor([0.0, 1.0, 3.0])          # float -> not the CE path

    got = U.calculate_loss(logits, class_idx)

    one_hot = torch.nn.functional.one_hot(class_idx.long(), num_classes=4).float()
    expected = U.calculate_loss(logits, one_hot)
    assert float(got) == pytest.approx(float(expected), rel=1e-6)
    # and it is emphatically NOT the cross-entropy answer
    ce = torch.nn.functional.cross_entropy(logits, class_idx.long())
    assert float(got) != pytest.approx(float(ce), rel=1e-3)


def test_calculate_loss_multilabel_1d_int32_target_is_one_hot_encoded():
    """int32 (non-long) 1-D targets take the same one-hot multilabel route."""
    logits = torch.zeros(2, 3)
    target = torch.tensor([1, 2], dtype=torch.int32)
    got = U.calculate_loss(logits, target)
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=3).float()
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, one_hot)
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_calculate_loss_multilabel_1d_focal_target_one_hot():
    """The one-hot path also feeds the focal variant."""
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    target = torch.tensor([0.0, 1.0])
    got = U.calculate_loss(logits, target, prefer_focal=True, reduction="sum")
    one_hot = torch.nn.functional.one_hot(target.long(), num_classes=2).float()
    expected = U.calculate_loss(logits, one_hot, prefer_focal=True, reduction="sum")
    assert float(got) == pytest.approx(float(expected), rel=1e-6)


def test_calculate_loss_focal_binary_unknown_reduction_returns_per_sample():
    """focal-BCE with a non mean/sum reduction returns the unreduced tensor."""
    logits = torch.tensor([[0.7], [-0.7]])
    target = torch.tensor([1.0, 0.0])
    weird = U.calculate_loss(logits, target, prefer_focal=True, reduction="elementwise")
    assert weird.shape == (2, 1)
    assert torch.allclose(
        weird, U.calculate_loss(logits, target, prefer_focal=True, reduction="none")
    )


# ---------------------------------------------------------------------------
# pick_best_model — filenames that do not match the epoch/acc pattern
# ---------------------------------------------------------------------------

def test_pick_best_model_ignores_unparsable_checkpoints(tmp_path):
    """Checkpoints without an _epoch_/_acc_ tag sort to the bottom via (0.0, 0)."""
    for name in ("no_tag_at_all.pth", "another_untagged.pth",
                 "m_epoch_2_acc_0.60.pth", "notes.txt"):
        (tmp_path / name).write_bytes(b"")

    best = U.pick_best_model(str(tmp_path))
    assert best == str(tmp_path / "m_epoch_2_acc_0.60.pth")


def test_pick_best_model_all_unparsable_still_returns_a_pth(tmp_path):
    """When nothing parses, every key is (0.0, 0) and a .pth is still returned."""
    names = {"alpha.pth", "beta.pth"}
    for name in names | {"ignore.csv"}:
        (tmp_path / name).write_bytes(b"")

    best = U.pick_best_model(str(tmp_path))
    import os
    assert os.path.basename(best) in names


def test_pick_best_model_prefers_accuracy_over_epoch(tmp_path):
    """Accuracy is the primary sort key; epoch only breaks ties."""
    for name in ("m_epoch_99_acc_0.10.pth", "m_epoch_1_acc_0.99.pth"):
        (tmp_path / name).write_bytes(b"")
    assert U.pick_best_model(str(tmp_path)).endswith("m_epoch_1_acc_0.99.pth")


# ---------------------------------------------------------------------------
# save_file_lists
# ---------------------------------------------------------------------------

def test_save_file_lists_writes_single_column_csv(tmp_path):
    """The list lands in <dst>/<data_set>.csv under a column named data_set."""
    paths = ["/a/one.png", "/a/two.png", "/a/three.png"]
    assert U.save_file_lists(str(tmp_path), "train", paths) is None

    out = tmp_path / "train.csv"
    assert out.exists()
    back = pd.read_csv(out)
    assert list(back.columns) == ["train"]
    assert back["train"].tolist() == paths


def test_save_file_lists_empty_list_writes_header_only(tmp_path):
    """An empty list still produces a well-formed, header-only CSV."""
    U.save_file_lists(str(tmp_path), "val", [])
    back = pd.read_csv(tmp_path / "val.csv")
    assert list(back.columns) == ["val"]
    assert len(back) == 0


# ---------------------------------------------------------------------------
# get_paths_from_db / split_my_dataset — behaviour anchors for this block
# ---------------------------------------------------------------------------

def test_get_paths_from_db_returns_empty_frame_when_nothing_matches():
    """A prcfo index disjoint from png_df yields an empty (but typed) frame."""
    df = pd.DataFrame(index=["p1_r1_c1_f1_o1"])
    png_df = pd.DataFrame({
        "png_path": ["/x/cell_png/a.png"],
        "prcfo": ["p9_r9_c9_f9_o9"],
    })
    out = U.get_paths_from_db(df, png_df)
    assert len(out) == 0
    assert list(out.columns) == ["png_path", "prcfo"]


def test_split_my_dataset_partitions_without_overlap():
    """The two subsets are disjoint and together cover every source index."""
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.arange(20).float().view(20, 1))
    train, val = U.split_my_dataset(ds, split_ratio=0.25)

    assert len(train) == 15 and len(val) == 5
    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices) | set(val.indices) == set(range(20))
