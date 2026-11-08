"""Held behaviour for the CPU fallback, input statistics and per-well scoring paths of spacr.deep_spacr.

The GPU memory governor is replaced by a stand-in that always takes the CPU
branch, so the fallback closures run on a tiny linear model. The remaining
tests hold the recorded input statistics of a scoring run, the loader facts
written into a checkpoint, label handling for one-hot targets, and the
train_test_model options (shuffled-label folder, fixed crop scaling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import spacr.deep_spacr as ds  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    import spacr.accelerator as accelerator

    monkeypatch.setattr(accelerator, "_CACHED", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def cpu_governor(monkeypatch):
    """Make the memory governor run every part through its CPU function."""
    import spacr._gpu_governor as gov

    seen = {}

    def chunked(fn, batch, *, site, cpu_fn=None, device=None, settings=None,
                on_retry=None):
        seen["site"] = site
        if on_retry is not None:
            on_retry()
        return cpu_fn(batch)

    monkeypatch.setattr(gov, "_chunked", chunked)
    return seen


def test_governed_predictions_fall_back_to_the_cpu(cpu_governor):
    model = nn.Linear(4, 2).eval()
    images = torch.rand(3, 4)
    with torch.no_grad():
        got = ds._governed_predictions(model, images)
        want = ds._inference_predictions(model, images)
    assert cpu_governor["site"] == "cnn_inference"
    assert len(got) == len(want)
    assert np.allclose(np.asarray(got[0], dtype=float), np.asarray(want[0], dtype=float))
    assert list(got[1]) == list(want[1])


@pytest.mark.parametrize("device", [None, torch.device("cpu")])
def test_governed_logits_fall_back_to_the_cpu_and_return_to_the_device(cpu_governor, device):
    model = nn.Linear(4, 2).eval()
    data = torch.rand(3, 4)
    logits = ds._governed_logits(model, data, device=device)
    assert cpu_governor["site"] == "cnn_evaluation"
    assert torch.allclose(logits, model(data))


@pytest.mark.parametrize("enabled", [False, True])
def test_train_batch_cpu_fallback_computes_gradients(cpu_governor, enabled):
    model = nn.Linear(4, 2)

    class Scaler:
        def is_enabled(self):
            return enabled

        def get_scale(self):
            return 4.0

        def scale(self, loss):
            return loss

    ds._train_batch(model, torch.rand(3, 4), torch.tensor([0, 1, 0]),
                    loss_fn=nn.CrossEntropyLoss(), scaler=Scaler(),
                    mixed_precision=False, device=torch.device("cpu"), head_dim=2)
    assert cpu_governor["site"] == "cnn_training"
    assert model.weight.grad is not None
    assert float(model.weight.grad.abs().sum()) > 0


def test_inference_contract_notes_stats_the_model_was_trained_without():
    metadata = {"preprocessing": {"contract_version": ds._CONTRACT_VERSION,
                                  "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
                                  "image_size": 8, "resize": "center_crop"}}
    mismatches = ds._inference_contract_check(
        metadata, image_size=8, stats=None, on_mismatch="warn")
    names = [m[0] for m in mismatches]
    assert "mean" in names
    assert ("mean", [0.5, 0.5, 0.5], "none") in mismatches


def test_inference_contract_refuses_fixed_scale_models():
    with pytest.raises(ValueError, match="crop_scaling='fixed'"):
        ds._inference_contract_check(
            {"preprocessing": {"crop_scaling": "fixed"}}, image_size=8,
            stats=None, on_mismatch="warn")


def test_tar_scores_report_when_no_prediction_carries_a_well(capsys, tmp_path):
    df = pd.DataFrame({"plateID": ["error", "error"], "prc": ["p_r1_c1", "p_r1_c2"],
                       "pred": [0.2, 0.9], "cv_predictions": [0, 1]})
    assert ds._score_tar_predictions(df, str(tmp_path / "out.csv"), {}) is None
    assert "No prediction carries a plate and well" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_training_pass_metrics_argmax_one_hot_targets():
    metrics = ds._TrainingPassMetrics()
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    target = torch.eye(3)
    metrics.add(logits, target, torch.tensor(0.5))
    result, (probs, labels) = metrics.metrics(1, num_classes=3)
    assert labels == [0, 1, 2]
    assert result["accuracy"] == pytest.approx(1.0)


def test_labelled_class_counts_argmax_one_hot_labels():
    class Data:
        labels = np.eye(3)[[0, 0, 1, 2, 2, 2]]

    class Loader:
        dataset = Data()

    counts = ds._labelled_class_counts(Loader(), 3)
    assert counts.tolist() == [2, 1, 3]


def test_legacy_contract_prefers_the_loaders_recorded_contract():
    class Loader:
        preprocessing_contract = {"crop_scaling": "fixed", "contract_version": 9}

    record = ds._legacy_contract({"train_channels": ["r"]}, "x", Loader())
    assert record == Loader.preprocessing_contract
    assert record is not Loader.preprocessing_contract


def test_crop_folder_facts_empty_dataset_and_marker_fields(tmp_path, monkeypatch):
    class Empty:
        dataset = []

    assert ds._crop_folder_facts(Empty()) == {}
    assert ds._crop_folder_facts(None) == {}

    png = tmp_path / "a.png"
    Image.fromarray(np.zeros((5, 7, 3), dtype=np.uint8)).save(png)

    class Loader:
        dataset = [(None, 0, str(png))]

    import spacr.crops as crops

    monkeypatch.setattr(crops, "crop_format_for_png", lambda path: 4)
    monkeypatch.setattr(crops, "read_crop_folder_marker",
                        lambda folder: {"mask_mode": "cell", "flat_field_sha256": "abc"})
    facts = ds._crop_folder_facts(Loader())
    assert facts == {"crop_format": 4, "crop_intensity": "raw", "crop_size": [7, 5],
                     "mask_mode": "cell", "flat_field_sha256": "abc"}


def _settings(src, **over):
    s = {"src": str(src), "model_type": "tinynet", "classes": ["nc", "pc"],
         "train_channels": ["r", "g"], "epochs": 1, "batch_size": 2,
         "image_size": 8, "n_jobs": 0, "pin_memory": False, "verbose": False,
         "augment": False, "val_split": 0.0}
    s.update(over)
    return s


def test_shuffled_label_control_trains_in_its_own_folder(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    with pytest.raises(ValueError, match="No classes provided"):
        ds.train_test_model(_settings(src, classes=[], train=False, test=False,
                                      control_shuffled_labels=True))
    assert (src / "model" / "tinynet" / "rg_shuffled_labels" / "epochs_1").is_dir()


def test_fixed_crop_scaling_is_refused_with_cross_validation(tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    with pytest.raises(ValueError, match="not available with cross-validation"):
        ds.train_test_model(_settings(src, train=True, test=False,
                                      crop_scaling="fixed",
                                      cross_validation_folds=3))


def test_fixed_crop_scaling_loads_crops_raw(tmp_path, monkeypatch):
    import spacr.io as sio

    src = tmp_path / "ds"
    src.mkdir()
    record = {}

    def fake_generate_loaders(src, mode="train", **kw):
        record.update(kw)
        return ["train_loader"], ["val_loader"], None

    def fake_train_model(**kw):
        return nn.Linear(2, 2), "/tmp/best_model.pth"

    monkeypatch.setattr(sio, "generate_loaders", fake_generate_loaders)
    monkeypatch.setattr(ds, "train_model", fake_train_model)
    ds.train_test_model(_settings(src, train=True, test=False, crop_scaling="fixed",
                                  loss_type="cross_entropy"))
    assert record["crop_loading_policy"] == ds.RAW_FIXED_SCALE
    assert record["crop_scaling"] == "fixed"


def test_train_model_prints_per_class_accuracy_without_validation(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: nn.Sequential(nn.Linear(4, 2)))
    g = torch.Generator().manual_seed(0)
    loader = [(torch.rand(4, 4, generator=g), torch.tensor([0, 1, 0, 1]),
               [f"f{i}.png" for i in range(4)]) for _ in range(2)]
    src = tmp_path / "data"
    src.mkdir()
    dst = tmp_path / "model"
    dst.mkdir()
    ds.train_model(str(src), str(dst), "resnet18", loader, epochs=1,
                   val_loaders=None, num_classes=2, classes=["neg", "pos"],
                   schedule=None, learning_rate=1e-3)
    assert "Train per-class acc.: neg" in capsys.readouterr().out


def test_train_model_without_validation_prints_no_per_class_line_when_metrics_have_none(
        tmp_path, monkeypatch, capsys):
    """A training-only epoch whose metrics carry no per-class breakdown.

    With ``train_metrics_full_pass`` the epoch's training metrics come from
    ``evaluate_model_performance``; when that dict has no
    ``per_class_accuracy`` the progress line is still printed, no empty
    per-class line follows it, and the epoch still counts as the best one.
    """
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: nn.Sequential(nn.Linear(4, 2)))
    seen = []

    def evaluate(model, loader, epoch, **kw):
        seen.append(epoch)
        return {"accuracy": 0.75, "loss": 0.5, "f1_macro": 0.7,
                "epoch": int(epoch)}, [None, None]

    monkeypatch.setattr(ds, "evaluate_model_performance", evaluate)
    g = torch.Generator().manual_seed(0)
    loader = [(torch.rand(4, 4, generator=g), torch.tensor([0, 1, 0, 1]),
               [f"f{i}.png" for i in range(4)]) for _ in range(2)]
    src = tmp_path / "data"
    src.mkdir()
    dst = tmp_path / "model"
    dst.mkdir()
    ds.train_model(str(src), str(dst), "resnet18", loader, epochs=1,
                   val_loaders=None, num_classes=2, classes=["neg", "pos"],
                   schedule=None, learning_rate=1e-3,
                   settings={"train_metrics_full_pass": True})
    out = capsys.readouterr().out
    assert seen == [1]
    assert "Train acc.: 0.750" in out
    assert "per-class acc." not in out
    train = pd.read_csv(dst / "train.csv")
    assert train["accuracy"].tolist() == [0.75]
    assert list(dst.glob("*_best_*.pth"))
