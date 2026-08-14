"""Defensive and edge-case coverage for :mod:`spacr.deep_spacr`.

These tests deliberately keep neural networks and data loaders tiny.  They
exercise validation and recovery paths which normal end-to-end training does
not reach, while remaining deterministic and CPU-only.
"""
from __future__ import annotations

import builtins
import os
import tarfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_unpack_supervised_batch_rejects_malformed_batches():
    from spacr.deep_spacr import _unpack_supervised_batch

    for batch in (None, torch.zeros(1), (), [torch.zeros(1)]):
        with pytest.raises(ValueError, match=r"at least \(images, labels\)"):
            _unpack_supervised_batch(batch)


def test_apply_model_multiclass_columns(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io

    class Dataset:
        def __init__(self, **_kwargs):
            pass

        def __len__(self):
            return 2

    monkeypatch.setattr(spacr_io, "NoClassDataset", Dataset)
    monkeypatch.setattr(
        ds, "DataLoader",
        lambda *_a, **_k: [(torch.zeros(2, 3, 4, 4), ["a.png", "b.png"])],
    )
    monkeypatch.setattr(
        ds, "_load_inference_model",
        lambda *_a, **_k: (lambda images: torch.tensor(
            [[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]]), {}),
    )
    monkeypatch.setattr("spacr.utils.print_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_empty_device_cache", lambda: None)
    model_path = tmp_path / "three-class.pth"

    frame = ds.apply_model(["a.png", "b.png"], str(model_path), n_jobs=0)

    assert list(frame["predicted_label"]) == [0, 2]
    assert {"prob_class_0", "prob_class_1", "prob_class_2"} <= set(frame)
    assert frame.filter(like="prob_class_").sum(axis=1).to_numpy() == pytest.approx(1)


def test_apply_model_to_tar_reports_crop_format(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io

    class Dataset:
        crop_format = "rgb-v1"

        def __init__(self, *_a, **_k):
            pass

        def __len__(self):
            return 0

    monkeypatch.setattr(spacr_io, "TarImageDataset", Dataset)
    monkeypatch.setattr(ds, "DataLoader", lambda *_a, **_k: [])
    monkeypatch.setattr(ds, "_load_inference_model", lambda *_a, **_k: (nn.Identity(), {}))
    monkeypatch.setattr(ds, "_empty_device_cache", lambda: None)
    monkeypatch.setattr("spacr.utils.process_vision_results", lambda frame, _t: frame)
    monkeypatch.setattr("spacr.utils.print_progress", lambda *_a, **_k: None)
    monkeypatch.setattr("spacr.crops.CROP_FORMAT_RGB", "rgb-v1")
    settings = {
        "tar_path": str(tmp_path / "images.tar"), "model_path": str(tmp_path / "m.pth"),
        "normalize": False, "image_size": 4, "verbose": False, "batch_size": 2,
        "n_jobs": 0, "score_threshold": 0.5,
    }

    frame = ds.apply_model_to_tar(settings)

    assert frame.empty
    assert "Tar crop format rgb-v1 (rgb)" in capsys.readouterr().out


def test_multiclass_metric_failure_is_reported(monkeypatch, caplog):
    import spacr.deep_spacr as ds

    monkeypatch.setattr(ds, "average_precision_score",
                        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad labels")))
    metrics = ds._multiclass_metrics(
        np.array([0, 1]), np.array([[0.8, 0.2], [0.1, 0.9]]))
    assert np.isnan(metrics["prauc"])
    assert "macro average-precision could not be computed" in caplog.text


def test_plot_training_curves_single_class_series_uses_plain_epoch_label():
    import matplotlib.pyplot as plt
    from spacr.deep_spacr import _plot_training_curves

    fig = _plot_training_curves(
        [{"epoch": 1, "loss": 1.0, "accuracy": 0.5,
          "per_class_accuracy": [0.5], "classes": ["only"]}], [], 1)
    assert fig.axes[2].get_xlabel() == "epoch"
    plt.close(fig)


def test_tensorboard_import_failure_is_soft(tmp_path, monkeypatch, capsys):
    from spacr.deep_spacr import _open_tensorboard_writer

    real_import = builtins.__import__

    def fail_tensorboard(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ModuleNotFoundError("tensorboard deliberately unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_tensorboard)
    writer, log_dir = _open_tensorboard_writer(str(tmp_path), enabled=True)
    assert writer is None
    assert log_dir.endswith("tensorboard")
    assert "TensorBoard logging is unavailable" in capsys.readouterr().out


def test_model_card_helpers_cover_irregular_inputs(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds

    split = tmp_path / "train"
    split.mkdir()
    assert ds.dataset_class_balance(tmp_path, classes=["missing"]) == {
        "train": {"missing": 0}}
    assert ds._training_counts({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert ds._imbalance_note({"bad": object(), "also_bad": "x"}) == ""

    markdown = ds.format_model_card({
        "model_file": "m.pth", "training_set": {}, "split_rule": "by well",
        "held_out": {"n": 1, "accuracy": 1.0, "f1_macro": 1.0,
                     "notes": ["one held-out well"], "classes": ["ok"],
                     "per_class_accuracy": [1.0], "class_support": [1]},
    })
    assert "> one held-out well" in markdown

    class BrokenRegistry:
        def register(self, **_kwargs):
            raise OSError("read only")

    assert ds.register_model_card(
        tmp_path / "m.pth", {"module": "train"}, registry=BrokenRegistry()) is None
    assert "card written but not registered" in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"teacher_paths": []}, "teacher_paths"),
        ({"data_loader": []}, "data_loader"),
        ({"alpha": -0.1}, "alpha"),
        ({"temperature": 0}, "temperature"),
        ({"epochs": 0}, "epochs"),
    ],
)
def test_knowledge_transfer_validates_arguments(tmp_path, kwargs, message):
    from spacr.deep_spacr import model_knowledge_transfer

    defaults = dict(teacher_paths=["teacher.pth"], student_save_path=str(tmp_path / "s"),
                    data_loader=[(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))],
                    pretrained=False, epochs=1)
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        model_knowledge_transfer(**defaults)


def test_knowledge_transfer_rejects_teacher_output_mismatch(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    class Teacher(nn.Linear):
        def __init__(self, outputs):
            super().__init__(2, outputs)
            self.num_classes = outputs

    teachers = iter((Teacher(2), Teacher(3)))
    monkeypatch.setattr(ds, "load_model_artifact", lambda *_a, **_k: (next(teachers), {}))
    with pytest.raises(ValueError, match="same output size"):
        ds.model_knowledge_transfer(
            ["a", "b"], str(tmp_path / "s"),
            [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))],
            pretrained=False, epochs=1)


def test_knowledge_transfer_binary_and_one_hot_paths(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    import spacr.utils as utils

    class Binary(nn.Module):
        num_classes = 1
        def __init__(self, **_kwargs):
            super().__init__()
            self.layer = nn.Linear(2, 1)
        def forward(self, x):
            return self.layer(x)

    monkeypatch.setattr(ds, "load_model_artifact", lambda *_a, **_k: (Binary(), {}))
    monkeypatch.setattr(utils, "TorchModel", Binary)
    saved = []
    monkeypatch.setattr(ds, "save_model_artifact", lambda *a, **k: saved.append((a, k)))
    loader = [(torch.zeros(2, 2), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), ["a", "b"])]

    model = ds.model_knowledge_transfer(
        ["teacher"], str(tmp_path / "student"), loader,
        pretrained=False, epochs=1)
    assert isinstance(model, Binary)
    assert saved


def test_model_fusion_rejects_empty_input(tmp_path):
    from spacr.deep_spacr import model_fusion
    with pytest.raises(ValueError, match="at least one checkpoint"):
        model_fusion([], str(tmp_path / "fused"), pretrained=False)


def test_model_fusion_rejects_same_keys_with_different_shapes(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    models = iter((nn.Linear(2, 2), nn.Linear(3, 2)))
    monkeypatch.setattr(ds, "load_model_artifact", lambda *_a, **_k: (next(models), {}))
    with pytest.raises(ValueError, match="identical architecture"):
        ds.model_fusion(["a", "b"], str(tmp_path / "fused"), pretrained=False)


def test_save_top_examples_validates_schema_n_and_labels(tmp_path):
    from spacr.deep_spacr import save_top_class_examples

    with pytest.raises(ValueError, match="prediction columns"):
        save_top_class_examples(pd.DataFrame({"path": []}), "x.tar", tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        save_top_class_examples(pd.DataFrame({"path": [], "pred": []}), "x.tar", tmp_path, n=0)
    multiclass = pd.DataFrame({"path": ["a"], "pred": [0.8],
                               "prob_class_0": [0.8], "prob_class_1": [0.2]})
    with pytest.raises(ValueError, match="1 class labels for 2"):
        save_top_class_examples(multiclass, "x.tar", tmp_path, classes=["only"])
    with pytest.raises(ValueError, match="More than two class labels"):
        save_top_class_examples(pd.DataFrame({"path": ["a"], "pred": [0.8]}),
                                "x.tar", tmp_path, classes=["a", "b", "c"])


def test_save_top_examples_skips_unextractable_tar_member(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    member = SimpleNamespace(name="a.png")
    class Archive:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def getmembers(self): return [member]
        def extractfile(self, _member): return None
    monkeypatch.setattr(tarfile, "open", lambda *_a, **_k: Archive())
    out = ds.save_top_class_examples(
        pd.DataFrame({"path": ["a.png"], "pred": [0.9]}), "unused.tar", tmp_path, n=1)
    assert out == tmp_path
    assert list(tmp_path.rglob("*.png")) == []


def test_deep_spacr_existing_split_cardinality_and_cv_best(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    import spacr.utils as utils

    base = {"train": True, "test": False, "generate_training_dataset": False,
            "generate_full_dataset": False, "apply_model_to_dataset": False,
            "src": ["a", "b"]}
    monkeypatch.setattr("spacr.settings.deep_spacr_defaults", lambda s: dict(s))
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="exactly one dataset root"):
        ds.deep_spacr(base)

    settings = dict(base, src=[str(tmp_path)], cv_best_model_path="best.pth")
    trained = []
    monkeypatch.setattr(ds, "train_test_model",
                        lambda supplied: trained.append(supplied) or "folds.csv")
    ds.deep_spacr(settings)
    assert trained[0]["src"] == [str(tmp_path)]
    assert trained[0]["model_path"] == "best.pth"
    assert trained[0]["cv_results_path"] == "folds.csv"


def test_deep_spacr_rejects_missing_generated_tar(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.utils as utils

    monkeypatch.setattr("spacr.settings.deep_spacr_defaults", lambda s: dict(s))
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(spacr_io, "generate_dataset", lambda _s: None)
    settings = {"src": str(tmp_path), "train": False, "test": False,
                "generate_full_dataset": True, "apply_model_to_dataset": False,
                "tar_path": None}
    with pytest.raises(RuntimeError, match="readable tar"):
        ds.deep_spacr(settings)


def test_generate_activation_map_rejects_unknown_method(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    import spacr.settings as settings_module

    dataset = tmp_path / "set" / "images.tar"
    dataset.parent.mkdir()
    defaults = {
        "dataset": str(dataset), "model_path": "model.pth", "model_type": "resnet18",
        "target_layer": None, "cam_type": "definitely_not_a_method", "n_jobs": 0,
    }
    monkeypatch.setattr(settings_module, "get_default_generate_activation_map_settings",
                        lambda supplied: {**defaults, **supplied})
    monkeypatch.setattr("spacr.utils.save_settings", lambda *_a, **_k: None)
    with pytest.raises(ValueError, match="unknown cam_type"):
        ds.generate_activation_map(defaults)


def test_generate_activation_map_registered_attribution_path(tmp_path, monkeypatch):
    import spacr.attribution as attribution
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.settings as settings_module
    import spacr.utils as utils

    dataset_path = tmp_path / "datasets" / "images.tar"
    dataset_path.parent.mkdir()
    dataset_path.write_bytes(b"present")
    supplied = {
        "dataset": str(dataset_path), "model_path": "model.pth", "model_type": "resnet18",
        "target_layer": None, "cam_type": "occlusion", "n_jobs": 0,
        "image_size": 4, "batch_size": 1, "channels": [1, 2, 3],
        "normalize": False, "normalize_input": False, "save": False,
        "plot": False, "correlation": False, "overlay": False, "shuffle": False,
        "manders_thresholds": [50],
    }
    monkeypatch.setattr(settings_module, "get_default_generate_activation_map_settings",
                        lambda value: {**supplied, **value})
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(utils, "print_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "_empty_device_cache", lambda: None)
    monkeypatch.setattr(ds, "_load_inference_model", lambda *_a, **_k: (nn.Identity(), {}))
    monkeypatch.setattr(spacr_io, "TarImageDataset", lambda *_a, **_k: [0])
    monkeypatch.setattr(
        ds, "DataLoader",
        lambda *_a, **_k: [(torch.zeros(1, 3, 4, 4), ["plate_A01_1_1.png"])],
    )
    calls = []

    class Generator:
        def __init__(self, *_a, **kwargs):
            calls.append(kwargs["method"])
        def compute_maps_and_predictions(self, inputs):
            return torch.ones(1, 4, 4), torch.tensor([1])

    monkeypatch.setattr(attribution, "AttributionMapGenerator", Generator)
    ds.generate_activation_map(dict(supplied))
    assert calls == ["occlusion"]
    assert (dataset_path.parent / "images" / "occlusion" / "class_1" /
            "plate" / "A01").is_dir()


def _tiny_training_loader():
    return [(torch.zeros(2, 2), torch.tensor([0, 1]), ["a.png", "b.png"])]


def _prepare_tiny_training(monkeypatch, *, loaded=None, payload=None, accuracy=0.5,
                           save_path=None):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.utils as utils

    model = nn.Linear(2, 2)
    monkeypatch.setattr(utils, "choose_model", lambda *_a, **_k: model)
    monkeypatch.setattr(utils, "build_loss", lambda **_k: nn.CrossEntropyLoss())
    monkeypatch.setattr(utils, "estimate_class_counts", lambda *_a, **_k: None)
    monkeypatch.setattr(utils, "suggest_training_changes",
                        lambda *_a, **_k: {"summary": {}, "flags": [], "suggestions": []})
    monkeypatch.setattr(spacr_io, "_save_progress", lambda *_a, **_k: None)
    monkeypatch.setattr(
        spacr_io, "_save_model",
        lambda *_a, **_k: save_path,
    )
    monkeypatch.setattr(
        ds, "evaluate_model_performance",
        lambda _m, _l, epoch, **_k: ({"epoch": epoch, "loss": 1.0,
                                      "accuracy": accuracy, "f1_macro": 0.5}, [[], []]),
    )
    monkeypatch.setattr(ds, "_open_tensorboard_writer",
                        lambda dst, enabled=True: (None, os.path.join(dst, "tensorboard")))
    if loaded is not None:
        monkeypatch.setattr(ds, "load_model_artifact",
                            lambda *_a, **_k: (loaded, payload or {}))
    return model


def test_train_model_rejects_missing_initial_checkpoint(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    _prepare_tiny_training(monkeypatch)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ds.train_model(str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(),
                       epochs=1, custom_model_path=str(tmp_path / "missing.pth"),
                       tensorboard=False, write_card=False)


def test_train_model_validates_loaded_checkpoint_state(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")

    wrong = nn.Linear(2, 3)
    wrong.num_classes = 3
    _prepare_tiny_training(monkeypatch, loaded=wrong, payload={})
    with pytest.raises(ValueError, match="3 output classes"):
        ds.train_model(str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(),
                       epochs=1, custom_model_path=str(checkpoint), num_classes=2,
                       tensorboard=False, write_card=False)

    correct = nn.Linear(2, 2)
    correct.num_classes = 2
    _prepare_tiny_training(monkeypatch, loaded=correct, payload={})
    with pytest.raises(ValueError, match="optimizer state is missing"):
        ds.train_model(str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(),
                       epochs=1, resume_checkpoint=str(checkpoint), num_classes=2,
                       tensorboard=False, write_card=False)


def test_train_model_fine_tunes_and_exercises_remaining_schedulers(
        tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    for schedule in ("cosine_warm_restarts", "exponential", "linear"):
        loaded = nn.Linear(2, 2)
        loaded.num_classes = 2
        _prepare_tiny_training(monkeypatch, loaded=loaded, payload={},
                               save_path=str(tmp_path / f"{schedule}.pth"))
        trained, path = ds.train_model(
            str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(),
            epochs=1, custom_model_path=str(checkpoint), num_classes=2,
            schedule=schedule, tensorboard=False, write_card=False)
        assert trained is loaded
        assert path.endswith(f"{schedule}.pth")
    assert "Fine-tuning model weights" in capsys.readouterr().out


def test_train_model_rejects_resume_past_requested_epochs(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    checkpoint = tmp_path / "resume.pth"
    checkpoint.write_bytes(b"stub")
    loaded = nn.Linear(2, 2)
    loaded.num_classes = 2
    payload = {"optimizer_state_dict": {"present": True}}
    _prepare_tiny_training(monkeypatch, loaded=loaded, payload=payload)
    monkeypatch.setattr(ds, "restore_training_state",
                        lambda *_a, **_k: {"epoch": 3, "best_metric": 0.8})
    with pytest.raises(ValueError, match="already completed epoch 3"):
        ds.train_model(str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(),
                       epochs=3, resume_checkpoint=str(checkpoint), num_classes=2,
                       tensorboard=False, write_card=False)


def test_train_model_disables_broken_tensorboard_and_keeps_fallback_checkpoint(
        tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds

    class BrokenWriter:
        def close(self):
            raise RuntimeError("close also failed")

    fallback = str(tmp_path / "fallback.pth")
    _prepare_tiny_training(monkeypatch, accuracy=float("nan"), save_path=fallback)
    monkeypatch.setattr(ds, "_open_tensorboard_writer",
                        lambda *_a, **_k: (BrokenWriter(), str(tmp_path / "tb")))
    monkeypatch.setattr(ds, "_log_tensorboard_epoch",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("write failed")))
    _model, path = ds.train_model(
        str(tmp_path), str(tmp_path), "tiny", _tiny_training_loader(), epochs=1,
        num_classes=2, tensorboard=True, write_card=False)
    assert path == fallback
    assert "TensorBoard logging disabled after an error: write failed" in capsys.readouterr().out


def _cv_settings(tmp_path, **overrides):
    settings = {
        "src": str(tmp_path), "dst": str(tmp_path / "cv"), "cross_validation_folds": 2,
        "image_size": 4, "batch_size": 2, "n_jobs": 0, "pin_memory": False,
        "normalize": False, "train_channels": ["r", "g", "b"], "augment": False,
        "verbose": False, "cv_group_by": "well", "class_balance": "none",
        "random_seed": 7, "model_type": "tiny", "epochs": 1, "learning_rate": 1e-3,
        "init_weights": False, "weight_decay": 0.0, "amsgrad": False,
        "optimizer_type": "adam", "use_checkpoint": False, "dropout_rate": 0.0,
        "intermedeate_save": False, "schedule": None, "loss_type": "cross_entropy",
        "gradient_accumulation": False, "gradient_accumulation_steps": 1,
        "tensorboard": False, "early_stopping_patience": 0,
        "class_folder_names": ["a", "b"], "classifier_evaluation": False,
        "nested_cv_inner_folds": 0, "evaluation_fail_on_leakage": False,
        "leakage_hash_content": False, "leakage_require_identity": True,
    }
    settings.update(overrides)
    os.makedirs(settings["dst"], exist_ok=True)
    return settings


def _patch_cv_dependencies(monkeypatch, *, metrics=None, labels=None,
                           validation_paths=None, train_results=None,
                           nested_layout=None):
    import spacr.classifier_evaluation as evaluation
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.utils as utils

    validation_paths = validation_paths or ["plate_A01_1.png"]
    train_loader = SimpleNamespace(dataset=["plate_B01_1.png"])
    val_loader = SimpleNamespace(dataset=list(validation_paths))
    base_dataset = list(range(4))
    info = {
        "dataset": base_dataset, "folds": [(np.array([0]), np.array([1]))],
        "labels": np.array([0, 1, 0, 1]), "groups": None,
        "fold_table": pd.DataFrame({"fold": [1]}),
    }
    monkeypatch.setattr(spacr_io, "generate_cv_loaders",
                        lambda *_a, **_k: ([(train_loader, val_loader)], info))
    monkeypatch.setattr(spacr_io, "dataset_filenames", lambda dataset: list(dataset))
    monkeypatch.setattr(spacr_io, "dataset_labels", lambda dataset: [0] * len(dataset))
    monkeypatch.setattr(spacr_io, "make_class_balance_sampler",
                        lambda *_a, **_k: (None, None))
    audit = SimpleNamespace(passed=False, critical_levels=["object"])
    monkeypatch.setattr(evaluation, "audit_cv_folds", lambda *_a, **_k: audit)
    monkeypatch.setattr(evaluation, "audit_split_leakage", lambda *_a, **_k: audit)
    monkeypatch.setattr(evaluation, "normalize_probabilities", lambda value, **_k: np.asarray(value))
    monkeypatch.setattr(evaluation, "nested_group_folds",
                        lambda *_a, **_k: nested_layout)
    monkeypatch.setattr(evaluation, "evaluate_predictions", lambda *_a, **_k: {})
    monkeypatch.setattr(evaluation, "write_evaluation_bundle",
                        lambda *_a, **_k: "evaluation.json")
    monkeypatch.setattr(utils, "augment_dataset", lambda dataset, **_k: dataset)
    results = iter(train_results or [(object(), "fold.pth")])
    monkeypatch.setattr(ds, "train_model", lambda **_k: next(results))
    returned_labels = iter(labels or [np.array([0])])
    metric_values = dict(metrics or {"accuracy": 0.8, "loss": 0.2})

    def evaluate(_model, _loader, **_kwargs):
        current = next(returned_labels)
        probabilities = np.tile([[0.8, 0.2]], (len(current), 1))
        return dict(metric_values), [probabilities, current]

    monkeypatch.setattr(ds, "evaluate_model_performance", evaluate)
    monkeypatch.setattr(ds, "summarize_cv_metrics",
                        lambda _frame: pd.DataFrame({"metric": [], "mean": []}))
    monkeypatch.setattr(ds, "_print_cv_report", lambda *_a, **_k: None)
    return info


def test_cross_validation_rejects_resume_and_invalid_nested_count(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    with pytest.raises(ValueError, match="cannot be shared"):
        ds._cross_validate_model(
            _cv_settings(tmp_path, resume_checkpoint="resume.pth"), 2)

    _patch_cv_dependencies(monkeypatch)
    with pytest.raises(ValueError, match="nested_cv_inner_folds=1"):
        ds._cross_validate_model(_cv_settings(tmp_path, nested_cv_inner_folds=1), 2)


@pytest.mark.parametrize("metrics", [
    {"accuracy": 0.8}, {"loss": 0.2}, {"f1_macro": 0.4},
])
def test_cross_validation_selects_best_model_for_available_metric(
        tmp_path, monkeypatch, metrics, capsys):
    import spacr.deep_spacr as ds

    _patch_cv_dependencies(monkeypatch, metrics=metrics)
    settings = _cv_settings(tmp_path, classifier_evaluation=True)
    result = ds._cross_validate_model(settings, 2)
    assert os.path.isfile(result)
    assert settings["cv_best_model_path"] == "fold.pth"
    assert settings["classifier_evaluation_path"] == "evaluation.json"
    output = capsys.readouterr().out
    assert "full CV partition leakage audit failed" in output
    assert "outer fold 1 leakage" in output
    assert "fewer than two successful outer folds" in output


def test_cross_validation_rejects_result_path_label_cardinality_mismatch(
        tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    _patch_cv_dependencies(
        monkeypatch, labels=[np.array([0])],
        validation_paths=["one.png", "two.png"])
    with pytest.raises(RuntimeError, match="1 labels for 2 validation paths"):
        ds._cross_validate_model(_cv_settings(tmp_path), 2)


def test_nested_cross_validation_records_all_inner_failures(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds

    layout = [{"inner": [(np.array([0]), np.array([1])),
                           (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(
        monkeypatch, nested_layout=layout,
        train_results=[(None, None), (None, None)])
    result = ds._cross_validate_model(
        _cv_settings(tmp_path, nested_cv_inner_folds=2, augment=True), 2)
    assert result is None
    assert "every inner model failed; fold skipped" in capsys.readouterr().out


def test_nested_cross_validation_rejects_inconsistent_member_labels(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    layout = [{"inner": [(np.array([0]), np.array([1])),
                           (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(
        monkeypatch, nested_layout=layout,
        train_results=[(object(), None), (object(), None)],
        labels=[np.array([0]), np.array([1])])
    with pytest.raises(RuntimeError, match="labels changed"):
        ds._cross_validate_model(
            _cv_settings(tmp_path, nested_cv_inner_folds=2), 2)


def test_nested_multiclass_cross_validation_uses_multiclass_metrics(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds

    layout = [{"inner": [(np.array([0]), np.array([1])),
                           (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(
        monkeypatch, nested_layout=layout,
        train_results=[(object(), None), (object(), None)],
        labels=[np.array([0]), np.array([0])])
    monkeypatch.setattr(
        ds, "evaluate_model_performance",
        lambda *_a, **_k: ({}, [np.array([[0.7, 0.2, 0.1]]), np.array([0])]),
    )
    settings = _cv_settings(
        tmp_path, nested_cv_inner_folds=2,
        class_folder_names=["a", "b", "c"])
    result = ds._cross_validate_model(settings, 3)
    assert os.path.isfile(result)


def _train_test_settings(tmp_path, **overrides):
    from spacr.settings import get_train_test_model_settings
    settings = get_train_test_model_settings({
        "src": str(tmp_path), "class_folder_names": ["a", "b"],
        "train": True, "test": False, "plot": False, "tensorboard": False,
        "leakage_audit_train_test": False,
    })
    settings.update(overrides)
    return settings


def test_train_test_model_enables_default_cv_reports_balance_and_snapshot_failure(
        tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    import spacr.settings as settings_module
    import spacr.utils as utils

    supplied = _train_test_settings(
        tmp_path, cross_validation_enabled=True, cross_validation_folds=0)
    monkeypatch.setattr(settings_module, "get_train_test_model_settings", lambda _s: supplied)
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "resolve_class_balance_loss",
                        lambda loss, mode, n: (loss, "weighted classes enabled"))
    monkeypatch.setattr(ds, "_cross_validate_model", lambda settings, _n: "folds.csv")
    monkeypatch.setattr(pd.DataFrame, "to_csv",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("read only")))
    assert ds.train_test_model(supplied) == "folds.csv"
    assert supplied["cross_validation_folds"] == 5
    output = capsys.readouterr().out
    assert "weighted classes enabled" in output
    assert "Could not write the per-run settings snapshot" in output


def test_train_test_model_aborts_cleanly_when_model_builder_fails(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.settings as settings_module
    import spacr.utils as utils

    supplied = _train_test_settings(tmp_path, cross_validation_enabled=False,
                                    cross_validation_folds=0)
    monkeypatch.setattr(settings_module, "get_train_test_model_settings", lambda _s: supplied)
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(spacr_io, "generate_loaders", lambda *_a, **_k: ([], [], None))
    monkeypatch.setattr(ds, "train_model", lambda **_k: (None, None))
    assert ds.train_test_model(supplied) is None
    assert "Training aborted" in capsys.readouterr().out


def test_cv_summary_and_reports_cover_empty_and_accuracy_outputs(capsys):
    import spacr.deep_spacr as ds

    folds = pd.DataFrame({"fold": [1, 2], "accuracy": [0.75, 0.85],
                          "loss": [np.nan, np.nan]})
    summary = ds.summarize_cv_metrics(folds)
    assert list(summary["metric"]) == ["accuracy"]
    assert summary.iloc[0]["mean"] == pytest.approx(0.8)
    ds._print_cv_report(folds, summary, 2)
    assert "accuracy across folds" in capsys.readouterr().out

    ds._print_cv_report(folds, pd.DataFrame(), 2)
    assert "no numeric metrics" in capsys.readouterr().out


def test_cross_validation_records_an_ordinary_failed_fold(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    _patch_cv_dependencies(monkeypatch, train_results=[(None, None)])
    assert ds._cross_validate_model(_cv_settings(tmp_path), 2) is None
    assert "could not be built; fold skipped" in capsys.readouterr().out


def test_nested_binary_cross_validation_uses_binary_metrics(tmp_path, monkeypatch):
    import spacr.deep_spacr as ds
    layout = [{"inner": [(np.array([0]), np.array([1])),
                           (np.array([1]), np.array([0]))]}]
    _patch_cv_dependencies(
        monkeypatch, nested_layout=layout,
        train_results=[(object(), None), (object(), None)],
        labels=[np.array([0]), np.array([0])])
    result = ds._cross_validate_model(
        _cv_settings(tmp_path, nested_cv_inner_folds=2), 2)
    assert os.path.isfile(result)


def test_training_counts_rejects_ambiguous_nested_populations():
    from spacr.deep_spacr import _training_counts
    assert _training_counts({"first": {"a": 1}, "second": {"b": 2}}) == {}


def test_train_test_model_one_fold_falls_back_and_aborts_bad_model(
        tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.settings as settings_module
    import spacr.utils as utils

    supplied = _train_test_settings(tmp_path, cross_validation_folds=1)
    monkeypatch.setattr(settings_module, "get_train_test_model_settings", lambda _s: supplied)
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(spacr_io, "generate_loaders", lambda *_a, **_k: ([], [], None))
    monkeypatch.setattr(ds, "train_model", lambda **_k: (None, None))
    assert ds.train_test_model(supplied) is None
    assert "cross_validation_folds=1" in capsys.readouterr().out


def test_train_test_model_tests_the_selected_checkpoint(tmp_path, monkeypatch, capsys):
    import spacr.deep_spacr as ds
    import spacr.io as spacr_io
    import spacr.settings as settings_module
    import spacr.utils as utils

    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"checkpoint")
    supplied = _train_test_settings(tmp_path, train=True, test=True)
    monkeypatch.setattr(settings_module, "get_train_test_model_settings", lambda _s: supplied)
    monkeypatch.setattr(utils, "save_settings", lambda *_a, **_k: None)
    monkeypatch.setattr(spacr_io, "generate_loaders", lambda *_a, **_k: ([], [], None))
    monkeypatch.setattr(spacr_io, "_copy_missclassified", lambda *_a, **_k: None)
    monkeypatch.setattr(ds, "train_model", lambda **_k: (object(), str(checkpoint)))
    loaded = object()
    monkeypatch.setattr(ds, "_load_inference_model", lambda *_a, **_k: (loaded, {}))
    monkeypatch.setattr(
        ds, "test_model_performance",
        lambda **_k: (pd.DataFrame({"pred": [0.5]}), pd.DataFrame({"accuracy": [1.0]})),
    )
    assert ds.train_test_model(supplied) == str(checkpoint)
    assert "Loading selected checkpoint for testing" in capsys.readouterr().out
