from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import flowview
from spacr.flowview import _classify_stages
from spacr.flowview.classify_blueprint import CLASSIFY_NODE_IDS, classify_graph
from spacr.flowview.collector import Collector
from spacr.flowview.model import NodeState


@pytest.fixture(autouse=True)
def _restore_trace_state():
    previous_collector = flowview.get_collector()
    previous_enabled = flowview.is_enabled()
    yield
    flowview.enable(previous_collector)
    if not previous_enabled:
        flowview.disable()


def _collector(family: str) -> Collector:
    settings = {
        "classifier_family": family,
        "model_type": "test_cv",
        "model_type_ml": "random_forest",
    }
    collector = Collector(
        classify_graph(settings, run_id=f"{family}-test", started_at=1.0)
    )
    flowview.enable(collector)
    return collector


def _assert_linear_timings(collector: Collector, boundaries: list[float]) -> None:
    collector.drain()
    graph = collector.snapshot()
    assert tuple(graph.nodes) == CLASSIFY_NODE_IDS
    for index, node_id in enumerate(CLASSIFY_NODE_IDS):
        node = graph.nodes[node_id]
        assert node.state is NodeState.DONE
        assert node.started_at == boundaries[index]
        assert node.ended_at == boundaries[index + 1]


def test_stage_coordinator_is_monotonic_and_failure_isolated(monkeypatch):
    collector = _collector("cv")
    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")

    assert _classify_stages._advance("tables", at=2.0)
    assert _classify_stages._advance("dataset", at=3.0)
    assert not _classify_stages._advance("tables", at=30.0)
    assert not _classify_stages._advance("dataset", at=31.0)

    original_emit = collector.emit
    monkeypatch.setattr(
        collector,
        "emit",
        lambda _event: (_ for _ in ()).throw(RuntimeError("display broke")),
    )
    assert _classify_stages._advance("split", at=4.0)
    monkeypatch.setattr(collector, "emit", original_emit)

    error = LookupError("scientific failure")
    assert _classify_stages._fail(error, at=5.0)
    collector.drain()
    graph = collector.snapshot()

    assert graph.nodes["source"].state is NodeState.DONE
    assert graph.nodes["source"].started_at is not None
    assert graph.nodes["source"].ended_at == 2.0
    assert graph.nodes["tables"].started_at == 2.0
    assert graph.nodes["tables"].ended_at == 3.0
    # The split start was deliberately lost, then the restored collector
    # accepted the failure. The display fault did not escape into science.
    assert graph.nodes["split"].started_at is None
    assert graph.nodes["split"].state is NodeState.FAILED
    assert all(
        graph.nodes[node_id].state is NodeState.SKIPPED
        for node_id in CLASSIFY_NODE_IDS[4:]
    )


@pytest.mark.parametrize("family", ["cv", "ml"])
def test_inference_only_jump_skips_intermediate_stages(
    family, monkeypatch
):
    collector = _collector(family)
    monkeypatch.setattr(_classify_stages, "_CLOCK", lambda: 1.0)
    assert _classify_stages._begin(
        {"classifier_family": family}, family
    )
    assert _classify_stages._advance("tables", at=2.0)
    assert _classify_stages._advance("evaluation", at=7.0)
    assert _classify_stages._advance("scores", at=8.0)
    assert _classify_stages._finish(at=9.0)

    collector.drain()
    graph = collector.snapshot()
    for node_id in ("source", "tables", "evaluation", "scores"):
        assert graph.nodes[node_id].state is NodeState.DONE
    for node_id in ("dataset", "split", "model", "training"):
        node = graph.nodes[node_id]
        assert node.state is NodeState.SKIPPED
        assert node.started_at is None
        assert node.ended_at == 7.0
    assert all(node.state is not NodeState.PENDING for node in graph.nodes.values())


def test_successful_train_only_finish_skips_trailing_stages(monkeypatch):
    collector = _collector("cv")
    monkeypatch.setattr(_classify_stages, "_CLOCK", lambda: 1.0)
    assert _classify_stages._begin({"classifier_family": "cv"}, "cv")
    for at, node_id in enumerate(
        ("tables", "dataset", "split", "model", "training"), start=2
    ):
        assert _classify_stages._advance(node_id, at=float(at))
    assert _classify_stages._finish(at=7.0)

    collector.drain()
    graph = collector.snapshot()
    assert all(
        graph.nodes[node_id].state is NodeState.DONE
        for node_id in CLASSIFY_NODE_IDS[:6]
    )
    for node_id in ("evaluation", "scores"):
        node = graph.nodes[node_id]
        assert node.state is NodeState.SKIPPED
        assert node.started_at is None
        assert node.ended_at == 7.0
    assert all(node.state is not NodeState.PENDING for node in graph.nodes.values())


def test_cv_pipeline_events_follow_real_orchestrator_calls(
    tmp_path, monkeypatch
):
    import torch

    import spacr.deep_spacr as deep
    import spacr.io as io
    import spacr.settings as settings_module
    import spacr.utils as utils

    collector = _collector("cv")
    clock = iter(float(value) for value in range(10, 19))
    monkeypatch.setattr(_classify_stages, "_CLOCK", lambda: next(clock))

    dataset_root = tmp_path / "training_all"
    train_path = dataset_root / "train"
    test_path = dataset_root / "test"
    train_path.mkdir(parents=True)
    test_path.mkdir()
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"model")
    tar_path = tmp_path / "dataset.tar"
    tar_path.write_bytes(b"tar")
    operations: list[str] = []

    run_settings = {
        "src": str(tmp_path),
        "train": True,
        "test": True,
        "generate_training_dataset": True,
        "generate_full_dataset": False,
        "apply_model_to_dataset": True,
        "tar_path": None,
        "model_type": "test_cv",
        "train_channels": ["r", "g", "b"],
        "epochs": 1,
        "batch_size": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "amsgrad": False,
        "optimizer_type": "adamw",
        "use_checkpoint": False,
        "dropout_rate": 0.0,
        "n_jobs": 0,
        "val_split": 0.2,
        "pin_memory": False,
        "normalize": False,
        "augment": False,
        "verbose": False,
        "init_weights": False,
        "intermedeate_save": None,
        "schedule": None,
        "loss_type": "cross_entropy",
        "label_smoothing": 0.0,
        "focal_gamma": 2.0,
        "focal_alpha": None,
        "logit_adjust_tau": 1.0,
        "gradient_accumulation": False,
        "gradient_accumulation_steps": 1,
        "image_size": 8,
        "plot": False,
        "tensorboard": False,
        "early_stopping_patience": 0,
        "custom_model_path": None,
        "resume_checkpoint": None,
        "mixed_precision": False,
        "deterministic": False,
        "random_seed": 7,
        "class_balance": "none",
        "cross_validation_enabled": False,
        "cross_validation_folds": 0,
        "leakage_audit_train_test": False,
        "n_top_examples": 1,
    }

    class FakeLoader:
        def __len__(self):
            return 0

        def __iter__(self):
            return iter(())

    class FakeModel:
        num_classes = 2

        def parameters(self):
            return [torch.nn.Parameter(torch.tensor(1.0))]

        def to(self, _device):
            return self

        def train(self):
            return self

    class FakeOptimizer:
        def __init__(self, *_args, **kwargs):
            self.param_groups = [{"lr": kwargs.get("lr", 0.001)}]

        def zero_grad(self, *args, **kwargs):
            return None

    fake_model = FakeModel()
    scored = pd.DataFrame({"path": ["image.png"], "pred": [0.5]})

    monkeypatch.setattr(
        settings_module, "deep_spacr_defaults", lambda supplied: supplied
    )
    monkeypatch.setattr(
        settings_module,
        "get_train_test_model_settings",
        lambda supplied: supplied,
    )
    monkeypatch.setattr(utils, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(deep, "_class_folder_names", lambda _settings: ["n", "p"])
    monkeypatch.setattr(deep, "_empty_device_cache", lambda: None)
    monkeypatch.setattr(deep, "seed_everything", lambda *a, **k: None)
    monkeypatch.setattr(deep, "resolve_seed", lambda _settings: 7)
    monkeypatch.setattr(
        deep, "pick_device", lambda **_kwargs: (torch.device("cpu"), "")
    )
    monkeypatch.setattr(deep, "AdamW", FakeOptimizer)
    monkeypatch.setattr(deep, "_gradient_scaler", lambda *a, **k: object())
    monkeypatch.setattr(
        deep, "_open_tensorboard_writer", lambda *a, **k: (None, None)
    )
    monkeypatch.setattr(
        deep,
        "evaluate_model_performance",
        lambda *a, **k: (
            {"loss": 0.2, "accuracy": 0.8, "f1_macro": 0.8, "epoch": 1},
            [np.empty((0, 2)), []],
        ),
    )
    monkeypatch.setattr(
        deep, "test_model_performance", lambda *a, **k: (pd.DataFrame(), pd.DataFrame())
    )
    monkeypatch.setattr(deep, "_load_inference_model", lambda *a, **k: (fake_model, {}))
    monkeypatch.setattr(
        deep,
        "model_card",
        lambda *a, **k: ({}, str(tmp_path / "model.card.json"), object()),
    )

    monkeypatch.setattr(
        io,
        "generate_training_dataset",
        lambda _settings: (operations.append("tables/dataset") or (str(train_path), str(test_path))),
    )
    monkeypatch.setattr(
        io,
        "generate_loaders",
        lambda *a, **k: (operations.append("split") or (FakeLoader(), FakeLoader(), None)),
    )
    monkeypatch.setattr(
        io,
        "_save_model",
        lambda *a, **k: (operations.append("training") or str(model_path)),
    )
    monkeypatch.setattr(io, "_save_progress", lambda *a, **k: None)
    monkeypatch.setattr(io, "_copy_missclassified", lambda *_args: None)
    monkeypatch.setattr(
        io,
        "generate_dataset",
        lambda _settings: (operations.append("inference dataset") or str(tar_path)),
    )
    monkeypatch.setattr(
        utils,
        "choose_model",
        lambda *a, **k: (operations.append("model") or fake_model),
    )
    monkeypatch.setattr(utils, "build_loss", lambda *a, **k: object())
    monkeypatch.setattr(utils, "estimate_class_counts", lambda *a, **k: None)
    monkeypatch.setattr(
        utils,
        "suggest_training_changes",
        lambda *_args: {"summary": {}, "flags": [], "suggestions": []},
    )
    monkeypatch.setattr(
        deep,
        "apply_model_to_tar",
        lambda _settings: (operations.append("evaluation") or scored),
    )
    monkeypatch.setattr(deep, "save_top_class_examples", lambda *a, **k: None)
    monkeypatch.setattr(
        deep,
        "merge_predictions_into_db",
        lambda *a, **k: operations.append("scores"),
    )

    assert deep.deep_spacr(run_settings) is None
    assert operations.index("tables/dataset") < operations.index("split")
    assert operations.index("split") < operations.index("model")
    assert operations.index("model") < operations.index("training")
    assert operations.index("training") < operations.index("evaluation")
    assert operations.index("evaluation") < operations.index("scores")
    _assert_linear_timings(collector, list(range(10, 19)))


def _ml_frame() -> pd.DataFrame:
    rows = []
    index = []
    for value, column in enumerate((["c1"] * 12) + (["c2"] * 12) + (["c3"] * 6)):
        signal = 0.0 if column == "c1" else 5.0 if column == "c2" else 2.5
        rows.append(
            {
                "columnID": column,
                "cell_channel_3_mean_intensity": signal + value / 100,
                "cell_channel_3_std_intensity": signal / 2 + value / 200,
            }
        )
        index.append(f"p1_r1_{column}_f{value % 3}_o{value}")
    return pd.DataFrame(rows, index=index)


def test_ml_pipeline_events_wrap_load_fit_evaluate_and_writeback(
    tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt

    import spacr.batch_correction as batch_correction
    import spacr.figure_sink as figure_sink
    import spacr.io as io
    import spacr.ml as ml
    import spacr.plot as plot
    import spacr.predictions as predictions
    import spacr.settings as settings_module
    import spacr.utils as utils

    collector = _collector("ml")
    clock = iter(float(value) for value in range(20, 29))
    monkeypatch.setattr(_classify_stages, "_CLOCK", lambda: next(clock))
    frame = _ml_frame()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    paths = [str(results_dir / f"artifact-{index}.csv") for index in range(10)]
    operations: list[str] = []
    figure = plt.figure()

    run_settings = {
        "src": str(tmp_path),
        "verbose": False,
        "nuclei_limit": True,
        "pathogen_limit": 3,
        "dataset_mode": "metadata",
        "annotation_column": None,
        "channel_of_interest": 3,
        "location_column": "columnID",
        "positive_control": "c2",
        "negative_control": "c1",
        "exclude": None,
        "n_repeats": 1,
        "top_features": 2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "learning_rate": 0.01,
        "n_estimators": 2,
        "test_size": 0.25,
        "model_type_ml": "random_forest",
        "n_jobs": 1,
        "remove_low_variance_features": False,
        "remove_highly_correlated_features": False,
        "prune_features": False,
        "cross_validation": False,
        "cv_group_by": "cell",
        "holdout_plate": None,
        "heatmap_feature": "predictions",
        "grouping": "mean",
        "min_max": None,
        "cmap": "viridis",
        "min_cell_count": 1,
    }

    monkeypatch.setattr(
        settings_module, "set_default_analyze_screen", lambda supplied: supplied
    )
    monkeypatch.setattr(utils, "save_settings", lambda *a, **k: None)
    monkeypatch.setattr(
        utils,
        "calculate_shortest_distance",
        lambda value, *a, **k: value,
    )
    monkeypatch.setattr(utils, "get_ml_results_paths", lambda *a, **k: tuple(paths))
    monkeypatch.setattr(
        io,
        "_read_and_merge_data",
        lambda *a, **k: (operations.append("tables") or (frame.copy(), None)),
    )
    monkeypatch.setattr(batch_correction, "correction_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(
        plot,
        "plot_permutation",
        lambda *_args, **_kwargs: figure,
    )
    monkeypatch.setattr(
        plot,
        "plot_feature_importance",
        lambda *_args, **_kwargs: figure,
    )
    monkeypatch.setattr(
        plot,
        "plot_plates",
        lambda *a, **k: (operations.append("evaluation") or figure),
    )
    monkeypatch.setattr(ml, "shap_analysis", lambda *a, **k: figure)
    monkeypatch.setattr(ml, "write_plot", lambda _fig, path, _name: path)
    monkeypatch.setattr(figure_sink, "publish", lambda _fig, path: path)
    monkeypatch.setattr(
        predictions,
        "merge_ml_predictions",
        lambda *a, **k: operations.append("scores"),
    )

    output, plate = ml.generate_ml_scores(run_settings)
    assert len(output) == 10
    assert plate is figure
    assert operations == ["tables", "evaluation", "scores"]
    _assert_linear_timings(collector, list(range(20, 29)))
    plt.close(figure)


def test_scientific_exception_is_identical_and_marks_descendants_skipped(
    monkeypatch
):
    import spacr.deep_spacr as deep
    import spacr.settings as settings_module

    collector = _collector("cv")
    sentinel = LookupError("science failed")

    def fail_science(_settings):
        raise sentinel

    monkeypatch.setattr(settings_module, "deep_spacr_defaults", fail_science)
    with pytest.raises(LookupError) as caught:
        deep.deep_spacr({})
    assert caught.value is sentinel

    collector.drain()
    graph = collector.snapshot()
    assert graph.nodes["source"].state is NodeState.FAILED
    assert "LookupError: science failed" in graph.nodes["source"].error
    assert all(
        graph.nodes[node_id].state is NodeState.SKIPPED
        for node_id in CLASSIFY_NODE_IDS[1:]
    )


def test_lifecycle_faults_cannot_replace_result_or_exception(monkeypatch):
    import spacr.deep_spacr as deep

    _collector("cv")
    sentinel_result = object()
    successful = deep._flowview_pipeline("cv")(
        lambda _settings: sentinel_result
    )
    monkeypatch.setattr(
        _classify_stages,
        "_finish",
        lambda: (_ for _ in ()).throw(RuntimeError("finish failed")),
    )
    assert successful({}) is sentinel_result

    sentinel_error = LookupError("same scientific exception")

    def failed_science(_settings):
        raise sentinel_error

    failed = deep._flowview_pipeline("cv")(failed_science)
    monkeypatch.setattr(
        _classify_stages,
        "_fail",
        lambda _error: (_ for _ in ()).throw(RuntimeError("failure event failed")),
    )
    with pytest.raises(LookupError) as caught:
        failed({})
    assert caught.value is sentinel_error


def test_disabled_direct_stage_gates_import_no_flowview_modules():
    script = """
import os
import sys

os.environ["SPACR_FLOWVIEW"] = "0"
import spacr.deep_spacr as deep
import spacr.ml as ml
assert not any(name.startswith("spacr.flowview") for name in sys.modules)
assert deep._flowview_event("advance", "source") is False
assert ml._flowview_event("advance", "source") is False
assert not any(name.startswith("spacr.flowview") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "SPACR_FLOWVIEW": "0"},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
