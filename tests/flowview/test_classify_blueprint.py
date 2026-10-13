from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from spacr.checkpoint import fingerprint
from spacr.classify import ClassifierFamilyError
from spacr.flowview.classify_blueprint import CLASSIFY_NODE_IDS, classify_graph
from spacr.flowview.model import NodeKind, NodeState


def test_cv_blueprint_is_the_approved_eight_node_active_path(tmp_path):
    settings = {
        "src": tmp_path,
        "classifier_family": "cv",
        "dataset_mode": "annotation",
        "model_type": "resnet50",
        "test_split": 0.25,
        "epochs": 4,
    }
    graph = classify_graph(
        settings,
        run_id="cv-run",
        started_at=12.5,
        spacr_version="test",
    )

    assert tuple(graph.nodes) == CLASSIFY_NODE_IDS
    assert len(graph.nodes) == 8
    assert [(edge.src, edge.dst) for edge in graph.edges] == list(
        zip(CLASSIFY_NODE_IDS, CLASSIFY_NODE_IDS[1:])
    )
    assert graph.nodes["source"].kind is NodeKind.INPUT
    assert graph.nodes["tables"].label == "PNG list"
    assert graph.nodes["dataset"].label == "Dataset build · annotation"
    assert graph.nodes["model"].label == "Model · CV (Torch: resnet50)"
    assert graph.nodes["model"].params == {
        "classifier_family": "cv",
        "model_type": "resnet50",
    }
    assert graph.nodes["scores"].kind is NodeKind.OUTPUT
    assert all(node.state is NodeState.PENDING for node in graph.nodes.values())
    assert graph.nodes["source"].params == {"src": os.fspath(tmp_path)}
    assert graph.started_at == 12.5
    assert graph.spacr_version == "test"
    assert graph.settings_digest == fingerprint(settings)


def test_ml_blueprint_names_only_the_selected_estimator_and_implicit_basis():
    settings = {
        "classifier_family": "ml",
        "model_type": "maxvit_t",
        "model_type_ml": "lightgbm",
        "annotation_column": "class_id",
    }
    graph = classify_graph(settings, run_id="ml-run")

    assert graph.started_at > 0
    assert graph.nodes["tables"].label == "Measurement tables"
    assert graph.nodes["dataset"].label == "Dataset build · annotation"
    assert graph.nodes["model"].label == "Model · ML (lightgbm)"
    assert "maxvit" not in graph.to_json()


def test_defaults_and_json_safe_source_values_are_stable():
    graph = classify_graph(
        {"src": {"plate-b", "plate-a"}},
        run_id="defaults",
        started_at=1,
    )

    assert graph.nodes["dataset"].label == "Dataset build · metadata"
    assert graph.nodes["model"].label == "Model · CV (Torch: maxvit_t)"
    assert graph.nodes["source"].params == {"src": ["plate-a", "plate-b"]}
    assert graph.nodes["split"].params == {"test_split": None}
    assert graph.nodes["training"].params == {"epochs": None}


def test_invalid_family_is_refused_before_a_graph_can_misreport_it():
    with pytest.raises(ClassifierFamilyError):
        classify_graph({"classifier_family": "quantum"}, run_id="bad")


def test_blueprint_import_does_not_load_training_or_gui_stacks():
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import spacr.flowview.classify_blueprint; "
            "blocked=('PySide6','torch','torchvision','pandas','sklearn','cv2'); "
            "assert not any(name == root or name.startswith(root + '.') "
            "for root in blocked for name in sys.modules)"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
