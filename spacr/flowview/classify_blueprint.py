"""The approved eight-node FlowView blueprint for one Classify run."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from spacr import __version__
from spacr.checkpoint import fingerprint, json_safe
from spacr.classify import resolve_family, resolve_ml_model_type
from spacr.training_basis import resolve_basis

from .model import Edge, Node, NodeKind, RunGraph

CLASSIFY_NODE_IDS = (
    "source",
    "tables",
    "dataset",
    "split",
    "model",
    "training",
    "evaluation",
    "scores",
)


def _source_value(settings: Mapping[str, Any]) -> Any:
    source = settings.get("src")
    if isinstance(source, os.PathLike):
        return os.fspath(source)
    return json_safe(source)


def classify_graph(
    settings: Mapping[str, Any],
    *,
    run_id: str,
    started_at: float | None = None,
    spacr_version: str = __version__,
) -> RunGraph:
    """Build the settled live-stage graph for an active Classify family.

    The graph shows only the family the run will execute.  It deliberately
    does not draw a dormant CV/ML branch: Classify's settings already retain
    and grey the inactive family, while FlowView reports what this run did.

    :param settings: Classify settings used to resolve the active family,
        source, dataset basis, and stage metadata.
    :param run_id: Identifier assigned to the resulting run graph.
    :param started_at: Optional Unix timestamp for the start of the run.
        The current time is used when this is omitted.
    :param spacr_version: spaCR version recorded with the graph provenance.
    :returns: Eight-stage run graph for the selected classifier family.
    """

    family = resolve_family(settings)
    dataset_mode = resolve_basis(settings)
    if family == "ml":
        model_name = resolve_ml_model_type(settings)
        table_label = "Measurement tables"
        model_label = f"Model · ML ({model_name})"
    else:
        model_name = str(settings.get("model_type") or "maxvit_t")
        table_label = "PNG list"
        model_label = f"Model · CV (Torch: {model_name})"

    nodes = (
        Node(
            "source",
            "Source folder",
            NodeKind.INPUT,
            params={"src": _source_value(settings)},
        ),
        Node("tables", table_label, NodeKind.INPUT),
        Node(
            "dataset",
            f"Dataset build · {dataset_mode}",
            NodeKind.PROCESS,
            params={"dataset_mode": dataset_mode},
        ),
        Node(
            "split",
            "Train/validation split",
            NodeKind.PROCESS,
            params={"test_split": json_safe(settings.get("test_split"))},
        ),
        Node(
            "model",
            model_label,
            NodeKind.PROCESS,
            params={"classifier_family": family, "model_type": model_name},
        ),
        Node(
            "training",
            "Training loop",
            NodeKind.PROCESS,
            params={"epochs": json_safe(settings.get("epochs"))},
        ),
        Node("evaluation", "Evaluation", NodeKind.PROCESS),
        Node("scores", "Scores written to database", NodeKind.OUTPUT),
    )
    edges = [
        Edge(source, target)
        for source, target in zip(CLASSIFY_NODE_IDS, CLASSIFY_NODE_IDS[1:])
    ]
    return RunGraph(
        run_id=run_id,
        started_at=time.time() if started_at is None else float(started_at),
        nodes={node.id: node for node in nodes},
        edges=edges,
        spacr_version=spacr_version,
        settings_digest=fingerprint(settings),
    )


def _install_classify_collector(settings: Mapping[str, Any]):
    """Install a fresh collector for one enabled Classify invocation.

    This is intentionally private: :mod:`spacr.classify` calls it through a
    lazy import so the disabled entry point does not import FlowView.  The
    caller owns failure isolation because tracing must never affect a
    scientific run.
    """

    from .collector import Collector
    from .trace import enable

    started_at = time.time()
    collector = Collector(
        classify_graph(
            settings,
            run_id=f"classify-{time.time_ns()}",
            started_at=started_at,
        )
    )
    enable(collector)
    return collector


__all__ = ["CLASSIFY_NODE_IDS", "classify_graph"]
