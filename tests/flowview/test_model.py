from dataclasses import FrozenInstanceError, fields

import pytest

from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph


@pytest.mark.parametrize("record", (Node, Edge, RunGraph))
def test_model_records_document_every_serialized_field(record):
    """Every value retained in the graph snapshot is explained in the API."""
    documentation = record.__doc__ or ""
    missing = [
        item.name
        for item in fields(record)
        if f":param {item.name}:" not in documentation
    ]
    assert not missing, f"{record.__name__}: {missing}"


def test_nodes_normalise_enums_copy_mappings_and_are_frozen():
    metrics = {"objects": 12}
    params = {"family": "torch"}
    node = Node(
        id="model",
        label="Model",
        kind="process",
        state="running",
        metrics=metrics,
        params=params,
    )

    metrics["objects"] = 99
    params["family"] = "xgboost"

    assert node.kind is NodeKind.PROCESS
    assert node.state is NodeState.RUNNING
    assert node.metrics == {"objects": 12}
    assert node.params == {"family": "torch"}
    with pytest.raises(FrozenInstanceError):
        node.label = "changed"


def test_graph_json_roundtrip_is_canonical_and_detaches_containers():
    pending = Node(id="input", label="Raw images", kind=NodeKind.INPUT)
    finished = Node(
        id="result",
        label="Scores",
        kind=NodeKind.OUTPUT,
        state=NodeState.DONE,
        started_at=1.25,
        ended_at=2.5,
        progress=(4, 4),
        metrics={"rows": 18, "loss": 0.25, "note": "kept"},
        thumbnail="cache/preview.png",
        params={"nested": {"b": 2, "a": 1}},
        error="retained diagnostic",
    )
    nodes = {"result": finished, "input": pending}
    edges = [
        Edge("result", "archive", volume=18),
        Edge("input", "result", label="files", volume=4),
    ]
    graph = RunGraph(
        run_id="run-1",
        started_at=1.0,
        nodes=nodes,
        edges=edges,
        spacr_version="1.5.0.4",
        settings_digest="abc123",
    )

    nodes.clear()
    edges.clear()
    encoded = graph.to_json()
    restored_from_text = RunGraph.from_json(encoded)
    restored_from_bytes = RunGraph.from_json(encoded.encode("utf-8"))
    restored_from_bytearray = RunGraph.from_json(bytearray(encoded, "utf-8"))

    assert encoded == graph.to_json()
    assert restored_from_text == restored_from_bytes == restored_from_bytearray
    assert restored_from_text.to_json() == encoded
    assert list(restored_from_text.to_dict()["nodes"]) == ["input", "result"]
    assert restored_from_text.nodes["input"].progress is None
    assert restored_from_text.nodes["result"].progress == (4, 4)
    assert restored_from_text.edges[0] == Edge("input", "result", "files", 4)


def test_graph_json_refuses_non_finite_numbers():
    graph = RunGraph(
        run_id="run",
        started_at=float("nan"),
        nodes={},
        edges=[],
        spacr_version="test",
        settings_digest="digest",
    )

    with pytest.raises(ValueError, match="JSON compliant"):
        graph.to_json()
