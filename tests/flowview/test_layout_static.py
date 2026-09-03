from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from spacr.flowview.layout import GraphLayout, NodeLayout, compute_layout, layout_graph
from spacr.flowview.model import Edge, Node, NodeKind, NodeState, RunGraph
from spacr.flowview.theme import (
    FAILURE,
    INPUT_ACCENT,
    KIND_ACCENTS,
    OUTPUT_ACCENT,
    PROCESS_ACCENT,
    STATE_LABELS,
    node_accent,
    state_label,
)


def graph(nodes: list[Node], edges: list[Edge] | None = None) -> RunGraph:
    return RunGraph(
        run_id="layout",
        started_at=1.0,
        nodes={node.id: node for node in nodes},
        edges=list(edges or ()),
        spacr_version="test",
        settings_digest="digest",
    )


def test_theme_tokens_cover_every_kind_and_non_colour_state_label():
    assert node_accent("input", "pending") == INPUT_ACCENT
    assert node_accent(NodeKind.PROCESS, NodeState.RUNNING) == PROCESS_ACCENT
    assert node_accent("output", "done") == OUTPUT_ACCENT
    assert node_accent("input", "failed") == FAILURE
    assert [state_label(state) for state in NodeState] == [
        "PENDING",
        "RUNNING",
        "DONE",
        "FAILED",
        "SKIPPED",
    ]
    with pytest.raises(TypeError):
        KIND_ACCENTS[NodeKind.INPUT] = "changed"
    with pytest.raises(TypeError):
        STATE_LABELS[NodeState.DONE] = "changed"


def test_layered_layout_is_deterministic_longest_path_and_semantically_pinned():
    nodes = [
        Node("z-output", "Scores", NodeKind.OUTPUT),
        Node("merge", "Model", NodeKind.PROCESS),
        Node("right", "Tables", NodeKind.PROCESS),
        Node("left", "Images", NodeKind.PROCESS),
        Node("input-b", "Metadata", NodeKind.INPUT),
        Node("input-a", "Raw images", NodeKind.INPUT),
        Node("short-output", "Preview", NodeKind.OUTPUT),
    ]
    edges = [
        Edge("input-a", "left"),
        Edge("input-b", "right"),
        Edge("left", "merge"),
        Edge("right", "merge"),
        Edge("merge", "z-output"),
        Edge("input-a", "short-output"),
        Edge("input-a", "left"),  # duplicate relation is harmless
    ]
    first = layout_graph(graph(nodes, edges))
    second = compute_layout(graph(list(reversed(nodes)), list(reversed(edges))))

    for field in ("nodes", "width", "height"):
        assert f":ivar {field}:" in (GraphLayout.__doc__ or "")
    for field in ("x", "y", "width", "height", "layer", "order"):
        assert f":ivar {field}:" in (NodeLayout.__doc__ or "")
    assert first == second
    assert first["input-a"].layer == first["input-b"].layer == 0
    assert first["left"].layer == first["right"].layer == 1
    assert first["merge"].layer == 2
    assert first["z-output"].layer == first["short-output"].layer == 3
    assert first.width > 0 and first.height > 0
    assert list(iter(first)) == list(first.nodes)
    assert len(first) == 7
    assert first["merge"].centre_y == (
        first["merge"].y + first["merge"].height / 2
    )
    with pytest.raises(FrozenInstanceError):
        first["merge"].x = 1


def test_median_sweeps_and_variable_heights_keep_each_layer_non_overlapping():
    nodes = [
        Node("i1", "I1", "input"),
        Node("i2", "I2", "input"),
        Node("a", "A", "process", metrics={"x": 1, "y": 2, "z": 3, "q": 4}),
        Node("b", "B", "process", thumbnail="missing.png"),
        Node("isolated", "No parent", "process"),
    ]
    edges = [Edge("i2", "a"), Edge("i1", "b")]
    result = layout_graph(
        graph(nodes, edges),
        sweeps=3,
        node_heights={"a": 150.0},
        row_gap=11.0,
    )

    assert result["a"].height == 150.0
    assert result["b"].height > result["isolated"].height
    assert result["isolated"].layer == 0
    for layer in {box.layer for box in result.values()}:
        boxes = sorted(
            (box for box in result.values() if box.layer == layer),
            key=lambda box: box.order,
        )
        assert all(
            before.y + before.height + 11.0 <= after.y
            for before, after in zip(boxes, boxes[1:])
        )


def test_zero_sweeps_output_only_and_empty_graph_have_defined_geometry():
    output_only = layout_graph(
        graph([Node("only", "Only", "output")]),
        sweeps=0,
        margin=0,
        column_gap=0,
    )
    assert output_only["only"].layer == 0
    assert output_only.width == output_only["only"].width

    empty = layout_graph(graph([]), margin=7)
    assert empty == GraphLayout({}, 14.0, 14.0)
    manual = GraphLayout({"x": NodeLayout(1, 2, 3, 4, 0, 0)}, 9, 10)
    assert manual["x"].centre_y == 4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"card_width": 0}, "layout dimensions"),
        ({"column_gap": -1}, "layout dimensions"),
        ({"sweeps": -1}, "sweeps"),
        ({"node_heights": {"x": 0}}, "node height"),
    ],
)
def test_invalid_layout_arguments_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        layout_graph(graph([Node("x", "X", "process")]), **kwargs)


def test_unknown_edge_endpoints_and_cycles_are_refused():
    known = Node("known", "Known", "process")
    with pytest.raises(ValueError, match="'missing'"):
        layout_graph(graph([known], [Edge("missing", "known")]))
    with pytest.raises(ValueError, match="'missing'"):
        layout_graph(graph([known], [Edge("known", "missing")]))

    cyclic = graph(
        [Node("a", "A", "process"), Node("b", "B", "process")],
        [Edge("a", "b"), Edge("b", "a")],
    )
    with pytest.raises(ValueError, match="directed acyclic graph"):
        layout_graph(cyclic)
