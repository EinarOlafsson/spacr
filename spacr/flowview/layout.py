"""Deterministic layered layout for FlowView directed acyclic graphs."""

from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from statistics import median

from .model import Node, NodeKind, RunGraph
from .theme import (
    CANVAS_MARGIN,
    CARD_MIN_HEIGHT,
    CARD_WIDTH,
    COLUMN_GAP,
    ROW_GAP,
    THUMBNAIL_SIZE,
)


@dataclass(frozen=True)
class NodeLayout:
    """Top-left card position and its stable layer/order assignment.

    :ivar x: horizontal coordinate of the card's left edge on the canvas.
    :ivar y: vertical coordinate of the card's top edge on the canvas.
    :ivar width: rendered card width used for edge routing and canvas bounds.
    :ivar height: rendered card height, including metrics and any thumbnail.
    :ivar layer: deterministic longest-path column assigned to the node.
    :ivar order: stable top-to-bottom position within that layer.
    """

    x: float
    y: float
    width: float
    height: float
    layer: int
    order: int

    @property
    def centre_y(self) -> float:
        """Vertical centre used when routing edges."""

        return self.y + self.height / 2.0


@dataclass(frozen=True)
class GraphLayout(Mapping[str, NodeLayout]):
    """A mapping of node identifiers plus deterministic canvas dimensions.

    :param nodes: node identifiers mapped to their computed card geometries.
    :param width: full canvas width, including both outer margins.
    :param height: full canvas height, including both outer margins.
    """

    nodes: dict[str, NodeLayout]
    width: float
    height: float

    def __getitem__(self, node_id: str) -> NodeLayout:
        return self.nodes[node_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)


def _node_height(node: Node) -> float:
    metric_height = min(len(node.metrics), 3) * 16.0
    text_height = 72.0 + metric_height
    if node.thumbnail is not None:
        text_height += THUMBNAIL_SIZE + 12.0
    return max(CARD_MIN_HEIGHT, text_height)


def _graph_links(
    graph: RunGraph,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    known = set(graph.nodes)
    for edge in sorted(graph.edges, key=lambda item: (item.src, item.dst)):
        if edge.src not in known or edge.dst not in known:
            missing = edge.src if edge.src not in known else edge.dst
            raise ValueError(f"edge references unknown node {missing!r}")
        if edge.dst not in children[edge.src]:
            children[edge.src].append(edge.dst)
            parents[edge.dst].append(edge.src)
    for values in (*parents.values(), *children.values()):
        values.sort()
    return parents, children


def _topological_order(
    node_ids: set[str],
    parents: Mapping[str, list[str]],
    children: Mapping[str, list[str]],
) -> list[str]:
    indegree = {node_id: len(parents.get(node_id, ())) for node_id in node_ids}
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        node_id = heapq.heappop(ready)
        ordered.append(node_id)
        for child in children.get(node_id, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(ordered) != len(node_ids):
        raise ValueError("FlowView layout requires a directed acyclic graph")
    return ordered


def _assign_layers(
    graph: RunGraph,
    ordered: list[str],
    parents: Mapping[str, list[str]],
) -> dict[str, int]:
    layers: dict[str, int] = {}
    for node_id in ordered:
        node = graph.nodes[node_id]
        if node.kind is NodeKind.INPUT:
            layers[node_id] = 0
        else:
            layers[node_id] = max(
                (layers[parent] + 1 for parent in parents.get(node_id, ())),
                default=0,
            )

    outputs = [
        node_id
        for node_id, node in graph.nodes.items()
        if node.kind is NodeKind.OUTPUT
    ]
    if outputs:
        rightmost = max(layers.values(), default=0)
        if any(graph.nodes[node_id].kind is not NodeKind.OUTPUT for node_id in layers):
            rightmost = max(1, rightmost)
        for node_id in outputs:
            layers[node_id] = rightmost
    return layers


def _reorder(
    layer_nodes: dict[int, list[str]],
    neighbours: Mapping[str, list[str]],
    layers: Mapping[str, int],
    layer_sequence: Iterator[int],
) -> None:
    positions = {
        node_id: order
        for nodes in layer_nodes.values()
        for order, node_id in enumerate(nodes)
    }
    for layer in layer_sequence:
        old_positions = {node_id: order for order, node_id in enumerate(layer_nodes[layer])}

        def key(
            node_id: str,
            current_layer: int = layer,
            current_positions: Mapping[str, int] = old_positions,
        ) -> tuple[int, float, int, str]:
            """Rank one node for the current median-sweep layer.

            :param node_id: node to rank within the layer being reordered.
            :param current_layer: loop layer captured when this key is built.
            :param current_positions: that layer's pre-sort positions, captured
                to preserve deterministic ordering for ties and isolated nodes.
            :returns: connected nodes first by the median position of their
                cross-layer neighbours, then prior position and identifier;
                unconnected nodes follow in their prior deterministic order.
            """
            adjacent = [
                positions[other]
                for other in neighbours.get(node_id, ())
                if layers[other] != current_layer
            ]
            if not adjacent:
                return (1, 0.0, current_positions[node_id], node_id)
            return (0, float(median(adjacent)), current_positions[node_id], node_id)

        layer_nodes[layer].sort(key=key)
        positions.update(
            {node_id: order for order, node_id in enumerate(layer_nodes[layer])}
        )


def layout_graph(
    graph: RunGraph,
    *,
    card_width: float = CARD_WIDTH,
    column_gap: float = COLUMN_GAP,
    row_gap: float = ROW_GAP,
    margin: float = CANVAS_MARGIN,
    sweeps: int = 4,
    node_heights: Mapping[str, float] | None = None,
) -> GraphLayout:
    """Lay out *graph* left-to-right using longest paths and median sweeps.

    Inputs are always assigned to layer zero and outputs to the common final
    layer.  All ties are resolved by node identifier, making the result
    independent of dictionary insertion order and edge-list order.
    """

    if min(card_width, column_gap, row_gap, margin) < 0 or card_width == 0:
        raise ValueError("layout dimensions must be non-negative and cards non-zero")
    if sweeps < 0:
        raise ValueError("sweeps must be non-negative")
    if not graph.nodes:
        return GraphLayout({}, margin * 2.0, margin * 2.0)

    parents, children = _graph_links(graph)
    ordered = _topological_order(set(graph.nodes), parents, children)
    layers = _assign_layers(graph, ordered, parents)
    layer_nodes: dict[int, list[str]] = defaultdict(list)
    for node_id in sorted(graph.nodes):
        layer_nodes[layers[node_id]].append(node_id)

    layer_numbers = sorted(layer_nodes)
    for _ in range(sweeps):
        _reorder(layer_nodes, parents, layers, iter(layer_numbers[1:]))
        _reorder(layer_nodes, children, layers, iter(reversed(layer_numbers[:-1])))

    heights: dict[str, float] = {}
    for node_id, node in graph.nodes.items():
        height = (
            float(node_heights[node_id])
            if node_heights is not None and node_id in node_heights
            else _node_height(node)
        )
        if height <= 0:
            raise ValueError(f"node height for {node_id!r} must be positive")
        heights[node_id] = height

    totals = {
        layer: sum(heights[node_id] for node_id in nodes)
        + row_gap * max(0, len(nodes) - 1)
        for layer, nodes in layer_nodes.items()
    }
    content_height = max(totals.values())
    canvas_height = content_height + 2.0 * margin
    placed: dict[str, NodeLayout] = {}
    for layer in layer_numbers:
        y = margin + (content_height - totals[layer]) / 2.0
        for order, node_id in enumerate(layer_nodes[layer]):
            placed[node_id] = NodeLayout(
                x=margin + layer * (card_width + column_gap),
                y=y,
                width=card_width,
                height=heights[node_id],
                layer=layer,
                order=order,
            )
            y += heights[node_id] + row_gap

    canvas_width = (
        2.0 * margin
        + card_width
        + max(layer_numbers) * (card_width + column_gap)
    )
    return GraphLayout(placed, canvas_width, canvas_height)


compute_layout = layout_graph

__all__ = ["GraphLayout", "NodeLayout", "compute_layout", "layout_graph"]
