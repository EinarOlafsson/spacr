"""Reachability in a provenance graph visits every artifact once.

Two branches fed into the same artifact put that artifact on the frontier
twice; a provenance cycle recorded in the registry would loop forever. Both
walks of :class:`spacr.pipeline_graph.PipelineGraph` -- downstream ("what does
re-running this invalidate?") and upstream ("what was this derived from?") --
must report each node once, in depth / newest / id order, and terminate.
"""
from __future__ import annotations

from spacr.pipeline_graph import Edge, Node, PipelineGraph


def _node(artifact_id: str, depth: int) -> Node:
    """A measurements-db node at ``depth``, created ``depth`` ns after 1000."""
    return Node(
        artifact_id=artifact_id,
        project="/p",
        kind="measurements-db",
        role="out",
        module="measure",
        path=f"/p/{artifact_id}.db",
        depth=depth,
        created_ns=1_000 + depth,
    )


def _graph(nodes, edges) -> PipelineGraph:
    """A one-project graph of ``(id, depth)`` nodes and ``(source, target)`` edges."""
    return PipelineGraph(
        project="/p",
        nodes=tuple(_node(name, depth) for name, depth in nodes),
        edges=tuple(Edge(source, target) for source, target in edges),
    )


_DIAMOND = _graph((("a", 0), ("b", 1), ("c", 1), ("d", 2)),
                  (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")))
_CYCLE = _graph((("x", 0), ("y", 1)), (("x", "y"), ("y", "x")))


def test_graph_walks_report_each_reachable_node_once_in_order(subtests):
    """Diamond joins are reported once, cycles terminate, the start is excluded."""
    cases = [
        ("diamond downstream of the root", _DIAMOND, "downstream", "a",
         ["b", "c", "d"]),
        ("diamond downstream of a branch", _DIAMOND, "downstream", "b", ["d"]),
        ("diamond upstream of the join", _DIAMOND, "upstream", "d",
         ["a", "b", "c"]),
        ("diamond upstream of the root", _DIAMOND, "upstream", "a", []),
        ("downstream of the leaf", _DIAMOND, "downstream", "d", []),
        ("cycle downstream", _CYCLE, "downstream", "x", ["y"]),
        ("cycle upstream", _CYCLE, "upstream", "x", ["y"]),
        ("unknown id", _DIAMOND, "downstream", "nope", []),
    ]
    for case_id, graph, direction, start, expected in cases:
        with subtests.test(msg=case_id):
            reached = getattr(graph, direction)(start)
            assert isinstance(reached, tuple)
            assert [node.artifact_id for node in reached] == expected
