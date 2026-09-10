"""Reachability in a diamond-shaped provenance graph must not double-count.

Two independent branches that were both fed into the same artifact put that
artifact on the frontier twice. If the walk did not remember what it had
already visited, the same node would be reported twice -- and a provenance
cycle would loop forever instead of returning.
"""
from __future__ import annotations

from spacr.pipeline_graph import Edge, Node, PipelineGraph


def _node(artifact_id: str, depth: int) -> Node:
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


def _diamond() -> PipelineGraph:
    """a -> b, a -> c, b -> d, c -> d: d is reachable by two routes."""
    nodes = tuple(_node(name, depth) for name, depth in
                  (("a", 0), ("b", 1), ("c", 1), ("d", 2)))
    edges = (Edge("a", "b"), Edge("a", "c"), Edge("b", "d"), Edge("c", "d"))
    return PipelineGraph(project="/p", nodes=nodes, edges=edges)


def test_a_node_reached_by_two_routes_is_reported_once():
    """The join of a diamond is one downstream artifact, not two."""
    reached = _diamond().downstream("a")
    ids = [n.artifact_id for n in reached]
    assert ids == ["b", "c", "d"], ids
    assert ids.count("d") == 1


def test_upstream_of_a_join_is_reported_once_per_ancestor():
    """Walking back up the same diamond visits the shared root only once."""
    reached = _diamond().upstream("d")
    ids = sorted(n.artifact_id for n in reached)
    assert ids == ["a", "b", "c"], ids


def test_a_provenance_cycle_terminates_instead_of_looping():
    """A cycle recorded in the registry must still return an answer."""
    nodes = (_node("x", 0), _node("y", 1))
    edges = (Edge("x", "y"), Edge("y", "x"))
    graph = PipelineGraph(project="/p", nodes=nodes, edges=edges)
    ids = sorted(n.artifact_id for n in graph.downstream("x"))
    assert ids == ["y"], ids
