"""Dependency-free data model for a FlowView run graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class NodeKind(str, Enum):
    """The role a node plays in a pipeline graph."""

    INPUT = "input"
    PROCESS = "process"
    OUTPUT = "output"


class NodeState(str, Enum):
    """Lifecycle state of a pipeline node."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Node:
    """One immutable snapshot of a pipeline stage or artifact."""

    id: str
    label: str
    kind: NodeKind
    state: NodeState = NodeState.PENDING
    started_at: float | None = None
    ended_at: float | None = None
    progress: tuple[int, int] | None = None
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    thumbnail: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        """Normalise enums and detach caller-owned mutable dictionaries."""

        object.__setattr__(self, "kind", NodeKind(self.kind))
        object.__setattr__(self, "state", NodeState(self.state))
        object.__setattr__(self, "metrics", dict(self.metrics))
        object.__setattr__(self, "params", dict(self.params))


@dataclass(frozen=True)
class Edge:
    """A directed transfer between two nodes."""

    src: str
    dst: str
    label: str | None = None
    volume: int | None = None


@dataclass(frozen=True)
class RunGraph:
    """A serialisable snapshot of one FlowView run."""

    run_id: str
    started_at: float
    nodes: dict[str, Node]
    edges: list[Edge]
    spacr_version: str
    settings_digest: str

    def __post_init__(self) -> None:
        """Detach the graph's containers from caller-owned containers."""

        object.__setattr__(self, "nodes", dict(self.nodes))
        object.__setattr__(self, "edges", list(self.edges))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical representation."""

        nodes = {
            node_id: {
                "id": node.id,
                "label": node.label,
                "kind": node.kind.value,
                "state": node.state.value,
                "started_at": node.started_at,
                "ended_at": node.ended_at,
                "progress": node.progress,
                "metrics": node.metrics,
                "thumbnail": node.thumbnail,
                "params": node.params,
                "error": node.error,
            }
            for node_id, node in sorted(self.nodes.items())
        }
        edges = [
            {
                "src": edge.src,
                "dst": edge.dst,
                "label": edge.label,
                "volume": edge.volume,
            }
            for edge in sorted(
                self.edges,
                key=lambda item: json.dumps(
                    [item.src, item.dst, item.label, item.volume],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        ]
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "nodes": nodes,
            "edges": edges,
            "spacr_version": self.spacr_version,
            "settings_digest": self.settings_digest,
        }

    def to_json(self) -> str:
        """Serialise the graph deterministically as strict JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunGraph":
        """Restore a graph from :meth:`to_dict` output."""

        nodes = {
            node_id: Node(
                id=node_payload["id"],
                label=node_payload["label"],
                kind=NodeKind(node_payload["kind"]),
                state=NodeState(node_payload["state"]),
                started_at=node_payload["started_at"],
                ended_at=node_payload["ended_at"],
                progress=(
                    tuple(node_payload["progress"])
                    if node_payload["progress"] is not None
                    else None
                ),
                metrics=node_payload["metrics"],
                thumbnail=node_payload["thumbnail"],
                params=node_payload["params"],
                error=node_payload["error"],
            )
            for node_id, node_payload in payload["nodes"].items()
        }
        edges = [
            Edge(
                src=edge_payload["src"],
                dst=edge_payload["dst"],
                label=edge_payload["label"],
                volume=edge_payload["volume"],
            )
            for edge_payload in payload["edges"]
        ]
        return cls(
            run_id=payload["run_id"],
            started_at=payload["started_at"],
            nodes=nodes,
            edges=edges,
            spacr_version=payload["spacr_version"],
            settings_digest=payload["settings_digest"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "RunGraph":
        """Restore a graph from its deterministic JSON record."""

        return cls.from_dict(json.loads(payload))
