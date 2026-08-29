"""Small, picklable event values emitted by FlowView instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .model import Edge, Node


@dataclass(frozen=True)
class NodeAdded:
    """Declare a node before it starts running."""

    node: Node


@dataclass(frozen=True)
class EdgeAdded:
    """Declare a directed relationship between two nodes."""

    edge: Edge


@dataclass(frozen=True)
class StageStarted:
    """Mark a declared or newly observed stage as running."""

    node: Node
    at: float


@dataclass(frozen=True)
class StageProgress:
    """Report completed and total work for a stage."""

    node_id: str
    current: int
    total: int


@dataclass(frozen=True)
class StageMetric:
    """Attach one scalar metric to a stage."""

    node_id: str
    name: str
    value: float | int | str


@dataclass(frozen=True)
class StageThumbnail:
    """Attach a cached thumbnail path to a stage."""

    node_id: str
    path: str


@dataclass(frozen=True)
class StageCompleted:
    """Mark a stage as successfully completed."""

    node_id: str
    at: float


@dataclass(frozen=True)
class StageFailed:
    """Mark a stage as failed and retain its formatted traceback."""

    node_id: str
    at: float
    error: str


@dataclass(frozen=True)
class _StageSkipped:
    """Mark one deliberately bypassed private pipeline stage."""

    node_id: str
    at: float


FlowEvent = Union[
    NodeAdded,
    EdgeAdded,
    StageStarted,
    StageProgress,
    StageMetric,
    StageThumbnail,
    StageCompleted,
    StageFailed,
    _StageSkipped,
]


__all__ = [
    "EdgeAdded",
    "FlowEvent",
    "NodeAdded",
    "StageCompleted",
    "StageFailed",
    "StageMetric",
    "StageProgress",
    "StageStarted",
    "StageThumbnail",
]
