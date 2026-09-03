"""Small, picklable event values emitted by FlowView instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .model import Edge, Node


@dataclass(frozen=True)
class NodeAdded:
    """Declare a node before it starts running.

    :ivar node: immutable node definition added to the displayed flow graph.
    """

    node: Node


@dataclass(frozen=True)
class EdgeAdded:
    """Declare a directed relationship between two nodes.

    :ivar edge: directed dependency added to the displayed flow graph.
    """

    edge: Edge


@dataclass(frozen=True)
class StageStarted:
    """Mark a declared or newly observed stage as running.

    :ivar node: stage definition whose run has begun.
    :ivar at: event time as seconds on the producer's clock.
    """

    node: Node
    at: float


@dataclass(frozen=True)
class StageProgress:
    """Report completed and total work for a stage.

    :ivar node_id: stable identifier of the stage being updated.
    :ivar current: number of work units completed so far.
    :ivar total: total work units expected, used as the progress denominator.
    """

    node_id: str
    current: int
    total: int


@dataclass(frozen=True)
class StageMetric:
    """Attach one scalar metric to a stage.

    :ivar node_id: stable identifier of the stage that produced the metric.
    :ivar name: human-readable metric name shown beside the stage.
    :ivar value: scalar value retained without numeric coercion.
    """

    node_id: str
    name: str
    value: float | int | str


@dataclass(frozen=True)
class StageThumbnail:
    """Attach a cached thumbnail path to a stage.

    :ivar node_id: stable identifier of the stage represented by the image.
    :ivar path: filesystem path of the cached thumbnail.
    """

    node_id: str
    path: str


@dataclass(frozen=True)
class StageCompleted:
    """Mark a stage as successfully completed.

    :ivar node_id: stable identifier of the stage that finished.
    :ivar at: completion time as seconds on the producer's clock.
    """

    node_id: str
    at: float


@dataclass(frozen=True)
class StageFailed:
    """Mark a stage as failed and retain its formatted traceback.

    :ivar node_id: stable identifier of the stage that failed.
    :ivar at: failure time as seconds on the producer's clock.
    :ivar error: formatted exception or traceback presented to the user.
    """

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
