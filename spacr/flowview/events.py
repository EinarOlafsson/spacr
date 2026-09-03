"""Small, picklable event values emitted by FlowView instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .model import Edge, Node


@dataclass(frozen=True)
class NodeAdded:
    """Declare a node before it starts running.

    :param node: node snapshot to insert by identifier when that identifier is
        not already declared. A duplicate declaration leaves the existing
        node unchanged.
    """

    node: Node


@dataclass(frozen=True)
class EdgeAdded:
    """Declare a directed relationship between two nodes.

    :param edge: directed transfer to add to the displayed flow graph. An
        equal edge is retained only once.
    """

    edge: Edge


@dataclass(frozen=True)
class StageStarted:
    """Mark a declared or newly observed stage as running.

    :param node: stage snapshot supplying its identifier, label, kind, and
        parameters. An absent node is declared; a prior node keeps its state
        while receiving the new snapshot's parameters.
    :param at: producer timestamp stored as the stage's start time. Starting
        clears any earlier end time and error.
    """

    node: Node
    at: float


@dataclass(frozen=True)
class StageProgress:
    """Report completed and total work for a stage.

    :param node_id: identifier of the existing stage to update. An unknown
        identifier is ignored by the collector.
    :param current: completed-work count stored verbatim, without clamping.
    :param total: expected-work count stored verbatim, without validation.
    """

    node_id: str
    current: int
    total: int


@dataclass(frozen=True)
class StageMetric:
    """Attach one scalar metric to a stage.

    :param node_id: identifier of the existing stage that owns the metric. An
        unknown identifier is ignored by the collector.
    :param name: metric key; a later event with the same name replaces it.
    :param value: float, integer, or string retained without coercion in the
        stage's detached metrics mapping.
    """

    node_id: str
    name: str
    value: float | int | str


@dataclass(frozen=True)
class StageThumbnail:
    """Attach a cached thumbnail path to a stage.

    :param node_id: identifier of the existing stage represented by the
        image. An unknown identifier is ignored by the collector.
    :param path: thumbnail path stored without checking that it exists; trace
        producers normalise path-like values before constructing the event.
    """

    node_id: str
    path: str


@dataclass(frozen=True)
class StageCompleted:
    """Mark a stage as successfully completed.

    :param node_id: identifier of the existing stage that finished. An
        unknown identifier is ignored by the collector.
    :param at: producer timestamp stored as the stage's end time.
    """

    node_id: str
    at: float


@dataclass(frozen=True)
class StageFailed:
    """Mark a stage as failed and retain its formatted traceback.

    :param node_id: identifier of the existing stage that failed. An unknown
        identifier is ignored and does not affect known descendants.
    :param at: failure timestamp stored as the stage's end time and assigned
        to known downstream stages when they are marked skipped.
    :param error: formatted exception or traceback retained on the failed
        stage; its known downstream stages are marked skipped.
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
