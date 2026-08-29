"""Headless-safe public API for optional FlowView instrumentation."""

from .collector import Collector
from .events import (
    EdgeAdded,
    FlowEvent,
    NodeAdded,
    StageCompleted,
    StageFailed,
    StageMetric,
    StageProgress,
    StageStarted,
    StageThumbnail,
)
from .model import Edge, Node, NodeKind, NodeState, RunGraph
from .trace import disable, enable, get_collector, is_enabled, stage

__all__ = [
    "Collector",
    "Edge",
    "EdgeAdded",
    "FlowEvent",
    "Node",
    "NodeAdded",
    "NodeKind",
    "NodeState",
    "RunGraph",
    "StageCompleted",
    "StageFailed",
    "StageMetric",
    "StageProgress",
    "StageStarted",
    "StageThumbnail",
    "disable",
    "enable",
    "get_collector",
    "is_enabled",
    "stage",
]
