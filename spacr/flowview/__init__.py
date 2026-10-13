"""Headless-safe public API for optional FlowView instrumentation."""

from .classify_blueprint import CLASSIFY_NODE_IDS, classify_graph
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
from .export import export
from .feeder import (
    MAX_EVENT_BYTES,
    MultiprocessingFeeder,
    is_transport_event,
    put_event_nowait,
)
from .model import Edge, Node, NodeKind, NodeState, RunGraph
from .trace import disable, enable, get_collector, is_enabled, stage

__all__ = [
    "Collector",
    "CLASSIFY_NODE_IDS",
    "Edge",
    "EdgeAdded",
    "FlowEvent",
    "MAX_EVENT_BYTES",
    "MultiprocessingFeeder",
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
    "export",
    "get_collector",
    "classify_graph",
    "is_enabled",
    "is_transport_event",
    "put_event_nowait",
    "stage",
]
