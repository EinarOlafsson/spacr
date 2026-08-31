"""Headless visual tokens shared by FlowView renderers.

This module deliberately contains data rather than Qt objects.  Static
renderers, a future live panel, and documentation examples can therefore use
one palette without importing a GUI toolkit.
"""

from __future__ import annotations

from types import MappingProxyType

from .model import NodeKind, NodeState

CANVAS = "#0E1216"
CARD = "#161C22"
TEXT_PRIMARY = "#F1F4F6"
TEXT_SECONDARY = "#B8C0C7"
INPUT_ACCENT = "#5FA8C7"
PROCESS_ACCENT = "#C79A5F"
OUTPUT_ACCENT = "#7FB08A"
FAILURE = "#C2605C"

FONT_FAMILY = "Arial, Helvetica, sans-serif"
LABEL_SIZE = 14
METRIC_SIZE = 11
STATE_SIZE = 10

CARD_WIDTH = 224.0
CARD_MIN_HEIGHT = 104.0
THUMBNAIL_SIZE = 96.0
COLUMN_GAP = 104.0
ROW_GAP = 36.0
CANVAS_MARGIN = 40.0
CORNER_RADIUS = 4.0

KIND_ACCENTS = MappingProxyType(
    {
        NodeKind.INPUT: INPUT_ACCENT,
        NodeKind.PROCESS: PROCESS_ACCENT,
        NodeKind.OUTPUT: OUTPUT_ACCENT,
    }
)

STATE_LABELS = MappingProxyType(
    {
        NodeState.PENDING: "PENDING",
        NodeState.RUNNING: "RUNNING",
        NodeState.DONE: "DONE",
        NodeState.FAILED: "FAILED",
        NodeState.SKIPPED: "SKIPPED",
    }
)


def node_accent(kind: NodeKind | str, state: NodeState | str) -> str:
    """Return the kind accent, reserving alarm red for failed nodes.

    :param kind: Input, process, or output role of the node.
    :param state: Current execution state of the node.
    :returns: Alarm red for a failed node, otherwise the accent for ``kind``.
    """

    normalised_state = NodeState(state)
    if normalised_state is NodeState.FAILED:
        return FAILURE
    return KIND_ACCENTS[NodeKind(kind)]


def state_label(state: NodeState | str) -> str:
    """Return the non-colour state marker printed on every node card.

    :param state: Current execution state of the node.
    :returns: Uppercase state label displayed on the node card.
    """

    return STATE_LABELS[NodeState(state)]


__all__ = [
    "CANVAS",
    "CANVAS_MARGIN",
    "CARD",
    "CARD_MIN_HEIGHT",
    "CARD_WIDTH",
    "COLUMN_GAP",
    "CORNER_RADIUS",
    "FAILURE",
    "FONT_FAMILY",
    "INPUT_ACCENT",
    "KIND_ACCENTS",
    "LABEL_SIZE",
    "METRIC_SIZE",
    "OUTPUT_ACCENT",
    "PROCESS_ACCENT",
    "ROW_GAP",
    "STATE_LABELS",
    "STATE_SIZE",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "THUMBNAIL_SIZE",
    "node_accent",
    "state_label",
]
