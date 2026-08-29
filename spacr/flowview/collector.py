"""Thread-safe bounded collection and event folding for FlowView."""

from __future__ import annotations

import queue
import threading
from collections import defaultdict, deque
from dataclasses import replace
from typing import Callable

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
from .model import Node, NodeState, RunGraph


class Collector:
    """Own the event queue and the only mutable copy of a run graph.

    Producers call :meth:`emit`, which never waits for queue capacity.  A
    saturated queue drops its oldest event and records that the resulting
    display is sampled.  Renderers consume immutable graph snapshots rather
    than sharing the collector's dictionaries.
    """

    def __init__(self, graph: RunGraph, *, max_queue_size: int = 2_000) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be greater than zero")
        self._run_id = graph.run_id
        self._started_at = graph.started_at
        self._spacr_version = graph.spacr_version
        self._settings_digest = graph.settings_digest
        self._nodes = dict(graph.nodes)
        self._edges = dict.fromkeys(graph.edges)
        self._queue: queue.Queue[FlowEvent] = queue.Queue(maxsize=max_queue_size)
        self._queue_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._sampled = threading.Event()
        self._revision = 0

    @property
    def sampled(self) -> bool:
        """Whether saturation has made the display a sampled view."""

        return self._sampled.is_set()

    @property
    def pending(self) -> int:
        """Number of events waiting to be folded."""

        return self._queue.qsize()

    @property
    def revision(self) -> int:
        """Monotonic graph revision for renderers that can skip idle work."""

        with self._state_lock:
            return self._revision

    def clear_sampled(self) -> None:
        """Acknowledge and clear the queue-saturation indicator."""

        self._sampled.clear()

    def emit(self, event: FlowEvent) -> bool:
        """Queue *event* without waiting for capacity.

        ``True`` means no event was discarded.  ``False`` means the oldest
        queued event was replaced and :attr:`sampled` was set.
        """

        with self._queue_lock:
            kept_every_event = not self._queue.full()
            if not kept_every_event:
                self._queue.get_nowait()
                self._sampled.set()
            self._queue.put_nowait(event)
        return kept_every_event

    def drain(self, limit: int | None = None) -> int:
        """Fold up to *limit* queued events and return the number consumed."""

        with self._queue_lock:
            available = self._queue.qsize()
            take = available if limit is None else min(available, max(0, limit))
            events = [self._queue.get_nowait() for _ in range(take)]
        with self._state_lock:
            changed = False
            for event in events:
                changed = self._fold_unlocked(event) or changed
            if changed:
                self._revision += 1
        return len(events)

    def fold(self, event: object) -> bool:
        """Fold one event immediately, returning whether it was recognised."""

        with self._state_lock:
            changed = self._fold_unlocked(event)
            if changed:
                self._revision += 1
            return changed

    def snapshot(self) -> RunGraph:
        """Return a detached, renderer-safe graph snapshot."""

        with self._state_lock:
            nodes = {
                node_id: replace(
                    node,
                    metrics=dict(node.metrics),
                    params=dict(node.params),
                )
                for node_id, node in self._nodes.items()
            }
            edges = list(self._edges)
        return RunGraph(
            run_id=self._run_id,
            started_at=self._started_at,
            nodes=nodes,
            edges=edges,
            spacr_version=self._spacr_version,
            settings_digest=self._settings_digest,
        )

    def _replace_node(self, node_id: str, change: Callable[[Node], Node]) -> bool:
        node = self._nodes.get(node_id)
        if node is None:
            return False
        self._nodes[node_id] = change(node)
        return True

    def _fold_unlocked(self, event: object) -> bool:
        if isinstance(event, NodeAdded):
            self._nodes.setdefault(event.node.id, event.node)
            return True
        if isinstance(event, EdgeAdded):
            self._edges[event.edge] = None
            return True
        if isinstance(event, StageStarted):
            previous = self._nodes.get(event.node.id, event.node)
            params = dict(previous.params)
            params.update(event.node.params)
            self._nodes[event.node.id] = replace(
                previous,
                label=event.node.label,
                kind=event.node.kind,
                state=NodeState.RUNNING,
                started_at=event.at,
                ended_at=None,
                params=params,
                error=None,
            )
            return True
        if isinstance(event, StageProgress):
            return self._replace_node(
                event.node_id,
                lambda node: replace(node, progress=(event.current, event.total)),
            )
        if isinstance(event, StageMetric):

            def add_metric(node: Node) -> Node:
                metrics = dict(node.metrics)
                metrics[event.name] = event.value
                return replace(node, metrics=metrics)

            return self._replace_node(event.node_id, add_metric)
        if isinstance(event, StageThumbnail):
            return self._replace_node(
                event.node_id,
                lambda node: replace(node, thumbnail=event.path),
            )
        if isinstance(event, StageCompleted):
            return self._replace_node(
                event.node_id,
                lambda node: replace(
                    node,
                    state=NodeState.DONE,
                    ended_at=event.at,
                ),
            )
        if isinstance(event, StageFailed):
            changed = self._replace_node(
                event.node_id,
                lambda node: replace(
                    node,
                    state=NodeState.FAILED,
                    ended_at=event.at,
                    error=event.error,
                ),
            )
            if changed:
                self._skip_descendants(event.node_id, event.at)
            return changed
        return False

    def _skip_descendants(self, node_id: str, at: float) -> None:
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in self._edges:
            adjacency[edge.src].append(edge.dst)

        pending = deque(adjacency[node_id])
        seen = {node_id}
        while pending:
            descendant = pending.popleft()
            if descendant in seen:
                continue
            seen.add(descendant)
            self._replace_node(
                descendant,
                lambda node: replace(
                    node,
                    state=NodeState.SKIPPED,
                    ended_at=at,
                ),
            )
            pending.extend(adjacency[descendant])


__all__ = ["Collector"]
