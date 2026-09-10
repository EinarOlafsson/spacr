"""Thread-safe bounded collection and event folding for FlowView."""

from __future__ import annotations

import operator
import queue
import threading
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import replace
from typing import Callable, cast

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
    _StageSkipped,
)
from .model import Node, NodeState, RunGraph


class Collector:
    """Own the event queue and the only mutable copy of a run graph.

    Producers call :meth:`emit`, which never waits for queue capacity but may
    briefly contend for the queue lock. A saturated queue drops its oldest
    event and records that the resulting display is sampled. Renderers consume
    recursively detached graph snapshots rather than sharing the collector's
    dictionaries.

    :param graph: run graph to copy recursively; later changes to the caller's
        node parameter dictionaries cannot alter collector state.
    :param max_queue_size: positive integer count of events that may wait
        before the oldest is dropped. It is a bound on memory, not on
        throughput: a producer never waits for capacity, so this is the number
        of events a slow renderer may fall behind by before the display becomes
        sampled rather than complete.
    :raises ValueError: when ``max_queue_size`` is not a positive integer.
    """

    def __init__(self, graph: RunGraph, *, max_queue_size: int = 2_000) -> None:
        """Copy one graph and initialize its bounded event stream.

        :param graph: run graph whose nodes, metrics, parameters, and edge
            container are detached from caller-owned state.
        :param max_queue_size: positive integer maximum of pending events
            retained before the oldest is discarded and :attr:`sampled`
            becomes true.
        :raises ValueError: if ``max_queue_size`` is not a positive integer.

        Recursive-copy errors from caller-provided node payloads propagate.
        """
        if isinstance(max_queue_size, bool):
            raise ValueError("max_queue_size must be a positive integer")
        try:
            queue_size = operator.index(max_queue_size)
        except TypeError as exc:
            raise ValueError(
                "max_queue_size must be a positive integer") from exc
        if queue_size <= 0:
            raise ValueError("max_queue_size must be a positive integer")
        self._run_id = graph.run_id
        self._started_at = graph.started_at
        self._spacr_version = graph.spacr_version
        self._settings_digest = graph.settings_digest
        self._nodes = {
            node_id: self._detached_node(node)
            for node_id, node in graph.nodes.items()
        }
        self._edges = dict.fromkeys(graph.edges)
        self._queue: queue.Queue[FlowEvent] = queue.Queue(maxsize=queue_size)
        self._queue_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._sampled = threading.Event()
        self._revision = 0

    @property
    def sampled(self) -> bool:
        """Whether saturation occurred since :meth:`clear_sampled`."""

        return self._sampled.is_set()

    @property
    def pending(self) -> int:
        """Point-in-time count of events waiting under concurrent producers."""

        return self._queue.qsize()

    @property
    def revision(self) -> int:
        """Monotonic graph revision for renderers that can skip idle work.

        It advances once per recognized immediate event and once per drained
        batch containing at least one recognized event.
        """

        with self._state_lock:
            return self._revision

    def clear_sampled(self) -> None:
        """Clear the sticky saturation indicator without draining the queue.

        :returns: ``None``.
        """

        self._sampled.clear()

    def emit(self, event: FlowEvent) -> bool:
        """Queue an event without waiting for capacity.

        :param event: FlowView event to retain for the next drain; node-bearing
            events are recursively detached before queueing.
        :returns: true when no event was discarded; false when the oldest
            queued event was replaced and :attr:`sampled` was set.
        """

        queued_event = cast(FlowEvent, self._detached_event(event))
        with self._queue_lock:
            kept_every_event = not self._queue.full()
            if not kept_every_event:
                self._queue.get_nowait()
                self._sampled.set()
            self._queue.put_nowait(queued_event)
        return kept_every_event

    def drain(self, limit: int | None = None) -> int:
        """Consume queued events in order and fold one batch.

        :param limit: maximum events to consume, all pending events when
            ``None``; a negative integer behaves as zero.
        :returns: number consumed, including unrecognized objects that entered
            the queue through a dynamically typed caller.
        :raises ValueError: if a non-``None`` limit is not an integer or is a
            boolean.
        """
        if limit is not None:
            if isinstance(limit, bool):
                raise ValueError("limit must be an integer or None")
            try:
                limit = operator.index(limit)
            except TypeError as exc:
                raise ValueError("limit must be an integer or None") from exc
        with self._state_lock:
            with self._queue_lock:
                available = self._queue.qsize()
                take = (available if limit is None
                        else min(available, max(0, limit)))
                events = [self._queue.get_nowait() for _ in range(take)]
            changed = False
            for event in events:
                changed = self._fold_unlocked(event) or changed
            if changed:
                self._revision += 1
            return len(events)

    def fold(self, event: object) -> bool:
        """Fold one event immediately.

        :param event: candidate event to apply under the state lock;
            node-bearing events are recursively detached first.
        :returns: true for a recognized event whose update was accepted;
            duplicate node and edge declarations remain recognized even when
            graph content does not change.
        """

        owned_event = self._detached_event(event)
        with self._state_lock:
            changed = self._fold_unlocked(owned_event)
            if changed:
                self._revision += 1
            return changed

    def snapshot(self) -> RunGraph:
        """Return a recursively detached, renderer-safe graph snapshot.

        :returns: graph whose containers, node metrics, and node parameters
            can be mutated without changing collector state.
        """

        with self._state_lock:
            nodes = {
                node_id: self._detached_node(node)
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
        """Replace an existing node through one callback.

        :param node_id: identifier of the node to replace.
        :param change: callback receiving the current node and returning its
            replacement; callback exceptions propagate.
        :returns: false when the node is absent, otherwise true after storing
            the callback result.
        """
        node = self._nodes.get(node_id)
        if node is None:
            return False
        self._nodes[node_id] = change(node)
        return True

    @staticmethod
    def _detached_node(node: Node) -> Node:
        """Return a node detached through every mutable payload level.

        :param node: node whose metrics and parameter mappings may contain
            caller-owned nested values.
        :returns: dataclass copy with recursively copied mappings.
        """
        return replace(
            node,
            metrics=deepcopy(node.metrics),
            params=deepcopy(node.params),
        )

    @classmethod
    def _detached_event(cls, event: object) -> object:
        """Detach the mutable node payload carried by an event.

        :param event: event supplied by a caller that retains ownership of its
            local values after :meth:`emit` or :meth:`fold` returns.
        :returns: a copy of node-added or stage-started events with a detached
            node; immutable scalar and edge events are returned unchanged.
        """
        if isinstance(event, (NodeAdded, StageStarted)):
            return replace(event, node=cls._detached_node(event.node))
        return event

    def _fold_unlocked(self, event: object) -> bool:
        """Apply one event while the caller holds the state lock.

        :param event: event candidate whose recognized state transition is
            applied to the owned graph.
        :returns: true for accepted event types, including duplicate node and
            edge declarations; false for unknown events or missing targets.
        """
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
                """Return a detached node carrying the captured metric event.

                :param node: existing run-graph node to update.
                :returns: a dataclass copy whose detached metrics mapping sets
                    the captured event's name to its value. The input node and
                    its original mapping are not mutated.
                """
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
        if isinstance(event, _StageSkipped):
            return self._replace_node(
                event.node_id,
                lambda node: replace(
                    node,
                    state=NodeState.SKIPPED,
                    ended_at=event.at,
                ),
            )
        return False

    def _skip_descendants(self, node_id: str, at: float) -> None:
        """Mark every known downstream node skipped using breadth-first order.

        :param node_id: failed ancestor from which traversal begins.
        :param at: failure timestamp copied to each known descendant.
        :returns: ``None`` after cycle-safe traversal; undeclared nodes are
            ignored but their declared descendants are still visited.
        """
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
