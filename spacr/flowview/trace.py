"""Low-cost, failure-isolated instrumentation for FlowView stages."""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import threading
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, TypeVar, cast

from spacr import __version__

from .collector import Collector
from .events import (
    EdgeAdded,
    NodeAdded,
    StageCompleted,
    StageFailed,
    StageMetric,
    StageProgress,
    StageStarted,
    StageThumbnail,
)
from .model import Edge, Node, NodeKind, NodeState, RunGraph

_LOG = logging.getLogger(__name__)
_TRUE_ENV_VALUES = frozenset({"1", "on", "true", "yes"})
_STATE_LOCK = threading.RLock()
_Function = TypeVar("_Function", bound=Callable[..., Any])


def _environment_enabled() -> bool:
    """Return whether the FlowView environment switch contains a true value."""
    return os.environ.get("SPACR_FLOWVIEW", "").strip().casefold() in _TRUE_ENV_VALUES


def _new_collector() -> Collector:
    """Create a collector around a new empty, versioned run graph."""
    started_at = time.time()
    graph = RunGraph(
        run_id=f"flowview-{time.time_ns()}",
        started_at=started_at,
        nodes={},
        edges=[],
        spacr_version=__version__,
        settings_digest=hashlib.sha256(b"{}").hexdigest(),
    )
    return Collector(graph)


_collector = _new_collector()
_enabled = _environment_enabled()


def enable(collector: Collector | None = None) -> Collector:
    """Enable tracing globally and optionally install a run collector."""

    global _collector, _enabled
    with _STATE_LOCK:
        if collector is not None:
            _collector = collector
        _enabled = True
        return _collector


def disable() -> None:
    """Disable tracing globally."""

    global _enabled
    with _STATE_LOCK:
        _enabled = False


def is_enabled() -> bool:
    """Return the process-wide tracing state."""

    with _STATE_LOCK:
        return _enabled


def get_collector() -> Collector:
    """Return the process-wide collector used by new trace events."""

    with _STATE_LOCK:
        return _collector


def _emit_event(factory: Callable[[], object]) -> None:
    """Construct and emit an event without allowing a fault to escape."""

    try:
        get_collector().emit(cast(Any, factory()))
    except Exception:
        _LOG.debug("FlowView event emission failed", exc_info=True)


def _stable_id(prefix: str, label: str) -> str:
    """Build a readable deterministic identifier from ``prefix`` and ``label``."""
    readable = re.sub(r"[^a-z0-9]+", "-", label.casefold()).strip("-") or prefix
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{readable}:{digest}"


class _NullStage:
    """No-op value returned by a disabled stage context."""

    node_id: None = None

    def __call__(self, function: _Function) -> _Function:
        """Return a decorated function unchanged while tracing is disabled."""
        return function

    def __enter__(self) -> "_NullStage":
        """Enter the reusable no-op stage."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        """Leave the no-op stage without suppressing an exception."""
        return False

    def progress(self, current: int, total: int) -> None:
        """Ignore a disabled stage's progress update."""
        return None

    def metric(self, name: str, value: float | int | str) -> None:
        """Ignore a disabled stage's metric."""
        return None

    def thumbnail(self, value: str | os.PathLike[str] | object) -> None:
        """Ignore a disabled stage's thumbnail."""
        return None


_NULL_STAGE = _NullStage()


class _StageRuntime:
    """One active invocation of a stage specification."""

    def __init__(self, spec: "_StageSpec") -> None:
        """Bind this invocation to its immutable stage specification."""
        self._spec = spec
        self.node_id = spec.node_id

    def __enter__(self) -> "_StageRuntime":
        """Emit stage, artifact, edge, and start events for this invocation."""
        stage_node = Node(
            id=self.node_id,
            label=self._spec.label,
            kind=self._spec.kind,
            params=self._spec.params,
        )
        _emit_event(lambda: NodeAdded(stage_node))

        for label in self._spec.consumes:
            artifact_id = _stable_id("artifact", label)
            artifact = Node(
                id=artifact_id,
                label=label,
                kind=NodeKind.INPUT,
                state=NodeState.DONE,
            )
            _emit_event(lambda artifact=artifact: NodeAdded(artifact))
            edge = Edge(src=artifact_id, dst=self.node_id, label="consumes")
            _emit_event(lambda edge=edge: EdgeAdded(edge))

        for label in self._spec.produces:
            artifact_id = _stable_id("artifact", label)
            artifact = Node(id=artifact_id, label=label, kind=NodeKind.OUTPUT)
            _emit_event(lambda artifact=artifact: NodeAdded(artifact))
            edge = Edge(src=self.node_id, dst=artifact_id, label="produces")
            _emit_event(lambda edge=edge: EdgeAdded(edge))

        _emit_event(lambda: StageStarted(stage_node, time.time()))
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        """Emit completion or failure events without suppressing exceptions."""
        ended_at = time.time()
        if exc is None:
            _emit_event(lambda: StageCompleted(self.node_id, ended_at))
            for label in self._spec.produces:
                artifact_id = _stable_id("artifact", label)
                _emit_event(
                    lambda artifact_id=artifact_id: StageCompleted(artifact_id, ended_at)
                )
        else:
            try:
                error = "".join(traceback.format_exception(exc_type, exc, tb))
            except Exception:
                error = f"{type(exc).__name__}: exception text was unavailable"
            _emit_event(lambda: StageFailed(self.node_id, ended_at, error))
        return False

    def progress(self, current: int, total: int) -> None:
        """Emit a progress event for this stage invocation."""
        _emit_event(lambda: StageProgress(self.node_id, current, total))

    def metric(self, name: str, value: float | int | str) -> None:
        """Emit one named metric for this stage invocation."""
        _emit_event(lambda: StageMetric(self.node_id, name, value))

    def thumbnail(self, value: str | os.PathLike[str] | object) -> None:
        """Emit the filesystem path of this stage's representative thumbnail."""
        _emit_event(lambda: StageThumbnail(self.node_id, os.fsdecode(os.fspath(value))))


class _StageSpec:
    """Object that serves as both a decorator and a context manager."""

    def __init__(
        self,
        label: str,
        *,
        kind: NodeKind | str,
        consumes: Iterable[str],
        produces: Iterable[str],
        params: Mapping[str, Any] | None,
        node_id: str | None,
    ) -> None:
        """Normalize the reusable metadata that describes a traced stage."""
        self.label = label
        self.kind = NodeKind(kind)
        self.consumes = tuple(consumes)
        self.produces = tuple(produces)
        self.params = dict(params or {})
        self.node_id = node_id or _stable_id("stage", label)
        self._context: _StageRuntime | None = None

    def __call__(self, function: _Function) -> _Function:
        """Return ``function`` wrapped in a fresh runtime when tracing is enabled."""
        if not is_enabled():
            return function

        @functools.wraps(function)
        def traced(*args: Any, **kwargs: Any) -> Any:
            """Call the captured function with optional stage tracing.

            :param args: positional arguments forwarded unchanged.
            :param kwargs: keyword arguments forwarded unchanged.
            :returns: the captured function's result. Enablement is rechecked
                at invocation time; enabled calls run inside a fresh stage
                runtime, which records completion or failure while preserving
                the original exception.
            """
            if not is_enabled():
                return function(*args, **kwargs)
            with _StageRuntime(self):
                return function(*args, **kwargs)

        return cast(_Function, traced)

    def __enter__(self) -> _StageRuntime | _NullStage:
        """Enter a fresh runtime, or the shared no-op stage when disabled."""
        if not is_enabled():
            return _NULL_STAGE
        runtime = _StageRuntime(self)
        self._context = runtime
        return runtime.__enter__()

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        """Exit and clear the active context runtime without suppressing errors."""
        runtime = self._context
        self._context = None
        if runtime is None:
            return False
        return runtime.__exit__(exc_type, exc, tb)


def stage(
    label: str,
    *,
    kind: NodeKind | str = NodeKind.PROCESS,
    consumes: Iterable[str] = (),
    produces: Iterable[str] = (),
    params: Mapping[str, Any] | None = None,
    node_id: str | None = None,
) -> _StageSpec | _NullStage:
    """Describe one stage for use as a decorator or context manager.

    :param label: Human-readable stage label used in trace events.
    :param kind: Input, process, or output role of the stage node.
    :param consumes: Labels of input artifacts consumed by the stage.
    :param produces: Labels of output artifacts produced by the stage.
    :param params: Optional stage metadata to retain in the trace graph.
    :param node_id: Optional stable node identifier. One is derived from
        ``label`` when this is omitted.
    :returns: Active stage specification, or a no-op stage while tracing is
        disabled.
    """

    if not is_enabled():
        return _NULL_STAGE
    return _StageSpec(
        label,
        kind=kind,
        consumes=consumes,
        produces=produces,
        params=params,
        node_id=node_id,
    )


__all__ = ["disable", "enable", "get_collector", "is_enabled", "stage"]
