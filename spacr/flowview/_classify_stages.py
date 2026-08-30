"""Failure-isolated lifecycle events for the approved Classify graph.

Core pipeline modules reach this private module only through a lazy
``sys.modules`` gate. Keeping the state machine here gives the CV and ML
implementations one clock and one ordering rule without importing FlowView on
their disabled path. Its callables remain private; the module documentation
explains why this instrumentation exists without promoting those helpers into
the public API.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Mapping

from .classify_blueprint import CLASSIFY_NODE_IDS, classify_graph
from .collector import Collector
from .events import (
    NodeAdded,
    StageCompleted,
    StageFailed,
    StageMetric,
    StageStarted,
    _StageSkipped,
)
from .model import Node, NodeKind
from .trace import enable, get_collector

_LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()
_CLOCK = time.time
_TIME_NS = time.time_ns
_INDEX = {node_id: index for index, node_id in enumerate(CLASSIFY_NODE_IDS)}
_FALLBACK_NODES = {
    "source": Node("source", "Source folder", NodeKind.INPUT),
    "tables": Node("tables", "Input tables", NodeKind.INPUT),
    "dataset": Node("dataset", "Dataset build", NodeKind.PROCESS),
    "split": Node("split", "Train/validation split", NodeKind.PROCESS),
    "model": Node("model", "Model", NodeKind.PROCESS),
    "training": Node("training", "Training loop", NodeKind.PROCESS),
    "evaluation": Node("evaluation", "Evaluation", NodeKind.PROCESS),
    "scores": Node("scores", "Scores written to database", NodeKind.OUTPUT),
}


@dataclass
class _RunState:
    current: str | None = None
    last_index: int = -1
    finished: bool = False


_STATES: weakref.WeakKeyDictionary[Collector, _RunState] = (
    weakref.WeakKeyDictionary()
)


def _emit(collector: Collector, event: object) -> bool:
    """Queue one lifecycle event without ever reaching the pipeline."""

    try:
        collector.emit(event)  # type: ignore[arg-type]
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView Classify event emission failed", exc_info=True)
        except BaseException:
            pass
        return False


def _journal_stage(
    node_id: str,
    *,
    label: str | None = None,
    state: str | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
    metrics: Mapping[str, float | int | str] | None = None,
) -> bool:
    """Mirror one FlowView observation into the active run journal."""

    try:
        journal = sys.modules.get("spacr.run_journal")
        current = getattr(journal, "current_run", None)
        if not callable(current):
            return False
        run = current()
        recorder = getattr(run, "_record_stage", None)
        if not callable(recorder):
            return False
        recorder(
            node_id,
            label=label,
            state=state,
            started_at=started_at,
            ended_at=ended_at,
            metrics=dict(metrics or {}),
        )
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView run-journal stage recording failed", exc_info=True)
        except BaseException:
            pass
        return False


def _node(collector: Collector, node_id: str) -> tuple[Node, bool]:
    """Return the predeclared node and whether it was already in the graph."""

    try:
        existing = collector.snapshot().nodes.get(node_id)
    except BaseException:
        existing = None
    return (existing or _FALLBACK_NODES[node_id], existing is not None)


def _skip(collector: Collector, node_id: str, at: float) -> None:
    """Terminalize a stage that this successful run deliberately bypassed."""

    node, declared = _node(collector, node_id)
    if not declared:
        _emit(collector, NodeAdded(node))
    _emit(collector, _StageSkipped(node_id, at))
    _journal_stage(node_id, label=node.label, state="skipped", ended_at=at)


def _fresh_collector(settings: Mapping[str, Any], family: str) -> Collector:
    effective = dict(settings)
    effective["classifier_family"] = family
    started_at = _CLOCK()
    collector = Collector(
        classify_graph(
            effective,
            run_id=f"classify-{_TIME_NS()}",
            started_at=started_at,
        )
    )
    enable(collector)
    return collector


def _begin(settings: Mapping[str, Any] | None, family: str) -> bool:
    """Begin one pipeline, reusing the fresh graph installed by Classify.

    A direct ``deep_spacr`` or ``generate_ml_scores`` call has only the
    generic collector created by :mod:`spacr.flowview.trace`; in that case a
    correctly labelled Classify graph is installed here.  The merged Classify
    entry point installs that graph a few instructions earlier, so it is
    recognised and retained rather than replaced.
    """

    try:
        supplied = settings if isinstance(settings, Mapping) else {}
        collector = get_collector()
        with _LOCK:
            prior_state = _STATES.get(collector)
            try:
                graph = collector.snapshot()
                has_blueprint = tuple(graph.nodes) == CLASSIFY_NODE_IDS
            except BaseException:
                has_blueprint = False
            if prior_state is not None or not has_blueprint:
                collector = _fresh_collector(supplied, family)
            _STATES[collector] = _RunState()
        _advance("source")
        source = supplied.get("src")
        source_count = (
            len(source)
            if isinstance(source, (list, tuple, set, frozenset))
            else int(source is not None)
        )
        _metric("sources", source_count)
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView Classify run setup failed", exc_info=True)
        except BaseException:
            pass
        return False


def _advance(node_id: str, *, at: float | None = None) -> bool:
    """Move forward to a real pipeline boundary at ``at``.

    The graph is deliberately monotonic.  Cross-validation calls the model
    and training helpers once per fold; those repeated lower-level calls are
    part of the same approved ``model``/``training`` stages, not another trip
    backwards through the eight-node pipeline.
    """

    try:
        target_index = _INDEX[node_id]
        collector = get_collector()
        with _LOCK:
            state = _STATES.setdefault(collector, _RunState())
            if state.finished or target_index <= state.last_index:
                return False

            boundary = _CLOCK() if at is None else float(at)
            previous = state.current
            if previous is not None:
                _emit(collector, StageCompleted(previous, boundary))
                _journal_stage(previous, state="done", ended_at=boundary)
            for skipped_id in CLASSIFY_NODE_IDS[
                state.last_index + 1 : target_index
            ]:
                _skip(collector, skipped_id, boundary)

            node, declared = _node(collector, node_id)
            if not declared:
                _emit(collector, NodeAdded(node))
            _emit(collector, StageStarted(node, boundary))
            _journal_stage(
                node_id,
                label=node.label,
                state="running",
                started_at=boundary,
            )
            state.current = node_id
            state.last_index = target_index
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView Classify stage transition failed", exc_info=True)
        except BaseException:
            pass
        return False


def _metric(name: str, value: float | int | str) -> bool:
    """Attach one count or scalar to the currently active Classify stage."""

    try:
        collector = get_collector()
        with _LOCK:
            state = _STATES.get(collector)
            if state is None or state.finished or state.current is None:
                return False
            node_id = state.current
            emitted = _emit(collector, StageMetric(node_id, str(name), value))
            journaled = _journal_stage(node_id, metrics={str(name): value})
        return emitted or journaled
    except BaseException:
        try:
            _LOG.debug("FlowView Classify metric emission failed", exc_info=True)
        except BaseException:
            pass
        return False


def _finish(*, at: float | None = None) -> bool:
    """Complete the last stage of a successful scientific run."""

    try:
        collector = get_collector()
        ended_at = _CLOCK() if at is None else float(at)
        with _LOCK:
            state = _STATES.get(collector)
            if state is None or state.finished or state.current is None:
                return False
            _emit(collector, StageCompleted(state.current, ended_at))
            _journal_stage(state.current, state="done", ended_at=ended_at)
            for skipped_id in CLASSIFY_NODE_IDS[state.last_index + 1 :]:
                _skip(collector, skipped_id, ended_at)
            state.finished = True
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView Classify completion failed", exc_info=True)
        except BaseException:
            pass
        return False


def _fail(error: BaseException, *, at: float | None = None) -> bool:
    """Fail the active stage while leaving ``error`` to its original caller."""

    try:
        collector = get_collector()
        ended_at = _CLOCK() if at is None else float(at)
        with _LOCK:
            state = _STATES.get(collector)
            if state is None or state.finished or state.current is None:
                return False
            try:
                detail = "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                )
            except BaseException:
                detail = f"{type(error).__name__}: exception text unavailable"
            _emit(collector, StageFailed(state.current, ended_at, detail))
            _journal_stage(state.current, state="failed", ended_at=ended_at)
            for skipped_id in CLASSIFY_NODE_IDS[state.last_index + 1 :]:
                node, _declared = _node(collector, skipped_id)
                _journal_stage(
                    skipped_id,
                    label=node.label,
                    state="skipped",
                    ended_at=ended_at,
                )
            state.finished = True
        return True
    except BaseException:
        try:
            _LOG.debug("FlowView Classify failure event failed", exc_info=True)
        except BaseException:
            pass
        return False
