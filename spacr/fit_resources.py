"""Record process and GPU memory use for each regression stage.

Stage readings are included in run summaries and failure reports so resource
exhaustion can be distinguished from other failures. Measurements are
best-effort: missing ``psutil``, an unavailable Torch runtime, or unsupported
container metrics produce an unavailable reading rather than failing the fit.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

__all__ = [
    "RESOURCE_KEY",
    "STAGE_KEY",
    "host_rss",
    "gpu_allocated",
    "readable",
    "record_stage",
    "peak",
    "describe_resources",
]

#: Where the per-stage readings accumulate on the settings dict.
RESOURCE_KEY = "_regression_resources"

#: Where the current stage name lives, for the failure report to name.
STAGE_KEY = "_regression_stage"


# Resource accounting deliberately lives behind private names until the Qt
# preference and every worker entry point have one settled integration seam.
# Keeping these names private also means adding the recorder does not silently
# expand the translated public API.  The persisted schema, not a Python class,
# is the interface users and support tooling consume.
_PERFORMANCE_LOG_ENV = "SPACR_PERFORMANCE_LOG"
_PERFORMANCE_MODES = frozenset({"off", "summary", "detailed"})
_PERFORMANCE_SCHEMA_VERSION = 1
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
_DEFAULT_SAMPLE_LIMIT = 3600
_MEMORY_MEASURE_ORDER = ("pss", "uss", "rss")


@dataclass(frozen=True)
class _ModeSelection:
    mode: str
    source: str
    requested: Optional[str]
    warning: str = ""


def _select_performance_mode(
        preference: Any = None,
        environ: Optional[Mapping[str, str]] = None) -> _ModeSelection:
    """Resolve the independent performance-log mode without tracing code."""
    environment = os.environ if environ is None else environ
    if _PERFORMANCE_LOG_ENV in environment:
        raw: Any = environment.get(_PERFORMANCE_LOG_ENV)
        source = "environment"
    elif preference is not None:
        raw = preference
        source = "preference"
    else:
        return _ModeSelection("summary", "default", None)

    if isinstance(raw, bool):
        token = "summary" if raw else "off"
    else:
        token = str(raw or "").strip().lower()
        token = {
            "0": "off",
            "false": "off",
            "no": "off",
            "1": "summary",
            "true": "summary",
            "yes": "summary",
            "on": "summary",
            "detail": "detailed",
            "full": "detailed",
        }.get(token, token)
    if not token:
        token = "summary"
    if token in _PERFORMANCE_MODES:
        return _ModeSelection(token, source, str(raw) if raw is not None else None)
    setting_name = _PERFORMANCE_LOG_ENV if source == "environment" else "performance_logging"
    return _ModeSelection(
        "summary",
        source,
        str(raw),
        f"unrecognised {setting_name} value {raw!r}; using summary",
    )


def _performance_mode(
        preference: Any = None,
        environ: Optional[Mapping[str, str]] = None) -> str:
    """Return only the resolved mode for callers that do not need provenance."""
    return _select_performance_mode(preference, environ).mode


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _process_key(pid: int, created: Optional[float]) -> str:
    if created is None:
        return f"{int(pid)}:unknown"
    return f"{int(pid)}:{float(created):.6f}"


def _worker_stamp(worker_kind: str, worker_id: Any) -> Dict[str, Any]:
    """Stamp the calling child so its parent can attribute sampler rows.

    A worker sends this tiny JSON-compatible dictionary over its existing
    result/status channel and the parent passes it to
    :meth:`_ResourceSampler._register_worker`.  PID plus process creation time
    prevents a recycled PID from inheriting an earlier trial's name.
    """
    created: Optional[float] = None
    try:
        import psutil

        created = float(psutil.Process().create_time())
    except Exception:                                            # noqa: BLE001
        pass
    return {
        "pid": os.getpid(),
        "create_time": created,
        "worker_kind": str(worker_kind),
        "worker_id": str(worker_id),
    }


def _memory_reading(process: Any, psutil_module: Any) -> Dict[str, Any]:
    """Read PSS, then USS, then RSS and record every fallback decision."""
    unavailable: List[Dict[str, str]] = []
    full = None
    try:
        full = process.memory_full_info()
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess):
        raise
    except Exception as exc:                                     # noqa: BLE001
        unavailable.extend([
            {"measure": "pss", "reason": exc.__class__.__name__},
            {"measure": "uss", "reason": exc.__class__.__name__},
        ])

    if full is not None:
        for measure in ("pss", "uss"):
            value = getattr(full, measure, None)
            if value is not None:
                return {
                    "memory_bytes": int(value),
                    "memory_measure": measure,
                    "memory_fallbacks": unavailable,
                }
            unavailable.append({"measure": measure, "reason": "not-exposed"})

    try:
        value = process.memory_info().rss
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess):
        raise
    except Exception as exc:                                     # noqa: BLE001
        unavailable.append({"measure": "rss", "reason": exc.__class__.__name__})
        return {
            "memory_bytes": None,
            "memory_measure": None,
            "memory_fallbacks": unavailable,
        }
    return {
        "memory_bytes": int(value),
        "memory_measure": "rss",
        "memory_fallbacks": unavailable,
    }


def _cpu_reading(process: Any, psutil_module: Any) -> Tuple[Optional[float], str]:
    try:
        cpu = process.cpu_times()
        return float(cpu.user) + float(cpu.system), ""
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess):
        raise
    except Exception as exc:                                     # noqa: BLE001
        return None, exc.__class__.__name__


def _python_thread_names() -> Dict[int, str]:
    names: Dict[int, str] = {}
    for thread in threading.enumerate():
        native_id = getattr(thread, "native_id", None)
        if native_id is not None:
            names[int(native_id)] = str(thread.name)
    return names


def _thread_reading(process: Any, psutil_module: Any,
                    names: Optional[Mapping[int, str]] = None) -> Dict[str, Any]:
    try:
        figures = process.threads()
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess):
        raise
    except Exception as exc:                                     # noqa: BLE001
        return {
            "thread_cpu_available": False,
            "thread_cpu_unavailable_reason": exc.__class__.__name__,
            "threads": None,
        }

    rows: List[Dict[str, Any]] = []
    known_names = names or {}
    for figure in figures:
        tid = int(figure.id)
        row = {
            "tid": tid,
            "cpu_user_seconds": float(figure.user_time),
            "cpu_system_seconds": float(figure.system_time),
            "cpu_total_seconds": (
                float(figure.user_time) + float(figure.system_time)
            ),
        }
        if tid in known_names:
            row["name"] = known_names[tid]
        rows.append(row)
    rows.sort(key=lambda item: item["tid"])
    return {
        "thread_cpu_available": True,
        "thread_cpu_unavailable_reason": "",
        "threads": rows,
    }


def _label_for_process(
        labels: Mapping[Tuple[int, Optional[float]], Mapping[str, Any]],
        pid: int,
        created: Optional[float]) -> Optional[Dict[str, str]]:
    label = labels.get((pid, created))
    if label is None:
        label = labels.get((pid, None))
    if label is None:
        return None
    return {
        "kind": str(label.get("worker_kind", "worker")),
        "id": str(label.get("worker_id", pid)),
    }


def _read_process(
        process: Any,
        *,
        root_pid: int,
        detailed: bool,
        labels: Mapping[Tuple[int, Optional[float]], Mapping[str, Any]],
        psutil_module: Any) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    pid = int(process.pid)
    try:
        created = float(process.create_time())
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess) as exc:
        return None, {"pid": pid, "reason": exc.__class__.__name__}
    except Exception:                                            # noqa: BLE001
        created = None

    try:
        memory = _memory_reading(process, psutil_module)
        cpu_seconds, cpu_error = _cpu_reading(process, psutil_module)
    except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess) as exc:
        return None, {
            "pid": pid,
            "create_time": created,
            "identity": _process_key(pid, created),
            "reason": exc.__class__.__name__,
        }

    try:
        name = str(process.name())
    except Exception:                                            # noqa: BLE001
        name = ""
    row: Dict[str, Any] = {
        "identity": _process_key(pid, created),
        "pid": pid,
        "create_time": created,
        "relation": "root" if pid == root_pid else "child",
        "name": name,
        **memory,
        "cpu_seconds": cpu_seconds,
        "cpu_available": cpu_seconds is not None,
        "cpu_unavailable_reason": cpu_error,
    }
    worker = _label_for_process(labels, pid, created)
    if worker is not None:
        row["worker"] = worker
    if detailed:
        thread_names = _python_thread_names() if pid == os.getpid() else None
        try:
            row.update(_thread_reading(process, psutil_module, thread_names))
        except (psutil_module.NoSuchProcess, psutil_module.ZombieProcess) as exc:
            row.update({
                "thread_cpu_available": False,
                "thread_cpu_unavailable_reason": exc.__class__.__name__,
                "threads": None,
            })
    return row, None


def _tree_measure(measures: Mapping[str, int]) -> Optional[str]:
    used = [name for name, count in measures.items() if count]
    if not used:
        return None
    if len(used) == 1:
        return used[0]
    return "mixed"


def _process_tree_snapshot(
        root_pid: Optional[int] = None,
        *,
        detailed: bool = False,
        labels: Optional[
            Mapping[Tuple[int, Optional[float]], Mapping[str, Any]]
        ] = None,
        process_factory: Any = None) -> Dict[str, Any]:
    """Take one race-safe process-tree sample.

    The tree total prefers PSS (proportional shared memory), then USS, and
    falls back to RSS only where neither richer measure is exposed.  A mixed
    tree says so explicitly instead of presenting incomparable bytes under a
    single unnamed definition.
    """
    import psutil

    pid = os.getpid() if root_pid is None else int(root_pid)
    factory = psutil.Process if process_factory is None else process_factory
    root = factory(pid)
    unavailable: List[Dict[str, Any]] = []
    try:
        children = list(root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        children = []
        unavailable.append({"pid": pid, "reason": exc.__class__.__name__,
                            "operation": "children"})
    except Exception as exc:                                     # noqa: BLE001
        children = []
        unavailable.append({"pid": pid, "reason": exc.__class__.__name__,
                            "operation": "children"})

    processes: Sequence[Any] = [root, *children]
    process_rows: List[Dict[str, Any]] = []
    seen_pids: Set[int] = set()
    label_map = labels or {}
    for process in processes:
        try:
            process_pid = int(process.pid)
        except Exception as exc:                                 # noqa: BLE001
            unavailable.append({"pid": None, "reason": exc.__class__.__name__})
            continue
        if process_pid in seen_pids:
            continue
        seen_pids.add(process_pid)
        row, missing = _read_process(
            process,
            root_pid=pid,
            detailed=detailed,
            labels=label_map,
            psutil_module=psutil,
        )
        if row is not None:
            process_rows.append(row)
        if missing is not None:
            unavailable.append(missing)

    process_rows.sort(key=lambda item: (item["relation"] != "root", item["pid"]))
    memory_bytes = [
        int(row["memory_bytes"])
        for row in process_rows
        if row.get("memory_bytes") is not None
    ]
    cpu_seconds = [
        float(row["cpu_seconds"])
        for row in process_rows
        if row.get("cpu_seconds") is not None
    ]
    measure_counts = Counter(
        str(row["memory_measure"])
        for row in process_rows
        if row.get("memory_measure") is not None
    )
    return {
        "utc": _utc_now(),
        "monotonic_ns": time.monotonic_ns(),
        "root_pid": pid,
        "tree_memory_bytes": sum(memory_bytes) if memory_bytes else None,
        "tree_memory_measure": _tree_measure(measure_counts),
        "memory_measure_counts": dict(sorted(measure_counts.items())),
        "tree_cpu_seconds": sum(cpu_seconds) if cpu_seconds else None,
        "process_count": len(process_rows),
        "processes": process_rows,
        "unavailable_processes": unavailable,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON document atomically and durably on local filesystems."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_descriptor = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


class _ResourceSampler:
    """Bounded, daemon-backed process-tree accounting for one run."""

    def __init__(
            self,
            output: Any,
            *,
            mode: Any = None,
            environ: Optional[Mapping[str, str]] = None,
            interval_seconds: float = _DEFAULT_SAMPLE_INTERVAL_SECONDS,
            sample_limit: int = _DEFAULT_SAMPLE_LIMIT,
            root_pid: Optional[int] = None,
            checkpoint_samples: int = 5) -> None:
        """Set up the sampler and the bounds that keep it bounded.

        :param output: where the accounting is written, or ``None`` to keep
            it in memory only.
        :param mode: the performance mode to sample under; resolved against
            ``environ`` when not given.
        :param environ: environment to read the mode from, defaulting to the
            process's own.
        :param interval_seconds: seconds between samples, floored at 0.001 so
            a zero cannot turn the loop into a spin.
        :param sample_limit: how many samples and events to keep. THIS IS THE
            BOUND IN "bounded": both deques carry it as ``maxlen``, so a long
            run drops its oldest samples rather than growing without limit,
            and the number dropped is counted rather than hidden.
        :param root_pid: the process whose tree is accounted for, defaulting
            to this one.
        :param checkpoint_samples: how many samples between writes to
            ``output``, floored at one.
        """
        selection = _select_performance_mode(mode, environ)
        self.mode = selection.mode
        self.output = Path(output) if output is not None else None
        self.interval_seconds = max(0.001, float(interval_seconds))
        self.sample_limit = max(1, int(sample_limit))
        self.root_pid = os.getpid() if root_pid is None else int(root_pid)
        self.checkpoint_samples = max(1, int(checkpoint_samples))
        self._selection = selection
        self._samples: Deque[Dict[str, Any]] = deque(maxlen=self.sample_limit)
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.sample_limit)
        self._labels: MutableMapping[
            Tuple[int, Optional[float]], Dict[str, Any]
        ] = {}
        self._seen_children: Dict[str, Dict[str, Any]] = {}
        self._samples_dropped = 0
        self._state_lock = threading.RLock()
        self._persist_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_utc: Optional[str] = None
        self._stopped_utc: Optional[str] = None
        self._stop_reason = ""
        self._write_error = ""
        self._summary: Dict[str, Any] = {
            "samples_recorded": 0,
            "last_tree_memory_bytes": None,
            "peak_tree_memory_bytes": None,
            "peak_tree_memory_utc": None,
            "last_tree_cpu_seconds": None,
            "peak_process_count": 0,
            "unavailable_process_reads": 0,
            "memory_measure_sample_counts": {
                name: 0 for name in (*_MEMORY_MEASURE_ORDER, "mixed")
            },
        }

    def _register_worker(self, stamp: Mapping[str, Any]) -> str:
        """Give a child process a stable identity, and return it.

        IDENTITY IS PID PLUS CREATE TIME, not the pid. A pid is reused, and on a
        long run it will be: a sampler keyed on the pid alone attributes a new
        worker's memory to the one that exited.
        """
        pid = int(stamp["pid"])
        raw_created = stamp.get("create_time")
        created = float(raw_created) if raw_created is not None else None
        row = {
            "pid": pid,
            "create_time": created,
            "worker_kind": str(stamp.get("worker_kind", "worker")),
            "worker_id": str(stamp.get("worker_id", pid)),
        }
        with self._state_lock:
            self._labels[(pid, created)] = row
            identity = _process_key(pid, created)
            worker = {
                "kind": row["worker_kind"],
                "id": row["worker_id"],
            }
            if identity in self._seen_children:
                self._seen_children[identity]["worker"] = worker
            else:
                self._seen_children[identity] = {
                    "identity": identity,
                    "pid": pid,
                    "create_time": created,
                    "worker": worker,
                }
        return identity

    def _update_summary(self, sample: Mapping[str, Any]) -> None:
        """Fold one sample into the running totals.

        Kept incrementally rather than recomputed from the samples, so the summary
        costs the same whether the run lasted a minute or a day.
        """
        summary = self._summary
        summary["samples_recorded"] += 1
        memory = sample.get("tree_memory_bytes")
        summary["last_tree_memory_bytes"] = memory
        if memory is not None and (
            summary["peak_tree_memory_bytes"] is None
            or int(memory) > int(summary["peak_tree_memory_bytes"])
        ):
            summary["peak_tree_memory_bytes"] = int(memory)
            summary["peak_tree_memory_utc"] = sample.get("utc")
        summary["last_tree_cpu_seconds"] = sample.get("tree_cpu_seconds")
        summary["peak_process_count"] = max(
            int(summary["peak_process_count"]), int(sample.get("process_count", 0))
        )
        summary["unavailable_process_reads"] += len(
            sample.get("unavailable_processes") or []
        )
        measure = sample.get("tree_memory_measure")
        if measure in summary["memory_measure_sample_counts"]:
            summary["memory_measure_sample_counts"][measure] += 1

    def _disappearance_events(
            self, sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """The children that have gone since the last sample.

        A child whose create time was never read is NOT reported as gone when its
        pid is still present -- that is the same process seen again rather than
        one that exited and another that started, and reporting it would put a
        spurious death in the log on every sample.
        """
        current = {
            str(row["identity"]): dict(row)
            for row in sample.get("processes") or []
            if row.get("relation") == "child"
        }
        current_by_pid = {
            int(row["pid"]): row
            for row in current.values()
            if row.get("pid") is not None
        }
        missing = sorted(set(self._seen_children) - set(current))
        events = []
        for identity in missing:
            old = self._seen_children[identity]
            old_pid = old.get("pid")
            if old.get("create_time") is None and old_pid in current_by_pid:
                worker = old.get("worker")
                if worker is not None:
                    current_by_pid[int(old_pid)]["worker"] = worker
                continue
            event: Dict[str, Any] = {
                "kind": "process_disappeared",
                "utc": sample.get("utc"),
                "identity": identity,
                "pid": old_pid,
                "exit_status": None,
                "exit_status_available": False,
            }
            if old.get("worker") is not None:
                event["worker"] = old["worker"]
            created = old.get("create_time")
            if old_pid is not None:
                self._labels.pop((int(old_pid), created), None)
                self._labels.pop((int(old_pid), None), None)
            events.append(event)
        self._seen_children = current
        return events

    def _sample_once(self, *, force_persist: bool = False) -> Dict[str, Any]:
        """Take one snapshot of the process tree and record what it shows."""
        with self._state_lock:
            labels = dict(self._labels)
        sample = _process_tree_snapshot(
            self.root_pid,
            detailed=self.mode == "detailed",
            labels=labels,
        )
        with self._state_lock:
            sequence = int(self._summary["samples_recorded"])
            sample["sequence"] = sequence
            events = self._disappearance_events(sample)
            if events:
                sample["events"] = events
                self._events.extend(events)
            self._update_summary(sample)
            if self.mode == "detailed":
                if len(self._samples) == self.sample_limit:
                    self._samples_dropped += 1
                self._samples.append(sample)
            should_persist = (
                force_persist
                or bool(events)
                or sequence == 0
                or (sequence + 1) % self.checkpoint_samples == 0
            )
        if should_persist:
            self._persist()
        return sample

    def _record_sampler_error(self, exc: BaseException) -> None:
        """Log a sampler failure INTO THE SAMPLES rather than raising.

        The sampler runs beside the work, not over it: a failure to measure must
        not end the run being measured, and a silent failure would leave a gap
        nobody could tell from an idle period.
        """
        event = {
            "kind": "sampler_error",
            "utc": _utc_now(),
            "error": exc.__class__.__name__,
        }
        with self._state_lock:
            self._events.append(event)
        self._persist()

    def _run(self) -> None:
        """Sample on an interval until stopped.

        The wait is the interval MINUS the time the sample took, so the cadence
        holds rather than drifting by the cost of each sample; and it waits on the
        stop event, so stopping is immediate rather than one interval away.
        """
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self._sample_once()
            except Exception as exc:                              # noqa: BLE001
                self._record_sampler_error(exc)
            elapsed = time.monotonic() - started
            if self._stop_event.wait(max(0.0, self.interval_seconds - elapsed)):
                break

    def _start(self) -> "_ResourceSampler":
        """Begin sampling, unless the mode is off.

        An output path is required rather than defaulted: a performance log with
        nowhere to go is a run that measured itself and threw the answer away.
        """
        if self.mode == "off":
            return self
        if self.output is None:
            raise ValueError("performance logging needs an output JSON path")
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._started_utc = self._started_utc or _utc_now()
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"spacr-resource-sampler-{self.root_pid}",
                daemon=True,
            )
            self._thread.start()
        return self

    def _stop(self, reason: str = "stopped") -> Optional[Path]:
        if self.mode == "off":
            return None
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(2.0, self.interval_seconds * 2.0))
        with self._state_lock:
            still_running = thread is not None and thread.is_alive()
        if not still_running:
            try:
                self._sample_once(force_persist=False)
            except Exception as exc:                              # noqa: BLE001
                self._record_sampler_error(exc)
        with self._state_lock:
            self._stopped_utc = _utc_now()
            self._stop_reason = str(reason)
            if still_running:
                self._events.append({
                    "kind": "sampler_stop_timeout",
                    "utc": self._stopped_utc,
                })
        self._persist()
        return self.output

    def _document(self) -> Dict[str, Any]:
        with self._state_lock:
            payload: Dict[str, Any] = {
                "schema_version": _PERFORMANCE_SCHEMA_VERSION,
                "mode": self.mode,
                "configuration": {
                    "source": self._selection.source,
                    "requested": self._selection.requested,
                    "warning": self._selection.warning,
                    "sample_interval_seconds": self.interval_seconds,
                    "sample_limit": self.sample_limit,
                    "memory_measure_preference": list(_MEMORY_MEASURE_ORDER),
                    "profile_hook_installed": False,
                },
                "root_pid": self.root_pid,
                "started_utc": self._started_utc,
                "stopped_utc": self._stopped_utc,
                "stop_reason": self._stop_reason,
                "summary": json.loads(json.dumps(self._summary)),
                "events": list(self._events),
                "samples_dropped": self._samples_dropped,
                "write_error": self._write_error,
            }
            if self.mode == "detailed":
                payload["samples"] = list(self._samples)
            return payload

    def _persist(self) -> bool:
        if self.mode == "off" or self.output is None:
            return False
        with self._persist_lock:
            try:
                _atomic_json(self.output, self._document())
            except Exception as exc:                              # noqa: BLE001
                with self._state_lock:
                    self._write_error = exc.__class__.__name__
                return False
        with self._state_lock:
            self._write_error = ""
        return True

    def __enter__(self) -> "_ResourceSampler":
        return self._start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        reason = "failed" if exc_type is not None else "completed"
        self._stop(reason)


def _tree_stage_reading() -> Dict[str, Any]:
    """Return the light process-tree fields recorded at existing fit stages."""
    try:
        sample = _process_tree_snapshot(detailed=False)
    except Exception:                                            # noqa: BLE001
        return {
            "tree_memory_bytes": None,
            "tree_memory_measure": None,
            "tree_process_count": None,
        }
    return {
        "tree_memory_bytes": sample.get("tree_memory_bytes"),
        "tree_memory_measure": sample.get("tree_memory_measure"),
        "tree_process_count": sample.get("process_count"),
    }


def host_rss() -> Optional[int]:
    """Resident bytes for this process, or ``None`` when unknowable.

    `/proc/self/statm` first because it needs no dependency and no import;
    psutil second. A container that reports neither gets ``None``, which the
    caller must not spell as zero -- "nothing was using memory" and "nobody
    measured" are opposite findings.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except Exception:                                            # noqa: BLE001
        pass
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:                                            # noqa: BLE001
        return None


def gpu_allocated() -> Optional[int]:
    """The HIGH-WATER mark of torch's CUDA allocation, or ``None``.

    ASKED ONLY IF TORCH IS ALREADY IMPORTED. Importing it to take a
    measurement would make the measurement the most expensive thing in the
    stage, and on a settings panel it is the import this project has twice
    had to keep out (`tests/test_a_settings_panel_does_not_import_torch.py`).

    Uses ``max_memory_allocated`` rather than the current allocation because
    fit tensors may already be released when a stage boundary is recorded.
    The high-water mark is cumulative across the process and therefore reports
    the largest allocation reached across a sequence of fits.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return int(max(torch.cuda.memory_allocated(),
                       torch.cuda.max_memory_allocated()))
    except Exception:                                            # noqa: BLE001
        return None


def readable(total: Optional[int]) -> str:
    """Bytes as the unit a person decides in, or "not measured".

    :param total: byte count to format, or None when no measurement exists.
    """
    if total is None:
        return "not measured"
    size = float(max(0, int(total)))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def record_stage(settings: Any, name: str) -> Dict[str, Any]:
    """Record the current fit stage and its memory use.

    Updates the stage and resource-history entries in ``settings`` when it is
    mutable. Measurement and storage failures are ignored so diagnostics do
    not interrupt the fit.

    :param settings: Mutable fit settings or another mapping-like object.
    :param name: Name of the stage being entered.
    :returns: Dictionary containing the stage, resident memory, and allocated
        GPU memory. Unavailable measurements are ``None``.
    """
    reading = {
        "stage": str(name),
        "rss": host_rss(),
        "gpu": gpu_allocated(),
        **_tree_stage_reading(),
    }
    try:
        settings[STAGE_KEY] = str(name)
        settings.setdefault(RESOURCE_KEY, []).append(reading)
    except Exception:                                            # noqa: BLE001
        pass
    return reading


def peak(settings: Any) -> Dict[str, Any]:
    """The largest reading recorded, and where it was taken.

    :param settings: mapping-like fit settings carrying the recorded resource
        history.

    Empty when nothing was recorded -- NOT zero, for the reason `host_rss`
    gives.
    """
    try:
        readings: List[Mapping[str, Any]] = list(
            settings.get(RESOURCE_KEY) or [])
    except Exception:                                            # noqa: BLE001
        return {}
    out: Dict[str, Any] = {}
    for key in ("rss", "gpu"):
        seen = [r for r in readings if r.get(key) is not None]
        if not seen:
            continue
        worst = max(seen, key=lambda r: r[key])
        out[key] = worst[key]
        out[f"{key}_stage"] = worst.get("stage", "")
    return out


def describe_resources(settings: Any) -> str:
    """The per-stage table, for a summary or a failure report. "" when empty.

    :param settings: mapping-like fit settings carrying the recorded resource
        history.
    """
    try:
        readings = list(settings.get(RESOURCE_KEY) or [])
    except Exception:                                            # noqa: BLE001
        return ""
    if not readings:
        return ""
    lines = [f"  {'stage':<34} {'resident':>12} {'GPU':>12}"]
    for reading in readings:
        lines.append(
            f"  {str(reading.get('stage', ''))[:34]:<34} "
            f"{readable(reading.get('rss')):>12} "
            f"{readable(reading.get('gpu')):>12}")
    high = peak(settings)
    if "rss" in high:
        lines.append(f"  PEAK resident {readable(high['rss'])} at "
                     f"{high.get('rss_stage', '')!r}")
    if "gpu" in high:
        lines.append(f"  PEAK GPU      {readable(high['gpu'])} at "
                     f"{high.get('gpu_stage', '')!r}")
    return "\n".join(lines)
