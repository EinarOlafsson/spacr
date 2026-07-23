"""
Plate queue — sequential execution of many pipelines.

Users often have 5–20 plates to segment, measure, or classify with the
same settings. Running each one manually from the Mask app is fine
for one plate and painful for twenty. This module lets them:

1. Enqueue a plate as ``(app_key, settings)`` (or import a batch of
   plates from a CSV).
2. Run the queue in the background — one item at a time — and see
   per-item status update live.
3. Pause between items, or stop cold.
4. Have every completed item show up in the run-journal history like
   a normal invocation, so nothing about downstream tooling changes.

The queue itself is a plain Python data structure — the Qt screen in
:mod:`spacr.qt.screens.queue` renders it. Keeping the logic separate
makes it unit-testable without a display.

Persistence: the queue serialises to ``~/.spacr/queue.json`` on every
mutation so a crash or restart doesn't lose the plan.
"""
from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger("spacr.qt.plate_queue")


def _queue_path() -> Path:
    """Return the on-disk queue file — creating the parent if needed."""
    p = Path.home() / ".spacr" / "queue.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Status(str, Enum):
    """Per-item lifecycle. Mirrors the run journal's terminology."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QueueItem:
    """One plate to process."""
    id:       str
    app_key:  str
    settings: Dict[str, Any]
    status:   Status = Status.QUEUED
    start_ts: Optional[float] = None
    end_ts:   Optional[float] = None
    error:    Optional[str] = None
    run_dir:  Optional[str] = None
    label:    str = ""

    @classmethod
    def build(cls, app_key: str, settings: Dict[str, Any],
                 label: str = "") -> "QueueItem":
        """Factory that mints an ID + resolves a display label."""
        item_id = uuid.uuid4().hex[:8]
        if not label:
            label = str(settings.get("src") or f"plate-{item_id}")
        return cls(id=item_id, app_key=app_key,
                     settings=dict(settings), label=label)

    @property
    def elapsed_s(self) -> Optional[float]:
        """Wall-clock seconds if the item has both timestamps."""
        if self.start_ts is None:
            return None
        end = self.end_ts if self.end_ts is not None else time.time()
        return end - self.start_ts


# ---------------------------------------------------------------------------
# Queue container
# ---------------------------------------------------------------------------

class PlateQueue:
    """Ordered list of :class:`QueueItem`s with atomic on-disk snapshots.

    The queue is thread-agnostic — the Qt screen owns exclusive
    access. If two callers ever need to touch it concurrently, wrap
    each mutation in a lock at the call site.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = path or _queue_path()
        self._items: List[QueueItem] = []
        self.load()

    # -- accessors ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def items(self) -> List[QueueItem]:
        """Return a shallow copy of the current items list."""
        return list(self._items)

    def find(self, item_id: str) -> Optional[QueueItem]:
        """Return the item with ``item_id`` or None."""
        return next((i for i in self._items if i.id == item_id), None)

    def next_queued(self) -> Optional[QueueItem]:
        """Return the first :attr:`Status.QUEUED` item, or None."""
        return next((i for i in self._items if i.status == Status.QUEUED),
                     None)

    def is_all_done(self) -> bool:
        """True iff no item is in QUEUED or RUNNING."""
        return not any(i.status in (Status.QUEUED, Status.RUNNING)
                        for i in self._items)

    # -- mutations ---------------------------------------------------------

    def add(self, item: QueueItem) -> None:
        self._items.append(item)
        self.save()

    def remove(self, item_id: str) -> bool:
        before = len(self._items)
        self._items = [i for i in self._items if i.id != item_id]
        changed = len(self._items) != before
        if changed:
            self.save()
        return changed

    def clear_finished(self) -> int:
        """Remove SUCCESS/FAILED/SKIPPED items; return count removed."""
        before = len(self._items)
        keep = {Status.QUEUED, Status.RUNNING}
        self._items = [i for i in self._items if i.status in keep]
        removed = before - len(self._items)
        if removed:
            self.save()
        return removed

    def update(self, item_id: str, **fields) -> None:
        """Patch fields on the item with ``item_id``. Saves on any change."""
        item = self.find(item_id)
        if item is None:
            return
        changed = False
        for k, v in fields.items():
            if hasattr(item, k) and getattr(item, k) != v:
                setattr(item, k, v)
                changed = True
        if changed:
            self.save()

    # -- persistence -------------------------------------------------------

    def save(self) -> None:
        try:
            payload = {"items": [self._serialise(i) for i in self._items]}
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception as e:
            LOG.warning("failed to persist queue: %s", e)

    def load(self) -> None:
        if not self._path.exists():
            self._items = []
            return
        try:
            payload = json.loads(self._path.read_text())
        except Exception as e:
            LOG.warning("queue file unreadable, starting empty: %s", e)
            self._items = []
            return
        raw = payload.get("items", []) if isinstance(payload, dict) else []
        self._items = []
        for entry in raw:
            try:
                self._items.append(self._deserialise(entry))
            except Exception as e:
                LOG.info("skipping malformed queue entry: %s", e)

    @staticmethod
    def _serialise(item: QueueItem) -> Dict[str, Any]:
        d = asdict(item)
        d["status"] = item.status.value
        return d

    @staticmethod
    def _deserialise(d: Dict[str, Any]) -> QueueItem:
        status = Status(d.get("status", "queued"))
        return QueueItem(
            id=str(d["id"]),
            app_key=str(d["app_key"]),
            settings=dict(d.get("settings", {})),
            status=status,
            start_ts=d.get("start_ts"),
            end_ts=d.get("end_ts"),
            error=d.get("error"),
            run_dir=d.get("run_dir"),
            label=str(d.get("label", "")),
        )


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------

def import_plates_from_csv(csv_path: Any,
                              base_settings: Dict[str, Any],
                              app_key: str = "mask") -> List[QueueItem]:
    """Parse a CSV of plates into :class:`QueueItem`s.

    The CSV must have a header row. Each remaining row is one plate.
    Columns other than ``src`` are merged over ``base_settings``;
    ``src`` becomes the item's src (and label). Rows missing ``src``
    are skipped.

    :param csv_path: path to a CSV with at least a ``src`` column.
    :param base_settings: settings dict applied to every row before
        the row's own overrides.
    :param app_key: pipeline id for every generated item.
    """
    import csv
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    items: List[QueueItem] = []
    with csv_path.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            src = (row.get("src") or "").strip()
            if not src:
                continue
            settings = copy.deepcopy(base_settings)
            settings["src"] = src
            for k, v in row.items():
                if k in (None, "", "src"):
                    continue
                # Try numeric coercion; fall back to raw string.
                vv: Any = v
                if v is not None:
                    try:
                        vv = int(v)
                    except ValueError:
                        try:
                            vv = float(v)
                        except ValueError:
                            # Preserve booleans + None-ish tokens
                            if v.lower() in ("true", "yes"):
                                vv = True
                            elif v.lower() in ("false", "no"):
                                vv = False
                            elif v.lower() in ("", "none", "null"):
                                vv = None
                settings[k] = vv
            items.append(QueueItem.build(app_key, settings, label=src))
    return items


# ---------------------------------------------------------------------------
# Runner — pure Python, injectable for testing
# ---------------------------------------------------------------------------

RunnerFn = Callable[[QueueItem], None]


def default_runner(item: QueueItem) -> None:
    """Execute ``item`` synchronously via the resolved pipeline entry
    point. Intended for CLI use or tests — the Qt screen uses a
    QThread wrapper instead so the UI stays responsive."""
    from .bridge import resolve_pipeline_entry
    fn = resolve_pipeline_entry(item.app_key)
    if fn is None:
        raise RuntimeError(f"no pipeline for app_key={item.app_key!r}")
    fn(item.settings)


def run_queue(queue: PlateQueue,
                 runner: RunnerFn = default_runner,
                 stop_on_error: bool = False) -> None:
    """Run every QUEUED item in ``queue`` sequentially.

    Each item's status transitions QUEUED → RUNNING → SUCCESS/FAILED.
    If ``stop_on_error`` is True, the first failure halts the loop
    with remaining items left as QUEUED.

    Not called by the Qt screen (which needs threads + signals) but
    exposed as a plain function for CLI / tests / scripting.
    """
    while True:
        item = queue.next_queued()
        if item is None:
            return
        queue.update(item.id, status=Status.RUNNING, start_ts=time.time())
        try:
            runner(item)
        except Exception as e:
            queue.update(item.id, status=Status.FAILED,
                            end_ts=time.time(), error=str(e))
            LOG.warning("queue item %s failed: %s", item.id, e)
            if stop_on_error:
                return
            continue
        queue.update(item.id, status=Status.SUCCESS, end_ts=time.time())
