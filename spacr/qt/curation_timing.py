"""Per-field timing log for a timed curation session.

One row is appended to ``<queue folder>/curate_timing.csv`` each time a field
is saved or skipped in Make Masks. Nothing here may block or crash curation:
:func:`log_visit` swallows and logs every failure.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

LOG = logging.getLogger(__name__)

TIMING_FILE = "curate_timing.csv"
SESSION_TAG_ENV = "SPACR_CURATION_SESSION_TAG"
#: Gaps between activity marks longer than this are idle and not counted.
IDLE_GAP_SECONDS = 60.0
MATCH_IOU = 0.5
MODIFIED_IOU = 0.9

COLUMNS = (
    "stem", "field_path", "session_tag", "decision", "t_open", "t_close",
    "wall_seconds", "active_seconds", "n_edits", "n_objects_premask",
    "n_objects_final", "n_added", "n_removed", "n_modified",
)


def count_object_changes(before, after) -> Tuple[int, int, int, int, int]:
    """Compare two label arrays by IoU matching.

    Objects are paired greedily by descending IoU; a pair needs IoU >= 0.5,
    and is *modified* when IoU < 0.9. Unpaired ``after`` objects are added,
    unpaired ``before`` objects are removed.

    :returns: ``(n_before, n_after, added, removed, modified)``.
    """
    a = np.asarray(before).ravel()
    b = np.asarray(after).ravel()
    if a.shape != b.shape:
        na = int(np.count_nonzero(np.unique(a)))
        nb = int(np.count_nonzero(np.unique(b)))
        return na, nb, nb, na, 0
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    za = 0 if ua.size and ua[0] == 0 else -1     # index of background
    zb = 0 if ub.size and ub[0] == 0 else -1
    na = ua.size - (1 if za == 0 else 0)
    nb = ub.size - (1 if zb == 0 else 0)
    if na == 0 or nb == 0:
        return na, nb, nb, na, 0
    area_a = np.bincount(ia, minlength=ua.size)
    area_b = np.bincount(ib, minlength=ub.size)
    fg = np.ones(ia.shape, bool)
    if za == 0:
        fg &= ia > 0
    if zb == 0:
        fg &= ib > 0
    pair = ia[fg].astype(np.int64) * ub.size + ib[fg]
    codes, inter = np.unique(pair, return_counts=True)
    pa, pb = codes // ub.size, codes % ub.size
    union = area_a[pa] + area_b[pb] - inter
    iou = inter / np.maximum(union, 1)
    order = np.argsort(-iou, kind="stable")
    used_a, used_b = set(), set()
    matched = []
    for k in order:
        if iou[k] < MATCH_IOU:
            break
        i, j = int(pa[k]), int(pb[k])
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append(float(iou[k]))
    n_match = len(matched)
    modified = sum(1 for v in matched if v < MODIFIED_IOU)
    return na, nb, nb - n_match, na - n_match, modified


class FieldTimer:
    """Wall and active time for one field on screen."""

    def __init__(self, path: str, premask, n_log_open: int = 0):
        self.path = path
        self.premask = None if premask is None else np.array(premask, copy=True)
        self.n_log_open = int(n_log_open)
        self.wall_open = time.time()
        self.mono_open = time.monotonic()
        self._last = self.mono_open
        self._active = 0.0

    def touch(self, now: Optional[float] = None) -> None:
        """Mark user activity; gaps over the idle limit are not counted."""
        now = time.monotonic() if now is None else now
        gap = now - self._last
        if 0 <= gap <= IDLE_GAP_SECONDS:
            self._active += gap
        self._last = now

    def active_seconds(self, now: Optional[float] = None) -> float:
        self.touch(now)
        return self._active


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def build_row(timer: FieldTimer, decision: str, final_mask, n_edits_total: int,
              session_tag: Optional[str] = None) -> Dict[str, object]:
    """The CSV row for one finished visit."""
    now_mono = time.monotonic()
    wall_close = time.time()
    if session_tag is None:
        session_tag = os.environ.get(SESSION_TAG_ENV, "")
    pre = timer.premask
    fin = final_mask if final_mask is not None else pre
    if pre is None:
        pre = fin
    if pre is not None and fin is not None:
        n_pre, n_fin, added, removed, modified = count_object_changes(pre, fin)
    else:
        n_pre = n_fin = added = removed = modified = ""
    path = timer.path
    stem = os.path.splitext(os.path.basename(path))[0]
    return {
        "stem": stem, "field_path": path, "session_tag": session_tag,
        "decision": decision, "t_open": _iso(timer.wall_open),
        "t_close": _iso(wall_close),
        "wall_seconds": round(now_mono - timer.mono_open, 3),
        "active_seconds": round(timer.active_seconds(now_mono), 3),
        "n_edits": max(0, int(n_edits_total) - timer.n_log_open),
        "n_objects_premask": n_pre, "n_objects_final": n_fin,
        "n_added": added, "n_removed": removed, "n_modified": modified,
    }


def append_row(folder, row: Dict[str, object]) -> str:
    """Append ``row`` to the folder's timing CSV, writing the header if new."""
    path = os.path.join(os.fspath(folder), TIMING_FILE)
    new = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    return path


def log_visit(folder, timer: Optional[FieldTimer], decision: str, final_mask,
              n_edits_total: int) -> Optional[str]:
    """Build and append one row; never raises."""
    if timer is None or folder is None:
        return None
    try:
        return append_row(folder, build_row(timer, decision, final_mask, n_edits_total))
    except Exception:                                        # noqa: BLE001
        LOG.warning("could not write the curation timing row", exc_info=True)
        return None
