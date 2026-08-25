"""Hand a frame to the next step without a file parse in between.

A step that has built a frame IN MEMORY and hands it to a step in the same
process has no need of a serialisation round trip. Writing one out and
parsing it back costs twice: once to write and once to read, and both costs
scale with the frame rather than with the work being done.

The merged measurement frame is the case this was built for. A four-plate
screen merges to about 2.75 GB; writing that as CSV takes around 160 seconds
and parsing it back takes longer still, all of it AFTER the frame already
existed in the writing process's memory, and all of it before the fit that
needs it can start.

Two halves, and both are needed:

``hold``/``held``
    the frame itself is offered under the path it was written to, so a reader
    that would have parsed that path gets the object instead. The reference
    is WEAK -- when the producer drops the frame the offer disappears and the
    file on disk is read as before, so nothing here can keep a multi-gigabyte
    frame alive after its owner is done with it.
``stage``
    the durable copy is written in a COLUMNAR format when one is available.
    The artefact is kept because a user can open it and because every fit of
    a queue then reads the same numbers; only the format changes, and Parquet
    both writes and reads several times faster than CSV at a fraction of the
    size.

A reader that knows nothing about this module keeps working: the path is a
real path to a real file, and the handoff is an optimisation on top of it.
"""
from __future__ import annotations

import os
import weakref
from typing import Any, Dict, Optional, Tuple

import pandas as pd

__all__ = [
    "COLUMNAR_SUFFIX",
    "describe",
    "held",
    "hold",
    "key_for",
    "release",
    "stage",
]

#: The columnar suffix :func:`stage` prefers. ``.parquet`` rather than
#: ``.feather`` because pandas writes it with either engine and it is the
#: format the rest of spaCR already reads.
COLUMNAR_SUFFIX = ".parquet"

#: Offered frames, keyed by the absolute path whose contents they are.
#:
#: WEAK VALUES. A strong reference here would make every merged frame in a
#: session immortal, which on this data is gigabytes per run held by a module
#: nobody remembers to clear. When the producer lets go, the entry vanishes
#: and the reader falls back to the file.
_OFFERED: "weakref.WeakValueDictionary[str, pd.DataFrame]" = \
    weakref.WeakValueDictionary()

#: What each offered frame was when it was offered: ``(rows, columns)``.
#: Kept separately because the reading is wanted for the log line even in the
#: run where the frame itself has already been collected.
_SHAPES: Dict[str, Tuple[int, int]] = {}


def key_for(path: Any) -> str:
    """The absolute path two callers must agree on to meet here.

    :param path: a path, or anything :func:`os.fspath` accepts.
    :returns: the expanded, absolute path used as the offer's key.
    """
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def hold(path: Any, frame: pd.DataFrame) -> str:
    """Offer ``frame`` as the contents of ``path``.

    :param path: where the frame was (or will be) written.
    :param frame: the frame the caller already has.
    :returns: the key the offer is filed under.

    The caller keeps ownership. Nothing here extends the frame's life, so a
    producer that finishes and drops it withdraws the offer by doing so.
    """
    if frame is None:
        raise ValueError("there is no frame to hand over")
    identity = key_for(path)
    _OFFERED[identity] = frame
    _SHAPES[identity] = (int(len(frame)), int(len(frame.columns)))
    return identity


def held(path: Any) -> Optional[pd.DataFrame]:
    """The frame offered for ``path``, or ``None`` when there is none.

    ``None`` is not a failure: it means the reader should read the file, which
    is what it would have done anyway.
    """
    try:
        return _OFFERED.get(key_for(path))
    except TypeError:
        # A caller that passed something with no path at all asked nothing of
        # this module; it gets the same answer as a path nobody offered.
        return None


def describe(path: Any) -> str:
    """One line naming what was handed over for ``path``, for a log.

    Empty when nothing was offered, so a caller can print it unconditionally
    and say nothing when there is nothing to say.
    """
    identity = key_for(path)
    shape = _SHAPES.get(identity)
    if shape is None or identity not in _OFFERED:
        return ""
    rows, columns = shape
    return (f"{os.path.basename(identity)}: {rows:,} rows x {columns} columns "
            f"handed over in memory, not parsed")


def release(path: Any = None) -> int:
    """Withdraw one offer, or every offer when ``path`` is ``None``.

    :returns: how many offers were withdrawn.

    Withdrawing is optional -- the references are weak -- but a producer that
    knows it has finished can say so, which makes the fallback to the file
    deterministic instead of dependent on when the frame is collected.
    """
    if path is None:
        count = len(_OFFERED)
        _OFFERED.clear()
        _SHAPES.clear()
        return count
    identity = key_for(path)
    _SHAPES.pop(identity, None)
    if identity in _OFFERED:
        del _OFFERED[identity]
        return 1
    return 0


def _columnar_engine() -> Optional[str]:
    """The Parquet engine pandas can use here, or ``None`` for neither."""
    for name in ("pyarrow", "fastparquet"):
        try:
            __import__(name)
            return name
        except Exception:                                        # noqa: BLE001
            continue
    return None


def stage(frame: pd.DataFrame, folder: Any, stem: str, *,
          columnar: bool = True, report=print) -> str:
    """Write the durable copy and offer the frame in memory under its path.

    :param frame: the frame to hand on.
    :param folder: directory for the artefact; created when absent.
    :param stem: the file name without a suffix.
    :param columnar: write Parquet when an engine is installed. ``False``
        forces CSV, which is what a caller wants when the file is meant to be
        opened in a spreadsheet rather than read back by spaCR.
    :param report: called with one line saying what was written and how long
        it took; ``None`` to say nothing.
    :returns: the path written.

    The frame is offered BEFORE the write returns, so a reader that starts
    while the artefact is still being written gets the object rather than a
    half-written file.
    """
    import time

    from . import tabular

    if frame is None:
        raise ValueError("there is no frame to stage")
    folder = os.path.abspath(os.path.expanduser(os.fspath(folder)))
    os.makedirs(folder, exist_ok=True)
    engine = _columnar_engine() if columnar else None
    suffix = COLUMNAR_SUFFIX if engine else ".csv"
    path = os.path.join(folder, f"{stem}{suffix}")

    hold(path, frame)
    started = time.time()
    tabular.write_table(frame, path)
    if report is not None:
        size = os.path.getsize(path) if os.path.exists(path) else 0
        report(f"Wrote {os.path.basename(path)}: {len(frame):,} rows, "
               f"{size / 1e6:.1f} MB in {time.time() - started:.1f} s"
               + ("" if engine else
                  " (no Parquet engine installed, so CSV)"))
    return path
