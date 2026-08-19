"""A crop path recorded on one computer, resolved on another.

``png_list.png_path`` is written ABSOLUTE at crop time. Move the screen to
another machine -- or mount the same NAS somewhere else -- and every one of
those paths is dead while the files themselves are all present, because what
the user preserved is the STRUCTURE BELOW THE PLATE, not the prefix above it:

    <recorded root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...
    <current  root>/plate1/data/single_nucleus/single_pathogen/plate1_H19/...

Measured on the TSG101 screen: 0 of 60,816 recorded paths existed, and
60,816 of 60,816 existed once the part below ``data/`` was rebuilt under the
plate folder the database was opened from.

THE RULE IS "RE-ROOT ONLY WHAT LANDS ON A FILE THAT EXISTS", and that is the
whole safety argument. ``io.py`` already carried this split as a nested local
function that rewrote unconditionally and let the caller discover later
whether the result was real; a path rewritten to somewhere equally absent is
strictly worse than the original, because the error then names a folder the
user never chose. Nothing here changes a path it cannot first confirm.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence

#: The folder every exported crop lives under, and the hinge the rebuild
#: turns on. It is `data/` because that is the layout `measure` writes and
#: the one the maintainer described as the thing a user keeps intact.
DATA_FOLDER = "data"


def _candidate_tails(path: str) -> List[str]:
    """The parts of ``path`` below a ``data/`` component, innermost first.

    Several, not one: a recorded root may itself contain a ``data``
    component (``/nas/data/proj/plate1/data/...``), and splitting on the
    first would keep ``proj/plate1/data`` in the tail. Innermost first is
    right for the real layout, and every candidate is checked against the
    disk before it is used, so a wrong guess cannot survive.
    """
    parts = str(path).replace("\\", "/").split("/")
    tails: List[str] = []
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == DATA_FOLDER and index + 1 < len(parts):
            tails.append("/".join(parts[index + 1:]))
    return tails


def reroot_crop_path(path: Optional[str], src_root: Optional[str]) -> Optional[str]:
    """``path`` as it exists under ``src_root``, or ``path`` unchanged.

    :param path: the recorded absolute path, or anything falsy.
    :param src_root: the folder the database was actually opened from -- the
        plate folder, which is what holds ``data/``.
    :returns: a path that EXISTS, when one could be built; otherwise the
        input untouched, so a caller's error still names what was recorded.
    """
    if not path or not isinstance(path, str) or not path.strip():
        return path
    if os.path.exists(path):
        return path
    if not src_root:
        return path
    root = os.path.abspath(os.path.expanduser(os.fspath(src_root)))
    for tail in _candidate_tails(path):
        candidate = os.path.join(root, DATA_FOLDER, *tail.split("/"))
        if os.path.exists(candidate):
            return candidate
    return path


def reroot_column(frame, column: str, src_root: Optional[str]):
    """Re-root one path column of ``frame`` in place, and say how many moved.

    :param frame: any DataFrame; missing columns are not an error, because
        the merged route and the PNG route carry different ones and a caller
        should be able to ask for both.
    :returns: how many values were rewritten.
    """
    if frame is None or column not in getattr(frame, "columns", ()):
        return 0
    original = frame[column]
    rebuilt = original.map(lambda value: reroot_crop_path(value, src_root))
    moved = int((rebuilt.astype(str) != original.astype(str)).sum())
    if moved:
        frame[column] = rebuilt
    return moved


def source_root_for_database(db_path: str) -> str:
    """The plate folder a ``measurements.db`` belongs to.

    ``<plate>/measurements/measurements.db`` -> ``<plate>``, which is the
    folder that holds ``data/``. Derived rather than passed so the montage
    gains portability without new plumbing through every caller.
    """
    if not db_path:
        return ""
    return os.path.dirname(os.path.dirname(os.path.abspath(db_path)))
