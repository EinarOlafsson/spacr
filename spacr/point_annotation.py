"""Point-count annotation over a folder of images (item 957, alpha).

Make Masks' point mode marks one point per object instead of painting a
mask: a curator clicks the centre of every nucleus in a crop, and a scorer
later asks which segmented object each point falls in. This module is the
session behind that mode. It holds the points of one image, the undo
stack, and the file the points are saved to, in plain Python, so the
round trip is tested without a display. :mod:`spacr.qt.point_annotator`
is the window that drives it.

The folder is opened as the resumable curation queue of
:mod:`spacr.curation_queue`: its ``curate_status.csv`` records which images
are done, so a session closed half way reopens where it stopped. A folder
that holds only images, with no ``masks/`` folder beside them, is accepted
as the ``images`` layout, because point mode reads no masks.

The points file
---------------

Each image's points are saved to
``<folder>/../annotations/<stem>_points.csv``, where ``<stem>`` is the
image's file name without its extension and without a trailing
``_clean``; ``crop01_site3_y1234_x567_clean.png`` saves as
``crop01_site3_y1234_x567_points.csv``. The columns are:

* ``X`` and ``Y``: the clicked pixel's column and row in image pixels,
  origin at the top left, written as floats. This is the convention of
  Fiji's Analyze -> Measure on a multi-point selection, and a scorer that
  reads ``P[["Y", "X"]]``, rounds and indexes a label image gets back
  exactly the pixel that was clicked;
* ``unsure``: 1 for a point the curator marked unsure, else 0;
* ``note``: free text, empty unless written.

Every change is written at once, through a temporary file and a rename, so
a crash never leaves a half-written file, and saving again overwrites.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .curation_queue import (
    LAYOUT_SEG,
    CurationQueue,
    CurationQueueError,
    LayoutError,
    QueueItem,
    QueueLayout,
    _image_files,
    _matches,
    _refuse_duplicate_stems,
    detect_layout,
    order_items,
    read_status,
    summarize,
)

__all__ = [
    "POINT_COLUMNS",
    "POINTS_SUFFIX",
    "LAYOUT_IMAGES",
    "Point",
    "PointSession",
    "annotations_folder",
    "points_path",
    "read_points",
    "write_points",
    "default_overlays",
    "overlays_for",
    "all_done",
    "build_point_queue",
]

PathLike = Union[str, "os.PathLike[str]", Path]

#: The columns of a points file, in the order they are written.
POINT_COLUMNS: Tuple[str, ...] = ("X", "Y", "unsure", "note")

#: How a points file is named after its image's stem.
POINTS_SUFFIX = "_points.csv"

#: Dropped from the end of an image's stem before naming its points file.
_CLEAN_SUFFIX = "_clean"

#: A folder of images with no masks folder, which only point mode opens.
LAYOUT_IMAGES = "images"

#: How far from a click, in image pixels, a point is still "the nearest".
_DEFAULT_RADIUS = 6.0


@dataclass(frozen=True)
class Point:
    """One clicked point.

    :ivar x: the column of the clicked pixel.
    :ivar y: the row of the clicked pixel.
    :ivar unsure: whether the curator marked the point unsure.
    :ivar note: free text about the point.
    """

    x: float
    y: float
    unsure: bool = False
    note: str = ""


def annotations_folder(folder: PathLike) -> Path:
    """Where point mode saves the points of a queue folder.

    :param folder: the queue folder.
    :returns: ``<folder>/../annotations``.
    """
    return Path(folder).absolute().parent / "annotations"


def points_path(image: PathLike, annotations: PathLike) -> Path:
    """The points file of one image.

    :param image: the image file.
    :param annotations: the folder points files are saved in.
    :returns: ``<annotations>/<stem>_points.csv``, the stem without a
        trailing ``_clean``.
    """
    stem = Path(image).stem
    if stem.endswith(_CLEAN_SUFFIX) and len(stem) > len(_CLEAN_SUFFIX):
        stem = stem[:-len(_CLEAN_SUFFIX)]
    return Path(annotations) / f"{stem}{POINTS_SUFFIX}"


def _as_flag(text: Optional[str]) -> bool:
    """Read an ``unsure`` cell: 1, true or yes in any case is unsure.

    :param text: the cell, or None when the column is missing.
    :returns: whether the point is unsure.
    """
    return (text or "").strip().lower() in ("1", "1.0", "true", "yes")


def read_points(path: PathLike) -> List[Point]:
    """Read a points file.

    A file saved by Fiji's Measure, with more columns than these, is read
    too; only ``X`` and ``Y`` are required.

    :param path: the points file.
    :returns: the points in file order, or ``[]`` when the file is absent.
    :raises CurationQueueError: when the file has no ``X`` or no ``Y``
        column, or a coordinate that is not a number.
    """
    path = Path(path)
    if not path.is_file():
        return []
    points: List[Point] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        if "X" not in names or "Y" not in names:
            raise CurationQueueError(
                f"{path} has no X and Y columns (found {list(names)})")
        for row in reader:
            try:
                x, y = float(row["X"]), float(row["Y"])
            except (TypeError, ValueError) as exc:
                raise CurationQueueError(
                    f"{path}: X={row.get('X')!r}, Y={row.get('Y')!r} is not "
                    f"a point") from exc
            points.append(Point(x=x, y=y, unsure=_as_flag(row.get("unsure")),
                                note=(row.get("note") or "")))
    return points


def write_points(path: PathLike, points: Iterable[Point]) -> Path:
    """Write a points file atomically, replacing what was there.

    :param path: the points file.
    :param points: the points, written in order.
    :returns: the path written.
    :raises CurationQueueError: when the file cannot be written.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(POINT_COLUMNS)
            for point in points:
                writer.writerow([repr(float(point.x)), repr(float(point.y)),
                                 int(bool(point.unsure)), point.note])
        os.replace(temporary, path)
    except OSError as exc:
        raise CurationQueueError(f"could not write {path}: {exc}") from exc
    return path


def default_overlays(folder: PathLike) -> List[Path]:
    """The overlay folders a point queue uses when none are named.

    :param folder: the queue folder.
    :returns: ``[<folder>/../overlays]`` when that folder exists, else ``[]``.
    """
    candidate = Path(folder).absolute().parent / "overlays"
    return [candidate] if candidate.is_dir() else []


def overlays_for(image: PathLike, folders: Sequence[PathLike]) -> List[Path]:
    """The overlay images of one image, found by its points stem.

    :param image: the image annotated.
    :param folders: folders of overlay images.
    :returns: the image files in ``folders`` named ``<stem>.<ext>`` or
        ``<stem>_<anything>.<ext>``, in folder order and then by name.
    """
    stem = points_path(image, ".").name[:-len(POINTS_SUFFIX)]
    found: List[Path] = []
    for folder in folders:
        for candidate in _image_files(Path(folder)):
            if candidate.stem == stem or candidate.stem.startswith(stem + "_"):
                found.append(candidate)
    return found


def all_done(queue: CurationQueue) -> bool:
    """Whether every image of a point queue is marked done, read from disk.

    :param queue: the queue :func:`build_point_queue` built.
    :returns: True when every image of the folder has the state ``done``.
    """
    status = read_status(queue.folder)
    return bool(queue.all_items) and all(
        item.stem in status and status[item.stem].state == "done"
        for item in queue.all_items)


def _images_layout(folder: Path) -> QueueLayout:
    """Read a folder for point mode: a curation layout, or bare images.

    :param folder: the queue folder.
    :returns: the layout.
    :raises LayoutError: when the folder is not a folder, or holds no
        image, or more than one layout.
    :raises CurationQueueError: when it holds ``_seg.npy`` bundles, or two
        images whose points files would share a name.
    """
    if not folder.is_dir():
        raise LayoutError(f"{folder} is not a directory, so there is no "
                          f"curation queue to open")
    if any(_matches(folder).values()):
        layout = detect_layout(folder)
    else:
        images = _image_files(folder)
        if not images:
            raise LayoutError(f"{folder} holds no image to annotate")
        items = [QueueItem(stem=p.stem, layout=LAYOUT_IMAGES, image=p)
                 for p in images]
        _refuse_duplicate_stems(items, folder)
        items.sort(key=lambda item: item.stem)
        layout = QueueLayout(kind=LAYOUT_IMAGES, folder=folder,
                             images_dir=folder, masks_dir=None,
                             items=tuple(items))
    if layout.kind == LAYOUT_SEG:
        raise CurationQueueError(
            f"point mode annotates image files; {folder} holds _seg.npy "
            f"bundles")
    names = {}
    for item in layout.items:
        name = points_path(item.image, ".").name
        if name in names:
            raise CurationQueueError(
                f"{item.image.name} and {names[name]} would both save their "
                f"points to {name}")
        names[name] = item.image.name
    return layout


def build_point_queue(folder: PathLike) -> CurationQueue:
    """Open a folder of images as a point-annotation session.

    Every image of the folder is offered, in name order, done ones
    included, so a curator can step back to an earlier image; the editor
    opens on the first one not done.

    :param folder: the queue folder: images with or without a ``masks/``
        folder, or an ``images/`` folder beside ``masks/``.
    :returns: the queue, ordered by name.
    :raises LayoutError: when the folder is not a folder, holds no image,
        or holds more than one layout.
    :raises CurationQueueError: when it holds ``_seg.npy`` bundles, or two
        images whose points files would share a name.
    """
    layout = _images_layout(Path(folder))
    status = read_status(layout.folder)
    items = order_items(layout.items, "name")
    return CurationQueue(layout=layout, items=tuple(items), status=status,
                         order="name", effective_order="name", limit=None,
                         summary=summarize(layout.items, status))


class PointSession:
    """The points of one image, with undo, saved on every change.

    :param image: the image annotated.
    :param annotations: the folder its points file is saved in.
    :param shape: the image's ``(height, width)``; a click outside it is
        refused. None accepts any point.
    :param autosave: write the points file after every change.
    """

    def __init__(self, image: PathLike, annotations: PathLike,
                 shape: Optional[Tuple[int, int]] = None,
                 autosave: bool = True) -> None:
        """Load the image's saved points, if it has any.

        :param image: the image annotated.
        :param annotations: the folder its points file is saved in.
        :param shape: the image's ``(height, width)``, or None.
        :param autosave: write the points file after every change.
        """
        self.image = Path(image)
        self.path = points_path(image, annotations)
        self.shape = shape
        self.autosave = autosave
        self._points: List[Point] = read_points(self.path)
        self._undo: List[List[Point]] = []

    @property
    def points(self) -> Tuple[Point, ...]:
        """The points, in the order they were added."""
        return tuple(self._points)

    @property
    def count(self) -> int:
        """How many points there are."""
        return len(self._points)

    @property
    def unsure_count(self) -> int:
        """How many points are marked unsure."""
        return sum(1 for point in self._points if point.unsure)

    def _changed(self, before: List[Point]) -> None:
        """Remember ``before`` for undo and save.

        :param before: the points as they were before the change.
        """
        self._undo.append(before)
        if self.autosave:
            self.save()

    def _nearest(self, x: float, y: float, radius: float) -> Optional[int]:
        """The index of the point nearest ``(x, y)`` within ``radius``.

        :param x: column.
        :param y: row.
        :param radius: the largest distance, in image pixels.
        :returns: the index, or None when no point is that close.
        """
        best, best_distance = None, float(radius)
        for index, point in enumerate(self._points):
            distance = math.hypot(point.x - x, point.y - y)
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def add(self, x: float, y: float) -> Optional[Point]:
        """Add a point at the pixel ``(x, y)``.

        :param x: the clicked pixel's column.
        :param y: the clicked pixel's row.
        :returns: the point added, or None when it lies outside the image.
        """
        if self.shape is not None:
            height, width = self.shape
            if not (0 <= x < width and 0 <= y < height):
                return None
        before = list(self._points)
        point = Point(x=float(x), y=float(y))
        self._points.append(point)
        self._changed(before)
        return point

    def remove_nearest(self, x: float, y: float,
                       radius: float = _DEFAULT_RADIUS) -> Optional[Point]:
        """Remove the point nearest ``(x, y)``, if one is within ``radius``.

        :param x: column.
        :param y: row.
        :param radius: the largest distance, in image pixels.
        :returns: the point removed, or None.
        """
        index = self._nearest(x, y, radius)
        if index is None:
            return None
        before = list(self._points)
        removed = self._points.pop(index)
        self._changed(before)
        return removed

    def toggle_unsure(self, x: Optional[float] = None,
                      y: Optional[float] = None,
                      radius: float = _DEFAULT_RADIUS) -> Optional[Point]:
        """Flip "unsure" on the point nearest ``(x, y)``, else on the last.

        :param x: column, or None to use the last point.
        :param y: row, or None to use the last point.
        :param radius: the largest distance, in image pixels.
        :returns: the point as it now is, or None when there is none.
        """
        index = None
        if x is not None and y is not None:
            index = self._nearest(x, y, radius)
        if index is None and self._points:
            index = len(self._points) - 1
        if index is None:
            return None
        before = list(self._points)
        point = self._points[index]
        self._points[index] = replace(point, unsure=not point.unsure)
        self._changed(before)
        return self._points[index]

    def undo(self) -> bool:
        """Take back the last add, removal or unsure change.

        :returns: whether there was a change to take back.
        """
        if not self._undo:
            return False
        self._points = self._undo.pop()
        if self.autosave:
            self.save()
        return True

    def save(self) -> Path:
        """Write the points file now.

        :returns: the path written.
        :raises CurationQueueError: when it cannot be written.
        """
        return write_points(self.path, self._points)
