"""``396`` — a curation session is a queue with a memory, not one field.

Make Masks is the better editor and the bespoke tool outside the repository
is the better workflow, and ledger item 396 is the observation that neither
is both. This module is the workflow half, pulled inside spaCR: which fields
are waiting, which order to offer them in, how many to offer before the
session ends, and what was already decided about each one.

It holds no pixels and draws nothing. That is deliberate and it is the whole
argument for bringing the work in here: the external tool's defects are found
in the middle of a curation session, because there is no way to test a
2,000-line Qt application, whereas everything below is ordinary
Python over a folder and a CSV, so a defect is found by
``tests/test_a_curation_session_resumes_where_it_stopped.py`` instead. The
module imports no Qt, needs no display, and touches ``numpy`` only inside the
functions that actually read a draft mask off the disk.

Three layouts, all accepted
---------------------------

Ledger item 396 names the layout mismatch as the thing to settle first, and
warns that silently assuming one is how a folder full of work gets loaded as
empty. So all three are accepted and the one in front of us is detected::

    nested    <folder>/*.tif           + <folder>/masks/*.tif
    sibling   <folder>/images/*.tif    + <folder>/masks/*.tif
    seg       <folder>/*_seg.npy       (Cellpose pickles, the external tool)

Accepting all three is safer than picking a winner. An existing curation
set opens unconverted, so nothing has to be moved before work can start.

A guess is not accepted. A folder that matches none of the three, or that
matches two of them, raises :class:`LayoutError`. The alternative would be an
empty queue, and an empty queue reads as "all done".

Three reviewed states, not two
------------------------------

:data:`REVIEWED` is ``done``, ``skip`` and ``recropped``. The ledger note says
two; it is wrong. ``skip`` is not a lesser ``done`` — it records "this one
cannot be curated", and collapsing the two means the same unusable field is
offered again every session. ``recropped`` is what the external tool writes
when a bundle has been split into children: the parent is out of the queue
and its pieces are in it.

The record lives at ``<folder>/curate_status.csv``, inside the queue folder
rather than beside it, so it syncs with the images and a session resumes on
the other machine. A row about a stem that is not in this folder is kept on
rewrite for exactly that reason: the curator may be resuming somewhere the
rest of the set has not arrived yet, and dropping those rows would hand back
work already done.

Ordering
--------

Ordering is the productivity feature: a curator's judgement is worth more on
the hard fields than on the first hundred easy ones. The four orders, and the
constants inside them, are the external tool's, chosen after a real 500-bundle
curation set. :func:`order_items` says what each one means.

Where this knowingly differs from the external tool
---------------------------------------------------

Five places, each because this module must serve three layouts and a test
suite rather than one folder of Cellpose pickles:

* Objects are counted as distinct non-zero labels, not ``masks.max()``. See
  :func:`count_draft_objects` for the two cases where those differ.
* The median object diameter is computed directly rather than through
  ``skimage.measure.regionprops``, so ordering a queue needs numpy alone.
  It is the same number, and a test pins the equality.
* An unknown ``order`` raises instead of silently becoming ``value``. A
  silent reorder is the failure this module is most careful about.
* ``name`` order sorts by stem rather than by file name, so the order does
  not change when one field is a PNG and the next a TIFF.
* Status rows are written sorted by stem, atomically. The file syncs between
  machines; a stable order keeps the diff to the rows that changed, and the
  rename keeps a half-written resume record off the disk.

The nested and sibling layouts also offer an image whose mask is missing,
rather than pairing strictly as the external tool's bundle builder does: a
field with no draft is work, drawing from scratch, and ``easy`` already knows
to offer it last.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .errors import SpacrError

__all__ = [
    "CurationQueueError",
    "LayoutError",
    "StatusFileError",
    "QueueItem",
    "QueueLayout",
    "StatusRow",
    "QueueSummary",
    "CurationQueue",
    "LAYOUT_NESTED",
    "LAYOUT_SIBLING",
    "LAYOUT_SEG",
    "LAYOUTS",
    "ORDERS",
    "DEFAULT_ORDER",
    "REVIEWED",
    "STATUS_FIELDS",
    "STATUS_FILENAME",
    "SCORES_FILENAME",
    "DRAFTS_FILENAME",
    "IMAGE_EXTS",
    "SEG_SUFFIX",
    "MIN_OBJECTS_FOR_VALUE",
    "MIN_DIAMETER_FOR_VALUE",
    "detect_layout",
    "discover_items",
    "status_path",
    "read_status",
    "write_status",
    "record_state",
    "mark_state",
    "is_reviewed",
    "pending_items",
    "load_probabilities",
    "load_draft_counts",
    "count_draft_objects",
    "draft_stats",
    "median_equivalent_diameter",
    "resolve_order",
    "order_items",
    "summarize",
    "build_queue",
]

PathLike = Union[str, "os.PathLike[str]", Path]

#: ``<folder>/*.tif`` with the masks in ``<folder>/masks/`` — what Make Masks
#: documents and what :func:`spacr.qt.mask_engine.mask_save_path` writes.
LAYOUT_NESTED = "nested"

#: ``<folder>/images/*.tif`` beside ``<folder>/masks/*.tif`` — how the
#: curation sets in the field are actually laid out.
LAYOUT_SIBLING = "sibling"

#: ``<folder>/*_seg.npy`` — Cellpose pickles, what the external tool edits.
LAYOUT_SEG = "seg"

#: Every layout this module can open, in the order it reports them.
LAYOUTS: Tuple[str, ...] = (LAYOUT_NESTED, LAYOUT_SIBLING, LAYOUT_SEG)

#: Image extensions, matching :data:`spacr.qt.mask_engine.IMAGE_EXTS`. Kept
#: literal here rather than imported because that module pulls in numpy and
#: imageio at import time and this one is deliberately stdlib-only;
#: ``test_a_curation_session_resumes_where_it_stopped`` holds the two equal.
IMAGE_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

#: How Cellpose names the pickle that holds an image and its draft masks.
SEG_SUFFIX = "_seg.npy"

#: The resume record, inside the queue folder so it syncs with the images.
STATUS_FILENAME = "curate_status.csv"

#: Optional per-stem probabilities, read by the ``prob`` and ``easy`` orders.
SCORES_FILENAME = "curate_scores.csv"

#: Cache of draft object counts, so ``easy`` does not reread every draft at
#: every launch. Counts drive ordering only: a stale one mis-sorts a field,
#: it never loses work.
DRAFTS_FILENAME = "curate_drafts.csv"

#: Column names accepted as the probability, most specific first.
PROB_COLUMNS: Tuple[str, ...] = ("prob", "probability", "plaque_prob", "p")

#: The columns of :data:`STATUS_FILENAME`. The external tool's Tk ancestor
#: writes ``updated`` too, so the union is declared and extras are ignored;
#: without that, writing a file the other tool created raises ``ValueError``.
STATUS_FIELDS: List[str] = ["stem", "state", "n_objects", "updated"]

#: The states that take a stem out of the queue. THREE, not two — see the
#: module docstring. Anything else, including the empty string, is pending.
REVIEWED = frozenset({"done", "skip", "recropped"})

#: Every ordering :func:`order_items` accepts.
ORDERS: Tuple[str, ...] = ("easy", "prob", "value", "name")

#: ``easy`` is the default because that is what the external tool settled on
#: after use: confirm a populated draft rather than draw an empty one.
DEFAULT_ORDER = "easy"

#: A draft with at least this many objects can be "rich" for ``value`` order.
#: Plaque-specific, hence a parameter with this as its documented default.
MIN_OBJECTS_FOR_VALUE = 3

#: ...and at least this median equivalent diameter, in pixels.
MIN_DIAMETER_FOR_VALUE = 40.0

#: What an unscored stem contributes to a probability sort key. The sign
#: matters: keys are negated probabilities, so an unscored stem yields
#: ``+1.0``, larger than any real ``-prob``, and sorts LAST rather than
#: pretending to be probability zero.
UNSCORED_PROB = -1.0


class CurationQueueError(SpacrError):
    """A curation queue could not be built, read or written.

    Subclass of :class:`spacr.errors.SpacrError`, so a caller that catches
    spaCR's own errors catches these too.
    """


class LayoutError(CurationQueueError):
    """The folder does not unambiguously hold one of the three layouts.

    Raised rather than returning an empty queue. An empty queue reads as
    "all done", which is the one wrong answer a curation tool must never
    give about a folder full of work.
    """


class StatusFileError(CurationQueueError):
    """``curate_status.csv`` exists but cannot be read as a resume record."""


@dataclass(frozen=True)
class QueueItem:
    """One field waiting to be curated.

    :ivar stem: the identity of the field, and the key of its status row.
    :ivar layout: which of :data:`LAYOUTS` this item came from.
    :ivar image: the image file, when the layout has one on disk. ``None``
        for a ``seg`` bundle that carries its own image, unless a display
        image of the same stem sits beside it.
    :ivar mask: the draft mask file, or ``None`` when no draft exists yet.
    :ivar bundle: the ``_seg.npy`` pickle, for the ``seg`` layout only.
    """

    stem: str
    layout: str
    image: Optional[Path] = None
    mask: Optional[Path] = None
    bundle: Optional[Path] = None

    @property
    def draft(self) -> Optional[Path]:
        """The file holding this item's draft labels, or ``None``.

        :returns: the bundle for a ``seg`` item, the mask file for the
            other two layouts, and ``None`` when there is no draft — which
            means drawing from scratch, not an error.
        """
        return self.bundle if self.bundle is not None else self.mask


@dataclass(frozen=True)
class QueueLayout:
    """What was found in the folder, and where.

    :ivar kind: one of :data:`LAYOUTS`.
    :ivar folder: the queue folder, which is also where the status file goes.
    :ivar images_dir: where the images are, when they are on disk.
    :ivar masks_dir: where masks are read from and written back to.
    :ivar items: every field found, sorted by stem.
    """

    kind: str
    folder: Path
    images_dir: Optional[Path]
    masks_dir: Optional[Path]
    items: Tuple[QueueItem, ...]

    @property
    def status_path(self) -> Path:
        """:returns: the path of this queue's ``curate_status.csv``."""
        return status_path(self.folder)

    def mask_destination(self, stem: str) -> Path:
        """Where a curated mask for ``stem`` is written.

        The ``seg`` layout writes back into the bundle it was read from, as
        the external tool does; the other two write
        ``<masks_dir>/<stem>.tif``, which is what
        :func:`spacr.qt.mask_engine.mask_save_path` already produces for the
        nested layout.

        :param stem: the field's stem.
        :returns: the path a curated mask belongs at.
        :raises CurationQueueError: when ``stem`` is not in this queue.
        """
        for item in self.items:
            if item.stem == stem:
                if item.bundle is not None:
                    return item.bundle
                if self.masks_dir is None:
                    raise CurationQueueError(
                        f"the {self.kind} layout at {self.folder} has no "
                        f"masks folder to write {stem} into")
                return self.masks_dir / f"{stem}.tif"
        raise CurationQueueError(
            f"{stem!r} is not in the queue at {self.folder}")


@dataclass(frozen=True)
class StatusRow:
    """One line of ``curate_status.csv``.

    :ivar stem: which field this is about.
    :ivar state: ``done``, ``skip``, ``recropped``, or anything else, which
        counts as pending. Normalised to lower case with the surrounding
        whitespace stripped, so a row typed by hand as ``Done`` still ends
        that stem's turn.
    :ivar n_objects: how many objects the field had when it was reviewed, or
        ``None`` when the file did not say.
    :ivar updated: ISO-8601 timestamp, as written.
    """

    stem: str
    state: str = ""
    n_objects: Optional[int] = None
    updated: str = ""

    @property
    def reviewed(self) -> bool:
        """:returns: whether this row takes its stem out of the queue."""
        return self.state in REVIEWED


@dataclass(frozen=True)
class QueueSummary:
    """What is left to do, in the terms ledger item 396 asks for.

    :ivar total: fields in the folder.
    :ivar done: curated.
    :ivar skip: refused — cannot be curated, and must not be offered again.
    :ivar recropped: split into children, which are in the queue instead.
    :ivar remaining: fields with no reviewed state.
    :ivar unknown: status rows about stems that are not in this folder.
        Kept, never dropped, and reported here rather than silently: on a
        second machine they are usually work already done elsewhere.
    """

    total: int
    done: int
    skip: int
    recropped: int
    remaining: int
    unknown: int = 0

    def describe(self) -> str:
        """One line a caller can print.

        The ledger's measured state, ``500 bundles, 366 done, 29 skip, 105
        remaining``, is exactly what this returns for it. The two rarer
        counts are appended only when they are not zero, so the common
        sentence stays the short one.

        :returns: a human-readable summary of the queue.
        """
        parts = [f"{self.total} bundles", f"{self.done} done",
                 f"{self.skip} skip"]
        if self.recropped:
            parts.append(f"{self.recropped} recropped")
        parts.append(f"{self.remaining} remaining")
        if self.unknown:
            parts.append(f"{self.unknown} elsewhere")
        return ", ".join(parts)


@dataclass(frozen=True)
class CurationQueue:
    """A bounded session of curation work over one folder.

    :ivar layout: what was found, and where.
    :ivar items: the fields to offer THIS session — unreviewed, ordered, and
        cut to ``limit``.
    :ivar status: every status row read, including rows about stems that are
        not in this folder.
    :ivar order: the ordering that was asked for.
    :ivar effective_order: the ordering actually used, which differs from
        ``order`` when ``prob`` or ``easy`` fell back to ``value`` for want
        of probabilities.
    :ivar limit: the cap that was applied, or ``None``.
    :ivar summary: counts over the WHOLE folder, not over this session.
    """

    layout: QueueLayout
    items: Tuple[QueueItem, ...]
    status: Dict[str, StatusRow]
    order: str
    effective_order: str
    limit: Optional[int]
    summary: QueueSummary

    @property
    def folder(self) -> Path:
        """:returns: the queue folder."""
        return self.layout.folder

    @property
    def all_items(self) -> Tuple[QueueItem, ...]:
        """:returns: every field in the folder, reviewed ones included."""
        return self.layout.items

    def describe(self) -> str:
        """:returns: a printable line naming the layout, order and counts."""
        order = self.effective_order
        if order != self.order:
            order = f"{order} (asked for {self.order})"
        return (f"{self.layout.kind} layout at {self.folder}: "
                f"{self.summary.describe()}; {len(self.items)} this session, "
                f"sorted by {order}")


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _listdir(folder: Path) -> List[Path]:
    """Entries of ``folder``, sorted, with dotfiles left out.

    macOS writes an AppleDouble ``._field.tif`` beside every file it copies,
    and those are not fields.

    :param folder: directory to list.
    :returns: sorted paths, or ``[]`` when ``folder`` is not a directory.
    """
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if not p.name.startswith(".")),
                  key=lambda p: p.name)


def _image_files(folder: Path) -> List[Path]:
    """:param folder: directory to list.

    :returns: the image files directly in ``folder``, sorted by name.
    """
    return [p for p in _listdir(folder)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _seg_files(folder: Path) -> List[Path]:
    """:param folder: directory to list.

    :returns: the ``*_seg.npy`` bundles directly in ``folder``, sorted.
    """
    return [p for p in _listdir(folder)
            if p.is_file() and p.name.endswith(SEG_SUFFIX)]


def _find_mask(masks_dir: Path, stem: str) -> Optional[Path]:
    """Locate an existing draft mask for one stem.

    ``.tif`` first because that is what everything spaCR writes uses, then
    the other image extensions, then ``.npy``.

    :param masks_dir: the folder masks are kept in.
    :param stem: the field's stem.
    :returns: the draft mask path, or ``None`` when there is no draft.
    """
    for ext in (".tif", ".tiff") + IMAGE_EXTS + (".npy",):
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _find_image(folder: Path, stem: str) -> Optional[Path]:
    """:param folder: where to look.

    :param stem: the field's stem.
    :returns: an image file of that stem in ``folder``, or ``None``.
    """
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _refuse_duplicate_stems(items: Sequence[QueueItem], folder: Path) -> None:
    """Refuse two files that share one stem.

    One stem is one status row. Two files claiming it means marking one
    ``done`` silently retires the other, which is the quiet-loss failure
    this module exists to prevent.

    :param items: the items discovered for a layout.
    :param folder: the queue folder, for the message.
    :raises CurationQueueError: when any stem appears twice.
    """
    seen: Dict[str, QueueItem] = {}
    clashes: List[str] = []
    for item in items:
        other = seen.get(item.stem)
        if other is not None:
            first = other.image or other.bundle
            second = item.image or item.bundle
            clashes.append(f"{item.stem}: {getattr(first, 'name', first)} "
                           f"and {getattr(second, 'name', second)}")
        else:
            seen[item.stem] = item
    if clashes:
        raise CurationQueueError(
            f"{folder} holds files that share a stem, and one stem is one "
            f"status row -- curating either would retire both:\n  "
            + "\n  ".join(clashes))


def _matches(folder: Path) -> Dict[str, bool]:
    """Which layouts this folder could be.

    :param folder: the queue folder.
    :returns: ``{layout: matched}`` for each of :data:`LAYOUTS`.
    """
    masks_dir = folder / "masks"
    images_dir = folder / "images"
    return {
        LAYOUT_NESTED: bool(_image_files(folder)) and masks_dir.is_dir(),
        LAYOUT_SIBLING: (images_dir.is_dir() and masks_dir.is_dir()
                         and bool(_image_files(images_dir))),
        LAYOUT_SEG: bool(_seg_files(folder)),
    }


def _no_layout_message(folder: Path) -> str:
    """Say what is in the folder and what each layout would have needed.

    :param folder: the queue folder.
    :returns: the body of the :class:`LayoutError` message.
    """
    images = len(_image_files(folder))
    segs = len(_seg_files(folder))
    has_masks = (folder / "masks").is_dir()
    has_images_dir = (folder / "images").is_dir()
    nested_images = len(_image_files(folder / "images"))
    return (
        f"{folder} is not a curation queue in any layout spaCR accepts.\n"
        f"  found: {images} image(s) in the folder, {segs} *_seg.npy "
        f"bundle(s), masks/ {'present' if has_masks else 'absent'}, "
        f"images/ {'present' if has_images_dir else 'absent'}"
        + (f" holding {nested_images} image(s)" if has_images_dir else "")
        + "\n"
        f"  nested  needs images in {folder} and a masks/ folder beneath it\n"
        f"  sibling needs {folder}/images and {folder}/masks\n"
        f"  seg     needs *_seg.npy files in {folder}\n"
        "This is refused rather than opened empty: an empty queue reads as "
        "\"all done\".")


def detect_layout(folder: PathLike) -> QueueLayout:
    """Work out which of the three layouts ``folder`` holds, and read it.

    :param folder: the queue folder.
    :returns: the :class:`QueueLayout`, with every field found in it.
    :raises LayoutError: when ``folder`` is not a directory, matches none of
        the three layouts, or matches more than one of them.
    :raises CurationQueueError: when two files in it share a stem.

    Both refusals are loud on purpose. Guessing between two layouts loads
    half a folder; assuming one that is not there loads none of it, and
    reports that as a finished queue.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise LayoutError(f"{folder} is not a directory, so there is no "
                          f"curation queue to open")

    found = _matches(folder)
    matched = [kind for kind in LAYOUTS if found[kind]]
    if not matched:
        raise LayoutError(_no_layout_message(folder))
    if len(matched) > 1:
        raise LayoutError(
            f"{folder} matches {len(matched)} layouts at once "
            f"({', '.join(matched)}), so which files are the queue is "
            f"ambiguous. Separate them -- move the images, or the bundles, "
            f"into a folder of their own -- rather than letting spaCR pick "
            f"one and load half the work.")

    kind = matched[0]
    if kind == LAYOUT_NESTED:
        masks_dir = folder / "masks"
        items = [QueueItem(stem=p.stem, layout=kind, image=p,
                           mask=_find_mask(masks_dir, p.stem))
                 for p in _image_files(folder)]
        images_dir: Optional[Path] = folder
    elif kind == LAYOUT_SIBLING:
        images_dir = folder / "images"
        masks_dir = folder / "masks"
        items = [QueueItem(stem=p.stem, layout=kind, image=p,
                           mask=_find_mask(masks_dir, p.stem))
                 for p in _image_files(images_dir)]
    else:
        images_dir = folder
        masks_dir = None
        items = [QueueItem(stem=p.name[:-len(SEG_SUFFIX)], layout=kind,
                           image=_find_image(folder,
                                             p.name[:-len(SEG_SUFFIX)]),
                           bundle=p)
                 for p in _seg_files(folder)]

    _refuse_duplicate_stems(items, folder)
    items.sort(key=lambda item: item.stem)
    return QueueLayout(kind=kind, folder=folder, images_dir=images_dir,
                       masks_dir=masks_dir, items=tuple(items))


def discover_items(folder: PathLike) -> Tuple[QueueItem, ...]:
    """Every field in ``folder``, whatever layout it is in.

    :param folder: the queue folder.
    :returns: the items, sorted by stem.
    :raises LayoutError: as :func:`detect_layout` does.
    """
    return detect_layout(folder).items


# ---------------------------------------------------------------------------
# The resume record
# ---------------------------------------------------------------------------

def status_path(folder: PathLike) -> Path:
    """:param folder: the queue folder.

    :returns: ``<folder>/curate_status.csv``, inside the queue folder so it
        travels with the images when the set is synced between machines.
    """
    return Path(folder) / STATUS_FILENAME


def is_reviewed(state: Optional[str]) -> bool:
    """:param state: a state string, or ``None``.

    :returns: whether that state takes a stem out of the queue. ``done``,
        ``skip`` and ``recropped`` all do, and they stay distinct: ``skip``
        means "cannot be curated", which is not a lesser ``done``.
    """
    return _normalise_state(state) in REVIEWED


def _normalise_state(state: Optional[str]) -> str:
    """:param state: a state string, or ``None``.

    :returns: the state stripped and lower-cased, so a status file edited by
        hand as ``Done`` or `` skip `` is still read as reviewed.
    """
    return (state or "").strip().lower()


def _as_int(text: Optional[str]) -> Optional[int]:
    """:param text: a CSV cell.

    :returns: the cell as an ``int``, or ``None`` when it is blank or not a
        number. The count is advisory — it feeds ordering and the summary —
        so an unreadable one is dropped rather than made fatal.
    """
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def read_status(folder: PathLike) -> Dict[str, StatusRow]:
    """Read the resume record.

    :param folder: the queue folder.
    :returns: ``{stem: StatusRow}``, empty when no status file exists yet.
        Rows about stems that are not in the folder are included; they are
        the reason :func:`write_status` writes back everything it is given.
    :raises StatusFileError: when the file exists but has no ``stem``
        column, which means it is not a status file at all.
    """
    path = status_path(folder)
    if not path.is_file():
        return {}
    rows: Dict[str, StatusRow] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        if "stem" not in reader.fieldnames:
            raise StatusFileError(
                f"{path} has no 'stem' column (found "
                f"{list(reader.fieldnames)}), so no field in it can be "
                f"identified; expected {STATUS_FIELDS}")
        for raw in reader:
            stem = (raw.get("stem") or "").strip()
            if not stem:
                continue
            rows[stem] = StatusRow(
                stem=stem,
                state=_normalise_state(raw.get("state")),
                n_objects=_as_int(raw.get("n_objects")),
                updated=(raw.get("updated") or "").strip(),
            )
    return rows


def write_status(folder: PathLike, rows: Mapping[str, StatusRow]) -> Path:
    """Write the resume record, atomically and in full.

    Every row given is written, including rows about stems that are not in
    this folder: the curator may be resuming on a machine where part of the
    set has not arrived, and dropping those rows offers work back that was
    already finished somewhere else.

    Rows are written sorted by stem. The file syncs between machines, and a
    stable order keeps a sync diff to the rows that actually changed.

    :param folder: the queue folder.
    :param rows: ``{stem: StatusRow}``, as :func:`read_status` returns.
    :returns: the path written.
    :raises StatusFileError: when the file cannot be written.
    """
    path = status_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            for stem in sorted(rows):
                row = rows[stem]
                writer.writerow({
                    "stem": row.stem,
                    "state": row.state,
                    "n_objects": ("" if row.n_objects is None
                                  else str(int(row.n_objects))),
                    "updated": row.updated,
                })
        os.replace(temporary, path)
    except OSError as exc:
        raise StatusFileError(f"could not write {path}: {exc}") from exc
    return path


def record_state(rows: Mapping[str, StatusRow], stem: str, state: str,
                 n_objects: Optional[int] = None,
                 updated: Optional[str] = None) -> Dict[str, StatusRow]:
    """Return ``rows`` with one stem's decision recorded.

    Pure: the mapping passed in is not modified, so a caller can decide
    whether the change reaches the disk.

    :param rows: the rows read so far.
    :param stem: the field being decided.
    :param state: ``done``, ``skip``, ``recropped``, or any other string,
        which leaves the stem pending.
    :param n_objects: how many objects it had, if known.
    :param updated: the timestamp to record; defaults to now, to the second.
    :returns: a new mapping including the recorded row.
    """
    if updated is None:
        updated = datetime.now().isoformat(timespec="seconds")
    merged = dict(rows)
    merged[stem] = StatusRow(stem=stem, state=_normalise_state(state),
                             n_objects=n_objects, updated=updated)
    return merged


def mark_state(folder: PathLike, stem: str, state: str,
               n_objects: Optional[int] = None,
               updated: Optional[str] = None) -> Dict[str, StatusRow]:
    """Record one decision and write the file, keeping every other row.

    This is the one call a curation session makes per field, and it is the
    call where rows about absent stems would otherwise be lost.

    :param folder: the queue folder.
    :param stem: the field being decided.
    :param state: the state to record.
    :param n_objects: how many objects it had, if known.
    :param updated: the timestamp to record; defaults to now.
    :returns: the rows as they now stand on disk.
    :raises StatusFileError: when the file cannot be read or written.
    """
    rows = record_state(read_status(folder), stem, state, n_objects, updated)
    write_status(folder, rows)
    return rows


def pending_items(items: Iterable[QueueItem],
                  status: Mapping[str, StatusRow]) -> List[QueueItem]:
    """:param items: fields in the folder.

    :param status: the rows read from the status file.
    :returns: the items with no reviewed state, in the order given.
    """
    return [item for item in items
            if not is_reviewed(
                status[item.stem].state if item.stem in status else "")]


def summarize(items: Iterable[QueueItem],
              status: Mapping[str, StatusRow]) -> QueueSummary:
    """Count the folder the way the ledger asks for it.

    :param items: every field in the folder.
    :param status: the rows read from the status file.
    :returns: the :class:`QueueSummary`. Counts are over fields that are
        actually here, so ``done + skip + recropped + remaining == total``;
        rows about stems that are not here are reported separately as
        ``unknown`` rather than subtracted from what is left to do.
    """
    items = list(items)
    stems = {item.stem for item in items}
    states = [status[stem].state if stem in status else "" for stem in stems]
    done = sum(1 for state in states if state == "done")
    skip = sum(1 for state in states if state == "skip")
    recropped = sum(1 for state in states if state == "recropped")
    remaining = sum(1 for state in states if state not in REVIEWED)
    unknown = sum(1 for stem in status if stem not in stems)
    return QueueSummary(total=len(items), done=done, skip=skip,
                        recropped=recropped, remaining=remaining,
                        unknown=unknown)


# ---------------------------------------------------------------------------
# Reading drafts — the only place numpy is needed
# ---------------------------------------------------------------------------

def _read_draft(item: QueueItem):
    """Load one item's draft labels.

    ``numpy`` is imported HERE rather than at module scope so this module
    stays importable in a bare environment, the way
    :mod:`spacr.validate` does it.

    :param item: the field to read.
    :returns: the draft label array, or ``None`` when there is no draft.
    :raises Exception: whatever reading a corrupt file raises; callers that
        order a queue treat that as "unreadable" rather than propagating.
    """
    path = item.draft
    if path is None:
        return None
    import numpy as np

    if item.bundle is not None and path == item.bundle:
        payload = np.load(path, allow_pickle=True).item()
        return np.asarray(payload["masks"])
    from .mask_io import load_mask

    return load_mask(path)


def count_draft_objects(item: QueueItem) -> int:
    """How many objects the draft holds.

    Counted as DISTINCT NON-ZERO LABELS rather than as ``masks.max()``,
    which is what the external tool uses. The two agree on a fresh Cellpose
    pickle, whose ids run 1..N, and disagree in the two places that matter
    here: a label TIFF that has had objects deleted has gaps in its ids, so
    ``max()`` overcounts; and a 0/255 binary mask, which the nested and
    sibling layouts can perfectly well hold, would report 255 objects.

    :param item: the field to count.
    :returns: the number of objects, and ``0`` for a draft that is missing
        or unreadable — as the external tool's count cache also does, since
        counts drive ordering only.
    """
    try:
        labels = _read_draft(item)
    except Exception:
        return 0
    if labels is None:
        return 0
    try:
        import numpy as np

        array = np.asarray(labels)
        if array.size == 0:
            return 0
        return int(np.count_nonzero(np.unique(array)))
    except Exception:
        return 0


def median_equivalent_diameter(labels) -> float:
    """Median circle-equivalent diameter of the objects in a label image.

    This is ``skimage.measure.regionprops``'s ``equivalent_diameter_area``
    for 2-D input, computed directly as ``sqrt(4 * area / pi)`` so that
    ordering a queue needs numpy and not scikit-image. The equality is
    pinned by a test rather than asserted here.

    :param labels: a 2-D integer label image.
    :returns: the median diameter in pixels, or ``0.0`` when there are no
        objects.
    """
    import numpy as np

    array = np.asarray(labels)
    if array.size == 0:
        return 0.0
    flat = array.ravel()
    positive = flat[flat > 0]
    if positive.size == 0:
        return 0.0
    areas = np.bincount(positive.astype(np.int64))
    areas = areas[areas > 0]
    if areas.size == 0:
        return 0.0
    return float(np.median(np.sqrt(4.0 * areas.astype(np.float64) / np.pi)))


def draft_stats(item: QueueItem) -> Tuple[int, float]:
    """:param item: the field to measure.

    :returns: ``(n_objects, median_diameter)`` for its draft.
    :raises Exception: whatever reading an unreadable draft raises. The
        ``value`` order catches that and sorts the field dead last.
    """
    labels = _read_draft(item)
    if labels is None:
        return 0, 0.0
    return count_draft_objects(item), median_equivalent_diameter(labels)


def value_key(item: QueueItem,
              min_objects: int = MIN_OBJECTS_FOR_VALUE,
              min_diameter: float = MIN_DIAMETER_FOR_VALUE
              ) -> Tuple[int, float, int]:
    """Sort key for ``value`` order: how much draft there is to correct.

    Exactly the external tool's ``_value``::

        n == 0                     -> (2, 0, 0)   nothing to correct, last
        unreadable                 -> (3, 0, 0)   dead last
        otherwise                  -> (0 if rich else 1, -diameter, -n)

    :param item: the field to rank.
    :param min_objects: objects a "rich" draft needs. Plaque-specific.
    :param min_diameter: median diameter a "rich" draft needs, in pixels.
        Plaque-specific.
    :returns: the sort key.
    """
    if item.draft is None:
        return (2, 0.0, 0)
    try:
        n_objects, diameter = draft_stats(item)
    except Exception:
        return (3, 0.0, 0)
    if n_objects == 0:
        return (2, 0.0, 0)
    rich = n_objects >= min_objects and diameter >= min_diameter
    return (0 if rich else 1, -diameter, -n_objects)


# ---------------------------------------------------------------------------
# Probabilities and cached draft counts
# ---------------------------------------------------------------------------

def load_probabilities(folder: PathLike) -> Dict[str, float]:
    """Read ``curate_scores.csv``, when there is one.

    :param folder: the queue folder.
    :returns: ``{stem: probability}``, and ``{}`` when the file is absent —
        which is what makes ``prob`` and ``easy`` fall back, loudly.
    :raises CurationQueueError: when the file is present but has no stem
        column or no recognised probability column. A scores file that
        exists and cannot be used is a mistake worth stopping for; a
        missing one is an ordinary state.
    """
    path = Path(folder) / SCORES_FILENAME
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "stem" not in fields:
            raise CurationQueueError(
                f"{path} has no 'stem' column (found {fields})")
        column = next((name for name in PROB_COLUMNS if name in fields), None)
        if column is None:
            raise CurationQueueError(
                f"{path} has no probability column; expected one of "
                f"{list(PROB_COLUMNS)} but found {fields}")
        probabilities: Dict[str, float] = {}
        for raw in reader:
            stem = (raw.get("stem") or "").strip()
            if not stem:
                continue
            try:
                probabilities[stem] = float(raw.get(column))
            except (TypeError, ValueError):
                continue
    return probabilities


def load_draft_counts(folder: PathLike, items: Sequence[QueueItem],
                      write_cache: bool = True,
                      announce: Optional[Callable[[str], None]] = None
                      ) -> Dict[str, int]:
    """Draft object counts for ``items``, cached in ``curate_drafts.csv``.

    Reading three hundred drafts at every launch is slow, so counts are
    cached by stem and only stems missing from the cache are read. Counts
    drive ORDERING only: a stale one mis-sorts a field, it never loses work.

    :param folder: the queue folder.
    :param items: the fields to count.
    :param write_cache: whether to write the cache back. A failure to write
        is announced, never raised — a read-only folder must still open.
    :param announce: where progress goes; defaults to :func:`print`.
    :returns: ``{stem: n_objects}`` for every item given.
    """
    say = announce if announce is not None else print
    path = Path(folder) / DRAFTS_FILENAME
    cache: Dict[str, int] = {}
    if path.is_file():
        try:
            with open(path, newline="", encoding="utf-8") as handle:
                for raw in csv.DictReader(handle):
                    stem = (raw.get("stem") or "").strip()
                    count = _as_int(raw.get("n_objects"))
                    if stem and count is not None:
                        cache[stem] = count
        except (OSError, csv.Error):
            cache = {}

    missing = [item for item in items if item.stem not in cache]
    if missing:
        say(f"caching draft counts for {len(missing)} field(s) (one-off)...")
        for item in missing:
            cache[item.stem] = count_draft_objects(item)
        if write_cache:
            try:
                with open(path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle,
                                            fieldnames=["stem", "n_objects"])
                    writer.writeheader()
                    for stem in sorted(cache):
                        writer.writerow({"stem": stem,
                                         "n_objects": cache[stem]})
            except OSError as exc:
                say(f"  (draft-count cache not written: {exc})")
    return {item.stem: cache.get(item.stem, 0) for item in items}


def _resolve(source, default):
    """Resolve a mapping that may be given as a callable.

    Passing a callable is how ``name`` order avoids loading probabilities at
    all: the callable is simply never called.

    :param source: a mapping, a zero-argument callable returning one, or
        ``None``.
    :param default: what ``None`` means.
    :returns: the mapping.
    """
    if source is None:
        return default
    if callable(source):
        return source()
    return source


def resolve_order(order: str, probabilities: Mapping[str, float],
                  scores_hint: Optional[PathLike] = None
                  ) -> Tuple[str, Optional[str]]:
    """Decide which ordering will actually be used, and what to say about it.

    ``prob`` and ``easy`` both need probabilities. Without them they fall
    back to ``value`` — AND SAY SO. The announcement is half the behaviour:
    a silent reorder leaves the curator believing they are working the
    uncertain fields first when they are not.

    :param order: the ordering asked for, one of :data:`ORDERS`.
    :param probabilities: the probabilities available, possibly empty.
    :param scores_hint: the scores file that would have supplied them,
        named in the message so the fix is obvious.
    :returns: ``(effective_order, announcement)``, the announcement being
        ``None`` when nothing changed.
    :raises ValueError: when ``order`` is not one of :data:`ORDERS`. A typo
        must not quietly become a different ordering.
    """
    if order not in ORDERS:
        raise ValueError(
            f"unknown order {order!r}; expected one of {list(ORDERS)}")
    if order in ("prob", "easy") and not probabilities:
        where = f" ({scores_hint})" if scores_hint is not None else ""
        return "value", (
            f"! no curation probabilities for this queue{where} -- "
            f"falling back to value order instead of {order}")
    return order, None


def order_items(items: Iterable[QueueItem], order: str = DEFAULT_ORDER,
                probs=None, counts=None,
                announce: Optional[Callable[[str], None]] = None,
                min_objects: int = MIN_OBJECTS_FOR_VALUE,
                min_diameter: float = MIN_DIAMETER_FOR_VALUE,
                scores_hint: Optional[PathLike] = None) -> List[QueueItem]:
    """Order a queue, with the external tool's semantics.

    ``name``
        By stem, and nothing else is read — not the probabilities, not the
        drafts. By stem rather than by file name so that the order does not
        change when one field is a PNG and the next a TIFF.

    ``prob``
        Most likely first: the key is ``-probs.get(stem, -1.0)``. The
        default is the point. An unscored field yields ``+1.0``, larger than
        any real negated probability, so it sorts LAST rather than
        pretending to be probability zero.

    ``easy``
        ``(0 if n > 0 else 1, -probs.get(stem, -1.0), -n)``. A populated
        draft is accept-or-light-edit; an EMPTY draft means drawing from
        scratch, so empty ones go last REGARDLESS of probability. Within
        each group, most likely first, then most objects.

    ``value``
        By :func:`value_key`: how much draft there is to correct.

    Ties are broken by stem, because the list is sorted by stem before any
    of this and Python's sort is stable.

    :param items: the fields to order.
    :param order: one of :data:`ORDERS`.
    :param probs: ``{stem: probability}``, or a callable returning one, or
        ``None`` for none available.
    :param counts: ``{stem: n_objects}``, or a callable returning one. When
        ``None``, ``easy`` reads the drafts itself.
    :param announce: where the fallback notice goes; defaults to
        :func:`print`, so it lands on stdout as the external tool's does.
    :param min_objects: passed to :func:`value_key`.
    :param min_diameter: passed to :func:`value_key`.
    :param scores_hint: the scores file named in the fallback notice.
    :returns: a new list, ordered.
    :raises ValueError: when ``order`` is not one of :data:`ORDERS`.
    """
    say = announce if announce is not None else print
    if order not in ORDERS:
        raise ValueError(
            f"unknown order {order!r}; expected one of {list(ORDERS)}")

    ordered = sorted(items, key=lambda item: item.stem)
    if order == "name":
        return ordered

    probabilities = _resolve(probs, {})
    order, notice = resolve_order(order, probabilities, scores_hint)
    if notice is not None:
        say(notice)

    if order == "easy":
        counted = _resolve(counts, None)
        if counted is None:
            counted = {item.stem: count_draft_objects(item)
                       for item in ordered}

        def easy_key(item: QueueItem) -> Tuple[int, float, int]:
            """:param item: one field.

            :returns: its ``easy`` sort key, populated drafts first.
            """
            n_objects = counted.get(item.stem, 0)
            return (0 if n_objects > 0 else 1,
                    -probabilities.get(item.stem, UNSCORED_PROB),
                    -n_objects)

        return sorted(ordered, key=easy_key)

    if order == "prob":
        return sorted(ordered,
                      key=lambda item: -probabilities.get(item.stem,
                                                          UNSCORED_PROB))

    return sorted(ordered, key=lambda item: value_key(item, min_objects,
                                                      min_diameter))


def build_queue(folder: PathLike, order: str = DEFAULT_ORDER,
                limit: Optional[int] = None, probs=None, counts=None,
                announce: Optional[Callable[[str], None]] = None,
                include_reviewed: bool = False,
                min_objects: int = MIN_OBJECTS_FOR_VALUE,
                min_diameter: float = MIN_DIAMETER_FOR_VALUE,
                cache_counts: bool = True) -> CurationQueue:
    """Open a folder as a bounded curation session.

    In order: detect the layout, read the resume record, drop everything
    already reviewed, order what is left, and only THEN apply the limit — so
    ``--limit 20`` means the twenty most worthwhile fields still to do, not
    the first twenty found.

    :param folder: the queue folder.
    :param order: one of :data:`ORDERS`; :data:`DEFAULT_ORDER` by default.
    :param limit: how many fields this session offers, or ``None`` for all.
    :param probs: probabilities to use instead of reading
        ``curate_scores.csv``.
    :param counts: draft counts to use instead of reading the drafts.
    :param announce: where notices go; defaults to :func:`print`.
    :param include_reviewed: offer reviewed fields too. Off by default;
        this is what "resume" means.
    :param min_objects: passed to :func:`value_key`.
    :param min_diameter: passed to :func:`value_key`.
    :param cache_counts: whether ``easy`` may write its count cache.
    :returns: the :class:`CurationQueue`.
    :raises LayoutError: when the folder is not one unambiguous layout.
    :raises StatusFileError: when the status file cannot be read.
    :raises ValueError: when ``order`` is not one of :data:`ORDERS`.
    """
    say = announce if announce is not None else print
    if order not in ORDERS:
        raise ValueError(
            f"unknown order {order!r}; expected one of {list(ORDERS)}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must not be negative, got {limit}")

    layout = detect_layout(folder)
    status = read_status(layout.folder)
    summary = summarize(layout.items, status)

    waiting = (list(layout.items) if include_reviewed
               else pending_items(layout.items, status))

    # Resolved ONLY for the orders that use them, so `name` and `value`
    # read no scores file and call no probability loader at all.
    probabilities: Optional[Mapping[str, float]] = None
    if order in ("prob", "easy"):
        probabilities = _resolve(probs, None)
        if probabilities is None:
            probabilities = load_probabilities(layout.folder)
    effective, notice = resolve_order(order, probabilities or {},
                                      Path(layout.folder) / SCORES_FILENAME)
    if notice is not None:
        say(notice)

    counted = _resolve(counts, None)
    if counted is None and effective == "easy":
        counted = load_draft_counts(layout.folder, waiting,
                                    write_cache=cache_counts, announce=say)

    ordered = order_items(waiting, effective, probs=probabilities or {},
                          counts=counted, announce=say,
                          min_objects=min_objects, min_diameter=min_diameter)
    if limit is not None:
        ordered = ordered[:limit]

    return CurationQueue(layout=layout, items=tuple(ordered), status=status,
                         order=order, effective_order=effective, limit=limit,
                         summary=summary)
