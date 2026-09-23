"""Pure-Python mask editing and persistence for the Qt Make Masks screen.

This module provides image and mask I/O plus non-brush label operations,
including fill, relabel, size and intensity filtering, Otsu detection, and
magic-wand selection. It has no Qt dependency, so the editing operations can
be tested without a display.

THREE INVERSIONS LIVE HERE AND THEY DO DIFFERENT THINGS (items 435 and 419
point 9). :func:`invert_intensity` is the photographic complement of an
IMAGE -- ``dtype_max - value``, what a viewer's Invert does, exactly
reversible on every integer dtype -- and it is what the Make Masks screen's
"Invert image" draws with. :func:`invert_for_detection` reflects an image
about its OWN range instead, and is what the screen's "Invert for detection"
hands the detectors; its docstring has the measurement that says why a
detector cannot use the complement, and it is NOT a duplicate to be merged
away. :func:`invert_mask` flips a LABEL image's foreground and background;
on an ordinary field that gives one object covering the frame, which is why
it was reported as doing nothing.

:func:`save_mask` passes labels through :func:`canonical_labels`, which
preserves existing nonzero object identifiers rather than renumbering
connected components. This maintains correspondence with measurements,
tracks, and crops keyed by those identifiers.

Saving also writes the artifact's :class:`spacr.curation.CurationLog`
sidecar, consistent with :mod:`spacr.napari_bridge` and
:mod:`spacr.qt.curation_tool`. The sidecar allows
:func:`spacr.curation.is_curated` to distinguish manually edited masks from
pipeline-generated masks.

Where a field's mask lives
--------------------------

The editor opens all three layouts :mod:`spacr.curation_queue` reads, and
edits each one in place rather than converting it:

``nested``
    ``<folder>/<image>`` with the mask at ``<folder>/masks/<stem>.tif``. The
    default, and what every function below does when told nothing else.
``sibling``
    ``<root>/images/<image>`` with the mask at ``<root>/masks/<stem>.tif``.
    The editor opens ``<root>/images`` and passes ``masks_dir=<root>/masks``;
    without it the mask would be read from and written to
    ``<root>/images/masks``, a folder the set does not have.
``seg``
    ``<folder>/<stem>_seg.npy``, a Cellpose bundle holding the image and the
    labels in one pickled dict. The editor's file list names the bundles
    themselves, and a name ending in :data:`SEG_SUFFIX` is read and written
    as a bundle by :func:`load_image_and_mask`, :func:`save_mask`,
    :func:`write_recrop` and :func:`retire_recropped_original`.

In place is sound for both, which is why there is no convert step. A sibling
set differs from a nested one only in where its masks folder is, so passing
that folder is the whole change. A bundle is rewritten with every key it
already had kept, ``masks`` replaced, and the two keys derived from the
masks -- ``outlines`` and ``ismanual`` -- brought up to date with it (see
:func:`save_seg_bundle`); that is what the external curation tool did to the
same files, less its stale outlines. A convert step would have been a second
copy of every field, and the curator would have had to remember which copy
was the truth.
"""
from __future__ import annotations

import csv
import json
import os
from collections import deque
from typing import Dict, List, NamedTuple, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from ..curation import LOG_SUFFIX, CurationLog
from ..curation_queue import SEG_SUFFIX
from ..tiff_io import write_tiff


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

#: What a ledger created by this screen records as having made the edits.
CURATION_SOURCE = "spacr-qt make_masks"

#: Eight-connectivity: two objects touching only at a corner are one blob to
#: the eye and must be one object to the label image too, or a hand-drawn
#: diagonal stroke arrives on disk as a string of separate cells.
def _ndimage():
    """``scipy.ndimage``, imported on first use rather than at module scope.

    THIS MODULE IS ON THE STARTUP PATH. `app.folded_children()` imports every
    fold host to read its `FOLDED_APPS`, and `make_masks` imports this one --
    so a module-level `from scipy.ndimage import ...` put scipy into the
    process before Home had painted. The packaged smoke test asserts Home
    crosses no operation-only import boundary and named scipy for exactly
    that reason.

    :returns: the ``scipy.ndimage`` module.
    """
    from scipy import ndimage

    return ndimage


_EIGHT = np.ones((3, 3), dtype=np.uint8)


def list_images(folder: str) -> List[str]:
    """Return filenames of image files in `folder`, sorted, or []."""
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTS)
    )


def is_seg_bundle(filename) -> bool:
    """Whether ``filename`` names a Cellpose ``_seg.npy`` bundle.

    :param filename: a file name or path.
    :returns: ``True`` when it ends in :data:`SEG_SUFFIX`, which is how a
        ``seg`` queue's fields are named in the editor's file list.
    """
    return os.path.basename(str(filename)).endswith(SEG_SUFFIX)


def field_stem(filename) -> str:
    """The stem a field is known by in ``curate_status.csv``.

    ``os.path.splitext`` gives ``well_A1_seg`` for ``well_A1_seg.npy``, and
    the queue calls that field ``well_A1``; a status row written under the
    first name would never take the field out of the queue.

    :param filename: an image file name, or a ``_seg.npy`` bundle name.
    :returns: the file name without its extension, or without
        :data:`SEG_SUFFIX` for a bundle.
    """
    name = os.path.basename(str(filename))
    if name.endswith(SEG_SUFFIX):
        return name[:-len(SEG_SUFFIX)]
    return os.path.splitext(name)[0]


def masks_folder(folder: str, masks_dir: Optional[str] = None) -> str:
    """Where the masks of the images in ``folder`` are kept.

    :param folder: the folder the editor opened.
    :param masks_dir: the masks folder, when it is not beneath ``folder`` --
        the ``sibling`` layout's ``<root>/masks`` beside ``<root>/images``.
    :returns: ``masks_dir`` when given, else ``<folder>/masks``.
    """
    if masks_dir:
        return os.fspath(masks_dir)
    return os.path.join(folder, "masks")


def _as_field_image(image: np.ndarray, image_path: str) -> np.ndarray:
    """Check one decoded image and bring it to the editor's uint16 grey.

    :param image: the decoded pixels.
    :param image_path: where they came from, for the messages.
    :returns: a 2-D uint16 image.
    :raises ValueError: for an unsupported shape or channel count, or
        non-finite or negative intensities.
    """
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        elif image.shape[2] == 3:
            image = np.dot(
                image, [0.2989, 0.5870, 0.1140]
            )
        else:
            raise ValueError(
                f"Unsupported channel count {image.shape[2]} in {image_path}; "
                "expected grayscale, RGB, or RGBA."
            )
    if image.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {image.shape} in {image_path}; "
            "Make Masks expects one 2-D field."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Image contains non-finite values: {image_path}")
    if image.size and float(image.min()) < 0:
        raise ValueError(f"Image contains negative intensities: {image_path}")
    if image.dtype != np.uint16:
        max_val = float(image.max()) if image.size else 1.0
        if max_val <= 0:
            max_val = 1.0
        image = (image / max_val * 65535.0).astype(np.uint16)
    return image


def _as_field_mask(mask: np.ndarray, shape, mask_path: str,
                   filename: str) -> np.ndarray:
    """Check one decoded label image against the image it belongs to.

    :param mask: the decoded labels.
    :param shape: the 2-D shape of the image they label.
    :param mask_path: where they came from, for the messages.
    :param filename: the field's name, for the shape-mismatch message.
    :returns: the labels as uint8 or uint16, whichever holds the largest id.
    :raises ValueError: for a mask that is not 2-D, does not match the
        image, holds non-integer or negative labels, or needs more than 16
        bits.
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    if mask.ndim != 2:
        raise ValueError(
            f"Unsupported mask shape {mask.shape} in {mask_path}; "
            "expected a 2-D label image."
        )
    if mask.shape != tuple(shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape "
            f"{tuple(shape)} for {filename}."
        )
    if not np.issubdtype(mask.dtype, np.integer):
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"Mask contains non-finite values: {mask_path}")
        if np.any(mask < 0) or np.any(mask != np.floor(mask)):
            raise ValueError(
                f"Mask must contain non-negative integer labels: {mask_path}"
            )
    maximum = int(mask.max()) if mask.size else 0
    if maximum > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Mask label {maximum} exceeds uint16 capacity: {mask_path}"
        )
    return mask.astype(np.uint8 if maximum <= 255 else np.uint16)


def load_image_and_mask(folder: str, filename: str,
                        masks_dir: Optional[str] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and its accompanying mask.

    - Multi-channel images are collapsed to grayscale via BT.601 weights.
    - Missing masks are created as zeros of the image shape.
    - Images are returned as uint16; masks preserve uint8/uint16 label IDs.
    - A mask saved by :func:`save_mask` is found even when the source image
      had a non-TIFF extension.
    - A ``filename`` ending in :data:`SEG_SUFFIX` is a Cellpose bundle and is
      read by :func:`load_seg_bundle` instead.

    :param folder: the folder holding the image, or the bundle.
    :param filename: the image, or the ``_seg.npy`` bundle, to load.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``;
        see :func:`masks_folder`.
    :returns: ``(image, mask)``.
    :raises ValueError: for unsupported dimensions or an image/mask shape
        mismatch.
    """
    if is_seg_bundle(filename):
        return load_seg_bundle(os.path.join(folder, filename))
    image_path = os.path.join(folder, filename)
    image = _as_field_image(imageio.imread(image_path), image_path)

    mask_dir = masks_folder(folder, masks_dir)
    stem = os.path.splitext(filename)[0]
    candidates = [
        os.path.join(mask_dir, filename),
        os.path.join(mask_dir, stem + ".tif"),
        os.path.join(mask_dir, stem + ".tiff"),
    ]
    mask_path = next((path for path in candidates if os.path.isfile(path)), "")
    if mask_path:
        mask = _as_field_mask(imageio.imread(mask_path), image.shape,
                              mask_path, filename)
    else:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
    return image, mask


def read_seg_bundle(path: str) -> Dict:
    """Read a Cellpose ``_seg.npy`` bundle as the dict it holds.

    A bundle is a pickle, and unpickling runs whatever the file says, so a
    bundle is only ever read from a queue folder the curator chose -- the
    same trust the external curation tool and Cellpose itself extend to it.

    :param path: the bundle.
    :returns: the bundle's dict, with every key it was written with.
    :raises ValueError: when the file does not hold a dict with a ``masks``
        entry.
    """
    loaded = np.load(path, allow_pickle=True)
    try:
        payload = loaded.item()
    except (AttributeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or "masks" not in payload:
        raise ValueError(
            f"{path} is not a Cellpose _seg.npy bundle: expected a dict "
            f"holding 'masks'.")
    return payload


def _bundle_image(path: str, payload: Dict, shape) -> Tuple[np.ndarray, str]:
    """Find the pixels a bundle's labels were drawn on.

    In the order the external curation tool used:

    1. the original the bundle names in ``source_image``, looked for BY
       NAME where that tool looked for it -- ``new_originals/`` and then
       ``training_data/`` beside the queue folder, then beside the bundle
       itself -- and used only when its shape matches the labels. The
       stored path itself is never touched, which is the one step of that
       tool's search left out: it is absolute and from whichever machine
       staged the set, and a stat on another machine's mount can hang the
       thread that asked;
    2. the ``img`` the bundle carries, often an 8-bit display copy of that
       original;
    3. an image of the bundle's own stem beside it, the display copy the
       external tool writes next to each bundle.

    :param path: the bundle.
    :param payload: its dict, from :func:`read_seg_bundle`.
    :param shape: the shape of its labels.
    :returns: ``(pixels, where they came from)``.
    :raises ValueError: when none of the three exists.
    """
    folder = os.path.dirname(path)
    source = payload.get("source_image")
    if source:
        name = os.path.basename(str(source))
        project = os.path.dirname(os.path.abspath(folder))
        for original in (os.path.join(project, "new_originals", name),
                         os.path.join(project, "training_data", name),
                         os.path.join(folder, name)):
            if not os.path.isfile(original):
                continue
            try:
                pixels = np.asarray(imageio.imread(original))
            except Exception:
                continue
            if pixels.shape[:2] == tuple(shape)[:2]:
                return pixels, original
    embedded = payload.get("img")
    if embedded is not None:
        pixels = np.asarray(embedded)
        if (pixels.ndim == 3 and pixels.shape[1:] == tuple(shape)[:2]
                and pixels.shape[:2] != tuple(shape)[:2]):
            pixels = np.moveaxis(pixels, 0, -1)
        return pixels, path
    stem = field_stem(path)
    for ext in IMAGE_EXTS:
        beside = os.path.join(folder, stem + ext)
        if os.path.isfile(beside):
            return np.asarray(imageio.imread(beside)), beside
    raise ValueError(
        f"{path} carries no image ('img') and there is no {stem}.<ext> beside "
        f"it, so there is nothing to draw its labels on.")


def load_seg_bundle(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Cellpose bundle as the editor's ``(image, mask)`` pair.

    :param path: the ``_seg.npy`` bundle.
    :returns: ``(image, mask)``, checked and converted exactly as
        :func:`load_image_and_mask` converts a TIFF pair.
    :raises ValueError: for a file that is not a bundle, a bundle with no
        image to show, or labels that do not fit the image.
    """
    payload = read_seg_bundle(path)
    labels = np.asarray(payload["masks"])
    pixels, source = _bundle_image(path, payload, labels.shape)
    image = _as_field_image(pixels, source)
    return image, _as_field_mask(labels, image.shape, path,
                                 os.path.basename(path))


def seg_outlines(labels: np.ndarray, like) -> np.ndarray:
    """The ``outlines`` entry of a bundle, redrawn for ``labels``.

    A pixel is on an outline when it belongs to an object and one of its four
    neighbours does not belong to the same one. ``like`` is the entry the
    bundle had: an entry holding only 0 and 1 gets a 0/1 outline back, and
    one holding ids gets each outline pixel's id, which is what Cellpose's
    own ``masks_flows_to_seg`` writes. An entry of all zeros says nothing
    about its form and gets ids, Cellpose's default.

    :param labels: the labels being saved.
    :param like: the bundle's previous ``outlines``, for its form and type.
    :returns: the new outlines, the shape of ``labels``.
    """
    labels = np.asarray(labels)
    old = np.asarray(like)
    padded = np.pad(labels, 1, mode="edge")
    height, width = labels.shape
    edge = np.zeros(labels.shape, dtype=bool)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        edge |= padded[dy:dy + height, dx:dx + width] != labels
    edge &= labels > 0
    peak = float(np.max(old)) if old.size else 0.0
    binary = old.dtype == bool or 0.0 < peak <= 1.0
    if binary:
        return edge.astype(old.dtype)
    outlined = np.where(edge, labels, 0)
    if np.issubdtype(old.dtype, np.integer) and \
            int(labels.max(initial=0)) <= np.iinfo(old.dtype).max:
        return outlined.astype(old.dtype)
    return outlined.astype(labels.dtype)


def _write_bundle(path: str, payload: Dict) -> None:
    """Write a bundle's dict so that a crash leaves the old file whole.

    Written beside the target under a dot-name the queue does not list,
    then renamed over it. ``np.save`` is given a handle rather than a path
    because, given a path, it appends ``.npy`` to one that does not end in
    it.

    :param path: the bundle to write.
    :param payload: the dict to put in it.
    """
    folder, name = os.path.split(path)
    temporary = os.path.join(folder, f".{name}.tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, payload, allow_pickle=True)
    os.replace(temporary, path)


def save_seg_bundle(path: str, mask: np.ndarray, *, preserve_ids: bool = False) -> str:
    """Write edited labels back into the bundle they came from.

    Every key the bundle already had is kept -- ``img``, ``flows``,
    ``filename``, ``source_image``, ``diameter``, and any other -- and
    ``masks`` is replaced by :func:`canonical_labels` of ``mask``. Two keys
    are DERIVED from the masks and would describe the old ones if left:

    * ``outlines`` is redrawn by :func:`seg_outlines`;
    * ``ismanual``, one flag per object, is resized to the new largest id:
      an id the bundle already flagged keeps its flag, and an id beyond the
      old list was drawn in this editor, so it is flagged manual.

    :param path: the ``_seg.npy`` bundle.
    :param mask: the edited labels.
    :param preserve_ids: keep every supplied label exactly, including a lone
        ID or disconnected pieces with the same ID. Labels must fit uint16.
    :returns: ``path``.
    :raises ValueError: when ``path`` is not a bundle.
    """
    payload = read_seg_bundle(path)
    labels = canonical_labels(mask, preserve_ids=preserve_ids)
    payload["masks"] = labels
    outlines = payload.get("outlines")
    if outlines is not None and np.shape(outlines) == labels.shape:
        payload["outlines"] = seg_outlines(labels, outlines)
    manual = payload.get("ismanual")
    if manual is not None and np.ndim(manual) == 1:
        old = np.asarray(manual, dtype=bool)
        count = int(labels.max(initial=0))
        flags = np.ones(count, dtype=bool)
        keep = min(count, old.size)
        flags[:keep] = old[:keep]
        payload["ismanual"] = flags
    _write_bundle(path, payload)
    return path


def mask_save_path(folder: str, filename: str,
                   masks_dir: Optional[str] = None) -> str:
    """Where this field's mask is written -- and where its ledger sits.

    :func:`load_image_and_mask` will accept a mask under the image's own
    extension, but everything :func:`save_mask` writes lands on
    ``<masks folder>/<stem>.tif``. The ledger is keyed on the file that was
    actually written, so both have to agree on one name; ask here rather
    than rebuilding it at each call site.

    :param folder: the folder the editor opened.
    :param filename: the field's image, or its ``_seg.npy`` bundle.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: the mask's path; for a bundle, the bundle itself, which is
        where its labels are written back.
    """
    if is_seg_bundle(filename):
        return os.path.join(folder, os.path.basename(str(filename)))
    stem = os.path.splitext(filename)[0]
    return os.path.join(masks_folder(folder, masks_dir), stem + ".tif")


#: The curation CSV's columns, in a fixed order. They are written on every
#: rewrite, so a reader never has
#: to guess which column is which.
CURATION_COLUMNS: Tuple[str, ...] = (
    "image path", "mask path", "object count", "keep",
)

#: The file itself. One per folder of images rather than one per field: a
#: verdict is only useful beside the others taken in the same sitting.
CURATION_CSV_NAME = "keep_discard.csv"


def curation_folder(folder: str) -> str:
    """Where a folder of images keeps its curation files.

    spaCR's layout: masks live at ``<images>/masks`` and the curation CSV at
    ``<images>/csv``. It is a folder rather than a
    file beside the images so that a folder listing of the fields is still
    a listing of the fields.

    :param folder: the folder the editor opened.
    :returns: ``<folder>/csv``.
    """
    return os.path.join(folder, "csv")


def curation_csv_path(folder: str) -> str:
    """The keep/discard CSV for the images in ``folder``.

    :param folder: the folder the editor opened.
    :returns: ``<folder>/csv/keep_discard.csv``.
    """
    return os.path.join(curation_folder(folder), CURATION_CSV_NAME)


def read_curation(folder: str) -> Dict[str, Dict[str, str]]:
    """Every verdict recorded for ``folder``, keyed by image path.

    A file that is missing, empty or unreadable is NO VERDICTS rather than
    an error: this is a curation aid, and refusing to open a folder because
    its CSV was edited by hand would be the wrong trade.

    :param folder: the folder the editor opened.
    :returns: image path -> the row, as strings.
    """
    path = curation_csv_path(folder)
    rows: Dict[str, Dict[str, str]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get(CURATION_COLUMNS[0]) or "").strip()
                if key:
                    rows[key] = dict(row)
    except (OSError, csv.Error, UnicodeDecodeError):
        return {}
    return rows


def curation_verdict(folder: str, image_path: str) -> Optional[bool]:
    """Whether this field is marked keep, discard, or not marked at all.

    :param folder: the folder the editor opened.
    :param image_path: the field.
    :returns: True for keep, False for discard, None for no row.
    """
    row = read_curation(folder).get(os.fspath(image_path))
    if row is None:
        return None
    value = str(row.get(CURATION_COLUMNS[3], "")).strip().lower()
    if value in ("true", "1", "yes", "keep"):
        return True
    if value in ("false", "0", "no", "discard"):
        return False
    return None


def record_curation(folder: str, image_path: str, mask_path: str,
                    object_count: int, keep: bool) -> str:
    """Record one verdict, replacing any the field already had.

    ONE ROW PER FIELD. Keep and then Discard on the same image leaves the
    later verdict and nothing else, because two rows that disagree are
    worse than no file -- whoever reads it downstream would have to guess
    which press came last, and a CSV does not say.

    Written to a dot-name in the same folder and renamed over the target,
    so a reader never sees half a file and a crash leaves the old one
    whole. The same shape as :func:`_write_bundle` above.

    :param folder: the folder the editor opened.
    :param image_path: the field being judged.
    :param mask_path: its mask, from :func:`mask_save_path`.
    :param object_count: how many objects the mask holds right now.
    :param keep: True for Keep, False for Discard.
    :returns: the CSV's path.
    """
    rows = read_curation(folder)
    rows[os.fspath(image_path)] = {
        CURATION_COLUMNS[0]: os.fspath(image_path),
        CURATION_COLUMNS[1]: os.fspath(mask_path),
        CURATION_COLUMNS[2]: str(int(object_count)),
        CURATION_COLUMNS[3]: "true" if keep else "false",
    }
    destination = curation_csv_path(folder)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    directory, name = os.path.split(destination)
    temporary = os.path.join(directory, f".{name}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CURATION_COLUMNS),
                                extrasaction="ignore")
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({column: rows[key].get(column, "")
                             for column in CURATION_COLUMNS})
    os.replace(temporary, destination)
    return destination


def canonical_labels(mask: np.ndarray, *, preserve_ids: bool = False) -> np.ndarray:
    """Return ``mask`` as uint16 labels, keeping every id it already had.

    The old behaviour here was ``label(mask > 0)``, which renumbers the
    connected components 1..N on every save. That throws away the identity
    of every object: erase object 7 of 20 and objects 8..20 each slide down
    by one, so the saved mask no longer keys against the measurements, the
    crops or the tracks derived from the segmentation it was edited from.

    So ids are kept, with two things settled:

    * **A mask with one foreground value carries no ids to keep.** That is
      what a purely brush-painted mask looks like -- every stroke writes the
      same value -- and it is a binary image, not a label image. Its
      components are numbered 1..N, which loses nothing.
    * **A label that names two separated blobs is split.** One id must mean
      one object. The largest piece keeps the id and the rest are given the
      smallest ids not already in use, so painting a second blob with the
      brush over a real segmentation adds an object instead of extending a
      distant one.

    Each id is examined inside its own bounding box
    (:func:`scipy.ndimage.find_objects`) rather than across the whole field,
    which gives the same pieces in the same order and makes the call cheap
    enough to run while the mouse moves: on a 2048 x 2048 field of 400
    objects it went from about 3.5 s to tens of milliseconds.

    :param mask: a label image; any integer or boolean dtype in default mode.
    :param preserve_ids: disable binary interpretation and component splitting.
        Use for primary/secondary relationships: a lone cell 900 remains 900,
        and separated pieces with the same primary ID remain one label.
        Requires a nonempty 2-D nonnegative integer array, not a boolean mask.
    :returns: the labels as ``uint16``.
    :raises ValueError: when an id does not fit in ``uint16``, or exact-ID
        mode receives invalid labels. Oversized IDs are never truncated.
    """
    m = np.asarray(mask)
    if preserve_ids:
        m = _primary_label_image(m, "Mask")
        if int(m.max()) > np.iinfo(np.uint16).max:
            raise ValueError("Exact mask IDs must fit in uint16; labels were not renumbered.")
        return m.astype(np.uint16, copy=True)
    boxes = None
    top = int(m.max()) if m.size and np.issubdtype(m.dtype, np.integer) \
        else None
    if top is not None and top <= np.iinfo(np.uint16).max:
        boxes = _ndimage().find_objects(m) if top > 0 else []
        values = [index + 1 for index, box in enumerate(boxes)
                  if box is not None]
    else:
        values = list(np.unique(m[m > 0]))
    if len(values) <= 1:
        labeled, count = _ndimage().label(m > 0, structure=_EIGHT)
        if count > np.iinfo(np.uint16).max:
            raise ValueError(
                f"mask needs {count} object labels, past what a uint16 mask can hold.")
        return labeled.astype(np.uint16)

    largest = int(max(values))
    if largest > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask carries label {largest}, past what a uint16 mask can hold.")
    whole = tuple(slice(None) for _axis in range(m.ndim))
    out = None
    used = {int(v) for v in values}
    candidate = 1
    for value in values:
        box = boxes[int(value) - 1] if boxes is not None else whole
        pieces, count = _ndimage().label(m[box] == value, structure=_EIGHT)
        if count <= 1:
            continue
        areas = np.bincount(pieces.ravel())
        keep = int(np.argmax(areas[1:])) + 1
        if out is None:
            out = m.astype(np.int64, copy=True)
        region = out[box]
        for piece in range(1, count + 1):
            if piece == keep:
                continue
            while candidate in used:
                candidate += 1
            region[pieces == piece] = candidate
            used.add(candidate)
    if out is None:
        out = m if boxes is not None else m.astype(np.int64)
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask carries label {top}, past what a uint16 mask can hold.")
    return out.astype(np.uint16)


def save_mask(folder: str, filename: str, mask: np.ndarray,
              log: Optional[CurationLog] = None,
              masks_dir: Optional[str] = None, *, preserve_ids: bool = False) -> str:
    """Write the mask to ``<folder>/masks/<stem>.tif`` and return that path.

    Object ids are preserved -- see :func:`canonical_labels` for what that
    costs and why the alternative is worse. A ``_seg.npy`` bundle is written
    back into itself by :func:`save_seg_bundle`, and ``masks_dir`` moves the
    TIFF to the sibling layout's masks folder.

    :param folder: field directory under which the ``masks`` directory is
        created.
    :param filename: source image name; its extension is discarded and its
        stem becomes the TIFF mask name. A name ending in
        :data:`SEG_SUFFIX` is the bundle to write into.
    :param mask: label image to canonicalise and write. Existing multi-label
        object identifiers are retained where possible.
    :param log: the session's :class:`spacr.curation.CurationLog` for this
        field. Given one holding at least one edit, it is written to
        ``<mask>.curation.json`` beside the mask, so
        :func:`spacr.curation.is_curated` reports the saved mask as
        hand-edited. The log is written whole, so it must have been seeded
        from :meth:`CurationLog.read_beside` to keep earlier sessions'
        entries -- which is what the screen does on load. A log with no
        edits writes no sidecar: a session that opened the editor and
        painted nothing has not curated anything, and a ledger that exists
        for every mask ever opened answers no question.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :param preserve_ids: retain exact primary/secondary IDs without interpreting
        single-valued masks as binary or splitting disconnected pieces. The
        explicit mode validates nonnegative 2-D integer labels and refuses
        IDs above 65535 before writing; it never silently renumbers them.
    :returns: the path written.
    """
    save_path = mask_save_path(folder, filename, masks_dir)
    if is_seg_bundle(filename):
        save_seg_bundle(save_path, mask, preserve_ids=preserve_ids)
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        write_tiff(save_path, canonical_labels(mask, preserve_ids=preserve_ids))
    if log is not None and len(log):
        if not log.artifact:
            log.artifact = save_path
        log.write_beside(save_path)
    return save_path


def normalize_uint16(image: np.ndarray,
                     lower_pct: float = 1.0,
                     upper_pct: float = 99.9) -> np.ndarray:
    """Return image clipped + rescaled to its dtype's full range."""
    if not image.size:
        return image
    lo = np.percentile(image, lower_pct)
    hi = np.percentile(image, upper_pct)
    if hi <= lo:
        hi = lo + 1
    out = np.clip(image, lo, hi)
    out = (out - lo) / (hi - lo)
    max_val = float(np.iinfo(image.dtype).max)
    return (out * max_val).astype(image.dtype)


def normalize_for_detection(image: np.ndarray, lower_pct: float = 1.0,
                            upper_pct: float = 99.9) -> np.ndarray:
    """``image`` stretched between two percentiles, as Make Masks draws it.

    :func:`normalize_uint16` for an integer field, so a detector reads the
    exact numbers the canvas paints; a float field has no integer range to
    fill and comes back on 0..1 instead.

    :param image: the field.
    :param lower_pct: the percentile mapped to the bottom of the range.
    :param upper_pct: the percentile mapped to the top.
    :returns: the stretched field, the same shape.
    """
    if np.issubdtype(image.dtype, np.integer):
        return normalize_uint16(image, lower_pct, upper_pct)
    if not image.size:
        return image
    lo = float(np.percentile(image, lower_pct))
    hi = float(np.percentile(image, upper_pct))
    if hi <= lo:
        hi = lo + 1.0
    return ((np.clip(image, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def invert_normalized(image: np.ndarray) -> np.ndarray:
    """Normalise ``image`` to 0..1 on its OWN range, take ``1 - v``, fit back.

    THE ONE INVERSION. The field is normalised to 0..1 and every pixel
    becomes ``1 - value``, which gives an image that Otsu and the magnifier
    can work on when the objects are dark. That is the reason for the
    button.

    So the purpose is a DETECTOR reading dark objects, and the picture the
    curator sees has to be the picture the detector reads -- one switch, one
    meaning. That decision replaced the two inversions this module used to
    carry for Make Masks, :func:`invert_intensity` (the dtype complement,
    drawn but never detected on) and :func:`invert_for_detection`
    (detected on but never drawn). Both are kept for
    callers outside Make Masks and neither is what the screen uses now.

    WHY NORMALISING FIRST IS THE POINT AND NOT A DETAIL. The Otsu threshold
    correction is a MULTIPLIER on an absolute level, so what it means depends
    on where the field's intensities sit. A dtype complement moves a 12-bit
    field (216..4095) up into 61440..65535, and a correction of 0.8 then asks
    for a cut below every pixel present while 1.3 asks for one above them all
    -- the dial becomes an on/off switch, which is the measurement recorded in
    :func:`invert_for_detection`. Normalising to the field's own range first
    puts EVERY field on the same 0..1 span before the multiplier is applied,
    so one correction value means the same thing on the next image.

    WHAT IS GIVEN UP, said plainly: this is not exactly reversible on the
    original numbers the way the dtype complement was. Inverting a field
    rescales it to the full range, and the original span cannot be recovered
    from the result. It does not need to be -- the screen keeps the untouched
    array and re-derives this one, so nothing measured, filtered or saved ever
    sees it -- but a caller that inverts an array and keeps only the result
    has lost where it sat.

    THE RETURN DTYPE IS THE INPUT'S, because the display path hands the result
    to :func:`normalize_uint16`, which reads ``np.iinfo`` and raises on a
    float. An integer field comes back spanning that dtype's full range; a
    float field comes back in 0..1, where a float already belongs.

    A FLAT FIELD has no range to normalise onto. ``v - min`` is 0 everywhere,
    so the normalised value is taken as 0 and the inverse as 1: a flat field
    inverts to a flat bright one, which is the literal reading of the formula
    and is what an inversion of "no contrast" should look like.

    :param image: the field, of any shape and any real dtype.
    :returns: a new array, same shape, same dtype, inverted.
    """
    array = np.asarray(image)
    if not array.size:
        return array.copy()
    if array.dtype == np.bool_:
        return ~array
    low = float(array.min())
    high = float(array.max())
    span = high - low
    unit = (np.zeros(array.shape, np.float64) if span <= 0
            else (array.astype(np.float64) - low) / span)
    flipped = 1.0 - unit
    if np.issubdtype(array.dtype, np.floating):
        return flipped.astype(array.dtype)
    info = np.iinfo(array.dtype)
    return (info.min + flipped * (float(info.max) - float(info.min))
            ).astype(array.dtype)


def invert_intensity(image: np.ndarray) -> np.ndarray:
    """Return the photographic complement of ``image``: dark becomes bright.

    Low intensity becomes high intensity and vice versa, fitted to the
    dtype. WHAT IS BUILT IS THE COMPLEMENT, ``dtype_max - value``,
    NOT THE RECIPROCAL, for three reasons that are worth having written
    down because the reciprocal is the obvious first thought:

    * it is what every image viewer means by Invert, so the picture that
      comes back is the one a reader expects from Invert;
    * it is EXACTLY reversible on an integer field -- inverting twice
      returns the identical array, which is what makes it safe to leave
      switched on while curating, and is asserted by comparing arrays;
    * ``1/value`` divides by zero on every background pixel, and it squashes
      the bright end non-linearly, so two objects a thousand counts apart
      come back indistinguishable while the background explodes.

    The reciprocal remains a reasonable SECOND mode for anyone who wants a
    log-like lift of the dim end; it is not this one.

    WHICH RANGE IS COMPLEMENTED depends on the dtype, because fitting to
    the dtype only has a meaning where the dtype has ends:

    ``unsigned integers``
        the dtype's own range, so a ``uint16`` field is ``65535 - value``.
        A 12-bit camera writing into ``uint16`` therefore comes back in the
        top sixteenth of the range; a display that stretches by percentiles
        puts that back where a reader can see it, and the array is still
        exactly invertible, which a data-range complement would not be
        across two fields of different brightness.
    ``signed integers``
        ``iinfo.min + iinfo.max - value``, the same complement on the range
        the dtype actually spans.
    ``bool``
        logical not.
    ``floating point``
        the ARRAY'S OWN range, ``min + max - value``, because a float image
        has no dtype maximum worth speaking of. The complement of a range
        maps its ends onto each other, so a second call computes the same
        two ends and comes back to the original -- but NOT bit for bit:
        ``s - (s - x)`` rounds twice, and the round trip is out by up to one
        unit in the last place of ``s``. Measured on a 200x200 field over
        0..65535: 0.002 in ``float32`` and 4e-12 in ``float64``, against an
        interval of one count. The round trip is EXACT for every integer and
        boolean dtype, which is every dtype a field is read in.

    :param image: any numeric or boolean array. It is not modified.
    :returns: a new array of the same shape and dtype.
    """
    values = np.asarray(image)
    if not values.size:
        return values.copy()
    kind = values.dtype.kind
    if kind == "b":
        return np.logical_not(values)
    if kind in "ui":
        info = np.iinfo(values.dtype)
        span = int(info.min) + int(info.max)
        return (span - values.astype(np.int64)).astype(values.dtype)
    if kind in "fc":
        span = values.min() + values.max()
        return (span - values).astype(values.dtype, copy=False)
    raise TypeError(
        f"invert_intensity needs a numeric or boolean image; got dtype "
        f"{values.dtype!r}.")


def invert_for_detection(image: np.ndarray, *, bounds=None) -> np.ndarray:
    """Reflect an image about its OWN range, for a DETECTOR to read.

    Masks are generated from the inverted image so a threshold written for
    bright objects can take dark ones. It is ``max + min - value`` on the
    field's own extremes.

    WHY THIS IS NOT :func:`invert_intensity`, which is the other inversion
    in this module and is one line away. The difference is not taste and it
    is not a duplicate that wants merging -- the two are read by different
    things and only one of them can afford to move the numbers:

    * :func:`invert_intensity` complements the DTYPE and is what "Invert
      image" draws with. It has to be exactly reversible, because a curator
      leaves it on all day, and nothing downstream reads its result.
    * this one is read by a THRESHOLD, and the Otsu threshold correction
      is a MULTIPLIER on the level Otsu finds, applied to absolute intensity
      in :func:`_otsu_levels`. Multiplying is not invariant to an offset, so
      an inversion that moves the field's span moves what the correction
      means. Measured on a 12-bit field (216..4095) with dark objects,
      inverted and put through :func:`_otsu_instances` on the bright side:

      =============  ==========================  =======================
      correction     dtype complement            reflection about range
      =============  ==========================  =======================
      0.8            1 object, 100% of the       47 objects, 20%
                     field -- everything
      1.0            3 objects, 8%               3 objects, 8%
      1.3            0 objects, 0% -- nothing    3 objects, 8%
      =============  ==========================  =======================

      The complement puts that field into 61440..65535, so a correction of
      0.8 asks for a cut at about 49000, below every pixel there is, and 1.3
      asks for one above all of them. The correction stops being a dial and
      becomes an on/off switch. At exactly 1.0 the two agree, which is why
      this is easy to miss.

    The other two candidates were considered and are worse. The DTYPE's
    maximum is the case above. The CONTRAST-STRETCHED view is a viewing
    choice, and a detector reading it would move when the percentiles moved,
    which is the argument :func:`filter_objects` already makes about
    intensity bounds.

    Reflecting about the image's own extremes keeps the span exactly, maps
    the darkest pixel onto the brightest and back, and is its own inverse on
    an image whose extremes it has not changed.

    Nonfinite pixels take no part in finding the extremes and are returned
    unchanged, since a NaN is not dark and is not bright.

    :param image: any 2-D field, as the canvas holds it.
    :param bounds: ``(lo, hi)`` to reflect about, instead of ``image``'s own
        extremes. WHAT A CROP IS GIVEN: a region inverted about its own
        extremes is inverted differently wherever the box is put, so the
        magnifier hands it the whole field's pair and the box stays a
        preview of what the detect button will do with the same setting.
        (:func:`invert_intensity` needs no such thing, being a function of
        the pixel value alone -- another way the two differ.)
    :returns: a NEW array of the input's dtype. The original is never
        touched: the readout and the filter must keep reporting the raw
        values whatever the detector was shown.
    """
    arr = np.asarray(image)
    if not arr.size:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.integer):
        work = arr.astype(np.int64)
        if bounds is None:
            lo, hi = int(work.min()), int(work.max())
        else:
            lo, hi = int(bounds[0]), int(bounds[1])
        return np.clip(hi + lo - work, lo, hi).astype(arr.dtype)
    work = arr.astype(np.float64)
    finite = np.isfinite(work)
    if not finite.any():
        return arr.copy()
    if bounds is None:
        lo = float(work[finite].min())
        hi = float(work[finite].max())
    else:
        lo, hi = float(bounds[0]), float(bounds[1])
    out = work.copy()
    out[finite] = np.clip(hi + lo - work[finite], lo, hi)
    return out.astype(arr.dtype)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colorized label mask onto a grayscale image, uint8 RGB."""
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)
    m = mask.astype(np.int32)
    max_label = int(np.max(m)) if m.size else 0
    rng = np.random.default_rng(0)
    colors = rng.integers(30, 255, size=(max_label + 1, 3), dtype=np.uint8)
    if max_label >= 0:
        colors[0] = [0, 0, 0]
    colored = colors[m]
    image_8bit = (image.astype(np.float32) / 256.0).clip(0, 255).astype(np.uint8)
    combined = np.where(
        m[..., None] > 0,
        np.clip(image_8bit * (1 - alpha) + colored * alpha, 0, 255),
        image_8bit,
    ).astype(np.uint8)
    return combined



def paint_disk(mask: np.ndarray, cx: int, cy: int, radius: int,
               value: int = 255) -> None:
    """In-place stamp a filled square (radius half-width) at (cx, cy)."""
    if radius < 1:
        radius = 1
    h, w = mask.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = value


def paint_line(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int,
               radius: int, value: int = 255) -> None:
    """In-place stamp a line of disks between two points (Bresenham)."""
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        paint_disk(mask, x, y, radius, value)
        if x == x1 and y == y1:
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy



#: Width, in image pixels, of the cut a divide draws through an object.
#:
#: Not cosmetic. Everything here calls two pixels that touch only at a
#: corner one object (:data:`_EIGHT`), so a one-pixel-wide cut across a
#: diagonal leaves the two halves corner-to-corner and they are still one
#: blob — the split does not take. Measured on a disk over 60 orientations:
#: a 1.0 px cut failed to separate in 58 of them, 1.5 px separated in all
#: 1800 orientation/offset combinations tried, and anything wider only eats
#: more of the object (1.5 px costs ~90 px of a 2800 px object, 2.0 px ~120).
DIVIDE_CUT_WIDTH = 1.5


def next_label(mask: np.ndarray) -> int:
    """The id to give the next object drawn on ``mask``: one past its top.

    Above the maximum rather than the lowest free id, because ids are what
    the ledger, the measurements and the crops name objects by. Handing a
    new object the id of one that was deleted makes two different cells
    share a name across a session, and nothing downstream can tell them
    apart afterwards.
    """
    return (int(mask.max()) if mask.size else 0) + 1


def _fit_label_width(out: np.ndarray, like: np.ndarray) -> np.ndarray:
    """Cast a working int64 mask back down, widening only when it must.

    Same rule as :func:`combine_masks`: the width follows the values. Keeping
    uint8 for a mask that has just been given label 256 would wrap it round
    to 0 and the new object would vanish into the background.
    """
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"mask needs label {top}, past what a uint16 mask can hold.")
    if np.issubdtype(like.dtype, np.integer) and top <= np.iinfo(like.dtype).max:
        return out.astype(like.dtype)
    return out.astype(np.uint8 if top <= 255 else np.uint16)


def fill_polygon(mask: np.ndarray, points, label_value: Optional[int] = None):
    """Fill a traced outline as ONE object; return ``(mask, label)``.

    This is the tool a brush is not. A brush stamps disks along the path, so
    tracing a cell's rim with it labels the rim and leaves the middle
    background; ``draw`` closes the path (last point back to the first) and
    fills what it encloses, so one gesture produces one solid object with
    one id. Anything already labelled inside the outline is overwritten,
    which is the point: the outline asserts "all of this is one object".

    :param mask: existing label image to copy and edit; labels outside the
        enclosed pixels are preserved and the dtype widens when the new id
        requires it.
    :param points: the traced path as image-pixel ``(x, y)`` pairs. A path
        that encloses less than one pixel -- two points, or a straight line
        traced back over itself -- is returned unchanged with label 0. It is
        a gesture that enclosed nothing, and the alternative is an object a
        pixel wide that the user then has to find and delete.
    :param label_value: the id to give it; by default :func:`next_label`.
    """
    from skimage.draw import polygon as _polygon

    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return mask.copy(), 0
    x, y = pts[:, 0], pts[:, 1]
    if abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) < 1.0:
        return mask.copy(), 0
    rows, cols = _polygon(pts[:, 1], pts[:, 0], shape=mask.shape[:2])
    if not len(rows):
        return mask.copy(), 0
    value = int(next_label(mask) if label_value is None else label_value)
    out = mask.astype(np.int64, copy=True)
    out[rows, cols] = value
    return _fit_label_width(out, mask), value


def _segment_band(shape, p0, p1, width: float) -> np.ndarray:
    """Boolean mask of the pixels within ``width``/2 of the segment p0-p1.

    A distance band rather than a rasterised line: the sampled-line cut the
    standalone curation tool uses (400 points between the ends) both leaves
    gaps on a segment longer than 400 px and is one pixel wide wherever it
    lands, and a one-pixel cut does not separate — see
    :data:`DIVIDE_CUT_WIDTH`.
    """
    height, width_px = shape[:2]
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    half = max(0.5, float(width) / 2.0)
    lo_x = max(0, int(np.floor(min(x0, x1) - half)))
    hi_x = min(width_px, int(np.ceil(max(x0, x1) + half)) + 1)
    lo_y = max(0, int(np.floor(min(y0, y1) - half)))
    hi_y = min(height, int(np.ceil(max(y0, y1) + half)) + 1)
    band = np.zeros((height, width_px), dtype=bool)
    if hi_x <= lo_x or hi_y <= lo_y:
        return band
    yy, xx = np.mgrid[lo_y:hi_y, lo_x:hi_x].astype(np.float64)
    dx, dy = x1 - x0, y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        t = np.zeros_like(xx)
    else:
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / length_sq, 0.0, 1.0)
    distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
    band[lo_y:hi_y, lo_x:hi_x] = distance <= half
    return band


def divide_object(mask: np.ndarray, p0, p1,
                  width: float = DIVIDE_CUT_WIDTH):
    """Cut every object the segment crosses in two; return ``(mask, splits)``.

    ``splits`` is a list of ``(id_split, id_created)`` pairs, empty when the
    line separated nothing.

    Three decisions make the result usable:

    * **Only the objects the line actually crosses are touched.** The cut is
      clipped to them, so a line drawn past a neighbour leaves that
      neighbour's every pixel where it was. (The standalone tool relabels
      the whole field after cutting, which renumbers every other object in
      it — the same re-keying :func:`canonical_labels` exists to avoid.)
    * **The larger piece keeps the original id**, and the smaller pieces get
      fresh ones above the mask's top label. That is the rule
      :func:`canonical_labels` already applies when one id names two blobs,
      so dividing and then saving does not renumber anything, and the id
      stays on the piece that carries most of what it used to name.
    * **A line that does not separate an object leaves it alone.** Stopping
      halfway across would otherwise carve a groove into the object and
      call it a division; treating it as a miss means the gesture can just
      be redrawn.
    """
    band = _segment_band(mask.shape, p0, p1, width)
    if not band.any():
        return mask.copy(), []
    crossed = [int(v) for v in np.unique(mask[band]) if int(v) > 0]
    if not crossed:
        return mask.copy(), []
    out = mask.astype(np.int64, copy=True)
    free_id = next_label(mask)
    splits = []
    for source in crossed:
        body = out == source
        remainder = body & ~band
        pieces, count = _ndimage().label(remainder, structure=_EIGHT)
        if count < 2:
            continue
        areas = np.bincount(pieces.ravel())
        keeps = int(np.argmax(areas[1:])) + 1
        out[body] = 0
        out[pieces == keeps] = source
        for piece in range(1, count + 1):
            if piece == keeps:
                continue
            out[pieces == piece] = free_id
            splits.append((source, free_id))
            free_id += 1
    if not splits:
        return mask.copy(), []
    return _fit_label_width(out, mask), splits

def fill_holes(mask: np.ndarray, *, preserve_ids: bool = False) -> np.ndarray:
    """Fill enclosed background pixels, optionally retaining primary identities.

    :param mask: label image to fill.
    :param preserve_ids: fill per object without merging or renumbering labels.
        Requires exact uint16-compatible IDs; default retains binary relabeling.
    """
    if preserve_ids:
        return _fill_label_holes(canonical_labels(mask, preserve_ids=True))
    binary = mask > 0
    filled = _ndimage().binary_fill_holes(binary)
    labeled, _ = _ndimage().label(filled)
    return labeled.astype(mask.dtype)


def relabel_objects(mask: np.ndarray) -> np.ndarray:
    """Return a mask whose connected components are labeled 1..N."""
    labeled, _ = _ndimage().label(mask > 0)
    return labeled.astype(mask.dtype)


def clear_mask(mask: np.ndarray) -> np.ndarray:
    """Return an all-zero array shaped like ``mask``."""
    return np.zeros_like(mask)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Swap object and background in a LABEL image, and relabel.

    NOT AN INTENSITY INVERSION -- that is :func:`invert_intensity`, and the
    Make Masks screen's "Invert image" is wired to that one. This flips the
    MASK: every labelled pixel becomes background and every background pixel
    becomes foreground, and what comes out is then labelled afresh. On an
    ordinary field the background is one connected region, so what comes back
    is a SINGLE field-sized object with holes where the objects were -- which
    is why it looks as if it does not invert at all: one flat overlay over
    the whole frame reads as nothing having happened.

    It is kept because it is a real thing to want -- a curator who has
    outlined the space BETWEEN the cells has drawn the complement of what is
    wanted -- but under the name that says what it does.
    """
    out = np.where(mask > 0, 0, 1).astype(mask.dtype)
    labeled, _ = _ndimage().label(out)
    return labeled.astype(mask.dtype)


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components with area < min_area (in pixels)."""
    if min_area <= 0:
        return mask.copy()
    labeled, n = _ndimage().label(mask > 0)
    if n == 0:
        return mask.copy()
    counts = np.bincount(labeled.ravel())
    keep = np.zeros_like(counts, dtype=bool)
    for i in range(1, len(counts)):
        if counts[i] >= min_area:
            keep[i] = True
    filtered = keep[labeled]
    out = np.where(filtered, mask, 0)
    labeled, _ = _ndimage().label(out > 0)
    return labeled.astype(mask.dtype)


def dilate_objects(mask: np.ndarray, distance: int = 1) -> np.ndarray:
    """Grow every object by ``distance`` pixels, without merging any two.

    A label takes the background pixels within ``distance`` of it; a pixel
    contested by two labels goes to the nearer one, and a pixel that already
    carries a label is never taken. SO THE OBJECT COUNT CANNOT CHANGE, which
    is what makes this safe on a mask that has been curated: ids survive, and
    an object that has been given the right id keeps it, along with every
    measurement, track and crop keyed by it.

    The distance is Euclidean (:func:`skimage.segmentation.expand_labels`),
    so ``1`` adds the four edge neighbours and not the corners -- the same
    metric :func:`shrink_objects` takes away by, which is what makes a shrink
    after a dilate land back where it started on an object with no neighbour
    close enough to have blocked the growth.

    :param mask: label image; 0 is background.
    :param distance: pixels to grow by. 0 or less returns a copy.
    :returns: a mask of the same dtype with the same label values.
    """
    from skimage.segmentation import expand_labels

    out = np.asarray(mask)
    if int(distance) <= 0 or not out.size:
        return out.copy()
    grown = expand_labels(out, distance=float(int(distance)))
    return np.asarray(grown).astype(mask.dtype, copy=False)


def shrink_objects(mask: np.ndarray, distance: int = 1) -> np.ndarray:
    """Erode every object by ``distance`` pixels, each one on its own.

    Each object is eroded against everything that is not itself -- background
    AND the objects touching it -- so two objects sharing a border both pull
    back from it and the seam between them widens. Eroding the foreground as
    one binary would instead leave that seam untouched, which is the opposite
    of what a curator reaching for Shrink wants.

    AN OBJECT THINNER THAN TWICE THE DISTANCE DISAPPEARS. That is what
    erosion means, and the screen says how many went rather than letting them
    go quietly; the edit is one undo step, so the way back is one press.

    The image border counts as background, matching
    :func:`scipy.ndimage.binary_erosion`'s own default, so an object the
    field cut off pulls back from the cut too.

    Each object is eroded inside its own bounding box, so the cost follows
    the area of the objects rather than the area of the field -- the same
    reason :func:`canonical_labels` works in boxes.

    :param mask: label image; 0 is background.
    :param distance: pixels to erode by. 0 or less returns a copy.
    :returns: a mask of the same dtype, holding the ids that survived.
    """
    out = np.array(mask, copy=True)
    steps = int(distance)
    if steps <= 0 or not out.size:
        return out
    ndimage = _ndimage()
    boxes = ndimage.find_objects(out.astype(np.int64, copy=False))
    for value, box in enumerate(boxes, start=1):
        if box is None:
            continue
        window = out[box]
        inside = window == value
        core = tuple(slice(steps, steps + n) for n in inside.shape)
        padded = np.zeros(tuple(n + 2 * steps for n in inside.shape),
                          dtype=bool)
        padded[core] = inside
        distances = ndimage.distance_transform_edt(padded)
        kept = distances[core] > float(steps)
        window[inside & ~kept] = 0
    return out


def erase_object_at(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Zero out the object under (x, y). No-op if no object there."""
    if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
        return mask
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return mask
    out = mask.copy()
    out[out == label_to_remove] = 0
    return out


def erase_object_in_place(mask: np.ndarray, x: int, y: int) -> int:
    """Zero the object under (x, y) *in place*; return the id removed, or 0.

    The copy-and-return :func:`erase_object_at` is right for a single click,
    where one edit is one undo step. A right-button sweep is dozens of move
    events and one thing the user did, so it deletes in place against a
    single pre-sweep snapshot: copying a 16-bit field per mouse event would
    both stutter and put every object of the sweep on its own undo step.

    The returned id is what the ledger records as the sweep's targets.
    """
    height, width = mask.shape[:2]
    if not (0 <= y < height and 0 <= x < width):
        return 0
    label_to_remove = int(mask[y, x])
    if label_to_remove <= 0:
        return 0
    mask[mask == label_to_remove] = 0
    return label_to_remove


def split_object_at(mask: np.ndarray, x: int, y: int, *,
                    min_area: int = 0) -> Tuple[np.ndarray, List[int]]:
    """Cut the object under (x, y) where its halves meet; ``(mask, new_ids)``.

    What Ctrl + left click does. The cut is a watershed on the
    object's own distance to background, the same recipe
    :func:`_split_touching_objects` runs on a whole field: every local
    maximum of that distance is one half's middle and the ridge between two
    of them is the waist where they meet.

    Three decisions, and the first is the one to read:

    * **AN OBJECT WITH ONE CENTRE IS LEFT ALONE** and reported as such,
      rather than being halved through the click. A single click carries no
      direction, so a forced cut would have to invent one, and the object
      that needs cutting is almost always a pair that merged -- which has
      two centres. The gesture for a cut the user aims themselves already
      exists and is the Divide tool (:func:`divide_object`).
    * **The largest piece keeps the id** and the others are given ids above
      the mask's top label, which is :func:`canonical_labels`' own rule, so
      splitting and then saving renumbers nothing.
    * **NO PIXEL IS LOST.** Every pixel of the object ends up under one of
      the new ids. ``min_area`` sets how far apart two centres must be to
      count as two (:func:`_split_touching_objects`' seed spacing) and is
      NOT applied as a drop here: a hand edit moves pixels between ids, and
      a gesture that quietly erased the smaller half would be a delete
      wearing a split's name.

    :param mask: the label image.
    :param x: column clicked, in image pixels.
    :param y: row clicked, in image pixels.
    :param min_area: the smallest object the screen is willing to keep, in
        pixels; it sets the seed spacing, so an object this size is not
        itself cut in two.
    :returns: ``(mask, new_ids)`` -- a new mask and the ids the split
        created, or a copy and an empty list when the click was on
        background, outside the field, or on an object with one centre.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    ndimage = _ndimage()
    height, width = mask.shape[:2]
    if not (0 <= int(y) < height and 0 <= int(x) < width):
        return mask.copy(), []
    target = int(mask[int(y), int(x)])
    if target <= 0:
        return mask.copy(), []

    where = np.argwhere(mask == target)
    y0, x0 = (int(v) for v in where.min(axis=0))
    y1, x1 = (int(v) + 1 for v in where.max(axis=0))
    window = mask[y0:y1, x0:x1]
    body = window == target

    distance = ndimage.gaussian_filter(
        ndimage.distance_transform_edt(body), 1.0)
    spacing = max(2, int(np.sqrt(max(int(min_area), 12) / np.pi)))
    peaks = peak_local_max(distance, min_distance=spacing,
                           labels=body.astype(np.int32),
                           exclude_border=False)
    if len(peaks) < 2:
        return mask.copy(), []
    markers = np.zeros(body.shape, dtype=np.int32)
    for index, point in enumerate(peaks, start=1):
        markers[tuple(point)] = index
    pieces = watershed(-distance, markers, mask=body)
    found = [int(v) for v in np.unique(pieces) if int(v) > 0]
    if len(found) < 2:
        return mask.copy(), []

    areas = {piece: int(np.count_nonzero(pieces == piece)) for piece in found}
    keeps = max(found, key=lambda piece: (areas[piece], -piece))
    out = mask.astype(np.int64, copy=True)
    free_id = next_label(mask)
    new_ids: List[int] = []
    for piece in found:
        if piece == keeps:
            continue
        out[y0:y1, x0:x1][pieces == piece] = free_id
        new_ids.append(free_id)
        free_id += 1
    return _fit_label_width(out, mask), new_ids


def relative_tolerance(image: np.ndarray, percent: float) -> float:
    """Magic-wand tolerance as ``percent`` of ``image``'s intensity range.

    An absolute tolerance is not a portable setting. The value that grabs
    one nucleus in an 8-bit field (range 0..255) selects nothing at all in
    a 16-bit one (range 0..65535), and the one tuned for 16-bit floods the
    entire 8-bit frame. A percentage of *this* image's own range means one
    number behaves the same on both.

    The floor of 1.0 keeps the wand usable on a flat field: a range of zero
    would otherwise give a tolerance of zero, and a tolerance of zero fills
    only pixels exactly equal to the seed.
    """
    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        return 1.0
    span = float(values.max() - values.min())
    return max(1.0, (float(percent) / 100.0) * span)


#: The four bounds :func:`filter_report` judges by, named as its keywords
#: are, in the order it applies them. A :class:`FilterRemoval` names the ones
#: an object failed with these strings, so the screen can put the number the
#: user typed beside the reason without a second vocabulary.
FILTER_BOUNDS = ("min_area", "max_area", "min_intensity", "max_intensity")


class FilterRemoval(NamedTuple):
    """One object the filter dropped, and which bound dropped it.

    The screen reports "object 22 with area x and intensity y was removed
    by minimum intensity", one row per object, so the filter has to say
    more than which ids went.

    :ivar label: the id :func:`canonical_labels` gave the object -- the same
        id the hover readout showed for it.
    :ivar area: its pixel count.
    :ivar mean_intensity: its mean value on the raw image.
    :ivar bounds: the names of every bound it failed, a subset of
        :data:`FILTER_BOUNDS` in that order. USUALLY ONE, and more when an
        object misses on two sides at once -- which is worth saying, because
        an object outside two bounds does not come back by moving one.
    """

    label: int
    area: int
    mean_intensity: float
    bounds: Tuple[str, ...]


def filter_report(mask: np.ndarray, image: np.ndarray, *,
                  min_area: int = 0, max_area: int = 0,
                  min_intensity: float = 0.0, max_intensity: float = 0.0,
                  preserve_ids: bool = False
                  ) -> Tuple[np.ndarray, List[FilterRemoval]]:
    """Filter as :func:`filter_objects` does, measuring what it removed.

    The same pass and the same arithmetic -- this is what
    :func:`filter_objects` now runs -- with each dropped object's area, mean
    and failed bounds kept instead of thrown away. Nothing else measures
    them a second time, so the ledger the screen prints cannot disagree with
    the mask it printed it about.

    :returns: ``(mask, removals)``, the removals sorted by id. Nothing to do
        returns the original array untouched and an empty list.
    :param preserve_ids: measure all pixels bearing an ID as one object,
        including lone or disconnected primary/secondary labels.
    """
    bounds = (int(min_area or 0), int(max_area or 0),
              float(min_intensity or 0.0), float(max_intensity or 0.0))
    lo_area, hi_area, lo_int, hi_int = bounds
    if not any(bounds) or mask is None or not mask.size or not mask.max():
        return mask, []

    from skimage.measure import regionprops

    grey = np.asarray(image, dtype=np.float32)
    if grey.ndim == 3:
        grey = grey.mean(axis=2)
    labels = canonical_labels(mask, preserve_ids=preserve_ids)
    removals: List[FilterRemoval] = []
    for region in regionprops(labels.astype(np.int32), intensity_image=grey):
        area = int(region.area)
        mean = float(region.intensity_mean
                     if hasattr(region, "intensity_mean")
                     else region.mean_intensity)
        failed = tuple(name for name, failure in (
            ("min_area", bool(lo_area and area < lo_area)),
            ("max_area", bool(hi_area and area > hi_area)),
            ("min_intensity", bool(lo_int and mean < lo_int)),
            ("max_intensity", bool(hi_int and mean > hi_int)),
        ) if failure)
        if failed:
            removals.append(
                FilterRemoval(int(region.label), area, mean, failed))
    if not removals:
        return mask, []
    removals.sort(key=lambda removal: removal.label)
    out = mask.copy()
    out[np.isin(labels, [removal.label for removal in removals])] = 0
    return out, removals


def filter_objects(mask: np.ndarray, image: np.ndarray, *,
                   min_area: int = 0, max_area: int = 0,
                   min_intensity: float = 0.0,
                   max_intensity: float = 0.0,
                   preserve_ids: bool = False) -> Tuple[np.ndarray, List[int]]:
    """Drop objects outside the size/intensity bounds. Each bound is off at 0.

    Area is the object's pixel count; intensity is its MEAN value on the
    *raw* image, not on the contrast-stretched display -- the display
    percentiles are a viewing choice and a filter that moved when you
    changed them would not be reproducible.

    Zero means "no bound on this side" rather than "reject everything",
    which is what makes all four bounds independently optional: a minimum
    area of 0 would exclude nothing anyway, so the value is free to carry
    the off switch.

    :returns: ``(mask, dropped)`` -- a new mask with the failing objects
        zeroed and the sorted ids that were dropped. The ids are what the
        curation ledger records, so an automatic filter is as traceable as a
        click. Nothing to do returns the original array untouched and an
        empty list.

    :func:`filter_report` is this function keeping what it measured; a
    caller that has to tell the user WHY an object went wants that one.

    :param preserve_ids: retain primary/secondary identities when measuring
        and removing labels; disconnected pieces sharing an ID count together.
    """
    out, removals = filter_report(
        mask, image, min_area=min_area, max_area=max_area,
        min_intensity=min_intensity, max_intensity=max_intensity,
        preserve_ids=preserve_ids)
    return out, [removal.label for removal in removals]


class PixelReadout(NamedTuple):
    """What the Make Masks readout says about one pixel of the open field.

    :ivar x: column, in image pixels.
    :ivar y: row, in image pixels.
    :ivar intensity: the raw image value at the pixel, before any display
        stretching.
    :ivar label: the id of the object under the pixel, as
        :func:`canonical_labels` numbers it; 0 on background.
    :ivar area: that object's pixel count; 0 on background.
    :ivar mean_intensity: that object's mean raw intensity, or ``None`` on
        background.
    """

    x: int
    y: int
    intensity: float
    label: int = 0
    area: int = 0
    mean_intensity: Optional[float] = None


class ObjectLookup:
    """The objects of one mask, measured as :func:`filter_objects` measures them.

    Built once for a mask state and then asked about one pixel at a time, so
    a readout that follows the mouse costs a bounding box per question rather
    than the whole field. The id is the one :func:`canonical_labels` gives,
    the area is that id's pixel count, and the mean is taken on the raw image
    in ``float32`` over the object's pixels in raster order, which is the
    arithmetic :func:`skimage.measure.regionprops` performs for the filter.
    So an intensity bound set to the mean shown keeps the object, and a bound
    just past it removes it.

    :param mask: the label image.
    :param image: the raw image under it, with the mask's height and width.
    """

    def __init__(self, mask: np.ndarray, image: np.ndarray, *,
                 preserve_ids: bool = False):
        """Number the objects and index their bounding boxes.

        :param mask: the label image.
        :param image: the raw image under it.
        :param preserve_ids: index exact IDs without binary interpretation
            or splitting disconnected pieces; agrees with exact-ID filtering.
        """
        self.labels = canonical_labels(mask, preserve_ids=preserve_ids)
        grey = np.asarray(image, dtype=np.float32)
        if grey.ndim == 3:
            grey = grey.mean(axis=2)
        self._grey = grey
        has_objects = bool(self.labels.size) and bool(self.labels.max())
        self._boxes = (_ndimage().find_objects(self.labels)
                       if has_objects else [])
        self._measured: dict = {}

    def measure(self, label: int) -> Optional[Tuple[int, float]]:
        """The area and mean intensity of object ``label``.

        :param label: an id in :attr:`labels`.
        :returns: ``(area, mean)``, or ``None`` when no object has that id.
        """
        label = int(label)
        if label in self._measured:
            return self._measured[label]
        if not 1 <= label <= len(self._boxes) \
                or self._boxes[label - 1] is None:
            return None
        box = self._boxes[label - 1]
        inside = self.labels[box] == label
        found = (int(np.count_nonzero(inside)),
                 float(np.mean(self._grey[box][inside])))
        self._measured[label] = found
        return found

    def at(self, x: int, y: int) -> Optional[PixelReadout]:
        """The readout for image pixel ``(x, y)``.

        :param x: column.
        :param y: row.
        :returns: the readout, or ``None`` for a pixel outside the field.
        """
        height, width = self.labels.shape[:2]
        if not (0 <= int(x) < width and 0 <= int(y) < height):
            return None
        x, y = int(x), int(y)
        intensity = float(self._grey[y, x])
        label = int(self.labels[y, x])
        measured = self.measure(label) if label else None
        if measured is None:
            return PixelReadout(x, y, intensity)
        return PixelReadout(x, y, intensity, label, *measured)


def connected_instances(binary: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Label every separated foreground region as its own object.

    :param binary: array whose truthy pixels are foreground. Regions touching
        diagonally are connected under the editor's eight-neighbour rule.
    :param min_area: regions smaller than this are dropped rather than
        labelled, so a detection does not hand back a field of single-pixel
        speckles for the user to delete by hand.
    """
    components, count = _ndimage().label(np.asarray(binary, dtype=bool), structure=_EIGHT)
    out = np.zeros(components.shape, dtype=np.int32)
    next_id = 1
    for old in range(1, count + 1):
        region = components == old
        if int(region.sum()) >= int(min_area):
            out[region] = next_id
            next_id += 1
    return out


def otsu_instances(image: np.ndarray, *, bright: bool = True,
                   min_area: int = 0) -> np.ndarray:
    """Threshold ``image`` at Otsu's level and label what is left.

    :param image: numeric intensity image. It is converted to float32 before
        the threshold is estimated and must contain at least one pixel.
    :param bright: objects are brighter than background (fluorescence).
        False takes the dark side instead, for a brightfield or a
        stained-plaque image where the objects absorb.
    :param min_area: passed to :func:`connected_instances`.
    :raises ValueError: on an empty image, which has no threshold to find.
    """
    from skimage.filters import threshold_otsu

    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        raise ValueError("Otsu needs an image; this one is empty.")
    threshold = float(threshold_otsu(values))
    binary = (values > threshold) if bright else (values < threshold)
    return connected_instances(binary, min_area=min_area)


def _otsu_values(image: np.ndarray, smoothing: float = 0.0) -> np.ndarray:
    """The float32 array every Otsu level in this module is measured on.

    One reader, so the level the histogram preview marks is measured on the
    same pixels the threshold is taken on rather than on the raw field: a
    preview drawn before the smoothing would mark a level that is not where
    the cut lands.

    :param image: the field, or a crop of it.
    :param smoothing: Gaussian sigma; 0 returns the values unsmoothed.
    :raises ValueError: on an empty image, which has no threshold to find.
    """
    values = np.asarray(image, dtype=np.float32)
    if not values.size:
        raise ValueError("Otsu needs an image; this one is empty.")
    sigma = max(0.0, float(smoothing))
    if sigma > 0.0:
        values = _ndimage().gaussian_filter(values, sigma)
    return values


#: The GLOBAL threshold algorithms this module can cut a field at, as
#: ``name -> the scikit-image function that finds the level``. Each takes
#: one image and returns one intensity, so all of them reach the detection
#: through the very same code: the correction, the smoothing, the
#: bright/dark side, fill holes, the split, the border rule and the minimum
#: area are found once, in :func:`_otsu_instances`, and an algorithm is the
#: one line that differs. ADDING ONE IS A ROW HERE.
#:
#: ``otsu`` is first and is the default everywhere, so a field thresholded
#: by a screen that knows nothing of this dictionary is thresholded exactly
#: as it always was.
GLOBAL_THRESHOLDS: Dict[str, str] = {
    "otsu": "threshold_otsu",
    "li": "threshold_li",
    "yen": "threshold_yen",
    "triangle": "threshold_triangle",
    "isodata": "threshold_isodata",
    "mean": "threshold_mean",
    "minimum": "threshold_minimum",
}

#: The LOCAL threshold algorithms, which return one level PER PIXEL rather
#: than one for the field, as ``name -> the scikit-image function``. They
#: read a window size, and Sauvola and Niblack read ``k`` as well.
#:
#: WHAT IS DELIBERATELY NOT HERE. Local MEAN and local GAUSSIAN
#: (``skimage.filters.threshold_local``) are not listed, because Make Masks
#: already offers them: they are what the Adaptive threshold mode runs,
#: through the organelle engine's own ``adaptive`` branch, with the same
#: block size and offset. Local OTSU is not listed either: it is the Otsu
#: category's "Local threshold (uneven illumination)" switch and is
#: :func:`_local_otsu_binary`. Listing either again would be two controls
#: for one operation.
LOCAL_THRESHOLDS: Dict[str, str] = {
    "sauvola": "threshold_sauvola",
    "niblack": "threshold_niblack",
}


def threshold_algorithms() -> Tuple[str, ...]:
    """Every threshold algorithm name, global then local."""
    return tuple(GLOBAL_THRESHOLDS) + tuple(LOCAL_THRESHOLDS)


def _global_level(values: np.ndarray, algorithm: str) -> float:
    """The one intensity ``algorithm`` cuts ``values`` at.

    :param values: the smoothed float image, from :func:`_otsu_values`.
    :param algorithm: a key of :data:`GLOBAL_THRESHOLDS`.
    :raises ValueError: for an algorithm this module does not know, rather
        than quietly thresholding by Otsu under another name.
    """
    from skimage import filters

    name = GLOBAL_THRESHOLDS.get(str(algorithm))
    if name is None:
        raise ValueError(
            f"{algorithm!r} is not a global threshold algorithm; "
            f"the ones there are: {sorted(GLOBAL_THRESHOLDS)}.")
    return float(getattr(filters, name)(values))


def _local_level_map(values: np.ndarray, algorithm: str, *, window: int,
                     k: float) -> np.ndarray:
    """A per-pixel threshold for ``values`` from a local algorithm.

    Niblack uses ``T = m - k*s`` and Sauvola uses
    ``T = m*(1 + k*(s/R - 1))``, with local mean m and deviation s.
    Values from :func:`_otsu_values` are float32 without range rescaling;
    scikit-image therefore defaults Sauvola's R to 1, not to the observed
    intensity range. Sauvola can consequently behave differently when the
    same image is multiplied by an intensity scale factor.

    :param values: the smoothed float image.
    :param algorithm: a key of :data:`LOCAL_THRESHOLDS`.
    :param window: the odd window size, in pixels.
    :param k: the algorithm's ``k``.
    :raises ValueError: for an algorithm this module does not know.
    """
    from skimage import filters

    name = LOCAL_THRESHOLDS.get(str(algorithm))
    if name is None:
        raise ValueError(
            f"{algorithm!r} is not a local threshold algorithm; "
            f"the ones there are: {sorted(LOCAL_THRESHOLDS)}.")
    return np.asarray(
        getattr(filters, name)(values, window_size=_odd_window(window),
                               k=float(k)),
        dtype=np.float32)


def _otsu_levels(image: np.ndarray, *, bright: bool = True,
                 correction: float = 1.0, smoothing: float = 0.0,
                 classes: int = 2, algorithm: str = "otsu") -> List[float]:
    """The intensity or intensities the field is actually cut at.

    The histogram preview shows the chosen level, and the only way a
    preview can be trusted to show it is for the detector to read the level
    from here too -- so :func:`_otsu_instances` calls this rather than
    finding its own, and a preview cannot drift from the button.

    The numbers are on the SMOOTHED image and already carry ``correction``,
    because that is where the cut is made.

    :param image: the field, or a crop of it.
    :param bright: objects are brighter than background. With two classes
        and a dark-object cut the level returned is the mirrored one --
        ``top - (top - level) * correction`` -- which is the value the
        detector compares against, so the preview marks the cut the user
        gets rather than Otsu's own number. Not read for three classes or
        more, where the class number says which side is meant.
    :param correction: Otsu's level is multiplied by this.
    :param smoothing: Gaussian sigma, applied before the level is found.
    :param classes: 2 for Otsu's own two-class split, 3 or more for
        multi-level Otsu (:func:`skimage.filters.threshold_multiotsu`),
        which returns ``classes - 1`` rising levels.
    :param algorithm: which of :data:`GLOBAL_THRESHOLDS` finds the level.
        Read only for two classes; multi-level Otsu is its own algorithm
        and is asked for by a class count above two.
    :raises ValueError: on an empty image, a correction that is not greater
        than 0, fewer than two classes, or an unknown algorithm.
    """
    from skimage.filters import threshold_multiotsu

    factor = _otsu_correction_factor(correction)
    count = int(classes)
    if count < 2:
        raise ValueError(
            f"Otsu needs at least two classes; got {classes!r}.")
    values = _otsu_values(image, smoothing)
    if count == 2:
        level = _global_level(values, algorithm)
        if bright:
            return [level * factor]
        top = float(values.max())
        return [top - (top - level) * factor]
    return [float(level) * factor
            for level in threshold_multiotsu(values, classes=count)]


def _otsu_histogram(image: np.ndarray, *, smoothing: float = 0.0,
                    bins: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges of the values the threshold is measured on.

    The picture behind the histogram preview. Measured on :func:`_otsu_values`
    for the same reason the levels are: a histogram of the raw field under a
    level found on the smoothed one would put the marker in the wrong valley.

    :param image: the field, or a crop of it.
    :param smoothing: Gaussian sigma, matching the detection's.
    :param bins: how many bars.
    :returns: ``(counts, edges)`` as :func:`numpy.histogram` returns them,
        so ``edges`` is one longer than ``counts``.
    :raises ValueError: on an empty image.
    """
    values = _otsu_values(image, smoothing)
    counts, edges = np.histogram(values, bins=max(2, int(bins)))
    return counts, edges


def _otsu_correction_factor(correction: float) -> float:
    """Validate and return the threshold correction multiplier."""
    factor = float(correction)
    if not factor > 0.0:
        raise ValueError(
            f"The Otsu threshold correction must be greater than 0; got "
            f"{correction!r}.")
    return factor


def _odd_window(window: int) -> int:
    """The local-threshold window as an odd number of pixels, validated.

    An even window has no centre pixel, so the level a pixel is judged
    against would be measured off-centre from it.

    :raises ValueError: for a window under 3.
    """
    size = int(window)
    if size < 3:
        raise ValueError(
            f"The local threshold window must be at least 3 px; got "
            f"{window!r}.")
    return size if size % 2 else size + 1


def _square_footprint(size: int) -> np.ndarray:
    """A square footprint of ``size`` px, whatever scikit-image calls it.

    ``footprint_rectangle`` arrived in scikit-image 0.25 and ``square`` is
    deprecated there and gone in 0.27, so both names are tried rather than
    pinning the package on a helper that returns an array of ones.
    """
    try:
        from skimage.morphology import footprint_rectangle

        return footprint_rectangle((size, size))
    except ImportError:
        from skimage.morphology import square

        return square(size)


def _local_otsu_binary(values: np.ndarray, *, window: int, bright: bool,
                       correction: float) -> np.ndarray:
    """Threshold every pixel against Otsu's level in the window around it.

    Adaptive Otsu: one level for the whole field loses
    an object wherever the illumination falls away, because the corner of a
    field can be dimmer than the background at its centre. Here each pixel is
    compared with the level found inside a ``window`` x ``window`` square
    centred on it (:func:`skimage.filters.rank.otsu`), so a dim corner is
    judged against its own corner.

    THE LEVELS ARE FOUND ON A 256-STEP RESCALING of ``values``, which is what
    the rank filters take and what keeps a megapixel field pressable: a
    16-bit rank filter builds a 65,536-bin histogram per pixel. The cut is
    then made on the same rescaling, so nothing is compared across the two.

    :param values: the smoothed float image.
    :param window: odd window size in pixels.
    :param bright: objects are brighter than background.
    :param correction: multiplies the local level, exactly as it multiplies
        the global one.
    :returns: a boolean foreground image.
    """
    from skimage.filters.rank import otsu as rank_otsu

    size = _odd_window(window)
    low = float(values.min())
    high = float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=bool)
    scaled = np.clip(
        np.rint((values - low) * (255.0 / (high - low))), 0.0, 255.0
    ).astype(np.uint8)
    levels = np.asarray(rank_otsu(scaled, _square_footprint(size)),
                        dtype=np.float32)
    here = scaled.astype(np.float32)
    factor = float(correction)
    if bright:
        return here > levels * factor
    return here < 255.0 - (255.0 - levels) * factor


def _otsu_instances(image: np.ndarray, *, bright: bool = True,
                    min_area: int = 0,
                    correction: float = 1.0,
                    smoothing: float = 0.0,
                    fill_holes: bool = False,
                    split_touching: bool = False,
                    exclude_border: bool = False,
                    classes: int = 2,
                    foreground_class: Optional[int] = None,
                    local: bool = False,
                    window: int = 51,
                    algorithm: str = "otsu",
                    local_k: float = 0.2) -> np.ndarray:
    """:func:`otsu_instances` with Otsu's level multiplied by ``correction``.

    OR ANOTHER ALGORITHM'S LEVEL. ``algorithm`` names one of
    :data:`GLOBAL_THRESHOLDS` or :data:`LOCAL_THRESHOLDS`, and it changes
    exactly one thing: where the number the field is cut at comes from.
    The smoothing, the correction, the bright-or-dark side, filling holes,
    the watershed split, the border rule and the minimum area are the same
    code for every one of them, which is the whole reason the algorithms
    are a dictionary and not ten functions.

    The "threshold correction", which is CellProfiler's threshold
    correction factor: the level Otsu finds is multiplied before it is used.
    For positive global thresholds and local Otsu, above 1 is stricter on
    either side: dark-object correction uses the distance below the image
    maximum (255 after rescaling for local Otsu).
    Sauvola and Niblack instead multiply their direct local level maps;
    for positive thresholds, raising the factor keeps fewer bright pixels
    but more dark pixels. Negative thresholds reverse those directions.

    A correction of exactly 1 with every switch below off IS
    :func:`otsu_instances`, looked up by name at call time, so the
    uncorrected path does not move at all.

    THE FOUR SWITCHES ARE THE OTSU MODE'S EXTRA SETTINGS, and they all
    default OFF -- a plain threshold and a connected-components
    labelling, which is what this did before they existed. Three of them are
    the steps the magnifier's Otsu mode has always taken and the detect
    button never did (:func:`_classical_region_labels`), which is why the
    two could disagree about the same field; turning them on is how a user
    makes the button do what the box under the mouse showed.

    :param correction: the factor, greater than 0.
    :param smoothing: Gaussian sigma applied before the level is estimated
        AND before the image is cut at it, so a noisy field is thresholded
        on what a reader sees rather than on its speckle.
    :param fill_holes: close the holes inside the thresholded foreground.
        A nucleus dimmer in the middle than at its rim arrives as a ring
        without this.
    :param split_touching: cut each blob where two objects meet
        (:func:`_split_touching_objects`) instead of labelling it whole.
    :param exclude_border: drop the objects the field's own edge cuts
        through (:func:`_drop_border_objects`).
    :param classes: multi-level Otsu. 2 is Otsu's own two-class
        split and is what this did before. 3 or more splits the histogram
        into that many brightness bands
        (:func:`skimage.filters.threshold_multiotsu`), which is how a field
        holding background, a dim halo and bright nuclei is cut at the
        boundary that matters instead of at the one compromise level between
        all three.
    :param foreground_class: with three classes or more, WHICH band becomes
        the objects, counting 0 for the dimmest. None takes the brightest,
        ``classes - 1``. Exactly that band is taken, so choosing a middle one
        gives the halo without the nuclei inside it -- which is the point of
        asking for more than two classes. ``bright`` is NOT read here: the
        class number already says which side is meant.
    :param local: the adaptive threshold. Each pixel is judged
        against Otsu's level in the ``window`` around it rather than against
        one level for the whole field (:func:`_local_otsu_binary`), which is
        what recovers objects in a corner the illumination has fallen away
        from. Two classes only.
    :param window: the local window, in pixels; rounded up to an odd number
        so it has a centre pixel. Read when ``local`` is on and by the
        algorithms in :data:`LOCAL_THRESHOLDS`.
    :param algorithm: which algorithm finds the level -- one of
        :data:`GLOBAL_THRESHOLDS` or :data:`LOCAL_THRESHOLDS`. ``otsu``,
        the default, is what this function did before there were others.
    :param local_k: dimensionless local contrast weight, default 0.2.
        Niblack uses ``T = m - k*s``; Sauvola uses
        ``T = m*(1 + k*(s/R - 1))``, with local mean m, standard deviation
        s and scikit-image's float-input default R=1. Increasing k lowers
        Niblack's threshold; Sauvola's direction depends on m and s/R.
        Bright foreground is strictly above the corrected level and dark
        foreground strictly below it. No automatic intensity rescaling.
    :raises ValueError: on an empty image, a correction that is not greater
        than 0, fewer than two classes, a foreground class outside them, a
        window under 3 px, or ``local`` asked for together with more than
        two classes -- which have no single meaning together and would
        otherwise silently drop one of the two.
    """
    factor = _otsu_correction_factor(correction)
    sigma = max(0.0, float(smoothing))
    count = int(classes)
    if count < 2:
        raise ValueError(f"Otsu needs at least two classes; got {classes!r}.")
    chosen = count - 1 if foreground_class is None else int(foreground_class)
    if not 0 <= chosen < count:
        raise ValueError(
            f"The foreground class must be one of 0..{count - 1} for "
            f"{count} classes; got {foreground_class!r}.")
    if local and count > 2:
        raise ValueError(
            "A local threshold finds one level per window, so it cannot "
            "also split the field into more than two classes. Turn one of "
            "the two off.")
    if str(algorithm or "otsu") in LOCAL_THRESHOLDS and count > 2:
        raise ValueError(
            f"{algorithm} finds one level per window, so it cannot also "
            f"split the field into {count} classes. Use Multi-Otsu, or "
            f"put the class count back to 2.")
    name = str(algorithm or "otsu")
    plain = (factor == 1.0 and sigma == 0.0 and not fill_holes
             and not split_touching and not exclude_border
             and count == 2 and not local and name == "otsu")
    if plain:
        return otsu_instances(image, bright=bright, min_area=min_area)
    values = _otsu_values(image, sigma)
    if name in LOCAL_THRESHOLDS:
        levels = _local_level_map(values, name, window=window, k=local_k)
        binary = (values > levels * factor if bright
                  else values < levels * factor)
    elif local:
        binary = _local_otsu_binary(values, window=window, bright=bright,
                                    correction=factor)
    elif count == 2:
        level = _otsu_levels(values, bright=bright, correction=factor,
                             smoothing=0.0, classes=2, algorithm=name)[0]
        binary = values > level if bright else values < level
    else:
        levels = _otsu_levels(values, bright=bright, correction=factor,
                              smoothing=0.0, classes=count)
        binary = np.digitize(values, levels) == chosen
    if fill_holes:
        binary = _ndimage().binary_fill_holes(binary)
    if split_touching:
        labels = _split_touching_objects(binary, min_area=min_area)
    else:
        labels = connected_instances(binary, min_area=min_area)
    if exclude_border:
        labels = _drop_border_objects(labels)
    return labels


def combine_masks(old: np.ndarray, new: np.ndarray,
                  mode: str = "replace") -> np.ndarray:
    """Fold a fresh detection into an existing mask.

    :param old: existing label image used as the merge base; ignored when
        ``mode`` is ``"replace"``.
    :param new: newly detected label image. In merge mode its positive labels
        are offset above ``old`` and copied only into background pixels.
    :param mode: ``"replace"`` -- the detection is the mask, and whatever
        was there is gone. ``"merge"`` -- keep every existing object and add
        the detected ones only where nothing is labelled yet, with fresh ids
        above the existing maximum. Merge never overwrites or splits an
        object that was curated by hand, which is the point of offering the
        choice: a detection run halfway through an editing session should
        not be able to silently undo the first half of it.
    :raises ValueError: for an unknown mode, rather than quietly picking
        one and discarding the user's edits.
    """
    if mode not in ("replace", "merge"):
        raise ValueError(
            f"combine mode must be 'replace' or 'merge', not {mode!r}")
    incoming = np.asarray(new).astype(np.int64)
    if mode == "replace":
        out = incoming
    else:
        base = int(old.max()) if old.size else 0
        out = old.astype(np.int64, copy=True)
        added = np.where(incoming > 0, incoming + base, 0)
        free = out == 0
        out[free] = added[free]
    top = int(out.max()) if out.size else 0
    if top > np.iinfo(np.uint16).max:
        raise ValueError(
            f"combined mask needs label {top}, past uint16; "
            "raise the minimum area so the detection makes fewer objects.")
    return out.astype(np.uint8 if top <= 255 else np.uint16)


#: The rules a live-magnifier commit knows for an object that lands on one
#: already labelled, in the order the editor offers them. ``clip`` keeps
#: only the new object's unlabelled pixels, so no existing object loses a
#: pixel; ``skip`` leaves out any object that touches an existing one;
#: ``replace`` lets the new object take every pixel it covers.
_MAGNIFIER_OVERLAP_RULES = ("clip", "skip", "replace")

#: How far one unit of magnifier sensitivity moves the classical cut, as a
#: fraction of the region's stretched intensity range. Positive lowers the
#: cut, so a more sensitive magnifier takes in dimmer pixels.
_CLASSICAL_SENSITIVITY_STEP = 0.05

#: Otsu's effectiveness -- between-class over total variance -- at or above
#: which a region is taken to hold two populations and is cut between them.
#: Below it the region is one population, background, with at most a few dim
#: objects in it; and background still HAS an Otsu level -- it splits the
#: noise in half, at an effectiveness near 2/pi for Gaussian noise -- so
#: cutting there hands back a sponge of objects that are not there. Such a
#: region is cut at a noise floor instead, :data:`_CLASSICAL_NOISE_SIGMAS`
#: robust deviations above its median.
_CLASSICAL_MIN_SEPARATION = 0.8

#: How many robust standard deviations (1.4826 x the median absolute
#: deviation) above the median a smoothed pixel must be to count as an
#: object in a one-population region. Each unit of sensitivity takes half a
#: deviation off, down to one.
_CLASSICAL_NOISE_SIGMAS = 4.0

#: Gaussian smoothing, in pixels, before either cut. It averages pixel noise
#: down about three and a half times, which is the difference between a dim
#: object being found and being broken into speckle.
_CLASSICAL_SMOOTHING = 1.0


def _magnifier_box(shape, x: int, y: int,
                   size: int) -> Tuple[int, int, int, int]:
    """The region a magnifier at image pixel ``(x, y)`` covers, clipped.

    :param shape: the image shape, ``(height, width, ...)``.
    :param x: column of the pixel under the cursor.
    :param y: row of the pixel under the cursor.
    :param size: side of the unclipped square, in image pixels. The cursor
        sits at index ``size // 2`` of it, so an even size puts the cursor
        just right of and below the middle.
    :returns: ``(x0, y0, x1, y1)`` with ``x1``/``y1`` exclusive -- the slice
        ``image[y0:y1, x0:x1]``. At the image border the box is cut short on
        that side only and never padded, so a crop is always real pixels and
        its top-left corner is always where its labels go.
    """
    height, width = int(shape[0]), int(shape[1])
    side = max(1, int(size))
    left = int(x) - side // 2
    top = int(y) - side // 2
    return (max(0, left), max(0, top),
            min(width, left + side), min(height, top + side))


def _drop_cut_objects(labels: np.ndarray, box, shape) -> np.ndarray:
    """Zero every object the box's own edge cuts through.

    An object touching an edge of the box that lies INSIDE the image is the
    part of an object the box happened to cover, and its boundary there is
    where the box ends -- the rule :func:`cut_recrop` applies to a recrop.
    An edge that is the image border cuts nothing off, so objects touching
    it are kept.

    :param labels: label image in crop coordinates.
    :param box: the crop's ``(x0, y0, x1, y1)`` in image pixels.
    :param shape: the image shape the box was clipped to.
    :returns: a copy of ``labels`` without the cut objects.
    """
    out = np.array(labels, copy=True)
    if not out.size or not out.max():
        return out
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    height, width = int(shape[0]), int(shape[1])
    edges = []
    if y0 > 0:
        edges.append(out[0, :])
    if y1 < height:
        edges.append(out[-1, :])
    if x0 > 0:
        edges.append(out[:, 0])
    if x1 < width:
        edges.append(out[:, -1])
    if edges:
        cut = np.unique(np.concatenate(edges))
        cut = cut[cut > 0]
        if cut.size:
            out[np.isin(out, cut)] = 0
    return out


def _split_touching_objects(binary: np.ndarray, min_area: int = 0) -> np.ndarray:
    """Label ``binary``, cutting each blob where two objects meet.

    A watershed on the blob's own distance transform: every local maximum of
    the distance to background is one object's middle, and the ridge between
    two of them is the line where they touch. A blob with a single maximum
    comes back whole, so this is not a splitter that cuts everything -- it
    cuts what has two centres.

    Seeds no nearer than the radius of an object of ``min_area``
    (:math:`\\sqrt{A/\\pi}`), so the smallest object the caller is willing to
    keep cannot itself be split in two; a blob the peak finder gave no seed
    at all gets one at its own deepest pixel, or it would be dropped.

    This is the tail :func:`_classical_region_labels` has always ended with,
    which :func:`_otsu_instances` now reaches too -- one recipe, so the Otsu
    mode in the magnifier and the Otsu detect button cut a pair of touching
    cells the same way.

    :param binary: truthy where there is foreground.
    :param min_area: objects smaller than this are dropped, and the seed
        spacing is taken from it.
    :returns: int32 labels 1..N, all zero for an empty ``binary``.
    """
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    ndimage = _ndimage()
    mask = np.asarray(binary, dtype=bool)
    empty = np.zeros(mask.shape, dtype=np.int32)
    if not mask.any():
        return empty
    distance = ndimage.gaussian_filter(
        ndimage.distance_transform_edt(mask), 1.0)
    components, count = ndimage.label(mask, structure=_EIGHT)
    spacing = max(2, int(np.sqrt(max(int(min_area), 12) / np.pi)))
    peaks = peak_local_max(distance, min_distance=spacing, labels=components,
                           exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    for index, point in enumerate(peaks, start=1):
        markers[tuple(point)] = index
    seeded = {int(v) for v in np.unique(components[markers > 0])}
    next_marker = len(peaks) + 1
    for component in range(1, count + 1):
        if component in seeded:
            continue
        where = int(np.argmax(np.where(components == component, distance, -1.0)))
        markers.flat[where] = next_marker
        next_marker += 1
    labels = watershed(-distance, markers, mask=mask)
    areas = np.bincount(labels.ravel())
    keep = areas >= max(1, int(min_area))
    keep[0] = False
    lookup = np.zeros(areas.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[labels]


def _drop_border_objects(labels: np.ndarray) -> np.ndarray:
    """Drop every object touching the edge of the field, and renumber.

    An object the frame cut through has an area and a mean intensity that
    are properties of where the frame fell, not of the object, so a
    detection meant to be measured is better off without it.

    :param labels: int label image.
    :returns: int32 labels 1..N holding only the objects clear of the edge.
    """
    lab = np.asarray(labels)
    if not lab.size:
        return np.zeros(lab.shape, dtype=np.int32)
    edge = set()
    for axis in range(lab.ndim):
        for index in (0, -1):
            edge.update(int(v) for v in np.unique(np.take(lab, index, axis)))
    edge.discard(0)
    keep = np.ones(int(lab.max()) + 1, dtype=bool)
    keep[0] = False
    for value in edge:
        keep[value] = False
    lookup = np.zeros(keep.size, dtype=np.int32)
    lookup[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lookup[lab]


#: How a propagation decides where to stop growing, as
#: ``name -> what the number beside it means``. See
#: :func:`maxima_propagate_instances`.
PROPAGATE_STOPS: Dict[str, str] = {
    "seed_fraction": "a fraction of THIS seed's own peak value",
    "absolute": "an absolute intensity",
    "percentile": "a percentile of the whole image",
    "threshold": "a global threshold algorithm's level",
}


class PropagateResult(NamedTuple):
    """What one maxima-and-propagate run found.

    :param labels: the objects, one label per seed that survived.
    :param seeds: how many local maxima were found. THE NUMBER THE USER
        TUNES AGAINST: too many and the minimum distance or the seed level
        is too low, too few and an object has no centre to grow from, and
        neither is visible from the objects alone.
    :param level: the intensity the growth stopped at, for the stop rules
        that have ONE -- absolute, percentile and a global threshold. None
        for ``seed_fraction``, which has a different level per object and
        so has no single number to report.
    """

    labels: np.ndarray
    seeds: int
    level: Optional[float]


class PrimarySecondaryReport(NamedTuple):
    """Label relationships between a primary mask and a secondary mask.

    All fields contain sorted tuples of Python integer IDs; 0 is excluded.
    ``matched_ids`` occur in both masks. ``missing_secondary_ids`` occur
    only in the primary mask, and ``orphan_secondary_ids`` only in the
    secondary mask. ``incomplete_primary_ids`` are matched IDs whose
    secondary does not contain every pixel of its primary. Matched IDs
    without any secondary pixels outside their own primary are listed in
    ``unexpanded_primary_ids``; these can indicate a threshold that stopped
    growth immediately. A match alone does not prove correct cell boundaries.

    :ivar primary_ids: nonzero IDs present in the primary mask.
    :ivar secondary_ids: nonzero IDs present in the secondary mask.
    :ivar matched_ids: IDs present in both masks.
    :ivar missing_secondary_ids: primary IDs absent from the secondary mask.
    :ivar orphan_secondary_ids: secondary IDs absent from the primary mask.
    :ivar incomplete_primary_ids: matched IDs whose secondary omits primary pixels.
    :ivar unexpanded_primary_ids: matched IDs with no growth beyond their primary.
    """

    primary_ids: Tuple[int, ...]
    secondary_ids: Tuple[int, ...]
    matched_ids: Tuple[int, ...]
    missing_secondary_ids: Tuple[int, ...]
    orphan_secondary_ids: Tuple[int, ...]
    incomplete_primary_ids: Tuple[int, ...]
    unexpanded_primary_ids: Tuple[int, ...]


class SecondaryResult(NamedTuple):
    """Secondary labels, their primary relationships and the common stop level.

    ``labels`` has the primary mask's dtype, shape and retained object IDs.
    ``relationships`` includes primaries removed by minimum-area filtering.
    ``level`` is None for the per-primary peak-ratio rule or an empty primary
    mask; otherwise it is the common threshold in processed intensity units.

    :ivar labels: secondary label array retaining primary IDs and dtype.
    :ivar relationships: primary/secondary identity report after filtering.
    :ivar level: shared stop threshold, or None when no common threshold applies.
    """

    labels: np.ndarray
    relationships: PrimarySecondaryReport
    level: Optional[float]


def _primary_label_image(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate a nonempty 2-D, nonnegative integer label image without casting."""
    values = np.asarray(labels)
    if values.ndim != 2 or not values.size:
        raise ValueError(f"{name} must be a nonempty 2-D label image.")
    if values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integer labels.")
    return values


def primary_secondary_report(primary: np.ndarray,
                             secondary: np.ndarray) -> PrimarySecondaryReport:
    """Report shared, missing, orphaned and incompletely enclosed object IDs.

    :param primary: nonempty 2-D nonnegative integer primary labels.
    :param secondary: secondary labels of the same shape. Sparse and uint64
        IDs are compared exactly, including values above signed int64.
    :returns: :class:`PrimarySecondaryReport`; no array is modified. An ID
        match means the IDs agree, not that spatial overlap was used to
        infer or repair a parent assignment.
    :raises ValueError: mismatched shapes or invalid label arrays.
    """
    first = _primary_label_image(primary, "Primary mask")
    second = _primary_label_image(secondary, "Secondary mask")
    if first.shape != second.shape:
        raise ValueError("Primary and secondary masks must have the same shape.")
    primary_ids = {int(value) for value in np.unique(first) if value}
    secondary_ids = {int(value) for value in np.unique(second) if value}
    matched = primary_ids & secondary_ids
    at_primary = first > 0
    primary_values = first[at_primary].astype(np.uint64)
    secondary_values = second[at_primary].astype(np.uint64)
    incomplete = {int(value) for value in np.unique(
        primary_values[primary_values != secondary_values])} & matched
    at_secondary = second > 0
    secondary_values = second[at_secondary].astype(np.uint64)
    primary_values = first[at_secondary].astype(np.uint64)
    expanded = {int(value) for value in np.unique(
        secondary_values[primary_values != secondary_values])}
    return PrimarySecondaryReport(
        tuple(sorted(primary_ids)), tuple(sorted(secondary_ids)),
        tuple(sorted(matched)), tuple(sorted(primary_ids - secondary_ids)),
        tuple(sorted(secondary_ids - primary_ids)), tuple(sorted(incomplete)),
        tuple(sorted(matched - expanded)))


def secondary_object_instances(
        image: np.ndarray, primary: np.ndarray, *, sigma: float = 2.0,
        stop: str = "threshold", stop_value: float = 0.4,
        stop_algorithm: str = "otsu", min_area: int = 0,
        fill_holes: bool = True, growth: str = "intensity") -> SecondaryResult:
    """Grow secondary objects from labelled primaries with a seeded watershed.

    Every positive primary label is a marker, including all its pixels.
    Intensity growth follows the negative, optionally Gaussian-smoothed image.
    Distance growth floods a flat surface from the primary pixels. Common
    stop thresholds constrain four-connected paths around excluded pixels;
    seed_fraction trims after growth. This is not unrestricted Euclidean
    nearest-primary assignment.
    Neither mode is CellProfiler's distance/intensity Propagation algorithm.

    The four rules in :data:`PROPAGATE_STOPS` use processed intensities.
    ``absolute``, ``percentile`` and ``threshold`` restrict growth with a
    common foreground mask. ``seed_fraction`` trims each watershed basin
    at a fraction of the brightest processed pixel inside its primary.
    That ratio is not a quantile and depends on background offset. With a
    dark nucleus in a cytoplasmic channel, use a common threshold instead
    of a nucleus-relative peak ratio.

    Primary pixels are always included before minimum-area filtering, even
    below the threshold. Hole filling can also restore below-threshold
    pixels. Filtering can remove a whole secondary together with its seed;
    the missing ID is reported. Remaining labels retain their primary IDs
    exactly, without splitting or renumbering disconnected components.
    Sparse IDs use compact internal markers, never arrays sized by max ID.

    :param image: finite nonempty 2-D intensities, converted to float32.
        The caller supplies normalization, background correction or inversion.
    :param primary: same-shape nonnegative integer primary labels; 0 means
        background. The returned labels retain this dtype and these IDs.
    :param sigma: finite nonnegative Gaussian sigma in pixels; 0 disables
        smoothing. Smoothing affects growth and threshold estimation.
    :param stop: one of :data:`PROPAGATE_STOPS`, default ``"threshold"``.
    :param stop_value: intensity for ``absolute``, percentile in [0,100] for
        ``percentile``, or a fraction in [0,1] for ``seed_fraction``. Ignored
        by ``threshold``. Fractions require nonnegative processed intensities.
    :param stop_algorithm: global threshold algorithm, default ``"otsu"``;
        read only for ``threshold``. The full supplied image determines it.
    :param min_area: minimum secondary area after hole filling; 0 disables
        filtering. The whole primary footprint counts toward the area.
    :param fill_holes: fill enclosed background pixels per label before
        filtering. Does not overwrite another primary's labelled pixels.
    :param growth: ``intensity`` (default) uses negative image intensity;
        ``distance`` floods a flat surface. Common stop rules constrain paths;
        the primary-relative fraction trims after growth.
        Both retain the same stop rules and exact primary IDs. Distance can
        help when bright structures attract an intensity basin across cells.
    :returns: :class:`SecondaryResult`, including ID relationship diagnostics.
        Empty primary masks yield an empty result and no common stop level.
    :raises ValueError: invalid images, labels, shape, sigma, stop rule,
        rule-specific stop value or negative minimum area.
    """
    markers = _primary_label_image(primary, "Primary mask")
    if growth not in ('intensity', 'distance'):
        raise ValueError("Secondary growth must be intensity or distance.")
    values = np.asarray(image, dtype=np.float32)
    if values.shape != markers.shape or not np.isfinite(values).all():
        raise ValueError("Image must be finite and match the primary mask's shape.")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Gaussian sigma must be finite and nonnegative.")
    if str(stop) not in PROPAGATE_STOPS:
        raise ValueError(f"Unknown propagation stop rule: {stop!r}.")
    minimum = float(min_area)
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("Minimum area must be finite and nonnegative.")
    if stop != "threshold":
        value = float(stop_value)
        if not np.isfinite(value):
            raise ValueError("Stop value must be finite.")
        if stop == "percentile" and not 0 <= value <= 100:
            raise ValueError("Stop percentile must be between 0 and 100.")
        if stop == "seed_fraction" and (not 0 <= value <= 1 or values.min() < 0):
            raise ValueError("Seed fraction needs a value in [0,1] and nonnegative intensities.")
    ids = np.unique(markers)
    ids = ids[ids > 0]
    dense = np.zeros(markers.shape, dtype=np.int32)
    foreground = markers > 0
    dense[foreground] = np.searchsorted(ids, markers[foreground]) + 1
    blurred = _ndimage().gaussian_filter(values, sigma) if sigma > 0 else values
    grown = _grow_markers(
        blurred, dense, stop=stop, stop_value=stop_value,
        stop_algorithm=stop_algorithm, min_area=int(minimum),
        fill_holes=fill_holes, keep_markers=True, relabel=False, growth=growth)
    lookup = np.concatenate((np.zeros(1, dtype=markers.dtype), ids))
    labels = lookup[grown.labels]
    return SecondaryResult(labels, primary_secondary_report(markers, labels),
                           grown.level)


def maxima_propagate_instances(
        image: np.ndarray, *, sigma: float = 2.0, min_distance: int = 10,
        seed_level: float = 90.0, seed_level_is_percentile: bool = True,
        exclude_border: bool = False, stop: str = "seed_fraction",
        stop_value: float = 0.4, stop_algorithm: str = "otsu",
        min_area: int = 0, fill_holes: bool = True) -> PropagateResult:
    """Segment bright objects with local maxima and an intensity watershed.

    Convert the field to float32, optionally blur it, and find centres with
    :func:`skimage.feature.peak_local_max`. Use those centres as markers for
    :func:`skimage.segmentation.watershed` on the negative blurred image.
    Distinct centres can split touching objects; noise can create extra
    centres, while smoothing or large centre spacing can remove real ones.
    No existing primary-object mask is accepted. This implementation has
    no CellProfiler propagation cost or distance/intensity weighting.

    The four stop rules operate on the blurred values:

    * ``seed_fraction`` first partitions the entire image by watershed,
      then retains pixels at or above ``stop_value`` times their basin's
      seed intensity. This is an intensity ratio, not a quantile. Adding a
      background offset changes the relative cut; equal measurements of
      bright and dim objects are not guaranteed. Trimming can leave
      disconnected pieces with the same label.
    * ``absolute`` restricts the watershed to pixels at or above
      ``stop_value``, in the input's intensity units.
    * ``percentile`` uses that percentile of all blurred input pixels as
      the common threshold. Changing the crop can change this level.
    * ``threshold`` obtains the common level from ``stop_algorithm`` and
      ignores ``stop_value``.

    Fill holes per label if requested, then discard labels smaller than
    ``min_area`` and renumber survivors. Hole filling can restore pixels
    below the selected intensity cut. The seed count is recorded before
    these operations and can exceed the number of surviving objects.

    Input preparation is the caller's responsibility: this function does
    not normalize intensities, subtract background, or invert dark objects.
    Use finite 2-D values; NaNs and infinities are not sanitized. Make Masks
    supplies the processed field or crop after its selected enhancements.
    Absolute levels and peak ratios therefore depend on that preparation.

    :param image: nonempty 2-D intensity array, converted to float32 without
        range rescaling. Output coordinates and shape match this array.
    :param sigma: Gaussian standard deviation in pixels; default 2.0.
        Positive values smooth both seed finding and growth; zero disables
        blur. Increasing it can suppress noise peaks or merge real peaks.
        Make Masks offers 0 to 50; the direct API also skips negative values.
    :param min_distance: centre separation in pixels; default 10. Passed
        to peak finding as ``max(1, int(min_distance))``, using its default
        Chebyshev distance. Increasing it suppresses nearby candidate seeds;
        reducing it can split an object into several detections. Make Masks
        offers 1 to 500.
    :param seed_level: default 90.0. Candidate maxima must exceed this
        intensity, or the intensity at this percentile when
        ``seed_level_is_percentile`` is true. Percentiles must be between
        0 and 100. Increasing the floor excludes dimmer candidate centres;
        it does not directly set the final object boundary.
    :param seed_level_is_percentile: default true. Compute the seed floor
        from all blurred input pixels; false uses an absolute intensity.
        A crop and a whole field can yield different percentile floors.
    :param exclude_border: default false. If true, exclude candidate centres
        within the effective ``min_distance`` of the input edge. This does
        not remove every object whose grown boundary touches the edge.
    :param stop: default ``"seed_fraction"``; one of
        :data:`PROPAGATE_STOPS`, with the behavior described above.
    :param stop_value: default 0.4. For ``seed_fraction``, use a ratio from
        0 to 1 with nonnegative intensities; increasing it removes dimmer
        basin pixels before hole filling. For ``absolute``, use an intensity;
        for ``percentile``, use 0 to 100. Higher common thresholds shrink
        the eligible mask. Ignored for ``threshold``. The API does not clip
        ratios or absolute values; the GUI number box alone does not enforce
        rule-specific limits.
    :param stop_algorithm: default ``"otsu"``; a key of
        :data:`GLOBAL_THRESHOLDS`, read only for ``stop="threshold"``.
        The threshold is estimated from the blurred field or crop.
    :param min_area: default 0, disabling size removal. Labels with fewer
        than ``int(min_area)`` pixels after hole filling are discarded.
        Increasing it removes small labels without merging touching ones.
    :param fill_holes: default true. Fill enclosed background pixels per
        label before size filtering. False preserves those holes.
    :returns: :class:`PropagateResult` containing an int32 label array
        (0 is background, surviving labels are 1 through N), the original
        seed count, and the common stop level. The level is None for
        ``seed_fraction`` or when no seeds were found. A constant image or
        an overly high seed floor can return all-zero labels and zero seeds;
        an empty stop mask or size filtering can remove every seeded object.
    :raises ValueError: for an unknown stop rule, an empty or non-2-D image,
        an out-of-range percentile when evaluated, or an unknown global
        threshold algorithm when that rule is evaluated.

    For example, ``maxima_propagate_instances(image, sigma=2,
    min_distance=10, seed_level=90, stop="seed_fraction", stop_value=0.4,
    min_area=20)`` retains each basin above 40 percent of its seed intensity
    before filling holes and removing labels smaller than 20 pixels.
    """
    from skimage.feature import peak_local_max

    if str(stop) not in PROPAGATE_STOPS:
        raise ValueError(
            f"{stop!r} is not a propagation stop rule; the ones there are: "
            f"{sorted(PROPAGATE_STOPS)}.")
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or not values.size:
        raise ValueError("Propagation needs a 2-D image; this one is empty.")
    empty = np.zeros(values.shape, dtype=np.int32)
    blurred = (_ndimage().gaussian_filter(values, float(sigma))
               if float(sigma) > 0.0 else values)

    floor = (float(np.percentile(blurred, float(seed_level)))
             if seed_level_is_percentile else float(seed_level))
    coordinates = peak_local_max(
        blurred, min_distance=max(1, int(min_distance)), threshold_abs=floor,
        exclude_border=max(1, int(min_distance)) if exclude_border else False)
    if not len(coordinates):
        return PropagateResult(empty, 0, None)

    markers = np.zeros(values.shape, dtype=np.int32)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

    return _grow_markers(
        blurred, markers, stop=stop, stop_value=stop_value,
        stop_algorithm=stop_algorithm, min_area=min_area, fill_holes=fill_holes)


def _grow_markers(blurred, markers, *, stop, stop_value, stop_algorithm,
                  min_area, fill_holes, keep_markers=False,
                  relabel=True, growth="intensity") -> PropagateResult:
    """Grow compact markers with shared stop, fill and size-filter semantics."""
    from skimage.segmentation import watershed

    seeds = int(markers.max())
    empty = np.zeros(markers.shape, dtype=np.int32)
    if not seeds:
        return PropagateResult(empty, 0, None)

    level: Optional[float] = None
    surface = np.zeros_like(blurred) if growth == 'distance' else -blurred
    if stop == "seed_fraction":
        grown = watershed(surface, markers)
        peaks = np.zeros(seeds + 1, dtype=np.float32)
        peaks[1:] = _ndimage().maximum(blurred, markers, np.arange(1, seeds + 1))
        keep = blurred >= peaks[grown] * float(stop_value)
        labels = np.where(keep, grown, 0).astype(np.int32)
    else:
        if stop == "absolute":
            level = float(stop_value)
        elif stop == "percentile":
            level = float(np.percentile(blurred, float(stop_value)))
        else:
            level = _global_level(blurred, stop_algorithm)
        mask = blurred >= level
        if keep_markers:
            mask |= markers > 0
        if not mask.any():
            return PropagateResult(empty, seeds, level)
        labels = np.asarray(watershed(surface, markers, mask=mask),
                            dtype=np.int32)

    if keep_markers:
        labels[markers > 0] = markers[markers > 0]
    if fill_holes:
        labels = _fill_label_holes(labels)
    return PropagateResult(_drop_small_labels(labels, min_area, relabel=relabel),
                           seeds, level)


def _fill_label_holes(labels: np.ndarray) -> np.ndarray:
    """Close the holes inside each object, without joining two of them.

    ``binary_fill_holes`` over the whole foreground would fill the gap
    BETWEEN two objects that happen to ring a piece of background, so the
    holes are filled per label and written back only where nothing else has
    a claim.
    """
    ndimage = _ndimage()
    out = np.asarray(labels, dtype=np.int32).copy()
    background = out == 0
    for value in np.unique(out):
        if value == 0:
            continue
        filled = ndimage.binary_fill_holes(out == value)
        out[filled & background] = value
    return out


def _drop_small_labels(labels: np.ndarray, min_area: int, *,
                       relabel: bool = True) -> np.ndarray:
    """Remove objects under ``min_area`` and renumber the rest from 1.

    Set ``relabel=False`` to retain marker IDs after dropping small labels.
    Inputs use compact integer labels; sparse external IDs must be mapped
    before calling this helper. Renumbering by remapping and NOT by
    re-labelling the foreground: two
    objects that touch are two objects, and connected-components would make
    them one again.
    """
    out = np.asarray(labels, dtype=np.int32)
    counts = np.bincount(out.ravel())
    if int(min_area) > 0:
        small = counts < int(min_area)
        small[0] = True
        out = np.where(small[out], 0, out)
    if not relabel:
        return out
    present = np.unique(out)
    present = present[present > 0]
    remap = np.zeros(int(out.max()) + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return remap[out]


def _classical_region_labels(region: np.ndarray, *, sensitivity: float = 0.0,
                             bright: bool = True,
                             min_area: int = 0,
                             correction: float = 1.0,
                             smoothing: float = _CLASSICAL_SMOOTHING,
                             fill_holes: bool = True,
                             split_touching: bool = True,
                             algorithm: str = "otsu",
                             window: int = 51,
                             local_k: float = 0.2) -> np.ndarray:
    """Threshold one magnifier region and split the objects that touch.

    The Otsu magnifier mode -- formerly named ``classical`` --
    and the fallback whenever a model cannot run: it needs nothing beyond
    scikit-image. The region is smoothed (``smoothing``, by default
    :data:`_CLASSICAL_SMOOTHING`) and then cut one of two ways. A region
    that holds two clear populations
    (:data:`_CLASSICAL_MIN_SEPARATION`) is cut at Otsu's level, moved by
    ``sensitivity`` steps of :data:`_CLASSICAL_SENSITIVITY_STEP` of its
    stretched range; any other region is background with at most a few dim
    objects, and is cut :data:`_CLASSICAL_NOISE_SIGMAS` robust deviations
    above its median. The foreground is opened, hole-filled and split with a
    watershed on its distance transform.

    THE THREE SWITCHES DEFAULT TO WHAT THIS DID BEFORE THEY EXISTED, so the
    call the magnifier has always made comes back the mask it has always
    come back. They are here because the Otsu settings panel shows them,
    and the panel drives both this and :func:`_otsu_instances`.

    :param region: 2-D intensity crop.
    :param sensitivity: 0 is the default cut; positive takes in dimmer
        pixels, negative keeps only the brightest.
    :param bright: objects are brighter than background; False takes the
        dark side, for brightfield.
    :param min_area: objects smaller than this are dropped. It also sets how
        far apart two seeds must be, so an object of the smallest allowed
        size is not split in two.
    :param correction: Otsu's level is multiplied by this before
        ``sensitivity`` moves it (the threshold correction; see
        :func:`_otsu_instances`). Above 1 is stricter. A region with no two
        clear populations is cut at its noise floor, which is not Otsu's
        level, and this does not apply to it.
    :param smoothing: Gaussian sigma applied before either cut. 0 cuts the
        raw region, which finds every speckle a noisy field has.
    :param fill_holes: close the holes inside the thresholded foreground
        before it is labelled. Off leaves a dim nucleus as a ring.
    :param split_touching: cut each blob at the ridge between two centres
        (:func:`_split_touching_objects`). Off labels each blob whole, so a
        pair of touching cells arrives as one object.
    :param algorithm: which of :data:`GLOBAL_THRESHOLDS` or
        :data:`LOCAL_THRESHOLDS` finds the level. ``otsu`` is the default
        and is what this did before there were others. A LOCAL ALGORITHM
        SKIPS THE TWO-POPULATION TEST below, because that test is a
        judgement about a whole region's histogram and a local algorithm
        does not take one.
    :param window: the window a local algorithm measures in, in pixels.
    :param local_k: Sauvola's and Niblack's ``k``.
    :returns: int32 labels 1..N shaped like ``region``; all zero for a
        region with nothing above its noise.
    """
    ndimage = _ndimage()
    values = np.asarray(region, dtype=np.float32)
    empty = np.zeros(values.shape, dtype=np.int32)
    if values.ndim != 2 or values.size < 4:
        return empty
    sigma = max(0.0, float(smoothing))
    smooth = (ndimage.gaussian_filter(values, sigma) if sigma > 0.0
              else values)
    if not bright:
        smooth = -smooth
    lo, hi = (float(v) for v in np.percentile(smooth, (1.0, 99.8)))
    if hi <= lo:
        return empty
    stretched = np.clip((smooth - lo) / (hi - lo), 0.0, 1.0)
    name = str(algorithm or "otsu")
    if name in LOCAL_THRESHOLDS:
        levels = _local_level_map(stretched, name, window=window, k=local_k)
        foreground = stretched > levels * float(correction)
        return _finish_region_binary(foreground, ndimage, empty,
                                     fill_holes=fill_holes,
                                     split_touching=split_touching,
                                     min_area=min_area)
    level = _global_level(stretched, name)
    upper = stretched > level
    share = float(upper.mean())
    total = float(stretched.var())
    separation = 0.0
    if 0.0 < share < 1.0 and total > 0.0:
        gap = float(stretched[upper].mean() - stretched[~upper].mean())
        separation = share * (1.0 - share) * gap * gap / total
    if separation >= _CLASSICAL_MIN_SEPARATION:
        cut = (level * float(correction)
               - _CLASSICAL_SENSITIVITY_STEP * float(sensitivity))
        foreground = stretched > cut
    else:
        centre = float(np.median(smooth))
        spread = 1.4826 * float(np.median(np.abs(smooth - centre)))
        if spread <= 0.0:
            return empty
        sigmas = max(1.0, _CLASSICAL_NOISE_SIGMAS - 0.5 * float(sensitivity))
        foreground = smooth > centre + sigmas * spread

    return _finish_region_binary(foreground, ndimage, empty,
                                 fill_holes=fill_holes,
                                 split_touching=split_touching,
                                 min_area=min_area)


def _finish_region_binary(foreground, ndimage, empty, *, fill_holes: bool,
                          split_touching: bool, min_area: int) -> np.ndarray:
    """Open, fill and label a magnifier region's foreground.

    The tail of :func:`_classical_region_labels`, in a function of its own
    because a local algorithm reaches it without passing through the
    two-population test in the middle of that one. Splitting it out is what
    keeps there being ONE description of what happens to a region's
    foreground after it has been decided.
    """
    binary = ndimage.binary_opening(foreground, structure=_EIGHT)
    if fill_holes:
        binary = ndimage.binary_fill_holes(binary)
    if not binary.any():
        return empty
    if not split_touching:
        return connected_instances(binary, min_area=min_area)
    return _split_touching_objects(binary, min_area=min_area)


def _paste_region_objects(mask: np.ndarray, labels: np.ndarray, origin, *,
                          overlap: str = "clip",
                          min_area: int = 0,
                          preserve_ids: bool = False) -> Tuple[np.ndarray, List[int]]:
    """Add a region's objects to ``mask`` as new objects.

    What a live-magnifier click commits. The labels arrive in the region's
    own coordinates and ``origin`` is where the region's top-left pixel sits
    in the image, so an object at ``labels[r, c]`` lands on
    ``mask[origin_y + r, origin_x + c]``. Anything that would fall outside
    the image is dropped rather than wrapped or clamped.

    :param mask: the label image to copy and add to. Its objects keep their
        ids; the dtype widens only when a new id needs it.
    :param labels: label image in region coordinates. Each distinct positive
        value is one object.
    :param origin: ``(x, y)`` of ``labels[0, 0]`` in image pixels.
    :param overlap: one of :data:`_MAGNIFIER_OVERLAP_RULES` -- what a new
        object does where the mask is already labelled. Under ``clip`` an
        object that an existing one splits in two keeps only its largest
        piece, because one id must name one object.
    :param min_area: an object left smaller than this once the rule has been
        applied is not added.
    :param preserve_ids: paste the supplied IDs, not newly allocated IDs.
        Same-ID pixels do not conflict under Clip or Skip. Disconnected
        pieces retain their shared identity. Exact-ID masks must fit uint16.
    :returns: ``(mask, new_ids)``. New ids start one past the mask's top id
        (:func:`next_label`) and follow the incoming labels' order, so they
        cannot collide with any id the mask holds. Nothing added returns a
        copy of ``mask`` and an empty list.
    :raises ValueError: for an unknown ``overlap`` rule.
    """
    if overlap not in _MAGNIFIER_OVERLAP_RULES:
        raise ValueError(
            f"overlap must be one of {_MAGNIFIER_OVERLAP_RULES}, "
            f"not {overlap!r}")
    incoming = np.asarray(labels)
    if preserve_ids:
        incoming = canonical_labels(incoming, preserve_ids=True)
        mask = canonical_labels(mask, preserve_ids=True)
    height, width = mask.shape[:2]
    ox, oy = int(origin[0]), int(origin[1])
    x0, y0 = max(0, ox), max(0, oy)
    x1 = min(width, ox + incoming.shape[1])
    y1 = min(height, oy + incoming.shape[0])
    if x1 <= x0 or y1 <= y0:
        return mask.copy(), []
    incoming = incoming[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    if not incoming.any():
        return mask.copy(), []
    occupied = np.asarray(mask)[y0:y1, x0:x1] > 0
    if preserve_ids:
        occupied &= np.asarray(mask)[y0:y1, x0:x1] != incoming
    kept = _surviving_region_objects(incoming, occupied, overlap=overlap,
                                    min_area=min_area, preserve_ids=preserve_ids)
    values = [int(v) for v in np.unique(kept) if int(v) > 0]
    if not values:
        return mask.copy(), []
    out = mask.astype(np.int64, copy=True)
    window = out[y0:y1, x0:x1]
    if preserve_ids:
        body = kept > 0
        window[body] = kept[body]
        return _fit_label_width(out, mask), values
    new_id = next_label(mask)
    added: List[int] = []
    renumber = np.zeros(int(kept.max()) + 1, dtype=np.int64)
    for value in values:
        renumber[value] = new_id
        added.append(new_id)
        new_id += 1
    body = kept > 0
    window[body] = renumber[kept[body]]
    return _fit_label_width(out, mask), added


def _largest_piece_of_each(labels: np.ndarray) -> np.ndarray:
    """Keep one connected piece of every object: the largest.

    One id must name one object, so an object an existing one has split in
    two cannot go into the mask as two islands under one label.

    A piece is a connected run of ONE id. :func:`skimage.measure.label` is
    what says so and :func:`scipy.ndimage.label` is not: the latter would
    take two different objects that touch as one piece, and the largest
    piece of a pair is not the largest piece of either.

    Ties go to the piece whose topmost-leftmost pixel comes first, which is
    what a per-object ``argmax`` over the areas used to pick.

    :param labels: a label image; 0 is background.
    :returns: the same image with every object's smaller pieces set to 0.
    """
    from skimage.measure import label as label_regions

    pieces = label_regions(labels, connectivity=2, background=0)
    count = int(pieces.max())
    if count <= 1:
        return labels
    areas = np.bincount(pieces.ravel(), minlength=count + 1)
    areas[0] = 0
    flat_pieces, flat_labels = pieces.ravel(), labels.ravel()
    inside = flat_pieces > 0
    owner = np.zeros(count + 1, dtype=np.int64)
    owner[flat_pieces[inside]] = flat_labels[inside]
    order = np.lexsort((-np.arange(count + 1), areas[:count + 1]))
    best = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    best[owner[order]] = order
    chosen = best[1:]
    survives = np.zeros(count + 1, dtype=bool)
    survives[chosen[chosen > 0]] = True
    return np.where(survives[pieces], labels, 0)


def _surviving_region_objects(labels: np.ndarray, occupied: np.ndarray, *,
                             overlap: str = "clip",
                             min_area: int = 0,
                             preserve_ids: bool = False) -> np.ndarray:
    """What is left of a region's objects once the Overlap rule has run.

    The live magnifier's Overlap rule and Min area in one place, so the box
    can draw what a click would add and the click can add exactly that. Both
    read this; nothing applies the rule twice and nothing can drift.

    IT IS COUNTED ONCE OVER THE PIXELS, NOT ONCE PER OBJECT. The loop this
    replaced ran a whole-region comparison and, under ``clip``, a whole
    connected-component pass for EVERY object, which is fine for the ten
    objects in a 128 px box and is not fine for the largest box the
    magnifier allows: on
    a 2048 px region holding 500 objects it took 6.1 s for ``clip`` and
    0.82 s for ``skip``, on the GUI thread, with the user's click waiting on
    it. The same work is 46 ms and 33 ms here.

    :param labels: the objects offered, 0 for background.
    :param occupied: where the mask already has an object, shaped like
        ``labels``.
    :param overlap: ``clip`` keeps only each object's unlabelled pixels (and
        only its largest piece, because one id names one object), ``skip``
        leaves out any object that touches one already there, ``replace``
        keeps everything.
    :param min_area: an object left smaller than this by the rule does not
        survive. 0 and 1 both mean "at least one pixel".
    :param preserve_ids: retain disconnected pieces sharing an ID; ``occupied``
        must exclude existing same-ID pixels. Validate labels as uint16 IDs.
    :returns: a copy of ``labels`` with everything the rule takes away set
        to 0. The surviving objects keep the ids they came in with.
    :raises ValueError: for an unknown ``overlap`` rule.
    """
    if overlap not in _MAGNIFIER_OVERLAP_RULES:
        raise ValueError(
            f"overlap must be one of {_MAGNIFIER_OVERLAP_RULES}, "
            f"not {overlap!r}")
    incoming = np.asarray(labels)
    if preserve_ids:
        incoming = canonical_labels(incoming, preserve_ids=True)
    taken = np.asarray(occupied, dtype=bool)
    kept = np.where(incoming > 0, incoming, 0).astype(np.int64)
    if overlap == "skip":
        touching = np.unique(kept[taken])
        touching = touching[touching > 0]
        if touching.size:
            kept[np.isin(kept, touching)] = 0
    elif overlap == "clip":
        kept[taken] = 0
        if not preserve_ids:
            kept = _largest_piece_of_each(kept)
    floor = max(1, int(min_area))
    if floor > 1 and kept.any():
        areas = np.bincount(kept.ravel())
        big = areas >= floor
        big[0] = False
        kept = np.where(big[kept], kept, 0)
    return kept



#: How many pixels the wand may EXAMINE per pixel it is allowed to change.
#:
#: The budget is on WORK, and it has to accommodate a case the change
#: budget deliberately does not bound: re-wanding an object the mask
#: already owns, with a wider tolerance, to grow it. Crossing owned
#: pixels costs nothing against ``max_pixels`` ON PURPOSE -- otherwise
#: the second click would stop at the first owned pixel and do nothing --
#: so the walk is bounded here instead.
VISIT_BUDGET_FACTOR = 4

#: The smallest visit budget, whatever ``max_pixels`` is, and the number
#: that actually matters.
#:
#: It has to be BIGGER THAN ANY OBJECT SOMEBODY RE-WANDS and smaller than
#: a frame. A hundred thousand is a 316x316 region, larger than any single
#: object in a field of cells, and it caps the pathological case -- a
#: mis-click on uniform background -- at about 0.7 s instead of the 4.8 s
#: measured on an 800x800 field or the half-minute at 2048x2048.
#:
#: It cannot be tight. A pixel count cannot tell "usefully growing a large
#: object" from "clicked on the background", because both walk until
#: tolerance stops them and on a uniform field neither does. So this is
#: set to keep every real fill working and to make the wrong click a
#: hitch rather than a hang, which is the honest trade rather than a
#: pretence that the two can be separated.
VISIT_BUDGET_FLOOR = 100_000


def magic_wand(
    image: np.ndarray,
    mask: np.ndarray,
    seed_x: int,
    seed_y: int,
    tolerance: float,
    max_pixels: int = 100_000,
    action: str = "add",
) -> np.ndarray:
    """BFS flood-fill from (seed_x, seed_y) filling pixels whose intensity
    is within `tolerance` (L2 distance) of the seed. Writes 255 (add) or
    0 (erase) into the returned mask copy.
    """
    if not (0 <= seed_y < image.shape[0] and 0 <= seed_x < image.shape[1]):
        return mask
    out = mask.copy()
    initial = image[seed_y, seed_x].astype(np.float32)
    visited = np.zeros(image.shape[:2], dtype=bool)
    q = deque([(seed_x, seed_y)])
    added = 0
    examined = 0
    visit_budget = max(VISIT_BUDGET_FACTOR * max_pixels,
                       max_pixels + VISIT_BUDGET_FLOOR)
    fill_val = 255 if action == "add" else 0
    while q and added < max_pixels and examined < visit_budget:
        cx, cy = q.popleft()
        if not (0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]):
            continue
        if visited[cy, cx]:
            continue
        visited[cy, cx] = True
        examined += 1
        cur = image[cy, cx].astype(np.float32)
        if float(np.linalg.norm(cur - initial)) > tolerance:
            continue
        if out[cy, cx] == 0 and action == "add":
            added += 1
        elif out[cy, cx] > 0 and action == "erase":
            added += 1
        out[cy, cx] = fill_val
        if added >= max_pixels:
            break
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and not visited[ny, nx]:
                q.append((nx, ny))
    return out



class MaskHistory:
    """Bounded undo/redo stack of mask arrays. Deep-copies on push so
    callers can mutate in place without corrupting older snapshots."""

    def __init__(self, capacity: int = 20):
        """Prepare an empty history with a bounded snapshot capacity.

        :param capacity: max snapshots kept in the undo (and redo) stack.
        """
        self.capacity = max(1, int(capacity))
        self._undo: deque = deque(maxlen=self.capacity)
        self._redo: deque = deque(maxlen=self.capacity)

    def clear(self) -> None:
        """Discard every snapshot from both the undo and redo stacks."""
        self._undo.clear()
        self._redo.clear()

    def push(self, mask: np.ndarray) -> None:
        """Store a deep-copy of ``mask`` and drop any redo history."""
        self._undo.append(np.array(mask, copy=True))
        self._redo.clear()

    def head(self) -> Optional[np.ndarray]:
        """The newest snapshot — what an edit in progress started from.

        Returned as held, not copied, so a caller that only wants to diff
        against it does not pay for a copy of a 16-bit field on every mouse
        release. It is already a private copy of whatever was pushed, so
        reading it cannot disturb the history; writing to it would.
        """
        return self._undo[-1] if self._undo else None

    def can_undo(self) -> bool:
        """Return True when at least one prior snapshot is available to undo to."""
        return len(self._undo) >= 2

    def can_redo(self) -> bool:
        """Return True when the redo stack has a snapshot to restore."""
        return bool(self._redo)

    def undo(self) -> Optional[np.ndarray]:
        """Pop the top snapshot, save it to the redo stack, and return the
        previous snapshot (i.e. one step back). None if not possible."""
        if not self.can_undo():
            return None
        current = self._undo.pop()
        self._redo.append(current)
        return np.array(self._undo[-1], copy=True)

    def redo(self) -> Optional[np.ndarray]:
        """Restore the most-recently-undone snapshot, or ``None`` if empty."""
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(np.array(snap, copy=True))
        return np.array(snap, copy=True)



#: Smallest side, in image pixels, a recrop box may have. A box smaller than
#: this is a mis-click or the tail of a drag that never really started, and
#: cutting it writes a field too small to hold the object it was aimed at.
RECROP_MIN_SIDE = 32

#: How much a new box may overlap one already cut out of this field, as
#: intersection over union. Above it the box is the SAME region drawn again,
#: not a second object, and it is refused rather than written: without that
#: refusal one object reached disk three times as three near-identical
#: fields, because nothing on screen said the first box had worked.
RECROP_MAX_OVERLAP = 0.5

#: Folder the retired originals are moved into, beside the images they were
#: enumerated with. Not a delete: see the note above.
RECROP_ARCHIVE_DIRNAME = "recropped_originals"

#: The record of every retirement, written inside the archive folder.
RECROP_MANIFEST = "recropped.json"

#: What separates a child crop's name from the field it was cut from:
#: ``<field>__r00``, ``<field>__r01``. Recropping a child re-uses the
#: original field's name rather than nesting, so a name says which field a
#: crop came from however many passes it took.
RECROP_INFIX = "__r"

#: The verb a recrop goes into the curation ledger under.
RECROP_KIND = "recrop"


class RecropRefused(ValueError):
    """Raised when a proposed recrop does not satisfy recropping rules.

    :ivar reason: Stable reason code: ``"no_field"``, ``"too_small"``, or
        ``"redraw"``.

    :param reason: the stable code above. Callers branch on it, so it must
        be one of the three rather than prose.
    :param message: what to say to the user. This is the exception's own
        message, so it is what an unhandled raise would print.
    """

    def __init__(self, reason: str, message: str):
        """Record why a re-crop was refused.

        :param reason: the machine-readable refusal code, kept so a caller can
            branch on it rather than parsing the message.
        :param message: the sentence shown to the user.
        """
        super().__init__(message)
        self.reason = str(reason)


def _ordered_box(shape, p0, p1) -> Tuple[int, int, int, int]:
    """``(x0, y0, x1, y1)`` for two dragged corners, clipped to ``shape``."""
    height, width = int(shape[0]), int(shape[1])
    x0, x1 = sorted((int(p0[0]), int(p1[0])))
    y0, y1 = sorted((int(p0[1]), int(p1[1])))
    return (max(0, min(width, x0)), max(0, min(height, y0)),
            max(0, min(width, x1)), max(0, min(height, y1)))


def box_overlap(a, b) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes.

    :returns: A value from 0 for disjoint boxes to 1 for identical boxes.
    """
    ax0, ay0, ax1, ay1 = (int(v) for v in a[:4])
    bx0, by0, bx1, by1 = (int(v) for v in b[:4])
    inter = (max(0, min(ax1, bx1) - max(ax0, bx0))
             * max(0, min(ay1, by1) - max(ay0, by0)))
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return (inter / union) if union > 0 else 0.0


def recrop_box(shape, p0, p1, existing=()) -> Tuple[int, int, int, int]:
    """Validate and clip a rectangular recrop selection.

    :param shape: Image shape as ``(height, width)``. Coordinates outside the
        image are clipped to this extent.
    :param p0: First selection corner as ``(x, y)``.
    :param p1: Opposite selection corner as ``(x, y)``.
    :param existing: Previously accepted boxes. A selection whose intersection
        over union exceeds :data:`RECROP_MAX_OVERLAP` is rejected.
    :returns: Validated ``(x0, y0, x1, y1)`` coordinates.
    :raises RecropRefused: If a side is shorter than
        :data:`RECROP_MIN_SIDE` or the selection duplicates an existing box.
    """
    x0, y0, x1, y1 = _ordered_box(shape, p0, p1)
    if (x1 - x0) < RECROP_MIN_SIDE or (y1 - y0) < RECROP_MIN_SIDE:
        raise RecropRefused(
            "too_small",
            f"Recrop box is {x1 - x0}x{y1 - y0} px — at least "
            f"{RECROP_MIN_SIDE} px on each side is needed.")
    for other in existing:
        if box_overlap((x0, y0, x1, y1), other) > RECROP_MAX_OVERLAP:
            name = other[4] if len(other) > 4 else "an earlier box"
            raise RecropRefused(
                "redraw",
                f"That region is already cut out as {name} — it is saved, "
                "draw the next one.")
    return (x0, y0, x1, y1)


def cut_recrop(image: np.ndarray, mask: np.ndarray,
               box) -> Tuple[np.ndarray, np.ndarray]:
    """Extract an image region and its complete labelled objects.

    Objects touching the crop boundary are removed because their masks are
    incomplete. Remaining labels are renumbered consecutively from one.

    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Crop coordinates as ``(x0, y0, x1, y1)``.
    :returns: Cropped image and relabelled mask.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image = np.ascontiguousarray(np.asarray(image)[y0:y1, x0:x1])
    sub_mask = np.ascontiguousarray(np.asarray(mask)[y0:y1, x0:x1])
    if sub_mask.size and sub_mask.max():
        edge = np.unique(np.concatenate([
            sub_mask[0, :], sub_mask[-1, :], sub_mask[:, 0], sub_mask[:, -1],
        ]))
        edge = edge[edge > 0]
        if edge.size:
            sub_mask = np.where(np.isin(sub_mask, edge), 0, sub_mask)
    relabelled, _ = _ndimage().label(sub_mask > 0, structure=_EIGHT)
    return sub_image, relabelled.astype(np.uint16)


def _recrop_base(filename: str) -> str:
    """The field name a child crop is named after.

    A recrop of a recrop is named after the ORIGINAL field, not after its
    parent: ``well_A1__r00`` recropped again yields ``well_A1__r03``, never
    ``well_A1__r00__r00``. Nesting would make the name grow with every pass
    while saying nothing more than the manifest already records. A bundle's
    ``_seg`` is not part of its name: ``well_A1_seg.npy`` yields
    ``well_A1__r00_seg.npy``.
    """
    return field_stem(filename).split(RECROP_INFIX)[0]


def recrop_child_name(folder: str, filename: str, ext: str = ".tif",
                      masks_dir: Optional[str] = None) -> str:
    """Return the next unused ``<field>__rNN`` filename.

    Names are checked against the image queue, mask directory, and recrop
    archive to prevent overwriting output from an earlier editing session.

    :param folder: the folder the editor opened.
    :param filename: the field being cut.
    :param ext: the child image's extension. Ignored for a bundle, whose
        child is a bundle too, ``<field>__rNN_seg.npy``, because a TIFF
        written into a folder of bundles would make it two layouts at once.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: the child's file name.
    """
    base = _recrop_base(filename)
    archive = os.path.join(folder, RECROP_ARCHIVE_DIRNAME)
    masks = masks_folder(folder, masks_dir)
    index = 0
    while True:
        stem = f"{base}{RECROP_INFIX}{index:02d}"
        taken = [os.path.join(masks, stem + ".tif"),
                 os.path.join(archive, "masks", stem + ".tif")]
        taken += [os.path.join(d, stem + e)
                  for d in (folder, archive)
                  for e in IMAGE_EXTS + (SEG_SUFFIX,)]
        if not any(os.path.exists(path) for path in taken):
            return stem + (SEG_SUFFIX if is_seg_bundle(filename) else ext)
        index += 1


class Recrop(NamedTuple):
    """Result of writing a recropped image and mask.

    :ivar name: Filename assigned to the recropped field.
    :ivar image_path: Path of the written image.
    :ivar mask_path: Path of the written label mask.
    :ivar n_objects: Number of complete labelled objects retained.
    :ivar box: Source coordinates as ``(x0, y0, x1, y1)``.
    """

    name: str
    image_path: str
    mask_path: str
    n_objects: int
    box: Tuple[int, int, int, int]


def write_recrop(folder: str, filename: str, image: np.ndarray,
                 mask: np.ndarray, box,
                 masks_dir: Optional[str] = None) -> "Recrop":
    """Write a recropped field, mask, and curation record.

    The image is stored as an unscaled uint16 TIFF beside the source images,
    and the relabelled mask is stored in ``<folder>/masks``. The curation
    record distinguishes deliberately removed boundary objects from missed
    segmentation objects.

    A field cut from a ``_seg.npy`` bundle becomes a bundle of its own,
    holding ``img``, ``masks``, empty ``flows`` and a ``filename`` naming the
    parent and the box -- the form the external curation tool gave its
    recrops, so either tool can open the other's.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param image: Source microscopy image.
    :param mask: Label image aligned with ``image``.
    :param box: Coordinates returned by :func:`recrop_box`.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: Filename and retained-object count for the new field.
    """
    x0, y0, x1, y1 = (int(v) for v in box[:4])
    sub_image, sub_mask = cut_recrop(image, mask, (x0, y0, x1, y1))
    child = recrop_child_name(folder, filename, masks_dir=masks_dir)
    image_path = os.path.join(folder, child)
    mask_path = mask_save_path(folder, child, masks_dir)
    if is_seg_bundle(child):
        _write_bundle(image_path, {
            "img": np.asarray(sub_image).astype(np.uint16),
            "masks": sub_mask,
            "flows": [None, None, None],
            "filename": (f"recrop of {field_stem(filename)} "
                         f"[{x0}:{x1},{y0}:{y1}]"),
        })
    else:
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        write_tiff(image_path, np.asarray(sub_image).astype(np.uint16))
        write_tiff(mask_path, sub_mask)
    log = CurationLog(mask_path, source=CURATION_SOURCE)
    log.append(RECROP_KIND, child,
               n_changed=int(np.count_nonzero(sub_mask)),
               parent=os.path.basename(str(filename)),
               box=[x0, y0, x1, y1],
               n_objects=int(sub_mask.max()))
    log.write_beside(mask_path)
    return Recrop(child, image_path, mask_path,
                  int(sub_mask.max()), (x0, y0, x1, y1))


def recrop_archive_dir(folder: str) -> str:
    """Return the archive directory for recropped source fields."""
    return os.path.join(folder, RECROP_ARCHIVE_DIRNAME)


def retire_recropped_original(folder: str, filename: str, *,
                              children=(), boxes=(),
                              masks_dir: Optional[str] = None) -> dict:
    """Archive a source field after recropped children have been created.

    The source image, mask, and curation ledger are moved to
    ``<folder>/recropped_originals`` and recorded in :data:`RECROP_MANIFEST`.
    This removes the multi-object source from the training queue without
    deleting it. A bundle goes with its ledger and with any display image of
    its stem beside it, as the external curation tool moved its ``.png``.

    :param folder: Image-queue directory.
    :param filename: Source image filename.
    :param children: Filenames created from the source field.
    :param boxes: Crop boxes corresponding to ``children``.
    :param masks_dir: where the masks are, when not in ``<folder>/masks``.
    :returns: Manifest record, including the original and archived paths.
    """
    archive = recrop_archive_dir(folder)
    name = os.path.basename(str(filename))
    if is_seg_bundle(filename):
        bundle = os.path.join(folder, name)
        moves = [(bundle, os.path.join(archive, name)),
                 (bundle + LOG_SUFFIX,
                  os.path.join(archive, name + LOG_SUFFIX))]
        moves += [(os.path.join(folder, field_stem(name) + ext),
                   os.path.join(archive, field_stem(name) + ext))
                  for ext in IMAGE_EXTS]
    else:
        mask_path = mask_save_path(folder, filename, masks_dir)
        moves = [
            (os.path.join(folder, filename),
             os.path.join(archive, name)),
            (mask_path, os.path.join(archive, "masks",
                                     os.path.basename(mask_path))),
            (mask_path + LOG_SUFFIX,
             os.path.join(archive, "masks",
                          os.path.basename(mask_path) + LOG_SUFFIX)),
        ]
    moved = []
    for source, target in moves:
        if not os.path.exists(source):
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.replace(source, target)
        moved.append([source, target])
    record = {
        "original": os.path.basename(str(filename)),
        "children": [str(c) for c in children],
        "boxes": [[int(v) for v in box[:4]] for box in boxes],
        "moved": moved,
    }
    _append_recrop_manifest(archive, record)
    return record


def _append_recrop_manifest(archive: str, record: dict) -> str:
    """Append one retirement to the archive's manifest; return its path.

    Read-modify-write of a list rather than a line per record, so the file
    is ordinary JSON that anything can open. A manifest that cannot be read
    is replaced rather than allowed to stop the retirement: the files
    themselves are the recovery, and the manifest is the map to them.
    """
    os.makedirs(archive, exist_ok=True)
    path = os.path.join(archive, RECROP_MANIFEST)
    records = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, list):
                records = loaded
        except (OSError, ValueError):
            records = []
    records.append(record)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=1)
    return path


def read_recrop_manifest(folder: str) -> List[dict]:
    """Return recrop-archive records in chronological order."""
    path = os.path.join(recrop_archive_dir(folder), RECROP_MANIFEST)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        return []
    return loaded if isinstance(loaded, list) else []


def restore_recropped_original(folder: str, original: str) -> List[str]:
    """Restore an archived source field to the image queue.

    Existing archived files listed for ``original`` are moved back to their
    source locations. Recropped child fields are not modified.

    :param folder: Image-queue directory.
    :param original: Original source filename recorded in the manifest.
    :returns: Paths restored to the queue.
    """
    restored: List[str] = []
    name = os.path.basename(str(original))
    for record in reversed(read_recrop_manifest(folder)):
        if record.get("original") != name:
            continue
        for pair in record.get("moved") or ():
            if len(pair) != 2:
                continue
            source, target = pair
            if not os.path.exists(target) or os.path.exists(source):
                continue
            os.makedirs(os.path.dirname(source), exist_ok=True)
            os.replace(target, source)
            restored.append(source)
        break
    return restored
