"""
Cellpose mask I/O — write + read masks as EITHER TIFF or NumPy.

Introduces two helpers to replace scattered ``np.save``/``np.load``
calls across :mod:`spacr.object` and :mod:`spacr.io`:

* :func:`save_mask` — writes a single 2-D uint16 mask. Emits
  ``.tif`` by default (smaller on disk + openable in Fiji /
  napari) but ``.npy`` is still supported via ``fmt="npy"`` for
  users who prefer numpy round-trips.
* :func:`load_mask` — reads a mask regardless of on-disk format.
  Given a path with no suffix, or a stem, it probes ``.tif``,
  ``.tiff`` and ``.npy`` in that order and returns the first hit.

Both helpers preserve uint16 dtype (spaCR's mask convention) and
return arrays with the same shape ``(H, W)``. TIFF is written with
LZW compression via ``tifffile``.

Migration path for existing scripts:

    # BEFORE
    np.save("foo_mask.npy", mask.astype(np.uint16))
    m = np.load("foo_mask.npy")

    # AFTER
    save_mask("foo_mask", mask)       # writes foo_mask.tif
    m = load_mask("foo_mask")         # finds foo_mask.tif or foo_mask.npy

The env var ``SPACR_MASK_FORMAT`` overrides the default across
the whole process: set to ``npy`` to keep the old behaviour.

Masks as ROIs: QuPath GeoJSON, ImageJ RoiSet and COCO
------------------------------------------------------

The second half of this module moves label masks to and from the formats
other tools curate and train on (item 545): :func:`masks_to_geojson` and
:func:`geojson_to_masks` for QuPath, :func:`masks_to_roiset` and
:func:`roiset_to_masks` for Fiji's ROI Manager, :func:`masks_to_coco` and
:func:`coco_to_masks` for COCO-style training sets, and
:func:`export_rois` / :func:`import_rois` choosing among them by file name.

Every object is one feature, ROI or annotation, carrying its object type
(``cell``, ``nucleus`` ...), its object id and its class; the class defaults to
the object type. Each object is traced on its own, so objects that touch stay
separate. Outlines follow PIXEL EDGES, not pixel centres: a pixel ``(row,
col)`` is the square from ``x = col`` to ``col + 1`` and ``y = row`` to
``row + 1``, the convention QuPath and ImageJ both use. A pixel is inside a
polygon when its centre is, so a traced outline rasterised back gives the
object exactly: no tolerance, holes and several-part objects included. A hole
is an interior ring (GeoJSON), a path of a composite ROI (ImageJ) or, since a
COCO polygon cannot have one, an RLE segmentation (COCO).

``roifile`` is needed only for ImageJ files and is imported when one is read
or written. COCO RLE is encoded and decoded here, so ``pycocotools`` is never
required; the files it writes are the ones ``pycocotools`` reads.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import numpy as np

LOG = logging.getLogger("spacr.mask_io")

PathLike = Union[str, os.PathLike, Path]

_requested_format = os.environ.get("SPACR_MASK_FORMAT", "tif").lower()
if _requested_format not in ("tif", "tiff", "npy"):
    LOG.warning("SPACR_MASK_FORMAT=%r not recognised; using 'tif'",
                _requested_format)
    _requested_format = "tif"

#: Mask file format used when a caller does not name one, overridable with
#: the ``SPACR_MASK_FORMAT`` environment variable. Assigned ONCE: autoapi
#: documents every module-level binding, so the earlier write-then-correct
#: form produced two entries for this constant on the API page.
DEFAULT_FORMAT = _requested_format


def _as_uint16_mask(mask: np.ndarray) -> np.ndarray:
    """Convert labels without narrowing, wrapping or renumbering identities.

    :param mask: real integer-valued labels; spatial shape is preserved.
    :returns: uint16 labels, sharing storage when conversion is unnecessary.
    :raises ValueError: labels are nonfinite, fractional, negative, nonnumeric,
        or exceed the uint16 capacity. Input values are never modified.
    """
    labels = np.asarray(mask)
    if labels.dtype.kind not in 'buif':
        raise ValueError('Mask labels must be real nonnegative integers fitting uint16')
    if labels.size:
        if not np.isfinite(labels).all() or np.min(labels) < 0:
            raise ValueError('Mask labels must be finite nonnegative integers fitting uint16')
        top = np.max(labels)
        if top > np.iinfo(np.uint16).max:
            raise ValueError(f'Mask holds object id {top}, which does not fit in uint16; '
                             'labels were not renumbered or saved')
        if labels.dtype.kind == 'f' and np.any(labels != np.floor(labels)):
            raise ValueError('Mask labels must be integer-valued before conversion to uint16')
    return labels.astype(np.uint16, copy=False)


def save_mask(path: PathLike, mask: np.ndarray,
                fmt: str = None) -> Path:
    """Write a Cellpose mask to disk.

    :param path: destination — extension is optional. ``foo``,
        ``foo.tif``, ``foo.npy`` all work.
    :param mask: 2-D integer mask; will be cast to uint16.
    :param fmt: force a format (``"tif"``, ``"tiff"``, or ``"npy"``).
        Defaults to :data:`DEFAULT_FORMAT` (env-overridable).
    :returns: the resolved on-disk path.
    :raises ValueError: labels are nonfinite, negative, fractional or will
        not fit in uint16, or ``fmt`` names a format this module cannot write.

    An id above 65535 is refused rather than cast. The cast wraps, and the
    first value it wraps to is 0 — background — so object 65536 would come
    back from disk fused with everything that was never segmented at all,
    and nothing downstream could tell that from a mask with one fewer object.
    """
    fmt = (fmt or DEFAULT_FORMAT).lower().lstrip(".")
    p = Path(path)

    mask = _as_uint16_mask(mask).copy()

    if p.suffix.lower() in (".tif", ".tiff", ".npy"):
        fmt = p.suffix.lower().lstrip(".")

    if fmt in ("tif", "tiff"):
        p = p.with_suffix(f".{fmt}")
        try:
            write_tiff = import_module(".tiff_io", __package__).write_tiff
        except Exception:
            LOG.warning("tifffile missing — falling back to npy for %s", p)
            return save_mask(path, mask, fmt="npy")
        write_tiff(
            str(p), mask, compression="lzw",
        )
    elif fmt == "npy":
        p = p.with_suffix(".npy")
        np.save(str(p), mask)
    else:
        raise ValueError(f"unknown mask format: {fmt!r}")
    return p


def load_mask(path: PathLike) -> np.ndarray:
    """Read a Cellpose mask regardless of on-disk format.

    :param path: mask file or extensionless stem to resolve and load.

    Accepts a full path (``foo.tif`` / ``foo.npy``) OR a stem
    (``foo``) — in the stem case, tif → tiff → npy is tried and the
    first extant file is loaded.

    :returns: uint16 2-D array. Shape unchanged.
    :raises FileNotFoundError: when no matching file exists.
    :raises ValueError: stored labels cannot be represented exactly as uint16.
    """
    p = Path(path)
    candidates = []
    if p.suffix.lower() in (".tif", ".tiff", ".npy"):
        candidates.append(p)
    else:
        stem = p.with_suffix("")
        candidates.extend([
            stem.with_suffix(".tif"),
            stem.with_suffix(".tiff"),
            stem.with_suffix(".npy"),
        ])

    for c in candidates:
        if c.is_file():
            return _read_one(c)
    raise FileNotFoundError(f"no mask found for {path!r} (tried "
                              f"{[str(c.name) for c in candidates]})")


def _read_one(p: Path) -> np.ndarray:
    """Read one TIFF or NumPy mask and return it as unsigned 16-bit data.

    :param p: Mask file with a ``.tif``, ``.tiff``, or ``.npy`` suffix.
    :returns: Mask array cast to ``numpy.uint16`` without changing its shape.
    :raises ValueError: ``p`` has an unsupported suffix or its stored labels
        cannot be represented exactly as uint16.
    """
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(p))
    elif p.suffix.lower() == ".npy":
        arr = np.load(str(p), allow_pickle=False)
    else:
        raise ValueError(f"unsupported mask extension: {p}")
    return _as_uint16_mask(arr)



#: The three ROI formats, as :func:`export_rois` and :func:`import_rois` name
#: them.
ROI_FORMATS = ("geojson", "imagej", "coco")

#: What an object is called when a single mask is handed over without a type.
DEFAULT_OBJECT_TYPE = "object"

#: The QuPath object kinds :func:`masks_to_geojson` can write.
#: ``"annotation"`` (the default) is editable in QuPath; ``"detection"`` is
#: lighter and is what QuPath's own cell detection makes.
QUPATH_OBJECT_KINDS = ("annotation", "detection")

_ROI_NAMESPACE = uuid.UUID("0c9a5f56-3a1e-5c56-9d3b-5a4c52a54545")
_NAME_PATTERN = re.compile(r"^\s*(?P<type>.+?)[\s_-]+(?P<id>\d+)\s*$")
_DIRECTIONS_X = (1, 0, -1, 0)
_DIRECTIONS_Y = (0, 1, 0, -1)


def _optional_module(module: str, package: str, purpose: str):
    """Import an optional dependency, or say exactly how to install it.

    :param module: the module to import.
    :param package: the pip package that provides it.
    :param purpose: what needs it, for the message.
    :returns: the imported module.
    :raises ImportError: the module is not installed; the message names the
        ``pip install`` line.
    """
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{purpose} needs the optional package '{package}', which is not "
            f"installed. Install it with:  pip install {package}") from exc


def _label_array(mask: Any, name: str) -> np.ndarray:
    """Check one label mask and return it as an unsigned integer array.

    :param mask: 2-D labels, 0 for background.
    :param name: the object type, for the messages.
    :returns: the labels as ``uint32`` (or ``uint64`` when they need it).
    :raises ValueError: the mask is not 2-D or its labels are not
        nonnegative integers.
    """
    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise ValueError(f"the {name} mask must be 2-D, got shape {labels.shape}")
    if labels.dtype.kind not in "biuf":
        raise ValueError(f"the {name} mask must hold integer labels")
    if labels.size:
        if labels.dtype.kind == "f" and (not np.isfinite(labels).all()
                                         or np.any(labels != np.floor(labels))):
            raise ValueError(f"the {name} mask must hold integer labels")
        if np.min(labels) < 0:
            raise ValueError(f"the {name} mask holds negative labels")
    wide = bool(labels.size) and np.max(labels) > np.iinfo(np.uint32).max
    return labels.astype(np.uint64 if wide else np.uint32, copy=False)


def _object_masks(masks: Any, object_type: Optional[str]) -> Dict[str, np.ndarray]:
    """Normalise one mask or a mapping of masks to ``{object type: labels}``.

    :param masks: a 2-D label mask, or a mapping of object type to mask.
    :param object_type: the type of a single mask; ignored for a mapping.
    :returns: the masks by object type, all the same shape.
    :raises ValueError: no mask, or masks of different shapes.
    """
    if isinstance(masks, Mapping):
        items = [(str(key), value) for key, value in masks.items()]
    else:
        items = [(str(object_type or DEFAULT_OBJECT_TYPE), masks)]
    if not items:
        raise ValueError("no masks to export")
    out = {name: _label_array(value, name) for name, value in items}
    shapes = {labels.shape for labels in out.values()}
    if len(shapes) != 1:
        raise ValueError(f"the masks differ in shape: {sorted(shapes)}")
    return out


def _object_classes(classes: Any, types: List[str]) -> Dict[str, Dict[int, str]]:
    """Normalise a class map to ``{object type: {object id: class}}``.

    :param classes: ``None``; ``{id: class}`` for a single object type; or
        ``{object type: {id: class}}``.
    :param types: the object types being exported.
    :returns: the class of every object that has one.
    :raises ValueError: a flat ``{id: class}`` map given for several types.
    """
    if not classes:
        return {}
    values = list(classes.values())
    if all(isinstance(value, Mapping) for value in values):
        return {str(key): {int(i): str(c) for i, c in value.items()}
                for key, value in classes.items()}
    if len(types) != 1:
        raise ValueError("a flat {object id: class} map needs exactly one "
                         "object type; pass {object type: {id: class}}")
    return {types[0]: {int(i): str(c) for i, c in classes.items()}}


def _trace_rings(binary: np.ndarray) -> List[np.ndarray]:
    """Trace the pixel-edge outlines of a binary image.

    Every edge between a foreground and a background pixel is used once. The
    edges run with the object on their right in image coordinates (``y``
    down), so an outer ring has a positive shoelace area and a hole a
    negative one. Where two foreground pixels meet only at a corner the trace
    turns towards the pixel it is following, keeping the two apart.

    :param binary: 2-D boolean image.
    :returns: closed rings, each an ``(n, 2)`` integer array of ``x, y``
        corner coordinates whose last point repeats the first; only corners
        where the outline turns are kept.
    """
    padded = np.pad(np.asarray(binary, dtype=bool), 1)
    inner = padded[1:-1, 1:-1]
    width = inner.shape[1]
    parts = []
    for direction, neighbour, start in (
            (0, padded[:-2, 1:-1], (0, 0)),
            (1, padded[1:-1, 2:], (1, 0)),
            (2, padded[2:, 1:-1], (1, 1)),
            (3, padded[1:-1, :-2], (0, 1))):
        rows, cols = np.nonzero(inner & ~neighbour)
        parts.append((cols + start[0], rows + start[1],
                      np.full(rows.shape, direction, dtype=np.int64)))
    xs = np.concatenate([part[0] for part in parts])
    ys = np.concatenate([part[1] for part in parts])
    directions = np.concatenate([part[2] for part in parts])
    if not xs.size:
        return []
    stride = width + 2
    starts = (ys * stride + xs).tolist()
    ends = ((ys + np.take(_DIRECTIONS_Y, directions)) * stride
            + xs + np.take(_DIRECTIONS_X, directions)).tolist()
    dirs = directions.tolist()
    outgoing: Dict[int, List[int]] = {}
    for index, key in enumerate(starts):
        outgoing.setdefault(key, []).append(index)
    used = [False] * len(starts)
    rings = []
    for first in range(len(starts)):
        if used[first]:
            continue
        chain = []
        edge = first
        while True:
            used[edge] = True
            chain.append(edge)
            candidates = outgoing[ends[edge]]
            if len(candidates) == 1:
                edge = candidates[0]
            else:
                turn = (dirs[edge] + 1) % 4
                edge = next(c for c in candidates if dirs[c] == turn)
            if edge == first:
                break
        chain_dirs = np.asarray([dirs[e] for e in chain])
        corners = np.nonzero(chain_dirs != np.roll(chain_dirs, 1))[0]
        if not corners.size:
            corners = np.asarray([0])
        picked = [chain[i] for i in corners.tolist()]
        ring = np.stack([xs[picked], ys[picked]], axis=1).astype(np.int64)
        rings.append(np.vstack([ring, ring[:1]]))
    return rings


def _ring_area(ring: np.ndarray) -> float:
    """Signed shoelace area of a closed ring, positive for an outer ring.

    :param ring: closed ``(n, 2)`` array of ``x, y`` points.
    :returns: the signed area.
    """
    x = np.asarray(ring[:, 0], dtype=float)
    y = np.asarray(ring[:, 1], dtype=float)
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _split_ring(ring: np.ndarray) -> List[np.ndarray]:
    """Split a ring that touches itself at a corner into simple loops.

    Where an object meets itself only at a pixel corner the traced outline
    passes that corner twice. The polygon is the same either way, but a ring
    touching itself is invalid to GEOS and JTS (so to shapely and QuPath),
    and a hole touching its shell at a point is valid; so the ring is cut at
    each repeated corner into loops whose orientation says which they are.

    :param ring: closed ``(n, 2)`` integer array.
    :returns: closed simple loops that together trace the same edges.
    """
    stack: List[Tuple[int, int]] = []
    where: Dict[Tuple[int, int], int] = {}
    loops = []
    for point in map(tuple, ring[:-1].tolist()):
        if point in where:
            start = where[point]
            loop = stack[start:]
            for other in loop[1:]:
                del where[other]
            del stack[start + 1:]
            loops.append(np.asarray(loop + loop[:1], dtype=np.int64))
        else:
            where[point] = len(stack)
            stack.append(point)
    loops.append(np.asarray(stack + stack[:1], dtype=np.int64))
    return [loop for loop in loops if len(loop) >= 4]


def _inside_point(ring: np.ndarray) -> np.ndarray:
    """The centre of a pixel just to the left of a ring's first edge.

    For a hole that is a pixel of the hole, so it is a point that lies
    inside whatever shell holds the hole and on no ring's edge.

    :param ring: closed ``(n, 2)`` integer array.
    :returns: ``[x, y]``.
    """
    step = np.sign(ring[1] - ring[0])
    return ring[0] + 0.5 * step + 0.5 * np.asarray([step[1], -step[0]])


def _crop_polygons(parts: np.ndarray, count: int, top: int,
                   left: int) -> List[List[np.ndarray]]:
    """Trace the labelled parts of one object's crop into polygons.

    :param parts: the crop, its 4-connected parts numbered 1..count.
    :param count: how many parts there are.
    :param top: the crop's first row in the full mask.
    :param left: the crop's first column in the full mask.
    :returns: one ``[exterior, holes...]`` list per polygon, in full-mask
        coordinates.
    """
    from skimage.measure import points_in_poly

    polygons = []
    offset = np.asarray([left, top])
    for part in range(1, count + 1):
        loops = [loop + offset for ring in _trace_rings(parts == part)
                 for loop in _split_ring(ring)]
        outer = sorted((loop for loop in loops if _ring_area(loop) > 0),
                       key=_ring_area)
        groups: List[List[np.ndarray]] = [[loop] for loop in outer]
        for hole in (loop for loop in loops if _ring_area(loop) < 0):
            point = _inside_point(hole)[None, :]
            home = next((group for group in groups
                         if points_in_poly(point, group[0])[0]), groups[-1])
            home.append(hole)
        polygons.extend(reversed(groups))
    return polygons


def object_polygons(mask: np.ndarray, label: int) -> List[List[np.ndarray]]:
    """The polygons that outline one object of a label mask, exactly.

    :param mask: 2-D label mask.
    :param label: the object id to outline.
    :returns: one ``[exterior, hole, hole, ...]`` list per 4-connected part
        of the object; each ring is a closed ``(n, 2)`` integer array of
        ``x, y`` pixel-corner coordinates. Empty when the id is absent.
    """
    from scipy import ndimage

    labels = np.asarray(mask)
    rows, cols = np.nonzero(labels == label)
    if not rows.size:
        return []
    top, left = int(rows.min()), int(cols.min())
    crop = labels[top:int(rows.max()) + 1, left:int(cols.max()) + 1] == label
    parts, count = ndimage.label(crop)
    return _crop_polygons(parts, count, top, left)


def _iter_objects(mask: np.ndarray) -> Iterator[Tuple[int, List[List[np.ndarray]]]]:
    """Every object of a label mask with its polygons, in id order.

    :param mask: 2-D label mask.
    :returns: an iterator of ``(object id, polygons)``; see
        :func:`object_polygons` for the polygons.
    """
    from scipy import ndimage

    labels = np.asarray(mask)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if not ids.size:
        return
    compact = (np.searchsorted(ids, labels) + 1).astype(np.int64)
    compact[labels == 0] = 0
    for index, box in enumerate(ndimage.find_objects(compact)):
        if box is None:
            continue
        parts, count = ndimage.label(compact[box] == index + 1)
        yield int(ids[index]), _crop_polygons(
            parts, count, box[0].start, box[1].start)


def _fill_rings(rings: List[Any], shape: Tuple[int, int], *,
                even_odd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """The pixels whose centres fall inside a polygon.

    :param rings: ``[exterior, holes...]``, each an ``(n, 2)`` array of
        ``x, y`` pixel-corner coordinates, closed or not; with
        ``even_odd`` every ring toggles instead.
    :param shape: the mask's ``(rows, columns)``; pixels outside are dropped.
    :param even_odd: fill by the even-odd rule over all rings, the rule of
        an ImageJ composite ROI, rather than exterior-minus-holes.
    :returns: ``(rows, columns)`` of the pixels inside.
    """
    from skimage.draw import polygon as draw_polygon

    arrays = [np.asarray(ring, dtype=float).reshape(-1, 2) for ring in rings]
    arrays = [ring for ring in arrays if len(ring) >= 3]
    empty = (np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp))
    if not arrays:
        return empty
    outline = arrays if even_odd else arrays[:1]
    points = np.vstack(outline)
    top = max(int(np.floor(points[:, 1].min())), 0)
    left = max(int(np.floor(points[:, 0].min())), 0)
    bottom = min(int(np.ceil(points[:, 1].max())), shape[0])
    right = min(int(np.ceil(points[:, 0].max())), shape[1])
    if bottom <= top or right <= left:
        return empty
    box = np.zeros((bottom - top, right - left), dtype=bool)
    for index, ring in enumerate(arrays):
        rr, cc = draw_polygon(ring[:, 1] - 0.5 - top, ring[:, 0] - 0.5 - left,
                              box.shape)
        if even_odd:
            box[rr, cc] ^= True
        elif index == 0:
            box[rr, cc] = True
        else:
            box[rr, cc] = False
    rows, cols = np.nonzero(box)
    return rows + top, cols + left


def _new_labels(shape: Tuple[int, int], top: int) -> np.ndarray:
    """An empty label mask wide enough for ids up to ``top``.

    :param shape: ``(rows, columns)``.
    :param top: the largest id it will hold.
    :returns: zeros of ``uint16`` when ``top`` fits, else ``uint32``.
    """
    dtype = np.uint16 if top <= np.iinfo(np.uint16).max else np.uint32
    return np.zeros((int(shape[0]), int(shape[1])), dtype=dtype)


def _paint(entries: List[Tuple[str, Optional[int], str, List[Any]]],
           shape: Tuple[int, int]) -> Tuple[Dict[str, np.ndarray],
                                            Dict[str, Dict[int, str]]]:
    """Rasterise imported objects into one label mask per object type.

    Objects keep the id they were exported with. An object with no id, or
    with an id already taken in its type, gets the next free one. Where two
    objects overlap, the one listed later wins the shared pixels.

    :param entries: ``(object type, id or None, class, pixel sets)`` per
        object, each pixel set a ``(rows, columns)`` pair.
    :param shape: the image's ``(rows, columns)``.
    :returns: ``(masks by object type, {object type: {id: class}})``.
    """
    wanted_ids: Dict[str, set] = {}
    for kind, wanted, _cls, _pixels in entries:
        wanted_ids.setdefault(kind, set())
        if wanted is not None and wanted > 0:
            wanted_ids[kind].add(int(wanted))
    claimed: Dict[str, set] = {kind: set() for kind in wanted_ids}
    numbers = []
    for kind, wanted, _cls, _pixels in entries:
        number = int(wanted) if wanted is not None and wanted > 0 else None
        if number is None or number in claimed[kind]:
            number = max(claimed[kind] | wanted_ids[kind] | {0}) + 1
            wanted_ids[kind].add(number)
        claimed[kind].add(number)
        numbers.append(number)
    masks: Dict[str, np.ndarray] = {}
    classes: Dict[str, Dict[int, str]] = {}
    for (kind, _wanted, cls, pixels), number in zip(entries, numbers):
        if kind not in masks:
            masks[kind] = _new_labels(shape, max(claimed[kind]))
            classes[kind] = {}
        for rows, cols in pixels:
            masks[kind][rows, cols] = number
        classes[kind][number] = cls
    for kind, labels in masks.items():
        present = set(np.unique(labels).tolist())
        classes[kind] = {k: v for k, v in classes[kind].items() if k in present}
    return masks, classes


def _class_colour(name: str) -> List[int]:
    """A stable RGB colour for a class name, so a class looks the same twice.

    :param name: the class.
    :returns: ``[r, g, b]`` in 0-255.
    """
    digest = uuid.uuid5(_ROI_NAMESPACE, name).bytes
    return [64 + digest[0] % 192, 64 + digest[1] % 192, 64 + digest[2] % 192]


def _ring_list(ring: np.ndarray) -> List[List[int]]:
    """A closed ring as JSON-ready ``[[x, y], ...]``.

    :param ring: closed ``(n, 2)`` integer array.
    :returns: the points as plain lists of ints.
    """
    return [[int(x), int(y)] for x, y in ring]


def _parse_name(name: Any) -> Tuple[Optional[str], Optional[int]]:
    """Read ``"<type> <id>"`` (or ``<type>-<id>``) back out of an ROI name.

    The type must hold a letter, so ImageJ's own ``"0012-0034"`` names,
    which are positions, are not read as a type and an id.

    :param name: the name; anything else gives ``(None, None)``.
    :returns: ``(object type, id)``, either ``None`` when absent.
    """
    match = _NAME_PATTERN.match(str(name or ""))
    if not match or not re.search(r"[^\W\d_]", match.group("type")):
        return None, None
    return match.group("type"), int(match.group("id"))


def _as_int(value: Any) -> Optional[int]:
    """An id read from a file, or ``None`` when it is not a whole number.

    :param value: the stored value.
    :returns: the int, or ``None``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number != int(number):
        return None
    return int(number)


def masks_to_geojson(masks: Any, *, classes: Any = None,
                     object_type: Optional[str] = None,
                     object_kind: str = "annotation",
                     image_name: str = "") -> Dict[str, Any]:
    """Label masks as a QuPath GeoJSON ``FeatureCollection``.

    One feature per object: a ``Polygon``, or a ``MultiPolygon`` for an
    object in several parts, with holes as interior rings and coordinates in
    pixels (pixel corners, ``x`` right and ``y`` down). The properties are
    QuPath's own -- ``objectType``, ``classification`` (``name`` and
    ``color``), ``name`` and ``isLocked`` -- plus ``object_type`` and
    ``object_id``, and ``measurements.object_id`` so the id survives a
    round trip through QuPath, which keeps measurements and names. The
    feature ``id`` is a UUID derived from the image name, the object type
    and the id, so exporting the same mask twice writes the same file.

    :param masks: a 2-D label mask, or ``{object type: mask}``.
    :param classes: the class of each object: ``{id: class}`` for a single
        mask or ``{object type: {id: class}}``; an object without one is
        classified as its object type.
    :param object_type: the type of a single mask (default ``"object"``).
    :param object_kind: ``"annotation"`` or ``"detection"``, the QuPath
        object each feature becomes.
    :param image_name: the image these masks belong to, used to make the
        feature ids unique across images.
    :returns: the ``FeatureCollection`` as a JSON-ready dict.
    :raises ValueError: an unknown ``object_kind`` or an invalid mask.
    """
    if object_kind not in QUPATH_OBJECT_KINDS:
        raise ValueError(f"object_kind must be one of {QUPATH_OBJECT_KINDS}")
    by_type = _object_masks(masks, object_type)
    class_map = _object_classes(classes, list(by_type))
    features = []
    for kind, labels in by_type.items():
        for number, polygons in _iter_objects(labels):
            cls = class_map.get(kind, {}).get(number, kind)
            coords = [[_ring_list(ring) for ring in polygon]
                      for polygon in polygons]
            geometry = ({"type": "Polygon", "coordinates": coords[0]}
                        if len(coords) == 1 else
                        {"type": "MultiPolygon", "coordinates": coords})
            ident = uuid.uuid5(_ROI_NAMESPACE, f"{image_name}|{kind}|{number}")
            features.append({
                "type": "Feature",
                "id": str(ident),
                "geometry": geometry,
                "properties": {
                    "objectType": object_kind,
                    "classification": {"name": cls,
                                       "color": _class_colour(cls)},
                    "name": f"{kind} {number}",
                    "isLocked": False,
                    "object_type": kind,
                    "object_id": number,
                    "measurements": {"object_id": number},
                },
            })
    return {"type": "FeatureCollection", "features": features}


def _geojson_features(data: Any) -> List[Mapping[str, Any]]:
    """The features of a GeoJSON document, however it is wrapped.

    :param data: a ``FeatureCollection``, a ``Feature``, a bare geometry or
        a list of features (QuPath writes the list form too).
    :returns: the features.
    :raises ValueError: the document is none of those.
    """
    if isinstance(data, list):
        return [f for f in data if isinstance(f, Mapping)]
    if isinstance(data, Mapping):
        kind = data.get("type")
        if kind == "FeatureCollection":
            return list(data.get("features") or [])
        if kind == "Feature":
            return [data]
        if kind in ("Polygon", "MultiPolygon", "GeometryCollection"):
            return [{"type": "Feature", "geometry": data, "properties": {}}]
    raise ValueError("not a GeoJSON FeatureCollection, Feature or feature list")


def _geometry_polygons(geometry: Any) -> List[List[Any]]:
    """The polygons of a GeoJSON geometry; points and lines have none.

    :param geometry: a GeoJSON geometry object.
    :returns: ``[exterior, holes...]`` coordinate lists.
    """
    if not isinstance(geometry, Mapping):
        return []
    kind = geometry.get("type")
    if kind == "Polygon":
        return [geometry.get("coordinates") or []]
    if kind == "MultiPolygon":
        return list(geometry.get("coordinates") or [])
    if kind == "GeometryCollection":
        return [polygon for part in geometry.get("geometries") or []
                for polygon in _geometry_polygons(part)]
    return []


def geojson_to_masks(data: Any, shape: Tuple[int, int], *,
                     object_type: Optional[str] = None,
                     with_classes: bool = False):
    """Rasterise a (QuPath) GeoJSON document into label masks.

    A pixel belongs to an object when its centre lies inside the polygon
    and outside its holes; an outline written by :func:`masks_to_geojson`
    therefore comes back pixel for pixel. The object type is, in order,
    ``object_type``, the feature's ``object_type`` property, the type in a
    ``"<type> <id>"`` name, or its classification -- so a QuPath project
    that was never in spaCR gives one mask per class. The id is the
    ``object_id`` property or measurement, or the id in the name; objects
    without one are numbered after the rest.

    :param data: the parsed document, or a path to a ``.geojson`` file.
    :param shape: the image's ``(rows, columns)``.
    :param object_type: put every feature in this one object type.
    :param with_classes: also return each object's class.
    :returns: ``{object type: uint16 labels}``, or
        ``(masks, {object type: {id: class}})`` with ``with_classes``.
    :raises ValueError: the document is not GeoJSON.
    """
    if isinstance(data, (str, os.PathLike)):
        with open(data, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    shape = (int(shape[0]), int(shape[1]))
    entries = []
    for feature in _geojson_features(data):
        polygons = _geometry_polygons(feature.get("geometry"))
        if not polygons:
            continue
        props = feature.get("properties") or {}
        classification = props.get("classification") or {}
        cls = (classification.get("name") if isinstance(classification, Mapping)
               else str(classification)) or None
        named_type, named_id = _parse_name(props.get("name"))
        measurements = props.get("measurements") or {}
        if not isinstance(measurements, Mapping):
            measurements = {m.get("name"): m.get("value") for m in measurements
                            if isinstance(m, Mapping)}
        kind = str(object_type or props.get("object_type") or named_type
                   or cls or DEFAULT_OBJECT_TYPE)
        number = _as_int(props.get("object_id"))
        if number is None:
            number = _as_int(measurements.get("object_id"))
        if number is None and named_type is not None:
            number = named_id
        pixels = [_fill_rings(polygon, shape) for polygon in polygons if polygon]
        entries.append((kind, number, str(cls or kind), pixels))
    masks, classes = _paint(entries, shape)
    return (masks, classes) if with_classes else masks


def _roi_bounds(points: np.ndarray, roi: Any) -> None:
    """Set an ImageJ ROI's bounding box from its points.

    :param points: ``(n, 2)`` array of ``x, y``.
    :param roi: the ``roifile.ImagejRoi`` to set.
    """
    low = np.floor(points.min(axis=0)).astype(int)
    high = np.ceil(points.max(axis=0)).astype(int)
    roi.left, roi.top = int(low[0]), int(low[1])
    roi.right, roi.bottom = int(high[0]), int(high[1])


def masks_to_roiset(masks: Any, path: PathLike, *, classes: Any = None,
                    object_type: Optional[str] = None) -> Path:
    """Write label masks as an ImageJ/Fiji ``RoiSet.zip``.

    An object in one piece without holes is a traced polygon ROI, the kind
    Fiji's wand makes; any other object is a composite ROI whose paths are
    all of its rings. Each ROI is named ``"<type>-<id>"`` (unique, and what
    the ROI Manager lists) and carries ``object_type``, ``object_id`` and
    ``class`` as ROI properties.

    :param masks: a 2-D label mask, or ``{object type: mask}``.
    :param path: the ``.zip`` to write; replaced if it exists.
    :param classes: ``{id: class}`` or ``{object type: {id: class}}``; an
        object without one is classified as its object type.
    :param object_type: the type of a single mask (default ``"object"``).
    :returns: the path written.
    :raises ImportError: ``roifile`` is not installed.
    :raises ValueError: an invalid mask.
    """
    roifile = _optional_module("roifile", "roifile", "Writing ImageJ ROIs")
    by_type = _object_masks(masks, object_type)
    class_map = _object_classes(classes, list(by_type))
    rois = []
    for kind, labels in by_type.items():
        for number, polygons in _iter_objects(labels):
            cls = class_map.get(kind, {}).get(number, kind)
            rings = [ring for polygon in polygons for ring in polygon]
            roi = roifile.ImagejRoi()
            if len(rings) == 1:
                ring = rings[0][:-1]
                roi.roitype = roifile.ROI_TYPE.TRACED
                _roi_bounds(ring, roi)
                roi.integer_coordinates = (
                    ring - [roi.left, roi.top]).astype(np.int32)
                roi.n_coordinates = len(ring)
            else:
                path_ops: List[float] = []
                for ring in rings:
                    path_ops += [0.0, float(ring[0][0]), float(ring[0][1])]
                    for x, y in ring[1:-1]:
                        path_ops += [1.0, float(x), float(y)]
                    path_ops.append(4.0)
                roi.roitype = roifile.ROI_TYPE.RECT
                _roi_bounds(np.vstack(rings), roi)
                roi.multi_coordinates = np.asarray(path_ops, dtype=np.float32)
                roi.shape_roi_size = len(path_ops)
            roi.name = f"{kind}-{number}"
            roi.props = (f"class: {cls}\nobject_id: {number}\n"
                         f"object_type: {kind}\n")
            rois.append(roi)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if rois:
        roifile.roiwrite(str(target), rois, mode="w")
    else:
        import zipfile

        with zipfile.ZipFile(str(target), "w"):
            pass
    return target


def _roi_props(roi: Any) -> Dict[str, str]:
    """An ImageJ ROI's properties as strings, whatever ``roifile`` it is.

    :param roi: a ``roifile.ImagejRoi``.
    :returns: ``{key: value}`` from its ``props`` text.
    """
    out = {}
    for line in str(getattr(roi, "props", "") or "").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def roiset_to_masks(path: PathLike, shape: Tuple[int, int], *,
                    object_type: Optional[str] = None,
                    with_classes: bool = False):
    """Rasterise an ImageJ ``RoiSet.zip`` (or one ``.roi``) into label masks.

    Polygon, freehand, traced, rectangle and oval ROIs are read, and a
    composite ROI by the even-odd rule over its paths, as ImageJ fills it;
    lines and points have no area and are skipped. A pixel belongs to a ROI
    when its centre is inside, ImageJ's own rule, so a set written by
    :func:`masks_to_roiset` comes back pixel for pixel. Type, id and class
    come from the ROI properties, else from a ``"<type>-<id>"`` name, else
    the ROI is an ``"object"`` numbered after the rest.

    :param path: the ``.zip`` or ``.roi`` file.
    :param shape: the image's ``(rows, columns)``.
    :param object_type: put every ROI in this one object type.
    :param with_classes: also return each object's class.
    :returns: ``{object type: uint16 labels}``, or
        ``(masks, {object type: {id: class}})`` with ``with_classes``.
    :raises ImportError: ``roifile`` is not installed.
    """
    roifile = _optional_module("roifile", "roifile", "Reading ImageJ ROIs")
    shape = (int(shape[0]), int(shape[1]))
    import zipfile

    if zipfile.is_zipfile(str(path)):
        with zipfile.ZipFile(str(path)) as archive:
            if not archive.namelist():
                masks, classes = _paint([], shape)
                return (masks, classes) if with_classes else masks
    read = roifile.roiread(str(path))
    rois = read if isinstance(read, list) else [read]
    area_types = {roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.FREEHAND,
                  roifile.ROI_TYPE.TRACED, roifile.ROI_TYPE.RECT,
                  roifile.ROI_TYPE.OVAL}
    entries = []
    for roi in rois:
        composite = bool(getattr(roi, "shape_roi_size", 0))
        if not composite and roi.roitype not in area_types:
            continue
        if roi.roitype == roifile.ROI_TYPE.RECT and not composite:
            rings = [np.asarray([[roi.left, roi.top], [roi.right, roi.top],
                                 [roi.right, roi.bottom],
                                 [roi.left, roi.bottom]], dtype=float)]
        else:
            rings = list(roi.coordinates(multi=True))
        props = _roi_props(roi)
        named_type, named_id = _parse_name(roi.name)
        kind = str(object_type or props.get("object_type") or named_type
                   or DEFAULT_OBJECT_TYPE)
        number = _as_int(props.get("object_id"))
        if number is None and named_type is not None:
            number = named_id
        pixels = [_fill_rings(rings, shape, even_odd=True)]
        entries.append((kind, number, str(props.get("class") or kind), pixels))
    masks, classes = _paint(entries, shape)
    return (masks, classes) if with_classes else masks


def rle_encode(binary: np.ndarray) -> Dict[str, Any]:
    """COCO run-length encoding of a binary mask, compressed as COCO writes it.

    :param binary: 2-D boolean mask.
    :returns: ``{"size": [rows, columns], "counts": str}``, what
        ``pycocotools.mask.encode`` returns with ``counts`` as text.
    """
    array = np.asarray(binary, dtype=bool)
    flat = array.ravel(order="F")
    changes = np.nonzero(np.diff(flat.astype(np.int8)))[0] + 1
    bounds = np.concatenate([[0], changes, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        runs = [0] + runs
    text = []
    for index, count in enumerate(runs):
        value = int(count) - int(runs[index - 2]) if index > 2 else int(count)
        more = True
        while more:
            chunk = value & 0x1F
            value >>= 5
            more = value != -1 if chunk & 0x10 else value != 0
            if more:
                chunk |= 0x20
            text.append(chr(chunk + 48))
    return {"size": [int(array.shape[0]), int(array.shape[1])],
            "counts": "".join(text)}


def rle_decode(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a COCO RLE, compressed (text) or uncompressed (list of runs).

    :param rle: ``{"size": [rows, columns], "counts": ...}``.
    :returns: the 2-D boolean mask.
    :raises ValueError: the runs do not add up to the size.
    """
    rows, cols = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    if isinstance(counts, str):
        runs: List[int] = []
        position = 0
        while position < len(counts):
            value = 0
            shift = 0
            more = True
            while more:
                chunk = ord(counts[position]) - 48
                value |= (chunk & 0x1F) << shift
                more = bool(chunk & 0x20)
                position += 1
                shift += 5
                if not more and chunk & 0x10:
                    value |= -1 << shift
            if len(runs) > 2:
                value += runs[-2]
            runs.append(value)
    else:
        runs = [int(v) for v in counts]
    if sum(runs) != rows * cols or any(v < 0 for v in runs):
        raise ValueError(f"RLE runs cover {sum(runs)} pixels, not {rows * cols}")
    values = np.zeros(len(runs), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, runs)
    return flat.reshape((rows, cols), order="F")


def masks_to_coco(masks: Any, *, file_name: str = "image",
                  classes: Any = None, object_type: Optional[str] = None,
                  rle: bool = False,
                  dataset: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Label masks as a COCO instance-segmentation dataset.

    Each object is one annotation with ``segmentation``, ``area`` (pixels),
    ``bbox`` (``[x, y, width, height]`` in pixel corners), ``iscrowd`` 0 and
    spaCR's ``object_type`` and ``object_id``. Categories are the
    ``(object type, class)`` pairs: ``name`` is the class and
    ``supercategory`` the object type. Segmentations are polygons -- one flat
    ``[x1, y1, x2, y2, ...]`` list per part -- except for an object with a
    hole, which a COCO polygon cannot express and which is written as RLE;
    ``rle=True`` writes every object as compressed RLE, which is exact
    whoever reads it.

    :param masks: a 2-D label mask, or ``{object type: mask}``.
    :param file_name: the image's file name, as ``images[].file_name``.
    :param classes: ``{id: class}`` or ``{object type: {id: class}}``; an
        object without one is classified as its object type.
    :param object_type: the type of a single mask (default ``"object"``).
    :param rle: write every segmentation as RLE.
    :param dataset: an existing COCO dict to add this image to, as a queue
        export does; ``None`` starts a new one.
    :returns: the dataset (``dataset`` itself when given).
    :raises ValueError: an invalid mask.
    """
    by_type = _object_masks(masks, object_type)
    class_map = _object_classes(classes, list(by_type))
    if dataset is None:
        dataset = {"info": {"description": "spaCR label masks",
                            "version": "1.0"},
                   "licenses": [], "images": [], "annotations": [],
                   "categories": []}
    shape = next(iter(by_type.values())).shape
    image_id = 1 + max([int(i["id"]) for i in dataset["images"]] or [0])
    dataset["images"].append({"id": image_id, "file_name": str(file_name),
                              "height": int(shape[0]), "width": int(shape[1])})
    categories = {(c.get("supercategory"), c.get("name")): int(c["id"])
                  for c in dataset["categories"]}
    next_ann = 1 + max([int(a["id"]) for a in dataset["annotations"]] or [0])
    for kind, labels in by_type.items():
        for number, polygons in _iter_objects(labels):
            cls = class_map.get(kind, {}).get(number, kind)
            key = (kind, cls)
            if key not in categories:
                categories[key] = 1 + max(list(categories.values()) or [0])
                dataset["categories"].append(
                    {"id": categories[key], "name": cls, "supercategory": kind})
            binary = labels == number
            rows, cols = np.nonzero(binary)
            if rle or any(len(polygon) > 1 for polygon in polygons):
                segmentation: Any = rle_encode(binary)
            else:
                segmentation = [[float(v) for v in polygon[0][:-1].ravel()]
                                for polygon in polygons]
            dataset["annotations"].append({
                "id": next_ann, "image_id": image_id,
                "category_id": categories[key],
                "segmentation": segmentation,
                "area": int(rows.size),
                "bbox": [int(cols.min()), int(rows.min()),
                         int(cols.max() - cols.min() + 1),
                         int(rows.max() - rows.min() + 1)],
                "iscrowd": 0, "object_type": kind, "object_id": number,
            })
            next_ann += 1
    return dataset


def _coco_image(data: Mapping[str, Any], file_name: Optional[str]
                ) -> Mapping[str, Any]:
    """The ``images`` entry a COCO import is for.

    :param data: the COCO dataset.
    :param file_name: the image to pick, matched on the full name, the base
        name or the stem; ``None`` for a dataset of one image.
    :returns: the image entry.
    :raises ValueError: no image, or several and no name to choose by.
    """
    images = list(data.get("images") or [])
    if file_name is None:
        if len(images) != 1:
            raise ValueError(f"the COCO file holds {len(images)} images; "
                             "name the one to import")
        return images[0]
    wanted = str(file_name)
    for key in (str, os.path.basename,
                lambda n: os.path.splitext(os.path.basename(n))[0]):
        for image in images:
            if key(str(image.get("file_name", ""))) == key(wanted):
                return image
    raise ValueError(f"the COCO file has no image named {wanted!r}")


def coco_image_names(data: Any) -> List[str]:
    """The file names of the images a COCO dataset annotates.

    :param data: the parsed dataset, or a path to its ``.json`` file.
    :returns: ``images[].file_name``, in file order.
    """
    if isinstance(data, (str, os.PathLike)):
        with open(data, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    return [str(image.get("file_name", "")) for image in data.get("images") or []]


def coco_to_masks(data: Any, *, file_name: Optional[str] = None,
                  shape: Optional[Tuple[int, int]] = None,
                  object_type: Optional[str] = None,
                  with_classes: bool = False):
    """Rasterise the annotations of one image of a COCO dataset.

    Polygons are filled by pixel centre and the parts of one annotation
    united, as ``pycocotools`` unites them; RLE (compressed or not) is
    decoded exactly. The object type is ``object_type``, the annotation's
    ``object_type``, or its category name; the id is ``object_id`` or the
    next free one; the class is the category name.

    :param data: the parsed dataset, or a path to its ``.json`` file.
    :param file_name: which image, for a dataset of several.
    :param shape: ``(rows, columns)``; default the image entry's height and
        width.
    :param object_type: put every annotation in this one object type.
    :param with_classes: also return each object's class.
    :returns: ``{object type: uint16 labels}``, or
        ``(masks, {object type: {id: class}})`` with ``with_classes``.
    :raises ValueError: the image is missing or an RLE does not fit it.
    """
    if isinstance(data, (str, os.PathLike)):
        with open(data, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    image = _coco_image(data, file_name)
    if shape is None:
        shape = (int(image["height"]), int(image["width"]))
    shape = (int(shape[0]), int(shape[1]))
    names = {int(c["id"]): str(c.get("name", ""))
             for c in data.get("categories") or []}
    entries = []
    for ann in data.get("annotations") or []:
        if ann.get("image_id") != image.get("id"):
            continue
        cls = names.get(_as_int(ann.get("category_id")) or 0) or None
        kind = str(object_type or ann.get("object_type") or cls
                   or DEFAULT_OBJECT_TYPE)
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, Mapping):
            binary = rle_decode(segmentation)
            if binary.shape != shape:
                raise ValueError(f"an RLE of shape {binary.shape} does not fit "
                                 f"an image of shape {shape}")
            pixels = [np.nonzero(binary)]
        else:
            pixels = [_fill_rings([np.asarray(part, dtype=float).reshape(-1, 2)],
                                  shape)
                      for part in segmentation or [] if len(part) >= 6]
        entries.append((kind, _as_int(ann.get("object_id")),
                        str(cls or kind), pixels))
    masks, classes = _paint(entries, shape)
    return (masks, classes) if with_classes else masks


def roi_format(path: PathLike, fmt: Optional[str] = None) -> str:
    """Which ROI format a file is: ``"geojson"``, ``"imagej"`` or ``"coco"``.

    :param path: the file; ``.geojson`` is GeoJSON, ``.zip`` and ``.roi``
        ImageJ, and an existing ``.json`` is read to tell COCO from GeoJSON
        (a new one is COCO).
    :param fmt: the answer, when the caller already knows it; ``"qupath"``
        and ``"roiset"`` are accepted too.
    :returns: the format.
    :raises ValueError: an unknown ``fmt`` or a file of no known format.
    """
    if fmt:
        name = str(fmt).lower()
        name = {"qupath": "geojson", "roiset": "imagej", "fiji": "imagej",
                "roi": "imagej", "zip": "imagej"}.get(name, name)
        if name not in ROI_FORMATS:
            raise ValueError(f"unknown ROI format {fmt!r}; use one of "
                             f"{ROI_FORMATS}")
        return name
    suffix = Path(path).suffix.lower()
    if suffix == ".geojson":
        return "geojson"
    if suffix in (".zip", ".roi"):
        return "imagej"
    if suffix == ".json":
        if not Path(path).is_file():
            return "coco"
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, Mapping) and "annotations" in data:
            return "coco"
        return "geojson"
    raise ValueError(f"cannot tell the ROI format of {os.fspath(path)!r}; "
                     "use .geojson, .zip, .roi or .json")


def roi_suffix(fmt: str) -> str:
    """The file suffix a ROI format is written with.

    :param fmt: ``"geojson"``, ``"imagej"`` or ``"coco"``.
    :returns: ``".geojson"``, ``".zip"`` or ``".json"``.
    """
    return {"geojson": ".geojson", "imagej": ".zip",
            "coco": ".json"}[roi_format("", fmt)]


def _write_json(data: Mapping[str, Any], path: PathLike) -> Path:
    """Write a JSON document, creating its folder.

    :param data: the document.
    :param path: where.
    :returns: the path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle)
    return target


def export_rois(masks: Any, path: PathLike, fmt: Optional[str] = None, *,
                classes: Any = None, object_type: Optional[str] = None,
                file_name: Optional[str] = None, rle: bool = False) -> Path:
    """Write label masks as QuPath GeoJSON, an ImageJ RoiSet or COCO JSON.

    :param masks: a 2-D label mask, or ``{object type: mask}``.
    :param path: the file to write; its suffix picks the format unless
        ``fmt`` does (``.geojson``, ``.zip``, ``.json`` for COCO).
    :param fmt: ``"geojson"``, ``"imagej"`` or ``"coco"``.
    :param classes: ``{id: class}`` or ``{object type: {id: class}}``.
    :param object_type: the type of a single mask (default ``"object"``).
    :param file_name: the image's name, recorded in GeoJSON ids and COCO
        ``images``; default the output file's stem.
    :param rle: COCO only: every segmentation as RLE.
    :returns: the path written.
    :raises ImportError: ImageJ output without ``roifile`` installed.
    :raises ValueError: an unknown format or an invalid mask.
    """
    target = Path(path)
    if fmt is None and target.suffix.lower() == ".json":
        fmt = "coco"
    name = roi_format(target, fmt)
    image = str(file_name or target.stem)
    if name == "imagej":
        return masks_to_roiset(masks, target, classes=classes,
                               object_type=object_type)
    if name == "geojson":
        return _write_json(masks_to_geojson(
            masks, classes=classes, object_type=object_type,
            image_name=image), target)
    return _write_json(masks_to_coco(
        masks, file_name=image, classes=classes,
        object_type=object_type, rle=rle), target)


def import_rois(path: PathLike, shape: Optional[Tuple[int, int]] = None,
                fmt: Optional[str] = None, *, file_name: Optional[str] = None,
                object_type: Optional[str] = None, with_classes: bool = False):
    """Read QuPath GeoJSON, an ImageJ RoiSet or COCO JSON into label masks.

    :param path: the file.
    :param shape: the image's ``(rows, columns)``; required for GeoJSON and
        ImageJ, optional for COCO, which records it.
    :param fmt: ``"geojson"``, ``"imagej"`` or ``"coco"``; default from the
        file (:func:`roi_format`).
    :param file_name: COCO only: which image of a dataset of several.
    :param object_type: put every object in this one object type.
    :param with_classes: also return each object's class.
    :returns: ``{object type: uint16 labels}``, or
        ``(masks, {object type: {id: class}})`` with ``with_classes``.
    :raises ValueError: no ``shape`` where one is needed, or an unreadable
        file.
    :raises ImportError: an ImageJ file without ``roifile`` installed.
    """
    name = roi_format(path, fmt)
    if name == "coco":
        return coco_to_masks(path, file_name=file_name, shape=shape,
                             object_type=object_type,
                             with_classes=with_classes)
    if shape is None:
        raise ValueError(f"reading {name} ROIs needs the image's shape")
    if name == "imagej":
        return roiset_to_masks(path, shape, object_type=object_type,
                               with_classes=with_classes)
    return geojson_to_masks(path, shape, object_type=object_type,
                            with_classes=with_classes)
