"""Touching label pairs, rendered for a person to call merge or keep.

A segmenter that cuts one object in two leaves two labels that touch along
a long shared edge. :func:`spacr.object_classifier.split_candidates` finds
such pairs by geometry; this module turns them into a review set that a
person labels one pair at a time.

The set is an ordinary Annotate source. Every pair is one PNG under
``<folder>/data/merge_pairs/`` with label A outlined white and label B
yellow, and one row in ``png_list`` of ``<folder>/measurements/
measurements.db``. The verdict goes to the ``merge`` column: ``1`` one
object cut in two, ``2`` two objects, ``3`` cannot tell. A set built here
can therefore also be opened in the Annotate grid, and any script that reads
``png_list`` reads the verdicts.

A PNG is named ``<mask stem>_<object><A>-<B>.png``, so a pair keeps the
field it came from and the two labels in its name.

The display range of each image plane is fixed for the whole set: the 1st
and 99.8th percentile of pixels pooled from every pair window. No crop is
stretched on its own, so two pairs that look alike are alike.
"""
from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MERGE_COLUMN = "merge"
MERGE_VALUES = {1: "merge", 2: "keep", 3: "unsure"}
SHARE_STEPS = (0.05, 0.15, 0.25, 0.40)
DEFAULT_SHARE = 0.25
PAIR_FOLDER = "merge_pairs"
COLOUR_A = (255, 255, 255)
COLOUR_B = (255, 210, 0)
_PAD = 16
_MIN_SIDE = 96
_IMAGE_SUFFIXES = (".npy", ".tif", ".tiff", ".png")


def _channels_last(array: np.ndarray) -> np.ndarray:
    """An image as ``(H, W, C)``: planes-first stacks are moved, 2-D gains an axis."""
    array = np.asarray(array)
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3 and array.shape[0] <= 8 and array.shape[0] < array.shape[-1]:
        return np.moveaxis(array, 0, -1)
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {array.shape}")
    return array


def _read_image(path: str) -> np.ndarray:
    """Read a ``.npy``, ``.tif`` or ``.png`` image as ``(H, W, C)`` with its own dtype."""
    low = path.lower()
    if low.endswith(".npy"):
        return _channels_last(np.load(path))
    if low.endswith((".tif", ".tiff")):
        import tifffile

        return _channels_last(tifffile.imread(path))
    from PIL import Image

    with Image.open(path) as handle:
        return _channels_last(np.asarray(handle))


def _mask_files(folder: str) -> List[str]:
    """The label images in ``folder``, by name."""
    return sorted(os.path.join(folder, name) for name in os.listdir(folder)
                  if name.lower().endswith(_IMAGE_SUFFIXES)
                  and not name.startswith("."))


def _image_for(mask_path: str, image_folder: str) -> Optional[str]:
    """The file in ``image_folder`` with the mask's stem, or None."""
    if not image_folder:
        return None
    stem = os.path.splitext(os.path.basename(mask_path))[0]
    for suffix in _IMAGE_SUFFIXES:
        candidate = os.path.join(image_folder, stem + suffix)
        if os.path.isfile(candidate):
            return candidate
    return None


def _plane(stack: np.ndarray, index: int) -> np.ndarray:
    """One plane of a channel-last stack; a negative index counts from the end."""
    if not -stack.shape[2] <= index < stack.shape[2]:
        raise ValueError(f"plane {index} is not in an image with "
                         f"{stack.shape[2]} planes")
    return stack[:, :, index]


def _share_step(mask: np.ndarray, pair: Tuple[int, int],
                found: Dict[float, set]) -> float:
    """The largest of :data:`SHARE_STEPS` at which ``pair`` is still a candidate."""
    best = 0.0
    for step in SHARE_STEPS:
        if pair in found.get(step, ()):
            best = step
    return best


def _pairs_in_mask(mask: np.ndarray, least: float
                   ) -> List[Tuple[int, int, float, int]]:
    """Touching pairs sharing at least ``least`` of the smaller perimeter.

    Pairs touching the field edge are left out, because
    :func:`spacr.object_classifier.merge_halves` is never asked to merge a
    cut object.

    :returns: ``(a, b, share step, merged by the default rule)`` per pair.
    """
    from .object_classifier import split_candidates, touches_border

    mask = np.asarray(mask)
    if mask.dtype.kind not in "iu":
        mask = mask.astype(np.int64)
    if not mask.any():
        return []
    steps = sorted({float(least), *[s for s in SHARE_STEPS if s >= least],
                    DEFAULT_SHARE})
    found = {step: set(split_candidates(mask, share=step)) for step in steps}
    out = []
    for a, b in sorted(found[float(least)]):
        if touches_border(mask, a) or touches_border(mask, b):
            continue
        out.append((a, b, _share_step(mask, (a, b), found),
                    int((a, b) in found[DEFAULT_SHARE])))
    return out


def _window(mask: np.ndarray, a: int, b: int) -> Tuple[int, int, int, int]:
    """The union box of ``a`` and ``b``, padded and grown to at least 96 px a side."""
    rows, cols = np.nonzero((mask == a) | (mask == b))
    r0, r1 = int(rows.min()) - _PAD, int(rows.max()) + 1 + _PAD
    c0, c1 = int(cols.min()) - _PAD, int(cols.max()) + 1 + _PAD
    height, width = mask.shape
    r0, r1 = _grow(r0, r1, height)
    c0, c1 = _grow(c0, c1, width)
    return r0, r1, c0, c1


def _grow(start: int, stop: int, size: int) -> Tuple[int, int]:
    """Widen ``[start, stop)`` to :data:`_MIN_SIDE` around its centre, inside ``size``."""
    if stop - start < _MIN_SIDE:
        centre = (start + stop) // 2
        start, stop = centre - _MIN_SIDE // 2, centre + _MIN_SIDE - _MIN_SIDE // 2
    start, stop = max(0, start), min(size, stop)
    return start, stop


def _display_ranges(windows: Sequence[np.ndarray], planes: int,
                    seed: int = 0) -> List[Tuple[float, float]]:
    """One ``(low, high)`` per plane over pixels pooled from every window."""
    rng = np.random.default_rng(seed)
    pooled: List[List[np.ndarray]] = [[] for _ in range(planes)]
    for window in windows:
        flat = window.reshape(-1, window.shape[-1])
        take = rng.choice(flat.shape[0], min(4000, flat.shape[0]), replace=False)
        for plane in range(planes):
            pooled[plane].append(flat[take, plane])
    out = []
    for plane in range(planes):
        values = np.concatenate(pooled[plane]).astype(np.float64)
        low, high = np.percentile(values, [1.0, 99.8])
        out.append((float(low), float(max(high, low + 1))))
    return out


def _render(window: Optional[np.ndarray], codes: np.ndarray,
            ranges: Sequence[Tuple[float, float]]) -> np.ndarray:
    """An 8-bit RGB picture of one pair, A outlined white and B yellow.

    One image plane is shown grey; two or three fill red, green and blue.
    Without an image, the two objects are drawn as grey shapes.
    """
    from skimage.segmentation import find_boundaries

    height, width = codes.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    if window is None:
        rgb[codes > 0] = 90
    else:
        planes = window.shape[-1]
        slots = [0, 0, 0] if planes == 1 else list(range(min(planes, 3)))
        for slot, plane in enumerate(slots):
            low, high = ranges[plane]
            value = (window[:, :, plane].astype(np.float32) - low) / (high - low)
            rgb[:, :, slot] = (np.clip(value, 0, 1) * 255 + 0.5).astype(np.uint8)
    for code, colour in ((1, COLOUR_A), (2, COLOUR_B)):
        rgb[find_boundaries(codes == code, mode="inner")] = colour
    return rgb


def _set_paths(folder: str) -> Tuple[str, str]:
    """The pair PNG folder and the database of a review set in ``folder``."""
    return (os.path.join(folder, "data", PAIR_FOLDER),
            os.path.join(folder, "measurements", "measurements.db"))


def _has_set(folder: str) -> bool:
    """Whether ``folder`` already holds a review set."""
    return os.path.isfile(_set_paths(folder)[1])


def _build_set(mask_folder: str, out_folder: str, *, object_name: str = "cell",
               mask_plane: int = 0, image_folder: str = "",
               image_planes: Sequence[int] = (0, 1, 2),
               least_share: float = SHARE_STEPS[0], limit: int = 0,
               progress=None) -> Dict[str, Any]:
    """Find touching pairs in every mask of ``mask_folder`` and write a review set.

    :param mask_folder: label images (``.npy``, ``.tif``, ``.png``).
    :param out_folder: where the set is written; it must not hold one yet.
    :param object_name: what the labels are, used in the file names.
    :param mask_plane: which plane of a multi-plane file holds the labels.
    :param image_folder: images with the masks' stems; may be the mask
        folder itself when the labels are one plane of the same stack.
    :param image_planes: up to three planes of the image to show.
    :param least_share: the least shared fraction of the smaller perimeter.
    :param limit: the most pairs to write, in file then label order; 0 for all.
    :param progress: called ``(done, total)`` after each mask file.
    :returns: ``{"pairs": n, "fields": n, "db": path}``.
    :raises ValueError: no masks, or a set is already there.
    """
    from PIL import Image

    from .tabular import write_database

    if _has_set(out_folder):
        raise ValueError(f"{out_folder} already holds a pair set; open it "
                         f"instead, or choose another folder")
    masks = _mask_files(mask_folder)
    if not masks:
        raise ValueError(f"no label images in {mask_folder}")
    planes = [int(p) for p in image_planes][:3]
    picks: List[Dict[str, Any]] = []
    windows: List[Optional[np.ndarray]] = []
    fields = 0
    for done, mask_path in enumerate(masks, 1):
        stack = _read_image(mask_path)
        mask = _plane(stack, mask_plane) if stack.shape[2] > 1 else stack[:, :, 0]
        found = _pairs_in_mask(mask, least_share)
        image_path = _image_for(mask_path, image_folder)
        image = None
        if image_path is not None and found:
            image = (stack if os.path.abspath(image_path) == os.path.abspath(mask_path)
                     else _read_image(image_path))
            image = np.stack([_plane(image, p) for p in planes], axis=-1) \
                if image.shape[2] > 1 else image
        if found:
            fields += 1
        stem = os.path.splitext(os.path.basename(mask_path))[0]
        for a, b, step, default in found:
            if limit and len(picks) >= limit:
                break
            r0, r1, c0, c1 = _window(mask, a, b)
            part = mask[r0:r1, c0:c1]
            codes = np.where(part == a, 1, np.where(part == b, 2, 0)).astype(np.uint8)
            windows.append(None if image is None else
                           np.asarray(image[r0:r1, c0:c1]).copy())
            picks.append(dict(
                name=f"{stem}_{object_name}{a}-{b}.png", codes=codes,
                source_mask=mask_path, source_image=image_path or "",
                object_type=object_name, label_a=a, label_b=b,
                object_label=f"{a}-{b}", share_at_least=step,
                merged_by_default_rule=default,
                window=f"{r0} {r1} {c0} {c1}"))
        if progress is not None:
            progress(done, len(masks))
        if limit and len(picks) >= limit:
            break
    shown = [w for w in windows if w is not None]
    ranges = _display_ranges(shown, shown[0].shape[-1]) if shown else []
    png_folder, db_path = _set_paths(out_folder)
    os.makedirs(png_folder, exist_ok=True)
    rows = []
    for pick, window in zip(picks, windows):
        path = os.path.join(png_folder, pick.pop("name"))
        Image.fromarray(_render(window, pick.pop("codes"), ranges), "RGB").save(path)
        rows.append(dict(png_path=path, file_name=os.path.basename(path), **pick))
    import pandas as pd

    frame = pd.DataFrame(rows, columns=[
        "png_path", "file_name", "source_mask", "source_image", "object_type",
        "label_a", "label_b", "object_label", "share_at_least",
        "merged_by_default_rule", "window"])
    frame[MERGE_COLUMN] = pd.array([None] * len(frame), dtype="Int64")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    write_database(frame, db_path, "png_list", if_exists="fail",
                   canonicalise=False)
    from .qt.annotate_engine import ensure_annotation_column

    ensure_annotation_column(db_path, MERGE_COLUMN)
    return {"pairs": len(rows), "fields": fields, "db": db_path}


def _read_set(folder: str) -> List[Dict[str, Any]]:
    """Every pair of the set in ``folder``, in table order, with its verdict."""
    from .tabular import read_table

    frame = read_table(_set_paths(folder)[1], table="png_list",
                       canonicalise=False, report=None)
    rows = []
    for record in frame.to_dict("records"):
        value = record.get(MERGE_COLUMN)
        record[MERGE_COLUMN] = (None if value is None or _is_missing(value)
                                else int(value))
        rows.append(record)
    return rows


def _is_missing(value: Any) -> bool:
    """Whether ``value`` is a missing cell."""
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _write_verdict(folder: str, png_path: str, value: Optional[int]) -> None:
    """Store ``value`` (or clear it with None) as the verdict of one pair."""
    from .database_concurrency import connect, transaction

    if value is not None and int(value) not in MERGE_VALUES:
        raise ValueError(f"a verdict is one of {sorted(MERGE_VALUES)}, not {value}")
    with contextlib.closing(connect(_set_paths(folder)[1], timeout=30)) as conn:
        with transaction(conn):
            conn.execute(f'UPDATE png_list SET "{MERGE_COLUMN}" = ? '
                         f'WHERE png_path = ?',
                         (None if value is None else int(value), str(png_path)))
