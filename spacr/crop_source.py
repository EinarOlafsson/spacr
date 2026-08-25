"""Where a classifier's training images come from.

TWO NAMES FOR THE ONE CHOICE, and they are the same two every other panel in
spaCR asks it with -- LOAD IMAGES and STREAM IMAGES. Training used to ask the
same question in a private vocabulary (``pre_generated`` / ``on_demand``),
which is one idea in two spellings: a user reading the annotation panel and
the training panel could not tell they were being asked the same thing, and
the two halves of the code could not tell either.

``png`` -- LOAD IMAGES, and the default
    Crops already written to disk by the measure step. Selected by
    ``path_string`` (a substring the path must contain) and ``file_type`` (the
    image extension). Nothing is cut here; the images exist.
``merged`` -- STREAM IMAGES
    Crops cut from ``merged/*.npy`` as training runs. A merged array holds
    both intensity planes and mask planes, so this needs to be told which are
    which: ``extract_channels`` names the intensity planes and ``object_array``
    names the object whose mask defines each crop's extent. Optionally the
    objects come from a DATABASE instead, via ``coordinate_columns``.
``generate`` -- an ACTION, not a third source
    Cut a full crop set to disk first, then train on it as LOAD IMAGES would.
    It is the one value that is not an answer to "where do the pixels come
    from", which is why it keeps its own name rather than being forced into a
    two-way naming that does not fit it.

**The stored values did not change.** ``png`` and ``merged`` are what
``spacr.crops.resolve_crop_source`` has always read, so a settings file
written under either vocabulary means what it always meant:
``pre_generated``, ``load_images`` and ``auto`` all arrive as LOAD IMAGES,
``on_demand`` and ``stream_images`` as STREAM IMAGES. :data:`CROP_SOURCE_ALIASES`
is that migration, in one place, and it is what stops a panel that has been
renamed from handing this module a word it refuses.

**Why streaming exists.** Pre-cutting every crop writes a copy of the dataset
to disk before a single epoch runs, and every change of crop size or channel
selection writes another. Cutting as training runs costs a slice per object
and no disk at all.

**Bounding box versus object.** A bounding box is the smallest rectangle
containing the object; an object crop masks everything outside it away. Both
are useful -- the background around a cell is sometimes signal and sometimes
contamination -- so it is a setting. Database-sourced objects can only ever be
bounding boxes: a coordinate has no outline to mask against.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("spacr.crop_source")

#: What a crop source can be, in the order the settings panel offers them.
#: LOAD IMAGES first, because it is the default.
CROP_SOURCES: Tuple[str, ...] = ("png", "merged", "generate")

#: Every spelling a settings file has ever carried, and the source it names.
#:
#: ACCEPTED, NOT REFUSED. Three panels and two renames have written this one
#: setting, and every value any of them wrote is still on somebody's disk:
#: ``pre_generated``/``on_demand`` from the training panel, ``load_images``/
#: ``stream_images`` from its rename, ``png``/``merged`` from the viewers, and
#: ``auto`` from before the question was asked out loud. They resolve here, so
#: this module accepts what the panels produce instead of raising on it --
#: which is what it did when the training panel was renamed and left this
#: reader behind: every computer-vision run refused at the door with
#: "crop_source='load_images' is not one of ...".
#:
#: ``auto`` means LOAD IMAGES rather than "whichever folder exists". It stays
#: readable because settings files hold it; it is not an answer a user is
#: offered, because "what is available here" is not an answer to which mode
#: they want.
CROP_SOURCE_ALIASES: Dict[str, str] = {
    "auto": "png",
    "png": "png",
    "load_images": "png",
    "pre_generated": "png",
    "merged": "merged",
    "stream": "merged",
    "stream_images": "merged",
    "on_demand": "merged",
    "generate": "generate",
}

#: The choice as a panel shows it: the value stored, and the words shown.
#:
#: The same words the annotation and montage panels use, because it is the
#: same question. A panel that renders these renders LOAD IMAGES first.
CROP_SOURCE_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("png", "load images — crops already in data/"),
    ("merged", "stream images — cut from merged/"),
    ("generate", "generate crops — write a crop set, then load it"),
)

#: Image extensions a pre-generated crop may have. The setting is a FILTER on
#: the extension, which is what it always should have been -- it and
#: ``png_type`` were two names for a path filter, and one of them pretended to
#: name a file type.
IMAGE_FILE_TYPES: Tuple[str, ...] = (
    "png", "tif", "tiff", "jpg", "jpeg", "bmp", "npy",
)

#: What a crop is cut as.
CROP_SHAPES: Tuple[str, ...] = ("bounding_box", "object")

#: Settings each source reads. Drives the greying -- a control the user can
#: edit that changes nothing is worse than one that is not there.
SOURCE_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "png": ("path_string", "file_type", "file_metadata", "tar_path"),
    "merged": ("extract_channels", "object_array", "coordinate_columns",
               "crop_shape", "image_size"),
    "generate": ("extract_channels", "object_array", "crop_shape",
                 "image_size", "path_string", "file_type"),
}


class CropSourceError(ValueError):
    """A crop source that cannot produce images, and why."""


def resolve_source(settings: Mapping[str, Any]) -> str:
    """Which crop source a settings dict asks for, in the two names.

    Every spelling in :data:`CROP_SOURCE_ALIASES` resolves, so a settings file
    from any panel spaCR has shipped answers this question. Unset means LOAD
    IMAGES, which is the default everywhere the question is asked.

    :raises CropSourceError: an unrecognised source. Guessing would train on a
        different set of images than was asked for and report success.
    """
    declared = str(settings.get("crop_source") or "").strip().lower()
    if not declared:
        return "png"
    resolved = CROP_SOURCE_ALIASES.get(declared)
    if resolved is None:
        raise CropSourceError(
            f"crop_source={settings.get('crop_source')!r} is not one of "
            f"{list(CROP_SOURCES)} (load images, stream images, or generate "
            f"a crop set); accepted spellings are "
            f"{sorted(name for name in CROP_SOURCE_ALIASES if name)}")
    return resolved


def inapplicable_settings(source: str) -> Tuple[str, ...]:
    """Settings belonging to the OTHER sources -- what the panel greys out.

    Greyed, never removed (INVARIANTS 6): a key absent from the dict makes the
    pipeline fall back to its own default, which can differ from the value the
    module needs and says nothing when it does.

    Any spelling :data:`CROP_SOURCE_ALIASES` knows is accepted, because what a
    panel has in hand is the value stored in the settings file, not the name
    this module resolved it to.
    """
    key = CROP_SOURCE_ALIASES.get(str(source).strip().lower(), "")
    if key not in SOURCE_SETTINGS:
        raise CropSourceError(
            f"{source!r} is not one of {list(CROP_SOURCES)}")
    mine = set(SOURCE_SETTINGS[key])
    return tuple(dict.fromkeys(
        s for name, keys in SOURCE_SETTINGS.items() if name != key
        for s in keys if s not in mine))


# ---------------------------------------------------------------------------
# Pre-generated: two filters that used to be one confused setting
# ---------------------------------------------------------------------------

def normalise_extension(file_type: Any) -> str:
    """The extension ``file_type`` names, without its dot and lower-cased.

    It used to hold ``'cell_png'`` -- a path substring wearing a file type's
    name, duplicating ``png_type``. Now it is an extension and only that, so
    ``'.TIF'``, ``'tif'`` and ``'tiff'`` all mean what they look like.

    :raises CropSourceError: an extension spaCR cannot read, named alongside
        the ones it can.
    """
    text = str(file_type or "").strip().lower().lstrip(".")
    if not text:
        return ""
    # Tolerated because it is what every old settings CSV holds: the old value
    # was `<object>_png`, whose extension is the part after the underscore.
    if "_" in text:
        text = text.rsplit("_", 1)[-1]
    if text not in IMAGE_FILE_TYPES:
        raise CropSourceError(
            f"{file_type!r} is not an image type spaCR reads; choose from "
            f"{', '.join(IMAGE_FILE_TYPES)}")
    return text


def matches_path(path: str, *, path_string: str = "",
                 file_type: Any = "") -> bool:
    """Whether one crop belongs in the dataset.

    Two independent tests, which is the whole point of splitting them: WHICH
    OBJECT the crop is of (a substring of its path, e.g. ``cell_png``) and
    WHAT FORMAT it is in (its extension). One setting could never express
    "every nucleus crop, whatever format" or "every TIFF, whatever object".
    """
    text = str(path)
    if path_string and str(path_string) not in text:
        return False
    extension = normalise_extension(file_type)
    if extension:
        actual = os.path.splitext(text)[1].lower().lstrip(".")
        if actual != extension and not (extension == "tif" and actual == "tiff"):
            return False
    return True


def select_crops(paths: Iterable[str], settings: Mapping[str, Any]
                 ) -> List[str]:
    """The crops a settings dict selects, in the order given."""
    path_string = settings.get("path_string") or settings.get("png_type") or ""
    return [p for p in paths
            if matches_path(p, path_string=path_string,
                            file_type=settings.get("file_type"))]


# ---------------------------------------------------------------------------
# On demand: cutting from merged
# ---------------------------------------------------------------------------

def _as_indices(value, what: str) -> List[int]:
    if value is None:
        raise CropSourceError(f"{what} is not set")
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise CropSourceError(f"{what}={value!r} is not a list of planes") from exc


def object_bounds(mask: np.ndarray, label: int) -> Optional[Tuple[int, int, int, int]]:
    """``(row0, row1, col0, col1)`` of one labelled object, or None if absent.

    Half-open on the far edge, like every other slice in Python, so the caller
    can index with it directly rather than remembering to add one.
    """
    rows, cols = np.nonzero(mask == label)
    if rows.size == 0:
        return None
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def crop_object(array: np.ndarray, mask: np.ndarray, label: int, *,
                channels: Sequence[int], shape: str = "bounding_box",
                size: Optional[int] = None,
                padding: int = 0) -> Optional[np.ndarray]:
    """Cut one object out of a merged array.

    :param array: the merged stack, ``(H, W, planes)``.
    :param mask: the plane holding this object's labels.
    :param label: which object.
    :param channels: which planes become image channels, in order.
    :param shape: ``bounding_box`` keeps the rectangle's contents;
        ``object`` zeroes everything outside the object itself. The background
        around a cell is sometimes signal and sometimes contamination, which
        is why this is a choice rather than a default.
    :param size: resize the result to ``size × size`` when given.
    :returns: ``(h, w, len(channels))``, or None if the object is not there.
    :raises CropSourceError: a plane the array does not have.
    """
    if shape not in CROP_SHAPES:
        raise CropSourceError(
            f"crop_shape={shape!r} is not one of {list(CROP_SHAPES)}")
    planes = _as_indices(channels, "extract_channels")
    if array.ndim != 3:
        raise CropSourceError(
            f"a merged array is (height, width, planes); this one is "
            f"{array.shape}")
    if max(planes) >= array.shape[2]:
        raise CropSourceError(
            f"extract_channels asks for plane {max(planes)} but the merged "
            f"array has {array.shape[2]}")

    bounds = object_bounds(mask, label)
    if bounds is None:
        return None
    row0, row1, col0, col1 = bounds
    if padding:
        row0 = max(0, row0 - padding)
        col0 = max(0, col0 - padding)
        row1 = min(mask.shape[0], row1 + padding)
        col1 = min(mask.shape[1], col1 + padding)

    cut = array[row0:row1, col0:col1, :][:, :, planes].astype(np.float32)
    if shape == "object":
        inside = (mask[row0:row1, col0:col1] == label)
        cut = cut * inside[:, :, None]
    if size:
        cut = _resize(cut, int(size))
    return cut


def crop_at(array: np.ndarray, row: float, column: float, *,
            channels: Sequence[int], size: int) -> Optional[np.ndarray]:
    """Cut a fixed box centred on a coordinate.

    The database path. Only a bounding box is possible here and that is not a
    limitation to be worked around: a coordinate has no outline, so there is
    nothing to mask against, and a crop that claimed to be object-shaped would
    be a rectangle wearing the wrong name.

    :param array: the merged stack, ``(H, W, planes)``. Its rank is not
        checked as :func:`crop_object` checks it, so a 2-D image raises
        ``IndexError`` from the slice rather than a ``CropSourceError``.
    :param row: centre on the FIRST axis -- an image row, not ``y``. Floats
        are rounded half-to-even, Python's rule, so a centroid of ``2.5``
        lands on row 2 while ``3.5`` lands on row 4.
    :param column: centre on the second axis; rounded the same way.
    :param channels: which planes become image channels, in the order given,
        so ``[2, 0]`` returns them swapped. A bare int counts as a one-plane
        list. NOT bounds-checked here as it is in :func:`crop_object`: a plane
        the array does not have raises ``IndexError``, and a negative one
        silently counts back from the last plane. An empty list yields a
        zero-channel crop rather than None.
    :param size: the box asked for, not the shape returned -- nothing on this
        path is resized or padded, unlike ``size`` in :func:`crop_object`. The
        side is rounded DOWN to even (``size=5`` cuts 4 px) and never falls
        below 2 (``0`` and negative values cut 2 px), and an array edge clips
        it further, so a coordinate near a border yields a smaller crop than
        one from the middle.
    :returns: ``(h, w, len(channels))`` as float32, or None when the box falls
        entirely off the array.
    :raises CropSourceError: ``channels`` is None, or is not planes.
    """
    planes = _as_indices(channels, "extract_channels")
    half = max(1, int(size) // 2)
    r, c = int(round(float(row))), int(round(float(column)))
    row0, row1 = max(0, r - half), min(array.shape[0], r + half)
    col0, col1 = max(0, c - half), min(array.shape[1], c + half)
    if row1 <= row0 or col1 <= col0:
        return None
    return array[row0:row1, col0:col1, :][:, :, planes].astype(np.float32)


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize, so a crop needs no image library.

    Nearest rather than interpolated: a crop is usually being made SMALLER,
    where interpolation blurs the boundary the classifier is being asked to
    look at, and this module has to stay importable on a cluster.
    """
    height, width = image.shape[:2]
    if height == size and width == size:
        return image
    rows = np.clip((np.arange(size) * height) // max(1, size), 0, height - 1)
    cols = np.clip((np.arange(size) * width) // max(1, size), 0, width - 1)
    return image[rows][:, cols]


def mask_plane_for(object_array: str, settings: Mapping[str, Any]) -> int:
    """Which plane of the merged array holds ``object_array``'s masks.

    Read from the ``*_mask_dim`` settings the mask step already writes, so the
    two cannot disagree about which plane is which.

    :raises CropSourceError: the object has no mask plane, naming the setting
        that would give it one.
    """
    name = str(object_array or "").strip().lower()
    key = f"{name}_mask_dim"
    plane = settings.get(key)
    if plane is None:
        raise CropSourceError(
            f"object_array={object_array!r} has no mask plane: {key} is not "
            f"set, so nothing says which plane of merged holds its objects")
    try:
        return int(plane)
    except (TypeError, ValueError) as exc:
        raise CropSourceError(f"{key}={plane!r} is not a plane number") from exc


def crops_from_merged(array: np.ndarray, settings: Mapping[str, Any], *,
                      labels: Optional[Sequence[int]] = None
                      ) -> List[Tuple[int, np.ndarray]]:
    """Every object's crop from one merged array.

    :param labels: only these objects; by default every label in the mask.
    :returns: ``(label, image)`` pairs, skipping objects that are not present.
    :raises CropSourceError: a setting that makes cutting impossible.
    """
    plane = mask_plane_for(settings.get("object_array", "cell"), settings)
    if plane >= array.shape[2]:
        raise CropSourceError(
            f"the mask plane is {plane} but the merged array has "
            f"{array.shape[2]} plane(s)")
    mask = array[:, :, plane]

    if labels is None:
        found = np.unique(mask)
        labels = [int(v) for v in found if v != 0]

    channels = settings.get("extract_channels")
    size = settings.get("image_size")
    shape = str(settings.get("crop_shape") or "bounding_box")

    out: List[Tuple[int, np.ndarray]] = []
    for label in labels:
        cut = crop_object(array, mask, int(label), channels=channels,
                          shape=shape, size=size)
        if cut is not None:
            out.append((int(label), cut))
    return out


def stream_planes(settings: Mapping[str, Any]) -> List[int]:
    """Which planes of a merged array become image channels, by either name.

    ``channel_arrays`` is the current spelling and ``extract_channels`` the
    older one; both are a list of plane indices and both are still written to
    settings files, so both are read here. The current spelling wins when a
    file carries both, because that is the one the panel is editing.

    :raises CropSourceError: neither is set, naming the one to set.
    """
    for name in ("channel_arrays", "extract_channels"):
        if settings.get(name) is not None:
            return _as_indices(settings.get(name), name)
    raise CropSourceError(
        "channel_arrays is not set, so nothing says which planes of the "
        "merged array become the image's channels")


def validate(settings: Mapping[str, Any]) -> str:
    """Check a settings dict can actually produce crops. Returns the source.

    Run before training rather than during it: discovering that the planes
    were never named after an hour of dataset building is a worse failure than
    refusing at the start, and the message here names the setting to fix.

    :raises CropSourceError: with what to change.
    """
    source = resolve_source(settings)
    if source == "png":
        if settings.get("file_type"):
            normalise_extension(settings.get("file_type"))
        return source

    stream_planes(settings)
    shape = str(settings.get("crop_shape") or "bounding_box")
    if shape not in CROP_SHAPES:
        raise CropSourceError(
            f"crop_shape={shape!r} is not one of {list(CROP_SHAPES)}")

    coordinates = settings.get("coordinate_columns")
    if coordinates:
        if shape != "bounding_box":
            raise CropSourceError(
                "objects taken from a database can only be cut as bounding "
                "boxes: a coordinate has no outline to mask against")
        # ONE COLUMN OR TWO, because there are two ways a database says where
        # an object is and both are in use. One column NAMES THE OBJECT --
        # `cell_id` -- and the mask plane supplies its extent; two give a
        # centroid's row and column, and the box is cut around it. Demanding
        # two refused spaCR's own derived value, which is the single
        # identifier column `stream_dataset.coordinate_column` produces.
        if not settings.get("image_size"):
            raise CropSourceError(
                "image_size is what decides how big a coordinate-centred crop "
                "is; without it there is no box to cut")
        return source

    mask_plane_for(settings.get("object_array", "cell"), settings)

    return source
