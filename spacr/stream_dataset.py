"""Create reproducible training datasets from merged image arrays.

The workflow first records a deterministic object selection table, then
extracts the selected crops from merged arrays. Object types, channels, crop
shape, and train/test allocation are therefore explicit settings rather than
properties of a previously exported directory. Crop names use the same naming
function as the standard object-crop exporter.
"""
from __future__ import annotations

import logging
import os
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.stream_dataset")

#: Supported selection methods represented as ``(value, display label)``.
STREAM_METHODS: Tuple[Tuple[str, str], ...] = (
    ("column", "coordinates from a column in a table"),
    ("array", "object numbers from a mask array"),
)

#: Settings consumed by each selection method.
METHOD_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "column": ("object_array", "channel_arrays"),
    "array": ("mask_array", "channel_arrays", "bounding_box"),
}

#: Canonical object-identifier column for each object-array type.
COORDINATE_COLUMNS: Dict[str, str] = {
    "cell": "cell_id",
    "nucleus": "nucleus_id",
    "pathogen": "pathogen_id",
    "cytoplasm": "cytoplasm_id",
}

#: What the selection table records for each object.
SELECTION_COLUMNS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "objectID",
    "object_array", "split", "source",
)

#: The file the decision is written to, in the destination folder.
SELECTION_FILE = "stream_selection.csv"


def coordinate_column(object_array: str) -> str:
    """Return the identifier column for an object-array type.

    Parameters
    ----------
    object_array : str
        Object type such as ``"cell"`` or ``"nucleus"``.

    Returns
    -------
    str
        Canonical identifier column.

    Raises
    ------
    KeyError
        If the object type is unsupported.
    """
    return COORDINATE_COLUMNS[str(object_array).strip().lower()]


def settings_for_method(method: str) -> Tuple[str, ...]:
    """Return settings used by a dataset-selection method.

    Raises
    ------
    KeyError
        If ``method`` is unsupported.
    """
    return METHOD_SETTINGS[str(method).strip().lower()]


# ---------------------------------------------------------------------------
# The table, which comes first
# ---------------------------------------------------------------------------

def _split_labels(count: int, test_split: float, seed: int) -> np.ndarray:
    """Generate deterministic train/test labels for ``count`` objects."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(count))
    n_test = int(round(float(test_split) * int(count)))
    out = np.array(["train"] * int(count), dtype=object)
    out[order[:n_test]] = "test"
    return out


def selection_from_objects(frame: pd.DataFrame, *, object_array: str = "cell",
                           test_split: float = 0.2, seed: int = 0
                           ) -> pd.DataFrame:
    """Build a streaming selection from an object table.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per measured object.
    object_array : str, default="cell"
        Object type to select.
    test_split : float, default=0.2
        Fraction assigned to the test set.
    seed : int, default=0
        Random seed controlling deterministic split assignment.

    Returns
    -------
    pandas.DataFrame
        Canonical identifiers, object type, split, and source for each object.

    Raises
    ------
    ValueError
        If the table is empty or contains no usable object identifier.
    """
    from .schema import COLUMN_KEY, FIELD_KEY, OBJECT_KEY, PLATE_KEY, ROW_KEY

    if frame is None or not len(frame):
        raise ValueError("the object table is empty, so no objects can be "
                         "selected to stream")
    column = coordinate_column(object_array)
    have = set(map(str, frame.columns))
    label = column if column in have else (
        OBJECT_KEY if OBJECT_KEY in have else None)
    if label is None:
        raise ValueError(
            f"the object table names no object: it has neither {column!r} "
            f"nor {OBJECT_KEY!r}, so a selection built from it would stream "
            f"crops it cannot identify")
    out = pd.DataFrame({
        PLATE_KEY: frame.get(PLATE_KEY, ""),
        ROW_KEY: frame.get(ROW_KEY, ""),
        COLUMN_KEY: frame.get(COLUMN_KEY, ""),
        FIELD_KEY: frame.get(FIELD_KEY, ""),
        OBJECT_KEY: frame[label].astype(str),
    })
    out["object_array"] = str(object_array)
    out["split"] = _split_labels(len(out), test_split, seed)
    out["source"] = "object table"
    return out[list(SELECTION_COLUMNS)]


def selection_from_arrays(merged_folder: str, *, object_array: str = "cell",
                          test_split: float = 0.2, seed: int = 0,
                          mask_index: Optional[int] = None) -> pd.DataFrame:
    """Build a streaming selection from labels in merged arrays.

    Parameters
    ----------
    merged_folder : str
        Directory containing merged ``.npy`` arrays.
    object_array : str, default="cell"
        Object type represented by the selected mask plane.
    test_split : float, default=0.2
        Fraction assigned to the test set.
    seed : int, default=0
        Random seed controlling deterministic split assignment.
    mask_index : int, optional
        Mask plane index. The final plane is used when omitted.

    Returns
    -------
    pandas.DataFrame
        Canonical identifiers, object type, split, and source array for every
        nonzero label.

    Raises
    ------
    FileNotFoundError
        If no merged arrays or nonzero object labels are available.
    """
    folder = str(merged_folder)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy")) \
        if os.path.isdir(folder) else []
    if not files:
        raise FileNotFoundError(
            f"{folder} holds no .npy stack, so the object numbers cannot be "
            f"read and there is nothing to stream")

    from .schema import COLUMN_KEY, FIELD_KEY, OBJECT_KEY, PLATE_KEY, ROW_KEY
    from .dependent_join import parts_from_path

    rows: List[Dict[str, Any]] = []
    for name in files:
        try:
            stack = np.load(os.path.join(folder, name), mmap_mode="r")
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read %s", name, exc_info=True)
            continue
        plane = stack[..., -1 if mask_index is None else int(mask_index)] \
            if getattr(stack, "ndim", 0) >= 3 else stack
        labels = np.unique(np.asarray(plane))
        parts = parts_from_path(name)
        for label in labels:
            if int(label) == 0:      # background is not an object
                continue
            rows.append({
                PLATE_KEY: parts.get(PLATE_KEY, ""),
                ROW_KEY: parts.get(ROW_KEY, ""),
                COLUMN_KEY: parts.get(COLUMN_KEY, ""),
                FIELD_KEY: parts.get(FIELD_KEY, ""),
                OBJECT_KEY: str(int(label)),
                "object_array": str(object_array),
                "source": f"npy: {name}",
            })
    if not rows:
        raise FileNotFoundError(
            f"{len(files)} .npy stack(s) in {folder} hold no object labels "
            f"other than background, so there is nothing to stream")
    out = pd.DataFrame(rows)
    out["split"] = _split_labels(len(out), test_split, seed)
    return out[list(SELECTION_COLUMNS)]


def build_selection(dst: str, *, objects: Optional[pd.DataFrame] = None,
                    merged_folder: str = "", object_array: str = "cell",
                    test_split: float = 0.2, seed: int = 0
                    ) -> Tuple[pd.DataFrame, str]:
    """Build and save the object-selection table.

    An available object table is preferred; otherwise labels are read from
    merged arrays.

    Returns
    -------
    pandas.DataFrame
        Selected objects and deterministic split assignments.
    str
        Path to the written ``stream_selection.csv`` file.
    """
    if objects is not None and len(objects):
        table = selection_from_objects(objects, object_array=object_array,
                                       test_split=test_split, seed=seed)
    else:
        table = selection_from_arrays(merged_folder,
                                      object_array=object_array,
                                      test_split=test_split, seed=seed)
    os.makedirs(str(dst), exist_ok=True)
    path = os.path.join(str(dst), SELECTION_FILE)
    # WRITTEN BEFORE ANY IMAGE IS. A training set decided at run time and
    # never recorded cannot be re-made or audited.
    #
    # Through `tabular.write_table`, not `to_csv`: the selection names wells
    # and plates, and those columns have three spellings in this codebase.
    # A record written in whichever one the source frame happened to use is
    # a record the next reader has to guess at.
    from .tabular import write_table

    write_table(table, path)
    return table, path


# ---------------------------------------------------------------------------
# The crop, named the way measure_crop names one
# ---------------------------------------------------------------------------

def crop_name(field_stem: str, object_id, *, crop_mode: str = "cell",
              nucleus_ids=(), pathogen_ids=(), timelapse: bool = False
              ) -> str:
    """Return the standard exported-crop name for an object.

    Naming is delegated to :func:`spacr.utils._generate_names`, ensuring that
    streamed and pre-exported crops from the same object are compatible.
    """
    from .utils import _generate_names

    name, _folder, _table = _generate_names(
        str(field_stem),
        np.atleast_1d(np.asarray([object_id], dtype=object)),
        np.atleast_1d(np.asarray(list(nucleus_ids) or [0])),
        np.atleast_1d(np.asarray(list(pathogen_ids) or [0])),
        "", crop_mode=str(crop_mode),
        timelapse=timelapse, object_id=object_id)
    return name


def cut(stack: np.ndarray, mask: np.ndarray, label: int, *,
        bounding_box: bool = True, channels: Sequence[int] = ()
        ) -> Optional[np.ndarray]:
    """Extract one labeled object from an image stack.

    Parameters
    ----------
    stack : numpy.ndarray
        Two- or three-dimensional image data.
    mask : numpy.ndarray
        Integer label mask aligned with the image dimensions.
    label : int
        Object label to extract.
    bounding_box : bool, default=True
        Return the complete bounding box when true. When false, pixels outside
        the selected object are set to zero within the box.
    channels : sequence of int, optional
        Channel indices to retain. All channels are used when omitted.

    Returns
    -------
    numpy.ndarray or None
        Extracted crop, or ``None`` when the label is absent.
    """
    from .crop_source import crop_object

    data = np.asarray(stack)
    if data.ndim == 2:
        data = data[..., None]
    planes = list(channels) if channels else list(range(data.shape[2]))
    return crop_object(data, np.asarray(mask), int(label),
                       channels=planes,
                       shape="bounding_box" if bounding_box else "object")


# ---------------------------------------------------------------------------
# The pass that writes the dataset
# ---------------------------------------------------------------------------

def _field_stem(row) -> str:
    """Return the canonical field stem used in crop names."""
    from .schema import COLUMN_KEY, FIELD_KEY, PLATE_KEY, ROW_KEY

    parts = [str(row.get(key, "") or "")
             for key in (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY)]
    return "_".join(p for p in parts if p)


def _stem_of(row) -> str:
    """Return the recorded source stem or reconstruct it from identifiers."""
    source = str(row.get("source", "") or "")
    if source.startswith("npy:"):
        return os.path.splitext(source.split(":", 1)[1].strip())[0]
    return _field_stem(row)


def _stack_for(merged_folder: str, stem: str) -> Optional[str]:
    """Find the merged array matching a recorded field stem."""
    if not os.path.isdir(str(merged_folder)):
        return None
    wanted = str(stem)
    for name in sorted(os.listdir(str(merged_folder))):
        if not name.endswith(".npy"):
            continue
        if os.path.splitext(name)[0] == wanted:
            return os.path.join(str(merged_folder), name)
    # A field written as `plate_A01_1_0.npy` has a trailing token the stem
    # does not; match on the prefix as a second pass rather than a first, so
    # an exact name always wins.
    for name in sorted(os.listdir(str(merged_folder))):
        if name.endswith(".npy") and name.startswith(wanted):
            return os.path.join(str(merged_folder), name)
    return None


def stream(selection: pd.DataFrame, merged_folder: str, dst: str, *,
           channel_arrays: Sequence[int] = (0, 1, 2),
           mask_index: Optional[int] = None,
           bounding_box: bool = True,
           crop_mode: str = "cell",
           write=None) -> Dict[str, Any]:
    """Write crops described by a saved selection table.

    Each merged field is loaded once and all selected objects from that field
    are extracted before advancing.

    Parameters
    ----------
    selection : pandas.DataFrame
        Selection produced by :func:`build_selection`.
    merged_folder : str
        Directory containing merged ``.npy`` arrays.
    dst : str
        Destination for split-specific crop directories.
    channel_arrays : sequence of int, default=(0, 1, 2)
        Image channels to retain.
    mask_index : int, optional
        Mask plane index. The final plane is used when omitted.
    bounding_box : bool, default=True
        Whether crops retain every pixel in the object bounding box.
    crop_mode : str, default="cell"
        Object type used for crop naming.
    write : callable, optional
        Function accepting ``(path, array)``. The default writes NumPy arrays.

    Returns
    -------
    dict
        Written and missing crop counts, processed field count, destination
        folders, and per-field problems.
    """
    from .schema import OBJECT_KEY

    if write is None:
        def write(path, array):
            np.save(os.path.splitext(path)[0] + ".npy", np.asarray(array))

    report: Dict[str, Any] = {"written": 0, "missing": 0, "fields": 0,
                              "folders": [], "trouble": []}
    if selection is None or not len(selection):
        report["trouble"].append(
            "the selection table is empty, so there is nothing to stream")
        return report

    frame = selection.copy()
    # THE TABLE ALREADY KNOWS WHICH FILE EACH OBJECT CAME FROM when it was
    # built from the .npy stacks, and that beats rebuilding the name from
    # the parsed parts: a merged file is named `plate1_A01_1_0.npy` while
    # the parts come back as r1/c1, so a rebuilt stem matches NOTHING. The
    # first attempt at this missed every field for exactly that reason.
    frame["_stem"] = [
        _stem_of(row) for _, row in frame.iterrows()]
    for split in sorted(set(map(str, frame.get("split", ["train"])))):
        folder = os.path.join(str(dst), str(split))
        os.makedirs(folder, exist_ok=True)
        report["folders"].append(folder)

    for stem, here in frame.groupby("_stem", sort=True):
        path = _stack_for(merged_folder, str(stem))
        if path is None:
            # COUNTED, NOT SKIPPED SILENTLY. A dataset short by a field is a
            # dataset trained on a different screen from the one the table
            # describes, and nothing else would say so.
            report["missing"] += int(len(here))
            report["trouble"].append(f"no merged stack for {stem}")
            continue
        try:
            stack = np.load(path)
        except Exception as error:                           # noqa: BLE001
            report["missing"] += int(len(here))
            report["trouble"].append(f"{stem}: {type(error).__name__}")
            continue
        report["fields"] += 1
        plane = -1 if mask_index is None else int(mask_index)
        mask = stack[..., plane] if getattr(stack, "ndim", 0) >= 3 else stack
        for _, row in here.iterrows():
            try:
                # THROUGH THE ONE PARSER. The coordinate method takes its
                # object id from the object table's own column, and
                # `png_list` spells it `cell_id = 'o2'` -- so `int(float(...))`
                # raised on every row of the crop table and each object was
                # counted as missing and dropped in silence. The array method
                # never hit it, because a label scanned off a mask plane is
                # already an integer, so the two methods produced different
                # datasets from the same screen.
                from .crops import object_label

                label = object_label(row[OBJECT_KEY])
            except Exception:                                # noqa: BLE001
                report["missing"] += 1
                continue
            crop = cut(stack, mask, label, bounding_box=bounding_box,
                       channels=list(channel_arrays))
            if crop is None:
                report["missing"] += 1
                continue
            name = crop_name(str(stem), label, crop_mode=crop_mode)
            out = os.path.join(str(dst), str(row.get("split", "train")), name)
            write(out, crop)
            report["written"] += 1
    return report


def stream_dataset(settings: Mapping[str, Any], dst: str, *,
                   objects: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Build, save, and execute a streamed-dataset selection.

    Parameters
    ----------
    settings : mapping
        Source, object type, split, channel, and crop-shape settings.
    dst : str
        Dataset destination.
    objects : pandas.DataFrame, optional
        Object table used to build the selection. Merged-array labels are used
        when the table is unavailable.

    Returns
    -------
    dict
        Streaming report augmented with the saved selection-table path.
    """
    merged = str(settings.get("merged_folder")
                 or os.path.join(str(settings.get("src", "")), "merged"))
    table, path = build_selection(
        dst, objects=objects, merged_folder=merged,
        object_array=str(settings.get("object_array") or "cell"),
        test_split=float(settings.get("test_split") or 0.2),
        seed=int(settings.get("random_seed") or 0))
    report = stream(
        table, merged, dst,
        channel_arrays=list(settings.get("channel_arrays") or (0, 1, 2)),
        bounding_box=bool(settings.get("bounding_box", True)),
        crop_mode=str(settings.get("object_array") or "cell"))
    report["selection"] = path
    return report
