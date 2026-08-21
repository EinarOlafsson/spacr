"""Stream training crops rather than reading them off disk (instruction 230).

TRAINING SHOULD BE POSSIBLE ON AS MANY COMBINATIONS OF THE DATA AS POSSIBLE.
Today the crops have to exist before a model can see them, so every
combination of objects, channels and bounding-box choice is a separate
export somebody has to remember to make. Streaming makes the combination a
setting rather than a directory.

THE TABLE COMES FIRST, AND IT IS SAVED. Streaming does not begin by walking
images. It begins by deciding WHICH objects are to be streamed and writing
that decision down -- because a training set that was decided at run time
and never written down cannot be re-made, compared against, or audited when
a model turns out to have learned something it should not have.

BORROWED FROM `measure_crop`, NOT RE-DERIVED. `spacr.utils._generate_names`
is what names an exported crop, and it is what names a streamed one -- so a
streamed crop and an exported crop of the same object are the same filename,
and anything downstream can treat them alike.
"""
from __future__ import annotations

import logging
import os
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.stream_dataset")

#: How the objects to stream are chosen.
#:
#: THE TWO NEED DIFFERENT SETTINGS, which is why this is one control rather
#: than a pair of flags: a panel showing the settings of the method the user
#: did NOT choose is a panel asking them to fill in something nothing reads.
STREAM_METHODS: Tuple[Tuple[str, str], ...] = (
    ("column", "coordinates from a column in a table"),
    ("array", "object numbers from a mask array"),
)

#: What each method reads. The panel greys the rest, and this is the single
#: source for that -- a list living in the GUI would drift from what the
#: streamer reads, and the symptom would be a control that changes nothing.
METHOD_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "column": ("object_array", "channel_arrays"),
    "array": ("mask_array", "channel_arrays", "bounding_box"),
}

#: The column an object's coordinates come from, per object array. DERIVED,
#: NOT ASKED FOR: "coordinate column will always be the same so figure that
#: out from object array" -- asking is asking the user to restate something
#: spaCR knows, and giving them a way to get it wrong.
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
    """The column an object of this type is identified by.

    :raises KeyError: for an object spaCR does not measure. Guessing a
        column name would produce a table that joins to nothing and reports
        no error, which is the failure this derivation exists to prevent.
    """
    return COORDINATE_COLUMNS[str(object_array).strip().lower()]


def settings_for_method(method: str) -> Tuple[str, ...]:
    """The settings ``method`` reads.

    :raises KeyError: for a method that does not exist -- silently returning
        an empty tuple would grey every setting and look like a UI bug.
    """
    return METHOD_SETTINGS[str(method).strip().lower()]


# ---------------------------------------------------------------------------
# The table, which comes first
# ---------------------------------------------------------------------------

def _split_labels(count: int, test_split: float, seed: int) -> np.ndarray:
    """``train``/``test`` for ``count`` objects, deterministically.

    SEEDED, because the selection table's whole purpose is that the same
    settings produce the same training set. A shuffle nobody can reproduce
    turns the saved table into a record of one run rather than a recipe.
    """
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(count))
    n_test = int(round(float(test_split) * int(count)))
    out = np.array(["train"] * int(count), dtype=object)
    out[order[:n_test]] = "test"
    return out


def selection_from_objects(frame: pd.DataFrame, *, object_array: str = "cell",
                           test_split: float = 0.2, seed: int = 0
                           ) -> pd.DataFrame:
    """Build the selection table from the OBJECT TABLE. The first route.

    :param frame: the object table -- one row per measured object.
    :raises ValueError: when the table names no object. A selection built
        from rows that cannot be identified would stream the wrong crops.
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
    """Build the selection table by reading the .npy files. THE FALLBACK.

    "if that does not exist another method must read all npy files in the
    merged folder and record the object numbers in the chosen mask array".

    :param mask_index: which plane of the stack is the mask. ``None`` means
        the LAST, which is where `measure_crop` writes it.
    :raises FileNotFoundError: when the folder holds no .npy at all. A
        selection table with no rows would stream nothing and report
        success.
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
    """Decide what to stream, WRITE IT DOWN, and return it.

    THE OBJECT TABLE FIRST, THE .npy FALLBACK SECOND, which is the order the
    instruction gives. The fallback is not a worse answer, it is the answer
    for a screen that was masked but never measured.

    :returns: ``(table, path)``.
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
    table.to_csv(path, index=False)
    return table, path


# ---------------------------------------------------------------------------
# The crop, named the way measure_crop names one
# ---------------------------------------------------------------------------

def crop_name(field_stem: str, object_id, *, crop_mode: str = "cell",
              nucleus_ids=(), pathogen_ids=(), timelapse: bool = False
              ) -> str:
    """The filename `measure_crop` would give this crop.

    THROUGH `spacr.utils._generate_names`, the same call, so a streamed crop
    and an exported crop of the same object have the SAME NAME. A second
    naming rule here would drift from that one, and the drift would only
    show up as two folders that cannot be pooled.
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
    """One object's pixels out of ``stack``.

    THROUGH `crop_source.crop_object`, which already solves this -- the
    plane check, the bounds and the mask-out are all there, and a second
    cutter would drift from the one the on-demand path already uses. This
    translates instruction 230's vocabulary into that function's:
    `bounding_box=False` is its `shape='object'`.

    :param bounding_box: the box around the object when True; ONLY THE
        PIXELS THAT OVERLAP THE MASK when False, with the rest zeroed. Both
        are wanted -- "using this method images can be streamed with bounding
        box or just the ppixels that overlap with the mask" -- and they are
        different training sets, not two renderings of one.
    :returns: the crop, or ``None`` when the label is not in the mask.
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
    """`plate_row_column_field`, which is what a crop name is built on."""
    from .schema import COLUMN_KEY, FIELD_KEY, PLATE_KEY, ROW_KEY

    parts = [str(row.get(key, "") or "")
             for key in (PLATE_KEY, ROW_KEY, COLUMN_KEY, FIELD_KEY)]
    return "_".join(p for p in parts if p)


def _stem_of(row) -> str:
    """The field this object belongs to, preferring what was recorded.

    `source` carries the .npy the object was READ from when the selection
    came by that route, and a name that was read is always right. The
    rebuilt stem is the fallback for a table built from the object table,
    where there is no file name to have kept.
    """
    source = str(row.get("source", "") or "")
    if source.startswith("npy:"):
        return os.path.splitext(source.split(":", 1)[1].strip())[0]
    return _field_stem(row)


def _stack_for(merged_folder: str, stem: str) -> Optional[str]:
    """The .npy holding this field, or None.

    MATCHED ON THE STEM rather than assembled from it: a merged file's name
    carries the plate exactly as the acquisition wrote it, and rebuilding it
    from the parsed parts is how a plate whose name contains an underscore
    stops being found.
    """
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
    """Write the dataset the selection table describes. Returns a report.

    AFTER THE TABLE, NEVER INSTEAD OF IT. This walks what was already
    decided and saved; it makes no choices of its own about which objects
    are in the training set, which is what makes the run reproducible from
    the file rather than from the code.

    ONE STACK READ PER FIELD, not per object. A field holds hundreds of
    objects and a merged stack is tens of megabytes; re-reading it per crop
    is the difference between a minute and an afternoon.

    :param write: called with ``(path, array)``. Injected so the pass can be
        tested without a dataset on disk -- the default writes a .npy beside
        the name `measure_crop` would have used.
    :returns: ``written``, ``missing``, ``fields``, ``folders``.
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
                label = int(float(row[OBJECT_KEY]))
            except (TypeError, ValueError):
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
    """Build the selection, save it, then stream. THE WHOLE PASS.

    THE ORDER IS THE CONTRACT: "after the table is generated the streamin
    begins to generate the datasets on disk". A pass that streamed first and
    recorded afterwards would record what it happened to write rather than
    what it set out to.
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
