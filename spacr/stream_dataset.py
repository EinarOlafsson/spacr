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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
