"""Which settings each selection method reads, and which column names it.

SEPARATE FROM :mod:`spacr.stream_dataset` FOR ONE REASON: this is data and
that module needs pandas. `spacr.settings` asks for `METHOD_SETTINGS` and
`coordinate_column` while a settings panel is being laid out, and reaching
them through `stream_dataset` charged a 203 ms pandas import to read two
dictionaries -- on the main thread, which the user sees as the interface
stopping before anything has been run.

The same move as :mod:`spacr.outlier_criteria`, for the same reason. A
settings default is a question about names and values; answering it should
not load machinery.
"""

from __future__ import annotations

from typing import Dict, Tuple

__all__ = ["STREAM_METHODS", "METHOD_SETTINGS", "COORDINATE_COLUMNS",
           "SELECTION_COLUMNS", "SELECTION_FILE",
           "coordinate_column", "settings_for_method"]


STREAM_METHODS: Tuple[Tuple[str, str], ...] = (
    ("column", "coordinates from a column in a table"),
    ("array", "object numbers from a mask array"),
)

#: Settings consumed by each selection method.
METHOD_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "column": ("object_array", "channel_arrays"),
    "array": ("object_array", "channel_arrays", "bounding_box"),
}

#: Canonical object-identifier column for each object-array type.
COORDINATE_COLUMNS: Dict[str, str] = {
    "cell": "cell_id",
    "nucleus": "nucleus_id",
    "pathogen": "pathogen_id",
    "cytoplasm": "cytoplasm_id",
    "organelle": "organelle_id",
    "organelleb": "organelleb_id",
    "organellec": "organellec_id",
    "organelled": "organelled_id",
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
