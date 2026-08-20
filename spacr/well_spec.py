"""Parse plate rows, columns, and wells through one validated vocabulary.

Supported forms include rows such as ``r1``, columns such as ``c1``, and
individual wells such as ``A01`` or ``r1_c1``. Every token is checked against
the selected plate layout so an invalid or out-of-range selection fails with
a useful message instead of silently selecting no wells.
"""
from __future__ import annotations

import re
import string
from typing import Iterable, Optional, Sequence, Set, Tuple

#: Settings whose documented values may contain plate rows, columns, or wells.
#: The list is derived from setting descriptions rather than key-name patterns,
#: which ensures generic fields such as ``filter_value`` are included.
WELL_SETTINGS = (
    "cell_loc", "cell_plate_metadata", "class_metadata", "classes",
    "control_wells", "filter_value", "metadata_item_1_value", "mix", "neg",
    "negative_control", "pathogen_loc", "pathogen_plate_metadata", "pos",
    "positive_control", "treatment_loc", "treatment_plate_metadata",
)

#: Settings whose complete value is a well specification and can therefore be
#: replaced safely by the plate-map picker. Mixed-vocabulary fields remain
#: outside this subset so non-well values are preserved.
WELL_ONLY_SETTINGS = (
    "cell_loc", "control_wells", "filter_value", "pathogen_loc",
    "treatment_loc",
)

#: Supported well counts mapped to their standard row and column dimensions.
LAYOUTS = {
    6: (2, 3),
    12: (3, 4),
    24: (4, 6),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48),
}

#: What the picker opens on.
DEFAULT_LAYOUT = 384

#: CASE IS THE DISCRIMINATOR, and it has to be one.
#:
#: `C04` is row C column 4 on every plate map ever printed. `c4` is column 4
#: in spaCR's own vocabulary -- `control_wells` documents "a column ('c12')".
#: Those two collide the moment the row/column patterns are case-insensitive,
#: and they did: `C04` on a 12-well plate came back as the whole of column 4.
#:
#: So the LOWERCASE forms are spaCR's row and column, and anything else is a
#: plate-map well. `c4` is a column, `C4` is well C4. The rule is stated in
#: the refusal text, because it is the one thing here a user can get wrong
#: while typing something that looks perfectly reasonable.
_ROW = re.compile(r"^r(\d{1,2})$")
_COLUMN = re.compile(r"^c(\d{1,3})$")
_WELL_RC = re.compile(r"^r(\d{1,2})[_-]?c(\d{1,3})$", re.IGNORECASE)
#: ``A01``, ``A1``, and the two-letter rows a 1536 needs (``AA1``).
_WELL_ALPHA = re.compile(r"^([A-Za-z]{1,2})(\d{1,3})$")


class WellSpecError(ValueError):
    """Raised when a well specification is invalid for its plate layout."""


def shape(layout: int = DEFAULT_LAYOUT) -> Tuple[int, int]:
    """Return the row and column count for a supported plate layout.

    Raises
    ------
    WellSpecError
        If ``layout`` is not one of the supported well counts.
    """
    try:
        return LAYOUTS[int(layout)]
    except (KeyError, TypeError, ValueError):
        raise WellSpecError(
            f"{layout!r} is not a plate layout spaCR knows. It offers "
            f"{sorted(LAYOUTS)}.") from None


def row_label(row: int) -> str:
    """Convert a one-based row number to a plate-map label."""
    if row <= 26:
        return string.ascii_uppercase[row - 1]
    # A 1536 has 32 rows, so the last six are AA..AF. Two letters is as far
    # as any real plate goes.
    return "A" + string.ascii_uppercase[row - 27]


def row_number(label: str) -> int:
    """Convert a plate-map row label to its one-based row number."""
    text = str(label).strip().upper()
    if len(text) == 1:
        return string.ascii_uppercase.index(text) + 1
    return 26 + string.ascii_uppercase.index(text[1]) + 1


def well_label(row: int, column: int) -> str:
    """Return the plate-map label for a one-based row and column."""
    return f"{row_label(row)}{column:02d}"


def parse_one(text: str, layout: int = DEFAULT_LAYOUT) -> Set[Tuple[int, int]]:
    """Resolve one row, column, or well token to one-based coordinates.

    Raises
    ------
    WellSpecError
        If the token is malformed or outside the selected layout. The error
        identifies both the token and plate dimensions.
    """
    rows, columns = shape(layout)
    token = str(text or "").strip()
    if not token:
        return set()

    match = _WELL_RC.match(token)
    if match:
        row, column = int(match.group(1)), int(match.group(2))
        _check(row, column, rows, columns, token, layout)
        return {(row, column)}

    match = _ROW.match(token)
    if match:
        row = int(match.group(1))
        _check(row, 1, rows, columns, token, layout)
        return {(row, c) for c in range(1, columns + 1)}

    match = _COLUMN.match(token)
    if match:
        column = int(match.group(1))
        _check(1, column, rows, columns, token, layout)
        return {(r, column) for r in range(1, rows + 1)}

    match = _WELL_ALPHA.match(token)
    if match:
        letters, digits = match.group(1), match.group(2)
        try:
            row = row_number(letters)
        except ValueError:
            raise WellSpecError(
                f"{token!r} does not name a row: {letters!r} is not a plate "
                f"row letter.") from None
        column = int(digits)
        _check(row, column, rows, columns, token, layout)
        return {(row, column)}

    raise WellSpecError(
        f"{token!r} is not a row (r1), a column (c1) or a well (A01 or "
        f"r1_c1). Note that case decides between the last two: 'c4' is "
        f"column 4 and 'C4' is well C4, because a plate map is labelled "
        f"C04 and spaCR writes c4.")


def _check(row: int, column: int, rows: int, columns: int, token: str,
           layout: int) -> None:
    if not (1 <= row <= rows and 1 <= column <= columns):
        raise WellSpecError(
            f"{token!r} is outside a {layout}-well plate, which has {rows} "
            f"row(s) and {columns} column(s) (A01 to "
            f"{well_label(rows, columns)}). Choose the layout the plate "
            f"actually is.")


def parse(text, layout: int = DEFAULT_LAYOUT) -> Set[Tuple[int, int]]:
    """Resolve a mixed well specification to unique one-based coordinates.

    The value ``"r1, c1, A01"`` selects the union of the named row, column,
    and individual well.
    """
    if text is None:
        return set()
    tokens: Iterable
    if isinstance(text, str):
        tokens = re.split(r"[,\s;]+", text)
    else:
        tokens = text
    out: Set[Tuple[int, int]] = set()
    for token in tokens:
        out |= parse_one(token, layout)
    return out


def to_text(cells: Iterable[Tuple[int, int]],
            layout: int = DEFAULT_LAYOUT) -> str:
    """Serialize selected coordinates using compact supported tokens.

    Complete rows and columns become ``rN`` and ``cN`` tokens. Remaining
    coordinates are written as individual well labels, ensuring the result
    can be parsed again without introducing an unsupported range syntax.
    """
    rows, columns = shape(layout)
    chosen = {(int(r), int(c)) for r, c in cells}
    if not chosen:
        return ""

    parts = []
    covered: Set[Tuple[int, int]] = set()
    for row in range(1, rows + 1):
        whole = {(row, c) for c in range(1, columns + 1)}
        if whole <= chosen:
            parts.append(f"r{row}")
            covered |= whole
    for column in range(1, columns + 1):
        whole = {(r, column) for r in range(1, rows + 1)}
        if whole <= chosen and not whole <= covered:
            parts.append(f"c{column}")
            covered |= whole
    rest = sorted(chosen - covered)
    parts.extend(well_label(r, c) for r, c in rest)
    return ",".join(parts)
