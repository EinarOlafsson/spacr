"""The one place that says what object kinds spaCR has.

Eleven modules used to spell this out independently -- ``OBJECT_TYPES`` in
:mod:`spacr.measure_hooks`, :mod:`spacr.feature_dict`, :mod:`spacr.schema`,
:mod:`spacr.crops` and :mod:`spacr.diameter`; ``OBJECT_TABLES`` in
:mod:`spacr.schema`, :mod:`spacr.filters` and :mod:`spacr.merge_tables`;
``CROP_OBJECT_TYPES`` in :mod:`spacr.io`; ``CROP_MODES`` in
:mod:`spacr.measure`; and ``OBJECT_NAMES`` in :mod:`spacr.validate`. Only two
of the eleven derived from another.

They agreed on MEMBERSHIP and disagreed on ORDER -- three different orderings
of the same five names, plus two four-name variants that leave out cytoplasm.
That is the shape of the problem: adding a sixth kind meant finding all
eleven, and missing one produced a column that silently vanished from a model
matrix rather than an error.

ORDER IS NOT INCIDENTAL, which is why this module does not impose one. The
merged-array plane order, the object-table order and the crop-mode order are
each load-bearing in their own module and are deliberately kept there. What
lives here is the MEMBERSHIP and the distinction between roles, so a new kind
is declared once.

Instruction 76 (support more than one organelle) is what this is for: the
vocabulary has to become derivable before a second organelle can be added to
it.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .schema import (ALL_ROLES, CHILD_ROLES, DERIVED_ROLES,
                     ORGANELLE_ROLES, SEGMENTED_ROLES)

#: Organelle slots use letter suffixes internally because object types are
#: embedded directly in underscore-separated object keys: ``organelle_2``
#: cannot round-trip through ``prcfo`` and ``organelle2`` is ambiguous with
#: label 2. The first spelling is the legacy slot; the display/index helpers
#: below present them as Organelle 1, 2, ... to users.
#: Kinds that are DERIVED from the segmented ones rather than found in a
#: channel. ``cytoplasm`` is cell-minus-nucleus-and-the-rest, so it has no
#: channel, no diameter and no detection method -- which is exactly why
#: :mod:`spacr.diameter` and :mod:`spacr.validate` leave it out of their own
#: lists, and why a per-object neighbour measurement is meaningless for it.
#: The segmented kinds that BELONG TO A CELL -- everything except the cell
#: itself. Each is many-rows-per-cell and carries its parent's label in a
#: ``cell_id`` column, which is what makes them roll up the same way.
#:
#: `io._read_and_join_tables` and `io._read_and_merge_data` both used to spell
#: this as the literal ``['nucleus', 'pathogen']``, so ORGANELLE WAS ABSENT
#: FROM BOTH: asking for it returned a frame with no organelle columns and no
#: message. Naming it once here is what lets a second organelle reach every
#: reader by being added in one place -- which is the whole of instruction 76.
def is_segmented(role: str) -> bool:
    """True when ``role`` is found in a channel rather than derived.

    :param role: object kind, e.g. ``"nucleus"``. Unknown names are False
        rather than an error, because callers use this to decide whether to
        look for a channel setting and an unknown kind simply has none.
    """
    return role in SEGMENTED_ROLES


def is_organelle(role: str) -> bool:
    """True when ``role`` is one of the closed organelle slots."""
    return str(role) in ORGANELLE_ROLES


def organelle_index(role: str) -> int:
    """Return the one-based user-facing index of an organelle slot."""
    try:
        return ORGANELLE_ROLES.index(str(role)) + 1
    except ValueError as exc:
        raise ValueError(
            f"{role!r} is not an organelle role; expected one of "
            f"{list(ORGANELLE_ROLES)}") from exc


def organelle_label(role: str) -> str:
    """Human-readable label for a slot (``Organelle 1``, ``Organelle 2``)."""
    return f"Organelle {organelle_index(role)}"


def setting_label(key: str) -> str:
    """Humanise a setting key, giving organelle slots numbered labels."""
    key = str(key)
    for role in sorted(ORGANELLE_ROLES, key=len, reverse=True):
        if key == role or key.startswith(f'{role}_'):
            suffix = key[len(role):].lstrip('_').replace('_', ' ')
            return (organelle_label(role) if not suffix else
                    f'{organelle_label(role)} — {suffix.capitalize()}')
    return key.replace('_', ' ').strip().capitalize()


def role_setting(role: str, suffix: str) -> str:
    """Return the setting key for ``suffix`` in one segmented role."""
    role = str(role)
    if role not in SEGMENTED_ROLES:
        raise ValueError(f"{role!r} is not a segmented role")
    return f"{role}_{str(suffix).lstrip('_')}"


def enabled_organelle_roles(settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """Organelle slots whose ``<role>_channel`` is enabled, in plane order."""
    return tuple(role for role in ORGANELLE_ROLES
                 if settings.get(role_setting(role, "channel")) is not None)


def organelle_settings_view(settings: Mapping[str, Any], role: str) -> Dict[str, Any]:
    """Return a copy exposing one slot through the legacy ``organelle_*`` API.

    The classical organelle segmenter predates slots and reads roughly forty
    ``organelle_*`` keys. Keeping that well-tested implementation and adapting
    one settings view at its boundary prevents four copies of the algorithm.
    """
    if role not in ORGANELLE_ROLES:
        raise ValueError(f"unknown organelle role {role!r}")
    out = dict(settings)
    if role == "organelle":
        return out
    prefix = f"{role}_"
    for key, value in settings.items():
        if str(key).startswith(prefix):
            out[f"organelle_{str(key)[len(prefix):]}"] = value
    recorded = settings.get(f"cellpose_{role}_channel")
    if recorded is not None:
        out["cellpose_organelle_channel"] = recorded
    return out


def ordered(*roles: str) -> Tuple[str, ...]:
    """Return ``roles`` as a tuple, checking each is a real object kind.

    For declaring a module's own ORDER while still being told when a name is
    wrong. The point is that a module keeps its ordering -- which may be a
    plane order or a table order and cannot be centralised -- without also
    keeping its own private copy of what the names are.

    :param roles: object kinds in this module's required order.
    :raises ValueError: if a name is not in :data:`ALL_ROLES`, naming it and
        listing the valid kinds.
    """
    unknown = [role for role in roles if role not in ALL_ROLES]
    if unknown:
        raise ValueError(
            f"unknown object role(s) {unknown}; expected some ordering of "
            f"{list(ALL_ROLES)}")
    return tuple(roles)

# ---------------------------------------------------------------------------
# The anchor: one concept, two column names
# ---------------------------------------------------------------------------

#: EVERY OBJECT TABLE IS ANCHORED TO THE CELL, and the column carrying that
#: anchor has two names depending on the table:
#:
#:     cell, cytoplasm                    'object_label'
#:     nucleus, pathogen, organelle,
#:     png_list                           'cell_id'
#:
#: cell and cytoplasm are ONE ROW PER CELL -- a cytoplasm is the cell minus
#: its interior objects, so its own label IS the cell's. The rest are MANY
#: rows per cell and carry the parent's label in ``cell_id``.
#:
#: A merge that does not translate between the two is joining on a
#: coincidence. `merge_tables` assumed every non-primary table was keyed by
#: ``cell_id`` and therefore SILENTLY DROPPED CYTOPLASM -- it logged a line
#: about an unlinkable table and returned a frame with no cytoplasm columns.
ANCHOR_COLUMN: Dict[str, str] = {
    "cell": "object_label",
    "cytoplasm": "object_label",
    "nucleus": "cell_id",
    "pathogen": "cell_id",
    **{role: "cell_id" for role in ORGANELLE_ROLES},
    "png_list": "cell_id",
}

#: Tables holding exactly one row per cell. They are JOINED to the cell
#: directly; everything else must be aggregated onto it first, or the cell's
#: own measurements fan out across its children.
ONE_ROW_PER_CELL: Tuple[str, ...] = ("cell", "cytoplasm")


def anchor_column(table: str) -> str:
    """The column in ``table`` that carries the cell it belongs to.

    :param table: an object table name.
    :returns: ``'object_label'`` or ``'cell_id'``.
    :raises ValueError: for a table with no declared anchor, naming the ones
        that have. Guessing produces a join on a coincidence.
    """
    key = str(table).strip().lower()
    if key not in ANCHOR_COLUMN:
        raise ValueError(
            f"no anchor column declared for table {table!r}; known tables are "
            f"{sorted(ANCHOR_COLUMN)}")
    return ANCHOR_COLUMN[key]


def is_one_row_per_cell(table: str) -> bool:
    """True when ``table`` holds one row per cell and needs no roll-up."""
    return str(table).strip().lower() in ONE_ROW_PER_CELL


#: How each table joins onto the cell, and why the answers differ.
#:
#: ``nucleus`` is INNER because a cell object with no nucleus is not a cell:
#: it is debris, a segmentation artefact, or a fragment at the image edge.
#: Keeping it adds a row with every nuclear measurement missing, which then
#: propagates as NaN through every ratio computed from it.
#:
#: ``png_list`` is INNER because a cell with no crop cannot be classified,
#: annotated or shown. Carrying it left the classification stage with rows it
#: could only drop later, after the counts had already been reported.
#:
#: ``cytoplasm`` is LEFT and it makes no difference: it is one row per cell,
#: derived from the cell mask, so it exists exactly when the cell does.
#:
#: ``pathogen`` and ``organelle`` are LEFT, and this one is a real choice --
#: an UNINFECTED cell is a cell, and in a screen it is usually the control
#: population. Dropping it would silently condition every result on infection.
#: :func:`join_how` takes the setting that reverses it.
JOIN_HOW: Dict[str, str] = {
    "cytoplasm": "left",
    "nucleus": "inner",
    "pathogen": "left",
    **{role: "left" for role in ORGANELLE_ROLES},
    "png_list": "inner",
}


def join_how(table: str, *, keep_uninfected: bool = True) -> str:
    """Whether ``table`` keeps cells it has no rows for.

    :param table: an object table name.
    :param keep_uninfected: ``False`` restricts the analysis to cells that
        actually contain a pathogen (or organelle), turning those joins
        inner. It does not touch nucleus or png_list, which are inner
        regardless -- a cell with no nucleus is not an uninfected cell, it is
        not a cell.
    :returns: ``'left'`` or ``'inner'``, for :meth:`pandas.DataFrame.merge`.
    """
    key = str(table).strip().lower()
    how = JOIN_HOW.get(key, "left")
    if not keep_uninfected and (key == "pathogen" or key in ORGANELLE_ROLES):
        return "inner"
    return how
