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

from typing import Tuple

#: Kinds that are SEGMENTED from an image channel -- each has its own mask,
#: its own ``<role>_channel`` setting and its own detection parameters.
SEGMENTED_ROLES: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "organelle")

#: Kinds that are DERIVED from the segmented ones rather than found in a
#: channel. ``cytoplasm`` is cell-minus-nucleus-and-the-rest, so it has no
#: channel, no diameter and no detection method -- which is exactly why
#: :mod:`spacr.diameter` and :mod:`spacr.validate` leave it out of their own
#: lists, and why a per-object neighbour measurement is meaningless for it.
DERIVED_ROLES: Tuple[str, ...] = ("cytoplasm",)

#: The segmented kinds that BELONG TO A CELL -- everything except the cell
#: itself. Each is many-rows-per-cell and carries its parent's label in a
#: ``cell_id`` column, which is what makes them roll up the same way.
#:
#: `io._read_and_join_tables` and `io._read_and_merge_data` both used to spell
#: this as the literal ``['nucleus', 'pathogen']``, so ORGANELLE WAS ABSENT
#: FROM BOTH: asking for it returned a frame with no organelle columns and no
#: message. Naming it once here is what lets a second organelle reach every
#: reader by being added in one place -- which is the whole of instruction 76.
CHILD_ROLES: Tuple[str, ...] = ("nucleus", "pathogen", "organelle")

#: Every object kind, segmented and derived.
ALL_ROLES: Tuple[str, ...] = SEGMENTED_ROLES + DERIVED_ROLES


def is_segmented(role: str) -> bool:
    """True when ``role`` is found in a channel rather than derived.

    :param role: object kind, e.g. ``"nucleus"``. Unknown names are False
        rather than an error, because callers use this to decide whether to
        look for a channel setting and an unknown kind simply has none.
    """
    return role in SEGMENTED_ROLES


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
