"""The repeated per-object settings, as one row per question.

Instruction 364 measured it: 78 of Mask's 201 settings are the SAME 20
questions asked once per object type -- ``cell_diameter``,
``nucleus_diameter``, ``pathogen_diameter``, ``organelle_diameter`` and so on
across twenty shapes. The maintainer chose a table over tabs and over leaving
the names flat.

WHY A TABLE IS NOT A COSMETIC CHOICE. Instruction 326 makes the organelle
count arbitrary, up to :data:`spacr.organelle_types.MAX_ORGANELLES` of 26. A
flat vocabulary grows by TWENTY SETTINGS per organelle, so the module would
be asking several hundred questions at the ceiling. In a table a new
organelle is a new COLUMN and the number of questions does not move. The two
instructions meet exactly here, and this is the shape that lets 326 land.

WHAT THIS MODULE IS AND IS NOT. It is the MODEL: it reads a flat settings
dict into a table and writes a table back out, losslessly. It draws nothing.
A GUI renders :func:`to_table` and calls :func:`from_table` on edit, and the
headless path never has to know the table exists -- the stored keys are
unchanged, so no settings file, notebook or tutorial migrates.

  THAT IS THE WHOLE POINT OF DOING IT THIS WAY. The obvious alternative --
  storing a nested ``{question: {object: value}}`` structure -- would be a
  file-format change, and every saved settings file, every published
  tutorial, and the headless ``spacr-run`` path would have to move with it.
  The presentation is what was wrong, so only the presentation changes.
"""
from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from .object_roles import organelle_index, organelle_label
from .organelle_types import ALL_ORGANELLE_ROLES

__all__ = [
    "OBJECT_ORDER",
    "column_label",
    "families",
    "from_table",
    "questions",
    "to_table",
]

#: The objects a question can be asked about, in the order a table shows
#: them. The four segmented kinds first, then the organelle slots, then
#: ``cytoplasm`` -- which is DERIVED (cell minus the rest) and so answers far
#: fewer of the questions, and putting it last keeps the sparse column out of
#: the middle of the table.
OBJECT_ORDER: Tuple[str, ...] = (
    ("cell", "nucleus", "pathogen") + tuple(ALL_ORGANELLE_ROLES) + ("cytoplasm",)
)

#: Longest object prefix first, so ``organelleb_min_area`` is split at
#: ``organelleb`` and not at ``organelle``. Getting this backwards silently
#: files every organelle slot under the first one.
_PREFIXES = tuple(sorted(OBJECT_ORDER, key=len, reverse=True))


def _split(key: str) -> Optional[Tuple[str, str]]:
    """``('cell', 'min_area')`` for ``'cell_min_area'``, else ``None``."""
    for prefix in _PREFIXES:
        if key.startswith(prefix + "_"):
            return prefix, key[len(prefix) + 1:]
    return None


def questions(keys: Iterable[str]) -> "List[str]":
    """The distinct questions in ``keys``, in first-seen order.

    ONE ENTRY PER SHAPE, however many objects ask it. ``cell_diameter`` and
    ``nucleus_diameter`` are one question, which is the entire saving.
    """
    seen: "OrderedDict[str, None]" = OrderedDict()
    for key in keys:
        split = _split(key)
        if split is not None:
            seen.setdefault(split[1], None)
    return list(seen)


def families(keys: Iterable[str]) -> "Dict[str, List[str]]":
    """``{question: [objects that ask it]}``, objects in display order."""
    found: "Dict[str, set]" = {}
    for key in keys:
        split = _split(key)
        if split is None:
            continue
        obj, question = split
        found.setdefault(question, set()).add(obj)
    order = {name: index for index, name in enumerate(OBJECT_ORDER)}
    return {question: sorted(objects, key=lambda o: order.get(o, len(order)))
            for question, objects in found.items()}


def column_label(obj: str) -> str:
    """What a column header reads.

    ``organelleb`` is a storage spelling, not a thing a user recognises. The
    letter suffixes exist because object types are embedded in
    underscore-separated object keys and ``organelle2`` is ambiguous with
    label 2 -- an implementation constraint that has no business appearing in
    a column header.
    """
    if obj.startswith("organelle"):
        return organelle_label(obj)
    return obj.replace("_", " ").capitalize()


def to_table(settings: Mapping[str, object]) -> "Dict[str, Dict[str, object]]":
    """Read a flat settings dict as ``{question: {object: value}}``.

    Only the keys that ARE per-object questions are taken. Everything else --
    ``src``, ``verbose``, ``n_jobs`` -- is not a table row and is left where
    it is; a caller wanting the whole settings dict back uses
    :func:`from_table` with the original.

    A question a given object does not ask is ABSENT from its row rather than
    present as ``None``. ``cytoplasm`` has no channel, no diameter and no
    detection method because it is derived rather than found in a channel,
    and a blank cell says that where a ``None`` would read as "not set yet".
    """
    order = {name: index for index, name in enumerate(OBJECT_ORDER)}
    table: "Dict[str, Dict[str, object]]" = OrderedDict()
    for key, value in settings.items():
        split = _split(key)
        if split is None:
            continue
        obj, question = split
        table.setdefault(question, {})[obj] = value
    for question, row in table.items():
        table[question] = {
            obj: row[obj]
            for obj in sorted(row, key=lambda o: order.get(o, len(order)))
        }
    return table


def from_table(table: Mapping[str, Mapping[str, object]],
               base: Optional[Mapping[str, object]] = None) -> "Dict[str, object]":
    """Flatten a table back to settings keys, over ``base`` if given.

    LOSSLESS WITH :func:`to_table`, which is what makes the table safe to use
    as an editing surface: what the user sees is the settings file, rearranged
    and not transformed. A test round-trips every default settings dict.

    :param table: ``{question: {object: value}}``.
    :param base: the settings the table came from. Keys the table does not
        cover are carried through unchanged, so a caller can edit the
        per-object half without holding the rest.
    """
    out: "Dict[str, object]" = dict(base or {})
    for question, row in table.items():
        for obj, value in row.items():
            out[f"{obj}_{question}"] = value
    return out


def widen(table: Mapping[str, Mapping[str, object]], obj: str,
          *, like: Optional[str] = None) -> "Dict[str, Dict[str, object]]":
    """Add a column for ``obj``, copying ``like``'s answers where it has them.

    THE OPERATION 326 NEEDS, and the reason the table exists. Adding an
    organelle is one call that adds one column; in the flat vocabulary it is
    twenty new settings that every consumer, tooltip table and translation
    catalog has to learn.

    :param obj: the object to add, e.g. ``'organellec'``.
    :param like: an existing object whose answers to copy as the starting
        point. Defaults to the first organelle when ``obj`` is one, because a
        second mitochondrion should start where the first one is rather than
        at a global default nobody chose.
    """
    if like is None and obj.startswith("organelle"):
        like = "organelle"
    widened: "Dict[str, Dict[str, object]]" = OrderedDict()
    order = {name: index for index, name in enumerate(OBJECT_ORDER)}
    for question, row in table.items():
        new_row = dict(row)
        if like is not None and like in row and obj not in row:
            new_row[obj] = row[like]
        widened[question] = {
            o: new_row[o] for o in sorted(new_row, key=lambda o: order.get(o, len(order)))
        }
    return widened
