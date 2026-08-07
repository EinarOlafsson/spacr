"""What defines a training class: a dict of name -> (column, value).

``settings['classes']`` was a list of names -- ``['nc', 'pc']`` -- and
everything about which OBJECTS those names referred to lived somewhere else:
``annotation_column`` said where to look, ``annotated_classes`` said which
values counted, ``write_random_annotation_column`` invented a comparison group,
and in metadata mode ``location_column`` plus ``positive_control`` and
``negative_control`` said it all again in different words.

Four settings to say one thing, and none of them able to say "class A is
value 1 of column X and class B is value 3 of column Y". So:

    {"infected": {"column": "annot_1", "value": 1},
     "uninfected": {"column": "annot_2", "value": 0}}

The name is the key, because the name is what the user picks. Each rule names
the VALUE and the COLUMN it came from, which is what makes more than one
annotation column usable at once -- the thing the old shape could not express
at all.

**The random complement.** Annotating one class is the normal case: a user
marks the infected cells and stops. The second class is then "everything not
annotated", chosen at random, and that is a KIND OF RULE rather than a button
pressed beforehand -- ``{"control": {"random_complement": true}}``. This
replaces ``write_random_annotation_column``.

**Metadata mode changes only which columns are on offer.** Under
``dataset_mode='metadata'`` the same dict is filled from plate / row / column /
field / well instead of user-defined annotation columns, which is why
``location_column``, ``positive_control`` and ``negative_control`` are no
longer needed: "positive control is column 3" is exactly a rule.

**Old settings keep working.** Every retired key is translated here, once, by
:func:`normalize_settings` -- the same discipline as
:mod:`spacr.training_basis`. A settings CSV written before this exists in
every user's project folder, and a run from one has to produce the same
training set it did before.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger("spacr.classify_classes")

#: The dict setting itself.
CLASSES = "classes"

#: Columns the dict offers under ``dataset_mode='metadata'``. Fixed, because
#: they are the plate's own coordinates rather than anything the user named.
METADATA_COLUMNS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "well",
)

#: Settings this module replaces. Each is translated on read and must not be
#: written back out; :data:`spacr.settings.DEAD_SETTINGS` is where they go once
#: nothing reads them.
RETIRED = (
    "annotation_column", "annotation_columns", "annotation_values",
    "annotated_classes", "write_random_annotation_column",
    "location_column", "positive_control", "negative_control",
)


class ClassDefinitionError(ValueError):
    """A class definition that cannot select objects, and why."""


@dataclass(frozen=True)
class ClassRule:
    """One class: its name, and what makes an object a member.

    Either a ``column``/``value`` pair, or ``random_complement`` -- never
    both. A rule that says both would have two answers for the same object and
    no way to choose between them.
    """

    name: str
    column: str = ""
    value: Any = None
    #: This class is "everything not claimed by another rule", sampled at
    #: random. At most one rule may say so.
    random_complement: bool = False

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ClassDefinitionError("a class must have a name")
        if self.random_complement:
            if self.column or self.value is not None:
                raise ClassDefinitionError(
                    f"class {self.name!r} is both a random complement and a "
                    f"rule on {self.column!r}; it can only be one")
            return
        if not str(self.column).strip():
            raise ClassDefinitionError(
                f"class {self.name!r} does not say which column its value "
                f"comes from, so it cannot select objects")

    def to_dict(self) -> Dict[str, Any]:
        if self.random_complement:
            return {"random_complement": True}
        return {"column": self.column, "value": self.value}


def candidate_columns(settings: Mapping[str, Any],
                      available: Sequence[str] = ()) -> Tuple[str, ...]:
    """The columns the Classes dict may be filled from.

    Under the metadata basis these are the plate's coordinates; otherwise they
    are whatever annotation columns the table actually has. The GUI populates
    the dict's keys from the VALUES of the chosen column, so this is the first
    half of "you set the column then the keys of this dict get populated".

    :param available: the table's columns, used to filter the metadata list --
        a database with no ``well`` column must not offer one.
    """
    from .training_basis import resolve_basis

    if resolve_basis(settings) == "metadata":
        if not available:
            return METADATA_COLUMNS
        lower = {str(c).lower(): str(c) for c in available}
        return tuple(lower[c.lower()] for c in METADATA_COLUMNS
                     if c.lower() in lower)
    return tuple(str(c) for c in available)


def values_in(frame: pd.DataFrame, column: str,
              *, limit: int = 100) -> Tuple[Any, ...]:
    """The distinct values of ``column`` -- the keys the dict is populated with.

    Nulls are excluded: "not annotated" is the absence of a class, and
    offering it as one is how a user ends up training on their own blanks.

    :param limit: refuse to enumerate a free-form column. Past this many
        distinct values it is a measurement, not a label, and the Gate Editor
        is what turns a measurement into a class.
    :raises ClassDefinitionError: the column is missing, or has too many
        distinct values to be a label.
    """
    if column not in frame.columns:
        raise ClassDefinitionError(
            f"column {column!r} is not in this table")
    values = frame[column].dropna().unique()
    if len(values) > limit:
        raise ClassDefinitionError(
            f"column {column!r} has {len(values):,} distinct values, which "
            f"makes it a measurement rather than a label; gate on it in the "
            f"Gate Editor instead")
    return tuple(values)


# ---------------------------------------------------------------------------
# Reading the setting
# ---------------------------------------------------------------------------

def class_rules(settings: Mapping[str, Any]) -> Tuple[ClassRule, ...]:
    """The classes a settings dict defines, in the order they were given.

    Order matters: it is the label order the model is trained with, so it has
    to be stable rather than whatever a set iterates in.

    :raises ClassDefinitionError: a malformed dict, or more than one random
        complement -- two classes both meaning "everything else" have no
        boundary between them.
    """
    raw = settings.get(CLASSES)
    if raw is None or (hasattr(raw, "__len__") and len(raw) == 0):
        return ()

    if not isinstance(raw, Mapping):
        # A plain list of names is the OLD shape. It says nothing about which
        # objects belong to which class, so it cannot be turned into rules
        # here -- `normalize_settings` translates it, using the other retired
        # keys, before anything asks.
        raise ClassDefinitionError(
            f"{CLASSES} is a {type(raw).__name__}, not a dict of "
            f"name -> {{column, value}}; run normalize_settings first")

    rules: List[ClassRule] = []
    for name, spec in raw.items():
        if isinstance(spec, Mapping):
            rules.append(ClassRule(
                name=str(name),
                column=str(spec.get("column", "") or ""),
                value=spec.get("value"),
                random_complement=bool(spec.get("random_complement", False))))
        else:
            raise ClassDefinitionError(
                f"class {name!r} is defined as {spec!r}; it needs a column "
                f"and a value, or random_complement")

    complements = [r for r in rules if r.random_complement]
    if len(complements) > 1:
        raise ClassDefinitionError(
            "more than one class is a random complement ("
            + ", ".join(r.name for r in complements)
            + "); two classes that both mean 'everything else' have no "
              "boundary between them")
    return tuple(rules)


def class_names(settings: Mapping[str, Any]) -> List[str]:
    """The class names, in order -- what ``settings['classes']`` used to be.

    Downstream (``deep_spacr``, ``model_zoo``, the evaluation code) reads a
    list of names and should keep doing so. This is what
    :func:`normalize_settings` writes back under ``class_names`` so none of
    that has to learn the dict.
    """
    raw = settings.get(CLASSES)
    if isinstance(raw, (list, tuple)):
        # The old shape, either untranslated or left alone because nothing
        # said what its names select. Reading the names off it is always
        # right; `class_rules` refuses it because it cannot make RULES from
        # names, which is a different question.
        return [str(n) for n in raw]
    return [r.name for r in class_rules(settings)]


# ---------------------------------------------------------------------------
# Translating what came before
# ---------------------------------------------------------------------------

def _rules_from_annotation(settings: Mapping[str, Any]) -> List[ClassRule]:
    """Rebuild rules from ``annotation_column`` + ``annotated_classes``."""
    columns = settings.get("annotation_columns") or settings.get(
        "annotation_column")
    if isinstance(columns, str):
        columns = [columns]
    columns = [str(c) for c in (columns or []) if str(c).strip()]
    if not columns:
        return []

    values = settings.get("annotation_values") or settings.get(
        "annotated_classes") or []
    if not isinstance(values, (list, tuple)):
        values = [values]

    names = settings.get(CLASSES)
    names = list(names) if isinstance(names, (list, tuple)) else []

    rules: List[ClassRule] = []
    for i, value in enumerate(values):
        name = str(names[i]) if i < len(names) else f"class_{value}"
        # One column and several values is the common case; several columns
        # pairs them off positionally, which is what the old readers did.
        column = columns[i] if i < len(columns) else columns[0]
        rules.append(ClassRule(name=name, column=column, value=value))

    if settings.get("write_random_annotation_column"):
        used = {r.name for r in rules}
        name = next((str(n) for n in names if str(n) not in used), "random")
        rules.append(ClassRule(name=name, random_complement=True))
    return rules


def _rules_from_metadata(settings: Mapping[str, Any]) -> List[ClassRule]:
    """Rebuild rules from ``location_column`` + the two control settings.

    "positive control is column 3" is exactly a rule, which is why those three
    settings retire rather than move.
    """
    column = str(settings.get("location_column") or "").strip()
    if not column:
        return []
    names = settings.get(CLASSES)
    names = [str(n) for n in names] if isinstance(names, (list, tuple)) else []

    rules: List[ClassRule] = []
    for i, key in enumerate(("negative_control", "positive_control")):
        value = settings.get(key)
        if value in (None, ""):
            continue
        name = names[i] if i < len(names) else key.replace("_", " ")
        # A control setting can name several wells.
        if isinstance(value, (list, tuple)):
            for item in value:
                rules.append(ClassRule(name=name, column=column, value=item))
        else:
            rules.append(ClassRule(name=name, column=column, value=value))
    return rules


def normalize_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``settings`` with :data:`CLASSES` as a dict. Never mutates.

    The translation happens ONCE, here, so no downstream reader has to know
    both shapes. A settings CSV written before this produces the same classes
    it did before -- which is the whole requirement, and what the tests
    assert.

    ``class_names`` is written alongside, in order, because that is what
    ``deep_spacr`` and the evaluation code read.
    """
    out = dict(settings)
    raw = out.get(CLASSES)

    if not isinstance(raw, Mapping):
        from .training_basis import resolve_basis

        basis = resolve_basis(out)
        rules = (_rules_from_metadata(out) if basis == "metadata"
                 else _rules_from_annotation(out))
        if not rules and isinstance(raw, (list, tuple)) and raw:
            # Names with nothing saying what they select. Left alone rather
            # than invented: a guessed column would train on the wrong labels
            # and report success.
            LOG.info("settings name %d class(es) but nothing says which "
                     "objects belong to them; leaving them as names", len(raw))
        elif rules:
            out[CLASSES] = {r.name: r.to_dict() for r in rules}

    names = class_names(out)
    if names:
        out.setdefault("class_names", names)
    return out


# ---------------------------------------------------------------------------
# Applying it
# ---------------------------------------------------------------------------

def assign_classes(frame: pd.DataFrame, settings: Mapping[str, Any], *,
                   seed: Optional[int] = 0) -> pd.Series:
    """Label every row with its class name, or NA.

    The random complement is drawn from the rows NO rule claimed, sized to
    match the largest explicit class so the training set is not lopsided by
    accident -- a comparison group ten times the size of the class it is
    compared against teaches the model the prior, not the difference.

    :param seed: fixes the random complement. A training set that changes
        every time it is built cannot be compared with the run before it.
    :returns: a Series of class names aligned to ``frame``.
    :raises ClassDefinitionError: a rule naming a column the table lacks.
    """
    rules = class_rules(settings)
    if not rules:
        raise ClassDefinitionError(
            "no classes are defined; set the column and name its values")

    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    claimed = pd.Series(False, index=frame.index)

    for rule in rules:
        if rule.random_complement:
            continue
        if rule.column not in frame.columns:
            raise ClassDefinitionError(
                f"class {rule.name!r} is defined on column {rule.column!r}, "
                f"which this table does not have")
        hit = frame[rule.column] == rule.value
        # First rule wins. Two rules matching the same object is a definition
        # the user has to fix, but silently relabelling is worse than keeping
        # the order they wrote.
        take = hit & ~claimed
        labels[take] = rule.name
        claimed |= hit

    complement = next((r for r in rules if r.random_complement), None)
    if complement is not None:
        pool = frame.index[~claimed]
        if len(pool) == 0:
            raise ClassDefinitionError(
                f"class {complement.name!r} is the unannotated objects, but "
                f"every object is already claimed by another class")
        counts = labels.value_counts()
        size = int(counts.max()) if len(counts) else len(pool)
        size = min(size, len(pool))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(np.asarray(pool), size=size, replace=False)
        labels.loc[chosen] = complement.name

    return labels
