"""Define classification classes from annotation or plate-metadata values.

Each class maps a display name to either a source-column/value pair or a
random-complement rule. For example::

    {"infected": {"column": "annot_1", "value": 1},
     "uninfected": {"column": "annot_2", "value": 0}}

This representation supports classes derived from different annotation
columns. In metadata mode, the same rules may reference plate, row, column,
field, or well identifiers. :func:`normalize_settings` converts supported
legacy classification settings to this representation without mutating the
input mapping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

LOG = logging.getLogger("spacr.classify_classes")

#: The dict setting itself.
CLASSES = "classes"

#: The ordered training-folder names. Split out of :data:`CLASSES`, which was
#: carrying both this and the class definitions -- see :func:`folder_names`.
CLASS_FOLDER_NAMES = "class_folder_names"

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
    "location_column", "positive_control_id", "negative_control_id",
)


class ClassDefinitionError(ValueError):
    """A class definition that cannot select objects, and why."""


@dataclass(frozen=True)
class ClassRule:
    """One class: its name, and what makes an object a member.

    :param name: nonblank class label written to matched rows and retained as
        the ordered training and folder name.
    :param column: source-table column compared by an explicit rule; leave it
        blank only for a random-complement rule.
    :param value: exact value selected by equality in ``column`` for an
        explicit rule.
    :param random_complement: when true, sample unclaimed rows with
        :func:`assign_classes`'s seed, up to the largest explicit class size;
        it cannot be combined with ``column`` or ``value``.

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
        """Reject blank names and contradictory or incomplete selectors."""
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
        """Return the serializable selector shape stored in ``classes``."""
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

    :param settings: classification settings whose resolved dataset basis
        decides whether coordinate metadata or annotation columns are offered.
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


def values_in(frame: Any, column: str,
              *, limit: int = 100) -> Tuple[Any, ...]:
    """The distinct values of ``column`` -- the keys the dict is populated with.

    Nulls are excluded: "not annotated" is the absence of a class, and
    offering it as one is how a user ends up training on their own blanks.

    :param frame: table containing the candidate class column.
    :param column: column whose distinct non-null values define class choices.
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



def class_rules(settings: Mapping[str, Any]) -> Tuple[ClassRule, ...]:
    """The classes a settings dict defines, in the order they were given.

    :param settings: classification settings containing the class definitions.

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
        names = ", ".join(repr(str(n)) for n in raw)
        raise ClassDefinitionError(
            f"{CLASSES} names {len(raw)} class(es) ({names}) but nothing "
            f"says which objects belong to them. Give each one a column and "
            f"a value in the Classes editor, or set annotation_column / "
            f"class_metadata so they can be derived. (If {CLASSES} has not "
            f"been through normalize_settings yet, that is what runs the "
            f"derivation.)")

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

    :param settings: classification settings whose class names are requested.

    Downstream (``deep_spacr``, ``model_zoo``, the evaluation code) reads a
    list of names and should keep doing so. This is what
    :func:`normalize_settings` writes back under
    :data:`CLASS_FOLDER_NAMES` so none of that has to learn the dict.
    """
    raw = settings.get(CLASSES)
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    return [r.name for r in class_rules(settings)]


def folder_names(settings: Mapping[str, Any]) -> List[str]:
    """Return ordered class-folder names for model training.

    Current settings derive folder names from the ordered keys of the
    ``classes`` definition. For compatibility, a legacy list-valued
    ``classes`` setting takes precedence. When no class definitions are
    present, the function uses ``class_folder_names``, which records the
    folders written by dataset generation. Invalid or absent definitions
    produce an empty list unless that recorded folder list is available.

    Parameters
    ----------
    settings : mapping
        Classification settings in current or legacy form.

    Returns
    -------
    list of str
        Folder names in class-label order.
    """
    legacy = settings.get(CLASSES)
    if isinstance(legacy, (list, tuple)):
        return [str(n) for n in legacy]

    if isinstance(settings.get(CLASSES), Mapping) and settings.get(CLASSES):
        try:
            defined = class_names(settings)
        except ClassDefinitionError:
            defined = []
        if defined:
            return defined

    raw = settings.get(CLASS_FOLDER_NAMES)
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    try:
        return class_names(settings)
    except ClassDefinitionError:
        return []


def _record_generated_folder_names(
    settings: MutableMapping[str, Any], names: Sequence[Any],
) -> List[str]:
    """Record folders written to disk and retire the ambiguous legacy list."""
    recorded = [str(name) for name in names]
    settings[CLASS_FOLDER_NAMES] = recorded
    if isinstance(settings.get(CLASSES), (list, tuple)):
        settings.pop(CLASSES, None)
    return recorded



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

    names = folder_names(settings)

    rules: List[ClassRule] = []
    for i, value in enumerate(values):
        name = str(names[i]) if i < len(names) else f"class_{value}"
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
    names = folder_names(settings)

    rules: List[ClassRule] = []
    for i, key in enumerate(("negative_control_id", "positive_control_id")):
        value = settings.get(key)
        if value in (None, ""):
            continue
        name = (names[i] if i < len(names)
                else key.removesuffix("_id").replace("_", " "))
        if isinstance(value, (list, tuple)):
            for item in value:
                rules.append(ClassRule(name=name, column=column, value=item))
        else:
            rules.append(ClassRule(name=name, column=column, value=value))
    return rules


def normalize_settings(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``settings`` with :data:`CLASSES` as a dict. Never mutates.

    :param settings: current or legacy classification settings to normalize.

    The translation happens ONCE, here, so no downstream reader has to know
    both shapes. A settings CSV written before this produces the same classes
    it did before -- which is the whole requirement, and what the tests
    assert.

    ``class_names`` is written alongside, in order, because that is what
    ``deep_spacr`` and the evaluation code read.
    """
    out = dict(settings)
    raw = out.get(CLASSES)
    legacy_names = (
        [str(name) for name in raw]
        if isinstance(raw, (list, tuple)) else None
    )

    if not isinstance(raw, Mapping) or not raw:
        from .training_basis import resolve_basis

        basis = resolve_basis(out)
        rules = (_rules_from_metadata(out) if basis == "metadata"
                 else _rules_from_annotation(out))
        if not rules and isinstance(raw, (list, tuple)) and raw:
            LOG.info("settings name %d class(es) but nothing says which "
                     "objects belong to them; leaving them as names", len(raw))
        elif rules:
            out[CLASSES] = {r.name: r.to_dict() for r in rules}

    if legacy_names is not None:
        names = legacy_names
        out[CLASS_FOLDER_NAMES] = names
        out["class_names"] = names
    else:
        names = folder_names(out)
        if CLASS_FOLDER_NAMES not in out and names:
            out[CLASS_FOLDER_NAMES] = names
        if names or isinstance(out.get(CLASS_FOLDER_NAMES), (list, tuple)):
            out["class_names"] = names
    return out



def assign_classes(frame: Any, settings: Mapping[str, Any], *,
                   seed: Optional[int] = 0) -> Any:
    """Label every row with its class name, or NA.

    The random complement is drawn from the rows NO rule claimed, sized to
    match the largest explicit class so the training set is not lopsided by
    accident -- a comparison group ten times the size of the class it is
    compared against teaches the model the prior, not the difference.

    :param frame: object table whose rows are to be labelled. Rule column names
        are resolved against this table.
    :param settings: classification settings containing the ordered
        :data:`CLASSES` definitions.
    :param seed: fixes the random complement. A training set that changes
        every time it is built cannot be compared with the run before it.
    :returns: a Series of class names aligned to ``frame``.
    :raises ClassDefinitionError: a rule naming a column the table lacks.
    """
    rules = class_rules(settings)
    if not rules:
        raise ClassDefinitionError(
            "no classes are defined; set the column and name its values")

    import pandas as pd

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



def annotation_column_of(settings) -> str:
    """Return the annotation column represented by class settings.

    The first non-empty ``column`` in the ordered ``classes`` mapping is
    returned. If no class rule supplies a column, the legacy
    ``annotation_column`` value is used.

    Parameters
    ----------
    settings : mapping
        Classification settings in current or legacy form.

    Returns
    -------
    str
        Annotation column name, or an empty string when none is defined.
    """
    raw = settings.get(CLASSES)
    if isinstance(raw, Mapping):
        for rule in raw.values():
            if isinstance(rule, Mapping):
                column = str(rule.get("column") or "").strip()
                if column:
                    return column
    return str(settings.get("annotation_column") or "").strip()


def class_metadata_of(settings) -> list:
    """Return legacy-shaped metadata values derived from class rules.

    Parameters
    ----------
    settings : mapping
        Classification settings in current or legacy form.

    Returns
    -------
    list
        Class values as ``[[value], ...]`` in class order. The legacy
        ``class_metadata`` list is returned when no current class rule has a
        value; otherwise an empty list is returned.
    """
    raw = settings.get(CLASSES)
    if isinstance(raw, Mapping) and raw:
        out = []
        for rule in raw.values():
            if isinstance(rule, Mapping) and rule.get("value") is not None:
                out.append([rule["value"]])
        if out:
            return out
    value = settings.get("class_metadata")
    return list(value) if isinstance(value, (list, tuple)) else []


def fold_into_classes(settings) -> dict:
    """Populate compatibility keys from current class definitions.

    Parameters
    ----------
    settings : mutable mapping
        Classification settings to update in place.

    Returns
    -------
    dict
        The same mapping, with non-empty ``annotation_column`` and
        ``class_metadata`` values derived from ``classes``.
    """
    column = annotation_column_of(settings)
    if column:
        settings["annotation_column"] = column
    values = class_metadata_of(settings)
    if values:
        settings["class_metadata"] = values
    return settings
