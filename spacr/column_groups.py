"""Choosing measurement columns by NAME, in groups. Instruction 49.

    "the user should be able to pick the columns that are dimentionally
     reduced, they should be able to do this through choosing individual
     columns for dimentional reduction and they should be able to chhose
     categories based on name lik cell, cor channel_1 or intensity
     measurements (have channel i thing), or morphology measurements, etc."

Three ways of naming the same set, because a measurement table names a column
three ways at once -- ``cell_channel_1_mean_intensity`` is a CELL measurement,
a CHANNEL 1 measurement and an INTENSITY measurement, and which of those a
user means depends on the question:

    object      cell, nucleus, pathogen, cytoplasm, organelle
    channel     channel_0, channel_1, ...
    family      morphology, intensity, texture, correlation, moment

THE FAMILIES ARE NOT INVENTED HERE. They come from
:data:`spacr.feature_dict.FEATURE_FAMILIES`, which is the dictionary that
already documents every column, and the classification is
:func:`spacr.feature_dict.parse_column`. A second taxonomy would be a second
thing to keep in step with the measurement code, and it would disagree first
in exactly the corners nobody checks.

Qt-free, so the picker's logic is testable without a display and the same
grouping can serve the CLI, a notebook, or a future screen.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

#: The three ways a group can be named, in the order a picker should offer
#: them: what the object IS, then which channel, then what kind of quantity.
GROUP_KINDS: Tuple[str, ...] = ("object", "channel", "family")

#: Families that are never useful as reduction inputs. `meta` is identifiers
#: and bookkeeping -- feeding a plate id to UMAP embeds the plate, which is
#: the batch effect rather than the biology. `unknown` is not a judgement
#: about the column, only that this dictionary could not classify it, so it
#: stays available and simply is not offered as a named group.
NON_FEATURE_FAMILIES: Tuple[str, ...] = ("meta",)


def _entry(column: str):
    from .feature_dict import parse_column

    try:
        return parse_column(str(column))
    except Exception:
        return None


def classify(columns: Iterable[str]) -> Dict[str, Dict[str, List[str]]]:
    """Group ``columns`` by object, by channel and by family.

    :param columns: the table's column names.
    :returns: ``{kind: {group name: [column, ...]}}`` for the kinds in
        :data:`GROUP_KINDS`. A column can appear in one group of each kind,
        which is the point -- ``cell_channel_1_mean_intensity`` is in
        ``object/cell``, ``channel/channel_1`` and ``family/intensity``.
    """
    out: Dict[str, Dict[str, List[str]]] = {k: {} for k in GROUP_KINDS}
    for column in columns:
        entry = _entry(column)
        if entry is None:
            continue
        family = str(getattr(entry, "family", "unknown") or "unknown")
        if family in NON_FEATURE_FAMILIES:
            # Not offered, and not silently deleted either: `meta` columns
            # are still in the table and still selectable one by one.
            continue
        object_type = getattr(entry, "object_type", None)
        if object_type:
            out["object"].setdefault(str(object_type), []).append(str(column))
        channel = getattr(entry, "channel", None)
        if channel is not None:
            out["channel"].setdefault(f"channel_{int(channel)}",
                                      []).append(str(column))
        out["family"].setdefault(family, []).append(str(column))
    for kind in out:
        for name in out[kind]:
            out[kind][name].sort()
    return out


def group_names(columns: Iterable[str]) -> Dict[str, List[str]]:
    """The group names a picker should offer, per kind, sorted.

    Channels sort NUMERICALLY -- ``channel_2`` before ``channel_10`` -- which
    a plain string sort gets wrong the moment a run has more than ten.
    """
    grouped = classify(columns)
    names: Dict[str, List[str]] = {}
    for kind, groups in grouped.items():
        if kind == "channel":
            names[kind] = sorted(
                groups, key=lambda n: int(n.rsplit("_", 1)[-1])
                if n.rsplit("_", 1)[-1].isdigit() else 0)
        else:
            names[kind] = sorted(groups)
    return names


def columns_in(columns: Iterable[str], kind: str, name: str) -> List[str]:
    """Every column in one named group.

    :raises KeyError: an unknown kind, naming the ones there are. A typo here
        would otherwise select nothing and read as "this table has no
        intensity measurements".
    """
    if kind not in GROUP_KINDS:
        raise KeyError(f"{kind!r} is not a group kind; choose from "
                       f"{list(GROUP_KINDS)}")
    return classify(columns)[kind].get(str(name), [])


def resolve(columns: Iterable[str],
            selection: Mapping[str, Sequence[str]] | None = None,
            *, explicit: Sequence[str] = ()) -> List[str]:
    """The columns a selection actually means, de-duplicated and in order.

    :param selection: ``{kind: [group name, ...]}`` -- the groups ticked.
    :param explicit: individual columns ticked, which are added to whatever
        the groups select. Both halves of the request are the same list in
        the end, so a user can tick "intensity" and then add one morphology
        column without the two mechanisms fighting.
    :returns: the columns, in the order they appear in ``columns``, so a
        reduction's input order does not depend on which checkbox was clicked
        first -- a UMAP whose axes depend on click order is not reproducible.
    """
    wanted = set()
    grouped = classify(columns)
    for kind, names in dict(selection or {}).items():
        if kind not in grouped:
            raise KeyError(f"{kind!r} is not a group kind; choose from "
                           f"{list(GROUP_KINDS)}")
        for name in names:
            wanted.update(grouped[kind].get(str(name), []))
    wanted.update(str(c) for c in explicit)
    ordered = [str(c) for c in columns if str(c) in wanted]
    return ordered


def summarise(columns: Iterable[str],
              selection: Mapping[str, Sequence[str]] | None = None,
              *, explicit: Sequence[str] = ()) -> str:
    """One line saying what is selected, for the picker to show.

    A reduction over 400 columns and one over 4 look identical in a dialog
    until something says which it is.
    """
    chosen = resolve(columns, selection, explicit=explicit)
    total = len([c for c in columns])
    if not chosen:
        return "no columns selected"
    return f"{len(chosen)} of {total} columns selected"
