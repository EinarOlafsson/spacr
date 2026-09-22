"""Shared object-role vocabulary and role-aware settings helpers.

The schema defines membership in segmented, derived, child, and organelle
roles. This module exposes those relationships to consumers that need labels,
setting keys, table anchors, or join behavior. Consumers retain their own
ordering where array planes, table layout, or crop modes require a specific
sequence; the shared registry defines which names are valid and how their
roles relate.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .schema import (  # noqa: F401
    ALL_ROLES,
    CHILD_ROLES,
    DERIVED_ROLES,
    ORGANELLE_ROLES,
    SEGMENTED_ROLES,
)


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
#: Centralising these relationships lets additional organelle slots reach all
#: table readers without duplicating role lists.
def is_segmented(role: str) -> bool:
    """True when ``role`` is found in a channel rather than derived.

    :param role: object kind, e.g. ``"nucleus"``. Unknown names are False
        rather than an error, because callers use this to decide whether to
        look for a channel setting; an unknown kind has none.
    :returns: ``True`` only for roles in :data:`SEGMENTED_ROLES`.
    """
    return role in SEGMENTED_ROLES


def is_organelle(role: str) -> bool:
    """True when ``role`` is one of the closed organelle slots.

    :param role: object-role name to test.
    :returns: ``True`` only for roles in :data:`ORGANELLE_ROLES`.
    """
    return str(role) in ORGANELLE_ROLES


def organelle_index(role: str) -> int:
    """Return the one-based user-facing index of an organelle slot.

    :param role: organelle role whose numeric slot is requested.
    :returns: The one-based slot encoded by ``role``.
    :raises ValueError: if ``role`` is not an organelle slot.

    ANSWERED FROM THE LETTER, not from a list of the slots that happen to
    segment today. The suffix IS the number -- organelle, organelleb,
    organellec -- so a slot the schema has no mask plane for still has a
    name, and a settings file carrying seven slots renders as
    "Organelle 5" rather than as "Organellee".
    """
    from .organelle_types import organelle_number

    try:
        return organelle_number(role)
    except ValueError as exc:
        raise ValueError(
            f"Cannot determine an organelle index: {exc}"
        ) from exc


def organelle_label(role: str) -> str:
    """Human-readable label for a slot (``Organelle 1``, ``Organelle 2``).

    :param role: organelle role to render for users.
    :returns: The numbered user-facing organelle label.
    :raises ValueError: if ``role`` is not an organelle slot.
    """
    from .organelle_types import organelle_slot_label

    return organelle_slot_label(role)


#: Terms whose established capitalization must survive label generation.
#: Python's ``str.capitalize`` lower-cases the remaining characters, so the
#: shared label formatter restores forms such as ``gRNA``, ``DNA``, and
#: ``UMAP`` consistently across every settings surface.
#: Keys whose label is not a de-underscored capitalisation at all.
#:
#: ``controls`` names guide or gene identifiers, whereas the neighbouring
#: control settings name wells. Use an explicit label so the two concepts are
#: not confused in settings forms.
EXACT_LABELS = {
    "nontargeting_control_grnas": "Control gRNA/Gene",
    "log_x": "Logarithmic x",
    "log_y": "Logarithmic y",
    "log_data": "Log-transform features",
    "src": "Source",
    "sample": "Sample size limit",
    "power": "Statistical power",
    "power_backend": "Statistical power — inference backend",
    "power_background_positive_rate": (
        "Statistical power — background positive-call rate"
    ),
    "power_cells_per_well": "Statistical power — cells per well",
    "power_constructs_per_well": (
        "Statistical power — library units per well"
    ),
    "power_detection_auroc": (
        "Statistical power — detection AUROC threshold"
    ),
    "power_effect_fold": "Statistical power — effect multiplier",
    "power_hit_rate": "Statistical power — hit probability",
    "power_n_genes": "Statistical power — genes",
    "power_n_grnas_per_gene": "Statistical power — gRNAs per gene",
    "power_n_plates": "Statistical power — plates",
    "power_n_replicates": "Statistical power — replicates",
    "power_reads_per_well": "Statistical power — reads per well",
    "power_score_per": "Statistical power — scoring level",
    "power_seed": "Statistical power — random seed",
    "power_wells_per_plate": "Statistical power — wells per plate",
}

CASED_TERMS = {
    "grna": "gRNA",
    "grnas": "gRNAs",
    "sgrna": "sgRNA",
    "sgrnas": "sgRNAs",
    "dna": "DNA",
    "rna": "RNA",
    "gpu": "GPU",
    "umap": "UMAP",
    "csv": "CSV",
    "png": "PNG",
    "qc": "QC",
    "id": "ID",
    "dog": "DoG",
}

#: Phrases that must be recased TOGETHER, because the word alone is
#: ambiguous. `log` is Laplacian of Gaussians in the blob-detector settings
#: and an ordinary logarithm in `log_x`, `log_y` and `log_data`, so it cannot
#: go in :data:`CASED_TERMS` -- a blanket rule would turn a log axis into a
#: filter. Same defect as `dog` above and found the same way: the label read
#: "Log max sigma", and the translators read it as a logbook. German
#: "Protokoll max sigma", Swedish "Logg max sigma", and Chinese rendered
#: `log_x` as "彩票X" -- lottery.
CASED_PHRASES = {
    "log max sigma": "LoG max sigma",
    "log min sigma": "LoG min sigma",
    "log num sigma": "LoG num sigma",
    "log threshold": "LoG threshold",
}


def _recase(text: str) -> str:
    """Restore the terms `capitalize()` flattened.

    :param text: Human-readable text whose established terms need recasing.
    :returns: The text with :data:`CASED_TERMS` spellings restored.
    """
    text = str(text)
    lowered = text.lower()
    for phrase, cased in CASED_PHRASES.items():
        if phrase in lowered:
            start = lowered.index(phrase)
            text = text[:start] + cased + text[start + len(phrase):]
            lowered = text.lower()
    return " ".join(CASED_TERMS.get(word.lower(), word)
                    for word in text.split(" "))


def _split_id_suffix(key: str) -> str:
    """Separate a terminal ``ID`` suffix from a camel-case setting key.

    Identifier columns such as ``plateID`` and ``objectID`` contain no
    underscore, so ordinary tokenization would render them as ``Plateid``.

    :param key: Setting key that may end in a camel-case ``ID`` suffix.
    :returns: The key with spacing inserted before a terminal ``ID``.
    """
    import re

    return re.sub(r'(?<=[a-z])(ID)\b', r' \1', str(key))


def setting_label(key: str) -> str:
    """Humanise a setting key, giving organelle slots numbered labels.

    :param key: canonical setting key to turn into a display label.
    :returns: The canonical human-readable setting label.
    """
    from .organelle_types import organelle_role_of

    key = str(key)
    role = organelle_role_of(key)
    if role is not None:
        suffix = key[len(role):].lstrip('_').replace('_', ' ')
        return (organelle_label(role) if not suffix else
                f'{organelle_label(role)} — '
                f'{_recase(suffix.capitalize())}')
    if key in EXACT_LABELS:
        return EXACT_LABELS[key]
    spaced = _split_id_suffix(key).replace('_', ' ').strip()
    return _recase(spaced.capitalize())


def role_setting(role: str, suffix: str) -> str:
    """Return the setting key for ``suffix`` in one segmented role.

    :param role: segmented object role that owns the setting.
    :param suffix: role-relative setting suffix such as ``channel``.
    :returns: The canonical ``<role>_<suffix>`` setting key.
    :raises ValueError: if ``role`` is not segmented.
    """
    role = str(role)
    if role not in SEGMENTED_ROLES:
        raise ValueError(f"{role!r} is not a segmented role")
    return f"{role}_{str(suffix).lstrip('_')}"


#: ``old suffix -> new suffix`` for a rename that belongs to a ROLE FAMILY
#: rather than to one key. 705 roles carry ``_flow_threshold``; writing the
#: old names out would be 3,522 entries nobody can read and six families
#: nobody can keep in step. The rename belongs to the SUFFIX, so it is
#: recorded once here and resolved per role.
#:
#: WHICH ROLES A RULE APPLIES TO IS NOT WRITTEN DOWN, and deliberately --
#: see :func:`spacr.settings.surviving_setting_name`, which applies a rule
#: only when the old key is NOT a live setting and the new one IS. A written
#: scope is a second list to keep in step with ``expected_types``, and the
#: measurement says a wrong one is expensive: seven ``_size`` keys are still
#: live (``cell_min_size``, ``cell_max_size``, ``nucleus_min_size``,
#: ``nucleus_max_size``, ``pathogen_min_size``, ``pathogen_max_size``,
#: ``cytoplasm_min_size``) while no organelle one is. A blanket
#: ``_size -> _area`` would retire all seven, and they do not even mean the
#: same thing -- ``cell_min_size`` filters at MEASUREMENT time and
#: ``cell_min_area`` at SEGMENTATION time, as `cell_min_size`'s own tooltip
#: says. Deriving the scope cannot make that mistake.
RENAMED_SETTING_SUFFIXES: Dict[str, str] = {
    "FT": "flow_threshold",
    "CP_prob": "cellprob_threshold",
    "Signal_to_noise": "signal_to_noise",
    "min_size": "min_area",
    "max_size": "max_area",
}


#: ``suffix -> why it went``, for settings WITHDRAWN from a whole role family.
#:
#: THE PARALLEL OF `RENAMED_SETTING_SUFFIXES`, AND IT EXISTS FOR THE SAME
#: REASON: 705 roles carry each of these, so a literal table would name the
#: first spelling and go silent for every generated organelle slot beside it.
#:
#: A WITHDRAWAL NEEDS A MESSAGE MORE THAN A RENAME DOES, not less. A renamed
#: key has somewhere to send its value; a withdrawn one does not, so the only
#: thing standing between the user and a silently ignored setting is being
#: told. The rule is explicit: do NOT silently drop an unrecognised
#: key -- a settings file that quietly loses a value the user set is worse
#: than one that refuses to load."
#:
#: Withdrawn merge/split controls and their older aliases stay here so old
#: files receive an explanation without migrating values onto the mean
#: intensity filters, which make a different decision about each object.
WITHDRAWN_SETTING_SUFFIXES: Dict[str, str] = {
    "minimum_area_to_split": "post-segmentation watershed splitting was removed",
    "min_watershed_distance": "post-segmentation watershed splitting was removed",
    "intensity_split": "post-segmentation watershed splitting was removed",
    "min_object_area": "post-segmentation watershed splitting was removed",
    "min_split_area": "post-segmentation watershed splitting was removed",
    "min_distance": "post-segmentation watershed splitting was removed",
    "intensity_threshold": "merging labels by boundary intensity was removed",
    "intensity_merge": "merging labels by boundary intensity was removed",
    "area_multiplier": (
        "post-segmentation watershed splitting was removed"),
    "intensity_threshold_method": (
        "merging labels by boundary intensity was removed"),
    "intensity_percentile": (
        "merging labels by boundary intensity was removed"),
    "min_intensity_percentile": (
        "the intensity band was removed: it dropped its share of objects "
        "however bright the field, which is a quota rather than a filter"),
    "max_intensity_percentile": (
        "the intensity band was removed: it dropped its share of objects "
        "however bright the field, which is a quota rather than a filter"),
}


def withdrawn_setting_reason(key: str):
    """Why ``key`` is no longer read, or ``None`` if it is not withdrawn.

    :param key: the key a settings file carries.
    :returns: a sentence naming what replaced it, or ``None``.
    """
    parts = split_role_setting(key)
    if parts is None:
        return None
    role, suffix = parts
    reason = WITHDRAWN_SETTING_SUFFIXES.get(suffix)
    return None if reason is None else reason.replace("<role>", role)


def split_role_setting(key: str):
    """``organellezz_min_area`` -> ``("organellezz", "min_area")``.

    The inverse of :func:`role_setting`, and the reason a suffix rename can
    cost six lines instead of 3,522. Every role is a single word with no
    underscore, so partitioning on the FIRST one separates the role from the
    suffix without scanning the 702 organelle slots.

    NOT :func:`role_setting`'s validation: this accepts any role in
    ``ALL_ROLES``, including ``cytoplasm``, which is derived rather than
    segmented and still declares ``cytoplasm_min_size``. Restricting to
    segmented roles here would silently skip it.

    A BARE KEY IS NOT A ROLE KEY, and that matters: ``FT``, ``CP_prob``,
    ``Signal_to_noise`` and ``flow_threshold`` are all LIVE settings in their
    own right in the standalone apply/test-model submodules. A key with no
    underscore has no role, so it can never match a suffix rule.

    :param key: a settings key.
    :returns: ``(role, suffix)``, or ``None`` when ``key`` names no role.
    """
    role, separator, suffix = str(key).partition("_")
    if not separator or not suffix or role not in ALL_ROLES:
        return None
    return role, suffix


def enabled_organelle_roles(settings: Mapping[str, Any]) -> Tuple[str, ...]:
    """Organelle slots whose ``<role>_channel`` is enabled, in plane order.

    :param settings: settings mapping carrying per-role channel assignments.
    :returns: Enabled organelle slots in schema and mask-plane order.
    """
    return tuple(role for role in ORGANELLE_ROLES
                 if settings.get(role_setting(role, "channel")) is not None)


def organelle_settings_view(settings: Mapping[str, Any], role: str) -> Dict[str, Any]:
    """Return a copy exposing one slot through the legacy ``organelle_*`` API.

    :param settings: complete settings mapping to adapt without mutating it.
    :param role: organelle slot to expose under legacy key names.
    :returns: A copied mapping exposing the selected slot through legacy
        ``organelle_*`` keys.
    :raises ValueError: if ``role`` is not an organelle slot.

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
    :returns: ``roles`` unchanged as a tuple.
    :raises ValueError: if a name is not in :data:`ALL_ROLES`, naming it and
        listing the valid kinds.
    """
    unknown = [role for role in roles if role not in ALL_ROLES]
    if unknown:
        raise ValueError(
            f"unknown object role(s) {unknown}; expected some ordering of "
            f"{list(ALL_ROLES)}")
    return tuple(roles)


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
    """True when ``table`` holds one row per cell and needs no roll-up.

    :param table: object-table name to classify.
    :returns: Whether the table needs no child-to-cell roll-up.
    """
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
