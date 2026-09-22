"""Which picture settings apply to which mode, and why the others do not.

THE KEYS ARE THE ANNOTATOR'S OWN, and that is the point of this module rather
than a second list. `spacr.settings.set_annotate_default_settings` already
names every one of them; a Cells tab with its own vocabulary for the same
picture would be two panels that disagree about what "normalize" means.

WHY A TABLE AND NOT A BRANCH IN THE WIDGET. The greying rule has to hold
wherever the settings are read -- a panel, a settings CSV, a macro -- and a
rule that lives only in the widget that greys it is a rule with one entry
point unguarded. That is what `ml._require_backend` says about backends and
what the volcano's adjusted axis needed at the API as well as in its menu.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .crops import (DEFAULT_PERCENTILES, DEFAULT_PNG_CHANNEL_MAPPING,
                    LOAD_IMAGES,
                    LOAD_IMAGES_LABEL, STREAM_FROM_DB,
                    STREAM_FROM_DB_LABEL, STREAM_IMAGES,
                    STREAM_IMAGES_LABEL, STREAMING_SOURCES,
                    percentile_pair)

#: Settings that shape the picture AFTER it has been obtained, so they mean
#: the same thing whichever route produced it.
BOTH_MODES: Tuple[str, ...] = (
    "crop_shape",
    "crop_size",
    "normalize_channels",
    "percentiles",
    "outline",
    "outline_threshold_factor",
    "outline_sigma",
    "edge_thickness",
    "edge_transparency",
    "edge_image",
    "object_size",
    "channels",
    "show_all_in_well",
    "crop_source",
    "half_widths",
    "baseline",
    "score_column",
    "cap",
    "cell_picking",
    "picking_threshold",
)

#: Defaults for the keys that are this panel's own rather than the
#: annotator's. Everything else comes from `set_annotate_default_settings`.
OWN_DEFAULTS: Dict[str, object] = {
    "show_all_in_well": True,
    "object_type": "cell",
    "crop_source": LOAD_IMAGES,
    "object_array": "",
    "red_channel": DEFAULT_PNG_CHANNEL_MAPPING.get("r", 2),
    "green_channel": DEFAULT_PNG_CHANNEL_MAPPING.get("g", 1),
    "blue_channel": DEFAULT_PNG_CHANNEL_MAPPING.get("b", 0),
    "half_widths": 1.0,
    "baseline": "screen_median",
    "score_column": "pred",
    "cap": 2000,
    "cell_picking": "rank",
    "picking_threshold": 0.55,
    "crop_shape": "object",
    "normalize_channels": "",
    "outline": "",
    "edge_image": False,
}

#: Settings that only mean something when the crops are read off disk.
LOAD_ONLY: Dict[str, str] = {
    "image_type": (
        "names which exported crop folder to read (cell_png, nucleus_png, "
        "...), and nothing is being read off disk when the images are "
        "streamed"),
}

#: Settings that only mean something for one way of picking cells.
PICKING_ONLY: Dict[str, Tuple[str, str]] = {
    "picking_threshold": (
        "attributed",
        "is the probability a cell must reach to be called, and the other "
        "pickers do not compute one"),
}

#: Settings that only mean something when the crops are cut on demand.
STREAM_ONLY: Dict[str, str] = {
    "object_array": (
        "names the labelled plane the object number is read from, and only "
        "the array route reads one"),
    "red_channel": (
        "chooses the array plane drawn in red, and a crop already on disk "
        "was made from a choice taken when it was written"),
    "green_channel": (
        "chooses the array plane drawn in green, and a crop already on "
        "disk was made from a choice taken when it was written"),
    "blue_channel": (
        "chooses the array plane drawn in blue, and a crop already on "
        "disk was made from a choice taken when it was written"),
}

#: Why a setting is silent in a mode that is not simply "the other mode".
#:
#: `crop_shape` applies in two of the three modes, so neither LOAD_ONLY nor
#: STREAM_ONLY can carry its reason, and a greyed control with no sentence
#: is the thing INVARIANTS 6 exists to prevent.
DERIVED_REASON: Dict[str, str] = {
    "crop_shape": (
        "chooses between an object-shaped cut and a bounding box, and the "
        "database route locates by coordinates -- there is no outline in a "
        "table to follow"),
}

#: Settings only the database route uses.
#:
#: It is the one route that has to be TOLD which object it is cutting: the
#: disk route is given it by the folder it reads, and the array route by
#: the labelled plane it reads the labels out of. Here the object type is
#: what names the coordinate columns, through
#: `spacr.stream_dataset.coordinate_column` -- which is also why the panel
#: does not ask for the columns as well.
DATABASE_ONLY: Dict[str, str] = {
    "object_type": (
        "names the object-table columns the database route reads its "
        "coordinates from, and the other two routes are given the object "
        "directly -- by the folder they read, or by the labelled plane"),
}

#: Every key this module has an opinion about, in the order a panel shows them.
ALL_KEYS: Tuple[str, ...] = (
    ("image_type",) + BOTH_MODES + tuple(DATABASE_ONLY) + tuple(STREAM_ONLY)
)


#: The tabs the picture-settings window is divided into, and which keys sit
#: on each: ``(title, keys)``, in the order they are shown.
#:
#: ONE LONG FORM IS NOT A PANEL. Twenty-eight controls in a single column ask
#: the reader to scroll past every question they are not asking to reach the
#: one they are, and the module screens already answer that with categories
#: (`spacr.qt.screens.settings_model._APP_CATEGORY_SPECS`) -- the same shape
#: is used here so the two panels are read the same way.
#:
#: THE GROUPING IS BY THE QUESTION EACH SETTING ANSWERS, not by which mode
#: uses it: a tab that appeared and vanished with the crop source would hide
#: the very controls whose greyed reason explains the mode, and greyed-never-
#: hidden is this panel's rule.
CATEGORY_SPEC: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Source", ("crop_source", "image_type", "object_type", "object_array",
                "crop_shape")),
    ("Channels", ("channels", "red_channel", "green_channel",
                  "blue_channel")),
    ("Picture", ("crop_size", "object_size", "normalize_channels",
                 "percentiles")),
    ("Outline", ("outline", "outline_threshold_factor", "outline_sigma",
                 "edge_thickness", "edge_transparency", "edge_image")),
    ("Which cells", ("cell_picking", "picking_threshold", "show_all_in_well",
                     "score_column", "baseline", "half_widths", "cap")),
)

#: Where a key lands when :data:`CATEGORY_SPEC` has not been told about it.
#:
#: A SETTING ADDED LATER MUST NOT VANISH. A panel built strictly from the
#: table would silently drop a key that reached `ALL_KEYS` without reaching
#: here, and a control that is not on any tab is a control the user cannot
#: reach -- the same failure as hiding one, arrived at by omission.
UNGROUPED_TITLE = "Other"


def categories() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """``(title, keys)`` for each tab, covering every key in :data:`ALL_KEYS`.

    Keys that :data:`CATEGORY_SPEC` names but :data:`ALL_KEYS` does not are
    dropped -- a retired setting must not leave an empty row behind -- and
    keys `ALL_KEYS` has that the spec does not are gathered onto a trailing
    :data:`UNGROUPED_TITLE` tab, which exists only when it has something on
    it.
    """
    known = set(ALL_KEYS)
    placed = []
    out = []
    for title, keys in CATEGORY_SPEC:
        kept = tuple(key for key in keys if key in known and key not in placed)
        placed.extend(kept)
        if kept:
            out.append((title, kept))
    left = tuple(key for key in ALL_KEYS if key not in placed)
    if left:
        out.append((UNGROUPED_TITLE, left))
    return tuple(out)


def category_of(key: str) -> str:
    """The tab ``key`` is shown on, or ``""`` when it is not offered here.

    :param key: picture-setting key to locate in the category specification.
    """
    name = str(key or "").strip()
    for title, keys in categories():
        if name in keys:
            return title
    return ""


def modes() -> Tuple[Tuple[str, str], ...]:
    """``(value, label)`` for the three modes, default first.

    Two of them stream from ``merged/*.npy`` and differ only in how they
    find the object -- by its label in a mask plane, or by its row in the
    measurement database. That difference decides whether the cut can
    follow an outline, so it is a choice the user makes here rather than
    something inferred from which other settings happen to be filled.
    """
    return ((LOAD_IMAGES, LOAD_IMAGES_LABEL),
            (STREAM_IMAGES, STREAM_IMAGES_LABEL),
            (STREAM_FROM_DB, STREAM_FROM_DB_LABEL))


def applies_to_picking(key: str, picking: str) -> bool:
    """Whether ``key`` means anything for the chosen way of picking cells.

    :param key: picture-setting key whose dependency is being checked.
    :param picking: selected cell-picking strategy.
    """
    entry = PICKING_ONLY.get(str(key or "").strip())
    if entry is None:
        return True
    return str(picking or "rank").strip().lower() == entry[0]


def applies_to(key: str, mode: str) -> bool:
    """Whether ``key`` means anything in ``mode``.

    :param key: picture-setting key whose source-mode applicability is tested.
    :param mode: selected crop-source mode.

    A key this module has never heard of applies: it is not this module's job
    to grey out a setting it does not know, and a panel that hid the unknown
    would hide new settings by default.
    """
    name = str(key or "").strip()
    chosen = str(mode or LOAD_IMAGES).strip().lower()
    streaming = chosen in STREAMING_SOURCES
    if name in LOAD_ONLY:
        return not streaming
    if name == "object_array":
        return chosen == STREAM_IMAGES
    if name in DATABASE_ONLY:
        return chosen == STREAM_FROM_DB
    if name == "crop_shape":
        return chosen != STREAM_FROM_DB
    if name in STREAM_ONLY:
        return streaming
    return True


def why_not(key: str, mode: str) -> str:
    """The sentence a greyed control carries, or ``""`` when it applies.

    :param key: picture-setting key whose inapplicability is explained.
    :param mode: selected crop-source mode.

    GREYED, NEVER HIDDEN (INVARIANTS 6). A control that vanishes cannot tell
    the user why their mode does not offer it.
    """
    if applies_to(key, mode):
        return ""
    name = str(key or "").strip()
    chosen = str(mode or LOAD_IMAGES).strip().lower()
    label = LOAD_IMAGES_LABEL if chosen != STREAM_IMAGES else STREAM_IMAGES_LABEL
    reason = (LOAD_ONLY.get(name) or STREAM_ONLY.get(name)
              or DATABASE_ONLY.get(name) or DERIVED_REASON.get(name) or "")
    return f"not used by '{label}': it {reason}" if reason else (
        f"not used by '{label}'")


def greyed_in(mode: str) -> Tuple[str, ...]:
    """The keys a panel must grey for ``mode``, in a stable order.

    :param mode: selected crop-source mode.
    """
    return tuple(key for key in ALL_KEYS if not applies_to(key, mode))


def bounding_box_only(settings) -> bool:
    """Whether the chosen cut can only be a bounding box.

    :param settings: picture settings containing the selected crop source.

    Coordinate-only sources have no object outline, so a panel should disable
    object-shaped crops before the cut rather than silently return a rectangle.
    """
    try:
        chosen = str(settings.get("crop_source") or LOAD_IMAGES).lower()
    except AttributeError:
        return False
    return chosen == STREAM_FROM_DB


#: The annotator's name for a setting -> the crop layer's name for it.
#: ONLY the settings that change how the crop is CUT. The rest of the
#: annotator's controls -- outline, edge_*, normalize_channels, percentiles,
#: object_size -- change how an obtained crop is DRAWN, and belong to the
#: renderer rather than the crop spec. They are not in this table because a
#: mapping that pretended to apply them would be worse than an absent one.
CUT_SETTINGS: Dict[str, str] = {
    "crop_size": "png_size",
    "channels": "png_dims",
}


def _as_channel_mapping(value, picture) -> Optional[Dict[str, object]]:
    """Colour letters as an explicit ``{r, g, b}`` source-channel mapping.

    ``None`` means "these are not colour letters" -- an index list, most
    often -- and the caller falls through to the index path.

    :param value: what the user chose, e.g. ``"r,g,b"`` or ``["r", "b"]``.
    :param picture: the whole settings dict, consulted for this screen's own
        ``png_channel_mapping`` before the default is used.
    """
    from .crops import DEFAULT_PNG_CHANNEL_MAPPING, PNG_COLOR_KEYS

    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip().lower() for p in value]
    else:
        return None
    parts = [p for p in parts if p]
    if not parts or not all(p in PNG_COLOR_KEYS for p in parts):
        return None

    known = (picture or {}).get("png_channel_mapping")
    base = dict(known) if isinstance(known, dict) else dict(
        DEFAULT_PNG_CHANNEL_MAPPING)
    return {key: (base.get(key) if key in parts else None)
            for key in PNG_COLOR_KEYS}


def _as_indices(value) -> Optional[list]:
    """``value`` as a list of source-channel indices, or ``None``.

    ``None`` means "these are not indices" -- colour letters, most often --
    and the caller leaves the setting to the renderer rather than guessing a
    number for it. See :func:`to_crop_settings`.
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip() for p in value]
    elif isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    else:
        return None
    parts = [p for p in parts if p]
    if not parts:
        return None
    out = []
    for part in parts:
        try:
            out.append(int(part))
        except (TypeError, ValueError):
            return None
    return out


def to_crop_settings(picture) -> Dict[str, object]:
    """The subset of a picture-settings dict the crop layer understands.

    :param picture: the annotator-named settings, as
        :class:`spacr.qt.widgets.picture_settings_dialog.PictureSettingsDialog`
        returns them.
    :returns: a mapping in the crop layer's own vocabulary, carrying only what
        this mode actually uses and only what is set.
    """
    try:
        items = dict(picture or {})
    except (TypeError, ValueError):
        return {}
    mode = str(items.get("crop_source") or LOAD_IMAGES).strip().lower()
    out: Dict[str, object] = {}
    for mine, theirs in CUT_SETTINGS.items():
        if not applies_to(mine, mode):
            continue
        value = items.get(mine)
        if value in (None, "", [], ()):
            continue
        if theirs == "png_dims":
            mapping = _as_channel_mapping(value, items)
            if mapping is not None:
                out["png_channel_mapping"] = mapping
                continue
            indices = _as_indices(value)
            if indices is None:
                continue
            value = indices
        if theirs == "png_size" and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            value = [int(value), int(value)]
        out[theirs] = value
    if applies_to("crop_shape", mode):
        shape = str(items.get("crop_shape") or "").strip().lower()
        if shape in ("bbox", "bounding_box", "box"):
            out["use_bounding_box"] = True
        elif shape == "object":
            out["use_bounding_box"] = False

    if mode in STREAMING_SOURCES:
        if mode == STREAM_FROM_DB:
            object_type = str(items.get("object_type") or "").strip().lower()
            if object_type:
                out["object_array"] = object_type
        chosen = {}
        for colour, key in (("r", "red_channel"), ("g", "green_channel"),
                            ("b", "blue_channel")):
            raw = items.get(key, "")
            text = str(raw).strip()
            if text in ("", "None", "-1"):
                chosen[colour] = None
                continue
            try:
                chosen[colour] = int(float(text))
            except (TypeError, ValueError):
                chosen[colour] = None
        if any(v is not None for v in chosen.values()):
            out["png_channel_mapping"] = chosen

        if mode == STREAM_FROM_DB:
            out["stream_method"] = "column"
        else:
            out["stream_method"] = "array"
            plane = str(items.get("object_array", "")).strip()
            if plane not in ("", "None"):
                out["object_array"] = plane
        if bounding_box_only(items):
            out["use_bounding_box"] = True
    return out



#: Settings that change how an obtained crop is DRAWN rather than how it is
#: cut. They are applied here, by the annotator's own functions, so a crop in
#: the Cells tab and the same crop in the annotation app look the same.
DRAW_SETTINGS: Tuple[str, ...] = (
    "normalize_channels", "percentiles", "channels", "outline",
    "outline_threshold_factor", "outline_sigma", "edge_thickness",
    "edge_transparency", "edge_image", "object_size",
)


#: A displayed image's channel position -> the colour the annotator's own
#: helpers name it by. `normalize_pil` and `filter_channels_pil` both key on
#: 'r'/'g'/'b' and silently skip anything else, so a user who typed 0,1,2 --
#: which is what every other channel setting in spaCR takes -- got a control
#: that accepted their input and did nothing.
#:
#: NOT the source-channel mapping. These are positions in the RGB picture
#: being drawn, which is why 0 is red here while `png_channel_mapping` may
#: put source channel 2 in red.
_POSITION_TO_COLOUR = {"0": "r", "1": "g", "2": "b"}


def _as_channel_list(value):
    """``['r','g']`` from whatever a settings field holds.

    Accepts the annotator's letters AND the index form the rest of spaCR
    uses, because a setting that quietly ignores half the spellings offered
    to it is worse than one that refuses them.
    """
    if value is None or value is False:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [p.strip().strip("'\"").lower() for p in text.split(",")
                 if p.strip().strip("'\"")]
    else:
        try:
            parts = [str(v).strip().lower() for v in value if str(v).strip()]
        except TypeError:
            return None
    return [_POSITION_TO_COLOUR.get(p, p) for p in parts] or None


def _as_pair(value, default=DEFAULT_PERCENTILES):
    """Resolve one percentile setting through the shared pair parser.

    :param value: Percentile pair in any form accepted by
        :func:`percentile_pair`.
    :param default: Pair returned when ``value`` does not provide one.
    :returns: Normalized low/high percentile pair.
    """
    return percentile_pair(value, default)


def draw_crop(array, picture):
    """``array`` drawn as the annotation application would draw it.

    :param array: an ``(H, W, 3)`` uint8 crop.
    :param picture: the annotator-named settings.
    :returns: an array of the same shape, or the input unchanged when nothing
        was asked for or the pipeline is unavailable.

    THE ANNOTATOR'S OWN FUNCTIONS DO THE WORK -- `normalize_pil`,
    `filter_channels_pil`, `outline_image` from
    :mod:`spacr.qt.annotate_engine`. A second implementation of "normalise a
    crop" is a second answer to what normalise means. Reusing the annotator's
    functions keeps identically configured images visually consistent.

    NEVER RAISES. A picture is the last thing this produces and the least
    important: losing a montage to an outline is the worst trade available.
    """
    items = dict(picture or {})
    if not any(items.get(key) for key in DRAW_SETTINGS):
        return array
    try:
        import numpy as np
        from PIL import Image

        from .qt.annotate_engine import (filter_channels_pil, normalize_pil,
                                         outline_image)
    except Exception:                                    # noqa: BLE001
        return array
    try:
        data = np.ascontiguousarray(np.asarray(array, dtype="uint8"))
        if data.ndim == 2:
            data = np.repeat(data[:, :, None], 3, axis=2)
        image = Image.fromarray(data[:, :, :3])
        full = image.copy()

        normalise = _as_channel_list(items.get("normalize_channels"))
        if normalise:
            image = normalize_pil(image, _as_pair(items.get("percentiles")),
                                  normalise)
        outline = _as_channel_list(items.get("outline"))
        if outline:
            size = items.get("object_size") or 0
            try:
                bounds = (int(size), int(size)) if not isinstance(
                    size, (list, tuple)) else (int(size[0]), int(size[1]))
            except (TypeError, ValueError):
                bounds = (0, 0)
            image = outline_image(
                image, full, outline_channels=outline,
                edge_sigma=float(items.get("outline_sigma") or 1.0),
                edge_thickness=float(items.get("edge_thickness") or 1.0),
                edge_transparency=float(items.get("edge_transparency") or 100.0),
                edge_image=bool(items.get("edge_image")),
                outline_threshold_factor=float(
                    items.get("outline_threshold_factor") or 1.0),
                object_size=bounds)
        shown = _as_channel_list(items.get("channels"))
        if shown and all(c in ("r", "g", "b") for c in shown):
            image = filter_channels_pil(image, shown)
        return np.asarray(image.convert("RGB"), dtype="uint8")
    except Exception:                                    # noqa: BLE001
        return array



def available_arrays(source) -> Tuple[str, ...]:
    """The mask planes this screen's merged arrays actually record.

    `object_array` chooses which mask the intensity channels are cut by, and
    offering it as free text asks the user to remember what their own screen
    contains -- and to spell it the way `measure` did. Every other chooser in
    spaCR is built from the data; this one was not.

    :param source: a :class:`spacr.crops.CropSource`, a
        :class:`CropSourceChoice`, or anything with a ``spec.mask_dims``.
    :returns: the plane names, in a stable order. Empty when the screen
        records none -- which is the answer, not a failure: a run whose
        merged arrays carry no mask planes cannot cut by one.
    """
    spec = getattr(getattr(source, "source", source), "spec", None)
    dims = dict(getattr(spec, "mask_dims", None) or {})
    return tuple(sorted(str(name) for name in dims))


def available_coordinate_columns(frame) -> Tuple[str, ...]:
    """The object-table columns a bounding box could be cut from.

    :param frame: measurement table whose coordinate columns are inspected.

    All four corners or none: three of them describe no box, so a chooser
    that offered them singly would let a user assemble a request that cannot
    be met.
    """
    names = getattr(frame, "columns", None)
    if names is None:
        return ()
    columns = {str(c) for c in names}
    if not columns:
        return ()
    out = []
    for spelling in (("bbox-0", "bbox-1", "bbox-2", "bbox-3"),
                     ("bbox_0", "bbox_1", "bbox_2", "bbox_3"),
                     ("min_row", "min_col", "max_row", "max_col")):
        if set(spelling) <= columns:
            out.append(", ".join(spelling))
    return tuple(out)


#: Explain how each annotation method selects cells: its inputs,
#: calculation, inclusion and exclusion rules, and principal limitation.
#:
#: The dropdown, documentation, and contextual help share this mapping so the
#: selection contract remains consistent across user-facing surfaces.
PICKING_HELP: Dict[str, str] = {
    "rank": (
        "TOP BY SCORE.\n"
        "Given: the guide's fraction in the well, and each cell's "
        "classification score.\n"
        "Computes: n = round(cells x fraction), then sorts by score -- "
        "descending for a positive coefficient, ascending for a negative "
        "one, because an inhibitory guide's cells are the LEAST consistent "
        "ones.\n"
        "Annotated: the top n. Not annotated: everyone below the cut, and "
        "any cell with no score at all.\n"
        "Wrong when: the fraction is wrong. It sets HOW MANY cells are "
        "taken, so a share inflated by normalisation reaches down into "
        "cells the score does not support. No probability is computed and "
        "none is reported."),
    "attributed": (
        "ATTRIBUTED.\n"
        "Given: every guide's fraction in the well and its regression "
        "effect, plus each cell's score.\n"
        "Computes: a posterior per cell per guide -- a likelihood from the "
        "score against each guide's expected effect, a prior from the "
        "fractions -- then iterative proportional fitting until every cell "
        "sums to 1 AND every guide holds the number of cells its reads "
        "imply.\n"
        "Annotated: the argmax, if it clears 0.55. Not annotated: anything "
        "below that, marked ambiguous, carrying the highest probability any "
        "guide reached.\n"
        "Wrong when: the effects are wrong, or a guide's effect is too "
        "small against the spread of scores to reach the threshold in any "
        "well -- then it selects nothing and the montage is empty."),
    "assigned": (
        "ASSIGNED.\n"
        "Given: the same as attributed.\n"
        "Computes: slots per guide = round(N x fraction), adjusted by "
        "largest remainders to sum to N exactly, then a Hungarian "
        "assignment minimising total cost.\n"
        "Annotated: EVERY cell -- each gets exactly one guide by "
        "construction. Not annotated: none.\n"
        "Wrong when: you need to know which cells are uncertain. It cannot "
        "abstain, so a well of pure noise is partitioned as confidently as "
        "a well of clear signal."),
    "multivariate": (
        "MULTIVARIATE.\n"
        "Given: attributed's inputs, plus one effect per MEASUREMENT per "
        "guide from the gene x measurement sweep.\n"
        "Computes: the same posterior over a vector of measurements rather "
        "than one score.\n"
        "Annotated: cells clearing 0.55. Not annotated: the rest.\n"
        "Wrong when: there is no sweep -- it falls back to attributed and "
        "SAYS so rather than substituting silently -- or when the swept "
        "measurements are correlated, which makes the effective dimension "
        "smaller than the count of columns suggests."),
    "sudoku": (
        "SUDOKU.\n"
        "Given: the cells' measurements across every well the guide "
        "appears in, plus the fractions.\n"
        "Computes: anchors from wells where a guide dominates, a "
        "nearest-neighbour graph over cells, label propagation from those "
        "anchors, then the same per-well constraint as attributed.\n"
        "Annotated: cells whose constrained posterior clears the decision "
        "bar. Not annotated: cells far from every anchor, and cells whose "
        "top two guides are too close to call.\n"
        "Wrong when: the anchors are wrong. They are chosen BY SCORE, so "
        "the score is deliberately left out of the graph -- with it in, "
        "every high-scoring cell would sit beside every guide's anchors "
        "and affirm all of them."),
}


def offered_values(key: str, source=None, frame=None) -> Tuple[str, ...]:
    """What a chooser for ``key`` should list, or ``()`` for free text.

    :param key: picture-setting key whose current choices are requested.

    ONE PLACE, so the Cells tab and the annotation app cannot offer different
    answers for the same screen.
    """
    name = str(key or "").strip()
    if name == "object_array":
        return available_arrays(source)
    if name == "crop_shape":
        return ("object", "bbox")
    if name == "crop_source":
        return modes()
    if name == "channels":
        return (
            ("r,g,b", "all three"),
            ("r", "red only"),
            ("g", "green only"),
            ("b", "blue only"),
            ("r,g", "red and green"),
            ("r,b", "red and blue"),
            ("g,b", "green and blue"),
        )
    if name in ("normalize_channels", "outline"):
        what = ("normalised" if name == "normalize_channels" else "outlined")
        return (
            ("", f"none — nothing is {what}"),
            ("r", f"red only"),
            ("g", f"green only"),
            ("b", f"blue only"),
            ("r,g", "red and green"),
            ("r,b", "red and blue"),
            ("g,b", "green and blue"),
            ("r,g,b", f"every channel"),
        )
    if name == "object_type":
        return ("cell", "nucleus", "pathogen", "cytoplasm",
                "organelle", "organelleb", "organellec", "organelled")
    if name == "baseline":
        return ("screen_median", "control_median", "zero")
    if name == "cell_picking":
        return (
            ("rank", "top by score — the count the fraction implies"),
            ("attributed", "attributed — each cell's probability, above the "
                           "threshold"),
            ("assigned", "assigned — every cell in the well gets one guide"),
            ("multivariate", "multivariate — every measurement, not just the "
                             "score (needs a sweep)"),
            ("sudoku", "sudoku — learns each guide's look from every well "
                       "it is in, then decides this one"),
        )
    return ()



#: How many crops fit on one page of a well tab, measured across the viewport
#: sizes the Cells tab is really used at.
#:
#: The page is the container's, not a setting: the column count is the user's
#: preference and the ROWS are whatever the viewport has room for. At six
#: columns and 96 px thumbnails that is 24 crops in the narrowest panel the
#: splitter allows (780x420), 36 in a comfortable one (780x700 and 1200x700
#: both), and 60 on a full 1920x1080 screen.
MEASURED_PAGE_SIZES: Tuple[int, int, int] = (24, 36, 60)

#: Bytes one crop occupies while its tab is open.
#:
#: THE TAB HOLDS EVERY CROP, not one page of them: the page decides how many
#: thumbnails are drawn, and the arrays behind them all stay so that turning
#: a page does not re-cut. 224x224x3 uint8 is the crop size the merged route
#: cuts by default.
MEASURED_CROP_BYTES = 224 * 224 * 3

#: Milliseconds to CUT one crop, by route: ``(best, worst)``.
#:
#: The merged route is priced by how many FIELDS a montage touches rather
#: than by how many crops it cuts, which is why its two numbers are four-fold
#: apart: 2.58 ms/crop over 6 fields against 11.43 ms/crop over 30. The
#: exported-PNG route is ~10x cheaper and flat in both.
MEASURED_MS_PER_CROP: Dict[str, Tuple[float, float]] = {
    "png": (0.49, 0.64),
    "merged": (2.58, 11.43),
}


def montage_cap_cost(cap) -> str:
    """Summarize the estimated cost of a montage containing ``cap`` objects.

    Estimates use the measured page capacities, crop memory, and per-crop
    timings in :data:`MEASURED_PAGE_SIZES`, :data:`MEASURED_CROP_BYTES`, and
    :data:`MEASURED_MS_PER_CROP`.

    :param cap: Maximum number of objects in one montage.
    :returns: Estimated page count, memory use, and crop-extraction time, or
        ``""`` when ``cap`` is not a positive integer.
    """
    try:
        count = int(cap)
    except (TypeError, ValueError):
        return ""
    if count <= 0:
        return ""
    widest, typical, narrowest = (max(MEASURED_PAGE_SIZES),
                                  MEASURED_PAGE_SIZES[1],
                                  min(MEASURED_PAGE_SIZES))
    fewest = -(-count // widest)
    most = -(-count // narrowest)
    memory = count * MEASURED_CROP_BYTES / float(1 << 20)
    quick = count * MEASURED_MS_PER_CROP["png"][0] / 1000.0
    slow = count * MEASURED_MS_PER_CROP["merged"][1] / 1000.0
    pages = (f"{fewest} pages" if fewest == most
             else f"{fewest}-{most} pages")
    return (f"{count:,} objects is {pages} "
            f"({typical} to a page on a typical panel), about "
            f"{memory:,.0f} MB of crops held while the tab is open, and "
            f"{_seconds(quick)}-{_seconds(slow)} s to cut -- the low end "
            f"reading exported PNGs, the high end cutting from merged arrays "
            f"across many fields.")


def _seconds(value: float) -> str:
    """A duration rounded to something a reader can act on.

    Under ten seconds a whole number rounds a real wait down to zero, which
    reads as free; over it the decimal is noise.
    """
    return f"{value:,.1f}" if value < 10 else f"{value:,.0f}"


#: Removed picture settings and the migration note reported for each one.
RETIRED: dict = {
    "cells_per_page": (
        "removed: the page size is now a consequence of the container size "
        "and the image size, so a configured count could only contradict "
        "the geometry -- producing a half-empty page or a clipped row, with "
        "no way for the user to tell which"),
}


def drop_retired(picture) -> tuple:
    """``(settings, [note])`` with the retired keys taken out.

    :param picture: saved picture-settings mapping to migrate.

    Called wherever a saved picture-settings blob is read. The notes are
    returned rather than printed, so the caller decides whether this is
    worth a line -- it is worth one the first time and noise every time
    after.

    A RENAMED key is moved rather than dropped: ``img_size`` became
    ``crop_size`` on 2026-09-19, and a saved run that chose a size keeps it.
    The rename is read from :func:`spacr.settings.surviving_setting_name`,
    the resolver the settings files use, and only onto a key this panel
    offers. The new name wins when a blob carries both.
    """
    out = dict(picture or {})
    notes = []
    for key, why in RETIRED.items():
        if key in out:
            out.pop(key)
            notes.append(f"{key}: {why}")
    offered = set(ALL_KEYS)
    try:
        from .settings import surviving_setting_name
    except Exception:                                        # noqa: BLE001
        return out, notes
    for key in [k for k in out if isinstance(k, str) and k not in offered]:
        try:
            survivors = surviving_setting_name(key)
        except Exception:                                    # noqa: BLE001
            survivors = ()
        if len(survivors) != 1 or survivors[0] not in offered:
            continue
        new = survivors[0]
        value = out.pop(key)
        if new in out:
            notes.append(f"{key}: renamed to {new}, which this blob also "
                         f"sets, so {key}={value!r} was not used")
            continue
        out[new] = value
        notes.append(f"{key}: renamed to {new}, and {value!r} moved across")
    return out, notes
