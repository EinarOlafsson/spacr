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

from .crops import (LOAD_IMAGES, LOAD_IMAGES_LABEL, STREAM_IMAGES,
                    STREAM_IMAGES_LABEL)

#: Settings that shape the picture AFTER it has been obtained, so they mean
#: the same thing whichever route produced it.
BOTH_MODES: Tuple[str, ...] = (
    "img_size",
    "normalize_channels",
    "percentiles",
    "outline",
    "outline_threshold_factor",
    "outline_sigma",
    "edge_thickness",
    "edge_transparency",
    "edge_image",
    "object_size",
    # Which channels are shown. It means "of the PNG" when loading and "of
    # the merged array" when streaming, which is the same question asked of
    # two sources -- not two settings.
    "channels",
    # THE WELL, OR THE CANDIDATES. Asked for 2026-08-19: "an option to show
    # all the images from each well and highlight the cells most likely to be
    # whatever gene is picked". Not the default, because the two answer
    # different questions -- the filtered view asks which cells look like the
    # effect, this asks what the whole well looks like and where they are in
    # it -- and a reader who cannot see the well cannot judge the window.
    "show_all_in_well",
    # How many cells one page of a well holds. A well capped at 300 drawn as
    # one grid is a scroll nobody reads to the end of.
    "cells_per_page",
    # THE MONTAGE'S OWN CONTROLS, moved off the toolbar 2026-08-19: "the
    # half-width baseline, score column, and max objects can be moved to the
    # settings panel", and the three combos beside them read as stray labels
    # once the row was crowded. A toolbar is for what you change constantly;
    # these are set once for a screen.
    "object_type",
    "crop_source",
    "half_widths",
    "baseline",
    "score_column",
    "cap",
    # WHICH CELLS BELONG TO THE COEFFICIENT (instructions 172 and 173).
    # 'rank' is heuristic 1 -- the top x by score. 'attributed' is each
    # cell's posterior of carrying the guide. 'assigned' is the constrained
    # assignment, where every cell in the well gets exactly one guide and each
    # guide gets exactly the cells its reads imply.
    "cell_picking",
    "picking_threshold",
)

#: Defaults for the keys that are this panel's own rather than the
#: annotator's. Everything else comes from `set_annotate_default_settings`.
OWN_DEFAULTS: Dict[str, object] = {
    "show_all_in_well": False,
    "cells_per_page": 60,
    "object_type": "cell",
    "crop_source": LOAD_IMAGES,
    "half_widths": 1.0,
    "baseline": "screen_median",
    "score_column": "pred",
    "cap": 300,
    "cell_picking": "rank",
    "picking_threshold": 0.55,
    # NAMED, not left to fall out of the dialog. Without an entry here the
    # settings dict said None while the settings WINDOW showed "object" --
    # the user reads one thing and the crop is cut by another.
    "crop_shape": "object",
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
        "chooses the mask plane the intensity channels are cut by, and a "
        "crop that was already written to disk has been cut already"),
    "coordinate_columns": (
        "names the object-table columns the crop is cut from, and a crop "
        "that was already written to disk has been cut already"),
    "crop_shape": (
        "chooses between an object-shaped cut and a bounding box, and a crop "
        "that was already written to disk has been cut already"),
}

#: Every key this module has an opinion about, in the order a panel shows them.
ALL_KEYS: Tuple[str, ...] = (
    ("image_type",) + BOTH_MODES + tuple(STREAM_ONLY)
)


def modes() -> Tuple[Tuple[str, str], ...]:
    """``(value, label)`` for the two modes, default first."""
    return ((LOAD_IMAGES, LOAD_IMAGES_LABEL),
            (STREAM_IMAGES, STREAM_IMAGES_LABEL))


def applies_to_picking(key: str, picking: str) -> bool:
    """Whether ``key`` means anything for the chosen way of picking cells."""
    entry = PICKING_ONLY.get(str(key or "").strip())
    if entry is None:
        return True
    return str(picking or "rank").strip().lower() == entry[0]


def applies_to(key: str, mode: str) -> bool:
    """Whether ``key`` means anything in ``mode``.

    A key this module has never heard of applies: it is not this module's job
    to grey out a setting it does not know, and a panel that hid the unknown
    would hide new settings by default.
    """
    name = str(key or "").strip()
    chosen = str(mode or LOAD_IMAGES).strip().lower()
    if name in LOAD_ONLY:
        return chosen != STREAM_IMAGES
    if name in STREAM_ONLY:
        return chosen == STREAM_IMAGES
    return True


def why_not(key: str, mode: str) -> str:
    """The sentence a greyed control carries, or ``""`` when it applies.

    GREYED, NEVER HIDDEN (INVARIANTS 6). A control that vanishes cannot tell
    the user why their mode does not offer it.
    """
    if applies_to(key, mode):
        return ""
    name = str(key or "").strip()
    chosen = str(mode or LOAD_IMAGES).strip().lower()
    label = LOAD_IMAGES_LABEL if chosen != STREAM_IMAGES else STREAM_IMAGES_LABEL
    reason = LOAD_ONLY.get(name) or STREAM_ONLY.get(name) or ""
    return f"not used by '{label}': it {reason}" if reason else (
        f"not used by '{label}'")


def greyed_in(mode: str) -> Tuple[str, ...]:
    """The keys a panel must grey for ``mode``, in a stable order."""
    return tuple(key for key in ALL_KEYS if not applies_to(key, mode))


def bounding_box_only(settings) -> bool:
    """Whether the chosen cut can only be a bounding box.

    Coordinate-only sources have no object outline, so a panel should disable
    object-shaped crops before the cut rather than silently return a rectangle.
    """
    try:
        chosen = str(settings.get("crop_source") or LOAD_IMAGES).lower()
        columns = settings.get("coordinate_columns")
    except AttributeError:
        return False
    return chosen == STREAM_IMAGES and bool(columns)


#: The annotator's name for a setting -> the crop layer's name for it.
#: ONLY the settings that change how the crop is CUT. The rest of the
#: annotator's controls -- outline, edge_*, normalize_channels, percentiles,
#: object_size -- change how an obtained crop is DRAWN, and belong to the
#: renderer rather than the crop spec. They are not in this table because a
#: mapping that pretended to apply them would be worse than an absent one.
CUT_SETTINGS: Dict[str, str] = {
    "img_size": "png_size",
    "channels": "png_dims",
}


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
            # THE ANNOTATOR'S `channels` IS NOT THE CROP LAYER'S `png_dims`,
            # and this mapping treated them as one thing. `channels` defaults
            # to 'r,g,b' -- which COLOUR PLANES OF AN EXISTING PNG to show,
            # a display choice the renderer makes with the annotator's own
            # `filter_channels_pil`. `png_dims` is which SOURCE ARRAY
            # CHANNELS to cut, and it must be indices.
            #
            # Handed the letters, `resolve_png_channel_mapping` reached
            # int('r') and the montage died with "invalid literal for int()
            # with base 10: 'r'" -- from inside the worker, so what the user
            # saw was "The montage load failed" with no mention of a setting.
            # This is 145 exactly: one idea, two vocabularies, and the code
            # in between assuming they agree.
            #
            # So only an INDEX form crosses over. Letters stay a display
            # setting and reach the renderer, which is where they mean
            # something.
            indices = _as_indices(value)
            if indices is None:
                continue
            value = indices
        if theirs == "png_size" and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            # `img_size` is ONE number -- a single spin box -- and `png_size`
            # is a (width, height) pair. Handed the scalar,
            # `crop_spec_from_settings` raised "'int' object is not
            # subscriptable" from inside the montage worker.
            value = [int(value), int(value)]
        out[theirs] = value
    # The SHAPE of the cut, which only streaming decides -- a crop already
    # written to disk was cut when it was written.
    if applies_to("crop_shape", mode):
        shape = str(items.get("crop_shape") or "").strip().lower()
        if shape in ("bbox", "bounding_box", "box"):
            out["use_bounding_box"] = True
        elif shape == "object":
            out["use_bounding_box"] = False
    return out


# --------------------------------------------------------------------------- #
#  Drawing a crop the way the annotator would
# --------------------------------------------------------------------------- #

#: Settings that change how an obtained crop is DRAWN rather than how it is
#: cut. They are applied here, by the annotator's own functions, so a crop in
#: the Cells tab and the same crop in the annotation app look the same.
DRAW_SETTINGS: Tuple[str, ...] = (
    "normalize_channels", "percentiles", "channels", "outline",
    "outline_threshold_factor", "outline_sigma", "edge_thickness",
    "edge_transparency", "edge_image", "object_size",
)


def _as_channel_list(value):
    """``['r','g']`` from whatever a settings field holds."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return [p.strip().strip("'\"").lower() for p in text.split(",")
                if p.strip().strip("'\"")]
    try:
        return [str(v).strip().lower() for v in value if str(v).strip()]
    except TypeError:
        return None


def _as_pair(value, default=(1.0, 99.0)):
    parts = _as_channel_list(value)
    if not parts or len(parts) < 2:
        return default
    try:
        return (float(parts[0]), float(parts[1]))
    except (TypeError, ValueError):
        return default


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
        # LAST, because zeroing a channel before the outline is computed would
        # outline a channel that is no longer there.
        shown = _as_channel_list(items.get("channels"))
        if shown and all(c in ("r", "g", "b") for c in shown):
            image = filter_channels_pil(image, shown)
        return np.asarray(image.convert("RGB"), dtype="uint8")
    except Exception:                                    # noqa: BLE001
        return array


# --------------------------------------------------------------------------- #
#  What THIS screen actually offers
# --------------------------------------------------------------------------- #

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

    All four corners or none: three of them describe no box, so a chooser
    that offered them singly would let a user assemble a request that cannot
    be met.
    """
    # `or ()` on a pandas Index raises -- an Index has no truth value. The
    # guard has to be an explicit None check.
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


def offered_values(key: str, source=None, frame=None) -> Tuple[str, ...]:
    """What a chooser for ``key`` should list, or ``()`` for free text.

    ONE PLACE, so the Cells tab and the annotation app cannot offer different
    answers for the same screen.
    """
    name = str(key or "").strip()
    if name == "object_array":
        return available_arrays(source)
    if name == "coordinate_columns":
        return available_coordinate_columns(frame)
    if name == "crop_shape":
        return ("object", "bbox")
    if name == "crop_source":
        # (value, label): the stored value never changes, and the label is
        # what the user named it (instruction 171).
        return ((LOAD_IMAGES, f"{LOAD_IMAGES_LABEL} — crops already in data/"),
                (STREAM_IMAGES, f"{STREAM_IMAGES_LABEL} — cut from merged/"))
    if name == "object_type":
        return ("cell", "nucleus", "pathogen", "cytoplasm")
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
        )
    return ()
