"""Which picture settings apply to which mode, and why the others do not.

Instruction 170 asks the Cells tab for "a settings button that spawns a
settings window like annotation aplication and gives the user the same
controll over how to show the images", with "settings that do not apply for
the chosen method are grayed out".

THE KEYS ARE THE ANNOTATOR'S OWN, and that is the point of this module rather
than a second list. `spacr.settings.set_annotate_default_settings` already
names every one of them; a Cells tab with its own vocabulary for the same
picture would be two panels that disagree about what "normalize" means, which
is the failure instruction 145 exists to stop.

WHY A TABLE AND NOT A BRANCH IN THE WIDGET. The greying rule has to hold
wherever the settings are read -- a panel, a settings CSV, a macro -- and a
rule that lives only in the widget that greys it is a rule with one entry
point unguarded. That is what `ml._require_backend` says about backends and
what the volcano's adjusted axis needed at the API as well as in its menu.
"""
from __future__ import annotations

from typing import Dict, Tuple

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
)

#: Defaults for the keys that are this panel's own rather than the
#: annotator's. Everything else comes from `set_annotate_default_settings`.
OWN_DEFAULTS: Dict[str, object] = {
    "show_all_in_well": False,
    "cells_per_page": 60,
}

#: Settings that only mean something when the crops are read off disk.
LOAD_ONLY: Dict[str, str] = {
    "image_type": (
        "names which exported crop folder to read (cell_png, nucleus_png, "
        "...), and nothing is being read off disk when the images are "
        "streamed"),
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

    "this could only do bounding box" -- the maintainer, describing cutting
    from object coordinates. A panel must say so BEFORE the cut is made
    rather than quietly squaring off an object-shaped request.
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
