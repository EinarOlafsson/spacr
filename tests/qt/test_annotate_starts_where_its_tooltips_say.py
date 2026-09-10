"""One table, or at least a shrinking count of places where there are two.

Instruction 364's Annotate audit
(``instructions/data/364_annotate_and_map_barcodes_settings_audit_2026-09-02.txt``)
found that Annotate answers "what is the default" twice, with two different
answers. ``set_annotate_default_settings`` in ``spacr/settings.py`` is what
the tooltips describe, what ``picture_defaults()`` seeds from, and what a
notebook or ``spacr-run`` gets. The Annotate SCREEN never consults it --
``spacr/qt/screens/annotate.py`` builds a bare ``AnnotateSettings()`` -- so
the dataclass field defaults are what a person annotating actually starts
from. The two tables shared 21 keys and disagreed on 15 of them.

Three of those disagreements were not preferences, which is why they are
asserted here individually rather than counted:

``outline_sigma`` and ``outline_threshold_factor``
    The tooltips end "Default 4." and "Default 1.25.". The factory ships
    exactly that. The dataclass shipped 1.0 and 1.0, and those two numbers
    are the whole shape of the outline an annotator draws -- so the help
    text named numbers that the only surface anyone draws on did not use.

``crop_source``
    Instructions 170 and 171 decided LOAD IMAGES by default, in answer to
    "in the annotation app how do i choose to stream images from database or
    dataset" -- for which the answer was that you could not. The fix landed
    in the factory as ``'png'``; the dataclass stayed ``'auto'``, so it
    landed everywhere except the screen. Nothing about a real dataset
    changes -- ``resolve_crop_source`` sends both values to the PNG folder
    when there is one and to ``merged/`` when there is not, and only records
    a different ``reason`` -- but the default is now the one that was chosen.

The rest are pinned as a budget rather than fixed, because which side wins
is a per-row judgement the audit deliberately leaves to the maintainer:
``annotation_column`` is ``'test'`` in the factory and ``'annotate'`` in the
dataclass, and 'test' looks like a placeholder while 'annotate' looks like
the intended column name. Guessing at that in a test would bake in the
guess. So this file measures the disagreement instead: it may shrink, and a
NEW key falling out of step fails the day it is added.
"""
from __future__ import annotations

from dataclasses import asdict

import pytest


#: Keys whose two spellings mean the same thing. These are noise in the
#: comparison, not divergence: a comma string and a list of the same three
#: letters, an empty string and ``None``, and ``src``, which the factory
#: fills in from the path handed to it and the dataclass leaves blank.
SAME_THING_SPELT_TWO_WAYS = {
    "channels",      # 'r,g,b'  vs  ['r', 'g', 'b']
    "measurement",   # ''       vs  None
    "threshold",     # ''       vs  None
    "src",           # the caller's path vs the empty placeholder
}

#: Keys that still answer differently, WITH the value each side gives, as of
#: 2026-09-02. This set may only shrink. Resolving one is a two-line change
#: here and a line in the audit saying which side won and why.
STILL_TWO_ANSWERS = {
    "annotation_column",     # 'test'        vs 'annotate'
    "edge_image",            # 'False' (str) vs False (bool)
    "edge_thickness",        # 0.1           vs 1.0
    "image_type",            # 'cell_png'    vs None
    "normalize_channels",    # None (off)    vs all three planes
    "percentiles",           # [2, 98]       vs (1.0, 99.0)
    "threshold_direction",   # 'higher'      vs None
}


def _the_two_tables():
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.settings import set_annotate_default_settings

    dataclass = asdict(AnnotateSettings())
    factory = set_annotate_default_settings({"src": "/tmp/not-a-real-run"})
    return dataclass, factory


def _keys_that_disagree():
    dataclass, factory = _the_two_tables()
    shared = set(dataclass) & set(factory)
    return {
        key for key in shared - SAME_THING_SPELT_TWO_WAYS
        if dataclass[key] != factory[key]
    }


@pytest.mark.parametrize("setting, promised", [
    ("outline_sigma", 4.0),
    ("outline_threshold_factor", 1.25),
])
def test_the_outline_starts_at_the_number_the_tooltip_names(setting, promised):
    """The help text ends "Default 4." -- so the screen must open at 4."""
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.settings import descriptions

    started_at = getattr(AnnotateSettings(), setting)
    assert started_at == promised, (
        f"{setting} opens at {started_at!r}, but its tooltip promises "
        f"{promised!r}. The screen builds a bare AnnotateSettings(), so the "
        f"field default IS what an annotator starts from."
    )

    tooltip = descriptions.get(setting, "")
    if tooltip:
        assert f"Default {promised:g}." in tooltip, (
            f"{setting}'s tooltip no longer says 'Default {promised:g}.'. If "
            f"the documented default changed, change it in BOTH tables and "
            f"here -- that divergence is the whole reason this test exists."
        )


def test_annotate_opens_on_load_images_the_way_170_decided():
    """``crop_source`` is the decision that landed everywhere but the screen."""
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.settings import set_annotate_default_settings

    screen_starts_at = AnnotateSettings().crop_source
    factory_ships = set_annotate_default_settings(
        {"src": "/tmp/not-a-real-run"})["crop_source"]

    assert screen_starts_at == "png", (
        f"Annotate opens on crop_source={screen_starts_at!r}. Instructions "
        f"170 and 171 chose LOAD IMAGES ('png'), and the factory ships "
        f"{factory_ships!r}; the screen never reads the factory, so this "
        f"field is the only place that choice can be made for a person "
        f"annotating."
    )
    assert screen_starts_at == factory_ships, (
        "the factory and the screen have drifted apart on crop_source again"
    )


def test_the_number_of_settings_with_two_answers_only_goes_down():
    disagree = _keys_that_disagree()

    appeared = disagree - STILL_TWO_ANSWERS
    assert not appeared, (
        f"new settings answer 'what is the default' twice: "
        f"{sorted(appeared)}. The Annotate screen builds a bare "
        f"AnnotateSettings() and never calls set_annotate_default_settings, "
        f"so a field added to one table and not the other silently gives "
        f"headless runs and the GUI different defaults. Add it to both, or "
        f"add it here with the two values and a reason."
    )

    fixed = STILL_TWO_ANSWERS - disagree
    assert not fixed, (
        f"these settings now agree: {sorted(fixed)}. Good -- remove them "
        f"from STILL_TWO_ANSWERS so the budget keeps ratcheting, and record "
        f"in the 364 audit which side won."
    )
