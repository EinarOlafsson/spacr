"""One name for loading, one for streaming, and a fallback that says so.

Instruction 171: "unify the terminology stream images and load images. the
default should always be loade images which loades from data folder. if that
fails it should always try the other."

The condition on the fallback is 170's original objection, kept: a fallback
nobody can see is what makes a user believe they are looking at a crop they
are not. So it falls back AND the reason says which route drew.
"""
import os

import numpy as np
import pytest

from spacr.crops import (LOAD_IMAGES, LOAD_IMAGES_LABEL, PICTURE_SOURCES,
                         STREAM_IMAGES, STREAM_IMAGES_LABEL, CropError,
                         picture_source_label, resolve_crop_source)


def _screen(tmp_path, name, *, png=False, merged=False):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    if png:
        crops = root / "data" / "w" / "cell_png"
        crops.mkdir(parents=True)
        (crops / "a.png").write_bytes(b"x")
    if merged:
        (root / "merged").mkdir(parents=True)
        np.save(root / "merged" / "a.npy", np.zeros((4, 4, 3)))
    return str(root)


def test_the_stored_values_did_not_change():
    """These are LABELS. A settings file already on disk must not move."""
    assert LOAD_IMAGES == "png"
    assert STREAM_IMAGES == "merged"


def test_load_images_is_offered_first():
    assert PICTURE_SOURCES[0] == (LOAD_IMAGES, LOAD_IMAGES_LABEL)
    assert picture_source_label("png") == "load images"
    assert picture_source_label("merged") == "stream images"


def test_each_mode_is_honoured_when_both_folders_are_there(tmp_path):
    root = _screen(tmp_path, "both", png=True, merged=True)

    assert resolve_crop_source({"src": root, "crop_source": LOAD_IMAGES}).kind == "png"
    assert resolve_crop_source({"src": root, "crop_source": STREAM_IMAGES}).kind == "merged"


def test_load_images_with_no_data_folder_streams_instead(tmp_path):
    root = _screen(tmp_path, "onlymerged", merged=True)

    source = resolve_crop_source({"src": root, "crop_source": LOAD_IMAGES})

    assert source.kind == "merged"
    assert LOAD_IMAGES_LABEL in source.reason, "say what was asked for"
    assert STREAM_IMAGES_LABEL in source.reason, "and what actually drew"


def test_stream_images_with_no_merged_folder_loads_instead(tmp_path):
    root = _screen(tmp_path, "onlypng", png=True)

    source = resolve_crop_source({"src": root, "crop_source": STREAM_IMAGES})

    assert source.kind == "png"
    assert STREAM_IMAGES_LABEL in source.reason
    assert LOAD_IMAGES_LABEL in source.reason


def test_an_explicit_load_no_longer_returns_a_source_that_cannot_read(tmp_path):
    """`crop_source='png'` used to return a PngCropSource without asking
    whether `data/` existed, so the failure surfaced later with less
    context."""
    root = _screen(tmp_path, "onlymerged2", merged=True)

    source = resolve_crop_source({"src": root, "crop_source": LOAD_IMAGES})

    assert source.kind == "merged", "it handed back a source it could not read"


def test_neither_folder_is_refused_naming_both(tmp_path):
    root = _screen(tmp_path, "neither")

    with pytest.raises(CropError) as raised:
        resolve_crop_source({"src": root, "crop_source": LOAD_IMAGES})

    message = str(raised.value)
    assert "data/" in message and "merged/" in message


def test_auto_still_answers_what_is_available(tmp_path):
    """'auto' is retired from the PANELS, not from the code."""
    root = _screen(tmp_path, "both2", png=True, merged=True)

    source = resolve_crop_source({"src": root, "crop_source": "auto"})

    assert source.kind == "png"
    assert LOAD_IMAGES_LABEL in source.reason


# ------------------------------------------------ the annotation app's choice


def test_the_annotation_app_defaults_to_load_images():
    """"in the annotation app how do i choose to stream images from database
    or dataset" -- the answer was that you did not: it shipped 'auto', which
    takes the PNG folder whenever one exists, and the choice was never
    offered."""
    from spacr.settings import set_annotate_default_settings

    assert set_annotate_default_settings({})["crop_source"] == LOAD_IMAGES


def test_the_panel_offers_the_two_modes_and_not_auto():
    """'auto' answers what is AVAILABLE, which is not an answer to somebody
    asked which mode they want."""
    import spacr.qt.screens.settings_model as model

    offered = None
    for name in dir(model):
        value = getattr(model, name)
        if isinstance(value, dict) and isinstance(value.get("annotate"), dict):
            offered = value["annotate"].get("crop_source")
            break

    assert offered == [LOAD_IMAGES, STREAM_IMAGES]
    assert "auto" not in (offered or [])


def test_auto_is_still_read_by_the_code():
    """Retired from the panels, not from the code."""
    import inspect

    from spacr import crops

    assert "auto" in inspect.getsource(crops.resolve_crop_source)


def test_the_tooltip_names_both_modes():
    from spacr.settings import tooltips

    text = tooltips["crop_source"]
    assert "LOAD IMAGES" in text and "STREAM IMAGES" in text
    assert "pre_generated" in text, (
        "the training vocabulary still exists and a reader meeting it needs "
        "to be told it is the same setting")
