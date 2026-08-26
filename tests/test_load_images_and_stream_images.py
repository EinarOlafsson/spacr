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
    """Two names for the two things, and streaming now says which route.

    "stream images" became "stream images (array)" when the database route
    joined it: both stream from merged/*.npy and they differ in how the
    object is found, so a label that named neither could not tell the user
    which one they had picked.
    """
    assert PICTURE_SOURCES[0] == (LOAD_IMAGES, LOAD_IMAGES_LABEL)
    assert picture_source_label("png") == "load images"
    assert picture_source_label("merged").startswith("stream images")
    assert "array" in picture_source_label("merged")


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

    # AN OPTION MAY NOW BE (value, label) -- instruction 171 wanted the two
    # modes offered in the WORDS "load images" and "stream images" while the
    # stored values stay 'png' and 'merged', so no settings file written
    # before that changed meaning. What this test is about is which VALUES
    # are offered and that 'auto' is not one of them.
    stored = [o[0] if isinstance(o, tuple) else o for o in (offered or [])]

    assert stored == [LOAD_IMAGES, STREAM_IMAGES]
    assert "auto" not in stored
    # And the labels really are the words, where labels are given.
    labels = " ".join(o[1] for o in (offered or []) if isinstance(o, tuple))
    if labels:
        assert "load images" in labels.lower()
        assert "stream images" in labels.lower()


def test_auto_is_still_read_by_the_code():
    """Retired from the panels, not from the code."""
    import inspect

    from spacr import crops

    assert "auto" in inspect.getsource(crops.resolve_crop_source)


def test_the_tooltip_names_both_modes():
    from spacr.settings import tooltips

    text = tooltips["crop_source"]
    assert "LOAD IMAGES" in text and "STREAM IMAGES" in text
    assert "'png'" in text and "'merged'" in text


def test_the_tooltip_does_not_teach_the_retired_training_vocabulary():
    """This assertion is the reverse of the one it replaces.

    The earlier test required 'pre_generated' in the tooltip, on the grounds
    that a reader meeting the training vocabulary had to be told it was the
    same setting. That is the failure this instruction names: a mapping in
    one settings tooltip sets two vocabularies side by side, it does not
    migrate one onto the other. Training writes 'load_images' /
    'stream_images' through `image_source`, and the old spellings survive as
    ALIASES with their migration written down where a grep lands --
    `crop_source.CROP_SOURCE_ALIASES` and `settings._IMAGE_SOURCES` -- so the
    tooltip describes the choice in the two words and nothing else.
    """
    from spacr.crop_source import CROP_SOURCE_ALIASES
    from spacr.settings import _IMAGE_SOURCES, tooltips

    text = tooltips["crop_source"]
    for retired in ("pre_generated", "on_demand"):
        assert retired not in text, retired
        # Retired from the wording, NOT from the readers: every settings CSV
        # in existence carries one of them.
        assert retired in CROP_SOURCE_ALIASES
        assert retired in _IMAGE_SOURCES

    assert "load_images" in text and "stream_images" in text, (
        "the two names training writes are what the tooltip migrates onto")


# --------------------------------- the annotation app's PANEL, not its table


def test_the_annotate_settings_window_offers_the_two_modes(qtbot):
    """A table nothing reads is not a control.

    `_APP_COMBO_OPTIONS['annotate']['crop_source']` holds both modes, but the
    generic settings model builds no widget for the annotate app at all --
    it is an interactive screen whose settings live in its own dialog. So the
    words had to reach THAT dialog for "how do i choose to stream images from
    database or dataset" to be answerable in the panel.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    dialog = _SettingsDialog(AnnotateSettings())
    qtbot.addWidget(dialog)
    combo = dialog._crop_source

    offered = [combo.itemData(i) for i in range(combo.count())]
    labels = " ".join(combo.itemText(i) for i in range(combo.count())).lower()
    assert offered == [LOAD_IMAGES, STREAM_IMAGES]
    assert "load images" in labels and "stream images" in labels
    assert "auto" not in offered


def test_the_annotate_settings_window_opens_on_load_images(qtbot):
    """LOAD IMAGES is the default, and a stored 'auto' still selects it."""
    pytest.importorskip("PySide6")
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    for stored in (None, "", "auto", "png", "pre_generated"):
        settings = AnnotateSettings()
        if stored is not None:
            settings.crop_source = stored
        dialog = _SettingsDialog(settings)
        qtbot.addWidget(dialog)
        assert dialog._crop_source.currentData() == LOAD_IMAGES, stored


def test_the_annotate_window_writes_the_mode_back(qtbot):
    """Stored values stay 'png' and 'merged'."""
    pytest.importorskip("PySide6")
    from spacr.qt.annotate_engine import AnnotateSettings
    from spacr.qt.screens.annotate import _SettingsDialog

    settings = AnnotateSettings()
    dialog = _SettingsDialog(settings)
    qtbot.addWidget(dialog)
    combo = dialog._crop_source
    combo.setCurrentIndex(
        [combo.itemData(i) for i in range(combo.count())].index(STREAM_IMAGES))

    assert dialog.collect().crop_source == STREAM_IMAGES
