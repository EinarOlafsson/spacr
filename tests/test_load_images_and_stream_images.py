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


# ------------------------------- the TRAINING panel, asked in the same words

#: Training persists its own spelling of the two names. The STORED VALUES
#: differ from the viewers' on purpose -- ``spacr.settings`` normalises every
#: choice into this pair and then copies `image_source` onto `crop_source` --
#: and instruction 171's point is that the WORDS shown may not.
TRAINING_LOAD_IMAGES = "load_images"
TRAINING_STREAM_IMAGES = "stream_images"

#: Every spelling a settings CSV has ever carried for this setting, and the
#: mode it has always meant. A panel that offered any of these AS A WORD
#: would be the fourth vocabulary 171 exists to delete; a panel that refused
#: them would stop old settings files loading.
RETIRED_SPELLINGS = [
    (None, TRAINING_LOAD_IMAGES),
    ("", TRAINING_LOAD_IMAGES),
    ("auto", TRAINING_LOAD_IMAGES),
    ("png", TRAINING_LOAD_IMAGES),
    ("pre_generated", TRAINING_LOAD_IMAGES),
    ("generate", TRAINING_LOAD_IMAGES),
    ("load_images", TRAINING_LOAD_IMAGES),
    ("merged", TRAINING_STREAM_IMAGES),
    ("stream", TRAINING_STREAM_IMAGES),
    ("on_demand", TRAINING_STREAM_IMAGES),
    ("stream_images", TRAINING_STREAM_IMAGES),
]


def _training_image_source_widget(app_key, stored="__unset__"):
    """The control the Classify panel actually builds for `image_source`."""
    from spacr.qt.screens.settings_model import SettingsWidgets

    current = None if stored == "__unset__" else {"image_source": stored}
    model = SettingsWidgets(app_key, current=current)
    widget = model._widget_for("entry", None,
                               model._defaults.get("image_source"),
                               "image_source")
    return model, widget


@pytest.mark.parametrize("app_key", ["classify", "classify_merged"])
def test_the_training_panel_offers_the_two_modes_and_not_a_text_box(qtbot,
                                                                   app_key):
    """The one panel the whole migration was for had no control at all.

    `image_source` is not in `_APP_HIDDEN_KEYS`, so the Classify screens lay
    out a row for it -- and a key absent from `_APP_COMBO_OPTIONS` gets
    whatever widget its default's TYPE implies, which for a string is a
    free-text box. So the panel that 171's "training vocabulary is migrated"
    section is about was the one panel where a user could type a fourth
    spelling, and a typo or a remembered 'pre_generated' refused the run at
    the door with a CropSourceError naming words no user had seen.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QComboBox

    _model, widget = _training_image_source_widget(app_key)
    qtbot.addWidget(widget)

    assert isinstance(widget, QComboBox), (
        f"{app_key} builds a {type(widget).__name__} for image_source, so "
        f"the two names are not offered -- they are typed")
    offered = [widget.itemData(i) for i in range(widget.count())]
    labels = [widget.itemText(i).lower() for i in range(widget.count())]
    assert offered == [TRAINING_LOAD_IMAGES, TRAINING_STREAM_IMAGES]
    assert labels[0].startswith("load images"), labels
    assert labels[1].startswith("stream images"), labels
    # AND NOTHING ELSE. 'auto' answers what is available (rule E) and
    # 'generate' is an ACTION rather than a source (rule D); neither is an
    # answer to which mode the user wants.
    for retired in ("pre_generated", "on_demand", "auto", "generate"):
        assert retired not in offered, retired
        assert retired not in " ".join(labels), retired


def test_the_training_panel_says_the_same_sentences_as_the_viewers():
    """One question, one wording. The two tables differ only in what they
    STORE, which is the thing 171 promised would not move."""
    from spacr.qt.screens.settings_model import (_CROP_SOURCE_OPTIONS,
                                                 _IMAGE_SOURCE_OPTIONS)

    assert [label for _v, label in _IMAGE_SOURCE_OPTIONS] == [
        label for _v, label in _CROP_SOURCE_OPTIONS], (
        "the training panel and the viewers word the same choice differently")
    assert [v for v, _l in _CROP_SOURCE_OPTIONS] == [LOAD_IMAGES,
                                                     STREAM_IMAGES]
    assert [v for v, _l in _IMAGE_SOURCE_OPTIONS] == [TRAINING_LOAD_IMAGES,
                                                      TRAINING_STREAM_IMAGES]


@pytest.mark.parametrize("app_key", ["classify", "classify_merged"])
def test_the_training_panel_opens_on_load_images(qtbot, app_key):
    """LOAD IMAGES IS THE DEFAULT (rule A), in the panel and not only in the
    settings dict."""
    pytest.importorskip("PySide6")

    model, widget = _training_image_source_widget(app_key)
    qtbot.addWidget(widget)

    assert model._defaults["image_source"] == TRAINING_LOAD_IMAGES
    assert widget.currentData() == TRAINING_LOAD_IMAGES
    assert widget.currentIndex() == 0


@pytest.mark.parametrize("stored,selects", RETIRED_SPELLINGS)
def test_an_older_settings_file_selects_its_mode_not_a_third_item(qtbot,
                                                                  stored,
                                                                  selects):
    """ACCEPTED, NOT REFUSED -- and not shown either.

    A combo whose stored value matches no item keeps that value as a new
    first item, which is right for a free alphabet and wrong for a two-way
    question: a settings CSV carrying 'on_demand' put a THIRD entry, spelled
    in a retired vocabulary, in front of the user. Every old spelling now
    SELECTS the mode it has always meant.
    """
    pytest.importorskip("PySide6")

    model, widget = _training_image_source_widget("classify", stored)
    qtbot.addWidget(widget)

    offered = [widget.itemData(i) for i in range(widget.count())]
    assert offered == [TRAINING_LOAD_IMAGES, TRAINING_STREAM_IMAGES], (
        f"crop_source={stored!r} added a third, retired spelling to the panel")
    assert widget.currentData() == selects, stored
    # And what the panel COLLECTS is the selected mode, so saving a screen
    # that was opened on an old file writes one of the two names.
    assert model._read_widget(widget) == selects, stored


def test_every_value_the_training_panel_offers_reaches_a_source():
    """The claim that would have caught the rename, bound to the panel's own
    table rather than to a hand-written list beside it.

    `io._canonical_crop_source` passes an unrecognised value through
    UNCHANGED so `resolve_crop_source` raises and names it -- which is how
    'load_images' produced "crop_source='load_images' is not one of [...]"
    on every computer-vision run.
    """
    from spacr.crop_source import CROP_SOURCE_ALIASES
    from spacr.io import _canonical_crop_source
    from spacr.qt.screens.settings_model import _IMAGE_SOURCE_OPTIONS

    landed = {}
    for stored, _label in _IMAGE_SOURCE_OPTIONS:
        assert stored in CROP_SOURCE_ALIASES, stored
        landed[stored] = _canonical_crop_source(stored)
    assert landed == {TRAINING_LOAD_IMAGES: LOAD_IMAGES,
                      TRAINING_STREAM_IMAGES: STREAM_IMAGES}


def test_training_load_images_with_no_data_folder_streams_instead(tmp_path):
    """Rule B and rule C, through the door a training run goes through.

    The viewers' fallback is tested above against `resolve_crop_source`
    directly; this asserts the value the TRAINING panel writes reaches it.
    """
    from spacr.io import open_crop_source

    root = _screen(tmp_path, "train_onlymerged", merged=True)

    source = open_crop_source({"src": root,
                               "crop_source": TRAINING_LOAD_IMAGES},
                              verbose=False)

    assert source is not None, "the training door refused the panel's own value"
    assert source.kind == STREAM_IMAGES
    assert LOAD_IMAGES_LABEL in source.reason, "say what was asked for"
    assert STREAM_IMAGES_LABEL in source.reason, "and what actually drew"


def test_training_stream_images_with_no_merged_folder_loads_instead(tmp_path):
    """And the other direction, which is what "always try the other" means."""
    from spacr.io import open_crop_source

    root = _screen(tmp_path, "train_onlypng", png=True)

    source = open_crop_source({"src": root,
                               "crop_source": TRAINING_STREAM_IMAGES},
                              verbose=False)

    assert source is not None
    assert source.kind == LOAD_IMAGES
    assert STREAM_IMAGES_LABEL in source.reason
    assert LOAD_IMAGES_LABEL in source.reason


@pytest.mark.parametrize("app_key", ["classify", "classify_merged"])
def test_the_training_modes_drive_the_panel_and_are_not_a_label(qtbot,
                                                                app_key):
    """A control nothing reads is a label with a dropdown arrow.

    `settings.setting_dependencies` gates the streaming-only settings on
    ``image_source``, and `_rules_for_this_panel` fires a rule only when the
    setting it READS is on the panel too. So the combo is the thing that
    makes "controls that do not apply to the selected source are disabled"
    -- which the shipped `crop_source` tooltip promises -- true on the
    training panel.

    NOTHING HERE CALLS `_refresh_setting_dependencies` BY HAND, which is the
    whole assertion. Driving the refresh from the test proves only that the
    rule can read the widget; it passes unchanged when
    `_connect_setting_dependency_signals` skips ``image_source`` altogether
    -- and a combo whose change reaches no rule leaves the streaming
    settings greyed out for a user who just asked to stream, which is the
    version of "a label with an arrow" that actually reaches a screen.
    Verified by skipping that key in the connect loop: all 36 tests in this
    file still passed, and so did `tests/qt/test_no_control_is_inert.py`.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import SettingsWidgets

    model = SettingsWidgets(app_key)
    model.build_sections()
    combo = model._widgets["image_source"]
    qtbot.addWidget(combo)
    stream_only = model._widgets["stream_method"]

    assert combo.currentData() == TRAINING_LOAD_IMAGES
    assert not stream_only.isEnabled(), (
        "stream_method is read only while streaming, and the panel opened on "
        "LOAD IMAGES")

    combo.setCurrentIndex(1)
    assert combo.currentData() == TRAINING_STREAM_IMAGES
    assert stream_only.isEnabled(), (
        "choosing STREAM IMAGES left the streaming settings greyed out")

    # And back, so the rule is bound to the CHOICE rather than to having been
    # touched once.
    combo.setCurrentIndex(0)
    assert combo.currentData() == TRAINING_LOAD_IMAGES
    assert not stream_only.isEnabled(), (
        "going back to LOAD IMAGES left the streaming settings editable")
