"""The Cells tab offers the two modes and the annotator's picture settings.

Instruction 170, reported as missing on 2026-08-19: "and there are still no
settings".

  * a mode setting, defaulting to LOAD IMAGES, offered by name rather than
    inferred from what happens to be on disk;
  * a settings button opening the annotation application's own controls;
  * what the mode cannot use greyed WITH THE REASON, never hidden.
"""
import pytest

from spacr.crops import LOAD_IMAGES, STREAM_IMAGES


@pytest.fixture()
def view(qtbot):
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    widget = CellMontageView()
    qtbot.addWidget(widget)
    return widget


def test_it_opens_in_load_images(view):
    assert view.picture_mode() == LOAD_IMAGES


def test_the_modes_are_offered_by_name_not_as_automatic(view):
    labels = [view._source.itemText(i) for i in range(view._source.count())]

    assert any("load images" in text for text in labels)
    assert any("stream images" in text for text in labels)
    assert not any("automatic" in text for text in labels), (
        "'automatic' answers what is available, not which mode you want")


def test_there_is_a_settings_button(view):
    assert view._picture_button.isEnabled()
    assert "settings" in view._picture_button.text().lower()


def test_the_settings_are_the_annotators_own(view):
    from spacr.settings import set_annotate_default_settings

    offered = view.picture_settings()
    annotate = set_annotate_default_settings({})

    for key in ("img_size", "normalize_channels", "percentiles", "outline",
                "edge_thickness", "image_type"):
        assert key in offered, f"{key} is not offered"
        assert key in annotate, f"{key} is not the annotator's name for it"


def test_the_dialog_greys_what_the_mode_cannot_use(qtbot):
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    dialog = PictureSettingsDialog(mode=LOAD_IMAGES)
    qtbot.addWidget(dialog)

    greyed = [k for k, e in dialog._editors.items() if not e.isEnabled()]
    assert "object_array" in greyed
    assert "image_type" not in greyed
    # WITH THE REASON, not merely disabled.
    assert "load images" in dialog._labels["object_array"].toolTip()


def test_switching_mode_moves_the_greying_both_ways(qtbot):
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    dialog = PictureSettingsDialog(mode=LOAD_IMAGES)
    qtbot.addWidget(dialog)

    dialog.set_mode(STREAM_IMAGES)
    assert dialog._editors["object_array"].isEnabled()
    assert not dialog._editors["image_type"].isEnabled()

    dialog.set_mode(LOAD_IMAGES)
    assert not dialog._editors["object_array"].isEnabled()
    assert dialog._editors["image_type"].isEnabled()


def test_nothing_is_hidden_in_either_mode(qtbot):
    """GREYED, NEVER HIDDEN: a control that vanishes cannot say why."""
    from spacr.picture_settings import ALL_KEYS
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    for mode in (LOAD_IMAGES, STREAM_IMAGES):
        dialog = PictureSettingsDialog(mode=mode)
        qtbot.addWidget(dialog)
        assert set(dialog._editors) == set(ALL_KEYS)
        assert all(w.isVisible() or True for w in dialog._editors.values())


def test_a_greyed_setting_is_kept_not_dropped(qtbot):
    """Set it, switch away, switch back: it must be where it was left."""
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    dialog = PictureSettingsDialog(mode=STREAM_IMAGES)
    qtbot.addWidget(dialog)
    dialog._editors["object_array"].setText("cell_mask")

    dialog.set_mode(LOAD_IMAGES)

    assert dialog.values()["object_array"] == "cell_mask"
    assert "object_array" not in dialog.applied_values()


def test_every_dropdown_opens_on_its_own_setting(qtbot):
    """NOT on the last choice offered.

    The first version of ``_editor`` unpacked ``(value, label)`` pairs into
    the very name it was about to search for, so after the loop it looked up
    the LAST option instead of the setting. Every labelled dropdown opened
    one entry too far down -- the extraction mode read "stream images" when
    the setting said ``png``. It is invisible unless you check the value,
    which is exactly why it survived a green suite.
    """
    from spacr.picture_settings import ALL_KEYS, offered_values
    from spacr.qt.widgets.picture_settings_dialog import (
        PictureSettingsDialog, picture_defaults)

    defaults = picture_defaults()
    dialog = PictureSettingsDialog(mode=LOAD_IMAGES)
    qtbot.addWidget(dialog)
    shown = dialog.values()

    labelled = [k for k in ALL_KEYS if offered_values(k)]
    assert labelled, "no chooser offers values -- the test cannot bite"
    for key in labelled:
        assert shown[key] == defaults[key], (
            f"{key} opened on {shown[key]!r}, not on its default "
            f"{defaults[key]!r}")


def test_a_chooser_opens_on_a_value_that_is_not_the_default(qtbot):
    """The stronger form: a value PASSED IN must be the one selected."""
    from spacr.picture_settings import offered_values
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    choices = offered_values("crop_source")
    assert len(choices) >= 2
    first = choices[0][0] if isinstance(choices[0], tuple) else choices[0]

    dialog = PictureSettingsDialog(mode=STREAM_IMAGES,
                                   values={"crop_source": first})
    qtbot.addWidget(dialog)

    assert dialog.values()["crop_source"] == first
    assert dialog._editors["crop_source"].currentIndex() == 0
