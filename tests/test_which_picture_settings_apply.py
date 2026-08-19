"""What a mode does not use is greyed, with the reason, and never hidden.

Instruction 170: "settings that do not apply for the chosen method are grayed
out". The keys are the annotator's own -- a Cells tab with its own vocabulary
for the same picture would be two panels that disagree about what "normalize"
means, which is what 145 exists to stop.
"""
import pytest

from spacr.crops import LOAD_IMAGES, STREAM_IMAGES
from spacr.picture_settings import (ALL_KEYS, BOTH_MODES, applies_to,
                                    bounding_box_only, greyed_in, modes,
                                    why_not)


def test_load_images_is_offered_first():
    assert modes()[0][0] == LOAD_IMAGES
    assert modes()[0][1] == "load images"
    assert modes()[1][1] == "stream images"


@pytest.mark.parametrize("key", BOTH_MODES)
@pytest.mark.parametrize("mode", [LOAD_IMAGES, STREAM_IMAGES])
def test_the_shared_settings_apply_to_both(key, mode):
    """They shape the picture AFTER it is obtained, so the route is
    irrelevant."""
    assert applies_to(key, mode)
    assert why_not(key, mode) == ""


def test_the_disk_only_setting_is_greyed_when_streaming():
    assert not applies_to("image_type", STREAM_IMAGES)
    assert applies_to("image_type", LOAD_IMAGES)
    assert "stream images" in why_not("image_type", STREAM_IMAGES)


@pytest.mark.parametrize("key", ["object_array", "coordinate_columns",
                                 "crop_shape"])
def test_the_cut_settings_are_greyed_when_loading(key):
    assert not applies_to(key, LOAD_IMAGES)
    assert applies_to(key, STREAM_IMAGES)
    assert "load images" in why_not(key, LOAD_IMAGES)


def test_every_reason_says_WHY_not_merely_that_it_does_not_apply():
    """A greyed control that says only 'not used' teaches nothing."""
    for mode in (LOAD_IMAGES, STREAM_IMAGES):
        for key in greyed_in(mode):
            reason = why_not(key, mode)
            assert ": it " in reason, f"{key} in {mode} gives no reason"


def test_the_two_modes_grey_different_things():
    assert greyed_in(LOAD_IMAGES) != greyed_in(STREAM_IMAGES)
    assert not set(greyed_in(LOAD_IMAGES)) & set(greyed_in(STREAM_IMAGES))


def test_an_unknown_setting_is_left_alone():
    """Not this module's job to grey a control it has never heard of, and a
    panel that hid the unknown would hide new settings by default."""
    assert applies_to("some_new_knob", LOAD_IMAGES)
    assert why_not("some_new_knob", STREAM_IMAGES) == ""


def test_a_coordinate_cut_declares_itself_a_bounding_box():
    """"this could only do bounding box" -- said BEFORE the cut is made."""
    assert bounding_box_only({"crop_source": STREAM_IMAGES,
                              "coordinate_columns": ["x", "y"]})
    assert not bounding_box_only({"crop_source": STREAM_IMAGES,
                                  "coordinate_columns": []})
    assert not bounding_box_only({"crop_source": LOAD_IMAGES,
                                  "coordinate_columns": ["x", "y"]})


def test_every_key_the_panel_shows_is_covered():
    for key in ALL_KEYS:
        assert applies_to(key, LOAD_IMAGES) or applies_to(key, STREAM_IMAGES)


# ------------------------------------------- the settings reach the picture


def test_the_cut_settings_are_translated_to_the_crop_layers_names():
    """A settings window whose values never reached the picture would be
    worse than no settings window."""
    from spacr.picture_settings import to_crop_settings

    got = to_crop_settings({"crop_source": LOAD_IMAGES, "img_size": 256,
                            "channels": [0, 1]})

    assert got == {"png_size": 256, "png_dims": [0, 1]}


def test_a_setting_the_mode_does_not_use_is_not_translated():
    from spacr.picture_settings import to_crop_settings

    got = to_crop_settings({"crop_source": LOAD_IMAGES, "crop_shape": "bbox"})

    assert "use_bounding_box" not in got, (
        "a crop already written to disk was cut when it was written")


def test_the_shape_reaches_the_cut_when_streaming():
    from spacr.picture_settings import to_crop_settings

    box = to_crop_settings({"crop_source": STREAM_IMAGES, "crop_shape": "bbox"})
    obj = to_crop_settings({"crop_source": STREAM_IMAGES,
                            "crop_shape": "object"})

    assert box["use_bounding_box"] is True
    assert obj["use_bounding_box"] is False


def test_unset_values_are_left_out_rather_than_sent_as_blanks():
    from spacr.picture_settings import to_crop_settings

    assert to_crop_settings({"crop_source": LOAD_IMAGES}) == {}
    assert to_crop_settings({"crop_source": LOAD_IMAGES, "channels": []}) == {}
    assert to_crop_settings(None) == {}


def test_the_display_only_settings_are_deliberately_absent():
    """outline / edge_* / normalize change how a crop is DRAWN, not how it is
    cut. A mapping that pretended to apply them would be worse than none."""
    from spacr.picture_settings import CUT_SETTINGS

    for key in ("outline", "edge_thickness", "normalize_channels",
                "percentiles", "object_size"):
        assert key not in CUT_SETTINGS


# ------------------------------------- the display settings reach the picture


def test_nothing_asked_for_changes_nothing():
    import numpy as np

    from spacr.picture_settings import draw_crop

    crop = np.full((16, 16, 3), 100, dtype="uint8")

    assert np.array_equal(draw_crop(crop, {}), crop)
    assert np.array_equal(draw_crop(crop, None), crop)


def test_normalisation_stretches_the_crop():
    import numpy as np

    from spacr.picture_settings import draw_crop

    rng = np.random.default_rng(0)
    crop = (rng.random((32, 32, 3)) * 60 + 90).astype("uint8")

    out = draw_crop(crop, {"normalize_channels": ["r", "g", "b"],
                           "percentiles": [2, 98]})

    assert out.max() > crop.max(), "the stretch did not widen the range"
    assert out.shape == crop.shape


def test_a_channel_the_user_turned_off_is_off():
    import numpy as np

    from spacr.picture_settings import draw_crop

    crop = np.full((8, 8, 3), 200, dtype="uint8")

    out = draw_crop(crop, {"channels": ["r"]})

    assert out[:, :, 0].max() > 0
    assert out[:, :, 1].max() == 0 and out[:, :, 2].max() == 0


def test_it_uses_the_ANNOTATORS_functions_not_its_own():
    """A second implementation of 'normalise a crop' is a second answer to
    what normalise MEANS."""
    import inspect

    from spacr import picture_settings

    source = inspect.getsource(picture_settings.draw_crop)
    assert "from .qt.annotate_engine import" in source
    for name in ("normalize_pil", "filter_channels_pil", "outline_image"):
        assert name in source


def test_the_outline_is_computed_before_a_channel_is_zeroed():
    """Zeroing a channel first would outline a channel that is not there."""
    import inspect

    from spacr import picture_settings

    source = inspect.getsource(picture_settings.draw_crop)
    assert source.index("outline_image(") < source.index("filter_channels_pil(")


def test_a_bad_setting_never_costs_the_montage_its_picture():
    """Losing a montage to an outline is the worst trade available."""
    import numpy as np

    from spacr.picture_settings import draw_crop

    crop = np.full((8, 8, 3), 120, dtype="uint8")

    out = draw_crop(crop, {"percentiles": "junk", "outline": ["r"],
                           "edge_thickness": "nonsense"})

    assert out.shape == crop.shape


# --------------------------------------------- built from the screen, not typed


def test_the_arrays_come_from_the_screens_own_spec():
    """Offering `object_array` as free text asks the user to remember what
    their own screen contains, and to spell it the way `measure` did."""
    from spacr.picture_settings import available_arrays

    class Spec:
        mask_dims = {"cell": 3, "nucleus": 4}

    class Source:
        spec = Spec()

    assert available_arrays(Source()) == ("cell", "nucleus")


def test_a_screen_with_no_mask_planes_offers_none():
    """That is the answer, not a failure: a run whose arrays carry no mask
    planes cannot cut by one."""
    from spacr.picture_settings import available_arrays

    assert available_arrays(None) == ()


def test_a_box_needs_all_four_corners():
    """Three of them describe no box, so a chooser that offered them singly
    would let a user assemble a request that cannot be met."""
    import pandas as pd

    from spacr.picture_settings import available_coordinate_columns

    four = pd.DataFrame(columns=["bbox-0", "bbox-1", "bbox-2", "bbox-3"])
    three = pd.DataFrame(columns=["bbox-0", "bbox-1", "bbox-2"])

    assert available_coordinate_columns(four)
    assert available_coordinate_columns(three) == ()
    assert available_coordinate_columns(None) == ()


def test_a_pandas_index_has_no_truth_value():
    """`getattr(frame, 'columns', ()) or ()` RAISES on a DataFrame."""
    import pandas as pd

    from spacr.picture_settings import available_coordinate_columns

    assert available_coordinate_columns(pd.DataFrame()) == ()


def test_a_setting_with_no_inventory_stays_free_text():
    from spacr.picture_settings import offered_values

    assert offered_values("img_size") == ()
    assert offered_values("percentiles") == ()


def test_the_dialog_offers_them_as_a_dropdown(qtbot):
    import pandas as pd
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    class Spec:
        mask_dims = {"cell": 3, "pathogen": 5}

    class Source:
        spec = Spec()

    dialog = PictureSettingsDialog(
        mode=STREAM_IMAGES, source=Source(),
        objects=pd.DataFrame(columns=["bbox-0", "bbox-1", "bbox-2", "bbox-3"]))
    qtbot.addWidget(dialog)

    editor = dialog._editors["object_array"]
    assert isinstance(editor, QComboBox)
    assert [editor.itemText(i) for i in range(editor.count())] == \
        ["cell", "pathogen"]
    assert dialog.values()["object_array"] == "cell"


def test_a_screen_that_offers_nothing_still_lets_it_be_typed(qtbot):
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    dialog = PictureSettingsDialog(mode=STREAM_IMAGES)
    qtbot.addWidget(dialog)

    assert not isinstance(dialog._editors["object_array"], QComboBox)
