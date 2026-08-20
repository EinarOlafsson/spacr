"""Instruction 176 -- three asks from 2026-08-19.

  A. "the measurements tab is ok now as long as only the attached databases
     tab in the measurements tab starts open"
  B. "there needs to be Tooltips for all the options in the cell settings"
  C. "the settings in regression want to use 0,1,2 instead of r,g,b but in
     the anotation app, r,g,b is used. i want this to be consistent, so use
     r,g,b in the regression cell feature"

C is the one with a trap in it. spaCR's default channel mapping is
{r: 2, g: 1, b: 0} -- 'r' is source channel TWO -- so reading "r,g,b" as
0,1,2 hands the streamer the planes in REVERSE and produces a crop that looks
entirely plausible and is wrong.
"""
import pytest

from spacr.picture_settings import ALL_KEYS, to_crop_settings


# ------------------------------------------------------- A: the calm tab


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    path = tmp_path / "spacr.ini"
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_only_the_attached_databases_section_starts_open(qtbot, store):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    panel = screen._scan_panel

    assert panel.is_section_expanded("Attached databases")
    folded = [t for t in panel.section_titles()
              if not panel.is_section_expanded(t)]
    assert set(folded) == set(panel.section_titles()) - {"Attached databases"}
    assert len(folded) == 3, folded


def test_the_stored_layout_still_wins(qtbot, store):
    """FIRST-RUN ONLY. 169 C says the arrangement persists, and a user who
    folded databases away last session must get that back."""
    from spacr.qt.screens.app_screen import AppScreen

    first = AppScreen("regression")
    qtbot.addWidget(first)
    first._scan_panel.set_section_expanded("Attached databases", False)
    first._scan_panel.set_section_expanded("Regression", True)

    again = AppScreen("regression")
    qtbot.addWidget(again)
    assert not again._scan_panel.is_section_expanded("Attached databases")
    assert again._scan_panel.is_section_expanded("Regression")


# ---------------------------------------------------------- B: the tooltips


def test_every_cell_setting_has_hover_help():
    from spacr.settings import tooltips

    missing = [k for k in ALL_KEYS if not str(tooltips.get(k, "")).strip()]
    assert not missing, f"no hover help for {missing}"


def test_the_help_is_in_the_one_table_the_dialog_already_reads():
    """NOT a second set of words. A tooltip written fresh beside this one
    would be a second answer to what "normalize" means -- the 145 failure
    this tab has already made once with the channel names."""
    import pathlib

    source = pathlib.Path(
        "spacr/qt/widgets/picture_settings_dialog.py").read_text()
    assert "from ...settings import tooltips" in source


def test_the_dialog_puts_the_help_on_the_name_not_the_field(qtbot):
    """Instruction 113: a tooltip on the box you are typing in is
    unreachable exactly when you wanted it."""
    from spacr.crops import LOAD_IMAGES
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    dialog = PictureSettingsDialog(mode=LOAD_IMAGES)
    qtbot.addWidget(dialog)

    for key in ALL_KEYS:
        assert dialog._labels[key].toolTip().strip(), key


def test_a_greyed_setting_says_why_instead_of_what():
    """The reason it does not apply is more use than its description when
    the control is greyed -- that is what the user is asking about."""
    from spacr.crops import LOAD_IMAGES
    from spacr.picture_settings import why_not

    assert why_not("object_array", LOAD_IMAGES).strip()


# ------------------------------------------------------ C: one vocabulary


def test_colour_letters_are_translated_not_dropped():
    out = to_crop_settings({"crop_source": "merged", "channels": "r,g,b"})
    assert "png_channel_mapping" in out


def test_r_is_the_channel_the_screen_put_in_red_not_channel_zero():
    """THE TRAP. spaCR's default is {r: 2, g: 1, b: 0}."""
    from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING

    out = to_crop_settings({"crop_source": "merged", "channels": "r,g,b"})
    assert out["png_channel_mapping"] == dict(DEFAULT_PNG_CHANNEL_MAPPING)
    assert out["png_channel_mapping"]["r"] == 2


def test_a_screens_own_mapping_wins_over_the_default():
    out = to_crop_settings({
        "crop_source": "merged", "channels": "r,g",
        "png_channel_mapping": {"r": 5, "g": 4, "b": 3}})
    assert out["png_channel_mapping"]["r"] == 5
    assert out["png_channel_mapping"]["g"] == 4


def test_a_colour_the_user_did_not_pick_is_blank_not_absent():
    """An absent key falls back to the default and quietly puts back a plane
    they turned off."""
    out = to_crop_settings({"crop_source": "merged", "channels": "r"})
    mapping = out["png_channel_mapping"]
    assert set(mapping) == {"r", "g", "b"}
    assert mapping["g"] is None and mapping["b"] is None


def test_an_index_form_still_works():
    """Unambiguous, and every settings CSV written before today carries it."""
    out = to_crop_settings({"crop_source": "merged", "channels": "0,1,2"})
    assert out["png_dims"] == [0, 1, 2]
    assert "png_channel_mapping" not in out


def test_the_cells_tab_asks_the_question_the_annotator_asks(qtbot, tmp_path):
    """r,g,b in both panels, so the same letters mean the same planes."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    assert view.picture_settings().get("channels") == "r,g,b"


def test_nonsense_channels_are_neither_letters_nor_indices():
    """Neither path claims it, so nothing reaches the crop layer to break on."""
    out = to_crop_settings({"crop_source": "merged", "channels": "wibble"})
    assert "png_dims" not in out and "png_channel_mapping" not in out


# ------------------------------------ the channel box accepts the annotator's


def test_the_channel_box_takes_the_annotation_apps_spelling():
    """Reported 2026-08-19: "'rgb' is not a list of channel indeces ... and
    this blocks the user from being able to spawn any images at all".

    THE TWO PANELS ASK DIFFERENT QUESTIONS IN THE SAME WORDS. The annotation
    app's `channels` is which COLOUR PLANES to show -- `_csv_to_list` keeps
    whatever strings it is given and `filter_channels_pil` reads 'r'/'g'/'b'
    directly. The Cells tab's box decides which SOURCE channels to cut, and
    answered in indices, so the annotator's answer raised ValueError and the
    whole tab refused to draw.
    """
    from spacr.qt.widgets.cell_montage_view import parse_channels

    assert parse_channels("r,g,b") is not None
    assert parse_channels("rgb") == parse_channels("r,g,b"), (
        "the compact form is what the report actually typed")
    assert parse_channels("RGB") == parse_channels("r,g,b")
    assert parse_channels("rg") == parse_channels("r,g")
    assert parse_channels("b") is not None


def test_the_letters_go_through_the_mapping_not_by_position():
    """spaCR's default is {r: 2, g: 1, b: 0}, so 'r' is source channel TWO.
    Reading it as 0 cuts the planes in reverse and produces a crop that looks
    entirely plausible and is wrong."""
    from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING
    from spacr.qt.widgets.cell_montage_view import parse_channels

    assert parse_channels("r") == (DEFAULT_PNG_CHANNEL_MAPPING["r"],)
    assert parse_channels("b") == (DEFAULT_PNG_CHANNEL_MAPPING["b"],)
    assert parse_channels("r,g,b") != (0, 1, 2), (
        "the letters were read positionally, which reverses the planes")


def test_channel_numbers_still_work():
    """Unambiguous, and anyone who knows their source channels can type them."""
    from spacr.qt.widgets.cell_montage_view import parse_channels

    assert parse_channels("0,1,2") == (0, 1, 2)


def test_an_empty_box_still_means_the_runs_own_channels():
    from spacr.qt.widgets.cell_montage_view import parse_channels

    assert parse_channels("") is None


def test_nonsense_is_still_refused():
    """Forgiving is not the same as silent: a word that is not a channel has
    no reading, and guessing one would draw the wrong planes."""
    from spacr.qt.widgets.cell_montage_view import parse_channels

    with pytest.raises(ValueError):
        parse_channels("wibble")


def test_the_tooltip_says_how_to_type_it(qtbot):
    """"instructions for doing this wil be in the tool tip"."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    hint = view._channels.toolTip()

    assert "r, g, b" in hint or "r,g,b" in hint
    assert "colour" in hint.lower()


def test_the_refusal_names_both_spellings(qtbot):
    """A refusal that only mentions numbers is what sent the user to numbers."""
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    view._channels.setText("wibble")
    reason = view.reason()

    if reason and "channel list" in reason:
        assert "r, g, b" in reason
        assert "0,1,2" in reason
