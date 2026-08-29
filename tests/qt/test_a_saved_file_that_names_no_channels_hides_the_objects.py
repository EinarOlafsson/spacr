"""A settings file restores the panel, and it restores the rule with it.

"it was in the mask modual that i saw the object settings eaven when object
channels were all none"

The rule that hides a setting whose object is not in the run is answered
against the panel's CURRENT VALUES, so the interesting case is not the fresh
panel -- it is the panel that has just been poured full from a file. Two
values do the damage if they are read wrongly: a stored ``None``, which
arrives from a CSV as the TEXT "None" and reads as a number to anything that
only checks for emptiness; and a stored ``0``, which is the FIRST PLANE and
reads as absent to anything written ``if not channel``.

Both are driven the whole way here: written with :func:`spacr.utils.
save_settings`, read back with the screen's own CSV loader, and applied
through the bulk apply an Import lands in -- with the event loop running,
because the panel answers on the next turn of it.  Cell is the deliberate
reference-object exception: instruction 300 keeps its controls available
even when no cell channel is named, while every optional object still follows
its own saved channel.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: One row per object that names a plane, and one setting of that object's
#: that must follow the plane on and off the form.
FOLLOWERS = {
    "cell": "cell_diameter",
    "nucleus": "nucleus_diameter",
    "pathogen": "pathogen_diameter",
    "organelle": "organelle_diameter",
}


def _screen(qtbot, app_key: str = "mask"):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    # The first visibility pass is scheduled, not immediate.
    qtbot.wait(1)
    return screen, screen._settings_model


def _write_and_read_back(tmp_path, settings: dict) -> dict:
    """Round-trip ``settings`` through a real spaCR settings CSV."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.utils import save_settings

    src = tmp_path / "merged"
    src.mkdir(exist_ok=True)
    out = dict(settings)
    out["src"] = str(src)
    save_settings(out, name="mask_settings", show=False)
    written = src / "settings" / "mask_settings.csv"
    assert written.is_file()
    return AppScreen._load_settings_csv(str(written))


# ---------------------------------------------------------------------------
# A file that names no channels
# ---------------------------------------------------------------------------

def test_a_file_that_names_no_channel_leaves_optional_objects_off_the_form(
        qtbot, tmp_path):
    """The whole route: save a panel with nothing switched on, open it."""
    screen, model = _screen(qtbot)
    loaded = _write_and_read_back(tmp_path, dict(model.collect()))
    for role in FOLLOWERS:
        assert loaded[f"{role}_channel"] is None, (
            f"{role}_channel came back from the CSV as "
            f"{loaded[f'{role}_channel']!r}")

    reopened, reopened_model = _screen(qtbot)
    reopened.apply_settings_dict(loaded)
    qtbot.wait(1)

    for role, follower in FOLLOWERS.items():
        expected = role == "cell"
        assert reopened.setting_row_is_visible(follower) is expected, (
            f"{follower} visibility disagrees with the saved "
            f"{role}_channel")
        if role == "organelle":
            # This file also says there are zero organelles. The optimized
            # form therefore builds no slot at all; the count is the control
            # that can ask for its first slot.
            assert reopened_model.collect()["number_of_organelles"] == 0
            assert f"{role}_channel" not in reopened_model._widgets
            assert reopened.setting_row_is_visible(
                "number_of_organelles") is True
        else:
            # Non-slot switches stay -- there would be nothing left to turn
            # the object back on with.
            assert reopened.setting_row_is_visible(
                f"{role}_channel") is True
    assert reopened_model.collect()["cell_channel"] is None


def test_a_file_that_omits_channels_keeps_only_the_reference_object(
        qtbot, tmp_path):
    """Absence is not a channel; cell alone remains available by design."""
    screen, model = _screen(qtbot)
    settings = {k: v for k, v in model.collect().items()
                if not k.endswith("_channel")}
    loaded = _write_and_read_back(tmp_path, settings)
    assert "cell_channel" not in loaded

    reopened, _model = _screen(qtbot)
    reopened.apply_settings_dict(loaded)
    qtbot.wait(1)

    assert reopened.setting_row_is_visible("cell_diameter") is True
    assert reopened.setting_row_is_visible("cell_channel") is True
    assert reopened.setting_row_is_visible("nucleus_diameter") is False
    assert reopened.setting_row_is_visible("nucleus_channel") is True


# ---------------------------------------------------------------------------
# A stored channel of zero
# ---------------------------------------------------------------------------

def test_a_stored_channel_of_zero_puts_its_object_back(qtbot, tmp_path):
    """Plane zero is the first plane, not an object that is not there."""
    screen, model = _screen(qtbot)
    settings = dict(model.collect())
    settings["cell_channel"] = 0
    loaded = _write_and_read_back(tmp_path, settings)
    assert loaded["cell_channel"] == 0

    reopened, reopened_model = _screen(qtbot)
    reopened.apply_settings_dict(loaded)
    qtbot.wait(1)

    assert reopened_model.collect()["cell_channel"] == 0
    assert reopened.setting_row_is_visible("cell_diameter") is True
    # And it says nothing about the objects the same file left unset.
    assert reopened.setting_row_is_visible("nucleus_diameter") is False


def test_the_rule_reads_a_saved_none_as_absent_and_a_saved_zero_as_present():
    """The two values, put to the rule itself, with nothing else moving."""
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    panel = ("nucleus_channel", "nucleus_diameter")
    for absent in (None, "None", "none", "", "  "):
        assert keys_hidden_by_their_object(
            panel, {"nucleus_channel": absent}) == {
                "nucleus_diameter"}, absent
    for present in (0, "0", 3, "3"):
        assert keys_hidden_by_their_object(
            panel, {"nucleus_channel": present}) == set(), present

    # Cell is the reference-object exception and therefore is never gated.
    assert keys_hidden_by_their_object(
        ("cell_channel", "cell_diameter"),
        {"cell_channel": None},
    ) == set()
