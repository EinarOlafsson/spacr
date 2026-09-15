"""317: a cached example must load the SHIPPED pack, not the user's run.

THE COLLISION IS BY DESIGN AND WAS WRITTEN DOWN AS A FEATURE.
`AppScreen._EXAMPLE_SETTINGS_FILES` lists the files an example may ship,
"best first", and for Mask that is::

    ("gen_mask_settings.csv", "gen_masks_settings.csv")

with a comment above it saying, correctly, that "a mask run saves
`gen_mask_settings.csv`, the older pack shipped `gen_masks_settings.csv`".
So the PREFERRED candidate is the name a completed run writes --
`utils.save_settings(settings, name='gen_mask_settings')` puts it at
`<src>/settings/gen_mask_settings.csv`, the same folder and the same name
this search looks in.

On a fresh download the pack is the only file there. After one run, the
user's own output sits under the preferred name and wins forever, because a
cached example is never re-fetched. It cannot happen until you have used the
thing once, which is why it only bites returning users and reads as
intermittent to everyone else.

Found by the other session's tutorial capture, which could not photograph
the Mask form until it moved two stale settings files aside.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

PACK_ROW = "flow_threshold,0.4\n"
RUN_ROW = "flow_threshold,100\n"


def _a_plate_with_both(tmp_path):
    """The shape a cached example is in after the user has run it once.

    `<dest>/settings/` is what the download fills -- a SIBLING of the plate,
    which no run writes into. `<dest>/plate1/settings/` is where a completed
    run saves. Both carry the same filename, which is the whole problem.
    """
    dest = tmp_path / "example"
    plate = dest / "plate1"
    (dest / "settings").mkdir(parents=True)
    (plate / "settings").mkdir(parents=True)
    (dest / "settings" / "gen_mask_settings.csv").write_text(
        "Key,Value\n" + PACK_ROW, encoding="utf-8")
    (plate / "settings" / "gen_mask_settings.csv").write_text(
        "Key,Value\n" + RUN_ROW, encoding="utf-8")
    return dest, plate


def test_the_shipped_pack_wins_over_the_users_own_run(tmp_path, qtbot,
                                                      monkeypatch):
    """THE DEFECT, as the user meets it.

    Both files exist, both are named `gen_mask_settings.csv`, and the one in
    the plate folder is the user's. The form must show the pack's value.
    """
    from spacr.qt.screens.app_screen import AppScreen

    dest, plate = _a_plate_with_both(tmp_path)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    seen = {}
    monkeypatch.setattr(AppScreen, "apply_settings_dict",
                        lambda self, loaded: seen.update(loaded) or len(loaded))
    monkeypatch.setattr(AppScreen, "reanchor_example_paths",
                        staticmethod(lambda loaded, destination: loaded))

    screen.apply_settings_that_came_with(plate, pack_folder=dest / "settings")

    assert str(seen.get("flow_threshold")) == "0.4", (
        f"the form was filled from {seen!r}; 100 is the user's own run "
        f"output and 0.4 is the shipped pack, so the cached example loaded "
        f"the wrong file")


def test_without_a_pack_folder_the_plate_is_still_read(tmp_path, qtbot,
                                                       monkeypatch):
    """THE HALF THAT MUST NOT REGRESS.

    Three of the four callers pass no pack folder, and the archives they
    unpack really do carry their settings inside the plate. A fix that only
    ever read the sibling would fill in nothing for them.
    """
    from spacr.qt.screens.app_screen import AppScreen

    _dest, plate = _a_plate_with_both(tmp_path)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    seen = {}
    monkeypatch.setattr(AppScreen, "apply_settings_dict",
                        lambda self, loaded: seen.update(loaded) or len(loaded))
    monkeypatch.setattr(AppScreen, "reanchor_example_paths",
                        staticmethod(lambda loaded, destination: loaded))

    screen.apply_settings_that_came_with(plate)

    assert str(seen.get("flow_threshold")) == "100", (
        "with no pack folder given, the plate's own settings are the only "
        "ones there are and must still be read")


def test_a_pack_folder_that_holds_nothing_falls_back_to_the_plate(tmp_path,
                                                                  qtbot,
                                                                  monkeypatch):
    """PROOF THE PREFERENCE IS NOT A REPLACEMENT.

    A pack folder that exists but carries no file for THIS app is the
    ordinary case -- `settings_pack` says plainly that "a pack legitimately
    carries settings for some apps and not others". Preferring it must not
    mean ignoring the plate when it is empty.
    """
    from spacr.qt.screens.app_screen import AppScreen

    dest, plate = _a_plate_with_both(tmp_path)
    (dest / "settings" / "gen_mask_settings.csv").unlink()
    screen = AppScreen("mask")
    qtbot.addWidget(screen)

    seen = {}
    monkeypatch.setattr(AppScreen, "apply_settings_dict",
                        lambda self, loaded: seen.update(loaded) or len(loaded))
    monkeypatch.setattr(AppScreen, "reanchor_example_paths",
                        staticmethod(lambda loaded, destination: loaded))

    screen.apply_settings_that_came_with(plate, pack_folder=dest / "settings")

    assert str(seen.get("flow_threshold")) == "100", (
        "an empty pack folder made the loader give up instead of falling "
        "back to the plate's own settings")
