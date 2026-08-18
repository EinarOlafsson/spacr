"""Dropping is the same gesture as "Add project folders…", and it ADDS.

The two handoffs instruction 109 left behind, both about the same change: on
the modules that merge databases, ``src`` stopped being a text box and became
a :class:`spacr.qt.widgets.database_set.DatabaseSetWidget`. Two pieces of code
were still asking the question the old way.

    1. ``dnd_handlers._set_src_on`` wrote through ``set_value_for_key``,
       which calls ``set_value``, which REPLACES the set. Dropping three
       plates on Image UMAP left one. Two of the three merges the user asked
       for were discarded with nothing said.
    2. ``app_screen._build_empty_state_banner`` decided "is a source set?"
       with ``isinstance(src, QLineEdit)``, so the "Point image umap at some
       data" card sat on top of three loaded databases for the whole session.

Both are measured here on the REAL Image UMAP screen and the real handler,
because both were invisible to a double: the double had a QLineEdit.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _plate(root: Path, name: str) -> Path:
    """A plate folder with the measurements database the handler looks for."""
    folder = root / name
    (folder / "measurements").mkdir(parents=True)
    con = sqlite3.connect(folder / "measurements" / "measurements.db")
    con.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
                "fieldID TEXT, object_label INTEGER, area REAL)")
    con.execute(f"INSERT INTO cell VALUES ('{name}','r1','c1','f1',1,10.0)")
    con.commit()
    con.close()
    return folder


@pytest.fixture()
def umap_screen(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("umap")
    qtbot.addWidget(screen)
    return screen


def test_three_dropped_plates_are_three_sources(umap_screen, tmp_path):
    """The bug, reproduced and fixed: it used to leave ['plate3']."""
    from spacr.qt.dnd_handlers import get_handler

    handler = get_handler("umap")
    src = umap_screen._settings_model._widgets["src"]
    plates = [_plate(tmp_path, name) for name in ("plate1", "plate2", "plate3")]

    for plate in plates:
        handler.apply(plate, umap_screen)

    assert src.sources() == [str(plate) for plate in plates]
    # And it reaches what the run would be given, not only the widget.
    assert umap_screen._settings_model.collect()["src"] == [
        str(plate) for plate in plates]


def test_the_same_plate_twice_is_one_source_and_still_succeeds(
        umap_screen, tmp_path):
    """`add_sources` returns how many were NEW; that is a different question.

    A user who drops the same folder twice has a screen pointing where they
    pointed it, so the drop succeeded. Reporting failure would raise "this
    module has no source field to receive the drop" at them.
    """
    from spacr.qt.dnd_handlers import _set_src_on, get_handler

    plate = _plate(tmp_path, "plate1")
    get_handler("umap").apply(plate, umap_screen)

    assert _set_src_on(umap_screen, str(plate)) is True
    assert umap_screen._settings_model._widgets["src"].sources() == [str(plate)]


def test_a_list_of_plates_is_added_whole(umap_screen, tmp_path):
    """Classify hands `_set_src_on` a list. A set must take all of it."""
    from spacr.qt.dnd_handlers import _set_src_on

    plates = [str(_plate(tmp_path, name)) for name in ("plateA", "plateB")]

    assert _set_src_on(umap_screen, plates) is True
    assert umap_screen._settings_model._widgets["src"].sources() == plates


def test_a_text_src_is_still_replaced(qtbot, tmp_path):
    """Adding is for a SET. A module whose src is one folder still takes one.

    Mask has a QLineEdit and dropping a second folder there means "look at
    this one instead" -- there is no set to join.
    """
    from PySide6.QtWidgets import QLineEdit
    from spacr.qt.dnd_handlers import _set_src_on
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert isinstance(screen._settings_model._widgets["src"], QLineEdit)

    assert _set_src_on(screen, "/data/plate1") is True
    assert _set_src_on(screen, "/data/plate2") is True
    assert screen._settings_model._widgets["src"].text() == "/data/plate2"


def test_the_empty_state_card_goes_away_once_databases_are_loaded(
        umap_screen, tmp_path):
    """It used to sit on top of three loaded databases for the whole session."""
    from spacr.qt.dnd_handlers import get_handler

    card = umap_screen._empty_state_card
    assert card is not None, "an unset Image UMAP still needs the card"
    assert not card.isHidden()

    get_handler("umap").apply(_plate(tmp_path, "plate1"), umap_screen)

    assert card.isHidden()


def test_emptying_the_set_brings_the_card_back(umap_screen, tmp_path):
    """A set can be emptied, unlike a path that has been typed.

    Removing the last database returns the screen to exactly the state the
    card describes, so leaving it hidden would leave a user with no data and
    nothing telling them how to get some.
    """
    from spacr.qt.dnd_handlers import get_handler

    src = umap_screen._settings_model._widgets["src"]
    get_handler("umap").apply(_plate(tmp_path, "plate1"), umap_screen)
    assert umap_screen._empty_state_card.isHidden()

    src.clear()

    assert not umap_screen._empty_state_card.isHidden()


def test_a_screen_reopened_with_a_loaded_set_never_builds_the_card(
        umap_screen, tmp_path):
    """`_build_empty_state_banner` reads the control, not its class."""
    src = umap_screen._settings_model._widgets["src"]
    src.set_value([str(_plate(tmp_path, "plate1"))])

    assert umap_screen._build_empty_state_banner() is None
