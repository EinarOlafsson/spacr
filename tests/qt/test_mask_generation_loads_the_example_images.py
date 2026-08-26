"""Mask Generation's example plate: what the button places, and what it says.

Regression's *Load the example screen…* fetches four plates of count and
score tables and drops them into its two table slots. A MASK run needs
neither: it needs a folder of images. ``spacr.example_data_manifest`` has no
images in it -- every entry is a ``counts`` or a ``scores`` CSV -- so the
button on this screen DRAWS a plate instead of downloading one, through
:func:`spacr.qt.synthetic.generate_mask_demo`, which is reproducible and
needs no network.

What is asserted here is the half that matters to somebody pressing it:

* the button is at the top of the section that holds the slot it fills,
  not in some other category;
* pressing it puts real image files on disk and puts that folder in ``src``;
* it fills the settings that say what is IN the images -- the channel each
  object is on, the diameters, the acquisition format -- because a folder
  whose channels are undeclared is not a run;
* and it SAYS all of that, naming every key it filled and every key this
  form has no control for. A file field that fills itself is
  indistinguishable from a button that did nothing.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import mask as mask_mod
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


class _Recorder:
    """Stands in for the console panel, and keeps what was written to it."""

    def __init__(self):
        self.lines = []

    def append_stdout(self, text):
        self.lines.append(str(text))

    def text(self):
        return "".join(self.lines)


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """A Mask Generation screen whose example plate lands in ``tmp_path``."""
    monkeypatch.setenv("SPACR_EXAMPLE_DATA", str(tmp_path / "cache"))
    widget = AppScreen(app_key="mask")
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Where the button is
# ---------------------------------------------------------------------------

def test_the_button_sits_at_the_top_of_the_section_that_holds_src(screen):
    """The section is found through the slot it fills, not by its heading.

    A button that writes into ``src`` from three categories away is a button
    whose effect is off screen when it is pressed.
    """
    button = mask_mod.install_example_data_button(screen)

    assert button is not None
    assert button.text() == mask_mod.EXAMPLE_BUTTON_TEXT
    src_widget = screen._settings_model._widgets["src"]
    holding = [section for section in screen._settings_sections
               if section.isAncestorOf(src_widget)]
    assert holding, "no settings section holds src"
    assert any(section.isAncestorOf(button) for section in holding), (
        "the example button is not in the section that holds src")


def test_installing_twice_leaves_one_button(screen):
    """The walk that installs it runs on every visit to this screen."""
    first = mask_mod.install_example_data_button(screen)
    second = mask_mod.install_example_data_button(screen)

    assert second is first


def test_the_fold_walk_installs_it_even_when_the_folds_do_not_mount(
        screen, monkeypatch):
    """A fold that cannot be built must not cost the example plate.

    The switches and the button arrive on the same walk, and the switches
    are the more fragile half: they mount settings categories onto a form
    that a preference may have hidden.
    """
    class _Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("the categories would not mount")

    monkeypatch.setattr(mask_mod, "CategoryFoldSet", _Broken)

    assert mask_mod.install_folds(screen) is None
    assert getattr(screen, "_example_images_button", None) is not None


def test_no_other_screen_gets_the_button(qtbot, qt_theme_applied):
    """It fills Mask Generation's src, and it is keyed on Mask Generation."""
    other = AppScreen(app_key="measure")
    qtbot.addWidget(other)

    assert mask_mod.install_example_data_button(other) is None


# ---------------------------------------------------------------------------
# What pressing it does
# ---------------------------------------------------------------------------

def test_pressing_it_writes_images_and_puts_the_folder_in_src(screen,
                                                              tmp_path):
    """The one thing a mask run cannot start without."""
    button = mask_mod.install_example_data_button(screen)
    screen._console = _Recorder()

    button.click()

    folder = mask_mod.example_plate_folder()
    assert folder.startswith(str(tmp_path))
    on_disk = sorted(name for name in os.listdir(folder)
                     if name.endswith(".tif"))
    assert len(on_disk) == 16, on_disk
    assert screen._settings_model.collect()["src"] == folder


def test_it_also_fills_what_says_what_is_in_the_images(screen):
    """A folder whose channels are undeclared is not a run.

    The demo draws the nucleus on C0, the cell on C1 and the pathogen on C2;
    a form left on its own defaults would segment the wrong planes and
    produce empty masks from images that are not empty.
    """
    mask_mod.install_example_data_button(screen)
    screen._console = _Recorder()

    placed = mask_mod.load_the_example_images(screen)

    values = screen._settings_model.collect()
    assert values["cell_channel"] == 1
    assert values["nucleus_channel"] == 0
    assert values["pathogen_channel"] == 2
    assert values["metadata_type"] == "cellvoyager"
    assert placed["applied"] >= 10
    assert "src" in placed["filled"]


def test_it_says_the_folder_the_count_and_every_key_it_filled(screen):
    """Saying nothing is indistinguishable from having done nothing."""
    mask_mod.install_example_data_button(screen)
    console = _Recorder()
    screen._console = console

    placed = mask_mod.load_the_example_images(screen)

    said = console.text()
    assert placed["folder"] in said
    assert f"{len(placed['images'])} image(s)" in said
    for key in placed["filled"]:
        assert key in said, f"{key} was filled and not reported"
    for key in placed["unplaced"]:
        assert key in said, f"{key} was dropped without being reported"


def test_pressing_it_twice_lands_on_the_same_plate(screen):
    """The plate is reproducible, so a second press is not a second copy.

    The files are redrawn -- identical inputs give identical bytes -- and
    what the second press reports is that it added nothing, which is what
    somebody who pressed the button twice needs to be told.
    """
    mask_mod.install_example_data_button(screen)
    screen._console = _Recorder()

    first = mask_mod.load_the_example_images(screen)
    second = mask_mod.load_the_example_images(screen)

    assert len(first["written"]) == len(first["images"])
    assert second["written"] == []
    assert second["images"] == first["images"]


def test_a_plate_that_cannot_be_written_is_said_and_not_raised(screen,
                                                               monkeypatch):
    """A refusal in a Qt slot has nowhere to go but the console.

    Raising out of ``clicked`` prints a traceback to a terminal the user is
    not looking at and leaves the form exactly as it was, with no hint that
    the button was pressed at all.
    """
    mask_mod.install_example_data_button(screen)
    console = _Recorder()
    screen._console = console

    def refuse(dst, **kwargs):
        raise OSError("Read-only file system")

    monkeypatch.setattr("spacr.qt.synthetic.generate_mask_demo", refuse)

    assert mask_mod.load_the_example_images(screen) == {}
    assert "Read-only file system" in console.text()
    assert "could not be written" in console.text()
    # And the form was left alone -- `src` still holds the default it opened
    # with -- rather than pointed at a folder that has nothing in it.
    assert screen._settings_model.collect()["src"] == "path"



# ---------------------------------------------------------------------------
# The gap this button is working around
# ---------------------------------------------------------------------------

def test_the_downloadable_example_screen_still_has_no_images():
    """The reason the plate is drawn rather than fetched.

    A ratchet, not a complaint: the day the release assets gain an image
    kind, this fails and the button should place THOSE instead of a drawing.
    """
    from spacr.example_data_manifest import FILES

    kinds = {entry["kind"] for entry in FILES}
    assert kinds == {"counts", "scores"}, (
        f"the example manifest now offers {sorted(kinds)}; Mask Generation's "
        f"button should place the real images rather than a drawn plate")
