"""Mask Generation offers its example data the way Regression does.

Regression's button fills two TABLE slots from the packaged example screen.
Mask Generation needs IMAGES, which is a different set from a different
place -- `einarolafsson/toxo_mito` on Hugging Face, which spaCR already
downloads for its end-to-end demo. So it is a separate button of the same
shape rather than the same one widened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


def _fake_download(images, settings, seen=None):
    class Result:
        dataset_path = images
        settings_path = settings

    def ask(parent, destination, done):
        if seen is not None:
            seen["destination"] = destination
        done(Result(), "")

    return ask


def test_mask_generation_has_the_button(mask):
    button = getattr(mask, "_example_images_button", None)
    assert button is not None, "Mask Generation offers no example data"
    assert "example" in button.text().lower()


def test_it_is_in_the_section_that_holds_src():
    """A button for filling `src` belongs where `src` is."""
    from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    section = EXAMPLE_DATA_SECTIONS["mask"]
    holds_src = [name for name, keys in _APP_CATEGORY_SPECS["mask"]
                 if "src" in keys]
    assert holds_src == [section], (section, holds_src)


def test_regression_keeps_its_own_and_does_not_gain_this_one(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert getattr(screen, "_example_data_button", None) is not None
    assert getattr(screen, "_example_images_button", None) is None


def test_pressing_it_fills_src(mask, tmp_path):
    images = tmp_path / "plate1"
    images.mkdir()
    (images / "a.tif").write_bytes(b"")
    settings = tmp_path / "settings"
    settings.mkdir()

    placed = mask.load_the_example_images(
        ask=_fake_download(images, settings))
    assert placed["src"] == str(images)
    assert mask._settings_model._widgets["src"].text() == str(images)


def test_it_says_where_everything_went(mask, tmp_path):
    """Filling a field silently is indistinguishable from a button that did
    nothing -- and this one may have just moved 400 MB."""
    images = tmp_path / "plate1"
    images.mkdir()
    (images / "a.tif").write_bytes(b"")
    settings = tmp_path / "settings"
    settings.mkdir()

    said = []
    mask._console.append_stdout = lambda text: said.append(text)
    mask.load_the_example_images(ask=_fake_download(images, settings))
    joined = "".join(said)
    assert str(images) in joined
    assert str(settings) in joined


def test_a_failed_download_says_so_and_does_not_fill_src(mask):
    def ask(parent, destination, done):
        done(None, "the network is down")

    # UNCHANGED, not empty: mask's `src` opens with its own placeholder,
    # and the point is that a failed download does not overwrite whatever
    # the user already had there.
    control = mask._settings_model._widgets["src"]
    before = control.text()

    said = []
    mask._console.append_stdout = lambda text: said.append(text)
    placed = mask.load_the_example_images(ask=ask)
    assert placed == {}
    assert "the network is down" in "".join(said)
    assert control.text() == before


def test_a_plate_already_cached_is_not_downloaded_again(mask, tmp_path,
                                                       monkeypatch):
    """400 MB once. The second press must be free."""
    cache = tmp_path / "example_images"
    plate = cache / "plate1"
    plate.mkdir(parents=True)
    (plate / "a.tif").write_bytes(b"")
    monkeypatch.setattr(mask, "example_images_destination", lambda: cache)

    called = []

    def ask(parent, destination, done):          # must not run
        called.append(destination)

    placed = mask.load_the_example_images(ask=ask)
    assert called == [], "it downloaded a plate it already had"
    assert placed["src"] == str(plate)


def test_the_download_goes_beside_the_other_example_data(mask):
    """One cache folder, not two."""
    destination = mask.example_images_destination()
    assert ".cache" in str(destination) and "spacr" in str(destination)
