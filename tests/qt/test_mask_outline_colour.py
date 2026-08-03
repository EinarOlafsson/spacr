"""The Mask live preview's outline colour, asserted on the drawn pixels.

Execution list 5.4. The colour was effectively stuck green: ``auto`` (the
default) hard-coded the compartment's own colour, the ``Masks`` view painted
straight from ``OBJECT_COLORS`` whatever the user picked, and the comparison
scrubber dropped the random mode on the floor. ``auto`` now means a random
colour, re-rolled once per preview run.

Every assertion here reads the colour back out of the *rendered pixmap*, never
out of the setting — reading the setting back is exactly what let the bug
through in the first place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from PySide6.QtGui import QImage

from spacr.qt.widgets import live_preview as LP


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """QPixmap aborts the process when no QGuiApplication exists."""
    return qapp


def _rendered(view) -> np.ndarray:
    """Return the RGB pixels currently displayed by a ``_ZoomView``."""
    item = view._pixmap_item
    assert item is not None, "nothing has been rendered into this canvas"
    image = item.pixmap().toImage().convertToFormat(QImage.Format_RGB888)
    width, height = image.width(), image.height()
    buffer = np.frombuffer(image.constBits(), dtype=np.uint8)
    return buffer.reshape(height, image.bytesPerLine())[:, :width * 3] \
        .reshape(height, width, 3).copy()


def _pixel(view, x: int, y: int):
    return tuple(int(v) for v in _rendered(view)[y, x])


def _mask(qtbot):
    panel = LP.LivePreviewPanel()
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def outlined(qtbot, tmp_path):
    """A Mask panel with one image and one square mask, ready to render."""
    arr = np.full((40, 40), 100, np.uint16)
    path = tmp_path / "tile.tif"
    tifffile.imwrite(str(path), arr)
    panel = _mask(qtbot)
    panel.load_image(path)
    mask = np.zeros((40, 40), np.int32)
    mask[10:26, 10:26] = 1
    panel._raw_masks = {"cell": mask}
    panel._masks = {"cell": mask}
    panel._refresh_canvases()
    return panel


@pytest.mark.parametrize("name,rgb", sorted(
    LP.LivePreviewPanel.OUTLINE_COLOURS.items()))
def test_every_named_outline_colour_is_the_colour_that_gets_drawn(
        outlined, name, rgb):
    outlined._outline_colour.setCurrentText(name)
    drawn = _rendered(outlined._mask_view)
    assert _pixel(outlined._mask_view, 10, 10) == rgb
    # And nothing else on the canvas is painted in another outline colour.
    painted = {tuple(c) for c in np.unique(drawn.reshape(-1, 3), axis=0)}
    assert rgb in painted


def test_automatic_is_a_random_colour_not_green(outlined):
    """The headline bug: 'auto' hard-coded the compartment colour (green)."""
    assert outlined._outline_colour.currentText() == "auto"
    drawn = _pixel(outlined._mask_view, 10, 10)
    assert drawn == outlined._auto_outline_colour("cell")
    assert drawn != LP.OBJECT_COLORS["cell"], \
        "'automatic' is still painting the hard-coded green"


def test_automatic_gives_a_different_colour_on_each_run(outlined):
    """Re-rolled per preview run — 30 runs must not all come out the same."""
    seen = set()
    for _ in range(30):
        outlined._recompute_masks(snapshot=True)
        seen.add(_pixel(outlined._mask_view, 10, 10))
    assert len(seen) > 1, f"'auto' produced one fixed colour: {seen}"
    # 30 independent hues essentially never collide; a fixed colour would
    # give exactly one entry, and the old behaviour gave exactly green.
    assert len(seen) >= 25
    assert seen != {LP.OBJECT_COLORS["cell"]}


def test_automatic_holds_still_while_a_display_knob_is_tuned(outlined):
    """Random must not mean flickering: only a *run* re-rolls the colour."""
    first = _pixel(outlined._mask_view, 10, 10)
    outlined._outline_thickness.setValue(3)
    outlined._normalise_check.setChecked(False)
    assert _pixel(outlined._mask_view, 11, 11) == first


def test_re_selecting_automatic_rolls_a_fresh_colour(outlined):
    outlined._outline_colour.setCurrentText("red")
    assert _pixel(outlined._mask_view, 10, 10) == (240, 60, 60)
    seen = set()
    for _ in range(30):
        outlined._outline_colour.setCurrentText("red")
        outlined._outline_colour.setCurrentText("auto")
        seen.add(_pixel(outlined._mask_view, 10, 10))
    assert len(seen) > 1


def test_random_outline_colour_helper_is_vivid_and_varied():
    colours = {LP.random_outline_colour() for _ in range(200)}
    assert len(colours) > 100
    for red, green, blue in colours:
        assert max(red, green, blue) >= 216          # value >= 0.85
        assert max(red, green, blue) - min(red, green, blue) >= 100   # saturated


def test_the_masks_view_honours_the_chosen_colour(outlined):
    """The ``Masks`` view painted straight from OBJECT_COLORS — cells stayed
    green no matter which colour was picked."""
    outlined._outline_colour.setCurrentText("red")
    outlined._view_mode.setCurrentText("Masks")
    inside = _pixel(outlined._mask_view, 15, 15)
    assert inside != (0, 0, 0)
    # Shaded, but unmistakably the red that was chosen.
    assert inside[0] > inside[1] and inside[0] > inside[2]
    assert inside[0] >= 120
    green_base = LP.OBJECT_COLORS["cell"]
    assert not (inside[1] > inside[0] and inside[1] > inside[2]), \
        f"the Masks view is still painting {green_base}"


def test_the_masks_view_follows_automatic_too(outlined):
    outlined._view_mode.setCurrentText("Masks")
    base = np.array(outlined._auto_outline_colour("cell"), dtype=float)
    shade = 0.5 + 0.5 * ((1 % 7) / 6.0)
    expected = np.clip(base * shade, 0, 255).astype(np.uint8)
    assert np.allclose(_pixel(outlined._mask_view, 15, 15), expected, atol=1)


def test_the_comparison_scrubber_honours_the_random_mode(outlined):
    """Scrubbing back dropped ``random_outline`` and repainted in green."""
    outlined._recompute_masks(snapshot=True)
    outlined._recompute_masks(snapshot=True)
    assert len(outlined._history) >= 2
    outlined._outline_colour.setCurrentText("color (random)")
    live = _pixel(outlined._mask_view, 10, 10)
    outlined._compare_slider.setValue(0)
    scrubbed = _pixel(outlined._mask_view, 10, 10)
    assert scrubbed == live
    assert scrubbed != LP.OBJECT_COLORS["cell"]


def test_the_comparison_scrubber_honours_a_named_colour(outlined):
    outlined._recompute_masks(snapshot=True)
    outlined._recompute_masks(snapshot=True)
    outlined._outline_colour.setCurrentText("cyan")
    outlined._compare_slider.setValue(0)
    assert _pixel(outlined._mask_view, 10, 10) == (32, 200, 220)


def test_overlay_masks_still_defaults_to_the_compartment_colours():
    """The pure helper's contract is unchanged — only the panel's 'auto' is."""
    image = np.zeros((20, 20), np.uint8)
    mask = np.zeros((20, 20), np.int32)
    mask[5:15, 5:15] = 1
    plain = LP.overlay_masks(image, {"cell": mask})
    assert tuple(plain[5, 5]) == LP.OBJECT_COLORS["cell"]
    tinted = LP.overlay_masks(image, {"cell": mask},
                              outline_colors={"cell": (7, 9, 11)})
    assert tuple(tinted[5, 5]) == (7, 9, 11)
    # An explicit global colour still beats the per-compartment map.
    forced = LP.overlay_masks(image, {"cell": mask}, outline_rgb=(1, 2, 3),
                              outline_colors={"cell": (7, 9, 11)})
    assert tuple(forced[5, 5]) == (1, 2, 3)


def test_the_outline_colour_combo_is_never_translated(qtbot):
    """A language pass rewriting these entries would make every choice miss
    the colour table and silently fall back to the compartment default."""
    panel = _mask(qtbot)
    assert panel._outline_colour.property("i18nSkipItems") is True
    assert panel._view_mode.property("i18nSkipItems") is True
