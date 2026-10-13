"""Mask's overlay figure draws every organelle slot, and reads the right planes.

Item 76, open point recorded 2026-09-19 ("Mask's overlay plots draw the first
slot only"), fixed 2026-09-30. Measured before the fix: with a second slot in
the merged stack, ``plot_image_mask_overlay`` counted the mask planes back
from the end without it, so the cell outline was drawn from the nucleus plane
and every other mask was one plane off as well.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr.crops import MERGED_LAYOUT_SIDECAR  # noqa: E402

SHAPE = (40, 40)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _square(top, left, size, label=1):
    plane = np.zeros(SHAPE, dtype=np.float32)
    plane[top:top + size, left:left + size] = label
    return plane


def _merged(tmp_path, with_sidecar=True):
    """A merged field: 4 intensity planes, then cell, nucleus, two slots."""
    rng = np.random.default_rng(3)
    masks = {
        "cell": _square(2, 2, 30),
        "nucleus": _square(5, 5, 6),
        "organelle": _square(20, 20, 4),
        "organelleb": _square(26, 8, 4),
    }
    planes = [rng.integers(100, 4000, SHAPE).astype(np.float32)
              for _ in range(4)] + list(masks.values())
    merged = tmp_path / "merged"
    merged.mkdir()
    path = merged / "plate1_A01_1.npy"
    np.save(path, np.stack(planes, axis=-1))
    if with_sidecar:
        (merged / MERGED_LAYOUT_SIDECAR).write_text(json.dumps({
            "version": 1, "intensity_channels": [0, 1, 2, 3],
            "mask_plane_order": list(masks),
            "mask_dims": {role: 4 + i for i, role in enumerate(masks)}}))
    return path, masks


def _panel(fig, index):
    return np.asarray(fig.axes[index].images[0].get_array())


def _painted(panel, colour):
    return np.all(np.isclose(panel, colour, atol=1e-6), axis=-1)


def _draw(path, **kwargs):
    from spacr.plot import plot_image_mask_overlay

    return plot_image_mask_overlay(
        str(path), [0, 1, 2, 3], 0, 1, None, organelle_channel=2,
        figuresize=2, thickness=1, save_pdf=False, **kwargs)


def test_the_second_slot_gets_its_own_panel_and_colour(tmp_path):
    path, masks = _merged(tmp_path)
    fig = _draw(path, organelle_channels={"organelleb": 3})
    assert [a.get_title() for a in fig.axes] == [
        "cell (channel 0)", "nucleus (channel 1)", "organelle (channel 2)",
        "organelleb (channel 3)", "combined objects"]
    drawn = _painted(_panel(fig, 3), (1.0, 0.0, 1.0))      # magenta
    square = masks["organelleb"] > 0
    assert drawn.any()
    assert not drawn[~square].any()
    combined = _panel(fig, -1).max(axis=-1) > 0
    assert combined[square].all()


def test_the_cell_outline_comes_from_the_cell_plane(tmp_path):
    path, masks = _merged(tmp_path)
    fig = _draw(path)                                     # slot 2 not asked for
    red = _painted(_panel(fig, 0), (1.0, 0.0, 0.0))
    ys, xs = np.nonzero(red)
    # The cell square spans 2..31; the nucleus plane's square spans 5..10.
    assert ys.min() == 2 and ys.max() == 31
    assert xs.min() == 2 and xs.max() == 31


def test_without_a_sidecar_the_old_plane_arithmetic_is_kept(tmp_path):
    from spacr.plot import _overlay_mask_dims

    path, _ = _merged(tmp_path, with_sidecar=False)
    assert _overlay_mask_dims(str(path), ["cell", "nucleus"], 8) == {
        "cell": 6, "nucleus": 7}


def test_slot_helpers_order_slots_and_cycle_colours():
    from spacr.plot import _extra_organelle_slots, _organelle_slot_colour

    assert _extra_organelle_slots(
        {"organellec": 5, "organelle": 1, "organelleb": 4, "cell": 0,
         "organelled": None}) == [("organelleb", 4), ("organellec", 5)]
    assert _extra_organelle_slots(None) == []
    assert _organelle_slot_colour("default", 0) == "magenta"
    assert _organelle_slot_colour("colourblind", 0) == "#CC79A7"
    assert _organelle_slot_colour("nonsense", 6) == "magenta"
