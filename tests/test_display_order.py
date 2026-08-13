"""Display order: how the user wants to LOOK at a crop. PR #74 / issue #73.

Commit 341f446 corrected the crop channel order on disk, which silently
mirrored what every pre-existing project SEES -- a parasite stain that was
always red opens blue. Red/green overlap is legible and blue/green is not, so
this is the difference between judging colocalisation by eye and not.

The two ways out before this were both bad: re-run measurement for hours to
fix a display, or mark the folder as a format it is not -- which works, and
then lies to every later reader.

THE DESIGN DECISION, and the reason there are two parameters rather than one
control: `stored_channel_order` answers "how was this file WRITTEN" (a fact
about the bytes) and `display_order` answers "how do I want to LOOK at it" (a
preference). One control doing both is what would confuse.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from spacr.crops import (
    DISPLAY_ORDERS,
    DISPLAY_ORDER_IDENTITY,
    CropError,
    apply_display_order,
    display_order_indices,
)


def _rgb():
    a = np.zeros((4, 4, 3), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 10, 20, 30
    return a


# ---------------------------------------------------------------------------
# The permutation
# ---------------------------------------------------------------------------

def test_all_six_orders_are_offered():
    assert set(DISPLAY_ORDERS) == {"rgb", "rbg", "grb", "gbr", "brg", "bgr"}


@pytest.mark.parametrize("order,expected", [
    ("rgb", (0, 1, 2)), ("bgr", (2, 1, 0)), ("grb", (1, 0, 2)),
    ("rbg", (0, 2, 1)), ("gbr", (1, 2, 0)), ("brg", (2, 0, 1)),
])
def test_each_order_names_its_source_planes(order, expected):
    assert display_order_indices(order) == expected


def test_the_identity_is_the_default_and_costs_nothing():
    image = _rgb()
    assert DISPLAY_ORDER_IDENTITY == "rgb"
    assert apply_display_order(image) is image, "the default must not copy"


def test_bgr_restores_a_pre_fix_project_s_picture():
    """The case the PR was opened for."""
    out = apply_display_order(_rgb(), "bgr")
    assert out[0, 0].tolist() == [30, 20, 10]


def test_spacing_and_commas_are_accepted():
    """A user typing 'b, g, r' means bgr."""
    assert display_order_indices("b, g, r") == (2, 1, 0)
    assert display_order_indices(" BGR ") == (2, 1, 0)


@pytest.mark.parametrize("bad", ["rg", "rgbb", "rrg", "xyz", "", None, "rgba"])
def test_an_order_that_is_not_a_permutation_is_refused(bad):
    """Silently ignoring a typed order shows a picture the user did not ask
    for and does not know they are not getting."""
    with pytest.raises(CropError):
        display_order_indices(bad)


def test_a_greyscale_crop_is_left_alone():
    """No three slots to permute; not wrong, just nothing to reorder."""
    grey = np.zeros((4, 4), np.uint8)
    assert apply_display_order(grey, "bgr") is grey


# ---------------------------------------------------------------------------
# Through the loader, which is the only seam
# ---------------------------------------------------------------------------

def test_the_loader_defaults_to_no_permutation(tmp_path):
    from spacr.qt.annotate_engine import load_crop_image

    path = str(tmp_path / "c.png")
    Image.fromarray(_rgb()).save(path)
    assert np.asarray(load_crop_image(path))[0, 0].tolist() == [10, 20, 30]


def test_the_loader_applies_the_chosen_order(tmp_path):
    from spacr.qt.annotate_engine import load_crop_image

    path = str(tmp_path / "c.png")
    Image.fromarray(_rgb()).save(path)
    out = np.asarray(load_crop_image(path, display_order="bgr"))
    assert out[0, 0].tolist() == [30, 20, 10]


def test_the_format_is_resolved_before_the_preference(tmp_path):
    """Reversing them would permute planes still in the wrong slots, and the
    two would compose into an order neither the file nor the user asked for."""
    import inspect

    from spacr.qt import annotate_engine

    source = inspect.getsource(annotate_engine.load_crop_image)
    assert source.index("read_crop_png(") < source.index("apply_display_order(")


def test_display_order_makes_no_claim_about_the_file(tmp_path):
    """The whole reason it is separate from stored_channel_order: setting it
    must not change how the bytes are interpreted."""
    from spacr.qt.annotate_engine import load_crop_image

    path = str(tmp_path / "c.png")
    Image.fromarray(_rgb()).save(path)

    as_written = np.asarray(load_crop_image(path, stored_channel_order="rgb"))
    permuted = np.asarray(load_crop_image(path, stored_channel_order="rgb",
                                          display_order="bgr"))
    # Same pixels, different slots -- nothing was re-decoded.
    assert sorted(as_written[0, 0].tolist()) == sorted(permuted[0, 0].tolist())
