"""The packaged setting animations are too small until they are zoomed.

Measured across all 94 assets as generated, the content — everything that is
neither the black background nor the rounded field the generator draws around
every scene — covers a median of 63.9 % of the square. 72 of the 94 are below
70 % and the smallest, ``nucleus_diameter``, covers 22.8 %. Shown at tooltip
size that is a handful of pixels of illustration adrift in black.

:mod:`spacr.qt.widgets.animation_zoom` measures each animation once and crops
and rescales it so the content lands in a 70-80 % band. These tests measure
the real assets, before and after, with the same helper the widget uses.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets import animation_zoom as az
from spacr.setting_animations import animations_by_setting, setting_animations


#: A deterministic spread across the whole measured range, chosen by
#: percentile from the full corpus: the two smallest, the median, and the
#: largest — which is the only case that has to be scaled *down*.
SAMPLE = (
    "nucleus_diameter",
    "smooth_lines",
    "cell_diameter",
    "t_project_for_tracking",
    "pathogen_perimeter_fraction",
    "organelle_perimeter_fraction",
    "nucleus_Signal_to_noise",
    "nucleus_max_area",
    "normalization_percentiles",
    "organelle_max_intensity_percentile",
    "organelle_mask_within_cells",
    "pathogen_remove_border_objects",
)

DISPLAY_SIZE = 220


def _animation(slug: str):
    for animation in setting_animations():
        if animation.slug == slug:
            return animation
    raise AssertionError(f"no packaged animation named {slug!r}")


def _source_extent(animation) -> float:
    frames, _delays = az.read_frames(animation.path)
    return az.content_extent(frames, az.chrome_mask(frames[0].shape[0]))


def _zoomed_extent(zoomed) -> float:
    """Measure the produced frames — not the arithmetic that made them."""
    return az.content_extent(zoomed.frames, zoomed.chrome_mask())


# ---------------------------------------------------------------------------
# The problem, measured
# ---------------------------------------------------------------------------

def test_the_animations_are_mostly_too_small_as_generated():
    """The baseline this whole feature exists for.

    Pinned as a measurement rather than a comment: if the assets are ever
    regenerated at a sensible size this test fails, and the zoom can be
    reconsidered instead of silently doing nothing.
    """
    extents = [_source_extent(a) for a in setting_animations()]
    assert len(extents) == 94
    below = [value for value in extents if value < az.MIN_FILL]
    assert len(below) > len(extents) / 2, (
        "the assets no longer under-fill their frame; re-check the zoom")
    assert min(extents) < 0.30


# ---------------------------------------------------------------------------
# The fix, measured
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slug", SAMPLE)
def test_a_real_animation_zooms_into_the_seventy_to_eighty_band(slug):
    animation = _animation(slug)
    zoomed = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
    assert zoomed is not None

    measured = _zoomed_extent(zoomed)
    assert az.MIN_FILL <= measured <= az.MAX_FILL, (
        f"{slug}: content covers {measured:.1%} of the square "
        f"(was {zoomed.source_extent:.1%})")
    assert zoomed.frames[0].shape == (DISPLAY_SIZE, DISPLAY_SIZE, 3)
    assert len(zoomed.frames) == len(zoomed.delays)


@pytest.mark.heavy
def test_every_packaged_animation_zooms_into_the_band():
    """All 94, not a sample — the slow, complete version of the test above."""
    out_of_band = []
    for animation in setting_animations():
        zoomed = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
        assert zoomed is not None, animation.slug
        measured = _zoomed_extent(zoomed)
        if not az.MIN_FILL <= measured <= az.MAX_FILL:
            out_of_band.append((animation.slug, measured))
    assert out_of_band == []


def test_content_larger_than_the_band_is_scaled_down_not_left_alone():
    """The ``remove_border_objects`` scenes fill 91.9 % and must shrink.

    They are also the one case where the rounded field is the point of the
    animation — objects are removed *because* they touch it — so it has to
    survive the transform whole rather than be cropped away.
    """
    animation = _animation("pathogen_remove_border_objects")
    zoomed = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
    assert zoomed.source_extent > az.MAX_FILL
    assert _zoomed_extent(zoomed) <= az.MAX_FILL
    assert zoomed.shows_field, "the field this animation is about was cropped"
    assert zoomed.crop[2] > az.SOURCE_SIZE, "scaled down means a crop bigger than the frame"


def test_a_field_boundary_that_the_crop_would_slice_is_erased():
    """A well cut by the crop reads as two stray lines, not as a boundary."""
    animation = _animation("nucleus_diameter")
    zoomed = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
    assert not zoomed.shows_field
    assert zoomed.chrome_mask() is None

    # Nothing lit is left hugging the edges of the output.
    union = az.content_mask(zoomed.frames)
    border = 2
    assert not union[:border, :].any()
    assert not union[-border:, :].any()
    assert not union[:, :border].any()
    assert not union[:, -border:].any()


# ---------------------------------------------------------------------------
# The measurement itself
# ---------------------------------------------------------------------------

def test_the_field_boundary_is_not_counted_as_content():
    """Otherwise every animation measures ~93 % full and nothing is zoomed."""
    size = az.SOURCE_SIZE
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    ring = az.field_ring_mask(size)
    frame[ring] = 255

    assert az.content_bounds([frame], az.chrome_mask(size)) is None
    # Without the chrome mask the same frame looks nearly full.
    assert az.content_extent([frame]) > 0.9


def test_an_isolated_speck_is_not_content():
    """One quantisation speck used to stretch the crop across an empty half.

    ``nucleus_intensity_merge`` carries exactly one pixel of value 9 in one
    frame of eighteen; before it was discounted it pulled the measured content
    from 132x86 to 231x203 and the zoom framed a corner with nothing in it.
    """
    size = az.SOURCE_SIZE
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[100:140, 100:140] = 200          # a real shape
    frame[300, 300] = 9                    # one speck, just over the floor

    bounds = az.content_bounds([frame], az.chrome_mask(size))
    assert bounds == (100, 100, 139, 139)
    # And the raw threshold, with speck rejection off, still sees it — so the
    # difference above is the filter and not a change of threshold.
    assert az.content_bounds([frame], az.chrome_mask(size), 0) == (
        100, 100, 300, 300)


def test_a_lone_pixel_goes_and_a_touching_pair_stays():
    size = 40
    lonely = np.zeros((size, size), dtype=bool)
    lonely[10, 10] = True
    assert not az.drop_specks(lonely).any()

    pair = np.zeros((size, size), dtype=bool)
    pair[10, 10] = True
    pair[10, 11] = True
    assert az.drop_specks(pair).sum() == 2


def test_specks_are_dropped_per_frame_not_after_the_union():
    """A speck must not borrow a neighbour from another frame's content.

    Filtering the union instead would let a stray pixel in one frame be
    rescued by whatever a *different* frame happened to draw next to it, and
    the crop would then be sized around noise.
    """
    size = az.SOURCE_SIZE
    speckled = np.zeros((size, size, 3), dtype=np.uint8)
    speckled[100:140, 100:140] = 255
    speckled[10, 10] = 9                    # isolated in its own frame

    neighbourly = np.zeros((size, size, 3), dtype=np.uint8)
    neighbourly[11:31, 11:31] = 255         # touches (10, 10) diagonally

    assert az.content_bounds([speckled, neighbourly]) == (11, 11, 139, 139)
    # Union-first would have kept the speck and started the box at (10, 10).
    assert az.content_bounds([speckled, neighbourly], None, 0) == (
        10, 10, 139, 139)


def test_the_measurement_spans_every_frame_not_one_of_them():
    """An object that drifts must be framed where it goes, not where it starts."""
    size = az.SOURCE_SIZE
    first = np.zeros((size, size, 3), dtype=np.uint8)
    first[100:140, 100:140] = 255
    second = np.zeros((size, size, 3), dtype=np.uint8)
    second[200:240, 200:240] = 255

    assert az.content_bounds([first], az.chrome_mask(size)) == (
        100, 100, 139, 139)
    assert az.content_bounds([first, second], az.chrome_mask(size)) == (
        100, 100, 239, 239)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_the_zoom_is_computed_once_and_cached():
    """Recomputing per frame — 40 ms of decode and scan — would be absurd."""
    animation = _animation("cell_diameter")
    az.clear_cache()
    first = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
    hits_before = az.zoomed_animation.cache_info().hits
    second = az.zoomed_animation(str(animation.path), DISPLAY_SIZE)
    assert second is first
    assert az.zoomed_animation.cache_info().hits == hits_before + 1


def test_an_unreadable_asset_degrades_to_nothing(tmp_path):
    """A missing or corrupt GIF must not raise into the Qt event loop."""
    broken = tmp_path / "not-a-gif.gif"
    broken.write_bytes(b"this is not an image")
    assert az.zoomed_animation(str(broken), DISPLAY_SIZE) is None
    assert az.zoomed_animation(str(tmp_path / "absent.gif"), DISPLAY_SIZE) is None


def test_the_registry_maps_every_sampled_key():
    """The sample is written as slugs; keep it anchored to real settings."""
    mapped = {a.slug for a in animations_by_setting().values()}
    assert set(SAMPLE) <= mapped


# ---------------------------------------------------------------------------
# Qt bridge
# ---------------------------------------------------------------------------

def test_a_frame_survives_the_round_trip_through_qimage(qapp):
    """``to_qimage`` must copy: QImage does not own a Python buffer."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[4:9, 3:11] = (10, 200, 30)
    image = az.to_qimage(frame)
    assert (az.from_qimage(image) == frame).all()
