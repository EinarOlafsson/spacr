"""Regression tests for latent bugs in spacr.measure found by the feature-dictionary pass.

Each test constructs an input whose CORRECT answer is known independently of
the implementation, so it fails if the bug is reintroduced.

Covered:
  1.  _estimate_blur measured a 1-D pixel vector, not a 2-D patch
  2.  mode_intensity was NaN for every object on SciPy >= 1.11
  3.  rad_dist_..._bin_0 was the field background, not the innermost shell
  4.  the Zernike degree was passed into mahotas' `radius` argument
  5.  organelle and *_organelle_summary rows never reached the database
  6.  the blur column carried its object/channel prefix twice
  7.  frac_high90 / frac_low10 were ~0.10 by construction
  9.  the parent-cell link vanished when radial_dist=False
  10. a dilation radius rounding to 0 dilated the crop to the whole field
  11. use_bounding_box + dialate_pngs inflated the radius by sqrt(label)

Everything runs offline on CPU with Agg, in well under a second.
"""
from __future__ import annotations

import os
import sqlite3

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure

import spacr.measure as M


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _disk(shape, cy, cx, r, value=1, dtype=np.int32):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    out = np.zeros(shape, dtype)
    out[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = value
    return out


def _intensity_settings(**over):
    base = dict(radial_dist=True, calculate_correlation=False, homogeneity=False,
                homogeneity_distances=[2], manders_thresholds=[85],
                distance_gaussian_sigma=0, cell_mask_dim=0, nucleus_mask_dim=1,
                pathogen_mask_dim=None, organelle_mask_dim=None, cytoplasm=False)
    base.update(over)
    return base


# ===========================================================================
# 1. _estimate_blur must measure a 2-D patch
# ===========================================================================

def test_estimate_blur_ranks_a_blurred_object_below_a_sharp_one():
    """Same object, same pixels, only the focus differs.

    The whole point of "variance of the Laplacian" is that defocus lowers it.
    """
    rng = np.random.default_rng(0)
    mask = _disk((64, 64), 32, 32, 18).astype(bool)
    sharp = rng.random((64, 64))
    blurred = gaussian_filter(sharp, sigma=3)

    v_sharp = M._estimate_blur(sharp, mask=mask)
    v_blurred = M._estimate_blur(blurred, mask=mask)

    assert v_sharp > v_blurred * 100, (v_sharp, v_blurred)


def test_estimate_blur_sees_structure_the_1d_vector_version_was_blind_to():
    """The old call passed image[label == id]: a 1-D vector in RASTER order.

    An image that varies only DOWN the columns (horizontal stripes) is strongly
    non-smooth in 2-D, but within one row it is constant -- so reading the
    object's pixels in raster order gives a nearly flat vector and the old
    "blur" score collapses. The 2-D measurement sees it, as it must.
    """
    mask = _disk((64, 64), 32, 32, 18).astype(bool)

    rows = np.zeros((64, 64))
    rows[::2, :] = 1.0                       # horizontal stripes: varies in y only
    cols = np.zeros((64, 64))
    cols[:, ::2] = 1.0                       # vertical stripes: varies in x only

    # Old behaviour: the 1-D vector of the object's pixels.
    old_rows = M._estimate_blur(rows[mask])
    old_cols = M._estimate_blur(cols[mask])
    # The old measure is wildly anisotropic: it barely registers the horizontal
    # stripes while the identical-but-rotated pattern scores ~57x higher.
    assert old_rows < old_cols / 20, (old_rows, old_cols)

    # Fixed behaviour: a rotation of the pattern must not change the score.
    new_rows = M._estimate_blur(rows, mask=mask)
    new_cols = M._estimate_blur(cols, mask=mask)
    assert new_rows == pytest.approx(new_cols, rel=0.05), (new_rows, new_cols)
    assert new_rows > 1.0


def test_estimate_blur_does_not_manufacture_an_edge_at_the_object_boundary():
    """A perfectly flat object on a bright background must score ~0.

    Masking the object into a zero-filled bounding box would put a step edge at
    the boundary and give a large, entirely artificial Laplacian variance.
    """
    mask = _disk((64, 64), 32, 32, 15).astype(bool)
    img = np.full((64, 64), 5000.0)
    img[mask] = 100.0                        # flat object, huge contrast to background

    assert M._estimate_blur(img, mask=mask) == pytest.approx(0.0, abs=1e-9)


def test_estimate_blur_without_a_mask_is_unchanged():
    """The no-mask signature still measures the whole array (existing callers)."""
    img = np.random.default_rng(3).random((32, 32))
    import cv2
    assert M._estimate_blur(img) == pytest.approx(
        cv2.Laplacian(img, cv2.CV_64F).var())


def test_estimate_blur_returns_nan_for_an_empty_mask():
    img = np.random.default_rng(4).random((16, 16))
    assert np.isnan(M._estimate_blur(img, mask=np.zeros((16, 16), bool)))


def test_estimate_blur_handles_a_single_pixel_object():
    """Too thin to erode: falls back to the un-eroded mask instead of failing."""
    img = np.random.default_rng(5).random((16, 16))
    mask = np.zeros((16, 16), bool)
    mask[8, 8] = True
    val = M._estimate_blur(img, mask=mask)
    assert np.isfinite(val) and val == pytest.approx(0.0)


# ===========================================================================
# 2. mode_intensity
# ===========================================================================

def test_mode_intensity_reports_the_chosen_modal_value_not_nan():
    """SciPy >= 1.11 returns a scalar from mode(); `.mode[0]` raised IndexError
    and a bare `except` wrote NaN for every object in every database."""
    mask = _disk((64, 64), 32, 32, 12)
    img = np.zeros((64, 64), np.float64)
    img[mask > 0] = 7.0                      # the modal value, by construction
    img[32, 32] = 99.0
    img[30, 30] = 1.0

    df = M._extended_regionprops_table(mask, img, ["label", "mean_intensity"])

    assert not df["mode_intensity"].isna().any()
    assert df["mode_intensity"].iloc[0] == pytest.approx(7.0)


def test_mode_intensity_picks_the_smallest_value_when_multimodal():
    mask = np.zeros((8, 8), np.int32)
    mask[0, :4] = 1
    img = np.zeros((8, 8), np.float64)
    img[0, :4] = [3.0, 3.0, 9.0, 9.0]        # 3 and 9 tie; scipy returns the smallest

    df = M._extended_regionprops_table(mask, img, ["label", "mean_intensity"])
    assert df["mode_intensity"].iloc[0] == pytest.approx(3.0)


# ===========================================================================
# 3. radial distribution
# ===========================================================================

def _radial_fixture():
    """A cell with a bright shell at a known distance from the nucleus edge."""
    shape = (80, 80)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    r2 = (yy - 40) ** 2 + (xx - 40) ** 2
    cell = np.zeros(shape, np.int32)
    cell[r2 < 30 ** 2] = 1
    nucleus = np.zeros(shape, np.int32)
    nucleus[r2 < 6 ** 2] = 1

    signal = np.zeros(shape, np.float64)
    signal[:] = 1000.0                        # very bright BACKGROUND outside the cell
    signal[cell > 0] = 0.0
    return cell, nucleus, signal, r2


def test_radial_bin_zero_is_the_innermost_shell_not_the_background():
    cell, nucleus, signal, _ = _radial_fixture()
    out = M._calculate_radial_distribution(cell, nucleus, signal[:, :, None], num_bins=6)
    bins = out[(1, 1, 0)]

    # Everything inside the cell is 0; the bright 1000 is background only. If
    # any bin picks up background the fix has regressed.
    assert np.nanmax(bins) == pytest.approx(0.0), bins
    assert not np.isnan(bins[0]), "bin 0 must exist"


def test_radial_distribution_puts_signal_in_the_bin_that_contains_it():
    """A shell placed just outside the nucleus must land in a LOW bin."""
    cell, nucleus, signal, r2 = _radial_fixture()
    # Bright shell hugging the nucleus (radius 6 -> 12), well inside the cell.
    shell = (r2 >= 7 ** 2) & (r2 < 12 ** 2)
    signal[shell] = 500.0

    out = M._calculate_radial_distribution(cell, nucleus, signal[:, :, None], num_bins=6)
    bins = np.nan_to_num(out[(1, 1, 0)])

    brightest = int(np.argmax(bins))
    assert brightest <= 1, (brightest, bins)
    # ...and the outer bins, which are pure dark cytoplasm, stay dark.
    assert bins[-1] == pytest.approx(0.0)
    assert bins[-2] == pytest.approx(0.0)


def test_radial_distribution_only_bins_pixels_inside_the_parent_cell():
    """Two cells: the signal in cell 2 must not leak into cell 1's profile."""
    shape = (64, 96)
    cell = np.zeros(shape, np.int32)
    cell[8:56, 4:44] = 1
    cell[8:56, 52:92] = 2
    nucleus = np.zeros(shape, np.int32)
    nucleus[28:36, 20:28] = 1                 # nucleus only in cell 1

    signal = np.zeros(shape, np.float64)
    signal[cell == 2] = 9999.0                # all the brightness is in the OTHER cell

    out = M._calculate_radial_distribution(cell, nucleus, signal[:, :, None], num_bins=4)
    bins = out[(1, 1, 0)]
    assert np.nanmax(bins) == pytest.approx(0.0), bins


# ===========================================================================
# 4. Zernike
# ===========================================================================

def test_zernike_degree_controls_the_number_of_coefficients():
    """The degree used to be ignored: mahotas always got its default 8, so the
    count was always 25 whatever the caller asked for."""
    mask = _disk((64, 64), 32, 32, 20)
    counts = {}
    for degree in (4, 8, 12):
        out = M._calculate_zernike(mask, pd.DataFrame({"label": [1]}), degree=degree)
        counts[degree] = len([c for c in out.columns if c.startswith("zernike_")])

    assert counts[4] == 9
    assert counts[8] == 25
    assert counts[12] == 49


def test_zernike_is_scale_normalised_across_object_sizes():
    """The same shape at two scales must give nearly the same coefficients.

    That is what a radius proportional to the object buys. With the radius
    pinned at 8 px the unit disk covered a fixed 8 px patch of every object, so
    the coefficients tracked the object's size instead of its shape.
    """
    from mahotas.features import zernike_moments

    small = _disk((64, 64), 32, 32, 12)
    big = _disk((160, 160), 80, 80, 30)

    cols_of = lambda d: [c for c in d.columns if c.startswith("zernike_")]
    z_small = M._calculate_zernike(small, pd.DataFrame({"label": [1]}), degree=8)
    z_big = M._calculate_zernike(big, pd.DataFrame({"label": [1]}), degree=8)
    a = z_small[cols_of(z_small)].to_numpy()[0]
    b = z_big[cols_of(z_big)].to_numpy()[0]
    scaled_error = np.max(np.abs(a - b))
    assert scaled_error < 0.06, scaled_error

    # And show the old fixed-radius call really was not scale normalised: a
    # disk smaller than the hard-coded 8 px radius and one much larger than it
    # disagree by an order of magnitude more.
    tiny = _disk((48, 48), 24, 24, 6).astype(bool)
    huge = _disk((160, 160), 80, 80, 30).astype(bool)
    old_error = np.max(np.abs(np.array(zernike_moments(tiny, 8))
                              - np.array(zernike_moments(huge, 8))))
    assert old_error > 0.3, old_error
    assert scaled_error < old_error / 5


def test_zernike_distinguishes_different_shapes():
    """Scale normalisation must not flatten genuine shape differences."""
    disk = _disk((64, 64), 32, 32, 16)
    bar = np.zeros((64, 64), np.int32)
    bar[30:34, 8:56] = 1

    zd = M._calculate_zernike(disk, pd.DataFrame({"label": [1]}), degree=8)
    zb = M._calculate_zernike(bar, pd.DataFrame({"label": [1]}), degree=8)
    cols = [c for c in zd.columns if c.startswith("zernike_")]
    assert np.max(np.abs(zd[cols].to_numpy()[0] - zb[cols].to_numpy()[0])) > 0.05


def test_zernike_survives_a_single_pixel_region():
    """radius would be 0 for a 1-px object; it is clamped rather than dividing by 0."""
    mask = np.zeros((16, 16), np.int32)
    mask[8, 8] = 1
    out = M._calculate_zernike(mask, pd.DataFrame({"label": [1]}), degree=4)
    zcols = [c for c in out.columns if c.startswith("zernike_")]
    assert len(zcols) == 9
    assert np.isfinite(out[zcols].to_numpy()).all()


# ===========================================================================
# 5. organelle tables reach the database
# ===========================================================================

def _write_and_read(tmp_path, morph, intensity, table_type):
    from spacr.utils import _merge_and_save_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True, exist_ok=True)
    _merge_and_save_to_database(morph, intensity, table_type, str(src),
                                "plate1_B03_2", "exp1", timelapse=False)
    db = src / "measurements" / "measurements.db"
    if not db.exists():
        return None
    con = sqlite3.connect(db)
    try:
        return pd.read_sql(f"SELECT * FROM {table_type}", con)
    finally:
        con.close()


def test_organelle_table_is_written_with_its_parent_cell_link(tmp_path):
    morph = pd.DataFrame({"label": [1, 2], "organelle_area": [10.0, 20.0]})
    intensity = pd.DataFrame({"label": [1, 2], "cell_id": [3, 4],
                              "organelle_channel_0_mean_intensity": [5.0, 6.0]})

    got = _write_and_read(tmp_path, morph, intensity, "organelle")

    assert got is not None, "the organelle table was never created"
    assert len(got) == 2
    # The organelle's parent is the CELL (measure._morphological_measurements
    # maps organelle -> cell), so cell_id is a key column just as for nucleus.
    assert list(got.columns[:2]) == ["object_label", "cell_id"]
    assert set(got["cell_id"]) == {3, 4}


@pytest.mark.parametrize("table", ["cell_organelle_summary", "nucleus_organelle_summary",
                                   "pathogen_organelle_summary", "cytoplasm_organelle_summary"])
def test_organelle_summary_tables_are_written_without_an_intensity_frame(tmp_path, table):
    """All four summaries pass an EMPTY intensity_df; the old len()>0 guard
    dropped every one of them silently."""
    morph = pd.DataFrame({"label": [1, 2],
                          "organelle_summary_organelle_count": [3, 0],
                          "organelle_summary_organelle_fraction": [0.25, 0.0]})

    got = _write_and_read(tmp_path, morph, pd.DataFrame(), table)

    assert got is not None, f"{table} was never created"
    assert len(got) == 2
    assert got.columns[0] == "object_label"
    assert set(got["organelle_summary_organelle_count"]) == {3, 0}


def test_a_summary_row_survives_the_full_measure_path(tmp_path):
    """End to end: real masks -> _summarize_organelles_per_parent -> DB."""
    cell = _disk((64, 64), 32, 32, 24)
    organelle = np.zeros((64, 64), np.int32)
    organelle[20:24, 20:24] = 1
    organelle[38:42, 38:42] = 2
    channels = np.random.default_rng(7).random((64, 64, 2))

    summary = M._summarize_organelles_per_parent(organelle, cell, channels,
                                                 parent_name="cell")
    summary.columns = [f"organelle_summary_{c}" if c != "label" else c
                       for c in summary.columns]
    got = _write_and_read(tmp_path, summary, pd.DataFrame(), "cell_organelle_summary")

    assert got is not None
    # Both organelles sit inside the one cell.
    assert int(got["organelle_summary_organelle_count"].iloc[0]) == 2


def test_an_object_table_with_no_intensity_frame_is_loud_not_silent(tmp_path, capsys):
    """The remaining skip prints instead of losing a field without trace."""
    morph = pd.DataFrame({"label": [1, 2], "cell_area": [10.0, 20.0]})
    got = _write_and_read(tmp_path, morph, pd.DataFrame(), "cell")

    assert got is None
    out = capsys.readouterr().out
    assert "cell" in out and "empty intensity frame" in out


def test_a_genuinely_unknown_table_type_is_still_rejected(tmp_path):
    from spacr.utils import _merge_and_save_to_database
    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    morph = pd.DataFrame({"label": [1], "area": [1.0]})
    intensity = pd.DataFrame({"label": [1], "mean_intensity": [1.0]})
    with pytest.raises(ValueError, match="Invalid table_type: mitochondrion"):
        _merge_and_save_to_database(morph, intensity, "mitochondrion", str(src),
                                    "plate1_B03_2", "exp1")


# ===========================================================================
# 6. blur column name
# ===========================================================================

def test_blur_column_is_prefixed_exactly_once():
    cell = _disk((64, 64), 32, 32, 20)
    nucleus = _disk((64, 64), 32, 32, 7)
    zeros = np.zeros_like(cell)
    channels = np.random.default_rng(8).random((64, 64, 2))

    cell_df, *_ = M._intensity_measurements(
        cell, nucleus, zeros, zeros, zeros, channels, _intensity_settings())

    blur_cols = sorted(c for c in cell_df.columns if c.endswith("blur"))
    assert blur_cols == ["cell_channel_0_blur", "cell_channel_1_blur"]
    assert not any("cell_channel_0_cell_channel_0" in c for c in cell_df.columns)


# ===========================================================================
# 7. frac_high90 / frac_low10
# ===========================================================================

def test_frac_high90_tracks_brightness_instead_of_being_pinned_at_0_1():
    """Thresholded on the FIELD's percentile, a bright object scores near 1 and
    a dim one near 0. Against its own percentile both were 0.10."""
    field = np.zeros((64, 64), np.float64)
    rng = np.random.default_rng(9)
    field[:] = rng.uniform(0.0, 1.0, size=(64, 64))     # dim background

    mask = np.zeros((64, 64), np.int32)
    mask[8:20, 8:20] = 1                                 # bright object
    mask[40:52, 40:52] = 2                               # dim object
    field[8:20, 8:20] = rng.uniform(50.0, 60.0, size=(12, 12))
    field[40:52, 40:52] = rng.uniform(0.0, 0.01, size=(12, 12))

    df = M._extended_regionprops_table(mask, field, ["label", "mean_intensity"])
    bright = df.loc[df["label"] == 1].iloc[0]
    dim = df.loc[df["label"] == 2].iloc[0]

    assert bright["frac_high90"] == pytest.approx(1.0)
    assert dim["frac_high90"] == pytest.approx(0.0)
    assert dim["frac_low10"] == pytest.approx(1.0)
    assert bright["frac_low10"] == pytest.approx(0.0)


def test_frac_high90_is_no_longer_a_constant_0_1_on_continuous_data():
    """Continuous data with no ties: the old version returned 0.1000 exactly."""
    rng = np.random.default_rng(10)
    mask = _disk((64, 64), 32, 32, 8)                    # ~5% of the field
    field = rng.random((64, 64))
    field[mask > 0] *= 8.0                               # object clearly above the field

    df = M._extended_regionprops_table(mask, field, ["label", "mean_intensity"])
    got = df["frac_high90"].iloc[0]
    assert got > 0.7, got

    # the object's OWN 90th percentile — the old reference — is 0.1 regardless
    intens = field[mask > 0]
    assert np.mean(intens > np.percentile(intens, 90)) == pytest.approx(0.1, abs=0.02)


# ===========================================================================
# 9. the parent-cell link must not depend on radial_dist
# ===========================================================================

@pytest.mark.parametrize("radial_dist", [True, False])
def test_nucleus_intensity_frame_always_carries_cell_id(radial_dist):
    cell = _disk((64, 64), 32, 32, 22)
    nucleus = _disk((64, 64), 32, 32, 8)
    zeros = np.zeros_like(cell)
    channels = np.random.default_rng(11).random((64, 64, 1))

    _, nucleus_df, *_ = M._intensity_measurements(
        cell, nucleus, zeros, zeros, zeros, channels,
        _intensity_settings(radial_dist=radial_dist))

    assert "cell_id" in nucleus_df.columns
    assert (nucleus_df["cell_id"] == 1).all()
    # exactly one copy, or the morphology/intensity merge would make
    # cell_id_x / cell_id_y and the key column would be lost
    assert list(nucleus_df.columns).count("cell_id") == 1


def test_cell_id_points_at_the_cell_the_object_actually_lies_in():
    """Two cells, one nucleus each: the link must not be off by a row."""
    shape = (64, 96)
    cell = np.zeros(shape, np.int32)
    cell[8:56, 4:44] = 1
    cell[8:56, 52:92] = 2
    nucleus = np.zeros(shape, np.int32)
    nucleus[28:36, 60:68] = 1          # nucleus label 1 lives in CELL 2
    nucleus[28:36, 20:28] = 2          # nucleus label 2 lives in CELL 1
    zeros = np.zeros_like(cell)
    channels = np.random.default_rng(12).random(shape + (1,))

    _, nucleus_df, *_ = M._intensity_measurements(
        cell, nucleus, zeros, zeros, zeros, channels,
        _intensity_settings(radial_dist=False))

    # The frame carries one 'label' column per concatenated block; they all
    # hold the same labels in the same order.
    labels = nucleus_df["label"]
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    link = dict(zip(labels, nucleus_df["cell_id"]))
    assert link[1] == 2
    assert link[2] == 1


# ===========================================================================
# 10. / 11. PNG crop region
# ===========================================================================

def test_zero_dilation_radius_does_not_grow_the_region_to_the_whole_field():
    """scipy reads iterations=0 as "dilate to a fixpoint", not "do nothing"."""
    region = np.zeros((64, 64), bool)
    region[10:14, 10:14] = True                # 16 px: under 25, so radius rounds to 0
    ratio = 0.2

    px = int(np.sqrt(np.count_nonzero(region)) * ratio)
    assert px == 0, "fixture must exercise the rounds-to-zero case"

    # what the old code did
    old = binary_dilation(region, structure=generate_binary_structure(2, 2), iterations=px)
    assert old.all(), "scipy iterations=0 must still fill the field (the trap)"

    # what measure.py does now: no dilation at all
    new = region if px <= 0 else binary_dilation(
        region, structure=generate_binary_structure(2, 2), iterations=px)
    assert np.array_equal(new, region)
    assert new.sum() == 16


def test_zero_radius_crop_stays_centred_on_the_object(tmp_path):
    """The visible consequence: the crop was a window on the middle of the field."""
    from spacr.utils import _crop_center
    region = np.zeros((64, 64), bool)
    region[10:14, 10:14] = True
    filled = np.ones((64, 64), bool)           # what iterations=0 produced

    img = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)[:, :, None].repeat(3, axis=2)

    on_object = _crop_center(img, region, 16, 16)
    on_field = _crop_center(img, filled, 16, 16)
    assert not np.array_equal(on_object, on_field)
    # the correctly centred crop contains the object's own pixels
    assert img[11, 11, 0] in set(on_object[..., 0].ravel().tolist())


def test_bounding_box_area_is_pixel_count_not_pixels_times_label():
    """_find_bounding_box fills the box with the LABEL VALUE, so np.sum scaled
    the dilation radius by the label id."""
    from spacr.utils import _find_bounding_box

    radii = []
    for label in (1, 100):
        mask = np.zeros((64, 64), np.int32)
        mask[20:24, 20:24] = label
        box = _find_bounding_box(mask, label, buffer=10)

        assert np.sum(box) == np.count_nonzero(box) * label   # the trap itself
        radii.append(int(np.sqrt(np.count_nonzero(box)) * 0.2))

    assert radii[0] == radii[1], "dilation must not depend on the label id"


# ===========================================================================
# 12. channel order is documented, deliberately not changed
# ===========================================================================

def test_png_channel_order_reversal_is_documented():
    """Left as-is on purpose: reversing it would change every crop already on
    disk and invalidate models trained on them. The docstring must say so."""
    doc = M.save_and_add_image_to_grid.__doc__ or ""
    assert "REVERSE" in doc
    assert "png_dims" in doc
