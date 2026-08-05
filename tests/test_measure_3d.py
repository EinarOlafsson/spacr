"""Measure module: 3-D (Z, Y, X) mask support, and the 2-D path staying put.

The 3D/4D mask settings (``spacr.zstack``) can emit ``(Z, Y, X)`` label volumes,
but ``spacr.measure`` was written for ``(Y, X)`` masks throughout. Twelve call
sites crashed outright on a volume and roughly three dozen returned a number
that was silently measured in the wrong place. This module covers both halves
of the fix:

* the 2-D path is **unchanged**, column-for-column and value-for-value -- a 3-D
  capability that perturbs 2-D results is a regression, not a feature;
* a 3-D volume measures without crashing, the 2-D-only properties are *absent*
  rather than wrong, distances are sampled with the voxel spacing, and the row
  says what units it is in.

Everything here is synthetic, CPU-only, offline and fast.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

from spacr import measure as M


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def _settings(**kw):
    """A minimal measure settings dict with every key the measurement pass reads."""
    s = {
        'cell_mask_dim': 4,
        'nucleus_mask_dim': 5,
        'pathogen_mask_dim': None,
        'organelle_mask_dim': None,
        'cytoplasm': False,
        'radial_dist': True,
        'calculate_correlation': True,
        'manders_thresholds': [15, 85],
        'homogeneity': True,
        'homogeneity_distances': [8, 16],
        'distance_gaussian_sigma': 0,
        'strict_errors': False,
    }
    s.update(kw)
    return s


def _masks_2d(shape=(48, 48)):
    """One cell holding one nucleus, plus a second smaller cell."""
    cell = np.zeros(shape, np.uint16)
    nucleus = np.zeros(shape, np.uint16)
    cell[6:26, 6:26] = 1
    cell[30:42, 30:42] = 2
    nucleus[12:20, 12:20] = 1
    nucleus[34:38, 34:38] = 2
    zero = np.zeros(shape, np.uint16)
    return cell, nucleus, zero


def _masks_3d(shape=(6, 48, 48), z0=1, z1=5):
    """The 2-D masks extruded through planes ``z0:z1``."""
    cell2, nucleus2, _ = _masks_2d(shape[1:])
    cell = np.zeros(shape, np.uint16)
    nucleus = np.zeros(shape, np.uint16)
    for z in range(z0, z1):
        cell[z] = cell2
        nucleus[z] = nucleus2
    zero = np.zeros(shape, np.uint16)
    return cell, nucleus, zero


def _channels(shape, n=2, seed=7):
    rng = np.random.RandomState(seed)
    return rng.rand(*shape, n).astype(np.float64)


# --------------------------------------------------------------------------
# 1. the acceptance test: 2-D is unchanged
# --------------------------------------------------------------------------

def test_2d_morphology_frame_is_unchanged_column_for_column():
    """The 2-D morphology frame is exactly what it was before 3-D support.

    Pinned as an explicit column list and against a direct ``regionprops_table``
    call with no ``spacing``: if any of the plumbing leaks a spacing, a dropped
    property or a renamed column into the 2-D path, one of the two halves of
    this assertion fails.
    """
    from skimage.measure import regionprops_table

    cell, nucleus, zero = _masks_2d()
    s = _settings()
    cell_df, nucleus_df, _p, _o, _c = M._morphological_measurements(
        cell, nucleus, zero, zero, zero, s)

    expected_props = [
        'label', 'area', 'area_filled', 'area_bbox', 'convex_area',
        'major_axis_length', 'minor_axis_length', 'eccentricity', 'solidity',
        'extent', 'perimeter', 'euler_number', 'equivalent_diameter_area',
        'feret_diameter_max']
    assert list(cell_df.columns)[:len(expected_props)] == \
        ['label'] + [f'cell_{p}' for p in expected_props[1:]]

    # Zernike columns are present when the optional backend is installed; the
    # automatic path intentionally omits them when it is unavailable.
    has_zernike = any(c.startswith('cell_zernike_') for c in cell_df.columns)
    assert has_zernike is (importlib.util.find_spec("mahotas") is not None)
    assert not [c for c in cell_df.columns if 'volume' in c]

    # Value-for-value against an unspaced regionprops_table.
    reference = pd.DataFrame(regionprops_table(cell, properties=expected_props))
    for prop in expected_props[1:]:
        np.testing.assert_array_equal(
            cell_df[f'cell_{prop}'].to_numpy(),
            reference[prop].to_numpy(),
            err_msg=f'2-D {prop} changed')

    # The nucleus frame still carries its cell link (prefixed, as always).
    assert 'nucleus_cell_id' in nucleus_df.columns


def test_2d_intensity_frame_is_unchanged_column_for_column():
    """The 2-D intensity frame keeps every column, including the 2-D-only ones."""
    cell, nucleus, zero = _masks_2d()
    chans = _channels((48, 48))
    s = _settings()
    cell_i, nucleus_i, _p, _o, _c = M._intensity_measurements(
        cell, nucleus, zero, zero, zero, chans, s,
        sizes=[1, 2], periphery=True, outside=True)

    # The 2-D centroid names are NOT renamed; the axis suffix stays numeric.
    assert 'cell_channel_0_centroid_weighted-0' in cell_i.columns
    assert 'cell_channel_0_centroid_weighted-1' in cell_i.columns
    assert not [c for c in cell_i.columns if c.endswith('centroid_weighted_z')]

    # GLCM homogeneity still runs in 2-D.
    assert 'cell_channel_0_homogeneity_distance_8' in cell_i.columns
    assert 'cell_channel_0_homogeneity_distance_16' in cell_i.columns

    # Periphery / outside / radial blocks still present on the child object.
    assert 'nucleus_channel_0_periphery_mean' in nucleus_i.columns
    assert 'nucleus_channel_0_outside_mean' in nucleus_i.columns
    assert 'nucleus_rad_dist_channel_0_bin_0' in nucleus_i.columns
    assert 'cell_channel_0_channel_1_Pearson_correlation' in cell_i.columns


def test_2d_outside_intensity_still_uses_iterated_dilation():
    """``_outside_intensity`` with no spacing is the historical dilation ring.

    The 3-D ring is built from a sampled EDT, which is a *different* set of
    pixels (Euclidean rather than city-block). The 2-D default must not move to
    it.
    """
    from scipy.ndimage import binary_dilation

    mask = np.zeros((40, 40), np.uint16)
    mask[15:25, 15:25] = 1
    image = np.arange(1600, dtype=float).reshape(40, 40)

    stats = M._outside_intensity(mask, image, distance=5)
    ring = binary_dilation(mask == 1, iterations=5) & ~(mask == 1)
    assert stats[0][0] == 1
    assert stats[0][1] == pytest.approx(image[ring].mean())


def test_2d_estimate_blur_is_bit_identical():
    """The 2-D blur is the same float it was, on a real object patch."""
    import cv2
    from scipy.ndimage import binary_erosion, generate_binary_structure

    rng = np.random.RandomState(3)
    image = rng.rand(40, 40)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 12:28] = True

    # The reference recomputes the documented 2-D recipe by hand.
    r0, r1 = 10 - 1, 29 + 1
    c0, c1 = 12 - 1, 27 + 1
    patch = image[r0:r1 + 1, c0:c1 + 1]
    sub = mask[r0:r1 + 1, c0:c1 + 1]
    interior = binary_erosion(sub, structure=generate_binary_structure(2, 2))
    expected = float(cv2.Laplacian(patch.astype(np.float64), cv2.CV_64F)[interior].var())

    assert M._estimate_blur(image, mask=mask) == expected


def test_2d_spacing_is_none_even_when_a_voxel_size_is_configured():
    """A configured voxel size does NOT rescale a 2-D run.

    Applying it would turn every existing ``*_area`` column from px^2 into um^2
    under an unchanged name, which is exactly the silent unit change the schema
    decision exists to prevent.
    """
    spacing, stamp = M.resolve_measurement_spacing(
        _settings(voxel_size_z_um=1.0, voxel_size_xy_um=0.13), ndim=2)
    assert spacing is None
    assert stamp['measurement_units'] == M.UNITS_PX
    assert stamp['measurement_ndim'] == 2
    assert stamp['n_z'] == 1


# --------------------------------------------------------------------------
# 2. a 3-D volume measures, and the 2-D-only properties are absent
# --------------------------------------------------------------------------

def test_3d_morphology_measures_and_drops_2d_only_properties():
    cell, nucleus, zero = _masks_3d()
    s = _settings(voxel_size_z_um=1.0, voxel_size_xy_um=0.25)

    cell_df, nucleus_df, _p, _o, _c = M._morphological_measurements(
        cell, nucleus, zero, zero, zero, s)

    assert len(cell_df) == 2
    # Absent rather than wrong.
    for absent in ('cell_eccentricity', 'cell_perimeter'):
        assert absent not in cell_df.columns
    assert not [c for c in cell_df.columns if 'zernike' in c]
    # Still measured.
    for present in ('cell_area', 'cell_solidity', 'cell_major_axis_length',
                    'cell_euler_number', 'cell_feret_diameter_max'):
        assert present in cell_df.columns
    assert 'nucleus_cell_id' in nucleus_df.columns


def test_3d_regionprops_2d_only_properties_would_have_crashed():
    """The survey's crash reproduces: one 2-D-only name kills the whole call."""
    from skimage.measure import regionprops_table

    cell, _n, _z = _masks_3d()
    for prop in M.PROPS_2D_ONLY:
        with pytest.raises(NotImplementedError):
            regionprops_table(cell, properties=['label', 'area', prop])
    # ... and the surviving list does not.
    kept = [p for p in M.MORPHOLOGICAL_PROPS if p not in M.PROPS_2D_ONLY]
    regionprops_table(cell, properties=kept, spacing=(1.0, 0.25, 0.25))


def test_3d_zernike_is_skipped_rather_than_crashing():
    """mahotas' zernike_moments unpacks a 2-D shape; a 3-D region raises."""
    pytest.importorskip(
        "mahotas",
        reason="numerical Zernike descriptors require the optional spacr[zernike] extra",
    )
    from mahotas.features import zernike_moments

    cell, _n, _z = _masks_3d()
    with pytest.raises(ValueError):
        zernike_moments(cell[1:5, 6:26, 6:26] > 0, 5.0, degree=8)

    frame = pd.DataFrame({'label': [1, 2]})
    out = M._calculate_zernike(cell, frame, degree=8)
    assert list(out.columns) == ['label']


def test_3d_homogeneity_is_refused_not_approximated():
    """GLCM is 2-D only; the 3-D run writes no homogeneity columns at all."""
    cell, nucleus, zero = _masks_3d()
    with pytest.raises(ValueError, match='2-D only'):
        M._calculate_homogeneity(cell, np.zeros(cell.shape))

    chans = _channels((6, 48, 48))
    s = _settings(homogeneity=True, anisotropy=4.0)
    cell_i, _n, _p, _o, _c = M._intensity_measurements(
        cell, nucleus, zero, zero, zero, chans, s,
        sizes=[1], periphery=True, outside=True)
    assert not [c for c in cell_i.columns if 'homogeneity' in c]


def test_3d_intensity_measures_end_to_end():
    cell, nucleus, zero = _masks_3d()
    chans = _channels((6, 48, 48))
    s = _settings(anisotropy=4.0)
    cell_i, nucleus_i, _p, _o, _c = M._intensity_measurements(
        cell, nucleus, zero, zero, zero, chans, s,
        sizes=[1], periphery=True, outside=True)

    assert len(cell_i) == 2
    assert 'cell_channel_0_mean_intensity' in cell_i.columns
    assert 'cell_channel_0_blur' in cell_i.columns
    assert 'nucleus_channel_0_outside_mean' in nucleus_i.columns
    assert 'nucleus_rad_dist_channel_0_bin_0' in nucleus_i.columns
    assert 'cell_channel_0_channel_1_Pearson_correlation' in cell_i.columns


def test_3d_centroid_columns_are_renamed_so_axis_zero_is_never_y():
    """``centroid_weighted-0`` is y in 2-D and z in 3-D; the 3-D one is renamed.

    A reader that pulls ``..._centroid_weighted-0`` out of a 3-D database would
    otherwise get a plane index where it expected a row. Absent beats wrong.
    """
    cell, nucleus, zero = _masks_3d()
    chans = _channels((6, 48, 48))
    s = _settings(anisotropy=4.0)
    cell_i, *_ = M._intensity_measurements(
        cell, nucleus, zero, zero, zero, chans, s,
        sizes=[1], periphery=False, outside=False)

    assert 'cell_channel_0_centroid_weighted-0' not in cell_i.columns
    for axis in ('z', 'y', 'x'):
        assert f'cell_channel_0_centroid_weighted_{axis}' in cell_i.columns
        assert f'cell_channel_0_centroid_weighted_local_{axis}' in cell_i.columns

    # And the z coordinate really is a plane index: the cell spans planes 1..4.
    z = cell_i['cell_channel_0_centroid_weighted_z'].to_numpy()
    assert np.all((z >= 1 * 4.0) & (z <= 4 * 4.0))   # anisotropy-scaled


def test_3d_measure_intensity_distance_no_longer_unpacks_two_values():
    """The ``minr, minc = ...`` unpacking crashed on a 3-D coordinate array."""
    cell, nucleus, zero = _masks_3d()
    with pytest.raises(ValueError, match='too many values to unpack'):
        minr, minc = np.min(np.argwhere(cell == 1), axis=0)

    chans = _channels((6, 48, 48))
    s = _settings(distance_gaussian_sigma=2, anisotropy=4.0)
    df = M._measure_intensity_distance(cell, nucleus, zero, chans, s)
    assert list(df['label']) == [1, 2]
    assert df['cell_channel_0_distance_to_nucleus'].notna().all()


# --------------------------------------------------------------------------
# 3. data[:, :, channels] on a 4-D array
# --------------------------------------------------------------------------

def test_channel_selection_on_a_4d_array_selects_channels_not_x():
    """``data[:, :, channels]`` indexed X on a (Z, Y, X, C) array.

    Constructed so the two are unmistakable: every voxel of channel ``c`` holds
    the value ``c``, so a correct channel selection contains only the requested
    channel values, while the old X-slice contains every channel.
    """
    z, y, x, n_ch = 4, 12, 20, 6
    data = np.zeros((z, y, x, n_ch), np.uint16)
    for c in range(n_ch):
        data[..., c] = c
    channels = [0, 1, 2]

    old = data[:, :, channels]
    assert old.shape == (z, y, len(channels), n_ch)      # X sliced, C intact
    assert sorted(np.unique(old)) == list(range(n_ch))   # every channel present

    new = data[..., channels]
    assert new.shape == (z, y, x, len(channels))
    assert sorted(np.unique(new)) == channels
    for i, c in enumerate(channels):
        assert np.all(new[..., i] == c)

    # And on the 2-D layout the two spellings are the same array.
    flat = data[0]
    np.testing.assert_array_equal(flat[:, :, channels], flat[..., channels])


def test_measure_crop_core_reads_channels_from_a_4d_array(tmp_path):
    """End to end: a 4-D merged array measures, and its numbers are the volume's.

    The mask slices carry real labels and the intensity channels carry a known
    constant per channel, so a wrong axis would show up immediately as the
    wrong mean intensity or the wrong object count.
    """
    src = tmp_path / 'merged'
    src.mkdir(parents=True)
    (tmp_path / 'measurements').mkdir(parents=True)
    z, y, x = 4, 40, 40
    data = np.zeros((z, y, x, 6), np.uint16)
    data[..., 0] = 11
    data[..., 1] = 22
    cell, nucleus, _zero = _masks_3d((z, y, x), z0=1, z1=3)
    data[..., 4] = cell
    data[..., 5] = nucleus
    np.save(src / 'plate1_A01_f1.npy', data)

    s = _settings(
        src=str(src), channels=[0, 1], cell_mask_dim=4, nucleus_mask_dim=5,
        pathogen_mask_dim=None, cell_min_size=0, nucleus_min_size=0,
        pathogen_min_size=0, cytoplasm_min_size=0, cytoplasm=False,
        uninfected=True, merge_edge_pathogen_cells=False,
        timelapse=False, timelapse_objects=['cell'], save_measurements=True,
        save_png=False, save_arrays=False, plot=False, verbose=False,
        experiment='exp', voxel_size_z_um=1.0, voxel_size_xy_um=0.25,
        homogeneity=False, radial_dist=False, calculate_correlation=False)

    index, avg, cells, figs = M._measure_crop_core(0, [], 'plate1_A01_f1.npy', s)
    assert not isinstance(cells, int), 'the field failed inside _measure_crop_core'

    db = tmp_path / 'measurements' / 'measurements.db'
    assert db.is_file()
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query('SELECT * FROM cell', conn)
    finally:
        conn.close()

    assert len(df) == 2
    # The channel really is the channel: constant 11 and 22, not a slab of X.
    assert df['cell_channel_0_mean_intensity'].round().tolist() == [11.0, 11.0]
    assert df['cell_channel_1_mean_intensity'].round().tolist() == [22.0, 22.0]


# --------------------------------------------------------------------------
# 4. blur is measured in the right plane
# --------------------------------------------------------------------------

def test_blur_on_a_zyx_volume_is_measured_in_the_xy_plane():
    """Focus is read from the xy plane, and the old code read it from zy.

    A single ``cv2.Laplacian`` call on a ``(Z, Y, X)`` array does not raise:
    OpenCV reads the array as ``Z`` rows by ``Y`` columns by ``X`` colour
    channels (it accepts up to 512), so it differentiates along **z and y** and
    treats x as a channel index. The two volumes below are built so the swap is
    unambiguous -- one varies along x only, the other along z only -- and each
    is invisible to exactly one of the two implementations.
    """
    import cv2

    z, y, x = 6, 30, 30
    mask = np.zeros((z, y, x), bool)
    mask[1:5, 5:25, 5:25] = True

    # Structure along x only: constant down z and across y.
    sharp_x = np.zeros((z, y, x))
    sharp_x[:, :, ::2] = 1.0

    # Structure along z only: every plane is uniform, plane value alternates.
    sharp_z = np.zeros((z, y, x))
    for k in range(z):
        sharp_z[k] = float(k % 2)

    # The in-plane measurement sees the x structure and not the z structure.
    assert M._estimate_blur(sharp_x, mask=mask) > 0.1
    assert M._estimate_blur(sharp_z, mask=mask) == pytest.approx(0.0, abs=1e-12)

    # The unguarded call is exactly the other way round: blind to x, and
    # reporting z structure as if it were focus.
    assert cv2.Laplacian(sharp_x, cv2.CV_64F)[mask].var() == pytest.approx(0.0, abs=1e-12)
    assert cv2.Laplacian(sharp_z, cv2.CV_64F)[mask].var() > 0.1

    # Real in-plane texture also scores, and identically on every plane.
    rng = np.random.RandomState(11)
    textured = np.repeat(rng.rand(y, x)[None, :, :], z, axis=0)
    assert M._estimate_blur(textured, mask=mask) > 0.01


def test_blur_on_a_single_plane_volume_matches_the_2d_measurement():
    rng = np.random.RandomState(5)
    image = rng.rand(30, 30)
    mask = np.zeros((30, 30), bool)
    mask[8:22, 8:22] = True

    flat = M._estimate_blur(image, mask=mask)
    vol = M._estimate_blur(image[None, :, :], mask=mask[None, :, :])
    assert vol == pytest.approx(flat)


def test_estimate_blur_refuses_a_4d_mask():
    with pytest.raises(ValueError, match='2-D .* or 3-D'):
        M._estimate_blur(np.zeros((2, 2, 4, 4)), mask=np.ones((2, 2, 4, 4), bool))


# --------------------------------------------------------------------------
# 5. EDT-derived measures respond to anisotropic sampling
# --------------------------------------------------------------------------

def test_outside_intensity_ring_follows_the_voxel_spacing():
    """The 'outside' ring is a physical shell, not an iteration count.

    With ``dz = 5 dxy`` an iterated dilation reaches 5 planes out -- 25 xy-pixel
    equivalents -- while reaching only 5 pixels sideways. Hand-computed: the
    sampled ring extends exactly one plane in z (5 * 0.2 um = 1.0 um, the ring
    width) and 5 pixels in xy.
    """
    from scipy.ndimage import distance_transform_edt

    z, y, x = 9, 30, 30
    mask = np.zeros((z, y, x), np.uint16)
    mask[4, 14:17, 14:17] = 1
    image = np.zeros((z, y, x), float)

    spacing = (1.0, 0.2, 0.2)          # dz = 1 um, dxy = 0.2 um -> anisotropy 5
    stats = M._outside_intensity(mask, image, distance=5, spacing=spacing)
    assert stats[0][0] == 1

    region = mask == 1
    edt = distance_transform_edt(~region, sampling=spacing)
    ring = (edt <= 5 * 0.2) & ~region
    # Reaches exactly one plane away in z ...
    assert ring[3].any() and ring[5].any()
    assert not ring[2].any() and not ring[6].any()
    # ... and five pixels away in xy on the object's own plane.
    assert ring[4, 14 - 5, 15]
    assert not ring[4, 14 - 6, 15]

    # The unsampled dilation, by contrast, reaches five planes.
    from scipy.ndimage import binary_dilation
    naive = binary_dilation(region, iterations=5) & ~region
    assert naive[0].any() or naive[8].any()


def test_outside_intensity_values_change_with_anisotropy():
    """The reported mean actually moves when the spacing changes."""
    z, y, x = 9, 24, 24
    mask = np.zeros((z, y, x), np.uint16)
    mask[4, 10:14, 10:14] = 1
    image = np.zeros((z, y, x), float)
    image[0] = 100.0
    image[8] = 100.0        # bright only at the top and bottom planes

    isotropic = M._outside_intensity(mask, image, distance=5, spacing=(1.0, 1.0, 1.0))
    anisotropic = M._outside_intensity(mask, image, distance=5, spacing=(5.0, 1.0, 1.0))
    # Isotropic voxels: the shell reaches the bright planes. Anisotropic: it
    # cannot, so the same object reports a different neighbourhood.
    assert isotropic[0][1] > 0
    assert anisotropic[0][1] == pytest.approx(0.0)


def test_radial_distribution_distance_map_is_sampled():
    """The radial bins are cut on a spaced distance map, not on a voxel count."""
    z, y, x = 7, 30, 30
    cell = np.zeros((z, y, x), np.uint16)
    cell[1:6, 5:25, 5:25] = 1
    obj = np.zeros((z, y, x), np.uint16)
    obj[3, 14:17, 14:17] = 1
    chans = _channels((z, y, x), n=1)

    flat = M._calculate_radial_distribution(cell, obj, chans, num_bins=4, spacing=None)
    spaced = M._calculate_radial_distribution(cell, obj, chans, num_bins=4,
                                              spacing=(5.0, 1.0, 1.0))
    key = (1, 1, 0)
    assert key in flat and key in spaced
    assert not np.allclose(flat[key], spaced[key], equal_nan=True), \
        'the sampling= argument had no effect on the radial bins'


def test_measure_intensity_distance_matches_a_hand_computed_edt():
    """The nucleus distance for one cell equals a hand-computed sampled EDT."""
    from scipy.ndimage import distance_transform_edt

    z, y, x = 5, 20, 20
    cell = np.zeros((z, y, x), np.uint16)
    cell[2, 2:5, 2:5] = 1          # odd width, so the centroid is exactly (2, 3, 3)
    nucleus = np.zeros((z, y, x), np.uint16)
    nucleus[2, 14:18, 14:18] = 1
    chans = np.ones((z, y, x, 1), float)

    s = _settings(distance_gaussian_sigma=0, anisotropy=3.0)
    df = M._measure_intensity_distance(cell, nucleus, np.zeros_like(cell), chans, s)

    spacing = (3.0, 1.0, 1.0)
    dt = distance_transform_edt(nucleus == 0, sampling=spacing)
    # The cell's intensity-weighted centroid on a flat image is its centroid.
    expected = dt[2, 3, 3]
    assert df['cell_channel_0_distance_to_nucleus'].iloc[0] == pytest.approx(expected)


# --------------------------------------------------------------------------
# 6. physical volume, and voxels when the size is unknown
# --------------------------------------------------------------------------

def test_known_voxel_size_gives_the_right_physical_volume():
    """A 20x20x4-voxel box at 0.25 x 0.25 x 1.0 um is 25 um^3."""
    cell, nucleus, zero = _masks_3d(z0=1, z1=5)
    s = _settings(voxel_size_z_um=1.0, voxel_size_xy_um=0.25)
    cell_df, *_ = M._morphological_measurements(cell, nucleus, zero, zero, zero, s)

    row = cell_df[cell_df['label'] == 1].iloc[0]
    n_vox = 20 * 20 * 4
    assert row['cell_volume_voxels'] == pytest.approx(n_vox)
    assert row['cell_volume_um3'] == pytest.approx(n_vox * 1.0 * 0.25 * 0.25)
    # The spaced `area` IS the physical volume: same number, same units.
    assert row['cell_area'] == pytest.approx(row['cell_volume_um3'])


def test_unknown_voxel_size_but_known_anisotropy_reports_xy_pixel_units():
    """Anisotropy alone gives correct geometry, and says the units are not um."""
    cell, nucleus, zero = _masks_3d(z0=1, z1=5)
    s = _settings(anisotropy=4.0)
    spacing, stamp = M.resolve_measurement_spacing(s, ndim=3, n_z=6)
    assert spacing == (4.0, 1.0, 1.0)
    assert stamp['measurement_units'] == M.UNITS_PX_XY
    assert stamp['voxel_size_z_um'] is None
    assert stamp['voxel_size_xy_um'] is None

    cell_df, *_ = M._morphological_measurements(cell, nucleus, zero, zero, zero, s)
    row = cell_df[cell_df['label'] == 1].iloc[0]
    # Volume in voxels is reported; there is no um^3 column, because there is
    # no micrometre.
    assert row['cell_volume_voxels'] == pytest.approx(20 * 20 * 4)
    assert 'cell_volume_um3' not in cell_df.columns
    assert row['cell_area'] == pytest.approx(20 * 20 * 4 * 4.0)   # xy-pixel^3


def test_unknown_anisotropy_is_refused_rather_than_assumed_isotropic():
    """No voxel size and no anisotropy: the 3-D run stops and explains."""
    from spacr.zstack import UnknownAnisotropyError

    with pytest.raises(UnknownAnisotropyError) as excinfo:
        M.resolve_measurement_spacing(_settings(), ndim=3, n_z=8)
    message = str(excinfo.value)
    assert 'voxel_size_z_um' in message and 'anisotropy' in message

    cell, nucleus, zero = _masks_3d()
    with pytest.raises(UnknownAnisotropyError):
        M._morphological_measurements(cell, nucleus, zero, zero, zero, _settings())


def test_a_bad_voxel_size_is_rejected():
    from spacr.errors import ConfigurationError

    for bad in ({'voxel_size_z_um': 0, 'voxel_size_xy_um': 0.25},
                {'voxel_size_z_um': -1.0, 'voxel_size_xy_um': 0.25},
                {'voxel_size_z_um': 1.0, 'voxel_size_xy_um': float('nan')}):
        with pytest.raises(ConfigurationError):
            M.resolve_measurement_spacing(_settings(**bad), ndim=3, n_z=4)

    with pytest.raises(ConfigurationError):
        M.resolve_measurement_spacing(_settings(anisotropy=0.0), ndim=3, n_z=4)

    with pytest.raises(ConfigurationError, match='2-D masks and 3-D'):
        M.resolve_measurement_spacing(_settings(), ndim=4, n_z=4)


# --------------------------------------------------------------------------
# 7. the schema decision
# --------------------------------------------------------------------------

def _run_field(tmp_path, volumetric, name='plate1_A01_f1', **extra):
    """Measure one synthetic field into ``tmp_path/measurements/measurements.db``.

    ``measurements/`` is created here because in a real run
    ``_save_settings_to_db`` makes it before the worker pool starts; this helper
    calls ``_measure_crop_core`` directly.
    """
    src = tmp_path / 'merged'
    src.mkdir(parents=True, exist_ok=True)
    (tmp_path / 'measurements').mkdir(parents=True, exist_ok=True)
    if volumetric:
        z, y, x = 4, 40, 40
        data = np.zeros((z, y, x, 6), np.uint16)
        cell, nucleus, _z = _masks_3d((z, y, x), z0=1, z1=3)
        data[..., 4] = cell
        data[..., 5] = nucleus
        data[..., 0] = 11
        data[..., 1] = 22
    else:
        y, x = 40, 40
        data = np.zeros((y, x, 6), np.uint16)
        cell, nucleus, _z = _masks_2d((y, x))
        data[..., 4] = cell
        data[..., 5] = nucleus
        data[..., 0] = 11
        data[..., 1] = 22
    np.save(src / f'{name}.npy', data)

    s = _settings(
        src=str(src), channels=[0, 1], cell_mask_dim=4, nucleus_mask_dim=5,
        pathogen_mask_dim=None, cell_min_size=0, nucleus_min_size=0,
        pathogen_min_size=0, cytoplasm_min_size=0, cytoplasm=False,
        uninfected=True, merge_edge_pathogen_cells=False,
        timelapse=False, timelapse_objects=['cell'], save_measurements=True,
        save_png=False, save_arrays=False, plot=False, verbose=False,
        experiment='exp', homogeneity=False, radial_dist=False,
        calculate_correlation=False)
    s.update(extra)
    return M._measure_crop_core(0, [], f'{name}.npy', s)


def _tiny_frames():
    """A fresh minimal (morph, intensity) pair.

    Built fresh per call because ``spacr.utils._check_integrity`` rewrites the
    frame it is handed in place, so a shared DataFrame cannot be written twice.
    """
    return (pd.DataFrame({'label': [1], 'cell_area': [10.0]}),
            pd.DataFrame({'label': [1], 'cell_channel_0_mean_intensity': [1.0]}))


def _read(tmp_path, table='cell'):
    conn = sqlite3.connect(tmp_path / 'measurements' / 'measurements.db')
    try:
        return pd.read_sql_query(f'SELECT * FROM {table}', conn)
    finally:
        conn.close()


def test_a_3d_row_is_distinguishable_from_a_2d_row(tmp_path):
    """Every row says what it is: ndim, units, plane count and voxel size."""
    two_d = tmp_path / 'flat'
    three_d = tmp_path / 'vol'
    _run_field(two_d, volumetric=False)
    _run_field(three_d, volumetric=True, voxel_size_z_um=1.0, voxel_size_xy_um=0.25)

    flat = _read(two_d)
    vol = _read(three_d)

    for col in ('measurement_ndim', 'measurement_units', 'n_z',
                'voxel_size_z_um', 'voxel_size_xy_um'):
        assert col in flat.columns and col in vol.columns

    assert set(flat['measurement_ndim']) == {2}
    assert set(flat['measurement_units']) == {'px'}
    assert set(flat['n_z']) == {1}
    assert flat['voxel_size_xy_um'].isna().all()

    assert set(vol['measurement_ndim']) == {3}
    assert set(vol['measurement_units']) == {'um'}
    assert set(vol['n_z']) == {4}
    assert set(vol['voxel_size_xy_um']) == {0.25}

    # The same-named column holds different quantities, and the row says so.
    assert 'cell_area' in flat.columns and 'cell_area' in vol.columns
    assert 'cell_volume_voxels' in vol.columns
    assert 'cell_volume_voxels' not in flat.columns


def test_mixing_2d_and_3d_rows_in_one_table_is_refused(tmp_path):
    """A px^2 area and a um^3 volume never land in one ``cell_area`` column."""
    from spacr.utils import MeasurementUnitsMismatch

    _run_field(tmp_path, volumetric=False, name='plate1_A01_f1')
    assert len(_read(tmp_path)) == 2

    # _measure_crop_core catches and ledgers, so call the writer directly to
    # see the refusal, and then confirm the field is reported as failed too.
    from spacr.utils import _merge_and_save_to_database
    morph, intensity = _tiny_frames()
    stamp = {'measurement_ndim': 3, 'measurement_units': 'um', 'n_z': 4,
             'voxel_size_z_um': 1.0, 'voxel_size_xy_um': 0.25}
    with pytest.raises(MeasurementUnitsMismatch, match='refusing to append'):
        _merge_and_save_to_database(
            morph, intensity, 'cell', str(tmp_path), 'plate1_A01_f2', 'exp',
            False, stamp=stamp)

    # Nothing was written: the table still holds only the two 2-D rows.
    after = _read(tmp_path)
    assert len(after) == 2
    assert set(after['measurement_ndim']) == {2}

    # And through the pipeline the 3-D field is recorded as a failure rather
    # than silently contributing rows.
    _index, _avg, cells, _figs = _run_field(
        tmp_path, volumetric=True, name='plate1_A01_f3',
        voxel_size_z_um=1.0, voxel_size_xy_um=0.25)
    assert cells == 0
    assert len(_read(tmp_path)) == 2


def test_a_legacy_unstamped_table_still_accepts_2d_rows(tmp_path):
    """Rows written before the stamp existed are 2-D/px as a matter of fact."""
    from spacr.utils import (_merge_and_save_to_database,
                             _existing_measurement_identity,
                             MeasurementUnitsMismatch)

    (tmp_path / 'measurements').mkdir(parents=True, exist_ok=True)
    # stamp=None writes no stamp columns at all -- the pre-3-D schema.
    _merge_and_save_to_database(*_tiny_frames(), 'cell', str(tmp_path),
                                'plate1_A01_f1', 'exp', False)
    db = str(tmp_path / 'measurements' / 'measurements.db')
    assert _existing_measurement_identity(db, 'cell') == {(2, 'px')}
    assert 'measurement_ndim' not in _read(tmp_path).columns

    # A stamped 2-D append is compatible with it ...
    _merge_and_save_to_database(
        *_tiny_frames(), 'cell', str(tmp_path), 'plate1_A01_f2', 'exp', False,
        stamp={'measurement_ndim': 2, 'measurement_units': 'px', 'n_z': 1,
               'voxel_size_z_um': None, 'voxel_size_xy_um': None})
    assert len(_read(tmp_path)) == 2

    # ... and a 3-D one is not, even though the legacy rows carry no stamp.
    with pytest.raises(MeasurementUnitsMismatch):
        _merge_and_save_to_database(
            *_tiny_frames(), 'cell', str(tmp_path), 'plate1_A01_f3', 'exp',
            False, stamp={'measurement_ndim': 3, 'measurement_units': 'um',
                          'n_z': 4, 'voxel_size_z_um': 1.0,
                          'voxel_size_xy_um': 0.25})


def test_units_differing_only_in_the_unit_string_are_also_refused(tmp_path):
    """um^3 volumes and xy-pixel^3 volumes are both 3-D, and still not mixable."""
    from spacr.utils import _merge_and_save_to_database, MeasurementUnitsMismatch

    (tmp_path / 'measurements').mkdir(parents=True, exist_ok=True)
    base = {'measurement_ndim': 3, 'n_z': 4}
    _merge_and_save_to_database(
        *_tiny_frames(), 'cell', str(tmp_path), 'plate1_A01_f1', 'exp', False,
        stamp={**base, 'measurement_units': 'um',
               'voxel_size_z_um': 1.0, 'voxel_size_xy_um': 0.25})
    with pytest.raises(MeasurementUnitsMismatch, match='3-D/um'):
        _merge_and_save_to_database(
            *_tiny_frames(), 'cell', str(tmp_path), 'plate1_A01_f2', 'exp', False,
            stamp={**base, 'measurement_units': 'px_xy',
                   'voxel_size_z_um': None, 'voxel_size_xy_um': None})


# --------------------------------------------------------------------------
# 8. a single-plane volume is the 2-D path
# --------------------------------------------------------------------------

def test_single_plane_volume_measures_identically_to_the_2d_path(tmp_path):
    """``(1, Y, X, C)`` is a 2-D field, measured as one -- no anisotropy needed.

    Same field, saved both ways; every numeric column must match exactly, and
    the row must be stamped 2-D/px, because it *is* a 2-D measurement.
    """
    flat_dir = tmp_path / 'flat'
    vol_dir = tmp_path / 'vol'
    for d in (flat_dir, vol_dir):
        (d / 'merged').mkdir(parents=True)
        (d / 'measurements').mkdir(parents=True)

    y, x = 40, 40
    field = np.zeros((y, x, 6), np.uint16)
    cell, nucleus, _z = _masks_2d((y, x))
    field[..., 4] = cell
    field[..., 5] = nucleus
    rng = np.random.RandomState(2)
    field[..., 0] = rng.randint(0, 500, (y, x))
    field[..., 1] = rng.randint(0, 500, (y, x))

    np.save(flat_dir / 'merged' / 'plate1_A01_f1.npy', field)
    np.save(vol_dir / 'merged' / 'plate1_A01_f1.npy', field[None, ...])

    common = dict(
        channels=[0, 1], cell_mask_dim=4, nucleus_mask_dim=5,
        pathogen_mask_dim=None, cell_min_size=0, nucleus_min_size=0,
        pathogen_min_size=0, cytoplasm_min_size=0, cytoplasm=False,
        uninfected=True, merge_edge_pathogen_cells=False, timelapse=False,
        timelapse_objects=['cell'], save_measurements=True, save_png=False,
        save_arrays=False, plot=False, verbose=False, experiment='exp',
        homogeneity=True, radial_dist=True, calculate_correlation=True)

    for d in (flat_dir, vol_dir):
        s = _settings(src=str(d / 'merged'), **common)
        _i, _a, cells, _f = M._measure_crop_core(0, [], 'plate1_A01_f1.npy', s)
        assert not isinstance(cells, int), f'{d} failed inside _measure_crop_core'

    flat = _read(flat_dir).sort_values('object_label').reset_index(drop=True)
    vol = _read(vol_dir).sort_values('object_label').reset_index(drop=True)

    assert list(flat.columns) == list(vol.columns)
    assert set(vol['measurement_ndim']) == {2}
    assert set(vol['n_z']) == {1}

    numeric = flat.select_dtypes(include=[np.number]).columns
    assert len(numeric) > 40, 'expected a full measurement frame'
    for col in numeric:
        np.testing.assert_allclose(
            flat[col].to_numpy(dtype=float), vol[col].to_numpy(dtype=float),
            rtol=0, atol=0, equal_nan=True,
            err_msg=f'{col} differs between the flat and one-plane runs')


# --------------------------------------------------------------------------
# 9. the 2-D-only crop path is refused, not approximated
# --------------------------------------------------------------------------

def test_3d_field_writes_measurements_but_refuses_crops(tmp_path, capsys):
    _index, _avg, cells, _figs = _run_field(
        tmp_path, volumetric=True, voxel_size_z_um=1.0, voxel_size_xy_um=0.25,
        save_png=True, png_size=[32, 32], png_dims=[0, 1, 2],
        normalize=False, normalize_by='png', crop_mode=['cell'],
        dialate_pngs=False, dialate_png_ratios=[0.2], use_bounding_box=False)
    assert not isinstance(cells, int)
    assert len(_read(tmp_path)) == 2

    out = capsys.readouterr().out
    assert 'no PNG' in out
    assert not list((tmp_path / 'merged').glob('**/*.png'))


def test_summarize_organelles_drops_eccentricity_in_3d():
    z, y, x = 5, 30, 30
    parent = np.zeros((z, y, x), np.uint16)
    parent[1:4, 5:25, 5:25] = 1
    organelle = np.zeros((z, y, x), np.uint16)
    organelle[2, 8:12, 8:12] = 1
    organelle[2, 16:20, 16:20] = 2
    chans = _channels((z, y, x), n=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = M._summarize_organelles_per_parent(
            organelle, parent, chans, parent_name='cell',
            spacing=(4.0, 1.0, 1.0))
    assert df['organelle_count'].iloc[0] == 2
    assert 'organelle_mean_eccentricity' not in df.columns
    assert 'organelle_mean_solidity' in df.columns
    assert np.isnan(df['organelle_mean_solidity'].iloc[0])
    assert 0 < df['organelle_fraction'].iloc[0] < 1
    assert not [warning for warning in caught
                if issubclass(warning.category, (RuntimeWarning, UserWarning))]

    # 2-D keeps it.
    flat = M._summarize_organelles_per_parent(
        organelle[2], parent[2], chans[2], parent_name='cell')
    assert 'organelle_mean_eccentricity' in flat.columns


def test_flat_3d_morphology_marks_convex_geometry_undefined_without_warning():
    """A one-plane label is valid 3-D input but has no convex *volume*."""
    cell = np.zeros((3, 12, 12), np.uint16)
    cell[1, 3:9, 3:9] = 1
    zero = np.zeros_like(cell)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cell_df, _n, _p, _o, _c = M._morphological_measurements(
            cell, zero, zero, zero, zero,
            _settings(cell_mask_dim=0, nucleus_mask_dim=None, anisotropy=1.0),
            zernike=False)

    assert np.isnan(cell_df['cell_convex_area'].iloc[0])
    assert np.isnan(cell_df['cell_solidity'].iloc[0])
    assert np.isnan(cell_df['cell_feret_diameter_max'].iloc[0])
    assert not [warning for warning in caught
                if issubclass(warning.category, (RuntimeWarning, UserWarning))]


def test_generate_object_dataset_refuses_a_volumetric_array(tmp_path):
    """Cropping a 4-D array with 2-D indexing would return a slab of X."""
    src = tmp_path
    (src / 'merged').mkdir(parents=True)
    (src / 'measurements').mkdir(parents=True)
    arr_path = src / 'merged' / 'plate1_A01_f1.npy'
    np.save(arr_path, np.zeros((3, 20, 20, 6), np.uint16))

    db = src / 'measurements' / 'measurements.db'
    conn = sqlite3.connect(db)
    try:
        pd.DataFrame({
            'object_label': [1], 'path_name': [str(arr_path)],
            'plateID': ['plate1'], 'rowID': ['r1'], 'columnID': ['c1'],
            'fieldID': ['f1'], 'cell_area': [100.0],
        }).to_sql('cell', conn, index=False)
    finally:
        conn.close()

    with pytest.raises(ValueError, match='crops 2-D merged arrays'):
        M.generate_object_dataset(str(src), object_type='cell', channels=(0, 1, 2),
                                  save_png=False, verbose=False)
