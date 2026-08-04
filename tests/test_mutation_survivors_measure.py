"""Gaps in the measure suite found by AST mutation testing of ``spacr.measure``.

Every test below was written against a specific *surviving mutant*: a
one-token change to ``spacr/measure.py`` that the whole targeted measure
suite (15 modules, ~700 tests) ran green over. The mutated line is named in
each docstring together with what the mutation silently changed, so the test
records the hole it fills rather than just the behaviour it asserts.

Each was verified twice -- red with the mutant loaded, green without it.

The recurring shape is worth naming, because it is not "a line nobody runs":
every line below is executed by the existing suite. What is missing is an
*oracle*. A default argument that only ever appears as ``f(x)`` at the call
site is executed on every run and asserted by nobody, and a test that phrases
its expectation as ``cores - M.N_JOBS_HEADROOM`` re-derives the answer from
the thing it is testing, so the number can move without the test moving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import measure as M


def _two_object_mask(size=32):
    m = np.zeros((size, size), dtype=np.int32)
    m[2:10, 2:10] = 1
    m[16:26, 16:26] = 2
    return m


def _nested_masks(size=32):
    """cell / nucleus / pathogen / organelle / cytoplasm co-registered masks."""
    cell = np.zeros((size, size), dtype=np.int32)
    cell[4:28, 4:28] = 1
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[8:14, 8:14] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[18:22, 18:22] = 1
    organelle = np.zeros((size, size), dtype=np.int32)
    organelle[10:12, 20:22] = 1
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    return cell, nucleus, pathogen, organelle, cytoplasm


# ---------------------------------------------------------------------------
# measure.py:122  N_JOBS_HEADROOM = 4
# ---------------------------------------------------------------------------

def test_the_worker_headroom_is_four_cores_and_not_merely_self_consistent():
    """``N_JOBS_HEADROOM = 4`` (measure.py:122).

    Mutant ``N_JOBS_HEADROOM = 5`` survived the entire measure suite.
    ``tests/test_measure_n_jobs.py`` phrases every expectation as
    ``cores - M.N_JOBS_HEADROOM``, which re-derives the answer from the
    constant under test: the pool can quietly shrink (or, mutated the other
    way, grow past the headroom the docstring promises an interactive
    machine) and every assertion still holds. The number is a user-facing
    promise -- "spaCR leaves you four cores" -- so pin the number.
    """
    assert M.N_JOBS_HEADROOM == 4
    assert M.resolve_n_jobs(None, cpu_count=32) == 28
    assert M.resolve_n_jobs(None, cpu_count=8) == 4
    # The floor still holds on a machine smaller than the headroom.
    assert M.resolve_n_jobs(None, cpu_count=4) == 1


# ---------------------------------------------------------------------------
# measure.py:1498  _outside_intensity(..., distance=5, ...)
# ---------------------------------------------------------------------------

def test_the_outside_intensity_ring_is_five_pixels_wide_by_default():
    """``def _outside_intensity(label_mask, image, distance=5, ...)``
    (measure.py:1498).

    Mutant ``distance=6`` survived. The only production call site,
    measure.py:1216, passes no ``distance``, so this default is the live
    width of the ring behind every ``outside_*`` intensity column spaCR
    writes for nuclei, pathogens and organelles -- widen it and every one of
    those numbers changes, in a direction nothing in the suite looks at.

    The oracle is exact rather than approximate: with the intensity image set
    to the L1 distance from a single-pixel object, ``binary_dilation`` grows a
    diamond, ``4k`` pixels sit at distance ``k``, and the ring mean is a
    closed form in the ring width.
    """
    size = 41
    centre = size // 2
    mask = np.zeros((size, size), dtype=np.int32)
    mask[centre, centre] = 1
    yy, xx = np.mgrid[0:size, 0:size]
    image = (np.abs(yy - centre) + np.abs(xx - centre)).astype(np.float64)

    stats = M._outside_intensity(mask, image)

    assert len(stats) == 1
    label, mean = stats[0][0], stats[0][1]
    assert label == 1
    width = 5
    expected = (sum(4 * k * k for k in range(1, width + 1))
                / sum(4 * k for k in range(1, width + 1)))
    assert mean == pytest.approx(expected)
    # The p95 of the ring reaches the outermost shell, so the test also fails
    # if the ring is narrowed rather than widened.
    assert stats[0][8] == pytest.approx(np.percentile(
        image[(image >= 1) & (image <= width)], 95))


# ---------------------------------------------------------------------------
# measure.py:576 and measure.py:833  degree=8
# ---------------------------------------------------------------------------

def test_the_default_zernike_degree_yields_the_documented_column_count():
    """``def _calculate_zernike(mask, df, degree=8)`` (measure.py:576).

    Mutant ``degree=9`` survived. ``tests/test_coverage_fill_measure_object``
    calls this with an explicit ``degree=8`` and then asserts only that *some*
    column starts with ``zernike_`` -- which is true of every degree. The
    docstring states the contract exactly ("9 for 4, 25 for 8, 49 for 12"), so
    assert it: the degree sets the WIDTH of the morphology table, and a
    silently wider table is a silently different measurement schema.
    """
    pytest.importorskip(
        "mahotas",
        reason="numerical Zernike descriptors require the optional spacr[zernike] extra",
    )
    out = M._calculate_zernike(_two_object_mask(), pd.DataFrame({"label": [1, 2]}))
    zernike_columns = [c for c in out.columns if c.startswith("zernike_")]
    assert len(zernike_columns) == 25
    assert zernike_columns[-1] == "zernike_24"


def test_morphology_defaults_to_the_documented_zernike_degree():
    """``def _morphological_measurements(..., degree=8)`` (measure.py:833).

    Mutant ``degree=9`` survived. measure.py:2375 -- the real measure run --
    calls this with neither ``zernike`` nor ``degree``, so this default is
    what every crop measurement in production actually uses, and it decides
    how many ``zernike_*`` columns land in the cell table.
    """
    pytest.importorskip(
        "mahotas",
        reason="numerical Zernike descriptors require the optional spacr[zernike] extra",
    )
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": 2, "organelle_mask_dim": 3,
        "cytoplasm": True,
    }
    cell_df = M._morphological_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm, settings,
        zernike=True)[0]
    zernike_columns = [c for c in cell_df.columns if c.startswith("cell_zernike_")]
    assert len(zernike_columns) == 25
    assert "cell_zernike_24" in zernike_columns
    assert "cell_zernike_25" not in zernike_columns


# ---------------------------------------------------------------------------
# measure.py:1930  stamp_crop_folder(os.path.dirname(img_path))
# ---------------------------------------------------------------------------

def test_writing_a_crop_marks_the_folder_as_current_format(tmp_path):
    """``stamp_crop_folder(os.path.dirname(img_path))`` (measure.py:1930).

    Deleting this statement outright left the whole measure suite green. The
    existing test asserts the PNG exists, and the crop-format suite tests
    ``stamp_crop_folder`` on its own -- nothing joined the two, so the only
    place the stamp is actually written on the measure path was unasserted.

    The consequence of the missing stamp is silent and wrong, not loud: an
    unmarked folder means *legacy* by definition, so
    ``spacr.crops.read_crop_png`` would channel-swap every correctly written
    crop on load. That is the exact defect the sidecar exists to prevent.
    """
    from spacr.crops import (
        CROP_FORMAT_CURRENT,
        CROP_FORMAT_SIDECAR,
        clear_crop_format_cache,
        read_crop_folder_marker,
    )

    clear_crop_format_cache()
    folder = tmp_path / "crops"
    folder.mkdir()
    png = np.zeros((8, 8, 3), dtype=np.uint16)

    M.save_and_add_image_to_grid(png, str(folder / "a.png"), [])

    assert (folder / "a.png").exists()
    assert (folder / CROP_FORMAT_SIDECAR).exists()
    marker = read_crop_folder_marker(str(folder), use_cache=False)
    assert marker is not None
    assert marker["spacr_crop_format"] == CROP_FORMAT_CURRENT
    clear_crop_format_cache()


# ---------------------------------------------------------------------------
# measure.py:1435  percentiles = [5, 10, 25, 75, 85, 95]
# ---------------------------------------------------------------------------

def test_the_intensity_percentile_columns_are_the_documented_six():
    """``percentiles = [5, 10, 25, 75, 85, 95]`` (measure.py:1435).

    Three mutants survived here -- ``5 -> 6``, ``10 -> 11`` and ``25 -> 26``.
    Each renames a column AND changes its value, i.e. it silently alters the
    measurement schema every spaCR database is written with: downstream code
    asking for ``percentile_5`` would find nothing, and a database written
    before and after the change would have incomparable columns. Every
    existing caller of ``_extended_regionprops_table`` asserts on the frame's
    shape or on one named statistic, never on the percentile set.
    """
    mask = _two_object_mask()
    rng = np.random.default_rng(4)
    image = (rng.random(mask.shape) * 1000).astype(np.float64)

    df = M._extended_regionprops_table(mask, image, ["label", "mean_intensity"])

    got = sorted(int(c.split("_")[1]) for c in df.columns
                 if c.startswith("percentile_"))
    assert got == [5, 10, 25, 75, 85, 95]

    # ...and each column really holds that percentile of its own object.
    first_object = image[mask == 1]
    for p in got:
        assert df[f"percentile_{p}"].iloc[0] == pytest.approx(
            np.percentile(first_object, p))


# ---------------------------------------------------------------------------
# measure.py:468  if dz is not None and dxy is not None
# ---------------------------------------------------------------------------

def test_micrometre_units_need_both_voxel_sizes_not_either_one():
    """``if dz is not None and dxy is not None:`` (measure.py:468).

    Mutant ``or`` survived. With ``or``, configuring only ``voxel_size_z_um``
    stamps the whole measurement ``measurement_units='um'`` and returns the
    spacing tuple ``(dz, None, None)`` -- a half-configured run that claims
    micrometres and hands ``None`` to ``regionprops_table`` as the xy
    sampling. The original refuses a half-configured 3-D run outright, which
    is the whole point of ``UnknownAnisotropyError``; nothing asserted that
    supplying exactly ONE of the two sizes still counts as unconfigured.
    """
    from spacr.zstack import UnknownAnisotropyError

    with pytest.raises(UnknownAnisotropyError):
        M.resolve_measurement_spacing(
            {"voxel_size_z_um": 2.0, "voxel_size_xy_um": None}, 3)
    with pytest.raises(UnknownAnisotropyError):
        M.resolve_measurement_spacing(
            {"voxel_size_z_um": None, "voxel_size_xy_um": 0.5}, 3)

    spacing, stamp = M.resolve_measurement_spacing(
        {"voxel_size_z_um": 2.0, "voxel_size_xy_um": 0.5}, 3)
    assert spacing == (2.0, 0.5, 0.5)
    assert stamp["measurement_units"] == M.UNITS_UM


# ---------------------------------------------------------------------------
# measure.py:332 and measure.py:365  max(1, ...)
# ---------------------------------------------------------------------------

def test_the_pool_never_shrinks_to_zero_workers():
    """``n_jobs = max(1, int(n_jobs))`` (measure.py:332) and
    ``cores = max(1, int(...))`` (measure.py:365).

    Both mutants ``max(0, ...)`` survived. The floor is not decorative:
    ``multiprocessing.Pool(0)`` raises ``ValueError: Number of processes must
    be at least 1``, so a zero here turns a measure run into a crash at pool
    construction -- the very failure the ``max(1, ...)`` was written for. The
    existing suite only ever exercises these two lines with values already
    above the floor.
    """
    assert M.resolve_pool_size(0, 10, start_method="spawn") == 1
    assert M.resolve_pool_size(-3, 10, start_method="spawn") == 1
    assert M.resolve_pool_size(0, 10, start_method="fork") == 1
    # cpu_count=0 is what a container with an unreadable cgroup reports.
    assert M.resolve_n_jobs(4, cpu_count=0) == 1
    assert M.resolve_n_jobs(None, cpu_count=0) == 1


# ---------------------------------------------------------------------------
# measure.py:2107  factor = 1.0
# ---------------------------------------------------------------------------

def test_promoting_ordinary_uint16_data_leaves_the_intensities_alone():
    """``factor = 1.0`` (measure.py:2107).

    Mutant ``factor = 0`` survived. ``factor`` is the shared rescale applied
    to every intensity plane, and ``1.0`` is the "nothing to do" case -- which
    is the ordinary path for any array already inside the 16-bit range. With
    ``0`` the function multiplies every intensity plane by zero and returns a
    black field, and the docstring's promise that "that path keeps behaving
    exactly as it did" is silently broken. Nothing asserted the no-op.
    """
    arr = np.zeros((8, 8, 2), dtype=np.uint16)
    arr[..., 0] = 3                       # label plane
    arr[2:6, 2:6, 1] = 1000               # intensity plane, inside uint16
    settings = {"cell_mask_dim": 0}

    out, factor = M._promote_merged_to_uint16(arr, settings)

    assert factor == 1.0
    assert out.dtype == np.uint16
    assert int(out[..., 1].max()) == 1000
    np.testing.assert_array_equal(out[..., 1], arr[..., 1])


# ---------------------------------------------------------------------------
# measure.py:1521  ring_width = float(distance) * float(spacing[-1])
# ---------------------------------------------------------------------------

def test_the_three_d_outside_ring_is_measured_in_micrometres_not_pixels():
    """``ring_width = float(distance) * float(spacing[-1])`` (measure.py:1521).

    Mutant ``/`` survived. This is the 3-D branch that exists precisely so the
    ring means ``distance`` xy pixels of PHYSICAL length on every axis; with
    ``/`` and a sub-micron ``dxy`` the ring blows up (here 5 px -> 10 um
    instead of 2.5 um) and 'outside intensity' becomes a measurement of
    whatever sits far away. The assertion is stated as physical extent, not as
    the formula: a voxel 3.0 um out must stay OUT, one 2.0 um out must be IN.
    """
    spacing = (2.0, 0.5, 0.5)             # dz, dxy, dxy in um
    shape = (5, 21, 21)
    mask = np.zeros(shape, dtype=np.int32)
    mask[2, 10, 10] = 1

    far = np.zeros(shape, dtype=np.float64)
    far[2, 10, 16] = 1.0                  # 6 px * 0.5 um = 3.0 um out
    assert M._outside_intensity(mask, far, spacing=spacing)[0][1] == 0.0

    near = np.zeros(shape, dtype=np.float64)
    near[2, 10, 14] = 1.0                 # 4 px * 0.5 um = 2.0 um out
    assert M._outside_intensity(mask, near, spacing=spacing)[0][1] > 0.0


# ---------------------------------------------------------------------------
# measure.py:2038, 2040, 2046  the _per_crop_mode length comparisons
# ---------------------------------------------------------------------------

def test_a_per_crop_setting_broadcasts_silently_and_complains_only_when_wrong(capsys):
    """``len(values) == 1`` / ``< n_modes`` / ``> n_modes`` (measure.py:2038,
    2040, 2046).

    Three mutants survived: ``== 0``, ``<=`` and ``>=``. All three leave the
    RETURNED list identical and change only whether ``_per_crop_mode`` prints
    at the exact-length boundary -- which is why the existing broadcast tests,
    which assert the list, cannot see them. The docstring makes the noise the
    contract, not a detail: a single value "is broadcast silently, which is
    what png_size has always done", while a short list is "said out loud,
    because losing every crop on a 1000-field plate to a typo'd list is worse
    than cropping two modes at the same ratio". A run that warns about a
    perfectly good setting teaches users to ignore the warning that matters.
    """
    assert M._per_crop_mode(0.2, 3, 'dialate_png_ratios') == [0.2, 0.2, 0.2]
    assert M._per_crop_mode([0.2], 3, 'dialate_png_ratios') == [0.2, 0.2, 0.2]
    assert capsys.readouterr().out == ''          # kills `len(values) == 0`

    exact = M._per_crop_mode([0.2, 0.3], 2, 'dialate_png_ratios')
    assert exact == [0.2, 0.3]
    assert capsys.readouterr().out == ''          # kills `<=` and `>=`

    short = M._per_crop_mode([0.2, 0.3], 4, 'dialate_png_ratios')
    assert short == [0.2, 0.3, 0.3, 0.3]
    assert 'reusing' in capsys.readouterr().out

    long = M._per_crop_mode([0.2, 0.3, 0.4], 2, 'dialate_png_ratios')
    assert long == [0.2, 0.3]
    assert 'ignoring the extra' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# measure.py:1807  sigma = settings.get('distance_gaussian_sigma', 1.0)
# ---------------------------------------------------------------------------

def test_the_distance_smoothing_defaults_to_sigma_one_not_to_no_smoothing():
    """``sigma = settings.get('distance_gaussian_sigma', 1.0)``
    (measure.py:1807).

    Mutant ``0`` survived. Zero sigma is not "a slightly different default",
    it is NO smoothing: the intensity-weighted centroid then sits on whatever
    single brightest pixel the channel happens to have, which is what the
    Gaussian is there to stop. Every caller in the suite either supplies the
    setting or never looks at the value, so the fallback was unasserted.

    The oracle is a comparison rather than a literal, because the reported
    distances are read at a rounded voxel index: measuring with the setting
    ABSENT must equal measuring with it set to 1.0, and must differ from
    measuring with it set to 0.
    """
    size = 40
    cell = np.zeros((size, size), dtype=np.int32)
    cell[10:19, 10:19] = 1
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[0:3, 0:3] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[36:39, 36:39] = 1
    channels = np.zeros((size, size, 1), dtype=np.float32)
    channels[10, 10, 0] = 1000.0          # one bright pixel on the cell corner

    def distances(settings):
        df = M._measure_intensity_distance(
            cell, nucleus, pathogen, channels, settings)
        return (float(df['cell_channel_0_distance_to_nucleus'].iloc[0]),
                float(df['cell_channel_0_distance_to_pathogen'].iloc[0]))

    default = distances({})
    explicit_one = distances({'distance_gaussian_sigma': 1.0})
    no_smoothing = distances({'distance_gaussian_sigma': 0})

    assert default == pytest.approx(explicit_one)
    assert default != pytest.approx(no_smoothing)


def test_the_distance_columns_measure_distance_to_the_object_not_from_it():
    """``nucleus_dt = distance_transform_edt(nucleus_mask == 0, ...)`` and the
    pathogen line beside it (measure.py:1823-1824).

    Mutant ``pathogen_mask == 1`` survived. ``distance_transform_edt`` measures
    distance to the nearest ZERO of what it is given, so passing
    ``mask == 0`` measures "how far to the pathogen" and passing ``mask == 1``
    measures "how far out of the pathogen" -- a different quantity entirely,
    written into the same column. Every existing caller checks the column
    NAMES and the frame shape; none checks a distance.

    With a uniform channel the intensity-weighted centroid of a square cell is
    its centre, so both distances are exact Euclidean lengths and can be
    written down.
    """
    size = 40
    cell = np.zeros((size, size), dtype=np.int32)
    cell[10:19, 10:19] = 1                # centre at (14, 14)
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[0:3, 0:3] = 1                 # nearest pixel (2, 2)
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[36:39, 36:39] = 1            # nearest pixel (36, 36)
    channels = np.ones((size, size, 1), dtype=np.float32)

    df = M._measure_intensity_distance(
        cell, nucleus, pathogen, channels, {})

    assert float(df['cell_channel_0_distance_to_nucleus'].iloc[0]) == \
        pytest.approx(float(np.hypot(14 - 2, 14 - 2)))
    assert float(df['cell_channel_0_distance_to_pathogen'].iloc[0]) == \
        pytest.approx(float(np.hypot(36 - 14, 36 - 14)))


# ---------------------------------------------------------------------------
# measure.py:1766  volumetric = np.asarray(image).ndim == 3
# ---------------------------------------------------------------------------

def test_blur_on_a_maskless_volume_is_measured_plane_by_plane():
    """``volumetric = np.asarray(image).ndim == 3`` (measure.py:1766).

    Mutant ``== 4`` survived. This is the maskless arm of ``_estimate_blur``
    (the whole-field QC path). Getting it wrong sends a ``(Z, Y, X)`` volume
    into a single ``cv2.Laplacian`` call, which reads the z axis as CHANNELS
    -- the exact "focus is defined in the xy plane" mistake the volumetric
    branch exists to avoid. Only the masked arm was exercised in 3-D.
    """
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(11)
    volume = rng.random((4, 16, 16)).astype(np.float64)

    got = M._estimate_blur(volume)

    per_plane = np.stack([cv2.Laplacian(np.ascontiguousarray(volume[z]),
                                        cv2.CV_64F)
                          for z in range(volume.shape[0])])
    assert got == pytest.approx(per_plane.var())


# ---------------------------------------------------------------------------
# measure.py:1540  _calculate_radial_distribution(..., num_bins=6, ...)
# ---------------------------------------------------------------------------

def test_the_radial_distribution_defaults_to_six_bins():
    """``def _calculate_radial_distribution(..., num_bins=6, ...)``
    (measure.py:1540).

    Mutant ``num_bins=7`` survived. The bin count is the column count --
    ``_create_dataframe`` emits one ``<obj>_rad_dist_channel_<c>_bin_<i>``
    per bin -- so the default is part of the measurement schema for any caller
    that does not override it.

    Lower-value than the other default-argument survivors and recorded as
    such: all three production call sites (measure.py:1241/1246/1251) pass
    ``num_bins=6`` explicitly, so the default is currently reachable only from
    direct callers. It is still the documented default ("Defaults to 6") and
    the free way to keep it honest.
    """
    size = 32
    cell = np.zeros((size, size), dtype=np.int32)
    cell[4:28, 4:28] = 1
    obj = np.zeros((size, size), dtype=np.int32)
    obj[10:16, 10:16] = 1
    channels = np.zeros((size, size, 1), dtype=np.float32)
    channels[..., 0] = np.random.default_rng(5).random((size, size))

    distributions = M._calculate_radial_distribution(cell, obj, channels)

    assert distributions
    for profile in distributions.values():
        assert len(profile) == 6


# ---------------------------------------------------------------------------
# measure.py:384  if n_jobs > cores
# ---------------------------------------------------------------------------

def test_asking_for_exactly_the_core_count_is_not_an_overshoot(capsys):
    """``if n_jobs > cores:`` (measure.py:384).

    Mutant ``>=`` survived. It returns the same number, so only the console
    says anything -- and what it says at the boundary is false: "n_jobs=8
    exceeds the 8 available cores". ``resolve_n_jobs`` exists because spaCR
    used to throw the user's request away silently, and the fix's whole point
    is that the user is told when, and only when, their value is changed.
    The existing boundary test uses ``n_jobs`` strictly above the core count,
    so ``>`` and ``>=`` are indistinguishable to it.
    """
    assert M.resolve_n_jobs(8, cpu_count=8) == 8
    assert capsys.readouterr().out == ''

    assert M.resolve_n_jobs(9, cpu_count=8) == 8
    assert 'exceeds the 8 available cores' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# measure.py:1105-1121  the per-parent organelle summary's empty-parent row
# ---------------------------------------------------------------------------

def test_a_parent_with_no_organelles_summarises_to_zero_not_to_one():
    """The ``... if len(org_subset) > 0 else 0.0`` fallbacks
    (measure.py:1109-1121) and ``parent_area_map.get(pid, 1)``
    (measure.py:1105).

    NINE mutants survived in this one block, every one of them turning a
    zero fallback into a one. That is not nine small holes, it is one big
    one: no test in the suite ever passes a parent that owns NO organelles,
    so the row spaCR writes for an empty cell -- which is most cells on a
    sparse plate -- is completely unasserted. The existing test asserts only
    ``isinstance(df, pd.DataFrame)``.

    The values matter downstream: ``organelle_fraction`` of 1.0 instead of
    0.0 says an empty cell is entirely filled by organelles, and
    ``organelle_mean_area`` of 1 instead of 0 shifts every plate-level mean.
    """
    size = 32
    parent = np.zeros((size, size), dtype=np.int32)
    parent[2:12, 2:12] = 1               # cell 1 -- owns an organelle
    parent[18:28, 18:28] = 2             # cell 2 -- owns nothing
    organelle = np.zeros((size, size), dtype=np.int32)
    organelle[4:8, 4:8] = 1              # 16 px, inside cell 1 only
    channels = np.ones((size, size, 1), dtype=np.float32)

    df = M._summarize_organelles_per_parent(
        organelle, parent, channels, parent_name='cell')

    by_label = df.set_index('label')
    assert sorted(by_label.index) == [1, 2]

    full = by_label.loc[1]
    assert full['organelle_count'] == 1
    assert full['organelle_total_area'] == pytest.approx(16.0)
    assert full['organelle_fraction'] == pytest.approx(16.0 / 100.0)

    empty = by_label.loc[2]
    assert empty['organelle_count'] == 0
    assert empty['organelle_total_area'] == 0
    assert empty['organelle_fraction'] == 0.0
    assert empty['organelle_mean_area'] == 0.0
    assert empty['organelle_std_area'] == 0.0
    assert empty['organelle_mean_solidity'] == 0.0
    assert empty['organelle_std_solidity'] == 0.0
    assert empty['organelle_mean_major_axis'] == 0.0
    assert empty['organelle_mean_minor_axis'] == 0.0
    assert empty['organelle_channel_0_mean_intensity_per_cell'] == 0.0
    assert empty['organelle_channel_0_std_intensity_per_cell'] == 0.0


# ---------------------------------------------------------------------------
# measure.py:1328  the Gini coefficient
# ---------------------------------------------------------------------------

def test_the_gini_intensity_feature_is_actually_a_gini_coefficient():
    """``np.sum((2 * index - n - 1) * array) / (n * np.sum(array))``
    (measure.py:1328).

    Mutant ``/ -> *`` survived: ``gini_intensity`` is written into every
    measurement database and no test has ever checked a value of it, only
    that the column exists. Multiplying instead of dividing turns a bounded
    inequality index on [0, 1] into an unbounded number that scales with the
    square of the object's brightness -- a feature that would still look
    plausible in a UMAP and be silently wrong in every model trained on it.

    The oracle is closed form: for pixel values 1, 2, 3, 4 the Gini
    coefficient is 10 / 40 = 0.25, and for a uniform object it is exactly 0.
    """
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[2:4, 2:4] = 1                    # object 1: four pixels
    mask[8:10, 8:10] = 2                  # object 2: four pixels, uniform
    image = np.zeros((12, 12), dtype=np.float64)
    image[2:4, 2:4] = np.array([[1.0, 2.0], [3.0, 4.0]])
    image[8:10, 8:10] = 7.0

    df = M._extended_regionprops_table(mask, image, ["label", "mean_intensity"])
    by_label = df.set_index("label")

    assert by_label.loc[1, "gini_intensity"] == pytest.approx(0.25)
    assert by_label.loc[2, "gini_intensity"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# measure.py:998 (and the ls.append calls at 897/927/954/990)
# ---------------------------------------------------------------------------

def test_the_five_morphology_frames_come_back_in_their_declared_slots():
    """``return df_ls[0], df_ls[1], df_ls[2], df_ls[3], df_ls[4]``
    (measure.py:998), and the ``ls.append(<object>)`` calls that name them
    (measure.py:897, 927, 954, 990).

    Mutant ``return ..., df_ls[4], df_ls[4]`` survived -- the ORGANELLE table
    replaced by a second copy of the CYTOPLASM table. So did deleting each
    ``ls.append``, which shifts the label list that prefixes every column.
    The caller at measure.py:2375 unpacks this tuple positionally straight
    into ``cell_df, nucleus_df, pathogen_df, organelle_df, cytoplasm_df``, so
    a slot swap writes one object's morphology into another object's table
    for the rest of the run, and the only existing assertion on this function
    is ``isinstance(out, tuple) and len(out) == 5``.

    Column prefixes are the identity check: each frame must carry its own
    object's name and no other's.
    """
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": 2, "organelle_mask_dim": 3,
        "cytoplasm": True,
    }

    frames = M._morphological_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm, settings, zernike=False)

    assert len(frames) == 5
    names = ("cell", "nucleus", "pathogen", "organelle", "cytoplasm")
    for frame, name in zip(frames, names):
        prefixed = [c for c in frame.columns if c != "label"]
        assert prefixed, f"{name} frame carried no measurements"
        assert all(c.startswith(f"{name}_") for c in prefixed), (
            f"{name} slot holds {sorted(prefixed)[:3]}")
        assert f"{name}_area" in frame.columns


@pytest.mark.parametrize("disabled", ["cell", "nucleus", "pathogen",
                                      "organelle", "cytoplasm"])
def test_a_disabled_object_still_holds_its_slot_in_the_morphology_tuple(disabled):
    """The ``else: ... ls.append(<object>)`` arms (measure.py:900, 930, 957,
    993 and their siblings).

    Deleting the ``ls.append`` on the NOT-MEASURED arm of each object
    survived. ``ls`` is the list the column prefixes are indexed out of, so a
    missing entry does not produce an error -- it slides every later object's
    prefix one slot to the left, and the nucleus's measurements come back
    named ``pathogen_*``. That is the worst kind of failure this module can
    have: no exception, plausible numbers, wrong object.

    It only shows when an object is switched OFF, which no existing test
    does while others are on -- so this is parametrised over which one.
    """
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1,
        "pathogen_mask_dim": 2, "organelle_mask_dim": 3,
        "cytoplasm": True,
    }
    if disabled == "cytoplasm":
        settings["cytoplasm"] = False
    else:
        settings[f"{disabled}_mask_dim"] = None

    frames = M._morphological_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm, settings, zernike=False)

    assert len(frames) == 5
    names = ("cell", "nucleus", "pathogen", "organelle", "cytoplasm")
    for frame, name in zip(frames, names):
        prefixed = [c for c in frame.columns if c != "label"]
        if name == disabled:
            assert prefixed == [], f"{name} was disabled but produced {prefixed[:3]}"
            continue
        assert prefixed, f"{name} frame carried no measurements"
        assert all(c.startswith(f"{name}_") for c in prefixed), (
            f"{name} slot holds {sorted(prefixed)[:3]}")
