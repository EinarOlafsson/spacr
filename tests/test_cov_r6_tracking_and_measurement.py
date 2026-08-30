"""Round-6 coverage for the per-object measurement and track-QC branches.

Two modules, one theme: what ``spacr.measure`` and ``spacr.timelapse`` do
with a field or a track table that is *missing* something.

Pinned here:

* ``measure._analyze_cytoskeleton`` skips an object whose channel is dark
  everywhere, and still reports the objects that are not.
* ``measure._spatial_adjacency`` tolerates a gap in the label numbering.
* ``measure._spatial_measurements`` silently ignores a *scalar* ``spacing``
  when scaling centroids, while a per-axis one is applied.
* ``measure._intensity_measurements`` drops the intensity-distance block for
  a non-integer ``distance_gaussian_sigma`` and when no child object type is
  configured, drops the radial block for an object type with no objects, and
  drops colocalisation for a single-channel field.
* ``measure._morphological_measurements`` leaves an absent mask out of the
  object-distance neighbour set.
* ``measure.process_measure_crop_results`` skips a job that returned no
  figures while still writing the ones that did.
* ``timelapse._apply_infection_intensity_qc`` keeps the original infection
  column when the strategy helper produced no ``adjusted_infected``, on all
  three scope routes (unknown scope, missing grouping columns, and the real
  per-plate group loop), and skips an empty group.

It also records the four branches in these two modules that no input can
reach, with the proof, rather than silencing them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import measure as M  # noqa: E402
from spacr import timelapse as T  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nested_masks(size=32):
    """cell / nucleus / pathogen / organelle / cytoplasm, co-registered."""
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


def _intensity_settings(**over):
    s = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1, "pathogen_mask_dim": 2,
        "organelle_mask_dim": 3, "cytoplasm": True,
        "radial_dist": True, "calculate_correlation": True,
        "homogeneity": False, "homogeneity_distances": [2, 4],
        "manders_thresholds": [15, 85],
        "distance_gaussian_sigma": 1,
    }
    s.update(over)
    return s


def _run_intensity(channels, **over):
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    if over.pop("_empty_nucleus", False):
        nucleus = np.zeros_like(nucleus)
    return M._intensity_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm, channels,
        _intensity_settings(**over), periphery=False, outside=False)


CH3 = np.random.default_rng(9).random((32, 32, 3)).astype(np.float32)


# ---------------------------------------------------------------------------
# measure._analyze_cytoskeleton
# ---------------------------------------------------------------------------

def test_a_dark_object_is_skipped_and_a_lit_one_is_not():
    """measure.py:664 -- ``if np.any(region_intensity)`` False.

    Two objects, one of which has no signal at all in the cytoskeleton
    channel. The dark one produces no row; the lit one still does, so the
    absence is a skip and not an empty result.
    """
    array = np.zeros((24, 24, 1), dtype=float)
    mask = np.zeros((24, 24), dtype=np.int32)
    mask[2:10, 2:10] = 1          # dark object
    mask[14:22, 14:22] = 2        # lit object
    array[14:22, 14:22, 0] = 1.0

    df = M._analyze_cytoskeleton(array, mask, 0)

    assert list(df["object_label"]) == [2]
    assert float(df.loc[0, "skeleton_length"]) > 0


# ---------------------------------------------------------------------------
# measure._spatial_adjacency
# ---------------------------------------------------------------------------

def test_a_gap_in_the_label_numbering_is_not_a_label():
    """measure.py:971 -- ``if n_boundary[lab] > 0`` False.

    ``np.bincount`` produces a slot for every integer below the largest
    label, so a mask labelled 1 and 3 carries an empty slot 2. It must not
    appear in the percent-touching map with a 0/0 division behind it.
    """
    mask = np.zeros((12, 12), dtype=np.int32)
    mask[1:6, 1:5] = 1
    mask[1:6, 5:10] = 3           # touches label 1, and 2 does not exist

    percent, neighbours = M._spatial_adjacency(mask)

    assert set(percent) == {1, 3}
    assert 2 not in percent
    assert percent[1] > 0 and percent[3] > 0
    assert neighbours == {1: 1, 3: 1}


# ---------------------------------------------------------------------------
# measure._spatial_measurements
# ---------------------------------------------------------------------------

def test_a_scalar_spacing_never_reaches_the_centroids():
    """measure.py:1033 -- ``scale.shape[1] == coords.shape[1]`` False.

    ``spacing`` is reshaped to ``(1, n)`` and only multiplied into the
    centroids when ``n`` matches the number of centroid axes. A scalar
    reshapes to ``(1, 1)``, so it is dropped -- the same field measures the
    same distance as with no spacing at all, while the per-axis form doubles
    it.
    """
    mask = np.zeros((30, 30), dtype=np.int32)
    mask[1:4, 1:4] = 1
    mask[20:24, 20:24] = 2

    scalar = M._spatial_measurements(mask, spacing=2.0, radius=50)
    per_axis = M._spatial_measurements(mask, spacing=(2.0, 2.0), radius=50)
    unscaled = M._spatial_measurements(mask, spacing=None, radius=50)

    d_scalar = float(scalar["nearest_neighbor_distance"].iloc[0])
    d_axis = float(per_axis["nearest_neighbor_distance"].iloc[0])
    d_none = float(unscaled["nearest_neighbor_distance"].iloc[0])

    assert d_scalar == pytest.approx(d_none)
    assert d_axis == pytest.approx(2.0 * d_none)
    # And it is visible in the answer the caller actually reads.
    assert int(scalar["neighbors_within_50"].iloc[0]) == 1
    assert int(per_axis["neighbors_within_50"].iloc[0]) == 0


# ---------------------------------------------------------------------------
# measure._intensity_measurements
# ---------------------------------------------------------------------------

def test_a_float_sigma_is_not_an_int_and_loses_the_distance_block():
    """measure.py:1636 -- ``isinstance(sigma, int)`` False.

    ``distance_gaussian_sigma`` is type-checked, not truth-checked, so 1.5
    turns the whole intensity-distance family off while 1 keeps it.
    """
    with_int = _run_intensity(CH3, distance_gaussian_sigma=1)[0]
    with_float = _run_intensity(CH3, distance_gaussian_sigma=1.5)[0]

    assert "cell_channel_0_distance_to_nucleus" in with_int.columns
    assert not [c for c in with_float.columns if "distance_to_" in c]


def test_no_child_object_means_no_intensity_distance():
    """measure.py:1639 -- neither nucleus nor pathogen dim is configured.

    The distance block measures a cell channel against its *children*, so
    with both child dims unset there is nothing to measure to.
    """
    both = _run_intensity(CH3, nucleus_mask_dim=1, pathogen_mask_dim=2)[0]
    neither = _run_intensity(
        CH3, nucleus_mask_dim=None, pathogen_mask_dim=None)[0]

    assert "cell_channel_0_distance_to_nucleus" in both.columns
    assert not [c for c in neither.columns if "distance_to_" in c]


def test_an_object_type_with_no_objects_gets_no_radial_profile():
    """measure.py:1644 -- ``np.max(nucleus_mask) != 0`` False.

    ``radial_dist`` is on for the whole field, so the pathogen profile is
    still written in the same call -- the missing nucleus profile is the
    empty mask, not a disabled setting.
    """
    frames = _run_intensity(CH3, _empty_nucleus=True)
    nucleus_frame, pathogen_frame = frames[1], frames[2]

    assert not [c for c in nucleus_frame.columns if "rad_dist" in c]
    assert [c for c in pathogen_frame.columns if "rad_dist" in c]


def test_one_channel_has_nothing_to_colocalise_with():
    """measure.py:1690 -- ``channel_arrays.shape[-1] >= 2`` False."""
    one = _run_intensity(CH3[..., :1])[0]
    three = _run_intensity(CH3)[0]

    assert "cell_channel_0_channel_1_Pearson_correlation" in three.columns
    assert not [c for c in one.columns if "_correlation" in c]


# ---------------------------------------------------------------------------
# measure._morphological_measurements  (object distances)
# ---------------------------------------------------------------------------

def test_an_absent_mask_is_not_a_distance_partner():
    """measure.py:1191 -- ``mask is not None and mask.size`` False.

    ``_all_masks`` builds the neighbour set for the object-distance block
    out of the masks this run actually has. With no nucleus mask the
    pathogen frame keeps every distance it measures to the cell and gains
    no nucleus distance at all -- the same call with a nucleus mask adds
    exactly those four columns.
    """
    cell, nucleus, pathogen, organelle, cytoplasm = _nested_masks()
    settings = {
        "cell_mask_dim": None, "nucleus_mask_dim": None,
        "pathogen_mask_dim": 1, "organelle_mask_dim": None,
        "cytoplasm": False, "object_distances": True, "channels": [0],
        "object_distance_intensity": False,
    }
    without = M._morphological_measurements(
        cell, None, pathogen, organelle, cytoplasm, settings,
        zernike=False)[2]
    with_nucleus = M._morphological_measurements(
        cell, nucleus, pathogen, organelle, cytoplasm,
        dict(settings, nucleus_mask_dim=0), zernike=False)[2]

    assert "pathogen_centre_to_cell_surface" in without.columns
    assert not [c for c in without.columns if "nucleus" in c]
    assert "pathogen_centre_to_nucleus_surface" in with_nucleus.columns


def test_a_link_only_pairs_labels_that_are_in_both_masks():
    """timelapse.py:388 -- why ``if union > 0`` can never be False.

    The IoU cost matrix is built over ``np.unique(mask)[1:]``, so every row
    label is present in the previous mask and every column label in the
    next one; the union of two non-empty regions is never empty. This pins
    that invariant from the outside: every label the matcher is offered is
    one the mask actually carries, and every pair it returns is a pair of
    real labels.
    """
    prev = np.zeros((16, 16), dtype=np.int32)
    prev[2:8, 2:8] = 1
    prev[10:14, 10:14] = 4          # deliberately not 2 or 3
    nxt = np.zeros((16, 16), dtype=np.int32)
    nxt[3:9, 3:9] = 7
    nxt[10:14, 10:14] = 9

    matches = T.link_by_iou(prev, nxt, iou_threshold=0.1)

    prev_labels = set(np.unique(prev)[1:].tolist())
    next_labels = set(np.unique(nxt)[1:].tolist())
    assert matches, "the two frames overlap and must link"
    for a, b in matches:
        assert int(a) in prev_labels
        assert int(b) in next_labels
        assert (prev == a).any() and (nxt == b).any()
    assert (4, 9) in [(int(a), int(b)) for a, b in matches]


# ---------------------------------------------------------------------------
# measure.process_measure_crop_results
# ---------------------------------------------------------------------------

def test_a_job_with_no_figures_is_skipped_not_saved(tmp_path):
    """measure.py:4130 -- ``if figs is not None`` False.

    Two completed jobs, one of which produced no figures at all. The one
    that did still writes its PDF, so the skip is a skip and not an early
    return.
    """
    src = tmp_path / "plate" / "merged"
    src.mkdir(parents=True)
    fig = plt.figure()
    try:
        results = [
            (0, 1.0, 3, None),
            (1, 1.0, 3, {"cells__field_a": fig}),
        ]
        M.process_measure_crop_results(results, {"src": str(src)})
    finally:
        plt.close(fig)

    written = sorted(p.name for p in
                     (tmp_path / "plate" / "results").rglob("*.pdf"))
    assert written == ["field_a.pdf"]


# ---------------------------------------------------------------------------
# timelapse._apply_infection_intensity_qc
# ---------------------------------------------------------------------------

def _qc_frame():
    return pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "wellID": ["A1", "A1", "B2", "B2"],
        "fieldID": [1, 1, 1, 1],
        "cellID": [1, 2, 1, 2],
        "infected": [0, 1, 1, 0],
        "pathogen_channel_1_mean_intensity": [0.1, 0.9, 0.8, 0.2],
    })


def _stub_qc(add_adjusted):
    def qc(all_df, settings, infection_col, pathogen_chan, motility_dir):
        out = all_df.copy()
        if add_adjusted:
            out["adjusted_infected"] = 1 - out[infection_col]
        settings["infection_hist_data"] = {"seen": len(out)}
        return out, infection_col
    return qc


def _apply(monkeypatch, tmp_path, df, scope, add_adjusted):
    monkeypatch.setattr(T, "_infection_qc_histogram", _stub_qc(add_adjusted))
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_qc_scope": scope,
    }
    out, col = T._apply_infection_intensity_qc(
        all_df=df, settings=settings, infection_col="infected",
        motility_dir=str(tmp_path / "motility"), pathogen_chan=1)
    return out, col, settings


@pytest.mark.parametrize("scope", ["sideways", "per_plate"])
def test_qc_that_adjusts_nothing_keeps_the_original_column(
        monkeypatch, tmp_path, scope):
    """timelapse.py:5413 / 5451 / 5510 -- no ``adjusted_infected`` produced.

    'sideways' is not a scope name, which routes through the
    unknown-scope fallback; 'per_plate' takes the real group loop. Both
    report the column the caller passed in when the strategy helper adjusted
    nothing, and both switch to 'adjusted_infected' when it did.
    """
    plain, col_plain, _ = _apply(
        monkeypatch, tmp_path, _qc_frame(), scope, add_adjusted=False)
    assert col_plain == "infected"
    assert "adjusted_infected" not in plain.columns
    assert len(plain) == 4

    adjusted, col_adj, _ = _apply(
        monkeypatch, tmp_path, _qc_frame(), scope, add_adjusted=True)
    assert col_adj == "adjusted_infected"
    assert sorted(adjusted["adjusted_infected"]) == [0, 0, 1, 1]


def test_a_grouped_scope_without_its_columns_falls_back_to_combined(
        monkeypatch, tmp_path):
    """timelapse.py:5451 -- the missing-grouping-columns fallback.

    'per_well' needs plateID and wellID. Given a frame with neither, QC
    still runs once over the whole table rather than not at all.
    """
    df = _qc_frame().drop(columns=["plateID", "wellID"])
    out, col, settings = _apply(
        monkeypatch, tmp_path, df, "per_well", add_adjusted=False)

    assert col == "infected"
    assert len(out) == 4
    assert settings["infection_hist_data"] == {"seen": 4}


def test_an_empty_plate_group_is_skipped(monkeypatch, tmp_path):
    """timelapse.py:5499 -- ``if df_group.empty: continue``.

    A categorical plateID with a category no row uses makes pandas hand the
    group loop an empty frame. It must be skipped, not sent to the QC
    helper, and the two real plates must both come back.
    """
    df = _qc_frame()
    df["plateID"] = pd.Categorical(
        df["plateID"], categories=["p1", "p2", "p3_never_imaged"])

    seen = []

    def qc(all_df, settings, infection_col, pathogen_chan, motility_dir):
        seen.append(len(all_df))
        return all_df.copy(), infection_col

    monkeypatch.setattr(T, "_infection_qc_histogram", qc)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "histogram",
        "infection_intensity_qc_scope": "plate",
    }
    out, col = T._apply_infection_intensity_qc(
        all_df=df, settings=settings, infection_col="infected",
        motility_dir=str(tmp_path / "motility"), pathogen_chan=1)

    assert seen == [2, 2]          # p3 never reached the helper
    assert len(out) == 4
    assert col == "infected"
