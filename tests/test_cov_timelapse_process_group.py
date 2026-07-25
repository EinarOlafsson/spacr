"""CPU coverage for the spacr.timelapse merged-group worker.

Covers ``_process_merged_group`` (the multiprocessing worker that turns a
group of ``merged/*.npy`` files into a per-cell-per-frame feature table) and
``_smooth_tracks_and_features`` (centroid glitch repair + z-score feature
smoothing).

Everything here is synthetic, deterministic and offline: merged arrays are
built by hand as float32 stacks of [intensity planes..., cell, nucleus,
pathogen] so the exact areas / centroids / intensities are known and can be
asserted numerically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# synthetic merged-array builders
# ---------------------------------------------------------------------------

H, W = 44, 52

# cell 1: 16x16 square that slides right by 2 px per frame
CELL1_AREA = 16 * 16
CELL2_AREA = 16 * 16
NUC_AREA = 6 * 6
PATH_AREA = 3 * 3


def _masks_for_frame(t, *, with_nucleus, with_pathogen, nucleus_fills_cell=False):
    """Cell / nucleus / pathogen label images for frame ``t``.

    Cell 1 (label 1) drifts +2 px in x per frame; cell 2 (label 2) is static.
    Nucleus and pathogen labels match their parent cell label.
    """
    shift = 2 * t

    cell = np.zeros((H, W), dtype=np.int32)
    cell[4:20, 4 + shift:20 + shift] = 1
    cell[24:40, 28:44] = 2

    nucleus = np.zeros((H, W), dtype=np.int32)
    if with_nucleus:
        if nucleus_fills_cell:
            nucleus = cell.copy()
        else:
            nucleus[8:14, 8 + shift:14 + shift] = 1
            nucleus[28:34, 32:38] = 2

    pathogen = np.zeros((H, W), dtype=np.int32)
    if with_pathogen:
        # only inside cell 1 -> cell 2 must end up with NaN pathogen features
        pathogen[16:19, 6 + shift:9 + shift] = 1

    return cell, nucleus, pathogen


def _planes_for_frame(t, n_channels, with_nucleus, with_pathogen,
                      with_masks=True, nucleus_fills_cell=False):
    cell, nucleus, pathogen = _masks_for_frame(
        t,
        with_nucleus=with_nucleus,
        with_pathogen=with_pathogen,
        nucleus_fills_cell=nucleus_fills_cell,
    )

    yy, xx = np.mgrid[:H, :W]
    base = (0.5 * yy + 0.25 * xx + 3.0 * t).astype(np.float32)

    all_channels = [
        base + 100.0 * (cell > 0),          # ch0: bright on the whole cell
        base + 250.0 * (pathogen > 0),      # ch1: bright only on pathogens
    ]
    planes = [c.astype(np.float32) for c in all_channels[:n_channels]]

    if with_masks:
        planes.append(cell.astype(np.float32))
        if with_nucleus:
            planes.append(nucleus.astype(np.float32))
        if with_pathogen:
            planes.append(pathogen.astype(np.float32))

    return planes


def _build_merged_group(tmp_path, *, n_frames=3, n_channels=2,
                        with_nucleus=True, with_pathogen=True,
                        with_masks=True, layout="planes_first",
                        nucleus_fills_cell=False, name="proj"):
    """Write ``<src>/merged/plate1_A01_1_<t>.npy`` files; return (src, basenames)."""
    src = tmp_path / name
    merged = src / "merged"
    merged.mkdir(parents=True)

    basenames = []
    for t in range(n_frames):
        planes = _planes_for_frame(
            t, n_channels, with_nucleus, with_pathogen,
            with_masks=with_masks, nucleus_fills_cell=nucleus_fills_cell,
        )
        arr = np.stack(planes).astype(np.float32)
        if layout == "channels_last":
            arr = np.moveaxis(arr, 0, -1)
        bn = f"plate1_A01_1_{t}.npy"
        np.save(merged / bn, arr)
        basenames.append(bn)
    return str(src), basenames


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# _process_merged_group — early-exit / defensive branches
# ---------------------------------------------------------------------------

def test_process_merged_group_empty_basenames_returns_empty_df():
    from spacr.timelapse import _process_merged_group

    out = _process_merged_group(("/nonexistent", [], 2, 0, 0, 1))
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert list(out.columns) == []


def test_process_merged_group_first_array_not_3d_returns_empty_df(tmp_path):
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_frames=2)
    # clobber the first (t=0) file with a 2-D array
    np.save(f"{src}/merged/{basenames[0]}", np.zeros((H, W), dtype=np.float32))

    out = _process_merged_group((src, basenames, 2, 0, 0, 1))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_process_merged_group_reorient_failure_returns_empty_df(tmp_path, monkeypatch):
    """Failure injection: the reorientation helper blows up on the first file."""
    import spacr.timelapse as tl

    src, basenames = _build_merged_group(tmp_path, n_frames=2)

    def _boom(arr, n_channels, max_extra_masks=3):
        raise ValueError("injected reorientation failure")

    monkeypatch.setattr(tl, "_reorient_merged_array", _boom)

    out = tl._process_merged_group((src, basenames, 2, 0, 0, 1))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_process_merged_group_without_mask_planes_returns_empty_df(tmp_path):
    """merged arrays carrying only intensity planes -> no cell masks at all."""
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_frames=3, with_masks=False)
    # sanity: the file really has n_channels planes only
    arr = np.load(f"{src}/merged/{basenames[0]}")
    assert arr.shape == (2, H, W)

    out = _process_merged_group((src, basenames, 2, 0, 0, 1))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_process_merged_group_empty_cell_props_returns_empty_df(tmp_path, monkeypatch):
    """Failure injection: regionprops yields nothing for a non-empty mask."""
    import spacr.timelapse as tl

    src, basenames = _build_merged_group(tmp_path, n_frames=3, n_channels=0)

    def _no_props(**kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(tl, "_compute_regionprops_stack", _no_props)

    out = tl._process_merged_group((src, basenames, 0, 0, 0, 1))
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# ---------------------------------------------------------------------------
# _process_merged_group — the full happy path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["planes_first", "channels_last"])
def test_process_merged_group_full_feature_table(tmp_path, layout):
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, n_frames=3, n_channels=2, layout=layout, name=f"proj_{layout}"
    )
    # feed the filenames out of order: the worker must sort them by timeID
    shuffled = [basenames[2], basenames[0], basenames[1]]

    df = _process_merged_group((src, shuffled, 2, 0, 0, 1))

    assert not df.empty
    # 2 cells x 3 frames
    assert len(df) == 6
    assert sorted(df["track_id"].unique().tolist()) == [1, 2]

    # --- metadata was attached and frames follow timeID order ---
    for col in ("plateID", "wellID", "rowID", "columnID", "fieldID",
                "timeID", "prcf", "prcft", "filename", "cellID"):
        assert col in df.columns, col
    assert (df["plateID"] == "plate1").all()
    assert (df["wellID"] == "A01").all()
    assert (df["rowID"] == "A").all()
    assert (df["columnID"] == 1).all()
    assert (df["fieldID"] == "1").all()
    frame_to_time = df.drop_duplicates("frame").set_index("frame")["timeID"].to_dict()
    assert frame_to_time == {0: 0, 1: 1, 2: 2}
    assert (df["cellID"] == df["track_id"]).all()
    assert set(df["prcft"]) == {"plate1_A01_1_0", "plate1_A01_1_1", "plate1_A01_1_2"}

    # --- cell geometry is exactly the square we drew ---
    assert (df["cell_area"] == CELL1_AREA).all()
    assert np.allclose(df["cell_centroid-0"], 11.5 * (df["track_id"] == 1)
                       + 31.5 * (df["track_id"] == 2))
    # cell 1 slides +2 px in x per frame, cell 2 is static
    c1 = df[df["track_id"] == 1].sort_values("frame")
    assert np.allclose(c1["cell_centroid-1"].to_numpy(), [11.5, 13.5, 15.5])
    c2 = df[df["track_id"] == 2].sort_values("frame")
    assert np.allclose(c2["cell_centroid-1"].to_numpy(), [35.5] * 3)

    # --- per-channel mean intensities, and cell_chan==0 consistency ---
    assert "cell_mean_intensity_ch0" in df.columns
    assert "cell_mean_intensity_ch1" in df.columns
    assert np.allclose(df["cell_mean_intensity"], df["cell_mean_intensity_ch0"])
    # ch0 is boosted by +100 over the whole cell, ch1 only over a 9 px pathogen
    assert (df["cell_mean_intensity_ch0"] > df["cell_mean_intensity_ch1"]).all()

    # --- intensity percentiles for every compartment / channel ---
    for prefix in ("cell", "nucleus", "pathogen", "cytoplasm"):
        for ch in (0, 1):
            assert f"{prefix}_p25_intensity_ch{ch}" in df.columns
            assert f"{prefix}_p75_intensity_ch{ch}" in df.columns
    assert (df["cell_p25_intensity_ch0"] <= df["cell_p75_intensity_ch0"]).all()

    # --- child summaries ---
    assert (df["n_nuclei"] == 1).all()
    assert np.allclose(df["nucleus_area"], NUC_AREA)
    assert (df["n_cytoplasm"] == 1).all()
    # cytoplasm = cell - nucleus - pathogen
    cyto = df.set_index("track_id")["cytoplasm_area"]
    assert np.allclose(cyto.loc[1], CELL1_AREA - NUC_AREA - PATH_AREA)
    assert np.allclose(cyto.loc[2], CELL2_AREA - NUC_AREA)
    # pathogens only exist in cell 1
    assert (df.loc[df["track_id"] == 1, "n_pathogens"] == 1).all()
    assert df.loc[df["track_id"] == 2, "n_pathogens"].isna().all()
    assert np.allclose(df.loc[df["track_id"] == 1, "pathogen_area"], PATH_AREA)


def test_process_merged_group_layouts_agree(tmp_path):
    """(planes, H, W) and (H, W, planes) merged files must give the same table."""
    from spacr.timelapse import _process_merged_group

    src_a, bn_a = _build_merged_group(tmp_path, layout="planes_first", name="a")
    src_b, bn_b = _build_merged_group(tmp_path, layout="channels_last", name="b")

    df_a = _process_merged_group((src_a, bn_a, 2, 0, 0, 1)).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    df_b = _process_merged_group((src_b, bn_b, 2, 0, 0, 1)).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)

    assert list(df_a.columns) == list(df_b.columns)
    num = df_a.select_dtypes("number").columns
    pd.testing.assert_frame_equal(df_a[num], df_b[num])


def test_process_merged_group_nucleus_only(tmp_path):
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=True, with_pathogen=False, name="nuc_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, 0, None))

    assert len(df) == 6
    assert (df["n_nuclei"] == 1).all()
    assert "n_pathogens" not in df.columns
    assert "pathogen_area" not in df.columns
    # cytoplasm = cell - nucleus (no pathogen subtracted)
    assert np.allclose(df["cytoplasm_area"], CELL1_AREA - NUC_AREA)


def test_process_merged_group_pathogen_only(tmp_path):
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=False, with_pathogen=True, name="path_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, None, 1))

    assert len(df) == 6
    assert "n_nuclei" not in df.columns
    assert (df.loc[df["track_id"] == 1, "n_pathogens"] == 1).all()
    assert df.loc[df["track_id"] == 2, "n_pathogens"].isna().all()
    # cell 1 loses the pathogen area from its cytoplasm, cell 2 keeps all of it
    cyto = df.set_index("track_id")["cytoplasm_area"]
    assert np.allclose(cyto.loc[1], CELL1_AREA - PATH_AREA)
    assert np.allclose(cyto.loc[2], CELL2_AREA)


def test_process_merged_group_cell_masks_only(tmp_path):
    """No nucleus and no pathogen planes -> no cytoplasm, no child summaries."""
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=False, with_pathogen=False, name="cell_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, None, None))

    assert len(df) == 6
    for col in ("n_nuclei", "n_pathogens", "n_cytoplasm",
                "nucleus_area", "pathogen_area", "cytoplasm_area"):
        assert col not in df.columns
    assert (df["cell_area"] == CELL1_AREA).all()
    assert "cell_mean_intensity_ch1" in df.columns


def test_process_merged_group_nucleus_covers_whole_cell_has_no_cytoplasm(tmp_path):
    """When the nucleus fills the cell the cytoplasm mask is empty."""
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=True, with_pathogen=False,
        nucleus_fills_cell=True, name="no_cyto",
    )
    df = _process_merged_group((src, basenames, 2, 0, 0, None))

    assert len(df) == 6
    assert (df["n_nuclei"] == 1).all()
    assert np.allclose(df["nucleus_area"], CELL1_AREA)
    assert "n_cytoplasm" not in df.columns
    assert "cytoplasm_area" not in df.columns


def test_process_merged_group_without_intensity_channels(tmp_path):
    """n_channels=0: geometry only, no percentile or mean-intensity columns."""
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_channels=0, name="no_chans")
    arr = np.load(f"{src}/merged/{basenames[0]}")
    assert arr.shape == (3, H, W)  # cell + nucleus + pathogen only

    df = _process_merged_group((src, basenames, 0, 0, 0, 1))

    assert len(df) == 6
    assert (df["cell_area"] == CELL1_AREA).all()
    assert (df["n_nuclei"] == 1).all()
    # no intensity information at all
    assert not [c for c in df.columns if "intensity" in c]
    assert "cell_solidity" in df.columns


def test_process_merged_group_single_frame_group(tmp_path):
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_frames=1, name="one_frame")
    df = _process_merged_group((src, basenames, 2, 0, 0, 1))

    assert len(df) == 2
    assert set(df["frame"]) == {0}
    assert set(df["timeID"]) == {0}


# ---------------------------------------------------------------------------
# _smooth_tracks_and_features
# ---------------------------------------------------------------------------

def _track_frame(cell_id, xs, ys=None, **extra):
    n = len(xs)
    if ys is None:
        ys = [10.0] * n
    data = {
        "plateID": ["plate1"] * n,
        "wellID": ["A01"] * n,
        "fieldID": ["1"] * n,
        "cellID": [cell_id] * n,
        "frame": list(range(n)),
        "cell_centroid-0": list(ys),
        "cell_centroid-1": list(xs),
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_smooth_tracks_empty_dataframe_returns_input():
    from spacr.timelapse import _smooth_tracks_and_features

    df = pd.DataFrame()
    out = _smooth_tracks_and_features(df)
    assert out is df
    assert out.empty


def test_smooth_tracks_missing_centroid_columns_is_noop():
    from spacr.timelapse import _smooth_tracks_and_features

    df = pd.DataFrame({
        "plateID": ["p1"] * 3,
        "wellID": ["A01"] * 3,
        "fieldID": ["1"] * 3,
        "cellID": [1, 1, 1],
        "frame": [2, 0, 1],
        "cell_area": [10.0, 11.0, 12.0],
    })
    out = _smooth_tracks_and_features(df)

    assert "cell_centroid-0" not in out.columns
    assert len(out) == 3
    # returned sorted by (plate, well, field, cellID, frame)
    assert out["frame"].tolist() == [0, 1, 2]
    assert out["cell_area"].tolist() == [11.0, 12.0, 10.0]


def test_smooth_tracks_interpolates_scalar_features_at_glitch_frame():
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    assert len(out) == 3
    # the scalar feature at the glitch frame is interpolated from its neighbours
    assert out["cell_area"].tolist() == [100.0, 110.0, 120.0]
    assert out["cell_centroid-0"].tolist() == [10.0, 10.0, 10.0]


def test_smooth_tracks_writes_back_glitch_centroid_for_float32_columns():
    """float32 centroids force ``to_numpy(dtype=float)`` to copy, so the
    write-back guard actually sees a difference and the fix lands in the
    returned frame."""
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0], ys=[10.0, 400.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    df["cell_centroid-0"] = df["cell_centroid-0"].astype(np.float32)
    df["cell_centroid-1"] = df["cell_centroid-1"].astype(np.float32)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    assert len(out) == 3
    assert out["cell_centroid-1"].tolist() == [10.0, 11.0, 12.0]
    assert out["cell_centroid-0"].tolist() == [10.0, 11.0, 12.0]
    assert out["cell_area"].tolist() == [100.0, 110.0, 120.0]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _smooth_tracks_and_features never writes the glitch-corrected "
           "centroid back for float64 columns - g[col].to_numpy(dtype=float) "
           "aliases the group's own buffer, so the in-place fix also mutates "
           "`g` and the guard `y[i] != g[y_col].iloc[i]` is always False.",
)
def test_smooth_tracks_writes_back_glitch_corrected_centroid():
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    # the teleporting frame should be replaced by the midpoint of its neighbours
    assert out["cell_centroid-1"].tolist() == [10.0, 11.0, 12.0]


def test_smooth_tracks_drops_track_with_unrecoverable_jump():
    from spacr.timelapse import _smooth_tracks_and_features

    bad = _track_frame(1, xs=[10.0, 10.0, 300.0])
    good = _track_frame(2, xs=[20.0, 21.0, 22.0])
    df = pd.concat([bad, good], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    assert out["cellID"].unique().tolist() == [2]
    assert len(out) == 3
    assert out["cell_centroid-1"].tolist() == [20.0, 21.0, 22.0]
    # index was reset after the drop
    assert out.index.tolist() == [0, 1, 2]


def test_smooth_tracks_keeps_short_tracks_untouched():
    """1-frame tracks are skipped; 2-frame tracks are never dropped."""
    from spacr.timelapse import _smooth_tracks_and_features

    one = _track_frame(1, xs=[5.0])
    two = _track_frame(2, xs=[5.0, 400.0])          # huge jump, but n < 3
    three = _track_frame(3, xs=[7.0, 8.0, 9.0])
    df = pd.concat([one, two, three], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    assert len(out) == 6
    assert sorted(out["cellID"].unique().tolist()) == [1, 2, 3]
    assert out.loc[out["cellID"] == 2, "cell_centroid-1"].tolist() == [5.0, 400.0]
    assert out.loc[out["cellID"] == 1, "cell_centroid-1"].tolist() == [5.0]


def test_smooth_tracks_zscore_smoothing_of_scalar_features():
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(
        1,
        xs=[20.0] * 5,
        cell_area=[100.0, 100.0, 1000.0, 100.0, 100.0],
        cell_solidity=[0.9] * 5,                        # std == 0 -> skipped
        cell_perimeter=[np.nan] * 5,                    # all non-finite -> skipped
        cell_max_intensity=[1.0, 2.0, np.nan, 8.0, 9.0],  # z is NaN at the spike
    )
    out = _smooth_tracks_and_features(df, max_displacement=50.0, zscore_thresh=1.5)

    assert len(out) == 5
    # the |z| > 1.5 outlier is replaced by the mean of its neighbours
    assert out["cell_area"].tolist() == [100.0, 100.0, 100.0, 100.0, 100.0]
    # constant / all-NaN / NaN-at-centre columns are left alone
    assert out["cell_solidity"].tolist() == [0.9] * 5
    assert out["cell_perimeter"].isna().all()
    assert out["cell_max_intensity"].isna().sum() == 1
    assert out["cell_max_intensity"].tolist()[0] == 1.0
    assert out["cell_max_intensity"].tolist()[-1] == 9.0


def test_smooth_tracks_zscore_leaves_mild_outliers_alone():
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(1, xs=[20.0] * 5,
                      cell_area=[100.0, 100.0, 130.0, 100.0, 100.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0, zscore_thresh=3.0)

    assert out["cell_area"].tolist() == [100.0, 100.0, 130.0, 100.0, 100.0]


def test_smooth_tracks_casts_integer_columns_to_float():
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(1, xs=[10, 11, 12], ys=[10, 10, 10],
                      cell_area=[100, 101, 102])
    assert df["cell_area"].dtype.kind == "i"

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    assert out["cell_area"].dtype == np.float64
    assert out["cell_centroid-0"].dtype == np.float64
    assert out["cell_centroid-1"].dtype == np.float64
    assert out["cell_area"].tolist() == [100.0, 101.0, 102.0]


def test_smooth_tracks_glitch_in_one_track_does_not_affect_others():
    from spacr.timelapse import _smooth_tracks_and_features

    glitchy = _track_frame(1, xs=[10.0, 500.0, 12.0], cell_area=[10.0, 90.0, 14.0])
    calm = _track_frame(2, xs=[30.0, 31.0, 32.0], cell_area=[20.0, 21.0, 22.0])
    df = pd.concat([calm, glitchy], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    g = out[out["cellID"] == 1].sort_values("frame")
    c = out[out["cellID"] == 2].sort_values("frame")
    # the glitchy track has its middle-frame feature interpolated ...
    assert g["cell_area"].tolist() == [10.0, 12.0, 14.0]
    # ... while the calm track is left completely untouched
    assert c["cell_centroid-1"].tolist() == [30.0, 31.0, 32.0]
    assert c["cell_area"].tolist() == [20.0, 21.0, 22.0]
    # and neither track is dropped
    assert len(out) == 6


# ---------------------------------------------------------------------------
# the two functions composed: worker output feeds the smoother
# ---------------------------------------------------------------------------

def test_process_merged_group_output_is_smoothable(tmp_path):
    from spacr.timelapse import _process_merged_group, _smooth_tracks_and_features

    src, basenames = _build_merged_group(tmp_path, n_frames=3, name="pipeline")
    df = _process_merged_group((src, basenames, 2, 0, 0, 1))

    smoothed = _smooth_tracks_and_features(df, max_displacement=50.0)

    # nothing in this synthetic movie jumps, so every row survives untouched
    assert len(smoothed) == len(df)
    assert sorted(smoothed["cellID"].unique().tolist()) == [1, 2]
    c1 = smoothed[smoothed["cellID"] == 1].sort_values("frame")
    assert np.allclose(c1["cell_centroid-1"].to_numpy(), [11.5, 13.5, 15.5])
    assert (smoothed["cell_area"] == float(CELL1_AREA)).all()
