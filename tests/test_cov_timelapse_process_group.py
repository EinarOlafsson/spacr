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

import functools
import inspect

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

def _case_group_cannot_be_measured(tmp_path, monkeypatch, fault):
    """No files, a 2-D first array, a reorientation failure, no mask planes or no cell regions: empty DataFrame, no error."""
    observed = {}
    import spacr.timelapse as tl

    if fault == "no_basenames":
        args = ("/nonexistent", [], 2, 0, 0, 1)
    else:
        n_channels = 0 if fault == "no_cell_props" else 2
        src, basenames = _build_merged_group(
            tmp_path, n_frames=3, n_channels=n_channels,
            with_masks=(fault != "no_mask_planes"))
        args = (src, basenames, n_channels, 0, 0, 1)
        if fault == "first_array_2d":
            np.save(f"{src}/merged/{basenames[0]}", np.zeros((H, W), dtype=np.float32))
        elif fault == "reorient_fails":
            def _boom(arr, n_channels, max_extra_masks=3):
                raise ValueError("injected reorientation failure")

            monkeypatch.setattr(tl, "_reorient_merged_array", _boom)
        elif fault == "no_cell_props":
            monkeypatch.setattr(tl, "_compute_regionprops_stack", lambda **kwargs: pd.DataFrame())

    out = tl._process_merged_group(args)

    observed['isinstance(out, pd.DataFrame)'] = bool(isinstance(out, pd.DataFrame))
    observed['out.empty'] = bool(out.empty)
    return observed


_EXPECT_group_cannot_be_measured = {
    'isinstance(out, pd.DataFrame)': True,
    'out.empty': True,
}


# ---------------------------------------------------------------------------
# _process_merged_group — the full happy path
# ---------------------------------------------------------------------------

def _case_full_feature_table(tmp_path, layout):
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, n_frames=3, n_channels=2, layout=layout, name=f"proj_{layout}"
    )
    # feed the filenames out of order: the worker must sort them by timeID
    shuffled = [basenames[2], basenames[0], basenames[1]]

    df = _process_merged_group((src, shuffled, 2, 0, 0, 1))

    observed['not df.empty'] = bool(not df.empty)
    # 2 cells x 3 frames
    observed['len(df)'] = len(df)
    observed["sorted(df['track_id'].unique().tolist())"] = sorted(df['track_id'].unique().tolist())

    # --- metadata was attached and frames follow timeID order ---
    for col in ("plateID", "wellID", "rowID", "columnID", "fieldID",
                "timeID", "prcf", "prcft", "filename", "cellID"):
        assert col in df.columns, col
    observed["(df['plateID'] == 'plate1').all()"] = bool((df['plateID'] == 'plate1').all())
    observed["(df['wellID'] == 'A01').all()"] = bool((df['wellID'] == 'A01').all())
    observed["(df['rowID'] == 'A').all()"] = bool((df['rowID'] == 'A').all())
    observed["(df['columnID'] == 1).all()"] = bool((df['columnID'] == 1).all())
    observed["(df['fieldID'] == '1').all()"] = bool((df['fieldID'] == '1').all())
    frame_to_time = df.drop_duplicates("frame").set_index("frame")["timeID"].to_dict()
    observed['frame_to_time'] = frame_to_time
    observed["(df['cellID'] == df['track_id']).all()"] = bool((df['cellID'] == df['track_id']).all())
    observed["set(df['prcft'])"] = set(df['prcft'])

    # --- cell geometry is exactly the square we drew ---
    observed["(df['cell_area'] == CELL1_AREA).all()"] = bool((df['cell_area'] == CELL1_AREA).all())
    observed["np.allclose(df['cell_centroid-0'], 11.5 * (df['track_id'] == 1) + 31.5 * (df['track_id'] == 2))"] = bool(np.allclose(df['cell_centroid-0'], 11.5 * (df['track_id'] == 1) + 31.5 * (df['track_id'] == 2)))
    # cell 1 slides +2 px in x per frame, cell 2 is static
    c1 = df[df["track_id"] == 1].sort_values("frame")
    observed["np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5])"] = bool(np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5]))
    c2 = df[df["track_id"] == 2].sort_values("frame")
    observed["np.allclose(c2['cell_centroid-1'].to_numpy(), [35.5] * 3)"] = bool(np.allclose(c2['cell_centroid-1'].to_numpy(), [35.5] * 3))

    # --- per-channel mean intensities, and cell_chan==0 consistency ---
    observed["'cell_mean_intensity_ch0' in df.columns"] = bool('cell_mean_intensity_ch0' in df.columns)
    observed["'cell_mean_intensity_ch1' in df.columns"] = bool('cell_mean_intensity_ch1' in df.columns)
    observed["np.allclose(df['cell_mean_intensity'], df['cell_mean_intensity_ch0'])"] = bool(np.allclose(df['cell_mean_intensity'], df['cell_mean_intensity_ch0']))
    # ch0 is boosted by +100 over the whole cell, ch1 only over a 9 px pathogen
    observed["(df['cell_mean_intensity_ch0'] > df['cell_mean_intensity_ch1']).all()"] = bool((df['cell_mean_intensity_ch0'] > df['cell_mean_intensity_ch1']).all())

    # --- intensity percentiles for every compartment / channel ---
    for prefix in ("cell", "nucleus", "pathogen", "cytoplasm"):
        for ch in (0, 1):
            assert f"{prefix}_p25_intensity_ch{ch}" in df.columns
            assert f"{prefix}_p75_intensity_ch{ch}" in df.columns
    observed["(df['cell_p25_intensity_ch0'] <= df['cell_p75_intensity_ch0']).all()"] = bool((df['cell_p25_intensity_ch0'] <= df['cell_p75_intensity_ch0']).all())

    # --- child summaries ---
    observed["(df['n_nuclei'] == 1).all()"] = bool((df['n_nuclei'] == 1).all())
    observed["np.allclose(df['nucleus_area'], NUC_AREA)"] = bool(np.allclose(df['nucleus_area'], NUC_AREA))
    observed["(df['n_cytoplasm'] == 1).all()"] = bool((df['n_cytoplasm'] == 1).all())
    # cytoplasm = cell - nucleus - pathogen
    cyto = df.set_index("track_id")["cytoplasm_area"]
    observed['np.allclose(cyto.loc[1], CELL1_AREA - NUC_AREA - PATH_AREA)'] = bool(np.allclose(cyto.loc[1], CELL1_AREA - NUC_AREA - PATH_AREA))
    observed['np.allclose(cyto.loc[2], CELL2_AREA - NUC_AREA)'] = bool(np.allclose(cyto.loc[2], CELL2_AREA - NUC_AREA))
    # pathogens only exist in cell 1
    observed["(df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all()"] = bool((df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all())
    observed["df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all()"] = bool(df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all())
    observed["np.allclose(df.loc[df['track_id'] == 1, 'pathogen_area'], PATH_AREA)"] = bool(np.allclose(df.loc[df['track_id'] == 1, 'pathogen_area'], PATH_AREA))
    return observed


_EXPECT_full_feature_table = {
    'not df.empty': True,
    'len(df)': 6,
    "sorted(df['track_id'].unique().tolist())": [1, 2],
    "(df['plateID'] == 'plate1').all()": True,
    "(df['wellID'] == 'A01').all()": True,
    "(df['rowID'] == 'A').all()": True,
    "(df['columnID'] == 1).all()": True,
    "(df['fieldID'] == '1').all()": True,
    'frame_to_time': {0: 0, 1: 1, 2: 2},
    "(df['cellID'] == df['track_id']).all()": True,
    "set(df['prcft'])": {'plate1_A01_1_0', 'plate1_A01_1_1', 'plate1_A01_1_2'},
    "(df['cell_area'] == CELL1_AREA).all()": True,
    "np.allclose(df['cell_centroid-0'], 11.5 * (df['track_id'] == 1) + 31.5 * (df['track_id'] == 2))": True,
    "np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5])": True,
    "np.allclose(c2['cell_centroid-1'].to_numpy(), [35.5] * 3)": True,
    "'cell_mean_intensity_ch0' in df.columns": True,
    "'cell_mean_intensity_ch1' in df.columns": True,
    "np.allclose(df['cell_mean_intensity'], df['cell_mean_intensity_ch0'])": True,
    "(df['cell_mean_intensity_ch0'] > df['cell_mean_intensity_ch1']).all()": True,
    "(df['cell_p25_intensity_ch0'] <= df['cell_p75_intensity_ch0']).all()": True,
    "(df['n_nuclei'] == 1).all()": True,
    "np.allclose(df['nucleus_area'], NUC_AREA)": True,
    "(df['n_cytoplasm'] == 1).all()": True,
    'np.allclose(cyto.loc[1], CELL1_AREA - NUC_AREA - PATH_AREA)': True,
    'np.allclose(cyto.loc[2], CELL2_AREA - NUC_AREA)': True,
    "(df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all()": True,
    "df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all()": True,
    "np.allclose(df.loc[df['track_id'] == 1, 'pathogen_area'], PATH_AREA)": True,
}


def _case_group_layouts_agree(tmp_path):
    """(planes, H, W) and (H, W, planes) merged files must give the same table."""
    observed = {}
    from spacr.timelapse import _process_merged_group

    src_a, bn_a = _build_merged_group(tmp_path, layout="planes_first", name="a")
    src_b, bn_b = _build_merged_group(tmp_path, layout="channels_last", name="b")

    df_a = _process_merged_group((src_a, bn_a, 2, 0, 0, 1)).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)
    df_b = _process_merged_group((src_b, bn_b, 2, 0, 0, 1)).sort_values(
        ["frame", "track_id"]).reset_index(drop=True)

    observed['list(df_a.columns) == list(df_b.columns)'] = bool(list(df_a.columns) == list(df_b.columns))
    num = df_a.select_dtypes("number").columns
    pd.testing.assert_frame_equal(df_a[num], df_b[num])
    return observed


_EXPECT_group_layouts_agree = {
    'list(df_a.columns) == list(df_b.columns)': True,
}


def _case_group_nucleus_only(tmp_path):
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=True, with_pathogen=False, name="nuc_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, 0, None))

    observed['len(df)'] = len(df)
    observed["(df['n_nuclei'] == 1).all()"] = bool((df['n_nuclei'] == 1).all())
    observed["'n_pathogens' not in df.columns"] = bool('n_pathogens' not in df.columns)
    observed["'pathogen_area' not in df.columns"] = bool('pathogen_area' not in df.columns)
    # cytoplasm = cell - nucleus (no pathogen subtracted)
    observed["np.allclose(df['cytoplasm_area'], CELL1_AREA - NUC_AREA)"] = bool(np.allclose(df['cytoplasm_area'], CELL1_AREA - NUC_AREA))
    return observed


_EXPECT_group_nucleus_only = {
    'len(df)': 6,
    "(df['n_nuclei'] == 1).all()": True,
    "'n_pathogens' not in df.columns": True,
    "'pathogen_area' not in df.columns": True,
    "np.allclose(df['cytoplasm_area'], CELL1_AREA - NUC_AREA)": True,
}


def _case_group_pathogen_only(tmp_path):
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=False, with_pathogen=True, name="path_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, None, 1))

    observed['len(df)'] = len(df)
    observed["'n_nuclei' not in df.columns"] = bool('n_nuclei' not in df.columns)
    observed["(df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all()"] = bool((df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all())
    observed["df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all()"] = bool(df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all())
    # cell 1 loses the pathogen area from its cytoplasm, cell 2 keeps all of it
    cyto = df.set_index("track_id")["cytoplasm_area"]
    observed['np.allclose(cyto.loc[1], CELL1_AREA - PATH_AREA)'] = bool(np.allclose(cyto.loc[1], CELL1_AREA - PATH_AREA))
    observed['np.allclose(cyto.loc[2], CELL2_AREA)'] = bool(np.allclose(cyto.loc[2], CELL2_AREA))
    return observed


_EXPECT_group_pathogen_only = {
    'len(df)': 6,
    "'n_nuclei' not in df.columns": True,
    "(df.loc[df['track_id'] == 1, 'n_pathogens'] == 1).all()": True,
    "df.loc[df['track_id'] == 2, 'n_pathogens'].isna().all()": True,
    'np.allclose(cyto.loc[1], CELL1_AREA - PATH_AREA)': True,
    'np.allclose(cyto.loc[2], CELL2_AREA)': True,
}


def _case_group_cell_masks_only(tmp_path):
    """No nucleus and no pathogen planes -> no cytoplasm, no child summaries."""
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=False, with_pathogen=False, name="cell_only"
    )
    df = _process_merged_group((src, basenames, 2, 0, None, None))

    observed['len(df)'] = len(df)
    for col in ("n_nuclei", "n_pathogens", "n_cytoplasm",
                "nucleus_area", "pathogen_area", "cytoplasm_area"):
        assert col not in df.columns
    observed["(df['cell_area'] == CELL1_AREA).all()"] = bool((df['cell_area'] == CELL1_AREA).all())
    observed["'cell_mean_intensity_ch1' in df.columns"] = bool('cell_mean_intensity_ch1' in df.columns)
    return observed


_EXPECT_group_cell_masks_only = {
    'len(df)': 6,
    "(df['cell_area'] == CELL1_AREA).all()": True,
    "'cell_mean_intensity_ch1' in df.columns": True,
}


def _case_group_nucleus_covers_whole_cell_has_no_cytoplasm(tmp_path):
    """When the nucleus fills the cell the cytoplasm mask is empty."""
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(
        tmp_path, with_nucleus=True, with_pathogen=False,
        nucleus_fills_cell=True, name="no_cyto",
    )
    df = _process_merged_group((src, basenames, 2, 0, 0, None))

    observed['len(df)'] = len(df)
    observed["(df['n_nuclei'] == 1).all()"] = bool((df['n_nuclei'] == 1).all())
    observed["np.allclose(df['nucleus_area'], CELL1_AREA)"] = bool(np.allclose(df['nucleus_area'], CELL1_AREA))
    observed["'n_cytoplasm' not in df.columns"] = bool('n_cytoplasm' not in df.columns)
    observed["'cytoplasm_area' not in df.columns"] = bool('cytoplasm_area' not in df.columns)
    return observed


_EXPECT_group_nucleus_covers_whole_cell_has_no_cytoplasm = {
    'len(df)': 6,
    "(df['n_nuclei'] == 1).all()": True,
    "np.allclose(df['nucleus_area'], CELL1_AREA)": True,
    "'n_cytoplasm' not in df.columns": True,
    "'cytoplasm_area' not in df.columns": True,
}


def _case_group_without_intensity_channels(tmp_path):
    """n_channels=0: geometry only, no percentile or mean-intensity columns."""
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_channels=0, name="no_chans")
    arr = np.load(f"{src}/merged/{basenames[0]}")
    observed['arr.shape'] = arr.shape

    df = _process_merged_group((src, basenames, 0, 0, 0, 1))

    observed['len(df)'] = len(df)
    observed["(df['cell_area'] == CELL1_AREA).all()"] = bool((df['cell_area'] == CELL1_AREA).all())
    observed["(df['n_nuclei'] == 1).all()"] = bool((df['n_nuclei'] == 1).all())
    # no intensity information at all
    observed["not [c for c in df.columns if 'intensity' in c]"] = bool(not [c for c in df.columns if 'intensity' in c])
    observed["'cell_solidity' in df.columns"] = bool('cell_solidity' in df.columns)
    return observed


_EXPECT_group_without_intensity_channels = {
    'arr.shape': (3, H, W),
    'len(df)': 6,
    "(df['cell_area'] == CELL1_AREA).all()": True,
    "(df['n_nuclei'] == 1).all()": True,
    "not [c for c in df.columns if 'intensity' in c]": True,
    "'cell_solidity' in df.columns": True,
}


def _case_group_single_frame_group(tmp_path):
    observed = {}
    from spacr.timelapse import _process_merged_group

    src, basenames = _build_merged_group(tmp_path, n_frames=1, name="one_frame")
    df = _process_merged_group((src, basenames, 2, 0, 0, 1))

    observed['len(df)'] = len(df)
    observed["set(df['frame'])"] = set(df['frame'])
    observed["set(df['timeID'])"] = set(df['timeID'])
    return observed


_EXPECT_group_single_frame_group = {
    'len(df)': 2,
    "set(df['frame'])": {0},
    "set(df['timeID'])": {0},
}


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


def _case_smooth_empty_dataframe_returns_input():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = pd.DataFrame()
    out = _smooth_tracks_and_features(df)
    observed['out is df'] = bool(out is df)
    observed['out.empty'] = bool(out.empty)
    return observed


_EXPECT_smooth_empty_dataframe_returns_input = {
    'out is df': True,
    'out.empty': True,
}


def _case_smooth_missing_centroid_columns_is_noop():
    observed = {}
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

    observed["'cell_centroid-0' not in out.columns"] = bool('cell_centroid-0' not in out.columns)
    observed['len(out)'] = len(out)
    # returned sorted by (plate, well, field, cellID, frame)
    observed["out['frame'].tolist()"] = out['frame'].tolist()
    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    return observed


_EXPECT_smooth_missing_centroid_columns_is_noop = {
    "'cell_centroid-0' not in out.columns": True,
    'len(out)': 3,
    "out['frame'].tolist()": [0, 1, 2],
    "out['cell_area'].tolist()": [11.0, 12.0, 10.0],
}


def _case_smooth_interpolates_scalar_features_at_glitch_frame():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    observed['len(out)'] = len(out)
    # the scalar feature at the glitch frame is interpolated from its neighbours
    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    observed["out['cell_centroid-0'].tolist()"] = out['cell_centroid-0'].tolist()
    return observed


_EXPECT_smooth_interpolates_scalar_features_at_glitch_frame = {
    'len(out)': 3,
    "out['cell_area'].tolist()": [100.0, 110.0, 120.0],
    "out['cell_centroid-0'].tolist()": [10.0, 10.0, 10.0],
}


def _case_smooth_writes_back_glitch_centroid_for_float32_columns():
    """float32 centroids force ``to_numpy(dtype=float)`` to copy, so the
    write-back guard actually sees a difference and the fix lands in the
    returned frame."""
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0], ys=[10.0, 400.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    df["cell_centroid-0"] = df["cell_centroid-0"].astype(np.float32)
    df["cell_centroid-1"] = df["cell_centroid-1"].astype(np.float32)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    observed['len(out)'] = len(out)
    observed["out['cell_centroid-1'].tolist()"] = out['cell_centroid-1'].tolist()
    observed["out['cell_centroid-0'].tolist()"] = out['cell_centroid-0'].tolist()
    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    return observed


_EXPECT_smooth_writes_back_glitch_centroid_for_float32_columns = {
    'len(out)': 3,
    "out['cell_centroid-1'].tolist()": [10.0, 11.0, 12.0],
    "out['cell_centroid-0'].tolist()": [10.0, 11.0, 12.0],
    "out['cell_area'].tolist()": [100.0, 110.0, 120.0],
}


def _case_smooth_writes_back_glitch_corrected_centroid():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(7, xs=[10.0, 500.0, 12.0],
                      cell_area=[100.0, 999.0, 120.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    # the teleporting frame should be replaced by the midpoint of its neighbours
    observed["out['cell_centroid-1'].tolist()"] = out['cell_centroid-1'].tolist()
    return observed


_EXPECT_smooth_writes_back_glitch_corrected_centroid = {
    "out['cell_centroid-1'].tolist()": [10.0, 11.0, 12.0],
}


def _case_smooth_drops_track_with_unrecoverable_jump():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    bad = _track_frame(1, xs=[10.0, 10.0, 300.0])
    good = _track_frame(2, xs=[20.0, 21.0, 22.0])
    df = pd.concat([bad, good], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    observed["out['cellID'].unique().tolist()"] = out['cellID'].unique().tolist()
    observed['len(out)'] = len(out)
    observed["out['cell_centroid-1'].tolist()"] = out['cell_centroid-1'].tolist()
    # index was reset after the drop
    observed['out.index.tolist()'] = out.index.tolist()
    return observed


_EXPECT_smooth_drops_track_with_unrecoverable_jump = {
    "out['cellID'].unique().tolist()": [2],
    'len(out)': 3,
    "out['cell_centroid-1'].tolist()": [20.0, 21.0, 22.0],
    'out.index.tolist()': [0, 1, 2],
}


def _case_smooth_keeps_short_tracks_untouched():
    """1-frame tracks are skipped; 2-frame tracks are never dropped."""
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    one = _track_frame(1, xs=[5.0])
    two = _track_frame(2, xs=[5.0, 400.0])          # huge jump, but n < 3
    three = _track_frame(3, xs=[7.0, 8.0, 9.0])
    df = pd.concat([one, two, three], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    observed['len(out)'] = len(out)
    observed["sorted(out['cellID'].unique().tolist())"] = sorted(out['cellID'].unique().tolist())
    observed["out.loc[out['cellID'] == 2, 'cell_centroid-1'].tolist()"] = out.loc[out['cellID'] == 2, 'cell_centroid-1'].tolist()
    observed["out.loc[out['cellID'] == 1, 'cell_centroid-1'].tolist()"] = out.loc[out['cellID'] == 1, 'cell_centroid-1'].tolist()
    return observed


_EXPECT_smooth_keeps_short_tracks_untouched = {
    'len(out)': 6,
    "sorted(out['cellID'].unique().tolist())": [1, 2, 3],
    "out.loc[out['cellID'] == 2, 'cell_centroid-1'].tolist()": [5.0, 400.0],
    "out.loc[out['cellID'] == 1, 'cell_centroid-1'].tolist()": [5.0],
}


def _case_smooth_zscore_smoothing_of_scalar_features():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(
        1,
        xs=[20.0] * 5,
        cell_area=[100.0, 100.0, 1000.0, 100.0, 100.0],
        cell_solidity=[0.9] * 5,                        # std == 0 -> skipped
        cell_perimeter=[np.nan] * 5,                    # all non-finite -> skipped
        cell_max_intensity=[1.0, 2.0, np.nan, 8.0, 9.0],  # z is NaN at the spike
    )
    out = _smooth_tracks_and_features(df, max_displacement=50.0, track_outlier_zscore=1.5)

    observed['len(out)'] = len(out)
    # the |z| > 1.5 outlier is replaced by the mean of its neighbours
    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    # constant / all-NaN / NaN-at-centre columns are left alone
    observed["out['cell_solidity'].tolist()"] = out['cell_solidity'].tolist()
    observed["out['cell_perimeter'].isna().all()"] = bool(out['cell_perimeter'].isna().all())
    observed["out['cell_max_intensity'].isna().sum()"] = out['cell_max_intensity'].isna().sum()
    observed["out['cell_max_intensity'].tolist()[0]"] = out['cell_max_intensity'].tolist()[0]
    observed["out['cell_max_intensity'].tolist()[-1]"] = out['cell_max_intensity'].tolist()[-1]
    return observed


_EXPECT_smooth_zscore_smoothing_of_scalar_features = {
    'len(out)': 5,
    "out['cell_area'].tolist()": [100.0, 100.0, 100.0, 100.0, 100.0],
    "out['cell_solidity'].tolist()": [0.9] * 5,
    "out['cell_perimeter'].isna().all()": True,
    "out['cell_max_intensity'].isna().sum()": 1,
    "out['cell_max_intensity'].tolist()[0]": 1.0,
    "out['cell_max_intensity'].tolist()[-1]": 9.0,
}


def _case_smooth_zscore_leaves_mild_outliers_alone():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(1, xs=[20.0] * 5,
                      cell_area=[100.0, 100.0, 130.0, 100.0, 100.0])
    out = _smooth_tracks_and_features(df, max_displacement=50.0, track_outlier_zscore=3.0)

    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    return observed


_EXPECT_smooth_zscore_leaves_mild_outliers_alone = {
    "out['cell_area'].tolist()": [100.0, 100.0, 130.0, 100.0, 100.0],
}


def _case_smooth_casts_integer_columns_to_float():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    df = _track_frame(1, xs=[10, 11, 12], ys=[10, 10, 10],
                      cell_area=[100, 101, 102])
    observed["df['cell_area'].dtype.kind"] = df['cell_area'].dtype.kind

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    observed["out['cell_area'].dtype"] = out['cell_area'].dtype
    observed["out['cell_centroid-0'].dtype"] = out['cell_centroid-0'].dtype
    observed["out['cell_centroid-1'].dtype"] = out['cell_centroid-1'].dtype
    observed["out['cell_area'].tolist()"] = out['cell_area'].tolist()
    return observed


_EXPECT_smooth_casts_integer_columns_to_float = {
    "df['cell_area'].dtype.kind": 'i',
    "out['cell_area'].dtype": np.float64,
    "out['cell_centroid-0'].dtype": np.float64,
    "out['cell_centroid-1'].dtype": np.float64,
    "out['cell_area'].tolist()": [100.0, 101.0, 102.0],
}


def _case_smooth_glitch_in_one_track_does_not_affect_others():
    observed = {}
    from spacr.timelapse import _smooth_tracks_and_features

    glitchy = _track_frame(1, xs=[10.0, 500.0, 12.0], cell_area=[10.0, 90.0, 14.0])
    calm = _track_frame(2, xs=[30.0, 31.0, 32.0], cell_area=[20.0, 21.0, 22.0])
    df = pd.concat([calm, glitchy], ignore_index=True)

    out = _smooth_tracks_and_features(df, max_displacement=50.0)

    g = out[out["cellID"] == 1].sort_values("frame")
    c = out[out["cellID"] == 2].sort_values("frame")
    # the glitchy track has its middle-frame feature interpolated ...
    observed["g['cell_area'].tolist()"] = g['cell_area'].tolist()
    # ... while the calm track is left completely untouched
    observed["c['cell_centroid-1'].tolist()"] = c['cell_centroid-1'].tolist()
    observed["c['cell_area'].tolist()"] = c['cell_area'].tolist()
    # and neither track is dropped
    observed['len(out)'] = len(out)
    return observed


_EXPECT_smooth_glitch_in_one_track_does_not_affect_others = {
    "g['cell_area'].tolist()": [10.0, 12.0, 14.0],
    "c['cell_centroid-1'].tolist()": [30.0, 31.0, 32.0],
    "c['cell_area'].tolist()": [20.0, 21.0, 22.0],
    'len(out)': 6,
}


# ---------------------------------------------------------------------------
# the two functions composed: worker output feeds the smoother
# ---------------------------------------------------------------------------

def _case_group_output_is_smoothable(tmp_path):
    observed = {}
    from spacr.timelapse import _process_merged_group, _smooth_tracks_and_features

    src, basenames = _build_merged_group(tmp_path, n_frames=3, name="pipeline")
    df = _process_merged_group((src, basenames, 2, 0, 0, 1))

    smoothed = _smooth_tracks_and_features(df, max_displacement=50.0)

    # nothing in this synthetic movie jumps, so every row survives untouched
    observed['len(smoothed) == len(df)'] = bool(len(smoothed) == len(df))
    observed["sorted(smoothed['cellID'].unique().tolist())"] = sorted(smoothed['cellID'].unique().tolist())
    c1 = smoothed[smoothed["cellID"] == 1].sort_values("frame")
    observed["np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5])"] = bool(np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5]))
    observed["(smoothed['cell_area'] == float(CELL1_AREA)).all()"] = bool((smoothed['cell_area'] == float(CELL1_AREA)).all())
    return observed


_EXPECT_group_output_is_smoothable = {
    'len(smoothed) == len(df)': True,
    "sorted(smoothed['cellID'].unique().tolist())": [1, 2],
    "np.allclose(c1['cell_centroid-1'].to_numpy(), [11.5, 13.5, 15.5])": True,
    "(smoothed['cell_area'] == float(CELL1_AREA)).all()": True,
}


# ---------------------------------------------------------------------------
# behaviour tests: one per public-function contract, one parametrised row per case
# ---------------------------------------------------------------------------

def _run_case(case, tmp_path, monkeypatch):
    """Call a case helper with only the fixtures its signature asks for."""
    wanted = inspect.signature(case).parameters
    available = {"tmp_path": tmp_path, "monkeypatch": monkeypatch}
    return case(**{k: v for k, v in available.items() if k in wanted})


_P = functools.partial

GROUP_CASES = [
    pytest.param(_P(_case_group_cannot_be_measured, fault="no_basenames"), _EXPECT_group_cannot_be_measured, id="empty-frame-no-basenames"),
    pytest.param(_P(_case_group_cannot_be_measured, fault="first_array_2d"), _EXPECT_group_cannot_be_measured, id="empty-frame-2d-first-array"),
    pytest.param(_P(_case_group_cannot_be_measured, fault="reorient_fails"), _EXPECT_group_cannot_be_measured, id="empty-frame-reorientation-fails"),
    pytest.param(_P(_case_group_cannot_be_measured, fault="no_mask_planes"), _EXPECT_group_cannot_be_measured, id="empty-frame-no-mask-planes"),
    pytest.param(_P(_case_group_cannot_be_measured, fault="no_cell_props"), _EXPECT_group_cannot_be_measured, id="empty-frame-no-cell-regions"),
    pytest.param(_P(_case_full_feature_table, layout="planes_first"), _EXPECT_full_feature_table, id="full-table-planes-first"),
    pytest.param(_P(_case_full_feature_table, layout="channels_last"), _EXPECT_full_feature_table, id="full-table-channels-last"),
    pytest.param(_case_group_layouts_agree, _EXPECT_group_layouts_agree, id="both-layouts-give-the-same-table"),
    pytest.param(_case_group_nucleus_only, _EXPECT_group_nucleus_only, id="nucleus-only"),
    pytest.param(_case_group_pathogen_only, _EXPECT_group_pathogen_only, id="pathogen-only"),
    pytest.param(_case_group_cell_masks_only, _EXPECT_group_cell_masks_only, id="cell-masks-only"),
    pytest.param(_case_group_nucleus_covers_whole_cell_has_no_cytoplasm, _EXPECT_group_nucleus_covers_whole_cell_has_no_cytoplasm, id="nucleus-fills-cell-no-cytoplasm"),
    pytest.param(_case_group_without_intensity_channels, _EXPECT_group_without_intensity_channels, id="no-intensity-channels"),
    pytest.param(_case_group_single_frame_group, _EXPECT_group_single_frame_group, id="single-frame-group"),
]


@pytest.mark.parametrize("case, expected", GROUP_CASES)
def test_process_merged_group_builds_the_per_cell_per_frame_table_or_an_empty_frame(
        case, expected, tmp_path, monkeypatch):
    """Metadata, geometry, intensities and child summaries per cell and frame; empty frame when nothing can be measured."""
    observed = _run_case(case, tmp_path, monkeypatch)
    assert observed == expected


SMOOTH_CASES = [
    pytest.param(_case_smooth_empty_dataframe_returns_input, _EXPECT_smooth_empty_dataframe_returns_input, id="empty-frame-returned-as-is"),
    pytest.param(_case_smooth_missing_centroid_columns_is_noop, _EXPECT_smooth_missing_centroid_columns_is_noop, id="missing-centroids-only-sorts"),
    pytest.param(_case_smooth_interpolates_scalar_features_at_glitch_frame, _EXPECT_smooth_interpolates_scalar_features_at_glitch_frame, id="glitch-frame-features-interpolated"),
    pytest.param(_case_smooth_writes_back_glitch_centroid_for_float32_columns, _EXPECT_smooth_writes_back_glitch_centroid_for_float32_columns, id="float32-centroids-written-back"),
    pytest.param(_case_smooth_writes_back_glitch_corrected_centroid, _EXPECT_smooth_writes_back_glitch_corrected_centroid, id="glitch-centroid-midpoint"),
    pytest.param(_case_smooth_drops_track_with_unrecoverable_jump, _EXPECT_smooth_drops_track_with_unrecoverable_jump, id="unrecoverable-jump-drops-track"),
    pytest.param(_case_smooth_keeps_short_tracks_untouched, _EXPECT_smooth_keeps_short_tracks_untouched, id="short-tracks-untouched"),
    pytest.param(_case_smooth_zscore_smoothing_of_scalar_features, _EXPECT_smooth_zscore_smoothing_of_scalar_features, id="zscore-outlier-replaced-by-neighbour-mean"),
    pytest.param(_case_smooth_zscore_leaves_mild_outliers_alone, _EXPECT_smooth_zscore_leaves_mild_outliers_alone, id="zscore-mild-outlier-kept"),
    pytest.param(_case_smooth_casts_integer_columns_to_float, _EXPECT_smooth_casts_integer_columns_to_float, id="integer-columns-cast-to-float"),
    pytest.param(_case_smooth_glitch_in_one_track_does_not_affect_others, _EXPECT_smooth_glitch_in_one_track_does_not_affect_others, id="glitch-stays-inside-its-track"),
]


@pytest.mark.parametrize("case, expected", SMOOTH_CASES)
def test_smooth_tracks_repairs_glitches_drops_bad_tracks_and_smooths_outlier_features(case, expected):
    observed = _run_case(case, None, None)
    assert observed == expected


def test_process_merged_group_output_feeds_the_smoother_without_changes(tmp_path):
    observed = _case_group_output_is_smoothable(tmp_path)
    assert observed == _EXPECT_group_output_is_smoothable
