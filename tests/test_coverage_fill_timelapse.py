"""Coverage-fill for spacr.timelapse pure-logic tracking helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import timelapse as TL


def _moving_masks(n=4, size=24):
    masks = []
    for t in range(n):
        m = np.zeros((size, size), dtype=np.int32)
        m[2 + t:6 + t, 2:6] = 1        # object 1 drifts down
        m[14:18, 14 - t:18 - t] = 2    # object 2 drifts left
        masks.append(m)
    return np.stack(masks)


def _square_pair():
    m0 = np.zeros((16, 16), dtype=np.int32); m0[2:8, 2:8] = 1
    m1 = np.zeros((16, 16), dtype=np.int32); m1[3:9, 3:9] = 1
    return m0, m1


def _prepared_frame():
    frame0 = np.zeros((16, 16), dtype=np.int32); frame0[2:6, 2:6] = 1
    frame1 = np.zeros((16, 16), dtype=np.int32); frame1[3:7, 3:7] = 1
    return TL._prepare_for_tracking(np.stack([frame0, frame1]))


def _tracked():
    return TL._track_by_iou(_moving_masks(), iou_threshold=0.1)


def _relabelled():
    masks = _moving_masks()
    out = TL._relabel_masks_based_on_tracks(masks, TL._track_by_iou(masks, iou_threshold=0.1))
    return out.shape == masks.shape and out.dtype == masks.dtype


def _bool_pair(identical):
    m1 = np.zeros((4, 4), dtype=bool)
    m1[1:3, 1:3] = True
    if identical:
        return m1, m1
    a = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b = np.zeros((4, 4), dtype=bool)
    b[3, 3] = True
    return a, b


def _tag(plates, wells):
    return TL._infer_plate_well_meta_tag(pd.DataFrame({"plateID": plates, "wellID": wells}))


def _full_filename_meta():
    meta = TL._parse_merged_filename("plate1_B03_2_t005.npy")
    return {k: meta[k] for k in ("plateID", "wellID", "rowID", "columnID", "fieldID", "timeID", "prcft")}


def _minimal_filename_meta():
    meta = TL._parse_merged_filename("plateX.npy")
    return {k: meta[k] for k in ("plateID", "timeID", "fieldID")}


def _reorient(shape, n_channels):
    out, planes, H, W = TL._reorient_merged_array(np.zeros(shape, dtype=np.float32), n_channels=n_channels)
    return out.shape[0], planes, (H, W)


def _pathogen_columns():
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame({
        "plateID": ["p1"] * n, "rowID": ["r1"] * n,
        "columnID": ["c1"] * n, "fieldID": ["f1"] * n,
        "timeID": [f"t{i}" for i in rng.integers(0, 3, n)],
        "cell_id": rng.integers(1, 4, n),
        "object_label": range(n),
        "area": rng.uniform(10, 100, n),
    })
    out = TL.preprocess_pathogen_data(df)
    return "parasite_count" in out.columns, "object_label" in out.columns


_HELPER_CASES = [
    pytest.param(lambda: TL._sort_key("/x/1_A02_3_5.npy"), ("1", "A02", "3", 5), id="sort-key-parses-name"),
    pytest.param(lambda: TL._sort_key("garbage.npy"), ("", "", "", 0), id="sort-key-default-lowest"),
    pytest.param(lambda: set(_prepared_frame()["frame"]), {0, 1}, id="prepare-frames"),
    pytest.param(lambda: {"frame", "y", "x", "mass", "original_label"} <= set(_prepared_frame().columns), True, id="prepare-columns"),
    pytest.param(lambda: TL.link_by_iou(*_square_pair(), iou_threshold=0.1), [(1, 1)], id="link-by-iou"),
    pytest.param(lambda: {"frame", "original_label", "track_id"} <= set(_tracked().columns), True, id="track-by-iou-columns"),
    pytest.param(lambda: isinstance(TL._filter_short_tracks(_tracked(), min_length=3), pd.DataFrame), True, id="filter-short-tracks"),
    pytest.param(lambda: TL.link_by_iou(*_bool_pair(False), iou_threshold=0.1), [],
                 id="link-by-iou-disjoint-masks-never-pair"),
    pytest.param(lambda: TL.link_by_iou(*_bool_pair(True), iou_threshold=0.1), [(True, True)],
                 id="link-by-iou-identical-masks-pair"),
    pytest.param(lambda: TL.exponential_decay(0, 2.0, 1.0, 0.5), pytest.approx(2.5), id="exponential-decay-at-zero"),
    pytest.param(lambda: _tag(["plate1"], ["A02"]), "plate1_A02", id="tag-single"),
    pytest.param(lambda: _tag(["p1", "p1"], ["A01", "A02"]), "p1_MULTI_WELLS", id="tag-multi-wells"),
    pytest.param(lambda: _tag(["p1", "p2"], ["A01", "A01"]), "MULTI_PLATES_A01", id="tag-multi-plates"),
    pytest.param(lambda: _tag(["p1", "p2"], ["A01", "A02"]), "MULTI_PLATES_MULTI_WELLS", id="tag-multi-both"),
    pytest.param(_full_filename_meta,
                 {"plateID": "plate1", "wellID": "B03", "rowID": "B", "columnID": 3,
                  "fieldID": "2", "timeID": 5, "prcft": "plate1_B03_2_5"}, id="parse-full-filename"),
    pytest.param(_minimal_filename_meta, {"plateID": "plateX", "timeID": 0, "fieldID": "1"}, id="parse-minimal-filename"),
    pytest.param(_relabelled, True, id="relabel-keeps-shape-and-dtype"),
    pytest.param(lambda: _reorient((4, 16, 16), 3), (4, 4, (16, 16)), id="reorient-planes-first"),
    pytest.param(lambda: _reorient((16, 16, 4), 3), (4, 4, (16, 16)), id="reorient-planes-last"),
    pytest.param(lambda: _reorient((2, 16, 16), 8), (2, 2, (16, 16)), id="reorient-smallest-axis-fallback"),
    pytest.param(_pathogen_columns, (True, True), id="pathogen-adds-count-and-renames-cell-id"),
]


@pytest.mark.parametrize("compute, expected", _HELPER_CASES)
def test_pure_tracking_helpers_return_the_documented_values(compute, expected):
    assert compute() == expected


def test_two_dimensional_input_is_refused_and_figure_and_results_helpers_write_into_results(tmp_path):
    import matplotlib.pyplot as plt
    with pytest.raises(ValueError):
        TL._reorient_merged_array(np.zeros((16, 16)), n_channels=3)
    fig, ax = plt.subplots(); ax.plot([0, 1], [0, 1])
    src = str(tmp_path / "run" / "data")
    (tmp_path / "run").mkdir()
    TL.save_figure(fig, src, 1)
    plt.close(fig)
    assert (tmp_path / "run" / "results" / "figure_1.pdf").exists()
    TL.save_results_dataframe(pd.DataFrame({"a": [1, 2]}), src, "res")
    assert (tmp_path / "run" / "results" / "res.csv").exists()
