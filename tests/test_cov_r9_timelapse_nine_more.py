"""Decisions in ``timelapse.py`` that need a driven run to see.

The merged preview caps at three colour channels, the QC panels describe the
first processed group, and the XGBoost panel matrix is owned rather than a
view of the measurements.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as T

_XGB_PATHOGEN_CHAN = 1


def _xgb_frame(n_per_class=18, wells=("A01", "A02"), n_frames=3, seed=3):
    """A frame-level table with a clean infected/uninfected separation.

    Shaped like the measurement table ``_infection_qc_xgboost`` reads:
    one tracked object per spec, repeated over ``n_frames`` frames, with
    an exact pathogen-channel p95 so the quartile thresholds are stable.
    """
    rng = np.random.default_rng(seed)
    chan = _XGB_PATHOGEN_CHAN
    rows = []
    cell_id = 0
    for well in wells:
        for _ in range(n_per_class):
            for infected, centre in ((True, 1000.0), (False, 300.0)):
                cell_id += 1
                intensity = float(rng.normal(centre, 120.0))
                area = float(rng.uniform(200.0, 900.0)) + (
                    300.0 if infected else 0.0)
                solidity = float(rng.uniform(0.70, 0.99))
                y0 = float(rng.uniform(10.0, 200.0))
                x0 = float(rng.uniform(10.0, 200.0))
                for frame_index in range(n_frames):
                    rows.append({
                        "plateID": "plate1", "wellID": well, "fieldID": "1",
                        "cellID": cell_id, "frame": frame_index,
                        "infected": bool(infected),
                        "n_pathogens": 3 if infected else 0,
                        f"cell_p95_intensity_ch{chan}": intensity,
                        f"cell_mean_intensity_ch{chan}": intensity * 0.6,
                        "cell_mean_intensity_ch0": float(
                            rng.uniform(100.0, 200.0)),
                        "cell_area": area,
                        "cell_perimeter": 0.4 * area,
                        "cell_solidity": solidity,
                        "cell_centroid-0": y0 + 1.5 * frame_index,
                        "cell_centroid-1": x0 + float(frame_index),
                        "nucleus_area": float(rng.uniform(50.0, 200.0)),
                    })
    return pd.DataFrame(rows)


def _write_merged(tmp_path, n_channels):
    """A merged file with n_channels distinct intensity planes and one mask plane."""
    planes = [np.full((16, 16), 10.0 * (i + 1)) + np.arange(16)[None, :] * (i + 1)
              for i in range(n_channels)]
    mask = np.zeros((16, 16))
    mask[4:8, 4:8] = 1
    arr = np.stack(planes + [mask], axis=-1)  # planes last
    (tmp_path / "merged").mkdir()
    np.save(tmp_path / "merged" / "s.npy", arr)


@pytest.mark.parametrize("n_channels, expected_filled", [
    pytest.param(1, [0], id="one-channel-leaves-green-and-blue-black"),
    pytest.param(2, [0, 1], id="two-channels-leave-blue-black"),
    pytest.param(3, [0, 1, 2], id="three-channels-fill-rgb"),
    pytest.param(5, [0, 1, 2], id="five-channels-still-cap-at-three"),
])
def test_the_merged_preview_fills_only_the_channels_that_exist_and_caps_at_three(
        tmp_path, monkeypatch, n_channels, expected_filled):
    """A preview merges up to three channels into RGB, and a stack with fewer
    leaves the rest black rather than repeating a channel, which would show a
    one-channel field as grey and read as a colour balance. A fourth channel
    must never be written into the RGB array."""
    _write_merged(tmp_path, n_channels)
    captured = {}

    def _grab(fig, path, **kwargs):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(T, "save_figure_to_path", _grab)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    T._debug_plot_merged_planes(str(tmp_path), "s.npy", n_channels, 0, 1, str(out_dir))

    merged_ax = captured["fig"].axes[-1]
    assert merged_ax.get_title() == "Merged channels + masks"
    rgb = np.asarray(merged_ax.images[0].get_array())
    assert rgb.shape == (16, 16, 3)
    for index in range(3):
        assert bool(rgb[..., index].any()) == (index in expected_filled)
    # one panel per intensity channel, plus one mask panel and the merged panel
    assert len(captured["fig"].axes) == n_channels + 2


def test_the_payload_comes_from_the_first_group_not_the_last(tmp_path, monkeypatch):
    """The QC panels describe ONE group. Keeping the FIRST processed one is
    the choice made; keeping the last would mean the panel changed depending
    on how the groups happened to be ordered."""
    processed = []

    def _recording_qc(all_df, settings, infection_col, pathogen_chan,
                      motility_dir):
        plate = str(all_df["plateID"].iloc[0])
        processed.append(plate)
        settings["infection_hist_data"] = {"plate": plate}
        settings["infection_intensity_qc_panel_type"] = plate
        return all_df, infection_col

    monkeypatch.setattr(T, "_infection_qc_histogram", _recording_qc)

    # plateB FIRST in the frame and second alphabetically, so the
    # first processed group is neither the last nor the sorted one.
    frame = pd.DataFrame({
        "plateID": ["plateB"] * 3 + ["plateA"] * 3,
        "wellID": ["A01"] * 6,
        "infected": [True, False, True, False, True, False]})
    settings = {"infection_intensity_qc": True,
                "infection_intensity_strategy": "histogram",
                "infection_intensity_qc_scope": "plate"}

    T._apply_infection_intensity_qc(
        frame, settings, "infected", 1, str(tmp_path / "motility"))

    assert processed == ["plateB", "plateA"]
    assert settings["infection_hist_data"] == {"plate": "plateB"}, (
        "the QC payload is no longer the first processed group, so "
        "which group the panel describes now depends on group order")
    assert settings["infection_intensity_qc_panel_type"] == "plateB"


def test_the_xgboost_panel_matrix_is_owned_rather_than_viewed(tmp_path):
    """A pandas-3 trap worth keeping: a homogeneous selection can
    return a READ-ONLY view, and the display-only imputation below
    writes into the matrix it is given.

    Driven on what owning it buys. The features carry a hole, so the
    imputation has to run: on an owned matrix it fills the hole and
    the PCA payload is produced, and on a read-only view it would
    raise into the enclosing ``except`` and leave the panel empty.
    The imputation is display-only either way: the measurements
    come back still carrying their hole.
    """
    all_df = _xgb_frame()
    holed = sorted(all_df["cellID"].unique())[::2]
    all_df.loc[all_df["cellID"].isin(holed), "cell_solidity"] = np.nan
    settings = {"tracked_object": "cell",
                "infection_xgb_n_estimators": 15,
                "infection_xgb_max_depth": 2,
                "infection_xgb_n_jobs": 1,
                "infection_intensity_mode": "relabel"}

    # COPY-ON-WRITE ON, DELIBERATELY. The assertions below describe what
    # owning the matrix buys, and they can only describe it in the mode
    # where a selection returns a read-only view. With copy-on-write off the
    # imputation writes into a doomed temporary, nothing raises, and the
    # payload appears whether the matrix was copied or not.
    with pd.option_context("mode.copy_on_write", True):
        out, _ = T._infection_qc_xgboost(
            all_df=all_df, settings=settings, infection_col="infected",
            pathogen_chan=_XGB_PATHOGEN_CHAN,
            motility_dir=str(tmp_path / "motility"))

    payload = settings.get("infection_pca_data")
    assert payload is not None, (
        "the PCA panel payload was lost, which is what a read-only "
        "panel matrix costs: the imputation raises into the enclosing "
        "except and the run finishes with no panel")
    assert np.isfinite(payload["coords"]).all(), (
        "the imputation did not fill the hole it was there to fill")

    survivors = out[out["cellID"].isin(holed)]
    assert len(survivors) > 0
    assert survivors["cell_solidity"].isna().all(), (
        "the display-only imputation wrote back into the "
        "measurements, so the matrix is a view of them after all")
