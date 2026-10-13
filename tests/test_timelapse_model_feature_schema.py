"""Regression tests for the model-feature boundary in timelapse QC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _numeric_text_frame(*, wells=("A01",), n_per_class=24):
    """Return realistic frame-level rows as pandas reads mixed SQLite types."""
    rows = []
    cell_id = 0
    for well in wells:
        for infected in (False, True):
            for offset in range(n_per_class):
                cell_id += 1
                intensity = (100.0 if not infected else 1000.0) + offset
                area = (200.0 if not infected else 600.0) + 2 * offset
                rows.append(
                    {
                        "plateID": "plate1",
                        "wellID": well,
                        "fieldID": "1",
                        "cellID": cell_id,
                        "frame": 0,
                        "infected": infected,
                        "n_pathogens": 2 if infected else 0,
                        # SQLite may contain numeric measurements as text.
                        "cell_p95_intensity_ch1": f"{intensity:.1f}",
                        "cell_area": f"{area:.1f}",
                        # An all-NULL REAL column is also read back as object.
                        "cell_empty_measurement": None,
                        # This is a different channel, so it is not a candidate
                        # for pathogen channel 1 and must not trigger an error.
                        "cell_mean_intensity_ch0": "not measured",
                        # Ordinary metadata must survive byte-for-byte.
                        "acquisition_note": f"{well}:keep-as-text",
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("scope", ["combined", "well"])
def test_pca_qc_uses_numeric_text_without_mutating_measurements(
    scope, tmp_path
):
    """Both live dispatcher scopes coerce candidates, not the durable table."""
    from spacr.timelapse import _apply_infection_intensity_qc

    frame = _numeric_text_frame(wells=("A01", "A02"))
    original = frame.copy(deep=True)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": "pca",
        "infection_intensity_qc_scope": scope,
        "infection_intensity_mode": "relabel",
    }

    with pytest.warns(UserWarning, match="stored as text") as warnings_seen:
        out, infection_col = _apply_infection_intensity_qc(
            all_df=frame,
            settings=settings,
            infection_col="infected",
            pathogen_chan=1,
            motility_dir=str(tmp_path / scope),
        )

    warning_text = "\n".join(str(item.message) for item in warnings_seen)
    assert "cell_p95_intensity_ch1" in warning_text
    assert "cell_area" in warning_text
    assert "cell_mean_intensity_ch0" not in warning_text
    assert infection_col == "adjusted_infected"
    assert out["adjusted_infected"].notna().all()
    assert set(out["wellID"]) == {"A01", "A02"}
    assert settings["infection_pca_data"]["coords"].shape[1] == 2

    # Coercion is model-local: neither the input nor the returned measurement
    # and metadata columns are rewritten as a side effect of fitting.
    pd.testing.assert_frame_equal(frame, original)
    assert out["cell_p95_intensity_ch1"].dtype == \
        original["cell_p95_intensity_ch1"].dtype
    assert out["cell_empty_measurement"].dtype == \
        original["cell_empty_measurement"].dtype
    assert out["cell_empty_measurement"].isna().all()
    assert out["acquisition_note"].tolist() == original["acquisition_note"].tolist()
    assert (out["cell_mean_intensity_ch0"] == "not measured").all()


@pytest.mark.parametrize("strategy", ["pca", "xgboost"])
def test_timelapse_qc_rejects_invalid_actual_candidate_with_schema_error(
    strategy, tmp_path
):
    """Malformed model inputs fail with the same actionable Classify error."""
    from spacr import schema
    from spacr.timelapse import _apply_infection_intensity_qc

    frame = _numeric_text_frame()
    frame.loc[frame.index[0], "cell_area"] = "not-a-number"
    original = frame.copy(deep=True)
    settings = {
        "infection_intensity_qc": True,
        "infection_intensity_strategy": strategy,
        "infection_intensity_qc_scope": "combined",
        "tracked_object": "cell",
    }

    with pytest.raises(schema.ModelFeatureSchemaError) as raised:
        _apply_infection_intensity_qc(
            all_df=frame,
            settings=settings,
            infection_col="infected",
            pathogen_chan=1,
            motility_dir=str(tmp_path / strategy),
        )

    message = str(raised.value)
    assert "cell_area" in message
    assert "not-a-number" in message
    assert "'Exclude' setting" in message
    assert "cell_mean_intensity_ch0" not in message
    pd.testing.assert_frame_equal(frame, original)


def test_xgboost_one_numeric_text_feature_keeps_a_2d_pca_payload(tmp_path):
    """A valid one-feature classifier still supplies the two-axis QC panel."""
    from spacr.timelapse import _infection_qc_xgboost

    frame = _numeric_text_frame().drop(columns=["cell_area"])
    original = frame.copy(deep=True)
    settings = {
        "tracked_object": "cell",
        "infection_xgb_n_estimators": 4,
        "infection_xgb_max_depth": 2,
        "infection_xgb_n_jobs": 1,
        "infection_xgb_min_cells_per_class": 2,
        "infection_xgb_drop_ambiguous": False,
        "infection_intensity_mode": "relabel",
    }

    with pytest.warns(UserWarning, match="cell_p95_intensity_ch1"):
        out, infection_col = _infection_qc_xgboost(
            all_df=frame,
            settings=settings,
            infection_col="infected",
            pathogen_chan=1,
            motility_dir=str(tmp_path),
        )

    assert infection_col == "adjusted_infected"
    assert settings["infection_xgb_importance"]["feature_names"] == [
        "cell_p95_intensity_ch1"
    ]
    coords = settings["infection_pca_data"]["coords"]
    assert coords.shape == (len(frame), 2)
    assert np.isfinite(coords).all()
    np.testing.assert_array_equal(coords[:, 1], np.zeros(len(frame)))

    pd.testing.assert_frame_equal(frame, original)
    assert out["cell_p95_intensity_ch1"].dtype == \
        original["cell_p95_intensity_ch1"].dtype
    assert out["cell_empty_measurement"].dtype == \
        original["cell_empty_measurement"].dtype
    assert out["acquisition_note"].tolist() == original["acquisition_note"].tolist()
