"""Bleach correction edges: no area, failed fits, empty inputs, odd tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl
from spacr.tabular import write_database
from tests.test_bleach_correction import _bleaching_table, _offset_series


@pytest.mark.parametrize("method", ["ratio", "histogram"])
def test_an_integrated_level_without_an_area_is_rescaled_whole(method):
    df = _offset_series().drop(columns=["nucleus_area"])
    corrected, fits = tl._bleach_correct_table(df, "nucleus", method)
    column = "nucleus_channel_1_integrated_intensity"
    assert corrected[column].notna().all() and len(fits) == 1


def test_a_failed_or_non_finite_decay_fit_is_none(monkeypatch):
    times, trend = [0, 1, 2, 3], [10.0, 8.0, 6.5, 5.0]

    def broken(*a, **k):
        raise RuntimeError("no convergence")

    monkeypatch.setattr(tl, "curve_fit", broken)
    assert tl._fit_bleach_decay(times, trend) is None
    monkeypatch.setattr(tl, "curve_fit",
                        lambda *a, **k: (np.array([np.nan, 1.0, 0.0]), None))
    assert tl._fit_bleach_decay(times, trend) is None


def test_matching_nothing_finite_gives_nan():
    out = tl._histogram_match(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert np.isnan(out).all()


def test_a_channel_without_its_mean_level_is_not_corrected():
    df = pd.DataFrame({"cell_channel_2_max_intensity": [1.0],
                       "cell_channel_1_mean_intensity": [1.0]})
    assert list(tl._bleach_channel_columns(df, "cell")) == [1]


def test_tables_without_time_are_skipped_and_empty_fits_draw_nothing(
        tmp_path, monkeypatch):
    db = tmp_path / "plate1" / "measurements" / "measurements.db"
    write_database(_bleaching_table(n_frames=4, n_objects=5), db, "cell",
                   if_exists="replace", canonicalise=False)
    write_database(pd.DataFrame({"nucleus_channel_0_mean_intensity": [1.0]}),
                   db, "nucleus", if_exists="replace", canonicalise=False)
    real = tl._bleach_correct_table

    def no_fits(df, table, method):
        corrected, fits = real(df, table, method)
        return corrected, fits.iloc[0:0]

    monkeypatch.setattr(tl, "_bleach_correct_table", no_fits)
    tl._correct_timelapse_bleaching(str(db), "ratio", plot=True)
    assert not (tmp_path / "plate1" / "results" / "bleach_correction").exists()


def test_calcium_correction_needs_the_channel_mean():
    frame = pd.DataFrame({"cell_channel_1_max_intensity": [1.0]})
    with pytest.raises(ValueError, match="selected level and its channel mean"):
        tl._calcium_shared_bleaching(frame, "cell_channel_1_max_intensity",
                                     "ratio")
