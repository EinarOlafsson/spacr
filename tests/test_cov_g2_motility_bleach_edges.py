"""Motility bleach correction edges around the real assay and its worker."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import timelapse as tl
from tests.test_motility_bleaching_f539 import close_figures, field  # noqa: F401


def test_the_worker_refuses_an_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="Unknown motility bleach correction"):
        tl._process_merged_group(field(tmp_path) + ("sepia",))


def test_the_worker_without_a_cell_channel_still_corrects(tmp_path):
    src, names, n_channels, _cell, nucleus, pathogen = field(tmp_path)
    result = tl._process_merged_group((src, names, n_channels, None, nucleus,
                                       pathogen, "ratio"))
    assert isinstance(result, tuple) or isinstance(result, pd.DataFrame)


def _settings(tmp_path, method):
    return dict(src=str(tmp_path), channels=[0, 1], cell_channel=0,
                nucleus_channel=None, pathogen_channel=1, n_jobs=1,
                bleach_correction=method, make_mask_panel=False,
                make_adjusted_panel=False, infection_intensity_strategy="none",
                infection_intensity_qc=False)


def test_histogram_matching_runs_beside_an_unrelated_database(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    field(tmp_path)
    (tmp_path / "measurements").mkdir()
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as con:
        con.execute("CREATE TABLE other (x INTEGER)")
    monkeypatch.setattr(tl, "_debug_plot_merged_planes", lambda **kwargs: None)
    monkeypatch.setattr(tl, "_feature_velocity_correlations", lambda *args: None)
    out = tl.automated_motility_assay(_settings(tmp_path, "histogram"))
    assert len(out)
    assert "do not interpret these values as quantitative" in capsys.readouterr().out


def test_an_assay_whose_workers_return_nothing_corrected(tmp_path, monkeypatch):
    field(tmp_path)
    monkeypatch.setattr(tl, "_process_merged_group",
                        lambda args: pd.DataFrame())
    monkeypatch.setattr(tl, "_debug_plot_merged_planes", lambda **kwargs: None)
    settings = _settings(tmp_path, "ratio")
    settings["reuse_existing_measurements"] = False
    with pytest.raises(RuntimeError, match="No measurements were produced"):
        tl.automated_motility_assay(settings)


def test_a_worker_with_no_intensities_or_fits_still_records_its_files(
        tmp_path, monkeypatch):
    monkeypatch.setattr(tl, "_compute_cell_mean_intensity_per_channel",
                        lambda **kwargs: pd.DataFrame())

    def no_fit(props, masks, stack, role, channel, times, method):
        return props, pd.DataFrame(), set()

    monkeypatch.setattr(tl, "_motility_role_bleaching", no_fit)
    corrected, raw, fits = tl._process_merged_group(field(tmp_path) + ("ratio",))
    assert list(fits.columns[:3]) == ["object_type", "channel", "method"]
    assert fits.empty
