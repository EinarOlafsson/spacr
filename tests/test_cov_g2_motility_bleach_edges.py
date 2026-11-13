"""Motility bleach correction edges around the real assay and its worker."""
from __future__ import annotations

import inspect
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import timelapse as tl
from tests.test_motility_bleaching_f539 import close_figures, field  # noqa: F401


def _scn_the_worker_refuses_an_unknown_method(tmp_path):
    _obs = []
    with pytest.raises(ValueError, match="Unknown motility bleach correction") as _raised:
        tl._process_merged_group(field(tmp_path) + ("sepia",))
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    return _obs


def _scn_the_worker_without_a_cell_channel_still_corrects(tmp_path):
    _obs = []
    src, names, n_channels, _cell, nucleus, pathogen = field(tmp_path)
    result = tl._process_merged_group((src, names, n_channels, None, nucleus,
                                       pathogen, "ratio"))
    assert isinstance(result, tuple) or isinstance(result, pd.DataFrame)
    _obs.append(_seen((isinstance(result, tuple) or isinstance(result, pd.DataFrame)), (isinstance(result, tuple) or isinstance(result, pd.DataFrame))))
    return _obs


def _settings(tmp_path, method):
    return dict(src=str(tmp_path), channels=[0, 1], cell_channel=0,
                nucleus_channel=None, pathogen_channel=1, n_jobs=1,
                bleach_correction=method, make_mask_panel=False,
                make_adjusted_panel=False, infection_intensity_strategy="none",
                infection_intensity_qc=False)


def _scn_histogram_matching_runs_beside_an_unrelated_database(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    _obs = []
    field(tmp_path)
    (tmp_path / "measurements").mkdir()
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as con:
        con.execute("CREATE TABLE other (x INTEGER)")
    monkeypatch.setattr(tl, "_debug_plot_merged_planes", lambda **kwargs: None)
    monkeypatch.setattr(tl, "_feature_velocity_correlations", lambda *args: None)
    out = tl.automated_motility_assay(_settings(tmp_path, "histogram"))
    assert len(out)
    _obs.append(_seen((len(out)), (len(out))))
    assert "do not interpret these values as quantitative" in capsys.readouterr().out
    _obs.append(_seen(("do not interpret these values as quantitative" in capsys.readouterr().out), ("do not interpret these values as quantitative" in capsys.readouterr().out)))
    return _obs


def _scn_an_assay_whose_workers_return_nothing_corrected(tmp_path, monkeypatch):
    _obs = []
    field(tmp_path)
    monkeypatch.setattr(tl, "_process_merged_group",
                        lambda args: pd.DataFrame())
    monkeypatch.setattr(tl, "_debug_plot_merged_planes", lambda **kwargs: None)
    settings = _settings(tmp_path, "ratio")
    settings["reuse_existing_measurements"] = False
    with pytest.raises(RuntimeError, match="No measurements were produced") as _raised:
        tl.automated_motility_assay(settings)
    _obs.append(_seen(type(_raised.value).__name__ + ": " + str(_raised.value), True))
    return _obs


def _scn_a_worker_with_no_intensities_or_fits_still_records_its_files(
        tmp_path, monkeypatch):
    _obs = []
    monkeypatch.setattr(tl, "_compute_cell_mean_intensity_per_channel",
                        lambda **kwargs: pd.DataFrame())

    def no_fit(props, masks, stack, role, channel, times, method):
        return props, pd.DataFrame(), set()

    monkeypatch.setattr(tl, "_motility_role_bleaching", no_fit)
    corrected, raw, fits = tl._process_merged_group(field(tmp_path) + ("ratio",))
    assert list(fits.columns[:3]) == ["object_type", "channel", "method"]
    _obs.append(_seen((list(fits.columns[:3])), (list(fits.columns[:3]) == ["object_type", "channel", "method"])))
    assert fits.empty
    _obs.append(_seen((fits.empty), (fits.empty)))
    return _obs


def _plain(value):
    """Return a plain comparable copy of a value, or raise ValueError."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return re.sub(r"\S*pytest-of-[^/]+/pytest-\d+/(?:popen-gw\d+/)?[^/\s]+", "<tmp>", value.replace("\\", "/"))
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    if isinstance(value, (list, tuple)) and len(value) <= 60:
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)) and len(value) <= 60:
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, dict) and len(value) <= 60:
        return {str(k): _plain(v) for k, v in value.items()}
    raise ValueError(type(value))


def _seen(value, held):
    """Observed value of one check: the value itself when plain, else whether the check held."""
    try:
        return _plain(value)
    except ValueError:
        return bool(held)


def _run_scenario(request, scenario, kwargs):
    """Run one scenario function, resolving its fixtures by name."""
    wanted = [n for n in inspect.signature(scenario).parameters if n not in kwargs]
    fixtures = {n: request.getfixturevalue(n) for n in wanted}
    return scenario(**fixtures, **kwargs)


@pytest.mark.parametrize("scenario, kwargs, expected", [
    pytest.param(_scn_the_worker_refuses_an_unknown_method, {}, ["ValueError: Unknown motility bleach correction: 'sepia'"], id='the-worker-refuses-an-unknown-method'),
    pytest.param(_scn_the_worker_without_a_cell_channel_still_corrects, {}, [True], id='the-worker-without-a-cell-channel-still-corrects'),
    pytest.param(_scn_histogram_matching_runs_beside_an_unrelated_database, {}, [6, False], id='histogram-matching-runs-beside-an-unrelated-database'),
    pytest.param(_scn_an_assay_whose_workers_return_nothing_corrected, {}, ['RuntimeError: No measurements were produced from merged .npy files.'], id='an-assay-whose-workers-return-nothing-corrected'),
    pytest.param(_scn_a_worker_with_no_intensities_or_fits_still_records_its_files, {}, [['object_type', 'channel', 'method'], True], id='a-worker-with-no-intensities-or-fits-still-records-its-files'),
])
def test_motility_bleach_correction_refuses_unknown_methods_and_survives_missing_intensities(request, scenario, kwargs, expected):
    observed = _run_scenario(request, scenario, kwargs)
    assert observed == expected
