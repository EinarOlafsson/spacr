"""The per-field Measure worker applies what the parent run recorded -- an
intensity calibration plan, an unmixing matrix -- and honours the settings
that change what it measures; the run itself hands a cloud source on and
runs its end-of-run steps (bleach correction, time to event, profiles)
without letting one of them fail the run.

The worker is called in this process, as the pool would call it, so what it
writes can be read back directly."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m


def _rows(db, sql):
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(sql, conn)


def test_the_worker_applies_the_calibration_plan_the_run_built(tmp_path):
    from spacr.intensity_rescale import CALIBRATION_SETTINGS_KEY
    from tests.test_intensity_calibration import _settings, _write_sessions

    merged = _write_sessions(tmp_path)
    (tmp_path / "measurements").mkdir()
    settings = _settings(merged, intensity_calibration=True)
    files = sorted(p.name for p in merged.glob("*.npy"))
    settings[CALIBRATION_SETTINGS_KEY] = m._build_intensity_calibration_plan(
        str(merged), files, settings)
    _i, _t, _c, _f, error = m._measure_crop_core(0, [], "plate2_B02_1.npy",
                                                 settings)
    assert not error
    db = tmp_path / "measurements" / "measurements.db"
    record = json.loads(_rows(db, "SELECT * FROM intensity_rescale")[
        "intensity_calibration"].iloc[0])
    assert record["plateID"] == "plate2"
    assert record["gain"]["0"] == pytest.approx(1 / 2.5, rel=0.02)


def test_a_calibration_that_saturates_the_field_says_so(tmp_path, capsys):
    from spacr.intensity_rescale import CALIBRATION_SETTINGS_KEY
    from tests.test_intensity_calibration import _settings, _write_sessions

    merged = _write_sessions(tmp_path)
    (tmp_path / "measurements").mkdir()
    settings = _settings(merged, intensity_calibration=True)
    files = sorted(p.name for p in merged.glob("*.npy"))
    plan = m._build_intensity_calibration_plan(str(merged), files, settings)
    for channel in plan["plates"]["plate2"]["gain"]:
        plan["plates"]["plate2"]["gain"][channel] = 1000.0
    settings[CALIBRATION_SETTINGS_KEY] = plan
    _i, _t, _c, _f, error = m._measure_crop_core(0, [], "plate2_B02_1.npy",
                                                 settings)
    assert not error
    assert ("WARNING: plate2_B02_1 intensity calibration clipped pixels"
            in capsys.readouterr().out)


def test_the_worker_unmixes_with_the_matrix_the_run_recorded(tmp_path):
    from spacr.psf_pipeline import _prepare_measure_unmixing
    from spacr.settings import get_measure_crop_settings
    from tests.test_spectral_unmixing import SHAPE, _dye, _field

    rng = np.random.default_rng(538)
    merged = tmp_path / "merged"
    merged.mkdir()
    (tmp_path / "measurements").mkdir()
    for index in (1, 2):
        np.save(merged / f"plate1_A01_{index}.npy",
                _field(rng, [_dye(rng, 2000.0), np.zeros(SHAPE)])[0])
        np.save(merged / f"plate1_B01_{index}.npy",
                _field(rng, [np.zeros(SHAPE), _dye(rng, 1500.0)])[0])
    field = _field(rng, [_dye(rng, 1800.0), np.zeros(SHAPE)])[0]
    np.save(merged / "plate1_C01_1.npy", field)
    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1], "cell_mask_dim": 2,
        "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": False, "verbose": False, "n_jobs": 1, "unmix": True,
        "unmix_controls": "0:A01; 1:B01"})
    assert _prepare_measure_unmixing(settings) is not None
    _i, _t, _c, _f, error = m._measure_crop_core(0, [], "plate1_C01_1.npy",
                                                 settings)
    assert not error
    cells = _rows(tmp_path / "measurements" / "measurements.db",
                  "SELECT * FROM cell")
    unmixed = float(cells["cell_channel_1_mean_intensity"].iloc[0])
    assert unmixed < float(field[..., 1].mean()), (
        "channel 1's bleed-through from dye 0 is taken out")


@pytest.fixture
def stack(tmp_path, synth_masks_multi, rng):
    from tests.test_measure_crop_core_synth import (_build_merged_stack,
                                                     _write_stack)

    data = _build_merged_stack(synth_masks_multi, rng, with_organelle=True)
    return _write_stack(tmp_path, data)


def test_field_settings_that_change_what_the_worker_measures(tmp_path,
                                                             stack, capsys):
    from tests.test_measure_crop_core_synth import _settings_for

    merged, name = stack
    settings = _settings_for(
        merged, normalize=False, normalize_by="fov", organelle_mask_dim=7,
        organelle_min_area=5, organelle_min_size=5, confluency=True, confluency_source="masks",
        verbose=False, plot=False)
    _i, _t, _c, figs, error = m._measure_crop_core(0, [], name, settings)
    assert not error and not figs
    db = tmp_path / "measurements" / "measurements.db"
    assert len(_rows(db, "SELECT * FROM organelle")) > 0
    assert len(_rows(db, f"SELECT * FROM {m._CONFLUENCY_TABLE}")) == 1
    assert "covered (masks)" not in capsys.readouterr().out


def test_nuclei_are_tracked_on_their_own_without_a_cell_mask(tmp_path, stack):
    from tests.test_measure_crop_core_synth import _settings_for

    merged, name = stack
    settings = _settings_for(merged, cell_mask_dim=None,
                             timelapse_objects="nucleus", cytoplasm=True,
                             crop_mode=["nucleus"], save_png=False)
    _i, _t, _c, _f, error = m._measure_crop_core(0, [], name, settings)
    assert not error
    tables = _rows(tmp_path / "measurements" / "measurements.db",
                   "SELECT name FROM sqlite_master WHERE type='table'")
    assert "nucleus" in set(tables["name"])
    assert "cytoplasm" not in set(tables["name"])


def test_a_cloud_source_is_handed_to_the_cloud_runner(monkeypatch):
    import spacr.ome_zarr as oz
    from spacr.settings import get_measure_crop_settings

    handed = []
    monkeypatch.setattr(oz, "_run_with_cloud_sources",
                        lambda fn, settings, kind: handed.append(
                            (fn, settings["src"], kind)) or "ran")
    settings = get_measure_crop_settings({})
    settings["src"] = "s3://bucket/plate1"
    assert m.measure_crop(settings) == "ran"
    assert handed == [(m.measure_crop, "s3://bucket/plate1", "measure")]


def test_the_end_of_run_steps_each_report_and_none_fails_the_run(
        tmp_path, capsys):
    from tests.test_scratch_wound_closure import _settings, _write_plate

    merged = _write_plate(tmp_path)
    m.measure_crop(_settings(
        merged, plot=False, wound_closure=False,
        bleach_correction="exponential", time_to_event=True,
        time_to_event_min_frames=1, profiling=True))
    printed = capsys.readouterr().out
    assert "Bleach correction" in printed
    assert "Time to event" in printed
    assert "Profiles" in printed
    assert "Successfully completed run" in printed
