"""Confluency on the fields and tables that sit at its edges: a blank
field, a field too flat to split, a source that is missing its input, a
database with no confluency in it, a time course, and a Measure field run
with its overlay figure drawn."""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import measure as m


def test_a_field_that_is_not_a_plane_or_a_stack_is_refused():
    with pytest.raises(ValueError, match="needs a 2-D field"):
        m._confluency_plane(np.zeros(5))


def test_flat_and_tiny_fields_have_no_separation():
    assert m._otsu_separation(np.array([])) == (0.0, 0.0)
    assert m._otsu_separation(np.array([3.0, 3.0, 3.0, 3.0])) == (3.0, 0.0)
    threshold, separation = m._otsu_separation(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0]))
    assert separation == 0.0 and 0.0 < threshold < 10.0
    assert not m._unit_scaled(np.full((4, 4), 7.0)).any()
    blank = m._texture_coverage(np.full((32, 32), 500, np.uint16))
    assert blank.confluency == 0.0 and blank.source == "texture"
    assert not blank.covered.any()


def test_a_source_without_its_input_is_refused():
    with pytest.raises(ValueError, match="unknown confluency source"):
        m._field_confluency(np.zeros((8, 8)), source="phase")
    with pytest.raises(ValueError, match="needs a cell mask"):
        m._field_confluency(np.zeros((8, 8)), source="masks")
    with pytest.raises(ValueError, match="the intensity confluency source "
                                         "needs an image"):
        m._field_confluency(None, source="intensity")


def test_the_confluency_channel_defaults_to_the_first_measured_one():
    assert m._confluency_channel({"channels": [2, 3]}) == 2
    assert m._confluency_channel({"confluency_channel": ""}) == 0
    assert m._confluency_channel({"confluency_channel": "1"}) == 1
    assert m._monolayer_ok(0.1, None) is True


def test_a_field_reads_its_channel_from_the_measured_planes_or_refuses():
    image = np.zeros((32, 32), np.uint16)
    image[:, :16] = 4000
    data = np.stack([np.zeros_like(image), image,
                     (image > 0).astype(np.uint16)], axis=-1)
    settings = {"confluency_source": "intensity", "confluency_channel": 1,
                "channels": [1], "cell_mask_dim": 2}
    result, plane = m._measure_field_confluency(
        data, settings, channel_arrays=data[..., [1]])
    assert result.confluency == pytest.approx(0.5, abs=0.05)
    assert np.array_equal(plane, image)
    masks, _plane = m._measure_field_confluency(
        data, dict(settings, confluency_source="masks"))
    assert masks.source == "masks" and masks.confluency == pytest.approx(0.5)
    with pytest.raises(ValueError, match="confluency_channel is 7"):
        m._measure_field_confluency(data, dict(settings, confluency_channel=7,
                                               channels=[0]))


def test_the_overlay_figure_names_the_field_and_its_coverage():
    import matplotlib.pyplot as plt

    image = np.zeros((16, 16), np.uint16)
    image[:8] = 1000
    result = m._field_confluency(image, source="intensity")
    figure = m._confluency_figure(image, result, "plate1_A01_1")
    try:
        assert figure.axes[0].get_title() == (
            f"plate1_A01_1: {result.confluency:.1%} covered (intensity)")
    finally:
        plt.close(figure)


def test_a_database_without_confluency_gives_empty_tables(tmp_path):
    missing = tmp_path / "absent.db"
    assert m._read_confluency(str(missing)).empty
    assert m._confluency_by_well(None).empty
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE cell (x INTEGER)")
    assert m._aggregate_confluency_by_well(str(db)).empty
    assert m._read_confluency_wells(str(db)).empty
    assert m._read_confluency_wells("").empty
    assert m._read_confluency_wells(str(missing)).empty

    plaques = pd.DataFrame({"plateID": ["p"], "rowID": ["r1"],
                            "columnID": ["c1"], "plaque_count": [4]})
    joined = m._monolayer_qc(plaques, str(db),
                             value_columns=("plaque_count",))
    assert joined["confluency"].isna().all()
    assert joined["monolayer_ok"].isna().all()
    with pytest.raises(ValueError, match="needs the well columns"):
        m._monolayer_qc(pd.DataFrame({"file": ["a.tif"]}), str(db))


def test_a_time_course_is_kept_per_time_point_through_the_database(tmp_path):
    fields = pd.DataFrame({
        "plateID": ["p"] * 4, "rowID": ["r1"] * 4, "columnID": ["c1"] * 4,
        "fieldID": ["f1", "f2", "f1", "f2"], "timeID": [1, 1, 2, 2],
        "confluency": [0.2, 0.4, 0.8, 1.0], "covered_px": [20, 40, 80, 100],
        "field_px": [100] * 4, "confluency_qc_threshold": [0.5] * 4})
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        fields.to_sql(m._CONFLUENCY_TABLE, conn, index=False)
    wells = m._aggregate_confluency_by_well(str(db))
    assert wells.set_index("timeID")["confluency"].round(6).to_dict() == {
        1: 0.3, 2: 0.9}
    read = m._read_confluency_wells(str(db))
    assert len(read) == 2 and "timeID" in read.columns

    counts = pd.DataFrame({"plateID": ["p", "p"], "rowID": ["r1", "r1"],
                           "columnID": ["c1", "c1"], "timeID": [1, 2],
                           "cells": [30, 90]})
    joined = m._monolayer_qc(counts, str(db), value_columns=("cells",))
    assert joined["cells_per_confluency"].round(6).tolist() == [100.0, 100.0]
    assert joined["monolayer_ok"].tolist() == [0, 1]


def test_a_measured_field_writes_its_confluency_and_draws_it(tmp_path,
                                                             capsys):
    from spacr.settings import get_measure_crop_settings

    merged = tmp_path / "merged"
    merged.mkdir()
    (tmp_path / "measurements").mkdir()
    rng = np.random.default_rng(0)
    cells = np.zeros((64, 64), np.uint16)
    cells[4:28, 4:28] = 1
    cells[36:60, 36:60] = 2
    signal = rng.integers(50, 100, (64, 64)).astype(np.uint16)
    signal[cells > 0] += 3000
    np.save(merged / "plate1_A01_1.npy",
            np.stack([signal, signal, cells], axis=-1))
    settings = get_measure_crop_settings({})
    settings.update({
        "src": str(merged), "channels": [0, 1], "cell_mask_dim": 2,
        "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "cell_min_size": 0, "nucleus_min_size": 0, "pathogen_min_size": 0,
        "cytoplasm_min_size": 0, "save_png": False, "save_arrays": False,
        "plot": True, "verbose": True, "n_jobs": 1, "confluency": True,
        "confluency_source": "masks", "confluency_qc_threshold": 0.1})
    _index, _time, _cells, figs, error = m._measure_crop_core(
        0, [], "plate1_A01_1.npy", settings)
    assert not error
    assert "plate1_A01_1: 28.1% covered (masks)" in capsys.readouterr().out
    figure = figs["plate1_A01_1__confluency"]
    assert figure.axes[0].get_title().startswith("plate1_A01_1: 28.1%")
    with sqlite3.connect(tmp_path / "measurements" / "measurements.db") as c:
        row = c.execute(f"SELECT confluency, monolayer_ok FROM "
                        f"{m._CONFLUENCY_TABLE}").fetchone()
    assert row == (pytest.approx(0.28125), 1)
