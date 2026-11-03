"""Feature QC at the end of Measure (item 756).

spacr.sp_stats.flag_degenerate_features flags per-object feature columns
that carry no signal. The case that motivated it is the first Zernike moment
in the BBBC021 reanalysis: numerically zero (~1e-15) apart from rare ~3e-3
spikes, which after per-plate scaling dominated cosine similarity and took
1-NN mechanism-of-action accuracy from 89.3% to chance. By default Measure
only reports the flagged columns; ``feature_qc_drop`` (alpha) removes them.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr.tabular import read_database, write_database


def _zernike_like(rng, n, spikes=0.005):
    """~1e-15 float noise with a few ~3e-3 spikes, as BBBC021 measured."""
    values = rng.normal(0, 1e-15, n)
    hit = rng.random(n) < spikes
    hit[:2] = True
    values[hit] = rng.normal(3e-3, 5e-4, int(hit.sum()))
    return values


def _well_behaved(rng, n):
    """Columns a real plate has that must never be flagged."""
    counts = rng.poisson(0.6, n).astype(float)           # mostly zero
    sparse_missing = rng.normal(50, 10, n)
    sparse_missing[rng.random(n) < 0.2] = np.nan          # some missing
    return pd.DataFrame({
        "cell_area": rng.normal(2800, 400, n),
        "cell_channel_0_mean_intensity": rng.lognormal(5, 0.5, n),
        "cell_eccentricity": rng.beta(2, 5, n),
        "cell_zernike_3": rng.normal(0.02, 0.01, n),
        "cell_neighbour_count": counts,
        "nucleus_channel_0_homogeneity_distance_8": sparse_missing,
        "cell_channel_1_negative_feature": rng.normal(-1e4, 2e3, n),
        "cell_tiny_but_real": rng.normal(1e-6, 2e-7, n),
    })


def test_the_zernike_spike_pattern_is_flagged_as_spike_dominated():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"nucleus_zernike_1": _zernike_like(rng, 4000),
                          "cell_zernike_1": _zernike_like(rng, 4000)})
    flagged = s.flag_degenerate_features(frame)
    assert flagged["feature"].tolist() == ["nucleus_zernike_1",
                                           "cell_zernike_1"]
    assert set(flagged["reason"]) == {"spike_dominated"}
    row = flagged.iloc[0]
    assert row["mad"] < 1e-12 and row["max_abs"] > 1e-3
    assert 0 < row["spike_fraction"] <= 0.05


def test_well_behaved_columns_are_not_flagged():
    rng = np.random.default_rng(2)
    frame = _well_behaved(rng, 4000)
    flagged = s.flag_degenerate_features(frame)
    assert flagged.empty, flagged.to_dict("records")
    assert list(flagged.columns) == list(s._FEATURE_QC_COLUMNS[1:])


def test_missing_zero_and_constant_columns_get_their_own_reason():
    rng = np.random.default_rng(3)
    n = 200
    half = rng.normal(1, 0.1, n)
    half[: n // 2 + 10] = np.nan
    frame = pd.DataFrame({
        "cell_pathogen_distance": np.full(n, np.inf),
        "cell_mostly_missing": half,
        "cell_zernike_1_no_spikes": rng.normal(0, 1e-15, n),
        "cell_zernike_0": np.full(n, 1 / np.pi, dtype=np.float32),
        "cell_area": rng.normal(100, 10, n),
    })
    flagged = s.flag_degenerate_features(frame).set_index("feature")
    assert flagged["reason"].to_dict() == {
        "cell_pathogen_distance": "all_missing",
        "cell_mostly_missing": "mostly_missing",
        "cell_zernike_1_no_spikes": "numerical_zero",
        "cell_zernike_0": "constant",
    }


def test_identity_and_position_columns_are_not_judged():
    n = 50
    frame = pd.DataFrame({"plateID": ["p1"] * n, "object_label": [1] * n,
                          "Metadata_n_nuclei": [1] * n,
                          "cell_channel_0_centroid_weighted-0": [5.0] * n,
                          "cell_area": np.arange(n, dtype=float)})
    assert s.flag_degenerate_features(frame).empty


def _plate_db(path, rng, n=600):
    cells = _well_behaved(rng, n)
    cells.insert(0, "plateID", "p1")
    cells.insert(1, "rowID", "r1")
    cells.insert(2, "columnID", [f"c{i % 12 + 1}" for i in range(n)])
    cells.insert(3, "object_label", np.arange(1, n + 1))
    cells["cell_zernike_1"] = _zernike_like(rng, n)
    cells["cell_zernike_0"] = 1 / np.pi
    nuclei = cells.loc[:, ["plateID", "rowID", "columnID", "object_label"]]
    nuclei = nuclei.assign(nucleus_area=rng.normal(700, 90, n),
                           nucleus_zernike_1=_zernike_like(rng, n),
                           nucleus_empty=np.nan)
    write_database(cells, path, "cell", if_exists="replace")
    write_database(nuclei, path, "nucleus", if_exists="replace")
    return path


def _digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_measure_reports_beside_the_database_and_leaves_it_untouched(
        tmp_path, capsys):
    from spacr.measure import _emit_feature_qc
    from spacr.settings import get_measure_crop_settings

    db = _plate_db(str(tmp_path / "measurements.db"),
                   np.random.default_rng(4))
    before = _digest(db)
    settings = get_measure_crop_settings({"src": str(tmp_path / "merged")})
    assert settings["feature_qc_drop"] is False
    _emit_feature_qc(settings, db)
    assert _digest(db) == before
    printed = capsys.readouterr().out
    assert "Feature QC: 4 degenerate feature column(s)" in printed
    assert "cell.cell_zernike_1 (spike_dominated)" in printed
    report = pd.read_csv(tmp_path / "feature_qc.csv")
    assert set(zip(report["table"], report["feature"], report["reason"])) == {
        ("cell", "cell_zernike_1", "spike_dominated"),
        ("cell", "cell_zernike_0", "constant"),
        ("nucleus", "nucleus_zernike_1", "spike_dominated"),
        ("nucleus", "nucleus_empty", "all_missing"),
    }
    summary = json.loads((tmp_path / "feature_qc.json").read_text(
        encoding="utf-8"))
    assert summary["flagged"] == 4 and summary["dropped"] == []
    assert summary["reasons"]["spike_dominated"] == 2
    assert summary["options"]["relative_tolerance"] == 1e-6


def test_feature_qc_drop_removes_only_the_flagged_columns(tmp_path, capsys):
    from spacr.measure import _emit_feature_qc

    db = _plate_db(str(tmp_path / "measurements.db"),
                   np.random.default_rng(5))
    (before,) = read_database(db, ["cell"], report=None)
    _emit_feature_qc({"feature_qc_drop": True}, db)
    assert "removed 4 column(s)" in capsys.readouterr().out
    (cells,) = read_database(db, ["cell"], report=None)
    (nuclei,) = read_database(db, ["nucleus"], report=None)
    assert "cell_zernike_1" not in cells and "cell_zernike_0" not in cells
    assert {"nucleus_zernike_1", "nucleus_empty"}.isdisjoint(nuclei.columns)
    kept = [c for c in before.columns
            if c not in {"cell_zernike_1", "cell_zernike_0"}]
    pd.testing.assert_frame_equal(cells.loc[:, kept], before.loc[:, kept])
    summary = json.loads((tmp_path / "feature_qc.json").read_text(
        encoding="utf-8"))
    assert sorted(summary["dropped"]) == [
        "cell.cell_zernike_0", "cell.cell_zernike_1",
        "nucleus.nucleus_empty", "nucleus.nucleus_zernike_1"]


def test_feature_qc_never_fails_a_finished_run(tmp_path, capsys):
    from spacr.measure import _emit_feature_qc

    broken = tmp_path / "measurements.db"
    broken.write_bytes(b"not a database")
    _emit_feature_qc({}, str(broken))
    assert "Feature QC could not run" in capsys.readouterr().out
    _emit_feature_qc({}, str(tmp_path / "absent.db"))


@pytest.mark.parametrize("key", ["feature_qc_drop"])
def test_the_drop_setting_is_alpha_gated(key):
    from spacr.settings import ALPHA_FEATURES

    assert key in ALPHA_FEATURES[756]["settings"]
