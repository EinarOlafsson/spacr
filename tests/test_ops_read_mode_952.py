"""Item 952: the read mode of the barcode reads (``ops_read_mode``: log, raw, auto).

``log`` is the read spaCR always took (the -LoG 3 x 3 maximum, the brightest base after
the per-cycle median normalisation, library calibration in the OPS decode). ``raw`` is
PERISCOPE's read: the unfiltered maximum over a disk of 3 px at the same peaks, the
brightest channel as read. ``auto`` chooses one of the two per well from decoy calls, as
the leading axis of the shared tuner :func:`spacr.ops_decode.tune_thresholds`, log last so
an exact tie keeps log. One implementation in :mod:`spacr.ops_sbs` serves the
spacer-prefix decode and the OPS decode.
"""
from __future__ import annotations

import inspect
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_decode as D
from spacr import ops_engine
from spacr import ops_prefix as P
from spacr import ops_sbs as S
from spacr.tabular import read_table
from tests.test_ops_spacer_prefix_decode_920 import (SPACERS, library_table, synthetic_site,
                                                     write_cpg_site)


def _disk(radius):
    """skimage's ``disk``, the footprint the round-3 harness used."""
    morphology = pytest.importorskip("skimage.morphology")
    return morphology.disk(radius).astype(bool)


def test_the_raw_read_is_the_disk_maximum_of_the_unfiltered_stack():
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 60000, (3, 4, 40, 50)).astype(np.uint16)
    peaks = np.array([[0, 0], [1, 49], [39, 2], [20, 25], [38, 47], [3, 3], [17, 0]])
    got = S._read_values(stack, None, peaks, "raw")
    footprint = _disk(3)
    want = np.empty_like(got)
    for c in range(3):
        for k in range(4):
            plane = ndimage.maximum_filter(stack[c, k].astype(np.float32), footprint=footprint)
            want[:, c, k] = plane[peaks[:, 0], peaks[:, 1]]
    assert got.dtype == np.float32 and got.shape == (7, 3, 4)
    assert np.array_equal(got, want)
    assert S._read_values(stack, None, np.zeros((0, 2), np.int64), "raw").shape == (0, 3, 4)


def test_the_log_read_is_the_filtered_3x3_maximum_and_other_modes_are_refused():
    rng = np.random.default_rng(1)
    filtered = rng.random((2, 4, 30, 30)).astype(np.float32)
    peaks = np.array([[5, 5], [10, 20], [0, 29]])
    assert np.array_equal(S._read_values(None, filtered, peaks, "log"),
                          S.extract_bases(filtered, peaks, window=1))
    with pytest.raises(ValueError, match="'log' or 'raw'"):
        S._read_values(filtered, filtered, peaks, "auto")


def test_raw_calls_take_the_brightest_channel_as_read():
    rng = np.random.default_rng(2)
    values = rng.random((400, 5, 4)).astype(np.float32)
    values[:, :, 3] *= 3.0
    data, codes, margin = S._raw_called_bases(values)
    plain = S.called_bases(values, compensate=False, normalise=False, gpu=False)
    assert np.array_equal(data, plain[0]) and codes == plain[1]
    assert np.array_equal(margin, plain[2])
    assert all(code == "".join(S.BASES[i] for i in row.argmax(axis=1))
               for code, row in zip(codes, values))
    assert codes != S.called_bases(values, gpu=False)[1]


def test_the_read_mode_setting_is_checked():
    assert S._read_mode_setting({}) == "log"
    assert S._read_mode_setting({"ops_read_mode": ""}) == "log"
    assert S._read_mode_setting({"ops_read_mode": " RAW "}) == "raw"
    with pytest.raises(ValueError, match="ops_read_mode must be one of"):
        S._read_mode_setting({"ops_read_mode": "brightest"})


def _two_mode_well(raw_gain, raw_leak, fields=24, seed=0):
    """Counts over (read mode raw/log, read threshold, 2) for one synthetic well."""
    rng = np.random.default_rng(seed)
    nuclei = rng.integers(900, 1100, fields)
    counts = np.zeros((fields, 2, 2, 2), np.int64)
    for a in range(2):
        real = rng.binomial(nuclei, 0.6 - 0.1 * a)
        counts[:, 1, a, 0] = real
        counts[:, 1, a, 1] = rng.poisson(real * 0.0005)
        raw = rng.binomial(nuclei, min(0.6 - 0.1 * a + raw_gain, 0.99))
        counts[:, 0, a, 0] = raw
        counts[:, 0, a, 1] = rng.poisson(raw * raw_leak)
    return counts, nuclei, np.arange(fields) + 1


AXES = [("read_mode", [1.0, 0.0]), ("read_threshold", [100.0, 200.0])]


def test_the_tuner_picks_raw_where_raw_wins_on_decoys_and_log_where_it_loses():
    counts, nuclei, sites = _two_mode_well(raw_gain=0.1, raw_leak=0.0005)
    raw = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=0.5)
    assert raw["auto_met"] and raw["chosen"]["read_mode"] == 1.0
    leaky = _two_mode_well(raw_gain=0.1, raw_leak=0.05)[0]
    log = D.tune_thresholds(leaky, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=0.5)
    assert log["auto_met"] and log["chosen"]["read_mode"] == 0.0
    assert {row["read_mode"] for row in log["grid"]} == {0.0, 1.0}
    assert set(log["tune_sites"]).isdisjoint(log["report_sites"])


def test_an_exact_tie_keeps_log():
    counts, nuclei, sites = _two_mode_well(raw_gain=0.0, raw_leak=0.0005)
    counts[:, 0] = counts[:, 1]
    tie = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=0.5)
    assert tie["chosen"]["read_mode"] == 0.0
    nothing = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=1e-9)
    assert not nothing["auto_met"] and nothing["chosen"]["read_mode"] == 0.0


def test_the_choice_reads_decoy_counts_only():
    """The tuner is given real and decoy call counts and nothing else: no reference call set."""
    params = set(inspect.signature(D.tune_thresholds).parameters)
    assert params == {"counts", "nuclei", "sites", "axes", "k", "scale", "objective", "fdr_max",
                      "f_beta", "unreachable", "ci", "tune_fraction", "seed"}
    counts, nuclei, sites = _two_mode_well(raw_gain=0.1, raw_leak=0.0005)
    first = D.tune_thresholds(counts, nuclei, sites, AXES, k=4, scale=1.0, fdr_max=0.5)
    shuffled = np.random.default_rng(5).permutation(len(sites))
    again = D.tune_thresholds(counts[shuffled], nuclei[shuffled], sites[shuffled], AXES, k=4,
                              scale=1.0, fdr_max=0.5)
    assert first["chosen"] == again["chosen"] and first["report"] == again["report"]


def _prefix(tmp_path, name, **extra):
    stack, labels = synthetic_site()
    root = tmp_path / "data"
    if not root.exists():
        for site in (1, 2, 3, 4):
            write_cpg_site(root, site, stack, labels)
        library_table(SPACERS).to_csv(tmp_path / "Barcodes.csv", index=False)
    settings = {"ops_layout": "cpg_cropped", "ops_library_mode": "spacer_prefix",
                "cpg_root": str(root / "images_corrected_cropped"),
                "cpg_phenotype_channels": ["CorrDNA"], "ops_library": str(tmp_path / "Barcodes.csv"),
                "ops_prefix_masks": str(root / "masks" / "{plate}_{well}_{site}_cells.tiff"),
                "ops_read_threshold": 10.0, "n_workers": 1,
                "dst_root": str(tmp_path / name), **extra}
    return P.run_prefix_decode(settings)


def _same_files(one, two):
    names = sorted(os.listdir(one))
    assert names == sorted(os.listdir(two))
    for name in names:
        with open(os.path.join(one, name), "rb") as a, open(os.path.join(two, name), "rb") as b:
            assert a.read() == b.read(), name


@pytest.mark.parametrize("extra", [{}, {"ops_prefix_decoy_k": 4}])
def test_log_writes_what_the_prefix_decode_wrote(tmp_path, extra):
    base = _prefix(tmp_path, "base", **extra)
    log = _prefix(tmp_path, "log", ops_read_mode="log", **extra)
    _same_files(base["folder"], log["folder"])
    saved = json.load(open(os.path.join(base["folder"], "quality_classes.json")))
    assert "ops_read_mode" not in saved["settings"]


def test_raw_and_auto_in_the_prefix_decode(tmp_path):
    raw = _prefix(tmp_path, "raw", ops_read_mode="raw", ops_prefix_decoy_k=4)
    saved = json.load(open(os.path.join(raw["folder"], "quality_classes.json")))
    assert saved["settings"]["ops_read_mode"] == "raw"
    assert raw["called_cells"] == 12 and raw["fdr"]["CP9_W1"]["assigned"] == 12
    reads = read_table(os.path.join(raw["folder"], "reads_CP9_W1.parquet"),
                       canonicalise=False, report=None)
    assert (reads["status"] == "exact").mean() > 0.5
    with pytest.raises(ValueError, match="per_well_decoy"):
        _prefix(tmp_path, "refused", ops_read_mode="auto", ops_prefix_decoy_k=4)
    grid = {"ops_read_threshold": [10, 5000], "ops_prefix_min_read_quality": [0, 0.5]}
    tuning = {"ops_prefix_decoy_k": 4, "ops_prefix_threshold_tuning": "per_well_decoy",
              "ops_prefix_tuning_grid": grid, "ops_prefix_fdr_target_pct": 50.0}
    auto = _prefix(tmp_path, "auto", ops_read_mode="auto", **tuning)
    choice = auto["tuning"]["CP9_W1"]
    assert choice["read_mode"] in ("raw", "log")
    assert choice["chosen"]["read_mode"] == (1.0 if choice["read_mode"] == "raw" else 0.0)
    stem = os.path.join(auto["folder"], "threshold_tuning_CP9_W1")
    table = pd.read_csv(stem + ".csv")
    assert len(table) == 2 * 2 * 2 * 1 and table["chosen"].sum() == 1
    assert set(table["read_mode"]) == {0.0, 1.0}
    assert json.load(open(stem + ".json"))["read_mode"] == choice["read_mode"]
    log = _prefix(tmp_path, "auto_log", ops_read_mode="log", **tuning)
    assert "read_mode" not in log["tuning"]["CP9_W1"]
    assert auto["called_cells"] == log["called_cells"] == 12
    again = _prefix(tmp_path, "auto_again", ops_read_mode="auto", **tuning)
    for name in ("reads_CP9_W1.parquet", "cells_CP9_W1.csv", "site_summary.csv",
                 "threshold_tuning_CP9_W1.csv"):
        pd.testing.assert_frame_equal(
            read_table(os.path.join(auto["folder"], name), canonicalise=False, report=None),
            read_table(os.path.join(again["folder"], name), canonicalise=False, report=None))


@pytest.fixture(scope="module")
def engine_well(tmp_path_factory):
    """The synthetic well of item 873's tests, decoded under each read mode."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_read_mode_952")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_preset": "rule_a_s2"}
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    log = ops_engine.run_ops({**settings, "ops_decode_threshold_mode": "manual"},
                             library=library)
    runs = {"log": log}
    runs["raw"] = ops_engine.run_ops({**settings, "ops_read_mode": "raw",
                                      "ops_decode_threshold_mode": "manual"},
                                     phases=("decode",), library=library)
    auto = {"ops_decode_threshold_mode": "auto", "ops_decode_fdr_max": 1000.0,
            "ops_decode_auto_spot_grid": []}
    runs["auto_log"] = ops_engine.run_ops({**settings, **auto}, phases=("decode",),
                                          library=library)
    runs["auto"] = ops_engine.run_ops({**settings, **auto, "ops_read_mode": "auto"},
                                      phases=("decode",), library=library)
    runs["plain_raw"] = ops_engine.run_ops(
        {**settings, "ops_decode_preset": "none", "ops_read_mode": "raw"},
        phases=("decode",), library=library)
    yield runs, settings, library
    monkeypatch.undo()


def test_the_ops_decode_reads_raw_on_request(engine_well):
    runs, _settings, _library = engine_well
    log = runs["log"]["wells"]["A1"]["decode"]
    raw = runs["raw"]["wells"]["A1"]["decode"]
    assert "read_mode" not in log["decode_rules"]
    assert raw["decode_rules"]["read_mode"] == "raw"
    assert raw["spots"] == log["spots"] and raw["objects_assigned"] > 0
    assert math.isfinite(raw["fdr"]["pct"])
    plain = runs["plain_raw"]["wells"]["A1"]["decode"]
    assert plain["read_mode"] == "raw" and plain["objects_assigned"] > 0


def test_auto_chooses_the_read_per_well_with_the_thresholds(engine_well):
    runs, _settings, _library = engine_well
    log = runs["auto_log"]["wells"]["A1"]["decode"]
    auto = runs["auto"]["wells"]["A1"]["decode"]
    assert "read_mode" not in log["thresholds"] and "read_mode" not in log["decode_rules"]
    mode = auto["thresholds"]["read_mode"]
    assert mode in ("raw", "log") and auto["decode_rules"]["read_mode"] == mode
    paths = auto["thresholds"]["paths"]
    curve = pd.read_csv(paths["curve_csv"])
    assert set(curve["read_mode"]) == {0.0, 1.0} and curve["chosen"].sum() == 1
    table = pd.read_csv(paths["plate_table"])
    assert list(table["read_mode"]) == [mode]
    with open(paths["thresholds"], encoding="utf-8") as handle:
        assert json.load(handle)["read_mode"] == mode
    if mode == "log":
        assert auto["objects_assigned"] == log["objects_assigned"]


def test_auto_needs_the_auto_thresholds(engine_well):
    _runs, settings, library = engine_well
    with pytest.raises(ValueError, match="ops_read_mode 'auto'"):
        ops_engine.run_ops({**settings, "ops_read_mode": "auto",
                            "ops_decode_threshold_mode": "manual"}, phases=("decode",),
                           library=library)
    with pytest.raises(ValueError, match="ops_read_mode must be one of"):
        ops_engine.run_ops({**settings, "ops_read_mode": "brightest"}, phases=("decode",),
                           library=library)


def test_alpha_registration():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS, OPS_TYPES
    from spacr.settings import ALPHA_FEATURES

    assert ALPHA_FEATURES[952]["settings"] == ("ops_read_mode",)
    assert OPS_DEFAULTS["ops_read_mode"] is None
    assert OPS_TYPES["ops_read_mode"] == (str, type(None))
    assert "ops_read_mode" in OPS_CATEGORIES["OPS decoding"]
    tip = OPS_TOOLTIPS["ops_read_mode"]
    assert tip.startswith("(str) - ") and "Default None. Alpha." in tip
