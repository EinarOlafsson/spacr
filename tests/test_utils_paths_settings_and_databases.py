"""Table-driven tests for the path, metadata/naming, cleanup, settings,
multiprocessing, organelle/database and small mask helpers of ``spacr.utils``.

Test-suite audit batch 5, group D: each test below replaces several older
coverage-chasing tests of one unit with one case table run through pytest's
``subtests`` fixture, so every former case is still reported on its own.

Everything is CPU-only and offline: CUDA, the Hugging Face listing, HTTP,
``nvcc``, psutil and ``os.close`` are faked where a case needs them, and all
I/O goes to ``tmp_path``.
"""
from __future__ import annotations

import builtins
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import types

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np
import pandas as pd
import psutil
import pytest

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from spacr import utils as U  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _read_table(db_path, table):
    con = sqlite3.connect(str(db_path))
    try:
        return pd.read_sql(f"SELECT * FROM {table}", con)
    finally:
        con.close()


def _npy(path, name):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, name), np.zeros((2, 2), np.uint16))


# ===========================================================================
# _extract_filename_metadata / _map_wells_png
# ===========================================================================

def test_extract_filename_metadata_groups_files_by_parsed_keys(tmp_path, subtests, capsys):
    """Each matching file lands under its (plate, well, field, channel, ...)
    key; cq1 numeric wells become A01-style ids, a missing plateID group falls
    back to the source folder name, and a regex without the wellID group is
    reported instead of raising."""
    full = re.compile(r"(?P<plateID>plate\d+)_(?P<wellID>[A-H]\d+)_"
                      r"(?P<fieldID>f\d+)_(?P<chanID>C\d+)\.tif")
    cq1 = re.compile(r"(?P<plateID>plate\d+)_(?P<wellID>\d+)_"
                     r"(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
    no_plate = re.compile(r"(?P<wellID>[A-P]\d{2})_(?P<fieldID>\d+)_(?P<chanID>\d+)\.tif")
    no_well = re.compile(r"(?P<plateID>.+)\.tif")
    cases = [
        # label, src folder, regex, files, metadata_type, expected keys, printed
        ("two_channels_one_unmatched", "p", full,
         ["plate1_A01_f1_C1.tif", "plate1_A01_f1_C2.tif", "nomatch.tif"],
         "cellvoyager",
         {("plate1", "A01", "f1", "C1", None, None): ["plate1_A01_f1_C1.tif"],
          ("plate1", "A01", "f1", "C2", None, None): ["plate1_A01_f1_C2.tif"]}, None),
        ("cq1_numeric_well", "q", cq1, ["plate1_1_1_1.tif"], "cq1",
         {("plate1", "A01", "1", "1", None, None): ["plate1_1_1_1.tif"]}, None),
        ("plate_from_src_basename", "plateXYZ", no_plate, ["A01_1_2.tif"], "cellvoyager",
         {("plateXYZ", "A01", "1", "2", None, None): ["A01_1_2.tif"]}, None),
        ("missing_well_group", "r", no_well, ["plate1_A01.tif"], "cellvoyager", {},
         "Could not extract information from filename"),
    ]
    for label, folder, rx, files, mtype, expected, printed in cases:
        with subtests.test(case=label):
            src = tmp_path / folder
            src.mkdir()
            capsys.readouterr()
            out = U._extract_filename_metadata(files, str(src), rx, metadata_type=mtype)
            assert {k: list(v) for k, v in out.items()} == {
                k: [str(src / f) for f in v] for k, v in expected.items()}
            if printed:
                assert printed in capsys.readouterr().out


def test_map_wells_png_parses_names_or_returns_an_error_tuple(subtests, capsys, monkeypatch):
    """Crop names become (plate, row, column, field[, time], prcfo, object);
    a numeric well is copied into row and column; a malformed name gives an
    all-'error' tuple of the right length and says which file failed."""
    # The bad-name report prints once per field per process; start clean so a
    # name another test already reported is reported here too.
    monkeypatch.setattr(U, "_REPORTED_BAD_NAMES", set())
    cases = [
        ("letter_well", "plate1_B03_2_10.png", False,
         ("plate1", "r2", "c3", "f2", "plate1_r2_c3_f2_o10", "o10"), None),
        ("timelapse", "plate1_B03_2_5_10.png", True,
         ("plate1", "r2", "c3", "f2", "t5", "plate1_r2_c3_f2_t5_o10", "o10"), None),
        ("numeric_well", "plate1_12_3_9.png", False,
         ("plate1", "12", "12", "f3", "plate1_12_12_f3_o9", "o9"), None),
        ("malformed", "garbage.png", False, ("error",) * 6,
         "Error processing filename: garbage.png"),
        ("malformed_timelapse", "plate1_A01_1.png", True, ("error",) * 7, None),
    ]
    for label, name, timelapse, expected, printed in cases:
        with subtests.test(case=label):
            capsys.readouterr()
            assert tuple(U._map_wells_png(name, timelapse=timelapse)) == expected
            if printed:
                assert printed in capsys.readouterr().out


# ===========================================================================
# _get_cellpose_batch_size
# ===========================================================================

def test_get_cellpose_batch_size_follows_the_vram_ladder(subtests, monkeypatch, capsys, caplog):
    """VRAM picks the Cellpose batch size (8/16/48/96) and prints it; no CUDA
    or a driver error falls back to 8 and says why."""
    import torch

    def fake_cuda(mp, vram_gb):
        mp.setattr(torch.cuda, "is_available", lambda: True)
        props = types.SimpleNamespace(total_memory=int(vram_gb * (1024 ** 3)), name="FakeGPU")
        mp.setattr(torch.cuda, "get_device_properties", lambda idx: props)

    def no_cuda(mp):
        mp.setattr(torch.cuda, "is_available", lambda: False)

    def broken_driver(mp):
        def boom(idx):
            raise RuntimeError("CUDA driver version is insufficient")
        mp.setattr(torch.cuda, "is_available", lambda: True)
        mp.setattr(torch.cuda, "get_device_properties", boom)

    cases = [(f"{v}GB", (lambda mp, v=v: fake_cuda(mp, v)), expected,
              ["cellpose batch size", f"{v:.2f} GB"], None)
             for v, expected in [(6.0, 8), (10.0, 16), (16.0, 48), (24.0, 96), (32.0, 96)]]
    cases += [
        ("no_cuda", no_cuda, 8, ["CUDA is not available"], None),
        ("driver_error", broken_driver, 8, [], "Could not inspect CUDA memory"),
    ]
    for label, install, expected, printed, logged in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            install(mp)
            capsys.readouterr()
            caplog.clear()
            assert U._get_cellpose_batch_size() == expected
            out = capsys.readouterr().out
            for fragment in printed:
                assert fragment in out
            if logged:
                assert logged in caplog.text


# ===========================================================================
# _merge_and_save_to_database
# ===========================================================================

def _morph_intensity_frames(n=3, with_cell_id=False):
    morph = pd.DataFrame({"label": list(range(1, n + 1)),
                          "area": np.arange(n, dtype=float) + 10.0})
    if with_cell_id:
        morph["cell_id"] = list(range(1, n + 1))
    intensity = pd.DataFrame({"label": list(range(1, n + 1)),
                              "mean_intensity": np.arange(n, dtype=float) + 100.0})
    return morph, intensity


def _measure_child_links():
    """Organelle rows from the real measure step, with parent-pathogen links."""
    from spacr import measure as M

    size = 32
    cell = np.zeros((size, size), dtype=np.int32)
    cell[4:28, 4:28] = 1
    nucleus = np.zeros((size, size), dtype=np.int32)
    nucleus[8:14, 8:14] = 1
    pathogen = np.zeros((size, size), dtype=np.int32)
    pathogen[18:22, 18:22] = 1
    organelle = np.zeros((size, size), dtype=np.int32)
    organelle[18:20, 18:20] = 1
    organelle[10:12, 20:22] = 2
    cytoplasm = cell.copy()
    cytoplasm[nucleus > 0] = 0
    settings = {
        "cell_mask_dim": 0, "nucleus_mask_dim": 1, "pathogen_mask_dim": 2,
        "organelle_mask_dim": 3, "cytoplasm": True,
        "radial_dist": False, "calculate_correlation": False,
        "homogeneity": False, "homogeneity_distances": [2, 4],
        "manders_thresholds": [15, 85], "distance_gaussian_sigma": 1,
    }
    channels = np.random.default_rng(9).random((32, 32, 3)).astype(np.float32)
    frames = M._intensity_measurements(cell, nucleus, pathogen, organelle, cytoplasm,
                                       channels, settings, periphery=False, outside=False)
    children = U._check_integrity(frames[3].copy()).set_index("object_label")
    assert children.loc[1, "pathogen_id"] == 1
    assert children.loc[1, "organelle_pathogen_overlap_fraction"] == 1
    assert np.isnan(children.loc[2, "pathogen_id"])
    morphology = pd.DataFrame({"label": [1, 2], "organelle_area": [4., 4.]})
    return morphology, frames[3]


def test_merge_and_save_to_database_writes_one_row_per_object(tmp_path, subtests):
    """Morphology and intensity rows are merged and written to the table of
    their object type, keyed and annotated from the field name (plate, row,
    column, field, time) with the parent-object links kept."""
    def cell_rows():
        return (*_morph_intensity_frames(n=3), "cell", "plate1_B03_2", False)

    def nucleus_rows():
        return (*_morph_intensity_frames(n=2, with_cell_id=True), "nucleus",
                "plate1_B03_2_7", True)

    def organelle_rows():
        return (*_measure_child_links(), "organelle", "plate1_r1_c1_f1", False)

    def check_cell(got, src):
        assert len(got) == 3
        assert list(got.columns[:8]) == ["object_label", "plateID", "rowID", "columnID",
                                         "fieldID", "prcf", "file_name", "path_name"]
        assert set(got["plateID"]) == {"plate1"}
        assert set(got["rowID"]) == {"r2"}
        assert set(got["columnID"]) == {"c3"}
        assert set(got["prcf"]) == {"plate1_r2_c3_f2"}
        assert set(got["path_name"]) == {os.path.join(str(src), "plate1_B03_2.npy")}
        assert "label_list_morphology" in got.columns
        assert "label_list_intensity" in got.columns
        assert got["object_label"].tolist() == [1, 2, 3]

    def check_nucleus(got, src):
        assert len(got) == 2
        assert list(got.columns[:2]) == ["object_label", "cell_id"]
        assert set(got["timeID"]) == {"t7"}
        assert set(got["prcf"]) == {"plate1_r2_c3_f2_t7"}

    def check_organelle(got, src):
        got = got.set_index("object_label")
        assert got.loc[1, "pathogen_id"] == 1
        assert pd.isna(got.loc[2, "pathogen_id"])

    cases = [("cell", cell_rows, check_cell),
             ("nucleus_timelapse", nucleus_rows, check_nucleus),
             ("organelle_child_links", organelle_rows, check_organelle)]
    for label, build, check in cases:
        with subtests.test(case=label):
            src = tmp_path / label
            (src / "measurements").mkdir(parents=True)
            morph, intensity, table, name, timelapse = build()
            U._merge_and_save_to_database(morph, intensity, table, str(src), name,
                                          "exp1", timelapse=timelapse)
            check(_read_table(src / "measurements" / "measurements.db", table), src)


def test_merge_and_save_to_database_refuses_bad_input_and_writes_nothing_empty(
        tmp_path, subtests, monkeypatch, capsys):
    """Empty morphology returns before any database is opened; an unknown
    table type or a merge that lost object_label raises; a database that
    cannot be opened is reported with the table name; and every frame that
    reaches the write has rows."""
    def run(src, morph=None, intensity=None, table="cell"):
        if morph is None:
            morph, intensity = _morph_intensity_frames()
        return U._merge_and_save_to_database(morph, intensity, table, str(src),
                                             "plate1_B03_2", "exp1")

    def empty_frames(src, mp):
        empty = pd.DataFrame({"label": pd.Series(dtype=int)})
        assert run(src, empty, empty.copy()) is None
        assert not (src / "measurements" / "measurements.db").exists()
        # the early return happens before any path is touched
        morph = pd.DataFrame({"object_label": [1, 2], "area": [10.0, 20.0]})
        assert U._merge_and_save_to_database(
            morph.iloc[0:0], pd.DataFrame(columns=["object_label", "mean"]),
            "cell", os.path.join(str(src), "nowhere"), "f0", "exp") is None

    def unknown_table(src, mp):
        with pytest.raises(ValueError, match="Invalid table_type: mitochondrion"):
            run(src, table="mitochondrion")

    def lossy_merge(src, mp):
        real_merge = pd.merge
        mp.setattr(pd, "merge", lambda left, right, *a, **k:
                   real_merge(left, right, *a, **k).drop(columns=["object_label"]))
        with pytest.raises(ValueError,
                           match=r"Columns missing in DataFrame: \['object_label'\]"):
            run(src)

    def unopenable_db(src, mp):
        import shutil
        shutil.rmtree(src / "measurements")
        capsys.readouterr()
        run(src)
        out = capsys.readouterr().out
        assert "SQLite error" in out and "cell" in out
        assert not (src / "measurements").exists()

    def write_always_has_rows(src, mp):
        written = []
        mp.setattr(U, "_check_integrity", lambda frame: frame)
        mp.setattr(U.schema, "validate_object_table_frame", lambda frame, *a, **k: frame)
        mp.setattr(U, "_assert_measurement_units_compatible", lambda *a, **k: None)
        mp.setattr(U, "_release_imported_rows_for_field", lambda *a, **k: 0)
        mp.setattr(U, "_append_to_measurements_db",
                   lambda _path, _table, frame, **k: written.append(frame.copy()))
        morph = pd.DataFrame({"object_label": [1, 2], "area": [10.0, 20.0]})
        U._merge_and_save_to_database(
            morph, pd.DataFrame({"object_label": [1], "mean": [5.0]}),
            "cell", "unused", "plate1_A1_1", "exp")
        assert len(written) == 1 and len(written[0]) == len(morph)
        import inspect
        assert "if len(merged_df) > 0:" not in inspect.getsource(
            U._merge_and_save_to_database)

    cases = [empty_frames, unknown_table, lossy_merge, unopenable_db, write_always_has_rows]
    for case in cases:
        with subtests.test(case=case.__name__), monkeypatch.context() as mp:
            src = tmp_path / case.__name__
            (src / "measurements").mkdir(parents=True)
            case(src, mp)


# ===========================================================================
# _organelle_diagnostic
# ===========================================================================

def _blob_img(size=48):
    """Two bright Gaussian spots on a black background, float32."""
    img = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    for cy, cx in ((14, 14), (32, 30)):
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 18.0) * 900
    return img


def _blob_count(title, prefix):
    assert title.startswith(prefix) and title.endswith("blobs)"), title
    return int(title.split("(")[1].split(" ")[0])


def test_organelle_diagnostic_spot_and_ring_filters(subtests, monkeypatch):
    """Spot morphologies paint each DoG/LoG detection to 1.0 (forwarding the
    sigma/threshold settings) or white-top-hat the image; ring morphology is
    the absolute DoG with default or configured sigmas."""
    from skimage.morphology import disk, white_tophat

    img = _blob_img()

    def painted_two_blobs(diag, title, prefix):
        assert diag.shape == img.shape
        assert diag.dtype == np.float64
        assert _blob_count(title, prefix) == 2
        assert diag[14, 14] == 1.0 and diag[32, 30] == 1.0
        assert diag[0, 0] < 1.0
        assert np.count_nonzero(diag == 1.0) > 2
        assert diag.max() == 1.0

    default_ring, _ = U._organelle_diagnostic(img, "ring", "otsu", {})

    def ring_default(diag, title):
        assert title == "DoG ring enhancement (σ=1.0/3.0)"
        assert diag.shape == img.shape
        assert np.all(diag >= 0) and diag.max() > 0
        assert diag[14, 14] > diag[0, 0]

    def ring_custom(diag, title):
        assert title == "DoG ring enhancement (σ=0.5/6.0)"
        assert not np.allclose(diag, default_ring), "custom sigmas were ignored"
        assert np.all(diag >= 0)

    def tophat(diag, title):
        assert title == "Top-hat filtered (r=8)"
        assert np.allclose(diag, white_tophat(img, disk(8)))
        assert np.all(diag >= 0) and np.all(diag <= img + 1e-3)

    def dog_nothing(diag, title):
        assert title == "DoG detections (0 blobs)"
        assert np.all(diag == 0.0)

    cases = [
        ("dog_paints", img, "spots", "dog",
         {"organelle_dog_sigma_low": 1.0, "organelle_dog_sigma_high": 3.0,
          "organelle_log_threshold": 0.01},
         lambda d, t: painted_two_blobs(d, t, "DoG detections (")),
        ("dog_empty_image", np.zeros((32, 32), np.float32), "spots", "dog",
         {"organelle_dog_sigma_low": 1.0, "organelle_dog_sigma_high": 3.0,
          "organelle_log_threshold": 0.5}, dog_nothing),
        ("log_paints", img, "spots", "log",
         {"organelle_log_min_sigma": 1, "organelle_log_max_sigma": 5,
          "organelle_log_num_sigma": 5, "organelle_log_threshold": 0.05},
         lambda d, t: painted_two_blobs(d, t, "LoG detections (")),
        ("otsu_tophat", img, "spots", "otsu", {"organelle_tophat_radius": 8}, tophat),
        ("ring_default", img, "ring", "otsu", {}, ring_default),
        ("ring_custom", img, "ring", "otsu",
         {"organelle_ring_sigma_inner": 0.5, "organelle_ring_sigma_outer": 6.0}, ring_custom),
    ]
    for label, image, morph, method, settings, check in cases:
        with subtests.test(case=label):
            check(*U._organelle_diagnostic(image, morph, method, settings))

    with subtests.test(case="dog_forwards_settings"), monkeypatch.context() as mp:
        captured = {}
        real = U.blob_dog

        def spy(image, **kwargs):
            captured.update(kwargs)
            return real(image, **kwargs)

        mp.setattr(U, "blob_dog", spy)
        U._organelle_diagnostic(img, "spots", "dog", {
            "organelle_dog_sigma_low": 2.0, "organelle_dog_sigma_high": 6.0,
            "organelle_log_threshold": 0.02})
        assert (captured["min_sigma"], captured["max_sigma"], captured["threshold"]) == \
            (2.0, 6.0, 0.02)


def test_organelle_diagnostic_network_irregular_and_fallback(subtests):
    """Network morphology runs hysteresis (thresholds < 1 are percentiles,
    >= 1 raw intensities), a named ridge filter, or a sigma-1 smooth;
    irregular smooths with sigma max(radius/2, 1); anything else is the
    1st/99th-percentile rescale, skipped for a flat image."""
    from skimage.filters import gaussian

    img = _blob_img()
    smooth = gaussian(img, sigma=1)

    def hysteresis(low_abs, high_abs, title_text):
        def check(diag, title):
            assert title == title_text
            assert diag.dtype == np.float64
            assert set(np.unique(diag)).issubset({0.0, 1.0})
            selected = diag > 0
            assert 0 < selected.sum() < diag.size
            assert np.all(smooth[selected] >= low_abs)
            assert np.all(diag[smooth >= high_abs] == 1.0)
        return check

    def no_seeds(diag, title):
        assert title == "Hysteresis (low=10.0, high=1000000000.0)"
        assert diag.sum() == 0.0

    def ridge(name):
        def check(diag, title):
            assert title == f"{name} ridge (sigmas=[1, 2])"
            assert diag.shape == img.shape
            assert np.all(np.isfinite(diag))
            assert diag.max() > 0, "ridge filter produced an all-zero response"
        return check

    def smoothed(sigma, title_text):
        def check(diag, title):
            assert title == title_text
            assert np.allclose(diag, gaussian(img, sigma=sigma))
        return check

    def network_smooth(diag, title):
        smoothed(1, "Gaussian smoothed (σ=1)")(diag, title)
        assert diag.max() < img.max()

    def normalised(diag, title):
        assert title == "Normalised image"
        p1, p99 = np.percentile(img.astype(np.float64), (1, 99))
        assert np.allclose(diag, np.clip((img.astype(np.float64) - p1) / (p99 - p1), 0, 1))
        assert diag.min() == pytest.approx(0.0) and diag.max() == pytest.approx(1.0)
        assert diag[14, 14] == pytest.approx(1.0)
        assert diag[0, 0] < 1e-6

    def flat(diag, title):
        assert title == "Normalised image"
        assert np.allclose(diag, 7.5)
        assert diag.dtype == np.float64

    cases = [
        ("hysteresis_percentiles", img, "network", "hysteresis",
         {"organelle_hysteresis_low": 0.9, "organelle_hysteresis_high": 0.99},
         hysteresis(np.percentile(smooth, 90), np.percentile(smooth, 99),
                    "Hysteresis (low=0.9, high=0.99)")),
        ("hysteresis_absolute", img, "network", "hysteresis",
         {"organelle_hysteresis_low": 50.0, "organelle_hysteresis_high": 300.0},
         hysteresis(50.0, 300.0, "Hysteresis (low=50.0, high=300.0)")),
        ("hysteresis_no_seeds", img, "network", "hysteresis",
         {"organelle_hysteresis_low": 10.0, "organelle_hysteresis_high": 1e9},
         no_seeds),
        *[(f"ridge_{name}", img, "network", "ridge",
           {"organelle_ridge_sigmas": [1, 2], "organelle_ridge_filter": name}, ridge(name))
          for name in ("frangi", "sato", "meijering")],
        ("network_other_method", img, "network", "otsu", {}, network_smooth),
        *[(f"irregular_r{radius}", img, "irregular", "otsu",
           {"organelle_morph_radius": radius},
           smoothed(sigma, f"Gaussian smoothed (σ={sigma:.1f})"))
          for radius, sigma in [(6, 3.0), (1, 1.0), (2, 1.0)]],
        ("unknown_morphology", img, "blobby", "otsu", {}, normalised),
        ("flat_image", np.full((16, 16), 7.5, np.float32), "unknown-morphology", "otsu",
         {}, flat),
    ]
    for label, image, morph, method, settings, check in cases:
        with subtests.test(case=label):
            check(*U._organelle_diagnostic(image, morph, method, settings))


# ===========================================================================
# _outline_and_overlay
# ===========================================================================

def _stack_with_masks(n_masks=1, h=48, w=48):
    """(H, W, 3 + n_masks) float image: 3 intensity channels + label masks."""
    img = np.zeros((h, w, 3 + n_masks), dtype=np.float32)
    img[..., 0] = 10.0
    img[..., 1] = 20.0
    img[..., 2] = 30.0
    for m in range(n_masks):
        mask = np.zeros((h, w), dtype=np.float32)
        mask[6 + m: 18 + m, 6: 18] = 1
        mask[28: 40, 28: 40] = 2
        img[..., 3 + m] = mask
    return img


def test_outline_and_overlay_paints_every_mask_outline_on_the_calling_thread(
        subtests, monkeypatch):
    """One uint8 outline per mask dimension, outline pixels repainted with
    the (cycled) outline colour and nothing else changed, an empty mask is
    the identity, and the contours are drawn on the calling thread (a worker
    thread aborts the process once Qt and Tk are both loaded)."""
    seen = set()
    real = U.find_contours

    def recording_find_contours(*args, **kwargs):
        seen.add(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(U, "find_contours", recording_find_contours)
    cases = [
        # label, n_masks, empty, mask_dims, colors, thickness
        ("one_mask", 1, False, [3], [[255, 0, 0]], 2),
        ("colour_cycles", 2, False, [3, 4], [[7, 8, 9]], 1),
        ("three_masks", 3, False, [3, 4, 5], [[255, 0, 0]], 1),
        ("empty_mask", 1, True, [3], [[1, 2, 3]], 3),
    ]
    for label, n_masks, empty, dims, colors, thickness in cases:
        with subtests.test(case=label):
            seen.clear()
            image = _stack_with_masks(n_masks=n_masks)
            if empty:
                image[..., 3] = 0
            rgb = U._gen_rgb_image(image, channels=[0, 1, 2])
            overlayed, outlines, returned = U._outline_and_overlay(
                image, rgb, mask_dims=dims, outline_colors=colors,
                outline_thickness=thickness)
            assert len(outlines) == len(dims)
            assert overlayed is not rgb
            assert returned is image
            union = np.zeros(image.shape[:2], bool)
            for i, outline in enumerate(outlines):
                assert outline.dtype == np.uint8
                assert outline.shape == image.shape[:2]
                painted = outline != 0
                union |= painted
                assert np.all(overlayed[painted] ==
                              np.array(colors[i % len(colors)], dtype=np.float32))
            assert np.array_equal(overlayed[~union], rgb[~union])
            if empty:
                assert outlines[0].max() == 0
                assert np.array_equal(overlayed, rgb)
            else:
                assert all(o.max() == 255 for o in outlines)
                assert seen == {threading.get_ident()}, (
                    "the contour drawing ran on a worker thread")
            if n_masks == 2:
                assert not np.array_equal(outlines[0], outlines[1])


# ===========================================================================
# _save_settings_json / save_settings / load_settings / pretty_print_settings
# ===========================================================================

def test_save_settings_json_writes_plain_json_or_notes_the_failure(tmp_path, subtests,
                                                                   monkeypatch, capsys):
    """numpy scalars are written as JSON numbers, a value numpy cannot be
    asked about falls back to its repr, and an unwritable path is a printed
    note, not an exception."""
    def numpy_scalars(path, mp):
        U._save_settings_json({"thr": np.float64(0.5), "n": np.int64(7)}, path)
        return {"thr": 0.5, "n": 7}

    def numpy_unimportable(path, mp):
        real_import = builtins.__import__

        def no_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy is gone")
            return real_import(name, *args, **kwargs)

        mp.setattr(builtins, "__import__", no_numpy)
        try:
            U._save_settings_json({"n": np.int64(7), "src": "here"}, path)
        finally:
            mp.undo()
        return {"n": repr(np.int64(7)), "src": "here"}

    def unwritable(path, mp):
        path = os.path.join(os.path.dirname(path), "no_such_folder", "settings.json")
        capsys.readouterr()
        U._save_settings_json({"src": "here"}, path)
        assert "could not write" in capsys.readouterr().out
        assert not os.path.exists(path)
        return None

    for case in (numpy_scalars, numpy_unimportable, unwritable):
        with subtests.test(case=case.__name__), monkeypatch.context() as mp:
            folder = tmp_path / case.__name__
            folder.mkdir()
            path = str(folder / "settings.json")
            expected = case(path, mp)
            if expected is not None:
                with open(path, encoding="utf-8") as fh:
                    assert json.load(fh) == expected


def test_save_settings_writes_a_headless_csv_copy_or_warns(tmp_path, subtests, monkeypatch,
                                                           capsys):
    """The CSV goes to <src>/settings/<name>.csv (<src[0]>/settings/<name>_list.csv
    for a list src) with test_mode and plot forced False and the caller's
    dict untouched; an unwritable src is a warning and the run continues."""
    def str_src(root, mp):
        src = root / "plate01"
        src.mkdir()
        settings = {"src": str(src), "test_mode": True, "plot": True,
                    "channels": [0, 1], "diameter": 30}
        assert U.save_settings(settings, name="gen_mask_settings") is None
        out = src / "settings" / "gen_mask_settings.csv"
        reloaded = U.load_settings(str(out), setting_key="Key", setting_value="Value")
        assert reloaded["test_mode"] is False and reloaded["plot"] is False
        assert reloaded["channels"] == [0, 1] and reloaded["diameter"] == 30
        assert settings["test_mode"] is True and settings["plot"] is True

    def list_src_show(root, mp):
        a, b = root / "a", root / "b"
        a.mkdir()
        b.mkdir()
        capsys.readouterr()
        U.save_settings({"src": [str(a), str(b)], "test_mode": True}, name="exp", show=True)
        assert (a / "settings" / "exp_list.csv").is_file()
        assert not (b / "settings").exists()
        printed = capsys.readouterr().out
        assert "Exp List" in printed and "Saving settings to" in printed

    def list_src_second_missing(root, mp):
        a = root / "a"
        a.mkdir()
        U.save_settings({"src": [str(a), str(root / "other")], "diameter": 30}, name="exp")
        assert (a / "settings" / "exp_list.csv").exists()
        assert not (root / "other").exists()

    def permission_denied(root, mp):
        def boom(*a, **kw):
            raise PermissionError(13, "Permission denied")
        mp.setattr(U.os, "makedirs", boom)
        capsys.readouterr()
        U.save_settings({"src": str(root), "test_mode": False}, name="settings")
        assert "Warning: could not save settings" in capsys.readouterr().out
        assert not (root / "settings").exists()

    def src_is_a_file(root, mp):
        bogus = root / "not_a_dir.txt"
        bogus.write_text("i am a file\n", encoding="utf-8")
        capsys.readouterr()
        assert U.save_settings({"src": str(bogus)}, name="settings") is None
        printed = capsys.readouterr().out
        assert "Warning: could not save settings to" in printed
        assert "Continuing without writing the settings copy." in printed
        assert bogus.read_text(encoding="utf-8") == "i am a file\n"

    for case in (str_src, list_src_show, list_src_second_missing, permission_denied,
                 src_is_a_file):
        with subtests.test(case=case.__name__), monkeypatch.context() as mp:
            root = tmp_path / case.__name__
            root.mkdir()
            case(root, mp)


def test_load_settings_parses_every_value_spelling(tmp_path, subtests):
    """Each settings-CSV value comes back as its Python type: booleans in any
    case and padding (matching the CLI loader), ints, floats, lists, tuples,
    dicts parsed recursively, blanks as None, broken literals verbatim,
    all-numeric columns as numbers; Key/Value and setting_key/setting_value
    headers both load."""
    from spacr import cli

    cases = [
        ("literal_types", "setting_key", "setting_value",
         [("flag_t", "True"), ("flag_f", "False"), ("an_int", "42"), ("a_float", "3.5"),
          ("a_list", "[1, 2, 3]"), ("a_tuple", "(4, 5)"), ("a_dict", "{'k': 'True', 'n': '7'}"),
          ("a_str", "cell"), ("blank", ""), ("broken_literal", "[1, 2")],
         {"flag_t": True, "flag_f": False, "an_int": 42, "a_float": 3.5, "a_list": [1, 2, 3],
          "a_tuple": (4, 5), "a_dict": {"k": True, "n": 7}, "a_str": "cell", "blank": None,
          "broken_literal": "[1, 2"}),
        ("more_literals", "setting_key", "setting_value",
         [("lst", "[1, 2]"), ("tup", "(3, 4)"), ("dct", "{'a': '1'}"), ("n_float", "3.14"),
          ("text", "hello"), ("bad_lit", "[unclosed")],
         {"lst": [1, 2], "tup": (3, 4), "dct": {"a": 1}, "n_float": 3.14, "text": "hello",
          "bad_lit": "[unclosed"}),
        ("all_numeric_column", "setting_key", "setting_value",
         [("nucleus_min_size", 5), ("cell_min_size", 100)],
         {"nucleus_min_size": 5, "cell_min_size": 100}),
        ("key_value_header", "Key", "Value", [("a", "cell")], {"a": "cell"}),
        ("spreadsheet_booleans", "Key", "Value",
         [("src", "data"), ("preprocess", "TRUE"), ("masks", "true"), ("unmix", "FALSE"),
          ("plot", "false"), ("save", " TrUe "), ("normalize", " FaLsE "),
          ("nested", "{'enabled': 'TRUE', 'off': 'FALSE'}"), ("experiment", "TRUE-control"),
          ("custom_model", " FALSE checkpoint "), ("one", "1"), ("zero", "0"),
          ("channels", "[0, 1, 2]")],
         {"src": "data", "preprocess": True, "masks": True, "unmix": False, "plot": False,
          "save": True, "normalize": False, "nested": {"enabled": True, "off": False},
          "experiment": "TRUE-control", "custom_model": " FALSE checkpoint ", "one": 1,
          "zero": 0, "channels": [0, 1, 2]}),
    ]
    for label, key_col, value_col, rows, expected in cases:
        with subtests.test(case=label):
            path = tmp_path / f"{label}.csv"
            pd.DataFrame(rows, columns=[key_col, value_col]).to_csv(path, index=False)
            out = U.load_settings(str(path))
            assert out == expected
            for key, value in expected.items():
                assert type(out[key]) is type(value), key
            assert U.load_settings(str(path), setting_key=key_col,
                                   setting_value=value_col) == expected
            if key_col == "Key":
                assert cli.load_settings_file(str(path)) == expected


def test_load_settings_refuses_what_is_not_a_settings_file_and_can_show_it(
        tmp_path, subtests, monkeypatch):
    """A table with neither header spelling raises naming setting_key; an
    Excel error cell in model_metadata is refused by the model-zoo check;
    show=True routes the table through display exactly once."""
    with subtests.test(case="neither_header"):
        for i, frame in enumerate([pd.DataFrame({"name": ["a"], "amount": ["cell"]}),
                                   pd.DataFrame({"a": [1], "b": [2]})]):
            path = tmp_path / f"u{i}.csv"
            frame.to_csv(path, index=False)
            with pytest.raises(ValueError, match="setting_key"):
                U.load_settings(str(path))

    with subtests.test(case="excel_error_cell"):
        openpyxl = pytest.importorskip("openpyxl")
        from spacr import tabular
        from spacr.model_zoo import ModelZooError

        path = tmp_path / "invalid-model-metadata.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.append(["Key", "Value"])
        sheet.append(["src", "data/plate"])
        sheet.append(["model_metadata", "#DIV/0!"])
        assert sheet["B3"].data_type == "e"
        workbook.save(path)
        workbook.close()
        raw = tabular.read_table(path, report=None, dtype=str, keep_default_na=False)
        value = raw.loc[raw["Key"] == "model_metadata", "Value"].iloc[0]
        assert not isinstance(value, str) and pd.isna(value)
        with pytest.raises(ModelZooError, match=r"model_metadata \(schema_version, settings\)"):
            U.load_settings(path)

    with subtests.test(case="show_true"), monkeypatch.context() as mp:
        seen = []
        mp.setattr(U, "display", lambda df: seen.append(df))
        path = tmp_path / "s.csv"
        pd.DataFrame({"setting_key": ["a", "b"], "setting_value": ["1", "cell"]}).to_csv(
            path, index=False)
        assert U.load_settings(str(path), show=True) == {"a": 1, "b": "cell"}
        assert len(seen) == 1 and isinstance(seen[0], pd.DataFrame)
        assert list(seen[0]["setting_key"]) == ["a", "b"]


def test_pretty_print_settings_draws_a_categorised_box(subtests, monkeypatch, capsys):
    """Settings print in a box titled by the caller, grouped under their
    category headers with unknown keys under 'Other', each key once, long
    values clipped to 41 characters plus an ellipsis; with no category table
    nothing gets a header."""
    import spacr.settings as S

    cat_name, cat_keys = next((c, ks) for c, ks in S.categories.items() if ks)
    known = cat_keys[0]

    def grouped(out):
        assert f"▸ {cat_name}" in out and "▸ Other" in out
        assert "zz_unknown_key" in out
        assert out.count("value-x") == 1

    def clipped(out):
        assert "▸" not in out
        assert "src" in out and "…" in out
        assert "z" * 41 in out and "z" * 42 not in out

    def empty(out):
        assert out.startswith("┌")
        assert "─" * 56 in out

    cases = [
        ("grouped", {known: "value-x", "zz_unknown_key": 7}, "My Run", False, grouped),
        ("no_categories_clips", {"src": "z" * 200}, "Fallback", True, clipped),
        ("empty", {}, "Nothing", False, empty),
    ]
    for label, settings, title, drop_categories, check in cases:
        with subtests.test(case=label), monkeypatch.context() as mp:
            if drop_categories:
                mp.delattr(S, "categories")
            capsys.readouterr()
            U.pretty_print_settings(settings, title=title)
            out = capsys.readouterr().out
            assert title in out
            check(out)


# ===========================================================================
# _update_database_with_merged_info / filepaths_to_database
# ===========================================================================

def _png_list_db(root):
    """A DB with a png_list table keyed on prcfo."""
    db = root / "m.db"
    con = sqlite3.connect(db)
    try:
        pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
                      "png_path": ["a/1.png", "a/2.png"]}).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db


def _info_df(with_object_label=True, with_cell_id=False):
    df = pd.DataFrame({
        "plateID": ["plate1", "plate1"], "rowID": ["r1", "r1"],
        "columnID": ["c1", "c1"], "fieldID": ["f1", "f1"],
        "pathogen": ["rh", "rh"], "treatment": ["cm", "cm"],
        "host_cells": ["HeLa", "HeLa"], "condition": ["HeLa_rh_cm", "HeLa_rh_cm"],
    })
    if with_object_label:
        df["object_label"] = [1, 2]
    if with_cell_id:
        df["cell_id"] = [1, 2]
    return df


def test_update_database_with_merged_info_merges_annotations_on_prcfo(tmp_path, subtests,
                                                                      monkeypatch, capsys):
    """Annotation columns are merged into png_list on prcfo, built from
    object_label (or, failing that, cell_id) unless the frame already has it;
    a missing table or a failed write is reported and changes nothing."""
    def from_object_label(db, mp):
        U._update_database_with_merged_info(str(db), _info_df())
        out = capsys.readouterr().out
        assert "generating prcfo columns" in out and "successfully updated" in out
        assert "cell_id" not in out  # the fallback must not run after a good build
        got = _read_table(db, "png_list")
        assert list(got["condition"]) == ["HeLa_rh_cm", "HeLa_rh_cm"]
        assert {"pathogen", "treatment", "host_cells", "condition"} <= set(got.columns)
        assert len(got) == 2

    def from_cell_id(db, mp):
        U._update_database_with_merged_info(
            str(db), _info_df(with_object_label=False, with_cell_id=True))
        assert "Merging on cell failed, trying with cell_id" in capsys.readouterr().out
        assert list(_read_table(db, "png_list")["host_cells"]) == ["HeLa", "HeLa"]

    def neither_id(db, mp):
        df = pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
                           "fieldID": ["f1"], "prcfo_missing": [1]})
        with pytest.raises(KeyError):
            U._update_database_with_merged_info(str(db), df, table="png_list",
                                                columns=["prcfo"])
        out = capsys.readouterr().out
        assert "Merging on cell failed, trying with cell_id" in out
        assert "cell_id" in out.split("trying with cell_id", 1)[1]

    def existing_prcfo_custom_columns(db, mp):
        df = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1", "plate1_r1_c1_f1_o2"],
                           "condition": ["a", "b"]})
        U._update_database_with_merged_info(str(db), df, columns=["condition", "prcfo"])
        assert "generating prcfo columns" not in capsys.readouterr().out
        got = _read_table(db, "png_list")
        assert list(got["condition"]) == ["a", "b"]
        assert "host_cells" not in got.columns

    def missing_table(db, mp):
        os.remove(db)
        sqlite3.connect(db).close()
        U._update_database_with_merged_info(str(db), _info_df(), table="png_list")
        assert "Failed to read table png_list" in capsys.readouterr().out
        con = sqlite3.connect(db)
        try:
            assert con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        finally:
            con.close()

    def write_fails(db, mp):
        def boom(self, *a, **kw):
            raise sqlite3.OperationalError("disk I/O error")
        mp.setattr(pd.DataFrame, "to_sql", boom)
        U._update_database_with_merged_info(str(db), _info_df())
        out = capsys.readouterr().out
        assert "Failed to update table png_list in the database" in out
        assert "disk I/O error" in out

    for case in (from_object_label, from_cell_id, neither_id, existing_prcfo_custom_columns,
                 missing_table, write_fails):
        with subtests.test(case=case.__name__), monkeypatch.context() as mp:
            root = tmp_path / case.__name__
            root.mkdir()
            db = _png_list_db(root)
            capsys.readouterr()
            case(db, mp)


def _crop_png_paths(directory, n=3, crop_mode="cell"):
    """Crop names as _generate_names emits them (child crops carry the parent id)."""
    if crop_mode in ("nucleus", "pathogen"):
        names = [f"plate1_A01_1_9_{i + 1}.png" for i in range(n)]
    else:
        names = [f"plate1_A01_1_{i + 1}.png" for i in range(n)]
    return [os.path.join(str(directory), nm) for nm in names]


def test_filepaths_to_database_registers_crops_in_png_list(tmp_path, subtests, capsys):
    """Every crop path becomes a png_list row with plate/row/column/field,
    prcfo and only its crop mode's <object>_id column (plus timeID for
    timelapse); an unregistered crop mode is refused; a database that cannot
    be opened is reported."""
    id_cols = {"cell_id", "nucleus_id", "pathogen_id", "cytoplasm_id"}
    for crop_mode in ("cell", "nucleus", "pathogen", "cytoplasm"):
        with subtests.test(case=crop_mode):
            src = tmp_path / crop_mode
            (src / "measurements").mkdir(parents=True)
            U.filepaths_to_database(_crop_png_paths(src, 3, crop_mode), {"timelapse": False},
                                    str(src), crop_mode)
            df = _read_table(src / "measurements" / "measurements.db", "png_list")
            id_col = f"{crop_mode}_id"
            assert len(df) == 3
            assert sorted(df[id_col]) == ["o1", "o2", "o3"]
            assert (set(df["plateID"]), set(df["rowID"]), set(df["columnID"]),
                    set(df["fieldID"])) == ({"plate1"}, {"r1"}, {"c1"}, {"f1"})
            assert sorted(df["prcfo"]) == [f"plate1_r1_c1_f1_o{i}" for i in (1, 2, 3)]
            assert not ((id_cols - {id_col}) & set(df.columns))
            assert "timeID" not in df.columns and "time_id" not in df.columns

    with subtests.test(case="timelapse"):
        src = tmp_path / "timelapse"
        (src / "measurements").mkdir(parents=True)
        paths = [os.path.join(str(src), f"plate1_A01_1_{t}_5.png") for t in (1, 2, 3)]
        U.filepaths_to_database(paths, {"timelapse": True}, str(src), "cell")
        df = _read_table(src / "measurements" / "measurements.db", "png_list")
        assert len(df) == 3 and "time_id" not in df.columns
        assert sorted(df["timeID"]) == ["t1", "t2", "t3"]
        assert set(df["fieldID"]) == {"f1"} and set(df["cell_id"]) == {"o5"}
        assert sorted(df["prcfo"]) == [f"plate1_r1_c1_f1_t{t}_o5" for t in (1, 2, 3)]

    with subtests.test(case="unregistered_crop_mode"):
        root = tmp_path / "unregistered"
        (root / "measurements").mkdir(parents=True)
        paths = [os.path.join(str(root), "data", "cell_png", f"plate1_A01_1_{i}.png")
                 for i in (1, 2)]
        U.filepaths_to_database(paths, {"timelapse": False}, str(root), "cell")
        registered = _read_table(root / "measurements" / "measurements.db", "png_list")
        assert list(registered["cell_id"]) == ["o1", "o2"]
        assert "unregistered" not in U.PNG_OBJECT_ID_COLUMNS
        with pytest.raises(ValueError, match="same length"):
            U.filepaths_to_database(paths, {"timelapse": False}, str(root), "unregistered")

    with subtests.test(case="unopenable_db"):
        src = tmp_path / "no_measurements"
        src.mkdir()
        capsys.readouterr()
        U.filepaths_to_database(_crop_png_paths(src, 2), {"timelapse": False}, str(src), "cell")
        assert "SQLite error" in capsys.readouterr().out
        assert not (src / "measurements").exists()


# ===========================================================================
# _generate_representative_images
# ===========================================================================

N_OBJ = 8


def _entity_frame(entity, n=N_OBJ, seed=0):
    rng = np.random.default_rng(seed)
    rows = ["r1" if i < n // 2 else "r2" for i in range(n)]
    cols = {"object_label": list(range(1, n + 1)), "plateID": ["plate1"] * n,
            "rowID": rows, "columnID": ["c1"] * n, "fieldID": ["f1"] * n,
            "prcf": [f"plate1_{r}_c1_f1" for r in rows]}
    if entity in ("nucleus", "pathogen"):
        cols["cell_id"] = list(range(1, n + 1))
    cols[f"{entity}_channel_1_mean_intensity"] = rng.uniform(100, 5000, n)
    cols[f"{entity}_area"] = rng.uniform(200, 4000, n)
    return pd.DataFrame(cols)


def _repr_db(src):
    """A measurements.db with the cell/cytoplasm/pathogen/png_list schema."""
    meas = src / "measurements"
    meas.mkdir(parents=True)
    db_path = meas / "measurements.db"
    rows = ["r1" if i < N_OBJ // 2 else "r2" for i in range(N_OBJ)]
    png_paths = [str(src / "data" / f"crop_{i + 1}.png") for i in range(N_OBJ)]
    con = sqlite3.connect(db_path)
    try:
        for seed, entity in enumerate(("cell", "cytoplasm", "pathogen")):
            _entity_frame(entity, seed=seed).to_sql(entity, con, index=False)
        pd.DataFrame({
            "cell_id": [f"o{i + 1}" for i in range(N_OBJ)], "png_path": png_paths,
            "plateID": ["plate1"] * N_OBJ, "rowID": rows, "columnID": ["c1"] * N_OBJ,
            "fieldID": ["f1"] * N_OBJ,
            "prcfo": [f"plate1_{rows[i]}_c1_f1_o{i + 1}" for i in range(N_OBJ)],
        }).to_sql("png_list", con, index=False)
    finally:
        con.close()
    return str(db_path), png_paths


def test_generate_representative_images_writes_one_figure_set_per_condition(
        tmp_path, subtests, monkeypatch):
    """For each cells/pathogens/treatments condition the top nr_imgs crops
    (ranked on the compartment ratio, a single compartment or cell_area) are
    plotted once combined and once per channel, saved as PDFs, optionally
    merged back into png_list; an unknown compartment names the options."""
    import spacr.plot

    calls = []

    def fake_grid(image_files, channel_indices, um_per_pixel, scale_bar_length_um=5,
                  fontsize=8, show_filename=True, channel_names=None, plot=False):
        calls.append({"files": list(image_files), "channel_indices": list(channel_indices)})
        fig = plt.figure(figsize=(2, 2))
        fig.add_subplot(111).plot([0, 1], [1, 0])
        return fig

    monkeypatch.setattr(spacr.plot, "_plot_images_on_grid", fake_grid)

    def pdfs(*conditions, channels):
        names = [f"measurements_proj_{c}.pdf" for c in conditions]
        names += [f"measurements_proj_channel_{ch}_{c}.pdf" for ch in channels
                  for c in conditions]
        return sorted(names)

    two_conditions = dict(cells="HeLa", pathogens=["rh"], treatments=["ctrl", "trt"],
                          treatment_loc=[["r1"], ["r2"]], channel_of_interest=1,
                          compartments=["pathogen", "cytoplasm"], nr_imgs=2, update_db=False)
    one_condition = dict(cells="HeLa", pathogens=["rh"], treatments=["ctrl"], nr_imgs=2,
                         channel_indices=[0], update_db=False)
    cases = [
        # label, kwargs, pdf names, channel_indices per call, files per call, png_list condition
        ("defaults", dict(cells=None, pathogens=None, treatments=None, compartments=None,
                          channel_indices=None, channel_of_interest=1, nr_imgs=2, plot=False,
                          update_db=True),
         pdfs("HeLa_rh_cm", channels=[0, 1, 2]), [[0, 1, 2], [0], [1], [2]], 2, {"HeLa_rh_cm"}),
        ("two_conditions_all_channels", dict(two_conditions, cell_loc=None, pathogen_loc=None,
                                             channel_indices=[0, 1], plot=False),
         pdfs("HeLa_rh_ctrl", "HeLa_rh_trt", channels=[0, 1]),
         [[0, 1], [0], [1], [0, 1], [0], [1]], 2, None),
        ("two_conditions_one_channel", dict(two_conditions, channel_indices=[0]),
         pdfs("HeLa_rh_ctrl", "HeLa_rh_trt", channels=[0]), [[0], [0], [0], [0]], 2, None),
        ("scalar_compartment", dict(one_condition, treatment_loc=None, compartments="cell",
                                    nr_imgs=3),
         pdfs("HeLa_rh_ctrl", channels=[0]), [[0], [0]], 3, None),
        ("single_compartment", dict(one_condition, compartments=["cell"]),
         pdfs("HeLa_rh_ctrl", channels=[0]), [[0], [0]], 2, None),
        ("no_compartment", dict(one_condition, compartments=[]),
         pdfs("HeLa_rh_ctrl", channels=[0]), [[0], [0]], 2, None),
    ]
    for label, kwargs, names, channel_calls, n_files, condition in cases:
        with subtests.test(case=label):
            calls.clear()
            src = tmp_path / label / "proj"
            db_path, png_paths = _repr_db(src)
            U._generate_representative_images(db_path, **kwargs)
            fig_dir = src / "figure"
            assert sorted(p.name for p in fig_dir.glob("*.pdf")) == names
            assert all((fig_dir / n).stat().st_size > 0 for n in names)
            assert [c["channel_indices"] for c in calls] == channel_calls
            assert all(len(c["files"]) == n_files for c in calls)
            assert all(set(c["files"]) <= set(png_paths) for c in calls)
            per_condition = len(channel_calls) // (2 if "two_conditions" in label else 1)
            # the combined figure's crops are reused for its per-channel figures
            for start in range(0, len(calls), per_condition):
                block = calls[start:start + per_condition]
                assert all(c["files"] == block[0]["files"] for c in block)
            if "two_conditions" in label:
                ctrl, trt = set(calls[0]["files"]), set(calls[-1]["files"])
                assert ctrl <= set(png_paths[:4]) and trt <= set(png_paths[4:])
                assert ctrl.isdisjoint(trt)
            png_list = _read_table(db_path, "png_list")
            if condition is None:
                assert "condition" not in png_list.columns
            else:
                assert set(png_list["condition"]) == condition
                assert set(png_list["host_cells"]) == {"HeLa"}
                assert set(png_list["pathogen"]) == {"rh"}
                assert len(png_list) == N_OBJ

    with subtests.test(case="unknown_compartment"):
        db_path, _ = _repr_db(tmp_path / "typo" / "proj")
        with pytest.raises(KeyError) as exc:
            U._generate_representative_images(db_path, **dict(one_condition,
                                                               compartments=["nucleuss"]))
        assert "nucleuss" in str(exc.value) and "no column" in str(exc.value)


# ===========================================================================
# debug decorator
# ===========================================================================

class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_debug_logs_entry_and_exit_at_debug_level_and_restores_the_logger(subtests):
    """Enabled, the wrapped call runs with its logger at DEBUG, logs
    '>>> Entering f' then '<<< Exiting f' (no exit line when it raises) and
    restores the caller's level; disabled it is a pure passthrough; the
    default logger is the wrapped function's module."""
    def add(a, b=1):
        return a + b

    def boom():
        raise ValueError("kaboom")

    cases = [
        # label, enabled, logger name, level, func, call args, expected, raises, logs
        ("disabled", False, "spacr.test.debug.disabled", logging.WARNING, add,
         ((2,), {"b": 3}), 5, None, []),
        ("enabled", True, "spacr.test.debug.enabled", logging.ERROR, add,
         ((6, 36), {}), 42, None, [">>> Entering add", "<<< Exiting add"]),
        ("raises", True, "spacr.test.debug.raises", logging.CRITICAL, boom,
         ((), {}), None, "kaboom", [">>> Entering boom"]),
        ("default_logger", None, None, logging.NOTSET, add,
         (("he", "llo"), {}), "hello", None, [">>> Entering add", "<<< Exiting add"]),
    ]
    for label, enabled, name, level, func, (args, kwargs), expected, raises, logs in cases:
        with subtests.test(case=label):
            log = logging.getLogger(name or __name__)
            during = []

            def recorded(*a, _func=func, **k):
                during.append(log.level)
                return _func(*a, **k)

            recorded.__name__ = func.__name__
            recorded.__module__ = __name__
            decorator = U.debug() if enabled is None else U.debug(enabled=enabled,
                                                                  logger_name=name)
            wrapped = decorator(recorded)
            assert wrapped.__name__ == func.__name__
            log.setLevel(level)
            handler = _Collector()
            log.addHandler(handler)
            log.propagate = False
            try:
                if raises:
                    with pytest.raises(ValueError, match=raises):
                        wrapped(*args, **kwargs)
                else:
                    assert wrapped(*args, **kwargs) == expected
                assert log.level == level
            finally:
                log.removeHandler(handler)
                log.propagate = True
                log.setLevel(logging.NOTSET)
            assert during == [level if enabled is False else logging.DEBUG]
            assert [r.getMessage() for r in handler.records] == logs
            assert all(r.levelno == logging.DEBUG for r in handler.records)


# ===========================================================================
# multiprocessing helpers
# ===========================================================================

class _FakeProc:
    """Minimal psutil.Process stand-in for the mp helpers."""

    def __init__(self, pid, cmdline, cmdline_exc=None, terminate_exc=None):
        self.info = {"pid": pid, "cmdline": cmdline}
        self._cmdline = cmdline
        self._cmdline_exc = cmdline_exc
        self._terminate_exc = terminate_exc
        self.terminated = False
        self.waited_timeout = None

    def cmdline(self):
        if self._cmdline_exc is not None:
            raise self._cmdline_exc
        return self._cmdline

    def terminate(self):
        if self._terminate_exc is not None:
            raise self._terminate_exc
        self.terminated = True

    def wait(self, timeout=None):
        self.waited_timeout = timeout


def test_is_multiprocessing_process_matches_the_command_line(subtests):
    """A process whose command line mentions multiprocessing is one; any
    other, or one psutil cannot read, is not."""
    cases = [
        ("spawn_import", _FakeProc(1, ["python", "-c", "from multiprocessing.spawn import x"]),
         True),
        ("plain_spawn", types.SimpleNamespace(
            cmdline=lambda: ["python", "-c", "multiprocessing.spawn"]), True),
        ("pytest", _FakeProc(2, ["python", "-m", "pytest"]), False),
        ("app", types.SimpleNamespace(cmdline=lambda: ["python", "app.py"]), False),
    ]
    cases += [(type(exc).__name__, _FakeProc(4242, [], cmdline_exc=exc), False)
              for exc in (psutil.NoSuchProcess(4242), psutil.AccessDenied(4242),
                          psutil.ZombieProcess(4242))]
    for label, proc, expected in cases:
        with subtests.test(case=label):
            assert U.is_multiprocessing_process(proc) is expected


def test_close_multiprocessing_processes_terminates_children_then_sweeps_fds(
        subtests, monkeypatch, capsys):
    """Every multiprocessing child except this process is terminated and
    waited on (5 s), failures are reported without stopping the loop, other
    processes are untouched, and the descriptor sweep always runs last."""
    me = os.getpid()
    cases = {
        "only_mp_children": [
            _FakeProc(me, ["python", "-c", "from multiprocessing import x"]),
            _FakeProc(me + 1, ["python", "-c", "from multiprocessing.spawn import y"]),
            _FakeProc(me + 2, ["bash", "-lc", "sleep 1"])],
        "failures_reported": [
            _FakeProc(me + 11, ["python", "multiprocessing.forkserver"],
                      terminate_exc=psutil.NoSuchProcess(me + 11)),
            _FakeProc(me + 12, ["python", "multiprocessing.resource_tracker"],
                      terminate_exc=psutil.AccessDenied(me + 12)),
            _FakeProc(me + 13, ["python", "-c", "import multiprocessing"])],
        "no_processes": [],
    }
    for label, procs in cases.items():
        with subtests.test(case=label), monkeypatch.context() as mp:
            order = []

            def process_iter(attrs=None, _procs=procs):
                order.append("iter")
                return iter(_procs)

            mp.setattr(U.psutil, "process_iter", process_iter)
            mp.setattr(U, "close_file_descriptors", lambda: order.append("sweep"))
            capsys.readouterr()
            U.close_multiprocessing_processes()
            out = capsys.readouterr().out
            assert order == ["iter", "sweep"]
            if label == "only_mp_children":
                assert [p.terminated for p in procs] == [False, True, False]
                assert procs[1].waited_timeout == 5
                assert f"Terminated process {me + 1}" in out
                assert f"Terminated process {me}" not in out.replace(f"Terminated process {me + 1}", "")
            elif label == "failures_reported":
                assert f"Failed to terminate process {me + 11}" in out
                assert f"Failed to terminate process {me + 12}" in out
                assert procs[2].terminated is True
                assert f"Terminated process {me + 13}" in out
            else:
                assert "Terminated" not in out


def test_close_file_descriptors_closes_3_up_to_the_soft_limit(subtests, monkeypatch):
    """Descriptors 3..soft-1 are closed (an already-closed one is ignored),
    0-2 never; with soft == 3 or no resource module nothing is closed."""
    cases = [("soft_6_fd4_already_closed", (6, 1024), [3, 5]),
             ("soft_3", (3, 1024), []),
             ("no_resource_module", None, [])]
    for label, limits, expected in cases:
        with subtests.test(case=label):
            if limits is not None and resource is None:
                pytest.skip("resource limits are POSIX-only")
            closed = []

            def fake_close(fd, _closed=closed):
                if fd == 4:
                    raise OSError(9, "Bad file descriptor")
                _closed.append(fd)

            with monkeypatch.context() as mp:
                if limits is None:
                    mp.setitem(sys.modules, "resource", None)
                else:
                    mp.setattr(resource, "getrlimit", lambda which, _l=limits: _l)
                mp.setattr(U.os, "close", fake_close)
                U.close_file_descriptors()
            assert closed == expected


# ===========================================================================
# print_progress
# ===========================================================================

def test_print_progress_reports_counts_timing_and_eta(subtests, capsys):
    """Progress prints done/total (lists counted as unique items, numeric
    strings and floats coerced, garbage as 0), time per batch and image (a
    list batch_size counts its length) and the ETA, staying finite for zero
    workers or batch size and printing None without timings."""
    cases = [
        ("list_batch_size", dict(files_processed=10, files_to_process=20, n_jobs=2,
                                 time_ls=[4.0, 4.0], batch_size=["f1", "f2", "f3", "f4"],
                                 operation_type="measure"),
         ["Progress: 10/20", "Time/batch: 4.000sec", "Time/image: 1.000sec",
          "Time_left: 0.333 min.", "operation_type: measure"]),
        ("zero_workers", dict(files_processed=2, files_to_process=10, n_jobs=0,
                              time_ls=[1.0, 3.0], batch_size=0, operation_type="in-process"),
         ["Time/image: 2.000sec", "Time_left: 0.267 min."]),
        ("coerced_counters", dict(files_processed="7", files_to_process=12.9, n_jobs=1,
                                  time_ls=None, operation_type="coerce"),
         ["Progress: 7/12,", "None"]),
        ("uncoercible_counters", dict(files_processed=object(), files_to_process="not-a-number",
                                      n_jobs=1, time_ls=[1.0], operation_type="junk"),
         ["Progress: 0/0,", "operation_type: junk", "Time_left: 0.000 min."]),
        ("lists_deduplicated", dict(files_processed=[1, 1, 2], files_to_process=[1, 2, 3, 3, 4],
                                    n_jobs=1, time_ls=[]),
         ["Progress: 2/4,", "Time/image: 0.000sec"]),
        ("plain_counts", dict(files_processed=5, files_to_process=10, n_jobs=1,
                              time_ls=[1.0, 2.0], operation_type="seg"),
         ["Progress: 5/10,", "operation_type: seg"]),
        ("list_counts_int_batch", dict(files_processed=[1, 2, 3], files_to_process=[1, 2, 3, 4],
                                       n_jobs=2, time_ls=[0.5], batch_size=4),
         ["Progress: 3/4,"]),
        ("no_timings", dict(files_processed=3, files_to_process=10, n_jobs=1, time_ls=None),
         ["Progress: 3/10,", "None"]),
    ]
    for label, kwargs, fragments in cases:
        with subtests.test(case=label):
            capsys.readouterr()
            U.print_progress(**kwargs)
            out = capsys.readouterr().out
            for fragment in fragments:
                assert fragment in out
            assert "nan" not in out.lower() and "inf" not in out.lower()
