"""Item 998: PERISCOPE painting features measured through sequencing-frame masks.

PERISCOPE publishes its painting crops in the painting frame and its Cells and
Nuclei masks in the sequencing frame; its 04Align logs the shift per site.
Pinned here (spec T1-T5): with the logged shift the features equal the
unshifted truth exactly, except for flagged cells at the zero-filled border
(T1); a site without a record, a NaN record, a failed registration and
sequencing-frame masks without a shift are refused before anything is
written (T2); 'none' with painting-frame masks is the plain measurement, byte
for byte, and the default cropped-site and tile phases do not run the new
phase (T3); the per-site shift table equals the Image.csv values and a rerun
from the stamped settings file gives identical tables (T4); and, opt-in on
the CP186A data, per-nucleus DNA against our CellProfiler 4.1.3 rerun of
PERISCOPE's pipeline (T5).
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import cpg_cropped as C
from spacr import cpg_measure as M
from spacr.ops_sample import measure_phenotype_field

BATCH = "20200101_TEST"
CHANNELS = ("CorrDNA", "CorrER")
LOGGED = {1: (6, -5), 2: (-4, 7)}
SIZE = 160
#: Thresholded on the whole field's percentiles, which the zero band moves.
FIELD_REFERENCED = ("_frac_high90", "_frac_low10")


def planted_site(seed):
    """Sequencing-frame masks and painting truth with nuclei along every edge."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    cells = np.zeros((SIZE, SIZE), np.int32)
    nuclei = np.zeros((SIZE, SIZE), np.int32)
    label = 0
    for cy in range(12, SIZE, 27):
        for cx in range(12, SIZE, 27):
            label += 1
            cy2, cx2 = cy + int(rng.integers(-2, 3)), cx + int(rng.integers(-2, 3))
            cells[(yy - cy2) ** 2 + (xx - cx2) ** 2 <= 11 ** 2] = label
            nuclei[(yy - cy2) ** 2 + (xx - cx2) ** 2 <= 6 ** 2] = label
    truth = np.zeros((SIZE, SIZE, len(CHANNELS)), np.uint16)
    for k in range(len(CHANNELS)):
        noise = rng.integers(50, 120, (SIZE, SIZE))
        signal = rng.integers(400, 3000, (SIZE, SIZE)) * (nuclei > 0 if k == 0 else cells > 0)
        truth[..., k] = (noise + signal).astype(np.uint16)
    return nuclei, cells, truth


def write_run(tmp_path, *, logged=LOGGED, empty=(), skip=()):
    """A cpg0021-like tree: painting crops in the painting frame, masks and Image.csv."""
    top = tmp_path / "cpg"
    well = top / "images" / BATCH / "images_corrected_cropped" / "CP9_Well1"
    masks = tmp_path / "masks"
    masks.mkdir(parents=True)
    truths = {}
    for site, (y, x) in logged.items():
        nuclei, cells, truth = planted_site(site)
        truths[site] = (nuclei, cells, truth)
        dy, dx = -y, -x
        for k, name in enumerate(CHANNELS):
            (well / name).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff",
                             C.shift_image(truth[..., k], -dy, -dx))
        (well / "Cycle01_DAPI").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(well / "Cycle01_DAPI" / f"Cycle01_DAPI_Site_{site}.tiff", truth[..., 0])
        tifffile.imwrite(masks / f"{site}_cells.tiff", cells.astype(np.uint16))
        tifffile.imwrite(masks / f"{site}_nuclei.tiff", nuclei.astype(np.uint16))
        if site in skip:
            continue
        folder = top / "workspace" / "analysis" / BATCH / f"CP9-Well1-{site}"
        folder.mkdir(parents=True, exist_ok=True)
        text = ("nan", "nan") if site in empty else (x, y)
        (folder / "Image.csv").write_text(
            "ImageNumber,Align_Xshift_DAPI_Painting,Align_Yshift_DAPI_Painting\n"
            f"1,{text[0]},{text[1]}\n")
    settings = {"ops_layout": "cpg_cropped", "cpg_root": str(well),
                "cpg_phenotype_channels": list(CHANNELS), "cpg_sbs_cycles": 0,
                "cpg_sbs_dapi": None, "dst_root": str(tmp_path / "run"),
                "ops_prefix_masks": str(masks / "{site}_cells.tiff"),
                "ops_prefix_nuclei_masks": str(masks / "{site}_nuclei.tiff"),
                "ops_prefix_masks_frame": "sbs",
                "cpg_painting_frame_shift": "periscope_logged"}
    return settings, truths


def results(settings):
    return os.path.join(settings["dst_root"], "results", "cpg_measure")


def test_t1_logged_shift_gives_the_truth_except_flagged_border_cells(tmp_path):
    settings, truths = write_run(tmp_path)
    out = M.run_cpg_measure(settings, report=lambda text: None)
    table = pd.read_parquet(out["tables"]["CP9_W1"])
    names = M.painting_column_names(CHANNELS)
    assert names == ["dna", "er"]
    for site, (nuclei, cells, truth) in truths.items():
        want = measure_phenotype_field(nuclei, cells, truth, names).set_index("label")
        got = table[table["site"] == site].set_index("label")
        border = got["painting_shift_border"]
        assert 0 < int(border.sum()) < len(got)
        dy, dx = -LOGGED[site][0], -LOGGED[site][1]
        np.testing.assert_array_equal(np.sort(border[border].index.to_numpy()),
                                      M.shift_border_labels(nuclei, cells, dy, dx))
        keep = border[~border].index
        columns = [c for c in want.columns if c in got.columns
                   and not c.endswith(FIELD_REFERENCED)]
        for column in ("nucleus_dna_integrated_intensity", "nucleus_dna_mean_intensity",
                       "cell_er_mean_intensity"):
            assert column in columns
        assert any("homogeneity" in c or "texture" in c for c in columns)
        pd.testing.assert_frame_equal(got.loc[keep, columns], want.loc[keep, columns],
                                      check_exact=True)
        unmoved = measure_phenotype_field(
            nuclei, cells, np.stack([C.shift_image(truth[..., k], -dy, -dx)
                                     for k in range(truth.shape[-1])], -1), names
        ).set_index("label")
        assert not np.allclose(unmoved.loc[keep, "nucleus_dna_integrated_intensity"],
                               want.loc[keep, "nucleus_dna_integrated_intensity"])


def test_t2_refusals_name_the_site_and_write_nothing(tmp_path, monkeypatch):
    settings, _ = write_run(tmp_path / "a", skip=(2,))
    with pytest.raises(ValueError, match="site 2 of CP9 Well1"):
        M.run_cpg_measure(settings, report=lambda text: None)
    assert not os.path.isdir(results(settings))
    settings, _ = write_run(tmp_path / "b", empty=(1,))
    with pytest.raises(ValueError, match="No recorded Align shift for site 1 of CP9 Well1"):
        M.run_cpg_measure(settings, report=lambda text: None)
    assert not os.path.isdir(results(settings))

    from spacr import ops_register

    class Refused:
        accepted, dy, dx, peak_ratio = False, 0, 0, 1.0

    monkeypatch.setattr(ops_register, "phase_correlate", lambda *a, **k: Refused())
    with pytest.raises(ValueError, match="site 1: the painting-to-sequencing registration"):
        M.run_cpg_measure({**settings, "cpg_painting_frame_shift": "register"},
                          report=lambda text: None)
    assert not os.path.isdir(results(settings))
    for mode in ("none", None, ""):
        with pytest.raises(ValueError, match="sequencing frame but the painting images are not"):
            M.run_cpg_measure({**settings, "cpg_painting_frame_shift": mode})
    with pytest.raises(ValueError, match="must not be moved"):
        M.painting_frame_mode({"ops_prefix_masks_frame": "painting",
                               "cpg_painting_frame_shift": "periscope_logged"})
    with pytest.raises(ValueError, match="cpg_painting_frame_shift must be one of"):
        M.painting_frame_mode({"cpg_painting_frame_shift": "guess"})
    assert not os.path.isdir(results(settings))


def test_t3_none_with_painting_masks_is_the_plain_measurement(tmp_path):
    settings, _ = write_run(tmp_path)
    settings = {**settings, "ops_prefix_masks_frame": "painting",
                "cpg_painting_frame_shift": "none"}
    out = M.run_cpg_measure(settings, report=lambda text: None)
    table = pd.read_parquet(out["tables"]["CP9_W1"])
    shifts = pd.read_csv(out["shifts"]["CP9_W1"])
    assert (shifts["dy"] == 0).all() and (shifts["dx"] == 0).all()
    assert not table["painting_shift_border"].any()
    index = C.cpg_cropped_index(settings["cpg_root"])
    for site in LOGGED:
        painting = C.load_cpg_site(index, "CP9", "Well1", site, phenotype_channels=CHANNELS,
                                   parts=("phenotype",))["phenotype"]
        nuclei = tifffile.imread(settings["ops_prefix_nuclei_masks"].format(site=site))
        cells = tifffile.imread(settings["ops_prefix_masks"].format(site=site))
        plain = measure_phenotype_field(nuclei.astype(np.int32), cells.astype(np.int32),
                                        painting, ["dna", "er"])
        got = table[table["site"] == site].drop(
            columns=["plate", "well", "site", "painting_shift_border"]).reset_index(drop=True)
        assert list(got.columns) == list(plain.columns)
        assert got.to_numpy().tobytes() == plain.to_numpy().tobytes()


def test_t3_default_and_tile_phases_do_not_measure(tmp_path):
    from spacr import ops_engine
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["cpg_painting_frame_shift"] == "periscope_logged"
    assert "cpg_measure" in ops_engine._CPG_PHASES
    assert ops_engine._CPG_PHASES[:2] == ("cpg_index", "cpg_merged")
    settings, _ = write_run(tmp_path)
    out = ops_engine._run_cpg(settings, ops_engine._PHASES)
    assert out["phases"] == ["cpg_index", "cpg_merged"] and "cpg_measure" not in out
    assert not os.path.isdir(results(settings))
    with pytest.raises(ValueError, match="cpg_cropped"):
        ops_engine._run_cpg({**settings, "ops_layout": "tiles"}, ("cpg_measure",))


def test_t4_shift_table_and_settings_file_rerun(tmp_path):
    from spacr import ops_engine

    settings, _ = write_run(tmp_path)
    out = ops_engine._run_cpg(settings, ("cpg_measure",))["cpg_measure"]
    shifts = pd.read_csv(out["shifts"]["CP9_W1"])
    assert list(shifts.columns) == list(M.SHIFT_TABLE_COLUMNS)
    assert shifts["site"].tolist() == sorted(LOGGED)
    for row in shifts.itertuples():
        assert (row.dy, row.dx) == (-LOGGED[row.site][0], -LOGGED[row.site][1])
        assert (row.border_y_px, row.border_x_px) == (abs(row.dy), abs(row.dx))
        assert row.mode == "periscope_logged" and row.source.endswith("Image.csv")
    table = pd.read_parquet(out["tables"]["CP9_W1"])
    assert shifts["cells"].sum() == len(table) == out["cells"]
    assert shifts["border_cells"].sum() == int(table["painting_shift_border"].sum())
    with open(out["settings"], encoding="utf-8") as handle:
        stamped = json.load(handle)
    assert stamped["cpg_painting_frame_shift"] == "periscope_logged"
    keep = tmp_path / "first"
    shutil.copytree(results(settings), keep)
    again = M.run_cpg_measure(stamped, report=lambda text: None)
    with open(again["shifts"]["CP9_W1"], "rb") as handle:
        assert handle.read() == (keep / "painting_frame_shift_CP9_W1.csv").read_bytes()
    pd.testing.assert_frame_equal(pd.read_parquet(again["tables"]["CP9_W1"]),
                                  pd.read_parquet(keep / "cells_CP9_W1.parquet"),
                                  check_exact=True)


CP186A = os.environ.get("SPACR_TEST_CP186A_998")


@pytest.mark.skipif(not CP186A, reason="real-data check: set SPACR_TEST_CP186A_998 to "
                    "'<cpg CP186A_Well1 crops>:<CellProfiler 4.1.3 site_<s> output root>'")
def test_t5_cp186a_dna_matches_cellprofiler(tmp_path):
    crops, cp_root = CP186A.split(":")
    sites = (45, 64, 98)
    base = {"ops_layout": "cpg_cropped", "cpg_root": crops, "cpg_sites": list(sites),
            "cpg_phenotype_channels": ["CorrDNA"], "cpg_sbs_cycles": 0, "cpg_sbs_dapi": None,
            "cpg_align_table": os.path.join(cp_root, "site_{site}", "Image.csv"),
            "ops_prefix_masks": os.path.join(
                cp_root, "site_{site}", "segmentation_masks",
                "Plate_{plate}_Well_{well}_Site_{site}_Cells_Objects.tiff"),
            "ops_prefix_nuclei_masks": os.path.join(
                cp_root, "site_{site}", "segmentation_masks",
                "Plate_{plate}_Well_{well}_Site_{site}_Nuclei_Objects.tiff"),
            "ops_ph_texture": False}
    cp = pd.concat([pd.read_csv(os.path.join(cp_root, f"site_{s}", "Nuclei.csv"),
                                usecols=["ObjectNumber",
                                         "Intensity_IntegratedIntensity_DAPI_Painting"]
                                ).assign(site=s) for s in sites])
    cp = cp.rename(columns={"ObjectNumber": "label",
                            "Intensity_IntegratedIntensity_DAPI_Painting": "cp"})
    found = {}
    for mode, frame in (("periscope_logged", "sbs"), ("none", "painting")):
        out = M.run_cpg_measure({**base, "dst_root": str(tmp_path / mode),
                                 "cpg_painting_frame_shift": mode,
                                 "ops_prefix_masks_frame": frame},
                                report=lambda text: None)
        table = pd.read_parquet(out["tables"]["CP186A_W1"])
        table = table[~table["painting_shift_border"]]
        joined = table.merge(cp, on=["site", "label"])
        spacr_dna = joined["nucleus_dna_integrated_intensity"].to_numpy(float)
        cp_dna = joined["cp"].to_numpy(float) * 65535.0
        r = float(np.corrcoef(np.log2(np.clip(spacr_dna, 1e-6, None)),
                              np.log2(np.clip(cp_dna, 1e-6, None)))[0, 1])
        found[mode] = {"nuclei": int(len(joined)), "pearson_log2": r,
                       "max_rel_diff": float(np.max(np.abs(spacr_dna - cp_dna)
                                                    / np.maximum(np.abs(cp_dna), 1e-9)))}
    print(json.dumps(found, indent=1))
    target = os.environ.get("SPACR_TEST_CP186A_998_OUT")
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(found, handle, indent=1)
    assert found["periscope_logged"]["pearson_log2"] >= 0.9999
    assert found["none"]["pearson_log2"] < 0.9
