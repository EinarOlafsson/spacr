"""Item 921: tiled segmentation of large cropped OPS sites, chunked
measurement, the RAM planner, the crop-edge policy and the Phalloidin preset."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from spacr import ops_large_sites as ols


def _discs(shape, centres, radius):
    """A boolean image with one disc per centre."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    out = np.zeros(shape, bool)
    for cy, cx in centres:
        out |= (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    return out


def _label(image):
    """A translation-invariant stand-in segmenter: connected components."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., 0]
    return ndimage.label(image > 0.5)[0].astype(np.int32)


class _FakeModel:
    """Anything with Cellpose's eval: normalises unless told not to, then
    thresholds its first channel."""

    def __init__(self, threshold=0.5):
        self.calls = []
        self.threshold = threshold

    def eval(self, x, batch_size=8, **kwargs):
        """Record the call and return ``(labels,)``."""
        self.calls.append(kwargs)
        x = ols.normalize_site(x) if kwargs.get("normalize", True) else np.asarray(x)
        x = x[..., 0] if x.ndim == 3 else x
        return (ndimage.label(x > self.threshold)[0].astype(np.int32),)


def _models():
    """Nucleus and cell stand-ins with thresholds that suit :func:`_site`."""
    return _FakeModel(2.0), _FakeModel(0.5)


def test_discs_crossing_tile_seams_are_found_once_with_their_area():
    shape = (6000, 6000)
    seams = [1792, 2048, 3584, 3840, 5376]
    centres = [(y, x) for y in seams for x in seams] + [(100, 3000), (3000, 5950)]
    radius = 40
    image = _discs(shape, centres, radius)
    labels = ols.segment_tiled(image, _label, tile_px=2048, overlap_px=256)
    assert labels.dtype == np.int32
    assert labels.max() == len(centres)
    expected = _label(image)
    areas = np.sort(np.bincount(labels.ravel())[1:])
    assert np.array_equal(areas, np.sort(np.bincount(expected.ravel())[1:]))
    assert np.array_equal(labels > 0, expected > 0)
    assert np.array_equal(labels, ols.relabel_raster(expected))


def test_a_site_within_one_tile_takes_the_untiled_path_unchanged():
    image = _discs((300, 400), [(100, 100), (200, 300)], 20)
    sentinel = np.full((300, 400), 7, np.int32)
    assert ols.segment_tiled(image, lambda x: sentinel, tile_px=2048) is not None
    assert np.array_equal(ols.segment_tiled(image, lambda x: sentinel, tile_px=2048), sentinel)
    assert np.array_equal(ols.segment_tiled(image, lambda x: sentinel, tile_px=0), sentinel)
    assert ols.tile_grid((300, 400), 0, 0) == [(0, 300, 0, 400)]


def test_tiled_equals_untiled_on_a_region_that_fits_untiled():
    rng = np.random.default_rng(0)
    centres = [tuple(c) for c in rng.integers(30, 2018, size=(120, 2))]
    image = _discs((2048, 2048), centres, 12)
    untiled = ols.relabel_raster(_label(image))
    tiled = ols.segment_tiled(image, _label, tile_px=700, overlap_px=100)
    assert np.array_equal(tiled, untiled)


def test_tile_grid_is_row_major_full_size_and_covers_the_image():
    boxes = ols.tile_grid((5500, 5500), 2048, 256)
    assert len(boxes) == 9
    assert boxes[0] == (0, 2048, 0, 2048) and boxes[-1] == (3452, 5500, 3452, 5500)
    assert all(y1 - y0 == 2048 and x1 - x0 == 2048 for y0, y1, x0, x1 in boxes)
    assert boxes == sorted(boxes)
    with pytest.raises(ValueError, match="smaller"):
        ols.tile_grid((5000, 5000), 256, 256)


def test_stitching_keeps_the_copy_farther_from_the_tile_edge():
    shape = (100, 200)
    left = np.zeros((100, 120), np.int32)
    left[40:60, 100:120] = 1
    right = np.zeros((100, 120), np.int32)
    right[40:60, 20:45] = 3
    out = ols.stitch_tile_labels([((0, 100, 0, 120), left), ((0, 100, 80, 200), right)], shape)
    assert out.max() == 1
    assert (out == 1).sum() == 20 * 25
    assert out[50, 124] == 1 and out[50, 125] == 0


def test_the_planner_halves_the_tile_and_refuses_an_impossible_cap():
    big = 2 ** 40
    plan = ols.plan_site_memory((5500, 5500), 5, {"seg_max_ram_gb": 24, "seg_tile_px": 2048,
                                                   "seg_workers": 4}, available_bytes=big)
    assert plan["tile_px"] == 2048 and plan["workers"] == 4 and plan["estimate_gb"] <= 24
    small = ols.plan_site_memory((5500, 5500), 5, {"seg_max_ram_gb": 2.2, "seg_tile_px": 2048,
                                                    "seg_tile_overlap_px": 64},
                                 available_bytes=big)
    assert small["tile_px"] < 2048 and small["estimate_gb"] <= 2.2
    whole = ols.plan_site_memory((5500, 5500), 5, {"seg_max_ram_gb": 3, "seg_tile_px": 0},
                                 available_bytes=big)
    assert whole["tile_px"] in (512, 1375, 2750)
    with pytest.raises(ols.MemoryPlanError, match="cannot hold"):
        ols.plan_site_memory((5500, 5500), 5, {"seg_max_ram_gb": 1}, available_bytes=big)
    with pytest.raises(ols.MemoryPlanError, match="available"):
        ols.plan_site_memory((5500, 5500), 5, {"seg_max_ram_gb": 24}, available_bytes=2 ** 30)
    assert ols.plan_site_memory((100, 100), 1, {"seg_max_ram_gb": 24})["workers"] >= 0


def test_edge_policies_count_and_drop_border_objects():
    labels = np.zeros((50, 50), np.int32)
    labels[0:5, 10:15] = 1
    labels[20:25, 20:25] = 2
    labels[45:50, 45:50] = 3
    kept, counts = ols.apply_edge_policy(labels, "keep")
    assert np.array_equal(kept, labels) and counts == {"edge_objects": 2, "excluded": 0, "owned": 0}
    out, counts = ols.apply_edge_policy(labels, "exclude")
    assert set(np.unique(out)) == {0, 2} and counts["excluded"] == 2
    with pytest.raises(ValueError, match="seg_edge_policy"):
        ols.apply_edge_policy(labels, "bogus")


def test_own_gives_a_cut_cell_to_exactly_one_of_two_crops():
    top = np.zeros((40, 40), np.int32)
    bottom = np.zeros((40, 40), np.int32)
    top[30:40, 5:15] = 1
    bottom[0:4, 5:15] = 1
    top[36:40, 25:35] = 2
    bottom[0:10, 25:35] = 2
    top[36:40, 0:2] = 3
    a, ca = ols.apply_edge_policy(top, "own", neighbours={"bottom": bottom})
    b, cb = ols.apply_edge_policy(bottom, "own", neighbours={"top": top})
    assert 1 in a and 1 not in b
    assert 2 not in a and 2 in b
    assert 3 not in a
    assert ca["owned"] == 1 and cb["owned"] == 1
    tie_top = np.zeros((10, 10), np.int32)
    tie_bottom = np.zeros((10, 10), np.int32)
    tie_top[8:10, 2:4] = 1
    tie_bottom[0:2, 2:4] = 1
    x, _ = ols.apply_edge_policy(tie_top, "own", neighbours={"bottom": tie_bottom})
    y, _ = ols.apply_edge_policy(tie_bottom, "own", neighbours={"top": tie_top})
    assert (1 in x) != (1 in y)
    with pytest.raises(ValueError, match="seam"):
        ols.apply_edge_policy(tie_top, "own", neighbours={"bottom": np.zeros((10, 12), np.int32)})


def test_chunked_measurement_equals_whole_image_measurement():
    rng = np.random.default_rng(1)
    centres = [tuple(c) for c in rng.integers(20, 480, size=(40, 2))]
    labels = ols.relabel_raster(_label(_discs((500, 500), centres, 9)))
    image = rng.random((500, 500, 2))
    whole = ols.measure_chunked(labels, image, band_px=0, channel_names=["a", "b"])
    banded = ols.measure_chunked(labels, image, band_px=64, channel_names=["a", "b"])
    assert len(banded) == labels.max() == banded["label"].nunique()
    pd.testing.assert_frame_equal(whole, banded, check_exact=False, rtol=1e-12)
    assert {"a_mean", "b_sum", "eccentricity", "centroid_y"} <= set(banded.columns)
    empty = ols.measure_chunked(np.zeros((300, 30), np.int32), image[:300, :30], band_px=64)
    assert empty.empty
    gray = ols.basic_features(labels, image[..., 0])
    assert "ch0_mean" in gray


def test_nucleus_diameter_is_clipped_to_the_cellprofiler_range():
    tiny = ols.relabel_raster(_label(_discs((100, 100), [(50, 50)], 2)))
    assert ols.estimate_nucleus_diameter([tiny]) == 10.0
    mid = ols.relabel_raster(_label(_discs((200, 200), [(50, 50), (150, 150)], 15)))
    assert 29 < ols.estimate_nucleus_diameter([mid]) < 32
    assert ols.estimate_nucleus_diameter([], (12.0, 80.0)) == 12.0
    big = ols.relabel_raster(_label(_discs((400, 400), [(200, 200)], 60)))
    assert ols.estimate_nucleus_diameter([big]) == 80.0


def test_preset_settings_expand_and_caller_values_win():
    s = ols.large_site_settings({"seg_preset": "cell_painting_phalloidin", "seg_max_ram_gb": 12})
    assert s["cell_channels"] == ["CorrPhalloidin", "CorrDNA"]
    assert s["seg_nucleus_diameter_range"] == [10.0, 80.0] and s["seg_max_ram_gb"] == 12
    assert ols.large_site_settings({})["seg_tile_px"] == 2048
    with pytest.raises(ValueError, match="seg_preset"):
        ols.large_site_settings({"seg_preset": "nope"})
    with pytest.raises(ValueError, match="seg_edge_policy"):
        ols.large_site_settings({"seg_edge_policy": "nope"})


def _site(shape=(600, 600), centres=None, seed=0):
    """A synthetic 5-channel site: nuclei in DNA, larger cells in Phalloidin,
    on a dim noisy background."""
    if centres is None:
        centres = [(100, 100), (100, 400), (300, 250), (500, 120), (480, 480), (5, 300)]
    noise = np.random.default_rng(seed).integers(0, 50, size=shape + (2,)).astype(np.uint16)
    dna = _discs(shape, centres, 10).astype(np.uint16) * 1000 + noise[..., 0]
    actin = _discs(shape, centres, 25).astype(np.uint16) * 1000 + noise[..., 1]
    other = np.zeros(shape, np.uint16)
    return np.stack([dna, actin, other, other, other], axis=-1), centres


def test_the_preset_pairs_one_nucleus_per_cell_and_tiles_both():
    stack, centres = _site()
    nucleus_model, cell_model = _models()
    settings = ols.large_site_settings({"seg_preset": "cell_painting_phalloidin",
                                        "seg_tile_px": 256, "seg_tile_overlap_px": 64})
    nuclei, cells, counts = ols.segment_cell_painting_site(
        stack, ols.LARGE_SITE_DEFAULTS["seg_site_channels"], settings,
        (nucleus_model, cell_model), nucleus_diameter=20.0)
    assert nuclei.max() == cells.max() == len(centres)
    for k in range(1, len(centres) + 1):
        assert np.all(cells[nuclei == k] == k)
    assert all(call.get("diameter") == 20.0 for call in nucleus_model.calls)
    assert all(call.get("channel_axis") == -1 for call in cell_model.calls)
    assert all(call.get("normalize") is False
               for call in nucleus_model.calls + cell_model.calls)
    assert len(nucleus_model.calls) == 9
    with pytest.raises(ValueError, match="preset needs"):
        ols.segment_cell_painting_site(stack, ["a", "b", "c", "d", "e"], settings,
                                       (nucleus_model, cell_model))


def test_mask_agreement_scores_matches_and_ignores_edge_objects():
    ref = ols.relabel_raster(_label(_discs((200, 200), [(50, 50), (150, 150), (100, 2)], 10)))
    same = mask_same = ref.copy()
    score = ols.mask_agreement(same, ref)
    assert score["n_reference"] == 2 and score["f1"] == 1.0 and score["median_iou"] == 1.0
    shifted = np.zeros_like(ref)
    shifted[:, 15:] = ref[:, :-15]
    score = ols.mask_agreement(shifted, ref)
    assert score["matched"] == 0 and score["f1"] == 0.0 and np.isnan(score["median_iou"])
    assert ols.mask_agreement(mask_same, ref, drop_edge=False)["n_reference"] == 3
    empty = ols.mask_agreement(np.zeros((5, 5), np.int32), np.zeros((5, 5), np.int32))
    assert np.isnan(empty["f1"])
    with pytest.raises(ValueError, match="shapes"):
        ols.mask_agreement(np.zeros((3, 3)), np.zeros((4, 4)))


def _write_well(tmp_path, stack, channels, sites=("1", "2")):
    """A PERISCOPE-layout well folder plus reference masks."""
    import tifffile

    well = tmp_path / "well"
    for c, name in enumerate(channels):
        (well / name).mkdir(parents=True)
        for site in sites:
            tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff", stack[..., c])
    ref = tmp_path / "ref" / "CP-Well1-1" / "segmentation_masks"
    ref.mkdir(parents=True)
    cells_ref = ols.relabel_raster(_label(stack[..., 1] > 500))
    tifffile.imwrite(ref / "Plate_CP_Well_Well1_Site_1_Cells_Objects.tiff", cells_ref)
    tifffile.imwrite(ref / "Plate_CP_Well_Well1_Site_1_Nuclei_Objects.tiff",
                     ols.relabel_raster(_label(stack[..., 0] > 500)))
    return well, tmp_path / "ref"


def test_the_settings_run_writes_labels_tables_report_and_reruns_identically(tmp_path):
    stack, centres = _site()
    channels = ols.LARGE_SITE_DEFAULTS["seg_site_channels"]
    well, ref = _write_well(tmp_path, stack, channels)
    assert ols.list_sites(str(well), channels) == ["1", "2"]
    base = {"seg_preset": "cell_painting_phalloidin", "seg_sites_src": str(well),
            "seg_tile_px": 256, "seg_tile_overlap_px": 64, "seg_report_reference": str(ref),
            "seg_measure_band_px": 200, "seg_diameter_min_objects": 3}
    first = ols.run_large_site_segmentation(dict(base, seg_dst=str(tmp_path / "a")),
                                            models=_models(),
                                            available_bytes=2 ** 40)
    second = ols.run_large_site_segmentation(dict(base, seg_dst=str(tmp_path / "b")),
                                             models=_models(),
                                             available_bytes=2 ** 40)
    sites = pd.read_csv(tmp_path / "a" / "sites.csv")
    assert list(sites["site"].astype(str)) == ["1", "2"]
    assert (sites["cells"] == len(centres) - 1).all()
    assert (sites["excluded_edge"] == 1).all()
    assert sites.loc[0, "cells_f1"] == 1.0 and sites.loc[0, "nuclei_f1"] == 1.0
    assert np.isnan(sites.loc[1, "cells_f1"])
    assert sites["peak_memory_gb"].gt(0).all() and sites["within_cap"].all()
    assert 10.0 <= sites.loc[0, "nucleus_diameter_px"] <= 80.0
    well_row = pd.read_csv(tmp_path / "a" / "agreement_well.csv")
    assert well_row.loc[0, "sites_with_reference"] == 1
    for name in ("1_cells.npy", "1_nuclei.npy", "2_cells.npy"):
        assert np.array_equal(np.load(tmp_path / "a" / "labels" / name),
                              np.load(tmp_path / "b" / "labels" / name))
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / "tables" / "1_cells.parquet"),
                                  pd.read_parquet(tmp_path / "b" / "tables" / "1_cells.parquet"))
    saved = json.loads((tmp_path / "a" / "settings.json").read_text())
    assert saved["seg_preset"] == "cell_painting_phalloidin"
    assert first["plan"]["tile_px"] == 256 and len(second["sites"]) == 2
    assert os.path.isfile(tmp_path / "a" / "resource_log" / "site_1.jsonl")


def test_the_run_reads_stacked_npy_sites_and_needs_its_folders(tmp_path):
    stack, centres = _site((300, 300))
    folder = tmp_path / "npy"
    folder.mkdir()
    np.save(folder / "7.npy", stack)
    out = ols.run_large_site_segmentation(
        {"seg_sites_src": str(folder), "seg_dst": str(tmp_path / "o"), "nucleus_diameter": 20,
         "seg_preset": "cell_painting_phalloidin", "seg_tile_px": 0},
        models=_models(), available_bytes=2 ** 40)
    assert out["sites"][0]["site"] == "7" and out["sites"][0]["nucleus_diameter_px"] == 20
    assert not os.path.exists(tmp_path / "o" / "agreement_well.csv")
    with pytest.raises(ValueError, match="required"):
        ols.run_large_site_segmentation({"seg_sites_src": str(folder)})
    with pytest.raises(FileNotFoundError, match="no CorrDNA"):
        ols.read_cropped_site(str(tmp_path), "3", ["CorrDNA"])


def test_preset_models_resolve_legacy_names(monkeypatch):
    import spacr.ops_engine as engine
    import spacr.utils as utils

    seen = []
    monkeypatch.setattr(utils, "_resolve_cellpose_pretrained",
                        lambda name, kind, setting=None: f"{name}->{kind}")
    monkeypatch.setattr(engine, "_cellpose_model",
                        lambda settings, gpu, key: seen.append((key, settings[key], gpu)) or key)
    out = ols.preset_models({"seg_nucleus_model": "nuclei", "cell_model": "cyto3"})
    assert out == ("seg_nucleus_model", "cell_model")
    assert seen == [("seg_nucleus_model", "nuclei->nucleus", False), ("cell_model", "cyto3->cell", False)]


def test_the_tiling_control_finds_tiled_and_untiled_runs_identical():
    rng = np.random.default_rng(4)
    centres = [tuple(c) for c in rng.integers(15, 985, size=(80, 2))]
    image = _discs((1000, 1000), centres, 10)
    out = ols.tiling_control(image, _label, tile_px=560, overlap_px=120)
    assert out["objects"] == out["away_from_seams"] + out["at_seams"]
    assert out["away_identical"] == out["away_from_seams"] > 0
    assert out["at_seams"] > 0 and out["at_seams_share"] == 1.0
    small = ols.tiling_control(image[:100, :100], _label, tile_px=2048, overlap_px=64)
    assert small["at_seams"] == 0 and np.isnan(small["at_seams_share"])


def _npy_well(tmp_path, stacks):
    """A folder of ``<site>.npy`` stacks."""
    folder = tmp_path / "npy"
    folder.mkdir()
    for site, stack in stacks.items():
        np.save(folder / f"{site}.npy", stack)
    return folder


def test_the_diameter_and_the_control_come_from_populated_sites_not_the_empty_first(tmp_path):
    empty, _ = _site(centres=[], seed=1)
    few, _ = _site(centres=[(300, 300)], seed=2)
    full, _ = _site(seed=3)
    crowded, _ = _site(centres=[(100, 100), (100, 400), (300, 250), (500, 120), (480, 480),
                                (300, 520), (520, 300)], seed=4)
    folder = _npy_well(tmp_path, {"1": empty, "2": few, "3": full, "4": crowded, "5": full})
    out = ols.run_large_site_segmentation(
        {"seg_sites_src": str(folder), "seg_dst": str(tmp_path / "o"),
         "seg_preset": "cell_painting_phalloidin", "seg_tile_px": 256, "seg_tile_overlap_px": 64,
         "seg_tiling_control_px": 400, "seg_diameter_sites": 5, "seg_diameter_min_objects": 3},
        models=_models(), available_bytes=2 ** 40)
    meta = json.loads((tmp_path / "o" / "nucleus_diameter.json").read_text())
    assert meta["sites_sampled"] == ["1", "2", "3", "4", "5"]
    assert meta["objects_per_site"] == {"1": 0, "2": 1, "3": 6, "4": 7, "5": 6}
    assert meta["sites_used"] == ["3", "4", "5"] and meta["diameter_per_site"]["1"] is None
    assert meta["nucleus_diameter_source"] == "estimated"
    assert 19 < meta["nucleus_diameter_px"] < 22
    assert meta["control_site"] == "5" and out["nucleus_diameter"] == meta
    sites = pd.read_csv(tmp_path / "o" / "sites.csv")
    assert (sites["nucleus_diameter_px"] == meta["nucleus_diameter_px"]).all()
    assert (sites["nucleus_diameter_sites"] == "3;4;5").all()
    control = json.loads((tmp_path / "o" / "tiling_control.json").read_text())
    assert control["site"] == "5" and control["site_nuclei"] == 6
    assert control["objects"] > 0 and control["away_identical"] == control["away_from_seams"]


def test_a_well_without_enough_nuclei_fails_instead_of_using_the_floor(tmp_path):
    empty, _ = _site(centres=[], seed=1)
    few, _ = _site(centres=[(300, 300)], seed=2)
    folder = _npy_well(tmp_path, {"1": empty, "2": few})
    with pytest.raises(ValueError, match="no sampled site has at least 3 nuclei.*site 2: 1"):
        ols.run_large_site_segmentation(
            {"seg_sites_src": str(folder), "seg_dst": str(tmp_path / "o"),
             "seg_preset": "cell_painting_phalloidin", "seg_tile_px": 0,
             "seg_diameter_min_objects": 3}, models=_models(), available_bytes=2 ** 40)


def test_a_fixed_diameter_is_recorded_and_the_control_site_still_surveyed(tmp_path):
    empty, _ = _site(centres=[], seed=1)
    full, _ = _site(seed=3)
    folder = _npy_well(tmp_path, {"1": empty, "2": full})
    ols.run_large_site_segmentation(
        {"seg_sites_src": str(folder), "seg_dst": str(tmp_path / "o"), "nucleus_diameter": 20,
         "seg_preset": "cell_painting_phalloidin", "seg_tile_px": 256, "seg_tile_overlap_px": 64,
         "seg_tiling_control_px": 400, "seg_diameter_min_objects": 3},
        models=_models(), available_bytes=2 ** 40)
    meta = json.loads((tmp_path / "o" / "nucleus_diameter.json").read_text())
    assert meta["nucleus_diameter_source"] == "fixed" and meta["nucleus_diameter_px"] == 20
    assert meta["control_site"] == "2"
    control = json.loads((tmp_path / "o" / "tiling_control.json").read_text())
    assert control["site"] == "2" and control["region_px"] == 400
    assert control["away_identical"] == control["away_from_seams"]
    sites = pd.read_csv(tmp_path / "o" / "sites.csv")
    assert sites["nucleus_diameter_sites"].isna().all()
    no_preset = ols.run_large_site_segmentation(
        {"seg_sites_src": str(folder), "seg_dst": str(tmp_path / "p"), "seg_tile_px": 0},
        models=_models(), available_bytes=2 ** 40)
    assert no_preset["nucleus_diameter"] == {"nucleus_diameter_px": None,
                                             "nucleus_diameter_source": "none"}


def _blobs(n):
    """``n`` 10 x 10 squares in a row."""
    out = np.zeros((40, 400), np.int32)
    for k in range(n):
        out[10:20, 12 * k:12 * k + 10] = k + 1
    return out


def test_survey_picks_the_median_count_site_and_sample_sites_spreads_evenly():
    nuclei = {"a": _blobs(0), "b": _blobs(5), "c": _blobs(9), "d": _blobs(7), "e": _blobs(4)}
    survey = ols.survey_sites(list(nuclei), nuclei, min_objects=5)
    assert survey["sites_used"] == ["b", "c", "d"] and survey["control_site"] == "d"
    assert 11 < survey["diameter_px"] < 12
    assert ols.survey_sites(["b", "c"], nuclei, min_objects=5)["control_site"] == "b"
    assert ols.sample_sites([str(i) for i in range(1, 101)], 5) == ["1", "26", "51", "75", "100"]
    assert ols.sample_sites(["1", "2"], 9) == ["1", "2"]
    assert ols.sample_sites(["7"], 0) == ["7"] and ols.sample_sites([], 3) == []


def test_site_normalisation_matches_cellpose_on_one_tile_and_zeroes_flat_channels():
    from cellpose.transforms import normalize_img

    rng = np.random.default_rng(5)
    image = rng.gamma(2.0, 300.0, size=(300, 320, 2)).astype(np.float32)
    image[..., 1] = 7.0
    out = ols.normalize_site(image)
    assert out.dtype == np.float32 and not np.shares_memory(out, image)
    assert np.array_equal(out, normalize_img(image.copy(), axis=-1))
    assert not out[..., 1].any()
    big = rng.gamma(2.0, 300.0, size=(2400, 2400)).astype(np.float32)
    assert np.array_equal(ols.normalize_site(big),
                          normalize_img(big[..., None].copy(), axis=-1)[..., 0])


def _uneven(shape=(1500, 1500), seed=7):
    """Soft-edged blobs of mixed brightness under a strong illumination gradient."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    image = 200.0 + 1800.0 * (xx / shape[1]) * (0.3 + yy / shape[0])
    for cy, cx in rng.integers(20, shape[0] - 20, size=(400, 2)):
        y0, x0 = max(cy - 30, 0), max(cx - 30, 0)
        patch = (slice(y0, cy + 31), slice(x0, cx + 31))
        image[patch] += rng.uniform(800, 4000) * np.exp(
            -((yy[patch] - cy) ** 2 + (xx[patch] - cx) ** 2) / (2 * 7.0 ** 2))
    return image.astype(np.float32)


def _per_tile(x):
    """The stand-in as Cellpose runs it by default: normalised per call."""
    return _FakeModel(0.6).eval(x)[0]


def _site_scaled(x):
    """The stand-in on an image normalised once for the whole site."""
    return _FakeModel(0.6).eval(x, normalize=False)[0]


def test_tiled_equals_untiled_away_from_seams_on_uneven_intensity():
    image = _uneven()
    tile = (image.shape[0] + 256) // 2 + 1
    before = ols.tiling_control(image, _per_tile, tile_px=tile, overlap_px=256)
    assert before["away_identical"] < before["away_from_seams"]
    after = ols.tiling_control(ols.normalize_site(image), _site_scaled, tile_px=tile,
                               overlap_px=256)
    assert after["objects"] > 200 and after["at_seams"] > 20
    assert after["away_identical"] == after["away_from_seams"]
    assert after["at_seams_share"] >= 0.99


def test_network_tile_starts_follow_cellpose_and_aligned_tiles_share_its_grid():
    from cellpose import transforms

    for length in (200, 256, 300, 816, 1200, 1999, 2048, 5500):
        y1, y2, _, _ = transforms.get_pad_yx(length, length, min_size=(256, 256))
        image = np.zeros((1, length + y1 + y2, 300), np.float32)
        _, ysub, _, _, _ = transforms.make_tiles(image, bsize=256, tile_overlap=0.1)
        starts = sorted({int(a) - y1 for a, _ in ysub})
        assert list(ols.network_tile_starts(length)) == starts
    tile, step, stride = ols.aligned_tiling(2048, 256)
    assert (tile, step, stride) == (1968, 1536, 192)
    layout = ols.tiling_layout((5500, 5500), 2048, 256, align=True)
    assert layout["aligned"] and layout["padded"] == (5616, 5616)
    assert layout["tile_px"] == 1968 and layout["overlap_px"] == 432
    grid = set(range(-8, 5616, stride))
    for y0, y1, _, _ in layout["boxes"]:
        assert y0 % stride == 0 and set(int(v) + y0 for v in ols.network_tile_starts(y1 - y0)) <= grid
    assert ols.tiling_layout((300, 300), 2048, 256, align=True)["boxes"] == [(0, 300, 0, 300)]
    assert ols.aligned_tiling(200, 64) is None
    assert ols.control_layout(2048, 256) == (1968, 1200, True)
    assert ols.control_layout(400, 64) == (400, 233, False)


class _TileMeanNet:
    """A stand-in network whose output depends on where its tiles fall:
    each network tile is centred on its own mean, as a real network's
    output depends on the whole tile it sees."""

    def __init__(self):
        import torch

        self.device, self.dtype = torch.device("cpu"), torch.float32

    def eval(self):
        """Cellpose puts the network in eval mode first."""
        return self

    def __call__(self, x):
        import torch

        y = x[:, :1] - x[:, :1].mean(dim=(2, 3), keepdim=True)
        return torch.cat([y, y, y], 1), torch.zeros(x.shape[0], 256)


def _network_like(x):
    """Cellpose's own network tiling and tile averaging around the stand-in."""
    from cellpose.core import run_net

    flows, _ = run_net(_TileMeanNet(), np.asarray(x, np.float32)[None, ..., None], bsize=256)
    return ndimage.label(flows[0, ..., 2] > 0.4)[0].astype(np.int32)


def test_tiles_on_the_network_grid_reproduce_the_untiled_run_away_from_seams():
    image = ols.normalize_site(_uneven((1200, 1200), seed=11))
    misplaced = ols.tiling_control(image, _network_like, tile_px=729, overlap_px=256)
    assert not misplaced["aligned"]
    assert misplaced["away_identical"] < misplaced["away_from_seams"]
    region, tile, aligned = ols.control_layout(1200, 256)
    assert (region, tile, aligned) == (1200, 816, True)
    out = ols.tiling_control(image, _network_like, tile_px=tile, overlap_px=256, align=True)
    assert out["aligned"] and out["tile_px"] == 816 and out["overlap_px"] == 432
    assert out["objects"] > 100 and out["at_seams"] > 20
    assert out["away_identical"] == out["away_from_seams"]
    assert out["at_seams_share"] >= 0.99
