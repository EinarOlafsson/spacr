"""Item 1025 (alpha, schema 969): spaCR makes its own SBS cells.

The decode can grow one SBS cell from every SBS nucleus, by Cellpose's cell
probability or a seeded watershed on a cytoplasm proxy of the base channels,
in the seeded watershed of the OPS phenotype path
(``spacr.ops_large_sites.cells_from_nucleus_seeds``), pairs nuclei with
cells in ``spacr.measure._assign_nuclei_to_cells``, the pairing of Measure
and OPS phenotype, and stores the masks in ``sbs/fov.ome.zarr``. The reads
still go to the nucleus discs. Off by default, and a run with it off
decodes exactly as before.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("tifffile")
pytest.importorskip("pyarrow")

from scipy import ndimage

from spacr import ops_decode, ops_engine, ops_sbs, ops_settings
from tests.ops_store_sql import ops_connect


def _field(seed=3):
    """A synthetic aligned SBS field: three nuclei, each in a hazy cell.

    :returns: ``(stack, centroids, areas)``; the stack is
        ``(cycles, channels, H, W)``.
    """
    rng = np.random.default_rng(seed)
    centroids = np.array([[30.0, 30.0], [30.0, 62.0], [80.0, 80.0]])
    areas = np.full(3, np.pi * 6 ** 2)
    yy, xx = np.mgrid[:120, :120]
    haze = np.zeros((120, 120))
    for cy, cx in centroids:
        haze[np.hypot(yy - cy, xx - cx) <= 15] = 400.0
    stack = np.empty((3, 4, 120, 120), np.float32)
    for c in range(3):
        for k in range(4):
            stack[c, k] = 100 + haze + rng.normal(0, 5, haze.shape)
    return stack, centroids, areas


class _ProbabilityModel:
    """A stand-in for Cellpose: the cell probability is the first channel, centred."""

    def __init__(self):
        self.calls = []

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Flows whose cell probability is the proxy channel minus its midpoint."""
        self.calls.append({"compute_masks": compute_masks, "channel_axis": channel_axis,
                           "shape": np.shape(image)})
        image = np.asarray(image, np.float32)
        first = image[..., 0] if channel_axis == -1 else image
        cellprob = first - 0.5 * (first.max() + first.min())
        flows = [np.zeros(first.shape + (3,), np.uint8),
                 np.zeros((2,) + first.shape, np.float32), cellprob]
        masks = ndimage.label(cellprob > 0)[0]
        return masks, flows, None


def _options(method, **extra):
    return ops_engine._sbs_cell_options({"ops_sbs_cell_method": method, **extra})


def test_off_is_the_default():
    assert ops_settings.OPS_DEFAULTS["ops_sbs_cell_method"] == "off"
    assert ops_engine._sbs_cell_options({}) is None
    with pytest.raises(ValueError, match="ops_sbs_cell_method"):
        _options("magic")
    with pytest.raises(ValueError, match="ops_sbs_cell_threshold"):
        _options("watershed", ops_sbs_cell_threshold="high")
    assert _options("watershed", ops_sbs_cell_threshold=50)["foreground"] == 50.0


def test_schema_row_and_alpha_registration():
    from spacr import settings

    assert settings.SETTINGS_SCHEMA_VERSION >= 969
    assert settings.DEFAULT_CHANGES[969] == {"ops_sbs_cell_method": "off"}
    assert set(settings.ALPHA_FEATURES[1025]["settings"]) == {
        "ops_sbs_cell_method", "ops_sbs_cell_model", "ops_sbs_cell_threshold"}
    assert len(ops_settings.OPS_CATEGORIES["SBS cells α"]) <= 20


def test_the_seeds_are_the_nucleus_discs():
    _stack, centroids, areas = _field()
    seeds = ops_engine._sbs_seed_labels(centroids, areas, (120, 120))
    assert sorted(np.unique(seeds).tolist()) == [0, 1, 2, 3]
    for index, (cy, cx) in enumerate(centroids, start=1):
        assert seeds[int(cy), int(cx)] == index
        assert abs((seeds == index).sum() - areas[0]) < 0.15 * areas[0]
    outside = ops_engine._sbs_seed_labels([[-50.0, -50.0]], [100.0], (20, 20))
    assert not outside.any()


@pytest.mark.parametrize("method", ["watershed", "cellpose"])
def test_both_methods_grow_one_cell_per_nucleus(monkeypatch, method):
    stack, centroids, areas = _field()
    model = _ProbabilityModel()
    seeds = ops_engine._sbs_seed_labels(centroids, areas, (120, 120))
    cells = ops_engine._sbs_segment_cells(stack, seeds, _options(method), model=model)
    for label in (1, 2, 3):
        mine = cells == label
        assert (mine & (seeds == label)).sum() == (seeds == label).sum()
        assert mine.sum() > 2 * (seeds == label).sum()
    assert cells[0, 119] == 0 and cells[119, 0] == 0
    if method == "cellpose":
        assert model.calls and all(call["compute_masks"] is False
                                   and call["channel_axis"] == -1 for call in model.calls)
    else:
        assert not model.calls


def test_each_nucleus_gets_one_cell_or_none():
    seeds = np.zeros((40, 60), np.int32)
    seeds[5:10, 5:10] = 1
    seeds[5:10, 15:18] = 2
    seeds[30:35, 50:55] = 3
    cells = np.zeros_like(seeds)
    cells[2:20, 2:25] = 7
    paired, has_cell, area = ops_engine._sbs_pair_cells(seeds, cells, 4)
    assert has_cell.tolist() == [True, False, False, False]
    assert set(np.unique(paired)) == {0, 1}
    assert area.tolist() == [18 * 23, 0, 0, 0]


def test_the_shared_functions_serve_phenotype_and_sbs(monkeypatch):
    """Phenotype pairing and SBS pairing reach the same pairing; SBS and the
    large-site phenotype path reach the same seeded watershed."""
    from spacr import measure, ops_large_sites, ops_sample

    calls = {"pair": 0, "seed": 0, "flows": 0}
    real_pair, real_seed = measure._assign_nuclei_to_cells, ops_large_sites.cells_from_nucleus_seeds
    real_flows = ops_large_sites._eval_flows

    def pair(*args, **kwargs):
        calls["pair"] += 1
        return real_pair(*args, **kwargs)

    def seed(*args, **kwargs):
        calls["seed"] += 1
        return real_seed(*args, **kwargs)

    def flows(*args, **kwargs):
        calls["flows"] += 1
        return real_flows(*args, **kwargs)

    monkeypatch.setattr(measure, "_assign_nuclei_to_cells", pair)
    monkeypatch.setattr(ops_large_sites, "cells_from_nucleus_seeds", seed)
    monkeypatch.setattr(ops_large_sites, "_eval_flows", flows)
    stack, centroids, areas = _field()
    seeds = ops_engine._sbs_seed_labels(centroids, areas, (120, 120))
    ops_engine._sbs_pair_cells(seeds, ops_engine._sbs_segment_cells(
        stack, seeds, _options("cellpose"), model=_ProbabilityModel()), 3)
    assert calls == {"pair": 1, "seed": 1, "flows": 1}
    ops_sample.pair_nuclei_with_cells(seeds, seeds)
    assert calls["pair"] == 2
    cellprob = np.ones(seeds.shape, np.float32)
    ops_large_sites.cells_from_nucleus_seeds(seeds, cellprob)
    assert calls["seed"] == 2


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The end-to-end synthetic well: objects once, then decodes under several settings."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setitem(ops_settings.OPS_DEFAULTS, "ops_raster_overlap", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    model = _ProbabilityModel()

    def fake_model(settings, gpu, key="ops_sbs_nucleus_model"):
        return model if key == "ops_sbs_cell_model" else base._ThresholdModel()

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake_model)
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    root = tmp_path_factory.mktemp("sbs_cells_1025")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    settings = {"genotype_source": str(raw), "dst_root": str(root / "out"),
                "plate": "synthetic", "ops_gpu": False, "n_workers": 1,
                "ops_decode_threshold_mode": "manual", "ops_decode_preset": "none"}
    ops_engine.run_ops(settings, library=library)
    captured = []
    real = ops_engine._decode_field

    def capture(task):
        out = real(task)
        captured.append((task, out))
        return out

    monkeypatch.setattr(ops_engine, "_decode_field", capture)
    runs = {}

    def run(name, **extra):
        captured.clear()
        result = ops_engine.run_ops({**settings, **extra}, phases=("decode",), library=library)
        with ops_connect(result["db"]) as conn:
            rows = conn.execute("SELECT * FROM ops_barcodes ORDER BY object_id").fetchall()
        runs[name] = {"result": result, "rows": rows, "fields": list(captured)}

    run("default")
    run("explicit_off", ops_sbs_cell_method="off")
    rule = {"ops_decode_preset": "rule_a_s2", "ops_decode_fdr_calibration": "none"}
    run("rule_default", **rule)
    run("rule_reach_watershed", **rule, ops_sbs_cell_method="watershed")
    run("rule_reach_cellpose", **rule, ops_sbs_cell_method="cellpose")
    run("legacy_watershed", ops_sbs_cell_method="watershed")
    yield runs, model, library
    monkeypatch.undo()


def test_default_off_is_identical(synthetic):
    runs, _model, _library = synthetic
    assert runs["default"]["rows"] and runs["default"]["rows"] == runs["explicit_off"]["rows"]
    for name in ("default", "explicit_off", "rule_default"):
        decode = runs[name]["result"]["wells"]["A1"]["decode"]
        assert "sbs_cells" not in decode
        for task, out in runs[name]["fields"]:
            assert not {"sbs_cells", "sbs_dapi", "cell_labels"} & set(task)
            assert "sbs_cells" not in out


@pytest.mark.parametrize("name, reference", [("rule_reach_watershed", "rule_default"),
                                             ("rule_reach_cellpose", "rule_default"),
                                             ("legacy_watershed", "default")])
def test_segmentation_changes_no_call(synthetic, name, reference):
    runs, model, _library = synthetic
    assert runs[reference]["rows"]
    assert runs[name]["rows"] == runs[reference]["rows"]
    report = runs[name]["result"]["wells"]["A1"]["decode"]["sbs_cells"]
    assert report["method"] == name.split("_")[-1]
    assert report["nuclei_with_cell"] == report["nuclei"] > 0
    if name == "rule_reach_cellpose":
        assert model.calls


def test_the_cells_are_stored_in_fov_ome_zarr(synthetic):
    import pandas as pd

    from spacr.image_store import field_key, read_labels

    runs, _model, _library = synthetic
    result = runs["rule_reach_watershed"]["result"]
    report = result["wells"]["A1"]["decode"]["sbs_cells"]
    assert report["store"].endswith("fov.ome.zarr")
    table = pd.read_parquet(report["table"])
    assert list(table.columns) == ["plate", "well", "site", "label", "has_cell",
                                   "cell_area_px", "owned"]
    assert not table.duplicated(["well", "site", "label"]).any()
    with ops_connect(result["db"]) as conn:
        ids = {row[0] for row in conn.execute("SELECT object_id FROM ops_objects")}
    for site in sorted(set(table["site"])):
        cells = np.asarray(read_labels(report["store"], "sbs_cell", key=field_key("A1", site)))
        nuclei = np.asarray(read_labels(report["store"], "sbs_nucleus",
                                        key=field_key("A1", site)))
        assert cells.shape == nuclei.shape
        labels = set(np.unique(cells[cells > 0]).tolist())
        assert labels <= ids
        rows = table[(table["site"] == site) & table["has_cell"]]
        assert labels == set(rows["label"].tolist())
        for label, area in zip(rows["label"], rows["cell_area_px"]):
            assert (cells == label).sum() == area
