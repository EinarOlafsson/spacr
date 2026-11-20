"""Item 970: pathogen and organelle objects in the OPS phenotype path.

Fields with planted pathogens and organelles of known number, size and
intensity per cell go through the shipped phenotype phases. Cellpose is
replaced by a threshold model, as in ``test_ops_engine_phenotype_coverage``;
the organelles take Mask Generation's own classical route ('spots', 'otsu')
unchanged. The default run is checked to write exactly what it wrote before.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("skimage")

from scipy import ndimage

from spacr import ops_engine
from spacr.ops_sample import _cell_object_features
from spacr.ops_store import write_table
from tests.test_ops_large_sites_921 import (  # noqa: F401
    _discs, _models, _site, _stand_in_masks)

PLATE = "P1"
SIZE = 96
CENTRES = ((20, 20), (20, 70), (70, 20), (70, 70))
PATHOGEN_OFFSETS = ((0, 6), (6, 0), (-6, 0))
ORGANELLE_OFFSETS = ((4, -5), (-5, 4), (-5, -5))
PATHOGENS_PER_CELL = (0, 1, 2, 3)
ORGANELLES_PER_CELL = (2, 0, 1, 3)
STRAY_PATHOGEN = (45, 45)
DISC_AREA = 13
CHANNELS = "DAPI,AF750,PATH,MITO"


class _ThresholdModel:
    """A stand-in for Cellpose: objects are the bright regions of the last channel."""

    def __init__(self, grow=0):
        self.grow = grow
        self.calls = []

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        """Label the last channel above half its maximum, grown by ``grow``."""
        self.calls.append({"shape": np.shape(image), "diameter": diameter})
        plane = np.asarray(image, float)
        if plane.ndim == 3:
            plane = plane[..., -1]
        labels, _n = ndimage.label(plane > plane.max() / 2.0)
        if self.grow:
            from skimage.segmentation import expand_labels

            labels = expand_labels(labels, self.grow)
        return labels, None, None


def _disc(centre, radius):
    """A boolean disc on the field."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return np.hypot(yy - centre[0], xx - centre[1]) <= radius


def _field(seed=0):
    """DAPI, the cell stain, planted pathogens and planted organelle spots."""
    rng = np.random.default_rng(seed)
    dapi = rng.normal(10, 1, (SIZE, SIZE))
    body = rng.normal(10, 1, (SIZE, SIZE))
    pathogen = np.zeros((SIZE, SIZE))
    mito = rng.normal(10, 1, (SIZE, SIZE))
    for index, (cy, cx) in enumerate(CENTRES):
        dapi[_disc((cy, cx), 4)] += 200
        body[_disc((cy, cx), 9)] += 120
        for dy, dx in PATHOGEN_OFFSETS[:PATHOGENS_PER_CELL[index]]:
            pathogen[_disc((cy + dy, cx + dx), 2)] = 200
        for dy, dx in ORGANELLE_OFFSETS[:ORGANELLES_PER_CELL[index]]:
            mito[_disc((cy + dy, cx + dx), 2)] += 300
    pathogen[_disc(STRAY_PATHOGEN, 2)] = 200
    return {"DAPI": dapi, "AF750": body, "PATH": pathogen, "MITO": mito}


def _write_fields(folder, well="A1", sites=(0,)):
    """Phenotype tiles, one file per channel per site, named as acquired."""
    folder.mkdir(parents=True, exist_ok=True)
    for site in sites:
        for name, plane in _field(site).items():
            tifffile.imwrite(
                folder / f"20X_DAPI-AF750_{well}_{name}_Site-{site}.tif",
                np.clip(plane, 0, 65535).astype(np.uint16))
    return folder


def _placed(db, well="A1", sites=(0,)):
    """``ops_phenotype`` rows for the sites."""
    write_table(db, "ops_phenotype", pd.DataFrame([{
        "plate": PLATE, "well": well, "site": site,
        "centre_y": 1000.0 + 100 * site, "centre_x": 1000.0,
        "sbs_site": 0, "is_anchor": int(site == 0)} for site in sites]))


@pytest.fixture
def models(monkeypatch):
    """The threshold model in place of Cellpose, one per model setting."""
    built = {}

    def fake(settings, gpu, key="cellpose_model"):
        built[key] = _ThresholdModel(grow=5 if key == "ops_ph_cell_model"
                                     else 0)
        return built[key]

    monkeypatch.setattr(ops_engine, "_cellpose_model", fake)
    return built


SETTINGS = {"ops_ph_channels": CHANNELS, "ops_ph_downsample": 1,
            "ops_ph_texture": False, "n_workers": 1,
            "ops_ph_cell_model": "cyto-like"}
ON = {**SETTINGS, "ops_ph_pathogen_channel": "PATH",
      "ops_ph_organelle_channel": "MITO"}


def _run(tmp_path, settings, sites=(0,)):
    """Segment and measure the planted well; the report and the cell table."""
    tiles = _write_fields(tmp_path / "pheno", sites=sites)
    db = str(tmp_path / "measurements.db")
    _placed(db, sites=sites)
    (tmp_path / "A1").mkdir(exist_ok=True)
    (tmp_path / "A1" / "ops_report.json").write_text(json.dumps(
        {"phenotype": {"expected_scale": 0.25}}))
    index = ops_engine._index_tiles(str(tiles), "uncycled")
    report = ops_engine._ph_fields(db, PLATE, "A1", index["A1"], settings,
                                   False, str(tmp_path), segment=True,
                                   measure=True)
    table = pd.read_parquet(tmp_path / "ops_phenotype_cells_A1.parquet")
    return report, table, index


def test_the_object_classes_follow_their_channel_settings():
    assert ops_engine._ph_object_roles({}) == []
    assert ops_engine._ph_object_roles(
        {"ops_ph_pathogen_channel": " ", "ops_ph_organelle_channel": None}) == []
    assert ops_engine._ph_object_roles(ON) == ["pathogen", "organelle"]
    assert ops_engine._ph_object_roles(
        {"ops_ph_organelle_channel": "MITO"}) == ["organelle"]


def test_planted_objects_are_segmented_and_summarised_per_cell(tmp_path,
                                                                models):
    report, table, _index = _run(tmp_path, ON)
    objects = report["ph_segment"]["objects"]
    assert objects["pathogen"] == sum(PATHOGENS_PER_CELL) + 1
    assert objects["organelle"] == sum(ORGANELLES_PER_CELL)
    with np.load(tmp_path / "A1" / "phenotype_masks" / "site-0.npz") as saved:
        assert set(saved.files) == {"nuclei", "cells", "pathogen", "organelle"}
        cells = saved["cells"].astype(np.int32)
    table = table.set_index("label")
    for index, centre in enumerate(CENTRES):
        label = int(cells[centre])
        row = table.loc[label]
        count = PATHOGENS_PER_CELL[index]
        assert row["cell_pathogen_count"] == count
        assert row["cell_pathogen_total_area"] == count * DISC_AREA
        assert row["cell_pathogen_fraction"] == pytest.approx(
            count * DISC_AREA / float((cells == label).sum()))
        assert row["cell_pathogen_path_mean_intensity"] == (200.0 if count
                                                            else 0.0)
        assert row["cell_pathogen_path_std_intensity"] == 0.0
        assert row["cell_organelle_count"] == ORGANELLES_PER_CELL[index]
    assert {"cell_pathogen_dapi_mean_intensity",
            "cell_organelle_mito_mean_intensity"} <= set(table.columns)
    assert report["ph_measure"]["features"] > len(
        ops_engine._ph_feature_columns(
            [c for c in table.columns if "_pathogen_" not in c
             and "_organelle_" not in c]))
    marker = json.loads((tmp_path / "A1" / "phenotype_cells"
                         / "part_options.json").read_text())
    assert marker["objects"]["pathogen"]["channel"] == "PATH"
    assert marker["objects"]["organelle"]["method"] == "otsu"


def test_the_default_run_writes_what_it_wrote_before(tmp_path, models):
    """The same mask classes are saved with a shared tile/precision resume marker."""
    report, table, _index = _run(tmp_path, SETTINGS)
    assert not {"pathogen", "organelle"} & set(report["ph_segment"]["objects"])
    assert "objects" not in report["ph_segment"]["seconds"]
    with np.load(tmp_path / "A1" / "phenotype_masks" / "site-0.npz") as saved:
        assert set(saved.files) == {"nuclei", "cells"}
    mask_options = json.loads((tmp_path / "A1" / "phenotype_masks"
                               / "part_options.json").read_text())
    assert mask_options == {"mask_eval_batch_size": 0, "ops_ph_tile_batch": 0,
                            "mask_inference_precision": "auto"}
    marker = json.loads((tmp_path / "A1" / "phenotype_cells"
                         / "part_options.json").read_text())
    assert "objects" not in marker
    assert not [c for c in table.columns
                if "_pathogen_" in c or "_organelle_" in c]


def test_objects_are_not_added_to_masks_made_without_them(tmp_path, models):
    _report, _table, index = _run(tmp_path, SETTINGS)
    with pytest.raises(ValueError, match="parts made with"):
        ops_engine._ph_fields(str(tmp_path / "measurements.db"), PLATE, "A1",
                              index["A1"], ON, False, str(tmp_path),
                              segment=True, measure=True)
    with pytest.raises(ValueError, match="no \\['pathogen', 'organelle'\\]"):
        ops_engine._ph_measure_table({
            "site": 0, "files": index["A1"][min(index["A1"])][0],
            "channels": tuple(CHANNELS.split(",")),
            "mask_path": str(tmp_path / "A1" / "phenotype_masks"
                             / "site-0.npz"),
            "texture": False, "objects": ["pathogen", "organelle"]})


def test_a_channel_outside_the_phenotype_channels_is_refused(models):
    stack = np.zeros((8, 8, 2), np.float32)
    with pytest.raises(ValueError, match="ops_ph_pathogen_channel"):
        ops_engine._ph_segment_objects(
            stack, ("DAPI", "AF750"), {"ops_ph_pathogen_channel": "PATH"},
            False, {})
    with pytest.raises(ValueError):
        ops_engine._ph_segment_objects(
            stack, ("DAPI", "AF750"),
            {"ops_ph_organelle_channel": "AF750",
             "ops_ph_organelle_morphology": "spots",
             "ops_ph_organelle_method": "ridge"}, False, {})


def test_a_minimum_area_drops_small_pathogens(models):
    field = _field()
    stack = np.stack([field[name] for name in CHANNELS.split(",")], axis=-1)
    small = np.zeros((SIZE, SIZE))
    small[_disc((20, 20), 2)] = 200
    small[_disc((70, 70), 5)] = 200
    stack[..., 2] = small
    cache = {"cpsam": _ThresholdModel()}
    kept = ops_engine._ph_segment_objects(
        stack, tuple(CHANNELS.split(",")),
        {"ops_ph_pathogen_channel": "PATH", "ops_ph_pathogen_min_area": 20},
        False, cache)["pathogen"]
    assert len(np.unique(kept[kept > 0])) == 1
    assert kept[70, 70] > 0 and kept[20, 20] == 0


def test_cells_without_objects_get_zeros_and_strays_belong_to_none():
    cells = np.zeros((20, 20), np.int32)
    cells[:10, :10] = 1
    cells[10:, 10:] = 2
    objects = np.zeros_like(cells)
    objects[2:4, 2:4] = 1
    objects[2:4, 12:14] = 2
    image = np.ones((20, 20, 1), np.float32) * 5
    frame = _cell_object_features(cells, {"pathogen": objects}, image,
                                  ["DAPI"]).set_index("label")
    assert frame.loc[1, "cell_pathogen_count"] == 1
    assert frame.loc[1, "cell_pathogen_total_area"] == 4
    assert frame.loc[1, "cell_pathogen_fraction"] == 4 / 100
    assert frame.loc[1, "cell_pathogen_dapi_mean_intensity"] == 5.0
    assert (frame.loc[2] == 0).all()
    none = _cell_object_features(cells, {"pathogen": np.zeros_like(cells)},
                                 image, ["DAPI"])
    assert list(none.columns) == list(frame.reset_index().columns)
    assert (none.drop(columns="label") == 0).all().all()


def test_the_summary_intensities_are_unchanged_by_the_bounding_boxes():
    from spacr.measure import _summarize_organelles_per_parent

    rng = np.random.default_rng(3)
    image = rng.normal(100, 20, (64, 64, 2)).astype(np.float32)
    cells = np.zeros((64, 64), np.int32)
    cells[:32] = 1
    cells[32:] = 2
    organelles, _n = ndimage.label(rng.random((64, 64)) > 0.93)
    frame = _summarize_organelles_per_parent(organelles, cells, image)
    for channel in range(2):
        expected = []
        for label in (1, 2):
            members = np.unique(organelles[(cells == label)
                                           & (organelles > 0)])
            means = []
            for member in members:
                owner = np.bincount(cells[organelles == member]).argmax()
                if owner == label:
                    means.append(image[..., channel][organelles == member]
                                 .mean())
            expected.append(np.mean(means))
        column = f"organelle_channel_{channel}_mean_intensity_per_cell"
        assert frame[column].tolist() == expected


def test_object_columns_reach_the_gene_profiles(tmp_path):
    rng = np.random.default_rng(0)
    n = 120
    db = str(tmp_path / "measurements.db")
    guides = ["AAAA", "CCCC", "GGGG", "TTTT"]
    genes = ["nontargeting", "nontargeting_b", "GENE1", "GENE1"]
    ids = np.arange(n) + 1
    guide_of = [guides[i % 4] for i in range(n)]
    write_table(db, "ops_merge", pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": 0, "label": ids,
        "object_id": ids, "distance": 0.5, "refined": 1}))
    write_table(db, "ops_barcodes", pd.DataFrame({
        "plate": PLATE, "well": "A1", "object_id": ids, "barcode": "x",
        "mapped_guide": guide_of}))
    pd.DataFrame({"sgRNA": [g + "TTTT" for g in guides],
                  "gene_symbol": genes}).to_csv(tmp_path / "design.csv",
                                                index=False)
    infected = np.array([4.0 if guide_of[i] in ("GGGG", "TTTT") else 1.0
                         for i in range(n)])
    pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": 0, "label": ids,
        "nucleus_area": rng.normal(100, 5, n),
        "cell_pathogen_count": rng.poisson(infected).astype(float),
        "cell_organelle_count": rng.poisson(3.0, n).astype(float)}).to_parquet(
        tmp_path / "ops_phenotype_cells_A1.parquet")
    report = ops_engine._ph_aggregate(
        db, PLATE, ["A1"], {"ops_design": str(tmp_path / "design.csv"),
                            "ops_ph_channels": "DAPI"}, str(tmp_path))
    profiles = pd.read_parquet(report["outputs"]["genes_parquet"]).set_index(
        "gene")
    assert {"cell_pathogen_count", "cell_organelle_count"} <= set(
        profiles.columns)
    assert profiles.loc["GENE1", "cell_pathogen_count"] > 1.0


def _large_site_well(tmp_path):
    """One synthetic large site with two organelle spots in each cell's CorrMito."""
    stack, centres = _site(centres=[(100, 100), (100, 400), (300, 250)])
    spots = [(cy + dy, cx + dx) for cy, cx in centres for dy, dx in ((0, 15), (15, 0))]
    stack[..., 3] = _discs(stack.shape[:2], spots, 3).astype(np.uint16) * 800
    folder = tmp_path / "npy"
    folder.mkdir()
    np.save(folder / "1.npy", stack)
    return folder, len(centres), len(spots)


def test_the_large_site_path_adds_organelles_only_when_asked(tmp_path,
                                                             _stand_in_masks):
    from spacr import ops_large_sites as ols

    folder, cells, spots = _large_site_well(tmp_path)
    base = {"seg_sites_src": str(folder), "nucleus_diameter": 20,
            "seg_preset": "cell_painting_phalloidin", "seg_tile_px": 0}
    ols.run_large_site_segmentation(dict(base, seg_dst=str(tmp_path / "off")),
                                    models=_models(), available_bytes=2 ** 40)
    ols.run_large_site_segmentation(
        dict(base, seg_dst=str(tmp_path / "on"),
             ops_ph_organelle_channel="CorrMito"),
        models=_models(), available_bytes=2 ** 40)
    off = pd.read_parquet(tmp_path / "off" / "tables" / "1_cells.parquet")
    on = pd.read_parquet(tmp_path / "on" / "tables" / "1_cells.parquet")
    assert not (tmp_path / "off" / "labels" / "1_organelle.npy").exists()
    assert "organelle_objects" not in pd.read_csv(tmp_path / "off" / "sites.csv")
    assert not [c for c in off.columns if "organelle" in c]
    pd.testing.assert_frame_equal(on[list(off.columns)], off)
    organelles = np.load(tmp_path / "on" / "labels" / "1_organelle.npy")
    assert len(np.unique(organelles[organelles > 0])) == spots
    assert pd.read_csv(tmp_path / "on" / "sites.csv").loc[0, "organelle_objects"] == spots
    assert len(on) == cells
    assert (on["cell_organelle_count"] == 2).all()
