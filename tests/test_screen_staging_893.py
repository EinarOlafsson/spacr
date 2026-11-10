"""Item 893: one settings file runs a screen plate, and its export and aggregate match the drivers'."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import screen_staging as ss


def _raw_plate(root: Path, wells=("B02", "C03"), sites=(1, 2), channels=(1, 2, 4)) -> Path:
    """BBBC021-named raw TIFFs: ``<plate>_<well>_s<site>_w<channel><GUID>.tif``."""
    import tifffile

    root.mkdir(parents=True)
    rng = np.random.default_rng(3)
    for well in wells:
        for site in sites:
            for channel in channels:
                image = rng.integers(100, 900, size=(64, 64)).astype(np.uint16)
                image[8:22, 8:22] += 4000 + 100 * site
                image[34:52, 30:50] += 3000 + 50 * channel
                tifffile.imwrite(root / f"Week1_{well}_s{site}_w{channel}ABCD-12.tif", image)
    return root


@pytest.fixture
def threshold_cellpose(monkeypatch):
    """Cellpose replaced by a model that labels the bright regions of what it is given."""
    import torch
    from skimage.measure import label

    import spacr.object as sobj
    import spacr.plot as splot

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, x, **kwargs):
            masks, flows = [], []
            for image in x:
                array = np.asarray(image, dtype=float)
                plane = array.max(axis=-1) if array.ndim == 3 else array
                masks.append(label(plane > plane.mean() + plane.std()).astype(np.uint16))
                flows.append(np.zeros(plane.shape, dtype=np.float32))
            return masks, flows, None

    cpu = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sobj.accelerator, "cellpose_kwargs", lambda: {"gpu": False, "device": cpu})
    monkeypatch.setattr(sobj, "cp_models", types.SimpleNamespace(CellposeModel=_Model))
    monkeypatch.setattr(splot, "plot_cellpose4_output", lambda *args, **kwargs: None)


def _mask_settings(src: Path, **over):
    settings = {
        "src": str(src), "metadata_type": "cellvoyager", "channels": [0, 1, 2],
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": None,
        "nucleus_diameter": 30, "cell_diameter": 30, "magnification": 20,
        "randomize": False, "plot": False, "verbose": False, "test_mode": False,
        "save": True, "n_jobs": 1, "batch_size": 50, "timelapse": False,
        "seg_qc": "off", "illumination_correction": False, "keep_intermediate": True,
    }
    settings.update(over)
    return settings


def _arrays(folder: Path):
    return {path.relative_to(folder).as_posix(): np.load(path)
            for path in sorted(folder.rglob("*.npy")) if not path.name.startswith(".")}


def test_one_mask_run_equals_the_two_pass_run(tmp_path, threshold_cellpose):
    from spacr.core import preprocess_generate_masks

    raw = _raw_plate(tmp_path / "raw" / "Week1")
    one, two = tmp_path / "one", tmp_path / "two"
    for target in (one, two):
        ss.import_plate({"src": str(target), "import_source": str(raw)})
    ss.run_screen({**_mask_settings(one), "screen_phases": "prep,masks", "record_timing": True})
    preprocess_generate_masks(_mask_settings(two, preprocess=True, masks=False))
    preprocess_generate_masks(_mask_settings(two, preprocess=False, masks=True))

    merged_one, merged_two = _arrays(one / "merged"), _arrays(two / "merged")
    assert len(merged_one) == 4 and merged_one.keys() == merged_two.keys()
    for name in merged_one:
        np.testing.assert_array_equal(merged_one[name], merged_two[name])
    masks_one, masks_two = _arrays(one / "masks"), _arrays(two / "masks")
    assert masks_one.keys() == masks_two.keys() and len(masks_one) == 8
    for name in masks_one:
        np.testing.assert_array_equal(masks_one[name], masks_two[name])
    assert all(array.max() > 0 for array in masks_one.values())
    saved = pd.read_csv(one / "settings" / "gen_mask_settings.csv")
    flags = dict(zip(saved.iloc[:, 0], saved.iloc[:, 1].astype(str)))
    assert flags["preprocess"] == "True" and flags["masks"] == "True"
    timing = json.loads((one / "settings" / "screen_timing.json").read_text())
    assert [row["phase"] for row in timing["phases"]] == ["prep+masks"]
    assert timing["phases"][0]["wall_s"] > 0 and timing["phases"][0]["device"] == "cpu"


def test_import_names_and_records_every_file_and_repeats_cleanly(tmp_path):
    raw = _raw_plate(tmp_path / "Week1", wells=("B02", "C03"), sites=(1,))
    (raw / "notes.txt").write_text("not an image")
    src = tmp_path / "work"
    first = ss.import_plate({"src": str(src), "import_source": str(raw), "import_wells": "B02"})
    assert first["files"] == 3 and first["fields"] == 1 and first["placed"] == 3
    names = sorted(path.name for path in src.glob("*.tif"))
    assert names == [f"Week1_B02_T0001F001L01A01Z01C{c:02d}.tif" for c in (1, 2, 4)]
    assert all((src / name).is_symlink() for name in names)
    recorded = pd.read_csv(src / "settings" / "import_map.csv")
    assert list(recorded.columns) == ["target", "source"]
    assert all(Path(path).exists() for path in recorded["source"])
    again = ss.import_plate({"src": str(src), "import_source": str(raw), "import_wells": "B02"})
    assert again["placed"] == 0 and again["files"] == 3


def test_an_import_map_round_trips_and_copies(tmp_path):
    raw = _raw_plate(tmp_path / "Week1", wells=("B02",), sites=(1,))
    first = ss.import_plate({"src": str(tmp_path / "a"), "import_source": str(raw)})
    mapping = first["map"].assign(source=[os.path.basename(p) for p in first["map"]["source"]])
    mapping.to_csv(tmp_path / "map.csv", index=False)
    second = ss.import_plate({"src": str(tmp_path / "b"), "import_source": str(raw),
                              "import_map": str(tmp_path / "map.csv"), "import_mode": "copy"})
    assert second["map"]["target"].tolist() == first["map"]["target"].tolist()
    assert second["map"]["source"].tolist() == first["map"]["source"].tolist()
    copied = sorted((tmp_path / "b").glob("*.tif"))
    assert len(copied) == 3 and not any(path.is_symlink() for path in copied)
    assert copied[0].read_bytes() == Path(first["map"]["source"].iloc[0]).read_bytes()


def test_import_refuses_what_it_cannot_name(tmp_path):
    raw = _raw_plate(tmp_path / "Week1", wells=("B02",), sites=(1,))
    with pytest.raises(ValueError, match="import_source"):
        ss.import_plate({"src": str(tmp_path / "w")})
    with pytest.raises(ValueError, match="import_naming"):
        ss.import_plate({"src": "w", "import_source": str(raw), "import_naming": "cq1"})
    with pytest.raises(ValueError, match="import_mode"):
        ss.import_plate({"src": "w", "import_source": str(raw), "import_mode": "move"})
    with pytest.raises(ValueError, match="chanID"):
        ss.import_plate({"src": str(tmp_path / "w"), "import_source": str(raw),
                         "import_regex": r"(?P<wellID>[A-P]\d{2})_s(?P<fieldID>\d+)"})
    pd.DataFrame({"source": ["x"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="target"):
        ss.import_plate({"src": str(tmp_path / "w"), "import_source": str(raw),
                         "import_map": str(tmp_path / "bad.csv")})
    pd.DataFrame({"source": ["a", "b"], "target": ["same.tif", "same.tif"]}).to_csv(
        tmp_path / "twice.csv", index=False)
    with pytest.raises(ValueError, match="one name"):
        ss.import_plate({"src": str(tmp_path / "w"), "import_source": str(raw),
                         "import_map": str(tmp_path / "twice.csv")})


def _toy_cells():
    rng = np.random.default_rng(0)
    rows = []
    for plate in ("P1", "P2"):
        for well in ("B02", "B03", "C02"):
            for cell in range(1, 21):
                rows.append({"plate": plate, "well": well, "site": 1, "Metadata_cell_id": cell,
                             "cell_area": float(rng.normal(100, 10)),
                             "nucleus_area": float(rng.normal(40, 5)),
                             "Location_cell_centroid-0": 1.0})
    return pd.DataFrame(rows).astype({"cell_area": "float32", "nucleus_area": "float32"})


def test_aggregate_writes_well_tables_heterogeneity_and_control_percentiles(tmp_path):
    cells = _toy_cells()
    files = []
    for plate, part in cells.groupby("plate"):
        path = tmp_path / f"{plate}.parquet"
        part.to_parquet(path, index=False)
        files.append(path)
    metadata = pd.DataFrame({"plate": ["P1", "P1", "P2", "P2", "P2"],
                             "well": ["B02", "B03", "B02", "B03", "C02"],
                             "compound": ["DMSO", "taxol", "DMSO", "DMSO", "taxol"]})
    controls = ss.control_wells(metadata, control="DMSO")
    assert controls == {("P1", "B02"), ("P2", "B02"), ("P2", "B03")}
    result = ss.aggregate_wells(files, tmp_path / "tables", controls=controls,
                                percentiles="1,99", heterogeneity=True)
    assert result["plates"] == 2 and result["n_cells"] == 120 and result["n_features"] == 2
    means = pd.read_parquet(tmp_path / "tables" / "wells_mean.parquet")
    expected = cells.groupby(["plate", "well"])[["cell_area", "nucleus_area"]].mean().reset_index()
    pd.testing.assert_frame_equal(means.drop(columns="n_cells"), expected)
    assert set(means["n_cells"]) == {20} and "Location_cell_centroid-0" not in means
    medians = pd.read_parquet(tmp_path / "tables" / "wells_median.parquet")
    assert medians["cell_area"].iloc[0] == pytest.approx(
        cells[(cells.plate == "P1") & (cells.well == "B02")]["cell_area"].median())
    het = pd.read_parquet(tmp_path / "tables" / "wells_heterogeneity.parquet")
    sub = cells[(cells.plate == "P2") & (cells.well == "C02")]["cell_area"]
    row = het[(het.plate == "P2") & (het.well == "C02")].iloc[0]
    assert row["cell_area__sd"] == pytest.approx(sub.std(), rel=1e-6)
    assert row["cell_area__p90mp10"] == pytest.approx(sub.quantile(0.9) - sub.quantile(0.1), rel=1e-6)
    limits = pd.read_parquet(tmp_path / "tables" / "dmso_percentiles.parquet")
    assert limits["q"].tolist() == [0.01, 0.99, 0.01, 0.99]
    p2 = cells[(cells.plate == "P2") & cells.well.isin(["B02", "B03"])]["cell_area"]
    assert limits[(limits.plate == "P2") & (limits.q == 0.99)]["cell_area"].iloc[0] == \
        pytest.approx(p2.quantile(0.99), rel=1e-6)


def test_aggregate_refuses_bad_requests(tmp_path):
    with pytest.raises(ValueError, match="levels"):
        ss.aggregate_wells([], tmp_path, levels="well_max")
    with pytest.raises(ValueError, match="no plate files"):
        ss.aggregate_wells([], tmp_path)
    pd.DataFrame({"plate": ["P"], "well": ["B02"], "size": [1.0]}).to_parquet(tmp_path / "x.parquet")
    with pytest.raises(ValueError, match="feature"):
        ss.aggregate_wells([tmp_path / "x.parquet"], tmp_path / "t")
    with pytest.raises(ValueError, match="0-100"):
        ss.aggregate_wells([tmp_path / "x.parquet"], tmp_path / "t", percentiles="1,150")
    with pytest.raises(ValueError, match="three"):
        ss.control_wells(pd.DataFrame(), columns="plate,well")
    with pytest.raises(ValueError, match="lacks"):
        ss.control_wells(pd.DataFrame({"plate": []}))


def _toy_database(src: Path, pathogen_column: bool = True) -> Path:
    db = src / "measurements" / "measurements.db"
    db.parent.mkdir(parents=True)
    keys = {"plateID": "Week1", "rowID": "r2", "columnID": "c2", "timeID": "t1"}
    cells, cyto, nuclei = [], [], []
    for field in (1, 2):
        prcf = f"Week1_r2_c2_f{field}"
        for label in (1, 2, 3):
            base = {**keys, "fieldID": f"f{field}", "prcf": prcf,
                    "file_name": f"Week1_B02_{field}_1", "path_name": "x", "object_label": label}
            cells.append({**base, "cell_area": 100.0 + label, "cell_centroid-0": 5.0,
                          "measurement_ndim": 2.0,
                          **({"cell_distance_to_pathogen": np.inf} if pathogen_column else {})})
            cyto.append({**base, "cytoplasm_area": 50.0 + label})
        for nucleus, (cell, area) in enumerate([(1, 30.0), (1, 40.0), (2, 25.0), (None, 9.0)], 1):
            nuclei.append({**keys, "fieldID": f"f{field}", "prcf": prcf, "file_name": "n",
                           "path_name": "x", "object_label": nucleus, "cell_id": cell,
                           "nucleus_area": area, "nucleus_cell_id": cell})
    with sqlite3.connect(db) as con:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        pd.DataFrame(cyto).to_sql("cytoplasm", con, index=False)
        pd.DataFrame(nuclei).to_sql("nucleus", con, index=False)
    return db


def test_export_joins_objects_and_writes_counts_to_the_image_table(tmp_path):
    src = tmp_path / "Week1"
    db = _toy_database(src)
    info = ss.export_plate_parquet(src)
    frame = pd.read_parquet(info["path"])
    assert info["n_cells"] == 6 and info["n_cells_db"] == 6 and info["n_fields"] == 2
    assert info["cells_without_nucleus"] == 2 and info["cells_multi_nucleus"] == 2
    assert info["nuclei_without_cell"] == 2 and info["n_wells"] == 1
    assert list(frame.columns[:4]) == ["plate", "well", "site", "Metadata_cell_id"]
    assert set(frame["well"]) == {"B02"} and set(frame["plate"]) == {"Week1"}
    assert "Location_cell_centroid-0" in frame and "measurement_ndim" not in frame
    assert not any("pathogen" in column for column in frame.columns)
    assert "nucleus_cell_id" not in frame
    first = frame[(frame.site == 1) & (frame.Metadata_cell_id == 1)].iloc[0]
    assert first["nucleus_area"] == 40.0 and first["Metadata_n_nuclei"] == 2
    assert frame["cell_area"].dtype == np.float32
    with sqlite3.connect(db) as con:
        image = pd.read_sql("select file_name, n_cells, cells_without_nucleus, "
                            "cells_multi_nucleus from image order by file_name", con)
    assert image["n_cells"].tolist() == [3, 3]
    assert image["cells_without_nucleus"].tolist() == [1, 1]
    assert image["cells_multi_nucleus"].tolist() == [1, 1]


def test_export_refuses_a_plate_it_cannot_read(tmp_path):
    with pytest.raises(ValueError, match="no measurements"):
        ss.export_plate_parquet(tmp_path)
    db = tmp_path / "measurements" / "measurements.db"
    db.parent.mkdir()
    with sqlite3.connect(db) as con:
        con.execute("create table nucleus (a int)")
    with pytest.raises(ValueError, match="no cell table"):
        ss.export_plate_parquet(tmp_path)
    src = tmp_path / "dup"
    db = _toy_database(src)
    with sqlite3.connect(db) as con:
        con.execute("insert into cell select * from cell limit 1")
    with pytest.raises(ValueError, match="twice"):
        ss.export_plate_parquet(src)


def test_run_screen_exports_and_aggregates_from_one_settings_dict(tmp_path, monkeypatch):
    src = tmp_path / "Week1"
    _toy_database(src)
    called = {}
    import spacr.measure as measure

    monkeypatch.setattr(measure, "measure_crop", lambda settings: called.update(settings))
    metadata = tmp_path / "meta.csv"
    pd.DataFrame({"plate": ["Week1"], "well": ["B02"], "compound": ["DMSO"]}).to_csv(metadata, index=False)
    result = ss.run_screen({
        "src": str(src), "screen_phases": "measure,export,aggregate", "channels": [0, 1, 2],
        "normalize": True, "measure_overrides": {"cell_min_size": 177},
        "aggregate_metadata": str(metadata), "aggregate_heterogeneity": True})
    assert called["cell_min_size"] == 177 and called["channels"] == [0, 1, 2]
    assert called["normalize"] == ss._measure_defaults({})["normalize"] is False
    assert result["phases"] == ["measure", "export", "aggregate"]
    assert result["export"][0]["n_cells"] == 6
    tables = sorted(path.name for path in (src / "tables").glob("*.parquet"))
    assert tables == ["dmso_percentiles.parquet", "wells_heterogeneity.parquet",
                      "wells_mean.parquet", "wells_median.parquet"]
    assert [row["phase"] for row in result["timing"]] == ["measure", "export", "aggregate"]
    assert (src / "settings" / "screen_settings.csv").exists()
    with pytest.raises(ValueError, match="screen_phases"):
        ss.run_screen({"src": str(src), "screen_phases": "segment"})
    with pytest.raises(ValueError, match="src"):
        ss.run_screen({"src": ""})


def test_the_screen_module_is_registered_and_alpha():
    from spacr import cli
    from spacr.settings import ALPHA_FEATURES

    module = cli.resolve_module("screen")
    assert module is not None and module.entry == "spacr.screen_staging:run_screen"
    defaults = cli.module_defaults(module)
    assert defaults["screen_phases"] == "import,prep,masks,measure,export,aggregate"
    assert "nucleus_channel" in defaults and "cell_min_size" in defaults
    assert set(ALPHA_FEATURES[893]["module_settings"]["screen"]) == set(ss._DEFAULTS)
    assert ss.screen_defaults({"src": "x", "aggregate_control": "none"})["aggregate_control"] == "none"
