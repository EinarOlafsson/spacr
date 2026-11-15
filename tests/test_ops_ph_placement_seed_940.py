"""Item 940, part 2: the phenotype phase placed from a phenotype placement seed.

A run on a few fields at a well's rim cannot align the six anchors the
phenotype phase fits its raster to. ``ops_ph_placement_seed`` names a file
(schema ``spacr.ops_ph_placement_seed`` v1, frame ``sbs_tile_anchored``) with
the whole-well run's raster, scale and rotation and the sequencing tiles they
were measured against; the phase moves the raster onto this run's stitch by
those tiles and aligns nothing. Empty (the default) leaves the phase as it was.

The offline tests use a seed written here; the last tests run the published
OPS end-to-end core tier (network, cached once).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spacr import stitch_core as core
from spacr.ops_layout import round_well_layout

TILES = {"321": {"y": 15142.084290753643, "x": 24118.35400420851},
         "322": {"y": 13875.109199167093, "x": 24109.209044432857}}


def _seed(**changes):
    """A small valid seed for a 1,281-field well, with ``changes`` applied."""
    seed = {
        "schema": core.PH_PLACEMENT_SCHEMA, "schema_version": 1,
        "plate": "P1", "well": "A1", "phenotype_fields": 1281,
        "sbs_tiles": 333,
        "frame": {"kind": "sbs_tile_anchored", "reference_tiles": TILES,
                  "tile_shape": [1480, 1480]},
        "raster": {"origin": [26180.4, 853.7], "column_step": [-4.6, 633.7],
                   "row_step": [633.5, 4.5]},
        "transform": {"scale": 0.25, "rotation_deg": -0.002,
                      "field_shape": [2960, 2960]},
        "fields": [],
    }
    seed.update(changes)
    return seed


def _write(tmp_path, seed, name="seed.json"):
    """Write ``seed`` as JSON and return its path."""
    path = tmp_path / name
    path.write_text(json.dumps(seed))
    return path


def test_a_phenotype_seed_loads_with_its_path_and_sha256(tmp_path):
    loaded = core.load_phenotype_placement_seed(_write(tmp_path, _seed()))
    assert loaded["schema"] == "spacr.ops_ph_placement_seed"
    assert loaded["path"].endswith("seed.json") and len(loaded["sha256"]) == 64


@pytest.mark.parametrize("change, message", [
    ({"schema": "spacr.ops_placement"}, "is not a phenotype placement seed"),
    ({"schema_version": 2}, "update spaCR"),
    ({"schema_version": "x"}, "update spaCR"),
    ({"frame": {"kind": "well_absolute"}}, "frame.kind"),
])
def test_another_schema_version_or_frame_is_refused(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        core.load_phenotype_placement_seed(_write(tmp_path, _seed(**change)))


def test_a_missing_key_and_a_corrupt_file_are_refused(tmp_path):
    seed = _seed()
    del seed["raster"]
    with pytest.raises(ValueError, match="is missing raster"):
        core.load_phenotype_placement_seed(_write(tmp_path, seed))
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(ValueError, match="not valid JSON"):
        core.load_phenotype_placement_seed(bad)
    with pytest.raises(ValueError, match="cannot be read"):
        core.load_phenotype_placement_seed(tmp_path / "absent.json")


def test_the_sbs_placement_reader_still_refuses_a_phenotype_seed(tmp_path):
    with pytest.raises(ValueError, match="is not a placement file"):
        core.load_placement_seed(_write(tmp_path, _seed()))


def test_the_raster_moves_by_the_mean_offset_of_the_reference_tiles_placed():
    placements = {321: (1276.0, 9.0), 322: (9.0, 0.0), 331: (0.0, 1267.0)}
    placed = core.phenotype_raster_from_seed(_seed(), placements, plate="P1",
                                             well="a01", fields=1281)
    expected = np.mean([[1276.0 - TILES["321"]["y"], 9.0 - TILES["321"]["x"]],
                        [9.0 - TILES["322"]["y"], 0.0 - TILES["322"]["x"]]],
                       axis=0)
    np.testing.assert_allclose(placed["offset"], expected, rtol=0, atol=1e-9)
    assert placed["reference_tiles"] == [321, 322]
    layout = round_well_layout(1281)
    column, row = layout.position(1264)
    centre = (np.array([26180.4, 853.7]) + expected
              + column * np.array([-4.6, 633.7]) + row * np.array([633.5, 4.5]))
    np.testing.assert_allclose(placed["centres"][1264], centre, rtol=0, atol=1e-9)
    assert len(placed["centres"]) == 1281
    assert (placed["scale"], placed["rotation_deg"], placed["field_shape"]) == (
        0.25, -0.002, (2960, 2960))


@pytest.mark.parametrize("kwargs, placements, message", [
    ({"plate": "P2", "well": "A1", "fields": 1281}, {321: (0.0, 0.0)},
     "is for P1 A1, not P2 A1"),
    ({"plate": "P1", "well": "B1", "fields": 1281}, {321: (0.0, 0.0)},
     "not P1 B1"),
    ({"plate": "P1", "well": "A1", "fields": 1280}, {321: (0.0, 0.0)},
     "describes 1281 phenotype fields"),
    ({"plate": "P1", "well": "A1", "fields": 1281}, {7: (0.0, 0.0)},
     "none of the phenotype placement seed's reference tiles"),
])
def test_a_seed_for_another_run_is_refused(kwargs, placements, message):
    with pytest.raises(ValueError, match=message):
        core.phenotype_raster_from_seed(_seed(), placements, **kwargs)


def test_the_default_is_off_and_needs_no_default_change_row():
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import DEFAULT_CHANGES

    assert OPS_DEFAULTS["ops_ph_placement_seed"] is None
    assert not any("ops_ph_placement_seed" in row
                   for row in DEFAULT_CHANGES.values())


def _settings(root: Path, dst: Path):
    """The core tier's shipped settings with absolute paths."""
    from spacr.cli import load_settings_file

    settings = dict(load_settings_file(str(root / "settings" /
                                           "ops_testdata_settings.csv")))
    for key in ("genotype_source", "phenotype_source", "ops_library",
                "ops_flatfield_maps"):
        settings[key] = str(root / settings[key])
    settings["dst_root"] = str(dst)
    return settings


@pytest.mark.network
@pytest.mark.integration
def test_the_core_tier_without_a_seed_still_needs_anchors(ops_e2e_core, tmp_path):
    from spacr.ops_engine import run_ops

    settings = _settings(ops_e2e_core, tmp_path)
    with pytest.raises(ValueError, match="phenotype fields aligned"):
        run_ops(settings, wells=["A1"], phases=("stitch", "phenotype"))


@pytest.mark.network
@pytest.mark.integration
def test_a_seeded_core_tier_gives_the_production_field_transforms(ops_e2e_core,
                                                                   tmp_path):
    """1255 and 1264 land on production's field_transform within 1e-6 px."""
    import sqlite3

    import pandas as pd

    from spacr.ops_engine import run_ops
    from spacr.ops_merge import field_transform

    root = ops_e2e_core
    seed_path = root / "reference" / "inputs" / "ops_ph_placement_seed.json"
    seed = json.loads(seed_path.read_text())
    settings = _settings(root, tmp_path)
    settings["ops_ph_placement_seed"] = str(seed_path)
    result = run_ops(settings, wells=["A1"], phases=("stitch", "phenotype"))
    report = result["wells"]["A1"]["phenotype"]
    assert report["placement"] == "seeded" and report["anchors_used"] == 0
    assert report["seed_reference_tiles"] == [321, 322, 331, 332]
    with sqlite3.connect(result["db"]) as con:
        rows = pd.read_sql("select * from ops_phenotype", con).set_index("site")
    assert len(rows) == 1281
    dy, dx = report["seed_offset_px"]
    for field in seed["fields"]:
        if field["site"] not in (1255, 1264):
            continue
        row = rows.loc[field["site"]]
        _, _, translation = field_transform(
            seed["transform"]["scale"], seed["transform"]["rotation_deg"],
            (field["centre_y"], field["centre_x"]),
            seed["transform"]["field_shape"])
        np.testing.assert_allclose([row.offset_y - dy, row.offset_x - dx],
                                   translation, rtol=0, atol=1e-6)
        np.testing.assert_allclose([row.offset_y - dy, row.offset_x - dx],
                                   [field["offset_y"], field["offset_x"]],
                                   rtol=0, atol=1e-6)
        assert int(row.sbs_site) == field["sbs_site"]
        assert row.transform_source == "seed"
        assert (row.scale, row.rotation_deg) == (0.24994, -0.002)


@pytest.mark.network
@pytest.mark.integration
def test_a_seed_with_another_field_size_is_refused(ops_e2e_core, tmp_path):
    from spacr.ops_engine import run_ops

    root = ops_e2e_core
    seed = json.loads((root / "reference" / "inputs" /
                       "ops_ph_placement_seed.json").read_text())
    seed["transform"]["field_shape"] = [1480, 1480]
    settings = _settings(root, tmp_path / "run")
    settings["ops_ph_placement_seed"] = str(_write(tmp_path, seed))
    with pytest.raises(ValueError, match="the phenotype placement seed describes"):
        run_ops(settings, wells=["A1"], phases=("stitch", "phenotype"))
