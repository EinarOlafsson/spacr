"""OPS end-to-end tests on the published well A1 test data.

The data is ``einarolafsson/spacr-example-ops-e2e`` at the pinned revision
(:data:`spacr.example_archives.OPS_E2E_REVISION`): a 2 x 2 block of
sequencing tiles of Funk et al. 2022 plate A well A1, all eleven cycles, and
three phenotype fields, with the settings files and spaCR's reference outputs
(``reference/PROVENANCE.json``). Two tiers of the same sites:

* core (about 194 MB, raw pixels inside a region of interest only): reader
  and layout, stitch and phenotype tests ONLY;
* full (1.25 GB, unmodified frames): decode tests, opt-in with
  ``SPACR_OPS_E2E_FULL=1``.

What runs by default once the network is reachable (or the tier is cached):
the reader, layout and stitch checks, in seconds. Segmentation takes about
40 minutes per tier on a CPU, so the objects check runs only with
``SPACR_OPS_E2E_SEGMENT=1``. The decode check uses the reference nuclei as
its fixed input, so a segmentation difference cannot move it. Everything is
``network``-marked, which every hosted CI selection excludes.

Tolerances are the design card's (2026-11-13, section 3): integers and
categories exact, placements within 1e-6 px, the composite window's md5 or
max |difference| <= 1e-3, call quality and fraction within 1e-6 relative.
Gate (c), restated by the maintainer: >= 99 % same guide where both called,
for nuclei more than 100 px from the subset edge, on the full tier.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.network, pytest.mark.integration]

WELL = "A1"
PLATE = "20200202_6W-LaC024A"
SBS_SITES = {321, 322, 331, 332}
PH_SITES = {1255, 1264, 1280}
TILE = 1480
PATH_KEYS = ("genotype_source", "phenotype_source", "ops_library",
             "ops_flatfield_maps")


def _settings(root: Path, dst: Path, name: str = "ops_testdata_settings.csv"):
    """The tier's shipped settings file, with its paths made absolute.

    :param root: the tier root.
    :param dst: the run folder (``dst_root``).
    :param name: the settings file in ``settings/``.
    :returns: the settings mapping.
    """
    from spacr.cli import load_settings_file

    settings = dict(load_settings_file(str(root / "settings" / name)))
    for key in PATH_KEYS:
        settings[key] = str(root / settings[key])
    settings["dst_root"] = str(dst)
    return settings


def _table(db: Path, name: str) -> pd.DataFrame:
    """One well's rows of an OPS table."""
    with sqlite3.connect(str(db)) as con:
        return pd.read_sql(f"select * from {name} where well = ?", con,
                           params=(WELL,))


def _run(settings, phases):
    """Run spaCR's OPS pipeline on well A1."""
    from spacr.ops_engine import run_ops

    return run_ops(settings, wells=[WELL], phases=tuple(phases))


def _pinned_environment(root: Path) -> bool:
    """Whether torch and cellpose are the versions the references used."""
    lock = json.loads((root / "reference" / "PROVENANCE.json").read_text())[
        "environment_lock"]
    try:
        torch = metadata.version("torch").split("+")[0]
        cellpose = metadata.version("cellpose")
    except metadata.PackageNotFoundError:
        return False
    return torch == lock["torch"] and cellpose == lock["cellpose"]


def _assert_geometry(found: pd.DataFrame, root: Path) -> None:
    """The stitch placements equal the reference's within 1e-6 px."""
    ref = pd.read_parquet(root / "reference" / "stitch" / "ops_geometry.parquet")
    keys = ["cycle", "site"]
    found = found.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    assert found[keys + ["tile_height", "tile_width"]].astype(int).equals(
        ref[keys + ["tile_height", "tile_width"]].astype(int))
    for column in ("y", "x", "origin_y", "origin_x"):
        np.testing.assert_allclose(found[column], ref[column], rtol=0, atol=1e-6)


def test_the_reader_finds_every_site_and_the_round_well_layouts(ops_e2e_core):
    """Reader and layout: 4 tiles x 11 cycles, 3 fields, 333 / 1281 layouts."""
    from spacr.ops_engine import _UNCYCLED_CYCLE, _index_tiles, _plane_sources
    from spacr.ops_layout import round_well_layout

    root = ops_e2e_core
    sbs = _index_tiles(str(root / PLATE / "sequencing"), "cycled")[WELL]
    assert sorted(sbs) == list(range(1, 12))
    for cycle, sites in sbs.items():
        assert set(sites) == SBS_SITES, cycle
        expected = {"CY3", "A594", "CY5", "CY7"} | ({"DAPI"} if cycle == 1 else set())
        for site, files in sites.items():
            assert set(_plane_sources(files)) == expected, (cycle, site)
    ph = _index_tiles(str(root / PLATE / "phenotype"), "uncycled")[WELL]
    assert set(ph[_UNCYCLED_CYCLE]) == PH_SITES
    assert round_well_layout(max(SBS_SITES) + 1).site_count == 333
    assert round_well_layout(max(PH_SITES) + 1).site_count == 1281


@pytest.mark.parametrize("settings_file", ["ops_testdata_settings.csv",
                                           "ops_testdata_settings_similarity.csv"])
def test_the_stitch_matches_the_reference(ops_e2e_core, tmp_path, settings_file):
    """Stitch: placements, edges and the feathered composite window."""
    import tifffile

    from spacr.ops_compose import Window, compose_window
    from spacr.ops_engine import _NUCLEAR, _index_tiles, _plane_sources, _read_plane

    root = ops_e2e_core
    result = _run(_settings(root, tmp_path, settings_file), ["stitch"])
    report = result["wells"][WELL]["stitch"]
    assert (report["placed"], report["edges_proposed"],
            report["edges_accepted"]) == (4, 4, 4)
    assert report["unreadable"] == []
    geometry = _table(Path(result["db"]), "ops_geometry")
    _assert_geometry(geometry, root)

    meta = json.loads((root / "reference" / "stitch" /
                       "composite_window.json").read_text())
    index = _index_tiles(str(root / PLATE / "sequencing"), "cycled")[WELL][1]
    sources = {s: _plane_sources(f).get(_NUCLEAR) for s, f in index.items()}
    placements = {int(r.site): (float(r.y), float(r.x))
                  for r in geometry.itertuples()}
    image, _ = compose_window(Window(**meta["window"]), placements,
                              lambda s: _read_plane(sources[s]),
                              tile_shape=(TILE, TILE), blend=meta["blend"])
    image = np.ascontiguousarray(image.astype(np.float32))
    if hashlib.md5(image.tobytes()).hexdigest() != meta["pixel_md5_float32"]:
        ref = tifffile.imread(root / "reference" / "stitch" / "composite_window.tif")
        assert float(np.max(np.abs(image - ref))) <= 1e-3


@pytest.mark.slow
@pytest.mark.heavy
def test_the_objects_match_the_reference(ops_e2e_core, tmp_path):
    """Stitch then objects: the SBS nuclei equal the reference's.

    About 40 minutes of Cellpose on a CPU, so it runs only with
    ``SPACR_OPS_E2E_SEGMENT=1``. On the pinned torch and cellpose (the
    references' lock) every row must match; on another build the count must
    agree within 0.5 % (the card's open question 7, exact on the lock).
    """
    if os.environ.get("SPACR_OPS_E2E_SEGMENT") != "1":
        pytest.skip("set SPACR_OPS_E2E_SEGMENT=1 to segment (about 40 min on CPU)")
    root = ops_e2e_core
    settings = _settings(root, tmp_path)
    settings["ops_gpu"] = False
    result = _run(settings, ["stitch", "objects"])
    found = _table(Path(result["db"]), "ops_objects").sort_values(
        "object_id").reset_index(drop=True)
    ref = pd.read_parquet(root / "reference" / "objects" / "ops_objects.parquet"
                          ).sort_values("object_id").reset_index(drop=True)
    if not _pinned_environment(root):
        assert abs(len(found) - len(ref)) <= 0.005 * len(ref)
        return
    assert len(found) == len(ref)
    for column in ref.columns:
        if column in ("plate", "well"):
            continue
        if np.issubdtype(ref[column].dtype, np.floating):
            np.testing.assert_allclose(found[column], ref[column], rtol=0,
                                       atol=1e-6, err_msg=column)
        else:
            assert found[column].tolist() == ref[column].tolist(), column


@pytest.mark.skip(reason=(
    "the phenotype -> merge/align references (similarity and tile_affine) are "
    "not in the published revision 4e45cdc1ae: that run needed the phenotype "
    "placement seed (item 940 part 2). They are built locally and wait for "
    "the maintainer's OK to publish a new dataset revision"))
def test_the_seeded_phenotype_and_merge_match_the_reference(ops_e2e_core):
    """Phenotype seeded from the shipped placement file, then merge/align."""


def _gate_c(found: pd.DataFrame, root: Path):
    """Gate (c): same guide where both called, beyond 100 px of the edge.

    The production calls are rebuilt from the reference: a nucleus listed in
    ``provenance_call_differences.parquet`` takes its ``prod_guide``, every
    other nucleus the reference call (they agreed). Nuclei production did not
    match within 3 px are therefore counted as agreeing with the reference;
    the build measured 0.9901 on 13,705 nuclei, this rebuild 0.9901 on 13,297.

    :param found: this run's ``ops_barcodes`` rows.
    :param root: the full tier root.
    :returns: ``(fraction, nuclei both called)``.
    """
    ref_dir = root / "reference"
    objects = pd.read_parquet(ref_dir / "objects" / "ops_objects.parquet")
    reference = pd.read_parquet(ref_dir / "genotype" / "ops_barcodes.parquet")
    differences = pd.read_parquet(ref_dir / "provenance_call_differences.parquet")
    geometry = pd.read_parquet(ref_dir / "stitch" / "ops_geometry.parquet")
    production = objects[["object_id"]].merge(
        reference[["object_id", "mapped_guide"]], on="object_id", how="left")
    listed = differences.set_index("object_id")["prod_guide"]
    differs = production.object_id.isin(listed.index)
    production.loc[differs, "mapped_guide"] = production.loc[
        differs, "object_id"].map(listed).to_numpy()
    frame = objects.merge(
        found[["object_id", "mapped_guide"]], on="object_id", how="left").merge(
        production.rename(columns={"mapped_guide": "prod_guide"}), on="object_id")
    top, left = geometry.y.min(), geometry.x.min()
    bottom, right = geometry.y.max() + TILE, geometry.x.max() + TILE
    y, x = frame.centroid_y, frame.centroid_x
    edge = np.minimum.reduce([y - top, bottom - y, x - left, right - x])
    inner = frame[edge > 100]
    both = inner.mapped_guide.notna() & inner.prod_guide.notna()
    same = (inner.mapped_guide == inner.prod_guide)[both]
    return float(same.mean()), int(both.sum())


@pytest.mark.slow
def test_the_full_tier_decode_meets_gate_c(ops_e2e_full, tmp_path):
    """Full tier: stitch, the reference nuclei, decode; gate (c) and identity.

    Opt-in (``SPACR_OPS_E2E_FULL=1``): 1.25 GB and about two minutes of
    decode. The nuclei are the reference ``ops_objects`` written into the
    run's database, so segmentation is not re-run here.
    """
    from spacr.ops_engine import _replace_well_rows

    root = ops_e2e_full
    settings = _settings(root, tmp_path)
    settings["n_workers"] = 1
    result = _run(settings, ["stitch"])
    db = Path(result["db"])
    _assert_geometry(_table(db, "ops_geometry"), root)
    objects = pd.read_parquet(root / "reference" / "objects" / "ops_objects.parquet")
    _replace_well_rows(str(db), "ops_objects", objects, PLATE, WELL)
    report = _run(settings, ["decode"])["wells"][WELL]["decode"]
    assert report["fields_decoded"] == 4 and report["channels_refused"] == 0
    found = _table(db, "ops_barcodes")

    fraction, both = _gate_c(found, root)
    assert both > 10_000
    assert fraction >= 0.99, f"gate (c): {fraction:.4f} same guide on {both}"

    ref = pd.read_parquet(root / "reference" / "genotype" / "ops_barcodes.parquet")
    found = found.sort_values("object_id").reset_index(drop=True)
    ref = ref.sort_values("object_id").reset_index(drop=True)
    assert found.object_id.tolist() == ref.object_id.tolist()
    for column in ("barcode", "mapped_guide", "n_reads", "n_agreeing", "n_cycles"):
        assert found[column].tolist() == ref[column].tolist(), column
    for column in ("quality", "fraction"):
        np.testing.assert_allclose(found[column], ref[column], rtol=1e-6,
                                   err_msg=column)
