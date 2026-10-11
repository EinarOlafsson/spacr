"""F534 gaps found on real infected Xenium (GSE300303, 2026-10-11).

* A crop window that reads only the tiles under it, so level 0 of a Xenium
  morphology image (about 33 GB) can be the analysis frame.
* That read never goes through zarr (tifffile's ``aszarr`` does not open
  under zarr 3 with tifffile 2023), so it works with zarr 2, zarr 3 or none.
* The number of pathogen transcripts that makes a host cell infected is a
  setting, default 1, and is written into the outputs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tracemalloc

import numpy as np
import pytest

from spacr import ops_engine as oe
from spacr.tabular import read_table

tifffile = pytest.importorskip("tifffile")

from tests.test_spatial_transcriptomics import (  # noqa: E402
    PIXEL, _masks, xenium_bundle)


def _stack(planes=3, rows=1536, cols=2048, seed=0):
    rng = np.random.default_rng(seed)
    ramp = np.add.outer(np.arange(rows), np.arange(cols)).astype(np.uint16)
    return np.stack([ramp + rng.integers(0, 50, (rows, cols), dtype=np.uint16)
                     for _ in range(planes)])


def _pyramid(path, data, *, tile=(128, 128), compression="zlib"):
    """A z-stack written as an OME pyramid: level 0 with one sub-resolution."""
    options = dict(photometric="minisblack", compression=compression,
                   metadata={"axes": "ZYX"})
    if tile:
        options["tile"] = tile
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as handle:
        handle.write(data, subifds=1, **options)
        handle.write(data[:, ::2, ::2], subfiletype=1, **options)
    return str(path)


def test_the_window_matches_a_full_read_at_every_level(tmp_path):
    data = _stack(rows=600, cols=700)
    path = _pyramid(tmp_path / "pyramid.ome.tif", data)
    with tifffile.TiffFile(path) as handle:
        assert len(handle.series[0].levels) == 2
    for level, window in ((0, (130, 70, 300, 257)), (0, (0, 0, 128, 128)),
                          (1, (31, 17, 100, 90)), (0, (650, 550, 200, 200))):
        full = oe._st_load_image(path, level)
        crop, origin = oe._st_read_window(path, level, window)
        x, y = origin
        assert np.array_equal(crop, full[y:y + crop.shape[0],
                                         x:x + crop.shape[1]]), (level, window)
        assert crop.dtype == full.dtype
    clipped, origin = oe._st_read_window(path, 0, (650, 550, 200, 200))
    assert origin == (650, 550) and clipped.shape == (50, 50)
    assert np.array_equal(clipped, data.max(axis=0)[550:, 650:])


def test_a_window_reads_only_its_tiles(tmp_path, monkeypatch):
    data = _stack(planes=4, rows=2048, cols=2048)
    path = _pyramid(tmp_path / "big.ome.tif", data)
    window = (900, 700, 200, 150)
    reads = []
    original = tifffile.FileHandle.read

    def counting(self, size=-1):
        out = original(self, size)
        reads.append(len(out))
        return out

    monkeypatch.setattr(tifffile.FileHandle, "read", counting)
    tracemalloc.start()
    crop, origin = oe._st_read_window(path, 0, window)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    monkeypatch.undo()
    level_bytes = data.nbytes
    assert peak < level_bytes / 20, (peak, level_bytes)
    assert sum(reads) < os.path.getsize(path) / 10
    assert np.array_equal(crop, data.max(axis=0)[700:850, 900:1100])
    assert origin == (900, 700)


@pytest.mark.parametrize("kind", ["strips", "rgb", "uncompressed"])
def test_striped_colour_and_raw_files_crop_the_same(tmp_path, kind):
    rng = np.random.default_rng(1)
    path = str(tmp_path / f"{kind}.tif")
    if kind == "strips":
        data = rng.integers(0, 4000, (2, 300, 260), dtype=np.uint16)
        tifffile.imwrite(path, data, rowsperstrip=16, compression="zlib",
                         photometric="minisblack", metadata={"axes": "ZYX"})
    elif kind == "rgb":
        data = rng.integers(0, 255, (300, 260, 3), dtype=np.uint8)
        tifffile.imwrite(path, data, tile=(64, 64), photometric="rgb")
    else:
        data = rng.integers(0, 4000, (300, 260), dtype=np.uint16)
        tifffile.imwrite(path, data)
    full = oe._st_load_image(path)
    crop = oe._st_load_image(path, window=(37, 101, 150, 120))
    assert np.array_equal(crop, full[101:221, 37:187])


def test_windows_are_parsed_and_checked():
    assert oe._st_parse_window("") is None
    assert oe._st_parse_window(None) is None
    assert oe._st_parse_window("1, 2;3 4") == (1.0, 2.0, 3.0, 4.0)
    assert oe._st_parse_window([5, 6, 7, 8]) == (5.0, 6.0, 7.0, 8.0)
    with pytest.raises(ValueError):
        oe._st_parse_window("1, 2, 3")
    with pytest.raises(ValueError):
        oe._st_parse_window("1, 2, 0, 4")
    with pytest.raises(ValueError):
        oe._st_clip_window((500, 500, 10, 10), (100, 100))


def test_the_windowed_read_needs_no_zarr(tmp_path):
    """The path is the same under zarr 2, zarr 3 or no zarr: it never imports
    it. Run with zarr made unimportable, as an environment without it."""
    data = _stack(planes=2, rows=300, cols=300)
    path = _pyramid(tmp_path / "nozarr.ome.tif", data)
    expected = data.max(axis=0)[40:140, 60:200]
    np.save(tmp_path / "expected.npy", expected)
    code = (
        "import sys; sys.modules['zarr'] = None\n"
        "import numpy as np\n"
        "from spacr import ops_engine as oe\n"
        f"crop, _ = oe._st_read_window({path!r}, 0, (60, 40, 140, 100))\n"
        f"assert np.array_equal(crop, np.load({str(tmp_path / 'expected.npy')!r}))\n"
        "assert sys.modules['zarr'] is None\n"
        "print('ok')\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr[-2000:]


def test_a_window_in_micrometres_registers_into_the_crop(tmp_path):
    folder = xenium_bundle(tmp_path / "x")
    bundle = oe._st_read_bundle(folder)
    whole = oe._st_register(bundle, {})
    saved = str(tmp_path / "crop" / "window.tif")
    cropped = oe._st_register(bundle, {
        "window": "20, 20, 40, 30", "window_unit": "um",
        "window_image": saved})
    x0, y0 = int(20 / PIXEL), int(20 / PIXEL)
    assert cropped["image"].shape == (int(30 / PIXEL), int(40 / PIXEL))
    assert np.array_equal(cropped["image"],
                          whole["image"][y0:y0 + 60, x0:x0 + 80])
    assert np.allclose(cropped["xy"], whole["xy"] - [x0, y0])
    window = cropped["registration"]["window"]
    assert (window["x"], window["y"], window["width"], window["height"]) == (
        40, 40, 80, 60)
    assert np.array_equal(tifffile.imread(saved), cropped["image"])
    in_px = oe._st_register(bundle, {"window": (40, 40, 80, 60),
                                     "window_unit": "px"})
    assert np.array_equal(in_px["image"], cropped["image"])


def test_counts_in_a_window_match_the_whole_frame(tmp_path):
    folder = xenium_bundle(tmp_path / "x")
    cell, nucleus, pathogen = _masks()
    rows, cols = slice(40, 120), slice(40, 160)
    paths = {}
    for kind, mask in (("cell", cell), ("pathogen", pathogen)):
        paths[kind] = str(tmp_path / f"crop_{kind}.npy")
        np.save(paths[kind], mask[rows, cols])
    summary = oe._st_run({
        "folder": folder, "masks": paths, "db": str(tmp_path / "m.db"),
        "prcf": "crop", "anndata": False, "window": "40, 40, 120, 80",
        "window_unit": "px"})
    bundle = oe._st_read_bundle(folder)
    whole = oe._st_register(bundle, {})
    labels, matrix, _ = oe._st_object_counts(bundle, whole["xy"], cell)
    inside = np.unique(cell[rows, cols])
    inside = inside[inside > 0]
    wide = read_table(str(tmp_path / "m.db"), table="cell_expression",
                      report=None)
    got = dict(zip(wide["prcfo"], wide["expr_total"]))
    for label in inside:
        expected = matrix[list(labels).index(label)].sum()
        assert got[f"crop_o{label}"] == expected
    assert summary["objects"]["cell"]["assigned_fraction_in_frame"] == 1.0
    assert summary["objects"]["cell"]["in_frame"] < len(bundle["transcripts"])


def _run(tmp_path, name, **extra):
    folder = xenium_bundle(tmp_path / "x")
    cell, _nucleus, pathogen = _masks()
    paths = {"cell": str(tmp_path / "plate1_r1_c1_f1_cell.npy"),
             "pathogen": str(tmp_path / "plate1_r1_c1_f1_pathogen.npy")}
    np.save(paths["cell"], cell)
    np.save(paths["pathogen"], pathogen)
    request = {"folder": folder, "masks": paths, "prcf": "plate1_r1_c1_f1",
               "db": str(tmp_path / name / "measurements.db"),
               "anndata": False}
    request.update(extra)
    return oe._st_run(request)


def test_the_default_threshold_keeps_one_pathogen_object(tmp_path):
    summary = _run(tmp_path, "default")
    assert summary["infection"] == {
        "rule": "pathogen_objects", "min_pathogen_transcripts": 1,
        "pathogen_genes": [], "infected": 10, "uninfected": 10}
    table = read_table(summary["files"]["infection"])
    assert set(table["min_pathogen_transcripts"]) == {1}
    assert int(table["infected"].sum()) == 10
    compared = read_table(summary["files"]["infected_vs_uninfected"])
    assert set(compared["min_pathogen_transcripts"]) == {1}
    assert summary["infected_vs_uninfected"]["min_pathogen_transcripts"] == 1


def test_the_threshold_counts_pathogen_gene_transcripts(tmp_path):
    every = _run(tmp_path, "one", pathogen_genes="GENE_UP*")
    assert every["infection"]["infected"] == 20
    assert any("Both groups" in message for message in every["messages"])
    two = _run(tmp_path, "two", pathogen_genes="GENE_UP",
               min_pathogen_transcripts=2)
    assert two["infection"]["rule"] == "pathogen_genes"
    assert two["infection"]["pathogen_genes"] == ["GENE_UP"]
    assert (two["infection"]["infected"], two["infection"]["uninfected"]) == (
        10, 10)
    table = read_table(two["files"]["infection"])
    assert set(table.loc[table["infected"], "pathogen_transcripts"]) == {25}
    assert set(table["min_pathogen_transcripts"]) == {2}
    compared = read_table(two["files"]["infected_vs_uninfected"])
    assert "GENE_UP" not in set(compared["gene"])
    many = _run(tmp_path, "many", pathogen_genes="GENE_UP",
                min_pathogen_transcripts=26)
    assert many["infection"]["infected"] == 0


def test_the_threshold_applies_to_pathogen_objects_without_genes(tmp_path):
    summary = _run(tmp_path, "objects", min_pathogen_transcripts=2)
    assert summary["infection"]["infected"] == 0
    assert "infected_vs_uninfected" not in summary["files"]
    missing = _run(tmp_path, "missing", pathogen_genes="NOT_A_GENE")
    assert missing["infection"]["rule"] == "pathogen_objects"
    assert any("None of the pathogen genes" in m for m in missing["messages"])


def test_pathogen_genes_are_matched_by_name_or_prefix():
    genes = np.array(["CHIKV_5_(0-7566)", "CHIKV_sgRNA_(7566-12036)",
                      "Itgax", "CHIK"], dtype=object)
    assert oe._st_pathogen_columns(genes, "CHIKV*").tolist() == [
        True, True, False, False]
    assert oe._st_pathogen_columns(genes, "Itgax; CHIK").tolist() == [
        False, False, True, True]
    assert not oe._st_pathogen_columns(genes, "").any()
