"""N692: the FIB-SEM example volume, its build tool and ``spacr-download fibsem``.

No network: the source volumes are synthetic, and the download is replaced by
one that hands over the archive the build tool made.
"""
from __future__ import annotations

import csv
import importlib.util
import shutil
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import cli_download
from spacr.example_archives import (EXAMPLE_ARCHIVES, FIBSEM_EXAMPLE_REPO,
                                    example_set)
from spacr.qt import mask_engine as engine

ROOT = Path(__file__).resolve().parent.parent


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_fibsem_example_dataset",
        ROOT / "tools" / "build_fibsem_example_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tool = _tool()
    source = tmp_path_factory.mktemp("source")
    rng = np.random.default_rng(692)
    image = rng.integers(0, 255, (200, 12, 10), dtype=np.uint8)
    labels = np.zeros(image.shape, dtype=np.uint8)
    labels[10:20, 1:3, 1:3] = 9
    labels[70:90, 2:6, 2:5] = 40
    labels[100:150, 7:10, 5:9] = 17
    labels[192:199, 0:2, 0:2] = 3
    tifffile.imwrite(source / tool.SOURCE_IMAGE, image)
    tifffile.imwrite(source / tool.SOURCE_LABELS, labels)
    dest = tmp_path_factory.mktemp("built")
    archive = tool.build(dest, source)
    return types.SimpleNamespace(tool=tool, dest=dest, archive=archive,
                                 image=image, labels=labels)


def test_the_crop_is_the_fixed_z_range_with_labels_renumbered(built):
    folder = built.dest / built.tool.FOLDER
    image = tifffile.imread(folder / "hela_fibsem.tif")
    labels = tifffile.imread(folder / "ground_truth_masks" / "hela_fibsem.tif")
    assert np.array_equal(image, built.image[64:192])
    assert labels.dtype == np.uint16 and labels.shape == image.shape
    assert sorted(np.unique(labels)) == [0, 1, 2]
    assert np.array_equal(labels > 0, built.labels[64:192] > 0)
    assert np.all(labels[built.labels[64:192] == 40] == 2)
    assert np.all(labels[built.labels[64:192] == 17] == 1)


def test_both_files_are_zyx_hyperstacks_calibrated_at_15_nm(built):
    folder = built.dest / built.tool.FOLDER
    for path in (folder / "hela_fibsem.tif",
                 folder / "ground_truth_masks" / "hela_fibsem.tif"):
        with tifffile.TiffFile(path) as handle:
            assert handle.series[0].axes == "ZYX"
            assert handle.imagej_metadata["unit"] == "nm"
            assert handle.imagej_metadata["spacing"] == 15.0
            x = handle.pages[0].tags["XResolution"].value
            assert x[1] / x[0] == pytest.approx(15.0)


def test_the_archive_ends_with_its_manifest_and_matches_the_registry(built):
    with tarfile.open(built.archive) as tar:
        names = tar.getnames()
    assert names[-1] == "manifest.csv"
    assert built.archive.name == EXAMPLE_ARCHIVES[FIBSEM_EXAMPLE_REPO]
    chosen = example_set("fibsem")
    assert chosen.repo == FIBSEM_EXAMPLE_REPO == built.tool.REPO
    assert chosen.folder == built.tool.FOLDER and not chosen.in_default
    for marker in chosen.markers:
        assert marker in names
    rows = list(csv.DictReader((built.dest / built.tool.FOLDER / "manifest.csv")
                               .open(encoding="utf-8")))
    assert rows[0]["modality"] == "FIB-SEM" and rows[0]["mitochondria"] == "2"
    card = (built.dest / "README.md").read_text(encoding="utf-8")
    assert "license: cc0-1.0" in card and "EMPIAR-10982" in card
    assert "10.1016/j.cels.2022.12.006" in card


def test_the_volume_opens_in_the_3d_editor_with_its_masks(built):
    folder = built.dest / built.tool.FOLDER
    session = engine.load_volume_image_and_mask(
        str(folder), "hela_fibsem.tif", str(folder / "ground_truth_masks"))
    assert session.axes == "ZYX" and session.spacing == (15.0, 15.0, 15.0)
    assert session.unit == "nm" and session.mask.max() == 2
    blank = engine.load_volume_image_and_mask(str(folder), "hela_fibsem.tif")
    assert not blank.mask.any()


def test_spacr_download_fibsem_unpacks_the_set(built, tmp_path, monkeypatch):
    calls = []

    def fake_download(repo, archive, folder, *, progress=None, chunk_size=None):
        calls.append((repo, archive))
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / archive
        shutil.copyfile(built.archive, target)
        if progress is not None:
            progress(target.stat().st_size, target.stat().st_size)
        return target

    monkeypatch.setattr(cli_download, "download_archive", fake_download)
    monkeypatch.setattr("shutil.disk_usage", lambda path: types.SimpleNamespace(
        total=10 ** 14, used=0, free=10 ** 14))
    dest = tmp_path / "spacr-data"
    assert cli_download.main(["fibsem", "--dest", str(dest), "--yes"]) == 0
    assert calls == [(FIBSEM_EXAMPLE_REPO, "spacr-example-fibsem.tar")]
    unpacked = list(dest.rglob("hela_fibsem.tif"))
    assert len(unpacked) == 2
    assert any(path.parent.name == "ground_truth_masks" for path in unpacked)
