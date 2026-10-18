"""712: Make Masks' timelapse test movies, their build tool and their pins.

No network: the source movies are tiny synthetic CTC-style sequences, and
every download is replaced by one that hands over bytes made here.
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import importlib.util
import shutil
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import cli_download
from spacr import example_archives as archives
from spacr.example_archives import (EXAMPLE_ARCHIVE_SHA256, EXAMPLE_ARCHIVES,
                                    check_archive_sha256, example_set)
from spacr.qt import mask_engine as engine

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("yeast", "bacteria", "mammalian")


def _tool():
    spec = importlib.util.spec_from_file_location(
        "build_timelapse_example_datasets",
        ROOT / "tools" / "build_timelapse_example_datasets.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_source(folder: Path, budding: bool) -> None:
    """Five frames: track 1 divides (or buds) at frame 2, track 4 does not."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(712)
    rows = [(4, 0, 4, 0)]
    for t in range(5):
        labels = np.zeros((24, 32), dtype=np.uint16)
        labels[18:22, 2 + t:6 + t] = 4
        if budding:
            labels[2:8, 2:8] = 1
            if t >= 2:
                labels[2:5, 9:12] = 2
        elif t < 2:
            labels[2:8, 2:10] = 1
        else:
            labels[2:8, 2:5] = 2
            labels[2:8, 7 + t:10 + t] = 3
        tifffile.imwrite(folder / f"man_track{t:03d}.tif", labels)
        tifffile.imwrite(folder / f"t{t:03d}.tif",
                         rng.integers(0, 4000, labels.shape, dtype=np.uint16))
    if budding:
        rows += [(1, 0, 4, 0), (2, 2, 4, 1)]
    else:
        rows += [(1, 0, 1, 0), (2, 2, 4, 1), (3, 2, 4, 1)]
    with (folder / "lineage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "t_start", "t_end", "parent"])
        writer.writerows(rows)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tool = _tool()
    source = tmp_path_factory.mktemp("timelapse_source")
    _write_source(source / "budding", budding=True)
    _write_source(source / "dividing", budding=False)
    dest = tmp_path_factory.mktemp("timelapse_built")
    tool.MOVIES = tuple(dataclasses.replace(spec, window=(0, 5))
                        for spec in tool.MOVIES)
    archives_built = tool.build(dest, {
        "timelapse_yeast": source / "budding",
        "timelapse_mammalian": source / "dividing"})
    return types.SimpleNamespace(tool=tool, dest=dest, archives=archives_built)


def _lineage(folder: Path) -> dict:
    state = None
    for name in engine.list_images(str(folder)):
        incoming = engine.load_mask_metadata(
            str(folder), name, str(folder / "ground_truth_masks"))["lineage"]
        state = engine.merge_lineage_states(state, incoming,
                                            frame=str(folder / name))
    return state


def test_the_three_sets_are_registered_beside_the_other_make_masks_data():
    tool = _tool()
    for kind in KINDS:
        chosen = example_set(f"timelapse_{kind}")
        spec = tool.movie(f"timelapse_{kind}")
        assert chosen.repo == spec.repo == f"einarolafsson/spacr-example-timelapse-{kind}"
        assert chosen.archive == EXAMPLE_ARCHIVES[chosen.repo] == spec.archive
        assert chosen.folder == spec.folder == f"make_masks_timelapse_{kind}"
        assert not chosen.in_default and 10_000_000 < chosen.bytes < 100_000_000
        assert chosen.markers == ("manifest.csv", "ground_truth_masks/*.tif")


def test_every_timelapse_archive_has_a_sha256_pin():
    for kind in KINDS:
        pin = EXAMPLE_ARCHIVE_SHA256[f"spacr-example-timelapse-{kind}.tar"]
        assert len(pin) == 64 and int(pin, 16) >= 0
    assert set(EXAMPLE_ARCHIVE_SHA256) <= set(EXAMPLE_ARCHIVES.values())


def test_a_pinned_archive_with_other_bytes_is_refused():
    name = "spacr-example-timelapse-yeast.tar"
    check_archive_sha256(name, EXAMPLE_ARCHIVE_SHA256[name])
    check_archive_sha256("spacr-example-tem.tar", "0" * 64)
    with pytest.raises(IOError, match="published checksum"):
        check_archive_sha256(name, "0" * 64)


def test_download_archive_refuses_a_mismatched_pin_and_leaves_nothing(
        tmp_path, monkeypatch):
    import requests

    payload = b"not the published archive"

    class Response:
        headers = {"Content-Length": str(len(payload))}

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield payload

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(IOError, match="published checksum"):
        archives.download_archive(archives.TIMELAPSE_YEAST_EXAMPLE_REPO,
                                  "spacr-example-timelapse-yeast.tar", tmp_path)
    assert list(tmp_path.iterdir()) == []
    pinned = "spacr-example-timelapse-mammalian.tar"
    monkeypatch.setitem(EXAMPLE_ARCHIVE_SHA256, pinned,
                        hashlib.sha256(payload).hexdigest())
    landed = archives.download_archive(
        archives.TIMELAPSE_MAMMALIAN_EXAMPLE_REPO, pinned, tmp_path)
    assert landed.read_bytes() == payload


def test_the_archive_ends_with_its_manifest_and_holds_every_frame(built):
    for key, archive in built.archives.items():
        with tarfile.open(archive) as tar:
            names = tar.getnames()
        assert names[-1] == "manifest.csv"
        folder = archive.parent / built.tool.movie(key).folder
        rows = list(csv.DictReader((folder / "manifest.csv").open(encoding="utf-8")))
        assert len(rows) == 5
        for row in rows:
            assert row["image"] in names and row["ground_truth_mask"] in names
        card = (archive.parent / "README.md").read_text(encoding="utf-8")
        assert "license: cc-by-4.0" in card and "doi" in card.lower()


def test_ctc_divisions_become_embedded_lineage_divisions(built):
    folder = built.dest / "timelapse_mammalian" / "make_masks_timelapse_mammalian"
    mask = tifffile.imread(folder / "ground_truth_masks" / "t98g_t003.tif")
    assert mask.dtype == np.uint32 and set(np.unique(mask)) == {0, 2, 3, 4}
    state = _lineage(folder)
    cells = {cell["name"]: cell for cell in state["cells"].values()}
    assert set(cells) == {"1", "1a", "1b", "4"}
    mother = cells["1"]
    assert mother["division_frame"] == str(folder / "t98g_t002.tif")
    assert sorted(state["cells"][child]["label"] for child in mother["children"]) == [2, 3]
    assert cells["4"]["parent_uuid"] is None and not cells["4"]["children"]
    for t in range(5):
        frame = state["frames"][str(folder / f"t98g_t{t:03d}.tif")]
        assert frame["4"] == next(key for key, cell in state["cells"].items()
                                  if cell["name"] == "4")


def test_a_bud_divides_its_mother_into_her_continuation_and_the_bud(built):
    folder = built.dest / "timelapse_yeast" / "make_masks_timelapse_yeast"
    state = _lineage(folder)
    divided = [cell for cell in state["cells"].values() if cell["children"]]
    assert len(divided) == 1
    children = [state["cells"][child] for child in divided[0]["children"]]
    assert sorted((child["name"], child["label"]) for child in children) == [
        ("1", 1), ("1a", 2)]
    assert divided[0]["name"] == "1" and divided[0]["label"] == 1
    early = state["frames"][str(folder / "yeast_sjr1_t001.tif")]["1"]
    late = state["frames"][str(folder / "yeast_sjr1_t003.tif")]["1"]
    assert early == next(key for key, cell in state["cells"].items()
                         if cell is divided[0])
    assert late in divided[0]["children"]


def test_a_moved_copy_reopens_with_every_frame_rebound(built, tmp_path):
    source = built.dest / "timelapse_mammalian" / "make_masks_timelapse_mammalian"
    moved = tmp_path / "elsewhere" / "make_masks_timelapse_mammalian"
    shutil.copytree(source, moved)
    state = _lineage(moved)
    assert all(path.startswith(str(moved)) for path in state["frames"])
    first = engine.load_mask_metadata(str(moved), "t98g_t000.tif",
                                      str(moved / "ground_truth_masks"))
    assert first["lineage"] == state


def test_spacr_download_fetches_a_timelapse_set_by_key(built, tmp_path, monkeypatch):
    calls = []
    archive = built.archives["timelapse_mammalian"]

    def fake_download(repo, name, folder, *, progress=None, chunk_size=None):
        calls.append((repo, name))
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / name
        shutil.copyfile(archive, target)
        return target

    monkeypatch.setattr(cli_download, "download_archive", fake_download)
    monkeypatch.setattr("shutil.disk_usage", lambda path: types.SimpleNamespace(
        total=10 ** 14, used=0, free=10 ** 14))
    dest = tmp_path / "spacr-data"
    assert cli_download.main(["timelapse_mammalian", "--dest", str(dest), "--yes"]) == 0
    assert calls == [(archives.TIMELAPSE_MAMMALIAN_EXAMPLE_REPO,
                      "spacr-example-timelapse-mammalian.tar")]
    manifests = list(dest.rglob("manifest.csv"))
    assert len(manifests) == 1
    assert manifests[0].parent.name == "make_masks_timelapse_mammalian"
    assert example_set("timelapse_mammalian").is_present(manifests[0].parent)
