"""712: the timelapse movies on Make Masks' "Load test data" menu.

No network and no GPU: a tiny synthetic CTC-style movie is built with the
real build tool, and the download is replaced by one that copies it.
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QMessageBox

from spacr.qt import make_masks_demo as demo
from spacr.qt.i18n import tr
from spacr.qt.screens.make_masks import MakeMasksScreen

ROOT = Path(__file__).resolve().parents[2]
CAPTIONS = ("Timelapse: yeast", "Timelapse: bacteria", "Timelapse: mammalian")
LOCALES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


@pytest.fixture
def screen(qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.Discard)
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    yield widget
    stage = widget._mask_stage
    qtbot.waitUntil(lambda: stage._worker.idle() and stage._mesher.idle(), timeout=5000)
    widget._lineage_dirty = False


@pytest.fixture(scope="module")
def movie(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "build_timelapse_example_datasets",
        ROOT / "tools" / "build_timelapse_example_datasets.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tool
    spec.loader.exec_module(tool)
    source = tmp_path_factory.mktemp("timelapse_source")
    rng = np.random.default_rng(7120)
    for t in range(4):
        labels = np.zeros((20, 24), dtype=np.uint16)
        if t < 2:
            labels[2:8, 2:10] = 1
        else:
            labels[2:8, 2:5] = 2
            labels[2:8, 8:11] = 3
        labels[14:18, 3 + 2 * t:7 + 2 * t] = 9
        tifffile.imwrite(source / f"man_track{t:03d}.tif", labels)
        tifffile.imwrite(source / f"t{t:03d}.tif",
                         rng.integers(0, 3000, labels.shape, dtype=np.uint16))
    with (source / "lineage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "t_start", "t_end", "parent"])
        writer.writerows([(1, 0, 1, 0), (2, 2, 3, 1), (3, 2, 3, 1), (9, 0, 3, 0)])
    tool.MOVIES = tuple(dataclasses.replace(spec, window=(0, 4))
                        for spec in tool.MOVIES)
    dest = tmp_path_factory.mktemp("timelapse_built")
    tool.build(dest, {"timelapse_mammalian": source})
    return dest / "timelapse_mammalian" / "make_masks_timelapse_mammalian"


def test_the_menu_offers_the_three_timelapse_movies(screen):
    actions = screen._test_data_actions
    for kind, caption in zip(demo.TIMELAPSE_EXAMPLE_REPOS, CAPTIONS):
        action = actions[f"timelapse_{kind}"]
        assert action.text() == tr(caption)
        assert action.objectName() == (
            f"MakeMasksTimelapse{kind.capitalize()}TestData")
        assert "CC BY 4.0" in action.toolTip()


def test_the_captions_are_translated_in_every_language():
    for caption in CAPTIONS:
        for code in LOCALES:
            translated = tr(caption, language=code)
            assert translated and translated != caption, (caption, code)


def test_the_movie_opens_with_masks_and_lineage_and_keeps_the_truth(screen, movie):
    truth = movie / "ground_truth_masks" / "t98g_t002.tif"
    published = truth.read_bytes()
    assert demo.load_the_timelapse_test_data(screen, "mammalian", folder=movie)
    assert screen._masks_dir == str(movie / "masks")
    assert sorted(path.name for path in (movie / "masks").iterdir()) == [
        f"t98g_t{t:03d}.tif" for t in range(4)]
    assert screen._image_files[0] == "t98g_t000.tif"
    assert screen._canvas.mask is not None and screen._canvas.mask.max() == 9
    assert screen._lineage_format and not screen._lineage_dirty
    cells = screen._lineage["cells"]
    divided = [cell for cell in cells.values() if cell["children"]]
    assert [cell["name"] for cell in divided] == ["1"]
    assert sorted(cells[child]["name"] for child in divided[0]["children"]) == ["1a", "1b"]
    assert truth.read_bytes() == published


def test_a_missing_movie_is_downloaded_then_opened(screen, movie, tmp_path):
    target = tmp_path / "cache" / "make_masks_timelapse_mammalian"
    seen = {}

    def fake_download(parent, dest, on_done):
        seen["dest"] = dest
        assert not screen._btn_test_data.isEnabled()
        shutil.copytree(movie, dest, ignore=shutil.ignore_patterns("masks"))
        on_done(object(), "")

    assert not demo.load_the_timelapse_test_data(
        screen, "mammalian", ask=fake_download, folder=target)
    assert seen["dest"] == target
    assert screen._btn_test_data.isEnabled()
    assert screen._folder == str(target)
    assert any(cell["children"] for cell in screen._lineage["cells"].values())


def test_the_download_worker_targets_each_movie_repository(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(demo, "download_toxo_mito_demo",
                        lambda parent, dest, on_done, **kwargs: seen.append(kwargs))
    for kind, repo in demo.TIMELAPSE_EXAMPLE_REPOS.items():
        demo.download_timelapse_example(None, tmp_path, lambda *args: None, kind)
        assert seen[-1]["worker_factory"].repo == repo
        assert seen[-1]["worker_factory"].archive == (
            f"spacr-example-timelapse-{kind}.tar")
        assert demo.timelapse_example_folder(kind).name == (
            f"make_masks_timelapse_{kind}")
    with pytest.raises(KeyError):
        demo.timelapse_example_folder("plants")
