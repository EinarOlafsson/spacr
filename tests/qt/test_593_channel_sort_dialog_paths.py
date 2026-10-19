"""The example-set check's thumbnails, offscreen."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import channel_sort_dialog as csd

pytestmark = pytest.mark.qt


def _tif(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _drawn(root: Path, masks=True, shape=(12, 12)) -> Path:
    folder = root / "drawn"
    for i in range(1, 4):
        for stain, value in (("nuc", 10), ("cell", 100)):
            name = f"{stain}_img{i:02d}.tif"
            _tif(folder / name, np.full(shape, value + i, np.uint16))
            if masks:
                _tif(folder / "masks" / name, np.full(shape, i, np.uint16))
    return folder


def _names(folder: Path):
    return sorted(n for n in os.listdir(folder) if n.endswith(".tif"))


def test_a_masked_icon_is_tinted_and_a_blank_one_is_grey(qt_theme_applied):
    grey = np.full((4, 4), 100, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 255
    image = csd._grey_icon(grey, mask).toImage()
    assert image.pixelColor(0, 0).red() == 200
    assert image.pixelColor(0, 0).green() == 50
    assert image.pixelColor(3, 3).red() == 100
    assert csd._grey_icon(None, size=8).width() == 8


def test_the_thumbnail_worker_reads_every_image_and_stops_when_asked(
        qt_theme_applied, tmp_path):
    folder = _drawn(tmp_path)
    worker = csd._ThumbWorker(str(folder), _names(folder))
    got = []
    worker.ready.connect(lambda name, image, mask: got.append(
        (name, image is not None, mask is not None)))
    worker.run()
    assert got == [(name, True, True) for name in _names(folder)]
    stopped = csd._ThumbWorker(str(folder), _names(folder))
    stopped.stop()
    seen = []
    stopped.ready.connect(lambda *args: seen.append(args))
    stopped.run()
    assert seen == []


def test_the_worker_survives_a_file_it_cannot_read(qt_theme_applied, tmp_path,
                                                   monkeypatch):
    folder = _drawn(tmp_path, masks=False)

    def broken(*_args):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(csd.cs, "mask_for", broken)
    worker = csd._ThumbWorker(str(folder), _names(folder)[:1])
    got = []
    worker.ready.connect(lambda name, image, mask: got.append((image, mask)))
    worker.run()
    assert got == [(None, None)]
