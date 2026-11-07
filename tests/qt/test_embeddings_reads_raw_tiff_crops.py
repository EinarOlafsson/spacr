"""The Embeddings screen reads a folder of raw 16-bit, multichannel TIFF crops.

A folder with no PNG and with TIFFs is read as stored: uint16 values and
every plane, planes-first stacks turned channels-last. The vectors the screen
computes from it are the ones ``embed_array`` gives for the same pixels as
float32, so nothing was narrowed to eight bits on the way.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

pytestmark = pytest.mark.qt


def _encoder(stack: np.ndarray) -> np.ndarray:
    """A deterministic stand-in backbone: per-plane mean and maximum."""
    flat = stack.reshape(stack.shape[0], -1, stack.shape[-1])
    return np.concatenate([flat.mean(1), flat.max(1)], axis=1).astype(np.float32)


_encoder.in_channels = None


@pytest.fixture
def tiff_folder(tmp_path):
    """Five 4 x 32 x 32 uint16 crops, planes first, with values above 255."""
    folder = tmp_path / "raw_16bit"
    folder.mkdir()
    rng = np.random.default_rng(560)
    crops = rng.integers(0, 60000, size=(5, 4, 32, 32)).astype(np.uint16)
    for index, crop in enumerate(crops):
        tifffile.imwrite(folder / f"crop_{index}.tif", crop,
                         photometric="minisblack", compression="zlib")
    return folder, np.moveaxis(crops, 1, -1)


def test_a_tiff_folder_loads_raw(tiff_folder):
    from spacr.crop_loader import CROP_SOURCE_FOLDER, CropQuery, load_crops, plan_crops

    folder, expected = tiff_folder
    plan = plan_crops(CropQuery(source=CROP_SOURCE_FOLDER, path=str(folder)))
    record = {}
    crops = load_crops(plan, record=record)

    assert crops.dtype == np.uint16
    assert np.array_equal(crops, expected)
    assert "TIFFs" in plan.source_label and "read raw" in plan.source_label
    assert record["crop_shape"] == [32, 32, 4]


def test_pngs_still_win_in_a_mixed_folder(tiff_folder):
    from PIL import Image

    from spacr.crop_loader import CROP_SOURCE_FOLDER, CropQuery, plan_crops

    folder, _ = tiff_folder
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(folder / "a.png")

    plan = plan_crops(CropQuery(source=CROP_SOURCE_FOLDER, path=str(folder)))

    assert plan.count == 1 and plan.rows[0].endswith("a.png")


def test_an_empty_folder_names_both_formats(tmp_path):
    from spacr.crop_loader import CROP_SOURCE_FOLDER, CropQuery, plan_crops

    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    plan = plan_crops(CropQuery(source=CROP_SOURCE_FOLDER, path=str(tmp_path)))

    assert plan.is_empty
    assert "no .png or .tif" in plan.empty_reason


def test_the_screen_embeds_a_16bit_tiff_like_the_float_input(qtbot, monkeypatch,
                                                            tiff_folder):
    from spacr import embeddings
    from spacr.crop_loader import CROP_SOURCE_FOLDER
    from dataclasses import replace

    from spacr.embeddings import embed_array
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    folder, expected = tiff_folder
    monkeypatch.setattr(embeddings, "_backbone_encoder", lambda spec: _encoder)
    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    screen._where.setCurrentIndex(screen._where.findData(CROP_SOURCE_FOLDER))
    screen._path.setText(str(folder))
    screen._foundation.setCurrentIndex(screen._foundation.findData(""))

    screen.load_crops()
    qtbot.waitUntil(lambda: getattr(screen, "_crops", None) is not None,
                    timeout=10000)
    assert screen._crops.dtype == np.uint16
    screen.embed()
    qtbot.waitUntil(lambda: screen._result is not None, timeout=10000)

    used = screen._result.spec
    reference = embed_array(expected.astype(np.float32), used, encoder=_encoder)
    assert np.allclose(screen._result.values, reference.values)
    scale = embed_array(expected.astype(np.float32),
                        replace(used, channel_scale=None),
                        encoder=_encoder).spec.channel_scale
    assert np.allclose(used.channel_scale, scale)
    assert max(used.channel_scale) > 255
