"""Item 1004: OPS Live Align & stitch shows the phenotype field before and after tile-affine placement.

Before is the field at its placement from the seed (or the raster fit); after is
the same field on the same sequencing crop at the finished merge's tile-affine
placement, moved into the preview's frame. Without a finished merge the tab says
how to get the comparison.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr import ops_live  # noqa: E402

from tests.qt.test_ops_live_preview_971 import _fake_result  # noqa: E402

pytestmark = pytest.mark.qt

TILES = {5: (100.0, 200.0)}
CROP = (4, 6, 32)
CENTRE = (100 + 4 + 16, 200 + 6 + 16)


def _ctx(test_root=None):
    return ops_live._Context(settings={}, test_root=test_root, folder=None, well="A1", steps={})


def _field():
    return {"site": 9, "files": {"DAPI": "x"}, "scale": 1.0, "rotation": 0.0,
            "centre": CENTRE, "shape": (32, 32), "source": "seed"}


def _merged_row(shift, frame_offset, groups=None):
    """A tile-affine row of field 9 in a run whose frame is ``frame_offset`` from the preview's."""
    ty = CENTRE[0] - 16 + shift[0] - frame_offset[0]
    tx = CENTRE[1] - 16 + shift[1] - frame_offset[1]
    return pd.Series({"site": 9, "centre_y": CENTRE[0] - frame_offset[0],
                      "centre_x": CENTRE[1] - frame_offset[1], "align_model": "tile_affine",
                      "align_a11": 1.0, "align_a12": 0.0, "align_a21": 0.0, "align_a22": 1.0,
                      "align_ty": ty, "align_tx": tx,
                      "align_tile_transforms": json.dumps(groups or [])})


@pytest.fixture
def dna(monkeypatch):
    from spacr import ops_engine

    plane = np.random.default_rng(1).random((32, 32)).astype(np.float32)
    monkeypatch.setattr(ops_engine, "_ph_channels", lambda settings: ["DAPI"])
    monkeypatch.setattr(ops_live, "_ph_stack", lambda ctx, info, channels: (plane[..., None], ("k",)))
    return plane


def test_the_field_is_read_at_its_tile_affine_placement_in_the_preview_frame(monkeypatch, dna):
    monkeypatch.setattr(ops_live, "_merged_placement_row",
                        lambda ctx, plate, site: _merged_row((2, 3), (500.0, -700.0)))
    after = ops_live._tile_affine_placement(_ctx(), "P", _field(), TILES, 5, CROP,
                                            np.zeros((32, 32), np.float32))
    assert after["site"] == 9 and after["affine"] == "field"
    np.testing.assert_allclose(after["shift"], (2.0, 3.0), atol=1e-9)
    np.testing.assert_allclose(after["phenotype"][2:-1, 3:-1], dna[:-3, :-4], atol=1e-5)


def test_the_crop_takes_its_own_tile_groups_affine(monkeypatch, dna):
    row = _merged_row((2, 3), (0.0, 0.0))
    group = [5, 40, 1.0, 0.0, 0.0, 1.0, row["align_ty"] - 2, row["align_tx"] - 3]
    row["align_tile_transforms"] = json.dumps([[4, 30, 1, 0, 0, 1, 0, 0], group])
    monkeypatch.setattr(ops_live, "_merged_placement_row", lambda ctx, plate, site: row)
    after = ops_live._tile_affine_placement(_ctx(), "P", _field(), TILES, 5, CROP,
                                            np.zeros((32, 32), np.float32))
    assert after["affine"] == "tile"
    np.testing.assert_allclose(after["shift"], (0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(after["phenotype"][:-1, :-1], dna[:-1, :-1], atol=1e-5)


def test_without_a_finished_merge_there_is_no_after_picture(monkeypatch, dna):
    monkeypatch.setattr(ops_live, "_merged_placement_row", lambda ctx, plate, site: None)
    assert ops_live._tile_affine_placement(_ctx(), "P", _field(), TILES, 5, CROP,
                                           np.zeros((32, 32), np.float32)) is None
    assert ops_live._tile_affine_placement(_ctx(), "P", None, TILES, 5, CROP, dna) is None


def test_the_test_data_reads_its_stored_tile_affine_merge(tmp_path):
    folder = tmp_path / "reference" / "merge" / "tile_affine"
    folder.mkdir(parents=True)
    rows = pd.DataFrame([dict(_merged_row((2, 3), (0, 0))), dict(_merged_row((0, 0), (0, 0)), site=4)])
    rows.loc[1, "align_model"] = None
    rows.to_parquet(folder / "ops_phenotype.parquet")
    ctx = _ctx(test_root=tmp_path)
    assert ops_live._merged_placement_row(ctx, "P", 9)["align_ty"] == rows.loc[0, "align_ty"]
    assert ops_live._merged_placement_row(ctx, "P", 4) is None
    assert ops_live._merged_placement_row(_ctx(), "P", 9) is None


def _align_result(after):
    rng = np.random.default_rng(2)
    result = dict(_fake_result("align"))
    result["placement"] = {"site": 7, "scale": 0.25, "rotation": 0.5, "centre": (24.0, 40.0),
                           "source": "seed", "sbs": rng.random((32, 32)),
                           "phenotype": rng.random((32, 32))}
    result["placed_after"] = after and {"site": 7, "affine": "tile", "shift": (1.5, -2.0),
                                        "sbs": rng.random((32, 32)), "phenotype": rng.random((32, 32))}
    return result


def test_the_align_tab_draws_the_field_before_and_after_placement(qapp):
    from spacr.qt.widgets.ops_live_preview import _DRAW, _theme_colours

    drawn = _DRAW["align"](_align_result(True), _theme_colours())
    titles = [title for title, _picture in drawn["images"]]
    before = next(i for i, t in enumerate(titles) if "before tile-affine placement" in t)
    assert "at the placement seed" in titles[before]
    assert titles[before + 1] == ("The same field after tile-affine placement "
                                  "(tile group affine; +1.5, -2.0 px)")
    assert drawn["images"][before][1].shape == drawn["images"][before + 1][1].shape
    assert any(row[0] == "Field 7" and row[1:3] == ["+1.5", "-2.0"] for row in drawn["rows"])


def test_without_a_merge_the_tab_says_how_to_compare(qapp):
    from spacr.qt.widgets.ops_live_preview import _DRAW, _theme_colours

    drawn = _DRAW["align"](_align_result(False), _theme_colours())
    titles = [title for title, _picture in drawn["images"]]
    assert not any("tile-affine" in t for t in titles)
    assert any("tile_affine" in row[3] for row in drawn["rows"])
