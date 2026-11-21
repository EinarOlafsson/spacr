"""OPS ph_crops cuts the cell's bounding box plus crop_margin_px, as Measure does (986, D1)."""
from __future__ import annotations

import numpy as np
import pytest

from spacr import ops_engine


def _field(tmp_path, monkeypatch):
    """A 2-channel 40 x 50 field with one 6 x 8 cell, its 2 x 2 nucleus and a bright background."""
    stack = np.full((40, 50, 2), 7.0, np.float32)
    stack[..., 1] = 9.0
    cells = np.zeros((40, 50), np.int32)
    cells[10:16, 20:28] = 3
    stack[10:16, 20:28, 0] = 100.0
    nuclei = np.zeros_like(cells)
    nuclei[12:14, 23:25] = 3
    masks = tmp_path / "masks.npz"
    np.savez(masks, cells=cells, nuclei=nuclei)
    monkeypatch.setattr(ops_engine, "_ph_read_field",
                        lambda files, channels, unreadable, secondary=None: stack.copy())
    return str(masks)


def _task(mask_path, **extra):
    """A one-cell ph_crops task in the native format."""
    return {"site": 1, "files": {}, "channels": ("DAPI", "CY3"),
            "crop_channels": ("DAPI", "CY3"), "mask_path": mask_path,
            "labels": {3: ("linked", 11)}, "size": 12, "flat_folder": None,
            "masks": True, "secondary": None, "plate": "P1", "well": "A1",
            "crop_format": "ome-zarr", **extra}


def test_the_margin_grows_the_bounding_box_and_keeps_the_surround(tmp_path, monkeypatch):
    """Margin 2 gives the 6 x 8 box grown to 10 x 12, with the background kept by default."""
    result = ops_engine._ph_crop_field(_task(_field(tmp_path, monkeypatch), margin=2,
                                             zero_outside=False))
    (_name, (pixels, mask), record), = result["members"]
    assert pixels.shape == (2, 10, 12)
    assert (record["bbox_y0"], record["bbox_x0"], record["bbox_y1"], record["bbox_x1"]) \
        == (8, 18, 18, 30)
    assert record["margin_px"] == 2 and record["window_edge"] == 0
    assert (record["cy"], record["cx"]) == (12, 24)
    assert pixels[0, 0, 0] == 7 and pixels[1, 0, 0] == 9
    assert (pixels[0, 2:8, 2:10] == 100).all()
    assert mask.dtype == np.uint8 and int((mask == 1).sum()) == 48 - 4
    assert int((mask == 2).sum()) == 4 and mask[0, 0] == 0


def test_zero_outside_clears_every_pixel_off_the_cell(tmp_path, monkeypatch):
    """crop_zero_outside keeps the cell's pixels and sets the margin to 0."""
    result = ops_engine._ph_crop_field(_task(_field(tmp_path, monkeypatch), margin=2,
                                             zero_outside=True))
    (_name, (pixels, mask), _record), = result["members"]
    assert (pixels[:, mask == 0] == 0).all()
    assert (pixels[0][mask > 0] == 100).all() and (pixels[1][mask > 0] == 9).all()


def test_the_margin_is_clipped_at_the_field_edge(tmp_path, monkeypatch):
    """A margin past the field edge is clipped and recorded as an edge crop."""
    result = ops_engine._ph_crop_field(_task(_field(tmp_path, monkeypatch), margin=15,
                                             zero_outside=False))
    (_name, (pixels, _mask), record), = result["members"]
    assert (record["bbox_y0"], record["bbox_y1"]) == (0, 31)
    assert (record["bbox_x0"], record["bbox_x1"]) == (5, 43)
    assert pixels.shape == (2, 31, 38) and record["window_edge"] == 1


def test_without_a_margin_the_fixed_square_is_unchanged(tmp_path, monkeypatch):
    """No crop_margin_px keeps the ops_crop_size square centred on the cell, 0 outside it."""
    result = ops_engine._ph_crop_field(_task(_field(tmp_path, monkeypatch)))
    (_name, (pixels, _mask), record), = result["members"]
    assert pixels.shape == (2, 12, 12) and "bbox_y0" not in record
    assert pixels[0, 0, 0] == 0


def test_the_pool_records_the_cut_box(tmp_path, monkeypatch):
    """The native pool stores the bounding box and margin of each crop."""
    pytest.importorskip("zarr")
    from spacr.crop_store import read_commit

    result = ops_engine._ph_crop_field(_task(_field(tmp_path, monkeypatch), margin=2,
                                             zero_outside=False))
    count, _pools, _bad = ops_engine._ph_write_crop_pool(
        str(tmp_path / "root"), "P1", "A1", [result], ("DAPI", "CY3"), 100, {"margin_px": 2})
    assert count == 1
    rows = read_commit(str(tmp_path / "root"), "P1", "A1", "cell")["rows"]
    row = rows.iloc[0] if hasattr(rows, "iloc") else rows[0]
    assert (int(row["bbox_y0"]), int(row["bbox_x0"]), int(row["bbox_y1"]),
            int(row["bbox_x1"]), int(row["margin_px"])) == (8, 18, 18, 30, 2)


def test_ops_defaults_keep_the_fixed_square():
    """The OPS default is no margin, so existing runs cut what they cut before."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["crop_margin_px"] is None
    assert OPS_DEFAULTS["crop_zero_outside"] is False


def test_a_negative_margin_is_refused(tmp_path):
    """A negative crop_margin_px stops the phase before anything is written."""
    with pytest.raises(ValueError, match="crop_margin_px"):
        ops_engine._ph_crops(str(tmp_path / "m.db"), "P1", [], {},
                             {"crop_margin_px": -1, "ops_ph_channels": "DAPI"},
                             str(tmp_path))
