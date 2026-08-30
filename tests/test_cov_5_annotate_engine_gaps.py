"""Reaching the crops when the joined measurements do not carry their path.

``png_list`` and the measurement tables are joined on ``prcfo``, and the join
does not always bring ``png_path`` with it. When it does not, the crop path
has to be merged back on — one-to-one, because a repeated key would show the
same cell several times under different crops — and when it cannot be, the
answer is no rows rather than a grid of objects nobody can open.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image                                          # noqa: E402

from spacr.qt import annotate_engine as AE                     # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def db(tmp_path):
    """A file that exists, since every entry point stats it first."""
    path = tmp_path / "measurements.db"
    path.write_bytes(b"")
    return str(path)


def _measurements():
    """A joined frame indexed by prcfo, with no crop path on it."""
    frame = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                          "cell_area": [50.0, 500.0, 900.0]})
    return frame.set_index("prcfo")


def _png_list():
    return pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                         "png_path": ["/crops/o1.png", "/crops/o2.png",
                                      "/crops/o3.png"]}).set_index("prcfo")


def _patch_readers(monkeypatch, joined, png_list):
    import spacr.io as io

    monkeypatch.setattr(io, "_read_and_join_tables",
                        lambda *_a, **_k: joined.copy())
    monkeypatch.setattr(io, "_read_db",
                        lambda *_a, **_k: [png_list.copy()])


# ---------------------------------------------------------------------------
# Channels and borders
# ---------------------------------------------------------------------------

def test_a_channel_left_out_is_blacked_out_not_left_alone():
    """The grid shows the stains asked for; a stray one misreads the crop."""
    image = Image.new("RGB", (4, 4), (200, 150, 100))

    only_green = AE.filter_channels_pil(image, ["g"])

    assert only_green.getpixel((0, 0)) == (0, 150, 0)
    assert AE.filter_channels_pil(image, ["r"]).getpixel((0, 0)) == (200, 0, 0)
    assert AE.filter_channels_pil(image, ["b"]).getpixel((0, 0)) == (0, 0, 100)
    assert AE.filter_channels_pil(image, ["r", "g", "b"]).getpixel((0, 0)) == (
        200, 150, 100)
    assert AE.filter_channels_pil(image, None).getpixel((0, 0)) == (
        200, 150, 100)


def test_a_bordered_crop_keeps_its_own_pixels_inside_the_frame():
    """The border is drawn around the crop, never over it."""
    image = Image.new("RGB", (4, 4), (10, 20, 30))

    bordered = AE.add_colored_border(image, 2, "red")

    assert bordered.size == (8, 8)
    assert bordered.getpixel((4, 4)) == (10, 20, 30), "the crop was painted over"
    assert bordered.getpixel((3, 0)) == (255, 0, 0)
    assert bordered.getpixel((0, 3)) == (255, 0, 0)


# ---------------------------------------------------------------------------
# The annotation column
# ---------------------------------------------------------------------------

def test_no_column_and_no_database_are_both_nothing_to_do(tmp_path):
    """Called on every page load, so it must be cheap and silent when idle."""
    absent = tmp_path / "absent.db"

    outcomes = (
        AE.ensure_annotation_column("", "test"),
        AE.ensure_annotation_column(str(absent), "test"),
    )

    assert outcomes == (None, None)
    assert not absent.exists(), "checking must not create an empty database"


# ---------------------------------------------------------------------------
# Threshold-filtered crops
# ---------------------------------------------------------------------------

def test_the_crop_path_is_merged_back_on_when_the_join_dropped_it(db,
                                                                   monkeypatch):
    _patch_readers(monkeypatch, _measurements(), _png_list())

    rows = AE.fetch_filtered_paths(db, "test", ["cell_area"], [100.0],
                                   ["higher"])

    assert [path for path, _annotation in rows] == ["/crops/o2.png",
                                                    "/crops/o3.png"]
    assert all(annotation is None for _path, annotation in rows)


def test_an_image_type_narrows_the_crops_after_the_merge(db, monkeypatch):
    _patch_readers(monkeypatch, _measurements(), _png_list())

    rows = AE.fetch_filtered_paths(db, "test", ["cell_area"], [10.0],
                                   ["higher"], image_type="o3")

    assert [path for path, _a in rows] == ["/crops/o3.png"]


def test_a_png_list_that_cannot_be_joined_yields_no_crops(db, monkeypatch):
    """Guessing a row order would open crops of the wrong objects."""
    unjoinable = pd.DataFrame({"png_path": ["/crops/o1.png"]})
    _patch_readers(monkeypatch, _measurements(), unjoinable)

    assert AE.fetch_filtered_paths(db, "test", ["cell_area"], [10.0],
                                   ["higher"]) == []


def test_a_database_that_is_not_there_yields_no_crops(tmp_path):
    assert AE.fetch_filtered_paths(str(tmp_path / "absent.db"), "test",
                                   ["cell_area"], [1.0], ["higher"]) == []
    assert AE.fetch_filtered_paths(str(tmp_path), "test", [], [], []) == []


# ---------------------------------------------------------------------------
# Gate-filtered crops
# ---------------------------------------------------------------------------

def _gate(low):
    from spacr.qt.widgets.gate_spec import ThresholdGate

    return ThresholdGate(name="big cells", column="cell_area", low=low)


def test_a_gated_population_reaches_its_crops_through_the_same_merge(
        db, monkeypatch):
    """A population gated on screen and one annotated from it must match."""
    _patch_readers(monkeypatch, _measurements(), _png_list())

    paths = AE.gate_paths(db, [_gate(100.0)])

    assert paths == ["/crops/o2.png", "/crops/o3.png"]


def test_gating_with_no_gates_selects_nothing(db):
    assert AE.gate_paths(db, []) == []


def test_a_gated_population_with_no_reachable_crops_is_empty(db, monkeypatch):
    _patch_readers(monkeypatch, _measurements(),
                   pd.DataFrame({"png_path": ["/crops/o1.png"]}))

    assert AE.gate_paths(db, [_gate(10.0)]) == []


# ---------------------------------------------------------------------------
# The outline model
# ---------------------------------------------------------------------------

def test_without_torch_the_outline_model_is_built_on_the_cpu(monkeypatch):
    """A machine with no torch import must still get outlines, not a crash."""
    built = {}

    class _FakeModel:
        def __init__(self, **kwargs):
            built.update(kwargs)

    fake_cellpose = types.ModuleType("cellpose")
    fake_models = types.ModuleType("cellpose.models")
    fake_models.CellposeModel = _FakeModel
    fake_cellpose.models = fake_models
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(AE, "_cellpose_outline_model", None, raising=False)

    try:
        model = AE._get_cellpose_outline_model()
        assert isinstance(model, _FakeModel)
        assert built["gpu"] is False
        assert built["pretrained_model"] == "cpsam"
        assert AE._get_cellpose_outline_model() is model, "not cached"
    finally:
        AE._cellpose_outline_model = None


# ---------------------------------------------------------------------------
# The writer thread
# ---------------------------------------------------------------------------

def test_a_writer_told_to_stop_while_idle_leaves_no_thread_behind(tmp_path):
    """``stop`` sets the flag before it queues the sentinel.

    The writer polls with a timeout, so it can wake on an empty queue with
    the flag already set and the sentinel not yet posted. It has to leave on
    the flag alone — a writer that only exits on the sentinel keeps a live
    SQLite connection open while the interpreter finalises the extension.
    """
    import sqlite3

    path = tmp_path / "measurements.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE png_list (png_path TEXT, test INTEGER)")
    connection.commit()
    connection.close()

    writer = AE.SaveWorker(str(path), "test")
    writer.start()
    assert writer.is_alive is True

    writer._terminate = True          # the flag, without the sentinel
    deadline = time.time() + 5.0
    while writer.is_alive and time.time() < deadline:
        time.sleep(0.02)

    assert writer.is_alive is False, "the writer outlived its terminate flag"
    writer.stop(wait=True)
