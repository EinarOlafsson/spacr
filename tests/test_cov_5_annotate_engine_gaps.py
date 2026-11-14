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
import time
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PIL import Image                                          # noqa: E402

from spacr.qt import annotate_engine as AE                     # noqa: E402

pytestmark = pytest.mark.qt



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


# ---------------------------------------------------------------------------
# Gate-filtered crops
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# reading the numbers a filter box holds
# ---------------------------------------------------------------------------

def test_a_bound_that_is_not_a_finite_number_is_no_bound_at_all():
    """NaN reaches here from an empty spin box and from a float('nan') setting.

    ``None`` means "no limit". NaN would be carried into the comparison
    instead, and every comparison with NaN is False -- so a NaN lower bound
    silently rejects EVERY object rather than accepting them all, and an
    annotator shows an empty grid with no explanation.
    """
    from spacr.qt.annotate_engine import filter_bound

    assert filter_bound(float("nan")) is None
    assert filter_bound(None) is None
    assert filter_bound("") is None
    assert filter_bound("not a number") is None
    # ...and a real bound survives, so the four refusals mean something.
    assert filter_bound("12.5") == 12.5
    assert filter_bound(3) == 3.0


def test_a_filter_for_a_channel_that_does_not_exist_is_ignored():
    """Settings outlive the form that wrote them.

    A saved filter naming a channel this run has not got must be dropped
    rather than added -- an unknown key in the bounds map is read later as a
    filter nothing can satisfy.
    """
    from spacr.qt.annotate_engine import (empty_object_filters,
                                          normalize_object_filters)

    bounds = normalize_object_filters(
        {"a_channel_that_was_never_defined": (1, 2)})

    assert bounds == empty_object_filters()


def test_a_filter_whose_value_is_not_a_pair_is_ignored():
    """A settings file can hold a single number where a pair belongs.

    Unpacking it raises, and the honest response is to leave that filter
    unset -- taking the run down because one saved bound was written by an
    older spaCR would make the settings file a liability.
    """
    from spacr.qt.annotate_engine import (FILTER_CHANNELS, filter_key,
                                          normalize_object_filters)

    key = filter_key(FILTER_CHANNELS[0], "area")

    for broken in (5, None, (1, 2, 3), object()):
        bounds = normalize_object_filters({key: broken})
        assert bounds[key] == (None, None), f"{broken!r} was accepted"

    # NOT a refusal, and worth pinning so nobody "fixes" it by accident: a
    # two-character string DOES unpack, so "12" is read as the pair
    # ("1", "2"). It is legal Python and it is what the settings file would
    # have to contain for it to matter.
    assert normalize_object_filters({key: "12"})[key] == (1.0, 2.0)

    # A real pair still lands.
    bounds = normalize_object_filters({key: (10, 200)})
    assert bounds[key] == (10.0, 200.0)


# ---------------------------------------------------------------------------
# the image-type expression
# ---------------------------------------------------------------------------
