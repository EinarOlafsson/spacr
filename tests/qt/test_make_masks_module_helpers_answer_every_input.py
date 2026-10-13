"""Make Masks' module-level helpers answer every input they can be handed.

Pins the answers of the helpers the magnifier, the detect buttons and the
Cellpose loader lean on, for the inputs the everyday tests never send:

* a CPU Cellpose whose constructor cannot be introspected is still loaded,
  just without the float32 switch;
* the tile count survives a Cellpose without ``get_pad_yx``; a broken spec
  lookup reads as "Cellpose not installed";
* an unknown magnifier mode falls back to Otsu and says so;
* an empty box stretches to an empty picture; a wide-integer field is drawn
  through the slow stretch, identical to the fast one's recipe;
* an object's window is found by scanning when the worker did not count
  extents, and a missing id has none;
* the newest-request worker runs a pinned request ahead of the waiting one,
  keeps going when delivering a result raises, and is idle again when its
  job dies with a BaseException.
"""
from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

# ---------------------------------------------------------------------------
# Cellpose loading and counting
# ---------------------------------------------------------------------------

def test_a_cpu_cellpose_that_cannot_be_introspected_is_still_loaded(
        monkeypatch):
    pytest.importorskip("cellpose")
    from cellpose import models as cp_models

    import spacr.accelerator as accelerator
    import spacr.utils as utils

    made = []

    class Opaque:
        __signature__ = "not a signature"

        def __init__(self, **kwargs):
            made.append(kwargs)

    monkeypatch.setattr(cp_models, "CellposeModel", Opaque)
    monkeypatch.setattr(utils, "_resolve_cellpose_pretrained",
                        lambda name: "/weights/" + name)
    monkeypatch.setattr(accelerator, "cellpose_kwargs", lambda: {"gpu": False})

    model = mm.load_cellpose_model("cpsam")
    assert isinstance(model, Opaque)
    assert made == [{"pretrained_model": "/weights/cpsam", "gpu": False}]


def test_the_tile_count_survives_a_cellpose_without_its_padding_helper(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose.transforms", None)
    assert mm._cellpose_tile_count((200, 200)) == 1
    assert mm._cellpose_tile_count((512, 512)) == 9
    assert mm._cellpose_tile_count((512, 512), diameter=60) == 1


def test_a_spec_lookup_that_raises_reads_as_not_installed(monkeypatch):
    def broken(name):
        raise ValueError("cellpose.__spec__ is None")

    monkeypatch.setattr(mm, "find_spec", broken)
    assert mm._cellpose_installed() is False


# ---------------------------------------------------------------------------
# The magnifier's region helpers
# ---------------------------------------------------------------------------

def _request(crop, mode):
    return mm._MagnifierRequest(
        key=("k",), crop=crop, box=(0, 0, crop.shape[1], crop.shape[0]),
        shape=crop.shape, mode=mode, sensitivity=1.0, bright=True,
        min_area=0, model_name="", diameter=0, colour=(255, 0, 0))


def test_an_unknown_magnifier_mode_falls_back_to_otsu_and_says_so():
    crop = np.zeros((40, 40), dtype=np.uint16)
    crop[10:30, 10:30] = 40000
    labels, used, note = mm._segment_region(_request(crop, "no-such-mode"))
    assert used == "otsu"
    assert note == "no magnifier mode is called 'no-such-mode'"
    assert labels[20, 20] > 0 and labels[2, 2] == 0


def test_an_empty_box_stretches_to_an_empty_picture():
    image = np.arange(100, dtype=np.uint16).reshape(10, 10)
    out = mm._stretch_for_box(image, (4, 4, 4, 9), (4, 4, 4, 9), 1.0, 99.0)
    assert out.size == 0 and out.flags["C_CONTIGUOUS"]


def test_a_wide_integer_field_is_drawn_through_the_slow_stretch():
    image = (np.arange(64, dtype=np.uint32).reshape(8, 8) * 70000)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    box = part = (0, 0, 8, 8)
    out = mm._box_picture(image, box, part, mask, 0.0, 100.0)
    stretched = mm._stretch_for_box(image, box, part, 0.0, 100.0)
    expected = mm._rgb32(engine.overlay_mask(stretched, mask, alpha=0.5))
    assert out.dtype == np.uint32 and out.shape == (8, 8)
    np.testing.assert_array_equal(out, expected)
    assert out[0, 0] != out[7, 7], "the stretch spans the field"


def _whole_image_result(labels, extents=None):
    request = _request(np.zeros(labels.shape, np.uint16), "otsu")
    return mm._MagnifierResult(request=request, labels=labels, mode="otsu",
                               note="", overlay=None, extents=extents)


def test_an_objects_window_is_found_by_scanning_without_counted_extents():
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[4:9, 11:17] = 3
    result = _whole_image_result(labels)
    assert mm._object_window(result, 3) == (11, 4, 17, 9)
    assert mm._object_window(result, 5) is None


def test_an_id_past_the_counted_extents_is_found_by_scanning():
    labels = np.zeros((20, 30), dtype=np.int32)
    labels[1:3, 2:4] = 1
    labels[10:12, 20:25] = 2
    result = _whole_image_result(labels, extents=mm._object_extents(
        labels[:5]))
    assert mm._object_window(result, 2) == (20, 10, 25, 12)


# ---------------------------------------------------------------------------
# The newest-request worker
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, key):
        self.key = key


def test_a_pinned_request_runs_ahead_of_the_one_waiting():
    gate, started = threading.Event(), threading.Event()
    ran, delivered = [], threading.Event()
    done = []

    def work(request):
        if request.key == "first":
            started.set()
            assert gate.wait(5)
        ran.append(request.key)
        return request.key

    def deliver(request, result, error):
        done.append(result)
        if len(done) == 3:
            delivered.set()

    worker = mm._NewestRequestWorker(work, deliver, name="test-worker")
    worker.submit(_Job("first"))
    assert started.wait(5)
    worker.submit(_Job("waiting"))
    worker.submit(_Job("clicked"), pin=True)
    gate.set()
    assert delivered.wait(5)
    assert ran == ["first", "clicked", "waiting"]
    assert worker.close()


def test_a_result_that_cannot_be_delivered_does_not_stop_the_worker():
    delivered = threading.Event()
    gate, started = threading.Event(), threading.Event()
    seen = []

    def deliver(request, result, error):
        seen.append(result)
        if result == "one":
            raise RuntimeError("the window is gone")
        delivered.set()

    def work(request):
        if request.key == "one":
            started.set()
            assert gate.wait(5)
        return request.key

    worker = mm._NewestRequestWorker(work, deliver)
    worker.submit(_Job("one"))
    assert started.wait(5)
    worker.submit(_Job("two"))
    gate.set()
    assert delivered.wait(5)
    assert seen == ["one", "two"]
    assert worker.close()


class _Fatal(BaseException):
    pass


def test_a_worker_whose_job_dies_is_idle_again_and_takes_new_work(
        monkeypatch):
    caught = []
    monkeypatch.setattr(threading, "excepthook",
                        lambda args: caught.append(args.exc_type))
    delivered = threading.Event()

    def work(request):
        if request.key == "fatal":
            raise _Fatal()
        return request.key

    worker = mm._NewestRequestWorker(
        work, lambda request, result, error: delivered.set())
    worker.submit(_Job("fatal"))
    for _ in range(250):
        if worker.idle():
            break
        threading.Event().wait(0.02)
    assert worker.idle()
    assert caught == [_Fatal]
    worker.submit(_Job("after"))
    assert delivered.wait(5)
    assert worker.close()
