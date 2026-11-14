"""Annotate's off-GUI-thread pieces: page and retrain workers, crop decoding, totals.

Everything here runs on a worker in the real screen, so each unit is driven
directly with its inputs and its output is asserted exactly: the signal
payloads a worker emits (or that it emits nothing to a screen that is gone),
the path a stale crop location is re-anchored to, the pixels of a thumbnail
or its placeholder, the arguments the outline and queue engines are called
with, and the dict ``_compute_total`` hands back to the screen.

A real on-disk experiment (8 PNG crops plus ``measurements/measurements.db``)
backs the cases that read files, because a moved dataset is about what is
and is not on disk.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import shiboken6
from PIL import Image
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QWidget

from spacr.qt import annotate_engine as engine
from spacr.qt.screens import annotate as annotate_mod

pytestmark = pytest.mark.qt


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """A small experiment folder: 8 crops and a png_list that names them."""
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "data" / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    paths = []
    for i in range(8):
        arr = rng.integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
        p = src / "data" / "images" / f"cell_{i:02d}.png"
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    db = src / "measurements" / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    engine.ensure_annotation_column(str(db), "annotate")
    return src


def _settings_for(experiment: Path) -> engine.AnnotateSettings:
    s = engine.AnnotateSettings()
    s.src = str(experiment)
    s.db_path = str(experiment / "measurements" / "measurements.db")
    s.image_size = (16, 16)
    return s


def _raises(exc):
    def _raise(*_args, **_kwargs):
        raise exc
    return _raise


# ---------------------------------------------------------------------------
# _PageLoadWorker
# ---------------------------------------------------------------------------

class _NoSignature:
    """A loader whose signature cannot be read."""

    __signature__ = "not a signature"

    def __call__(self, row):
        return (row[0] + "!", row[1])


def test_the_page_worker_reports_each_page_once_or_not_at_all(qapp, subtests):
    """A page comes back whole, empty on a decode error, or never once abandoned."""
    def stoppable(row, should_stop):
        return (row[0], should_stop())

    class _GoneAtEmit(annotate_mod._PageLoadWorker):
        def isInterruptionRequested(self):
            raise RuntimeError("Internal C++ object already deleted")

    class _Interrupted(annotate_mod._PageLoadWorker):
        def isInterruptionRequested(self):
            return True

    rows = [("a.png", 1), ("b.png", 2)]
    cases = [
        ("every decoded crop is handed back", annotate_mod._PageLoadWorker,
         3, rows, lambda row: (row[0].upper(), row[1]),
         [(3, [("A.PNG", 1), ("B.PNG", 2)])]),
        ("a ValueError delivers an empty page", annotate_mod._PageLoadWorker,
         5, [("a.png", None)], _raises(ValueError("not an image")), [(5, [])]),
        ("an OSError delivers an empty page", annotate_mod._PageLoadWorker,
         7, [("a.png", None)], _raises(OSError("truncated PNG")), [(7, [])]),
        ("a loader that can stop is handed the stop flag",
         annotate_mod._PageLoadWorker, 2, rows, stoppable,
         [(2, [("a.png", False), ("b.png", False)])]),
        ("a loader with no readable signature is called plainly",
         annotate_mod._PageLoadWorker, 4, rows, _NoSignature(),
         [(4, [("a.png!", 1), ("b.png!", 2)])]),
        ("a cancelled outline abandons the page", annotate_mod._PageLoadWorker,
         6, rows, _raises(annotate_mod.OutlineCancelled("stop")), []),
        ("an interrupted page reports nothing", _Interrupted, 8, rows,
         lambda row: row, []),
        ("a worker whose C++ half is gone reports nothing", _GoneAtEmit, 9,
         [], lambda row: row, []),
    ]
    for case_id, cls, gen, paths, load_fn, expected in cases:
        with subtests.test(msg=case_id):
            worker = cls(gen, paths, load_fn)
            seen = []
            worker.done.connect(lambda g, loaded: seen.append((g, loaded)))
            worker.run()
            assert seen == expected


# ---------------------------------------------------------------------------
# _RetrainWorker
# ---------------------------------------------------------------------------

def test_the_retrain_worker_hands_back_its_round_or_says_why_not(
        qapp, monkeypatch, subtests):
    """A round reaches a live screen; failures are named; a dead screen gets nothing."""
    from spacr import active_learning as al

    class _Interrupted(annotate_mod._RetrainWorker):
        def isInterruptionRequested(self):
            return True

    def run(cls=annotate_mod._RetrainWorker, delete=False, options=None):
        given = {"round_index": 4} if options is None else options
        worker = cls("db", "annotate", given)
        given["round_index"] = 99          # edited after the worker started
        done, failed = [], []
        worker.done.connect(done.append)
        worker.failed.connect(failed.append)
        if delete:
            shiboken6.delete(worker)
        worker.run()
        return done, failed

    seen = []
    with subtests.test(msg="a finished round is handed back with a copy of the options"):
        monkeypatch.setattr(al, "retrain_round",
                            lambda db, column, **options: seen.append(
                                (db, column, options)) or "round-result")
        assert run() == (["round-result"], [])
        assert seen == [("db", "annotate", {"round_index": 4})]

    with subtests.test(msg="an interrupted round is not handed back"):
        assert run(_Interrupted) == ([], [])

    with subtests.test(msg="a finished round for a dead screen is dropped"):
        assert run(delete=True) == ([], [])

    with subtests.test(msg="a failure names the exception and its message"):
        monkeypatch.setattr(al, "retrain_round",
                            _raises(ValueError("only one class")))
        assert run() == ([], ["ValueError: only one class"])

    with subtests.test(msg="a failure reported to a dead screen is dropped"):
        monkeypatch.setattr(al, "retrain_round",
                            _raises(ValueError("no usable features")))
        assert run(delete=True) == ([], [])


# ---------------------------------------------------------------------------
# _reanchor_png_path
# ---------------------------------------------------------------------------

def test_a_stored_crop_path_is_reanchored_under_the_open_database(
        experiment, subtests):
    """Stale or relative paths are rebuilt from their data/ segment, else kept."""
    db = str(experiment / "measurements" / "measurements.db")
    images = experiment / "data" / "images"
    cases = [
        ("an empty path stays empty", "", db, ""),
        ("a path that exists is kept", str(images / "cell_03.png"), db,
         str(images / "cell_03.png")),
        ("no open database leaves the path as stored", "/gone/data/x.png", "",
         "/gone/data/x.png"),
        ("a moved dataset is found from its data segment",
         "/somewhere/that/never/existed/data/images/cell_00.png", db,
         str(images / "cell_00.png")),
        ("a Windows path is found from its data segment",
         "C:\\old\\data\\images\\cell_02.png", db, str(images / "cell_02.png")),
        ("a relative path is rebuilt under the database root",
         "data/images/cell_01.png", db, str(images / "cell_01.png")),
        ("a crop missing under the new root too is left alone",
         "/old/data/images/not_here_either.png", db,
         "/old/data/images/not_here_either.png"),
        ("a relative crop missing under the new root is left alone",
         "data/nope.png", db, "data/nope.png"),
        ("a path with no data segment is left alone", "/old/cell_00.png", db,
         "/old/cell_00.png"),
    ]
    for case_id, path, database, expected in cases:
        with subtests.test(msg=case_id):
            assert annotate_mod._reanchor_png_path(path, database) == expected


# ---------------------------------------------------------------------------
# _load_thumb_image_worker
# ---------------------------------------------------------------------------

def _pixels(img):
    return np.asarray(img.convert("RGB"))


def _displayed(img, s):
    """What the thumbnail pipeline makes of ``img`` without an outline."""
    shown = annotate_mod.normalize_pil(img, s.percentiles, s.normalize_channels)
    shown = annotate_mod.filter_channels_pil(shown, s.channels)
    return _pixels(shown.resize(s.image_size))


class _MergedSource:
    kind = "merged"

    def __init__(self, get):
        self.get = get


def test_a_thumbnail_is_decoded_or_drawn_as_a_placeholder_keeping_its_label(
        experiment, monkeypatch, subtests):
    """Every row yields an image of the tile size and keeps its annotation."""
    s = _settings_for(experiment)
    crop = str(experiment / "data" / "images" / "cell_00.png")
    on_disk = annotate_mod.load_crop_image(crop, db_path=s.db_path)
    grey = np.full((16, 16, 3), 30, dtype=np.uint8)

    def hatched(path):
        img, annotation = annotate_mod._load_thumb_image_worker(
            {"png_path": path, "annotation": 3}, None, s)
        return img, annotation

    with subtests.test(msg="a missing file is a hatched tile naming the path"):
        img, annotation = hatched("/nowhere/x.png")
        assert annotation == 3
        assert img.size == (16, 16)
        assert img.info[annotate_mod._MISSING_CROP] == "/nowhere/x.png"
        assert tuple(_pixels(img)[0, 0]) == (150, 150, 150)
        assert tuple(_pixels(img)[0, 8]) == (96, 96, 96)

    with subtests.test(msg="a row with no path is a hatched tile naming nothing"):
        img, annotation = hatched(None)
        assert (annotation, img.info[annotate_mod._MISSING_CROP]) == (3, "")

    with subtests.test(msg="a crop on disk is normalised, filtered and resized"):
        img, annotation = annotate_mod._load_thumb_image_worker(
            (crop, 2), None, s)
        assert annotation == 2
        assert annotate_mod._MISSING_CROP not in img.info
        np.testing.assert_array_equal(_pixels(img), _displayed(on_disk, s))

    with subtests.test(msg="a non-merged source still reads the png"):
        class _PngSource:
            pass
        img, _ = annotate_mod._load_thumb_image_worker(
            {"png_path": crop}, _PngSource(), s)
        np.testing.assert_array_equal(_pixels(img), _displayed(on_disk, s))

    with subtests.test(msg="the stored and display channel orders are passed on"), \
            monkeypatch.context() as m:
        calls = []
        m.setattr(annotate_mod, "load_crop_image",
                  lambda path, **kw: calls.append((path, kw)) or on_disk)
        ordered = _settings_for(experiment)
        ordered.stored_channel_order = "legacy_bgr"
        ordered.display_order = "bgr"
        ordered.display_primaries = "cmy"
        annotate_mod._load_thumb_image_worker((crop, None), None, ordered)
        assert calls == [(crop, {"db_path": s.db_path,
                                 "stored_channel_order": "legacy_bgr",
                                 "display_order": "bgr",
                                 "display_primaries": "cmy"})]

    with subtests.test(msg="a crop that will not decode is a grey tile"), \
            monkeypatch.context() as m:
        m.setattr(annotate_mod, "load_crop_image",
                  _raises(OSError("bad header")))
        img, annotation = annotate_mod._load_thumb_image_worker(
            (crop, 2), None, s)
        assert annotation == 2
        np.testing.assert_array_equal(_pixels(img), grey)

    with subtests.test(msg="a merged source that cannot cut the crop is a grey tile"):
        rows = []
        img, annotation = annotate_mod._load_thumb_image_worker(
            {"png_path": "x.png", "annotation": None},
            _MergedSource(lambda row: rows.append(row) or
                          _raises(KeyError("no such object"))()), s)
        assert annotation is None
        assert rows == [{"png_path": "x.png", "annotation": None}]
        np.testing.assert_array_equal(_pixels(img), grey)

    with subtests.test(msg="a merged source crop is decoded from its array"):
        rows = []
        arr = np.zeros((8, 8, 3), dtype=np.uint8)
        arr[:4] = 200
        img, annotation = annotate_mod._load_thumb_image_worker(
            ("x.png", 4), _MergedSource(lambda row: rows.append(row) or arr), s)
        assert annotation == 4
        assert rows == [{"png_path": "x.png"}]
        np.testing.assert_array_equal(
            _pixels(img), _displayed(Image.fromarray(arr).convert("RGB"), s))

    with subtests.test(msg="the outline sees the full image, the display the filtered one"), \
            monkeypatch.context() as m:
        outlined = _settings_for(experiment)
        outlined.outline = ["r"]
        outlined.channels = ["g"]
        seen = {}
        stop = object()

        def _outline(**kwargs):
            seen.update(kwargs)
            return Image.new("RGB", (5, 5), (1, 2, 3))

        m.setattr(annotate_mod, "outline_image", _outline)
        img, annotation = annotate_mod._load_thumb_image_worker(
            (crop, None), None, outlined, should_stop=stop)
        assert annotation is None
        np.testing.assert_array_equal(
            _pixels(img), np.tile(np.array([1, 2, 3], np.uint8), (16, 16, 1)))
        full = annotate_mod.normalize_pil(on_disk, outlined.percentiles,
                                          outlined.normalize_channels)
        np.testing.assert_array_equal(_pixels(seen.pop("full_img")),
                                      _pixels(full))
        np.testing.assert_array_equal(
            _pixels(seen.pop("base_img")),
            _pixels(annotate_mod.filter_channels_pil(full, ["g"])))
        assert seen == {
            "outline_channels": ["r"],
            "edge_sigma": outlined.outline_sigma,
            "edge_thickness": outlined.edge_thickness,
            "edge_transparency": outlined.edge_transparency,
            "edge_image": outlined.edge_image,
            "outline_threshold_factor": outlined.outline_threshold_factor,
            "object_size": outlined.object_size,
            "object_filters": getattr(outlined, "object_filters", None),
            "outline_method": outlined.outline_method,
            "should_stop": stop,
        }

    with subtests.test(msg="an outline that fails still shows the crop"), \
            monkeypatch.context() as m:
        outlined = _settings_for(experiment)
        outlined.outline = ["r"]
        m.setattr(annotate_mod, "outline_image",
                  _raises(RuntimeError("no cellpose")))
        img, annotation = annotate_mod._load_thumb_image_worker(
            (crop, 1), None, outlined)
        assert annotation == 1
        np.testing.assert_array_equal(_pixels(img),
                                      _displayed(on_disk, outlined))

    with subtests.test(msg="a cancelled outline abandons the crop"), \
            monkeypatch.context() as m:
        outlined = _settings_for(experiment)
        outlined.outline = ["r"]
        m.setattr(annotate_mod, "outline_image",
                  _raises(annotate_mod.OutlineCancelled("page left")))
        with pytest.raises(annotate_mod.OutlineCancelled):
            annotate_mod._load_thumb_image_worker((crop, 1), None, outlined)


# ---------------------------------------------------------------------------
# _compute_total
# ---------------------------------------------------------------------------

def test_the_page_total_comes_from_the_queue_the_filter_or_the_whole_table(
        experiment, monkeypatch, subtests):
    """The dict handed back names rows, total, queue summary and any note."""
    from spacr import active_learning as al

    with subtests.test(msg="no filter and no queue counts the whole table"):
        assert annotate_mod._compute_total(
            _settings_for(experiment), filter_active=False) == {
                "filtered_rows": None, "total": 8, "queue_summary": "",
                "note": ""}

    with subtests.test(msg="the uncertainty queue orders the crops"), \
            monkeypatch.context() as m:
        s = _settings_for(experiment)
        s.queue_by_uncertainty = True
        s.queue_measure = "margin"
        s.queue_diversity = ""
        s.queue_limit = 0
        s.image_type = "cell"
        calls = []
        m.setattr(al, "build_queue",
                  lambda *a, **k: calls.append((a, k)) or "the-queue")
        m.setattr(al, "queue_rows",
                  lambda q: [("a.png", None), ("b.png", 1)] if q == "the-queue"
                  else [])
        m.setattr(al, "format_queue_summary", lambda q: f"2 by entropy ({q})")
        assert annotate_mod._compute_total(s, filter_active=True) == {
            "filtered_rows": [("a.png", None), ("b.png", 1)], "total": 2,
            "queue_summary": "2 by entropy (the-queue)", "note": ""}
        assert calls == [((s.db_path, "annotate"),
                          {"measure": "margin", "diversity": "none",
                           "limit": None, "image_type": "cell", "seed": 0})]

    for exc in (ValueError("no scores yet"), FileNotFoundError("no scores yet")):
        with subtests.test(msg=f"a queue that cannot be built ({type(exc).__name__}) falls back"), \
                monkeypatch.context() as m:
            s = _settings_for(experiment)
            s.queue_by_uncertainty = True
            m.setattr(al, "build_queue", _raises(exc))
            assert annotate_mod._compute_total(s, filter_active=False) == {
                "filtered_rows": None, "total": 8, "queue_summary": "",
                "note": "Uncertainty queue unavailable: no scores yet"}

    with subtests.test(msg="any other queue failure is not swallowed"), \
            monkeypatch.context() as m:
        s = _settings_for(experiment)
        s.queue_by_uncertainty = True
        m.setattr(al, "build_queue", _raises(RuntimeError("database gone")))
        with pytest.raises(RuntimeError, match="database gone"):
            annotate_mod._compute_total(s, filter_active=False)

    filters = [
        ("a single threshold is wrapped into lists", "cell_area", 100.0,
         "higher", (["cell_area"], [100.0], ["higher"])),
        ("threshold lists are passed as they are",
         ["cell_area", "nucleus_area"], [100.0, 5.0], ["higher", "lower"],
         (["cell_area", "nucleus_area"], [100.0, 5.0], ["higher", "lower"])),
    ]
    for case_id, measurement, threshold, direction, expected in filters:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            s = _settings_for(experiment)
            s.measurement = measurement
            s.threshold = threshold
            s.threshold_direction = direction
            s.image_type = "nucleus"
            calls = []
            rows = [("a.png", None), ("b.png", None), ("c.png", 1)]
            m.setattr(annotate_mod, "fetch_filtered_paths",
                      lambda *a: calls.append(a) or rows)
            assert annotate_mod._compute_total(s, filter_active=True) == {
                "filtered_rows": rows, "total": 3, "queue_summary": "",
                "note": ""}
            assert calls == [(s.db_path, "annotate", *expected, "nucleus")]


# ---------------------------------------------------------------------------
# _ZoomOverlay
# ---------------------------------------------------------------------------

def test_the_zoom_overlay_paints_only_when_it_has_a_crop(qtbot, subtests):
    """No crop paints nothing at all; a crop is fitted inside the margin over a scrim."""
    def overlay_with(pixmap):
        overlay = annotate_mod._ZoomOverlay()
        qtbot.addWidget(overlay)
        overlay.resize(120, 90)
        overlay._pixmap = pixmap
        return overlay

    blank = QWidget()
    qtbot.addWidget(blank)
    blank.resize(120, 90)
    untouched = blank.grab().toImage()

    for case_id, pixmap in (("no pixmap", None), ("a null pixmap", QPixmap())):
        with subtests.test(msg=case_id):
            overlay = overlay_with(pixmap)
            assert overlay.picture_rect() == QRectF()
            assert overlay.grab().toImage() == untouched

    with subtests.test(msg="a crop is drawn at its aspect inside the margin"):
        red = QPixmap(10, 20)
        red.fill(QColor(255, 0, 0))
        overlay = overlay_with(red)
        margin = float(annotate_mod.SPACING["md"])
        box_w, box_h = 120 - 2 * margin, 90 - 2 * margin
        scale = min(box_w / 10, box_h / 20)
        expected = QRectF(margin + (box_w - 10 * scale) / 2,
                          margin + (box_h - 20 * scale) / 2,
                          10 * scale, 20 * scale)
        assert overlay.picture_rect() == expected
        image = overlay.grab().toImage()
        assert image.pixelColor(60, 45) == QColor(255, 0, 0)
        # Outside the picture: the background under a black scrim of alpha 170.
        under, got = untouched.pixelColor(1, 1), image.pixelColor(1, 1)
        for channel in ("red", "green", "blue"):
            expected_value = getattr(under, channel)() * (255 - 170) / 255
            assert abs(getattr(got, channel)() - expected_value) <= 1.5
