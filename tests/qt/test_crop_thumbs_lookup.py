"""Crop lookup for hover previews: re-anchoring, plot-time resolution, the cache.

``resolve_crop_path`` re-anchors a stored crop path when a dataset moved and
must never raise under a mouse handler; ``crop_paths_for_keys`` resolves a
whole plot with one scan and bisects only around keys that have no crop; and
``CropThumbnails`` keeps a bounded cache whose counters a status line reports.
"""
from __future__ import annotations

import os
import sqlite3
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import active_learning
from spacr.qt import crop_thumbs as CT
from spacr.qt.screens import annotate as annotate_screen

pytestmark = pytest.mark.qt

STALE = "/some/other/disk/olddir/data/plate1/cell_png/plate1_A01_1_7.png"


@pytest.fixture
def moved_dataset(tmp_path):
    """A crop tree with its database, as ``(db_path, real_crop_path)``."""
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    crop_dir = root / "data" / "plate1" / "cell_png"
    crop_dir.mkdir(parents=True)
    real = crop_dir / "plate1_A01_1_7.png"
    real.write_bytes(b"not really a png")
    db = root / "measurements" / "measurements.db"
    db.write_bytes(b"")
    return str(db), str(real)


def _raise(exc):
    def fail(_path, _db):
        raise exc
    return fail


def test_resolve_crop_path_answers_with_the_best_path_it_has(
        subtests, monkeypatch, moved_dataset):
    """Existing, empty and undatabased paths pass through; a moved one is rebuilt."""
    db, real = moved_dataset
    plain = [
        ("existing file kept whatever the db", real, db, real),
        ("existing file without a db", real, "", real),
        ("empty string", "", db, ""),
        ("None path", None, db, ""),
        ("no db at all", "/gone/x.png", "", "/gone/x.png"),
        ("default db argument", "/gone/x.png", None, "/gone/x.png"),
        ("moved dataset re-anchored under its db", STALE, db, real),
    ]
    for case_id, path, db_path, expected in plain:
        with subtests.test(msg=case_id):
            got = (CT.resolve_crop_path(path) if db_path is None
                   else CT.resolve_crop_path(path, db_path))
            assert got == expected

    with subtests.test(msg="the re-anchor rule is asked with (path, db)"):
        seen = []

        def record(path, db_path):
            seen.append((path, db_path))
            return "/here/data/crop.png"

        with monkeypatch.context() as m:
            m.setattr(annotate_screen, "_reanchor_png_path", record)
            assert CT.resolve_crop_path("/there/data/crop.png",
                                        "/here/measurements.db") \
                == "/here/data/crop.png"
        assert seen == [("/there/data/crop.png", "/here/measurements.db")]

    for case_id, exc in [("OSError while re-anchoring", OSError("disk gone")),
                         ("ValueError while re-anchoring",
                          ValueError("no /data/ segment"))]:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as m:
                m.setattr(annotate_screen, "_reanchor_png_path", _raise(exc))
                assert CT.resolve_crop_path(STALE, db) == STALE

    with subtests.test(msg="Annotate screen not importable keeps the stored path"):
        with monkeypatch.context() as m:
            m.setitem(sys.modules, "spacr.qt.screens.annotate", None)
            assert CT.resolve_crop_path(STALE, db) == STALE
            assert CT.resolve_crop_path("/there/data/crop.png",
                                        "/here/measurements.db") \
                == "/there/data/crop.png"


@pytest.fixture
def crop_table(tmp_path):
    """A real ``png_list`` with four crops keyed on their own paths."""
    db = tmp_path / "measurements.db"
    con = sqlite3.connect(db)
    try:
        con.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT)")
        rows = [(f"/crops/plate1_A01_1_{i}.png", f"plate1_A01_1_o{i}")
                for i in range(1, 5)]
        con.executemany("INSERT INTO png_list VALUES (?, ?)", rows)
        con.commit()
    finally:
        con.close()
    return str(db), [path for path, _ in rows]


def test_crop_paths_for_keys_resolves_once_and_bisects_only_around_misses(
        subtests, monkeypatch, crop_table):
    """Exact answers and exact scan batches, against the real table and a fake."""
    db, paths = crop_table
    real = active_learning.crops_for_object_keys
    p0, p1, p2, p3 = paths

    real_cases = [
        ("every key present costs one scan", paths,
         {p: p for p in paths}, [paths]),
        ("reversed order pairs each key with its own crop",
         list(reversed(paths)), {p: p for p in paths},
         [list(reversed(paths))]),
        ("a missing key is absent, and the range is halved onto it",
         [p0, p1, "no_such_object", p3], {p0: p0, p1: p1, p3: p3},
         [[p0, p1, "no_such_object", p3], [p0, p1],
          ["no_such_object", p3], ["no_such_object"], [p3]]),
        ("keys are stringified", [p2], {p2: p2}, [[p2]]),
    ]
    for case_id, keys, expected, expected_batches in real_cases:
        with subtests.test(msg=case_id):
            batches = []

            def counted(db_path, batch, **kwargs):
                assert db_path == db
                batches.append(list(batch))
                return real(db_path, batch, **kwargs)

            with monkeypatch.context() as m:
                m.setattr(active_learning, "crops_for_object_keys", counted)
                out = CT.crop_paths_for_keys(db, keys)
            assert out == expected
            assert batches == expected_batches

    def fake_known(_db, batch):
        known = {"k2": "/plate/data/k2.png"}
        scans.append(list(batch))
        return [(known[key], 1) for key in batch if key in known]

    def fake_two_rows(_db, batch):
        scans.append(list(batch))
        if list(batch) == ["k1"]:
            return [("/plate/data/k1_a.png", 1), ("/plate/data/k1_b.png", 1)]
        return []

    def explode(*_args, **_kwargs):
        raise AssertionError("the crop table must not be scanned")

    fake_cases = [
        ("one resolvable key among missing ones", fake_known,
         "/plate/measurements.db", ["k1", "k2", "k3"],
         {"k2": "/plate/data/k2.png"},
         [["k1", "k2", "k3"], ["k1"], ["k2", "k3"], ["k2"], ["k3"]]),
        ("a key with two crop rows is filed under its first", fake_two_rows,
         "/plate/measurements.db", ["k1"], {"k1": "/plate/data/k1_a.png"},
         [["k1"]]),
        ("a single missing key resolves to nothing", fake_two_rows,
         "/plate/measurements.db", ["k9"], {}, [["k9"]]),
        ("no keys scans nothing", explode, "/plate/measurements.db", [], {},
         []),
        ("no database scans nothing", explode, "", ["k1"], {}, []),
    ]
    for case_id, fake, db_path, keys, expected, expected_scans in fake_cases:
        with subtests.test(msg=case_id):
            scans = []
            with monkeypatch.context() as m:
                m.setattr(active_learning, "crops_for_object_keys", fake)
                assert CT.crop_paths_for_keys(db_path, keys) == expected
            assert scans == expected_scans


@pytest.fixture
def real_crop(tmp_path):
    """One genuine RGB crop PNG on disk."""
    from PIL import Image

    array = np.zeros((300, 240, 3), dtype="uint8")
    array[:150, :, 0] = 200
    array[150:, :, 2] = 120
    path = tmp_path / "plate1_A01_1_1.png"
    Image.fromarray(array).save(path)
    return str(path)


def _counters(cache):
    return {"decodes": cache.decodes, "hits": cache.hits,
            "misses": cache.misses, "failures": cache.failures,
            "len": len(cache)}


def test_crop_thumbnails_counts_every_decode_hit_miss_and_failure(
        subtests, qapp, tmp_path, real_crop):
    """Blank paths are never entries; failures are cached; stamps invalidate."""
    from PIL import Image

    with subtests.test(msg="a blank path asks for nothing"):
        cache = CT.CropThumbnails()
        assert [cache.peek(""), cache.pixmap(""), cache.peek(None),
                cache.pixmap(None)] == [None, None, None, None]
        assert _counters(cache) == {"decodes": 0, "hits": 0, "misses": 0,
                                    "failures": 0, "len": 0}
        assert cache.describe() == ("0/{} crops cached · 0 decode(s), "
                                    "0 hit(s), 0 unreadable"
                                    .format(CT.DEFAULT_CAPACITY))

    missing = str(tmp_path / "not_there.png")
    with subtests.test(msg="an unreadable crop is decoded once, then a hit"):
        cache = CT.CropThumbnails()
        assert cache.pixmap(missing) is None
        assert _counters(cache) == {"decodes": 1, "hits": 0, "misses": 1,
                                    "failures": 1, "len": 1}
        assert cache.pixmap(missing) is None
        assert cache.peek(missing) is None
        assert _counters(cache) == {"decodes": 1, "hits": 2, "misses": 1,
                                    "failures": 1, "len": 1}
        assert missing in cache

    with subtests.test(msg="describe names occupancy, decodes, hits, failures"):
        cache = CT.CropThumbnails(size=64, capacity=4)
        cache.pixmap(missing)
        cache.peek(missing)
        assert cache.describe() == ("1/4 crops cached · 1 decode(s), "
                                    "1 hit(s), 1 unreadable")

    with subtests.test(msg="a rewritten crop is decoded again"):
        cache = CT.CropThumbnails(size=32)
        assert cache.pixmap(real_crop) is not None
        assert cache.decodes == 1
        Image.fromarray(np.full((300, 240, 3), 40, dtype="uint8")).save(
            real_crop)
        os.utime(real_crop, (0, 0))
        assert real_crop not in cache
        cache.pixmap(real_crop)
        assert _counters(cache) == {"decodes": 2, "hits": 0, "misses": 2,
                                    "failures": 0, "len": 2}

    with subtests.test(msg="clear drops entries but keeps the counters"):
        cache = CT.CropThumbnails(size=32)
        cache.pixmap(real_crop)
        cache.peek(real_crop)
        assert len(cache) == 1 and real_crop in cache
        cache.clear()
        assert real_crop not in cache
        assert _counters(cache) == {"decodes": 1, "hits": 1, "misses": 1,
                                    "failures": 0, "len": 0}
        assert cache.peek(real_crop) is None
        assert cache.hits == 1
