"""The hover seam's two edges: a moved dataset, and a key with no crop.

`resolve_crop_path` exists because `png_list` stores absolute paths built at
measure time, so a dataset copied to another disk resolves to nothing. Its
happy path is one line; the three that matter are the ones a screen hits when
the Annotate module is not importable, when re-anchoring throws, and when it
succeeds against a rebuilt tree.

`crop_paths_for_keys` bisects when some keys have no crop, and the bisection
is what makes hover a dict lookup rather than one table scan per point. It is
driven here against a real sqlite crop table so the subsequence arithmetic is
checked against the real resolver's real dropping behaviour rather than
against a stand-in that agrees with the test by construction.
"""

import os
import sqlite3
import sys

import pytest

from spacr.qt.crop_thumbs import (DEFAULT_CAPACITY, DEFAULT_SIZE,
                                  CropThumbnails, crop_paths_for_keys,
                                  resolve_crop_path)


# ---------------------------------------------------------------------------
# a dataset that moved
# ---------------------------------------------------------------------------

@pytest.fixture
def moved_dataset(tmp_path):
    """A crop tree with the DB beside it, and the stale path it was recorded as.

    :returns: ``(db_path, stale_path, real_path)``.
    """
    root = tmp_path / "run"
    (root / "measurements").mkdir(parents=True)
    crop_dir = root / "data" / "plate1" / "cell_png"
    crop_dir.mkdir(parents=True)
    real = crop_dir / "plate1_A01_1_7.png"
    real.write_bytes(b"not really a png")
    db = root / "measurements" / "measurements.db"
    db.write_bytes(b"")
    stale = "/some/other/disk/olddir/data/plate1/cell_png/plate1_A01_1_7.png"
    return str(db), stale, str(real)


def test_a_path_that_still_resolves_is_left_alone(moved_dataset):
    """An existing file is returned untouched, whatever the database says."""
    db, _stale, real = moved_dataset
    assert resolve_crop_path(real, db) == real


def test_nothing_to_resolve_is_answered_without_a_database():
    """An empty path, or one with no database to anchor against, comes back."""
    assert resolve_crop_path("") == ""
    assert resolve_crop_path(None) == ""
    assert resolve_crop_path("/gone/x.png") == "/gone/x.png"
    assert resolve_crop_path("/gone/x.png", "") == "/gone/x.png"


def test_a_moved_dataset_is_re_anchored_under_its_own_database(moved_dataset):
    """A stale absolute path is rebuilt from its `/data/` segment.

    This is the whole point of the function: without it every hover preview
    on a copied dataset is blank.
    """
    db, stale, real = moved_dataset
    assert not os.path.isfile(stale)
    assert resolve_crop_path(stale, db) == real


def test_without_the_annotate_screen_the_stored_path_is_kept(monkeypatch,
                                                             moved_dataset):
    """A trimmed install returns the stored path rather than raising.

    The rebuild rule lives in the Annotate screen; a headless install that
    cannot import it still has to answer, and the honest answer is the path
    it was given.
    """
    db, stale, _real = moved_dataset
    # `None` in sys.modules is how Python spells "this import is blocked".
    monkeypatch.setitem(sys.modules, "spacr.qt.screens.annotate", None)
    assert resolve_crop_path(stale, db) == stale


def test_a_re_anchor_that_throws_does_not_throw_at_the_cursor(monkeypatch,
                                                              moved_dataset):
    """An exception inside the rebuild degrades to the stored path.

    This runs under a mouse handler; an exception here would leave the plot
    dead rather than the preview blank.
    """
    db, stale, _real = moved_dataset
    from spacr.qt.screens import annotate as annotate_screen

    def explode(_path, _db):
        raise OSError("the disk went away mid-hover")

    monkeypatch.setattr(annotate_screen, "_reanchor_png_path", explode)
    assert resolve_crop_path(stale, db) == stale


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

def test_an_empty_path_is_never_a_cache_entry(qapp):
    """`peek` and `pixmap` answer None for '' without touching the cache.

    A blank `png_path` is a row with no crop, and caching one under the
    key of the empty string would serve it to every other blank row.
    """
    cache = CropThumbnails()
    assert cache.peek("") is None
    assert cache.pixmap("") is None
    assert cache.peek(None) is None
    assert cache.pixmap(None) is None
    assert len(cache) == 0


def test_an_unreadable_crop_is_remembered_as_unreadable(qapp, tmp_path):
    """A failed decode is cached as None, so a hover sweep decodes it once.

    Retrying sixty times a second is the failure mode this guards; the
    counters are how a status line can say it happened.
    """
    missing = tmp_path / "not_there.png"
    cache = CropThumbnails()

    assert cache.pixmap(str(missing)) is None
    assert cache.decodes == 1
    assert cache.failures == 1
    assert cache.misses == 1

    assert cache.pixmap(str(missing)) is None
    assert cache.decodes == 1, "a known-bad crop was decoded twice"
    assert cache.hits == 1
    assert str(missing) in cache


def test_the_cache_says_how_it_is_doing(qapp, tmp_path):
    """`describe` names the occupancy, the decodes, the hits and the failures."""
    cache = CropThumbnails(size=64, capacity=4)
    cache.pixmap(str(tmp_path / "gone.png"))
    cache.peek(str(tmp_path / "gone.png"))

    line = cache.describe()
    assert "1/4 crops cached" in line
    assert "1 decode(s)" in line
    assert "1 hit(s)" in line
    assert "1 unreadable" in line


def test_the_cache_stays_bounded_under_a_sweep(qapp, tmp_path):
    """Past capacity the least recently used entry goes, not the newest."""
    cache = CropThumbnails(capacity=3)
    paths = [str(tmp_path / f"c{i}.png") for i in range(5)]
    for path in paths:
        cache.prime(path)
    assert len(cache) == 3
    assert paths[0] not in cache
    assert paths[-1] in cache


def test_the_defaults_are_sane_and_the_floors_hold(qapp):
    """Silly sizes and capacities are clamped rather than accepted."""
    assert CropThumbnails().size == DEFAULT_SIZE
    assert CropThumbnails().capacity == DEFAULT_CAPACITY
    assert CropThumbnails(size=0).size == 16
    assert CropThumbnails(capacity=0).capacity == 1


# ---------------------------------------------------------------------------
# resolving a whole plot at once
# ---------------------------------------------------------------------------

@pytest.fixture
def crop_table(tmp_path):
    """A `png_list` with four crops, keyed on their own paths."""
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


def test_nothing_to_resolve_costs_no_scan():
    """No keys, or no database, is an empty answer rather than a query."""
    assert crop_paths_for_keys("", ["a", "b"]) == {}
    assert crop_paths_for_keys("/nowhere/measurements.db", []) == {}


def test_a_plot_whose_points_all_resolve_costs_one_scan(crop_table,
                                                        monkeypatch):
    """Every key present means one call to the resolver, not one per key."""
    db, paths = crop_table
    from spacr import active_learning

    calls = []
    real = active_learning.crops_for_object_keys

    def counted(db_path, batch, **kwargs):
        calls.append(list(batch))
        return real(db_path, batch, **kwargs)

    monkeypatch.setattr(active_learning, "crops_for_object_keys", counted)

    out = crop_paths_for_keys(db, paths)
    assert out == {path: path for path in paths}
    assert len(calls) == 1


def test_a_key_with_no_crop_is_absent_rather_than_empty(crop_table):
    """Missing keys are dropped; the ones around them still resolve.

    Mapping a missing key to '' would claim the object's crop is at nowhere,
    which is a different -- and false -- statement from having none.
    """
    db, paths = crop_table
    keys = [paths[0], paths[1], "plate9_Z99_9_o9", paths[3]]

    out = crop_paths_for_keys(db, keys)
    assert out == {paths[0]: paths[0], paths[1]: paths[1],
                   paths[3]: paths[3]}
    assert "plate9_Z99_9_o9" not in out


def test_the_bisection_narrows_onto_the_missing_key(crop_table, monkeypatch):
    """A miss costs a handful of extra scans, not one scan per key.

    The resolver keeps order but drops what it cannot find, so a short answer
    no longer lines up with the request; the range is halved until a half is
    either fully resolved or a single key.
    """
    db, paths = crop_table
    from spacr import active_learning

    batches = []
    real = active_learning.crops_for_object_keys

    def counted(db_path, batch, **kwargs):
        batches.append(list(batch))
        return real(db_path, batch, **kwargs)

    monkeypatch.setattr(active_learning, "crops_for_object_keys", counted)

    keys = [paths[0], paths[1], "no_such_object", paths[3]]
    out = crop_paths_for_keys(db, keys)

    assert set(out) == {paths[0], paths[1], paths[3]}
    # the whole range, then the two halves, then the failing half split again
    assert batches[0] == keys
    assert ["no_such_object"] in batches
    # and it stopped there rather than scanning once per key
    assert len(batches) < 2 * len(keys)


def test_every_resolved_key_maps_to_its_own_crop(crop_table):
    """Order is kept by the resolver, so the zip pairs each key with its own.

    A pairing that slipped by one would show the neighbouring object's image
    under every point, which looks plausible and is wrong.
    """
    db, paths = crop_table
    out = crop_paths_for_keys(db, list(reversed(paths)))
    for path in paths:
        assert out[path] == path


# ---------------------------------------------------------------------------
# a crop that really decodes
# ---------------------------------------------------------------------------

@pytest.fixture
def real_crop(tmp_path):
    """One genuine RGB crop PNG on disk, wider than a thumbnail."""
    from PIL import Image
    import numpy as np

    array = np.zeros((300, 240, 3), dtype="uint8")
    array[:150, :, 0] = 200          # a red band, so a channel swap shows
    array[150:, :, 2] = 120
    path = tmp_path / "plate1_A01_1_1.png"
    Image.fromarray(array).save(path)
    return str(path)


def test_a_real_crop_decodes_once_and_is_served_from_memory(qapp, real_crop):
    """The first ask decodes; the second is a hit and does not decode again."""
    cache = CropThumbnails(size=64)

    assert cache.peek(real_crop) is None, "peek decoded on the hover path"

    pixmap = cache.pixmap(real_crop)
    assert pixmap is not None
    assert not pixmap.isNull()
    # thumbnailed down the long edge, aspect kept
    assert max(pixmap.width(), pixmap.height()) == 64
    assert pixmap.width() < pixmap.height()

    assert cache.decodes == 1
    assert cache.failures == 0

    again = cache.peek(real_crop)
    assert again is not None
    assert cache.decodes == 1
    assert cache.hits == 1


def test_a_rewritten_crop_is_re_read_rather_than_served_stale(qapp, real_crop):
    """The key carries the file's stamp, so a re-run's crop is decoded again.

    Serving last session's decode for a crop the pipeline has just rewritten
    is the kind of staleness nobody notices until the image is wrong.
    """
    from PIL import Image
    import numpy as np

    cache = CropThumbnails(size=32)
    cache.pixmap(real_crop)
    assert cache.decodes == 1

    bigger = np.full((300, 240, 3), 40, dtype="uint8")
    Image.fromarray(bigger).save(real_crop)
    # a different size on disk is a different identity
    os.utime(real_crop, (0, 0))

    cache.pixmap(real_crop)
    assert cache.decodes == 2


def test_clearing_drops_the_decodes_but_keeps_the_counters(qapp, real_crop):
    """`clear` empties the cache -- what a screen closing does."""
    cache = CropThumbnails(size=32)
    cache.pixmap(real_crop)
    assert len(cache) == 1

    cache.clear()
    assert len(cache) == 0
    assert real_crop not in cache
    assert cache.decodes == 1, "clearing rewrote the health counters"
