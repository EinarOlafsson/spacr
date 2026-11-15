"""The annotate engine's last unwalked turnings.

Every branch pinned here is one the engine only takes when something has
already gone sideways, and each one has a rule that is invisible from the
happy path:

* a ``should_stop`` hook that RAISES instead of answering. A QThread whose
  C++ wrapper has been deleted does exactly that, and the caller behind it
  is gone either way, so the unanswerable question has to count as "stop"
  rather than as "carry on and build a 1.2 GB checkpoint";
* the memory policy asking the engine to evict a decoded array **after the
  plate changed and the caches were dropped**, which must be a quiet "no"
  and not an ``AttributeError`` on ``None``;
* an outline filter on a crop with **no objects in it at all**, where the
  label count is zero and there is nothing to take a mean of;
* the save worker being **stopped before it ever ran**, which must keep the
  queued edit rather than swallow it;
* a writer that fails a transaction and is then restarted onto a database
  that has vanished — the FIRST, actionable error is the one the user keeps.

The measurement joins and the image-filter parser are covered by
``tests/qt/test_annotate_engine_crop_queries.py``.

Offscreen, offline, no real Cellpose checkpoint, no sleeps.
"""
from __future__ import annotations

import shutil
import sqlite3

import numpy as np
import pytest
from PIL import Image

from spacr.qt import annotate_engine as ae
from spacr.qt.annotate_engine import (OutlineCancelled, SaveWorker,
                                      cache_budget_entries, class_counts,
                                      drop_cache_budget_entry,
                                      forget_outline_masks, outline_image)
from tests.conftest import (MISSING_CHANNEL_AXIS,
                            check_cellpose_eval_call)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_outline_caches():
    """The decoded-array caches are module globals; no test may inherit one."""
    forget_outline_masks()
    yield
    forget_outline_masks()


def _blob(value: int = 200, plane: int = 0):
    """A 24x24 crop with one solid 12x12 square in ``plane``."""
    arr = np.zeros((24, 24, 3), dtype=np.uint8)
    arr[6:18, 6:18, plane] = value
    return arr


class _StubCellposeModel:
    """Stands in for cpsam: returns one square label, counts its calls."""

    def __init__(self):
        self.calls = 0

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
             normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
             anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
             min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile_overlap=0.1, bsize=256,
             compute_masks=True, progress=None):
        # THE INSTALLED SIGNATURE, WRITTEN OUT, and no **kwargs. A
        # double that accepts anything cannot fail when spaCR passes
        # an argument cellpose has removed, which is what
        # tests/cellpose_api_contract.py exists to catch.
        # channel_axis IS NAMED AND IS READ. Absorbing it into **kwargs
        # leaves the double unable to tell a working call from the
        # `channel_axis=3` that raised on every real run, which is how that
        # bug survived fifteen tests. `check_cellpose_eval_call` raises on
        # an axis this image cannot take, and the sentinel default keeps
        # "omitted" distinguishable from "passed None".
        # require_channel_axis=False: the outline path passes a single
        # 2-D plane (annotate_engine.py:338), where Cellpose's own
        # auto-detect is correct and spaCR deliberately omits the
        # axis -- the same choice spacr.submodules makes. The value is
        # still checked when one IS passed, which
        # is what catches the channel_axis=3 that broke every real run.
        check_cellpose_eval_call([image], channel_axis,
                                 require_channel_axis=False)
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=np.int32)
        mask[6:18, 6:18] = 1
        return mask, None, None


@pytest.fixture
def seeded_cellpose_model():
    """Seed the module's model cache instead of monkeypatching the getter.

    ``_get_cellpose_outline_model`` returns the cached model when there is
    one, so a seeded cache is the same door the real 1.2 GB checkpoint comes
    through -- including the ``should_stop`` questions asked on the way to
    it. Reaching for the private global is the only way in: there is no
    public setter, and building the real model would read a checkpoint.
    """
    previous = ae._cellpose_outline_model
    model = _StubCellposeModel()
    ae._cellpose_outline_model = model
    try:
        yield model
    finally:
        ae._cellpose_outline_model = previous


@pytest.fixture
def png_list_db(tmp_path):
    """A real ``png_list`` with an ``annotate`` column and three crops."""
    path = tmp_path / "measurements.db"
    con = sqlite3.connect(path)
    try:
        con.execute('CREATE TABLE "png_list" '
                    '(png_path TEXT PRIMARY KEY, annotate INTEGER)')
        con.executemany('INSERT INTO "png_list" VALUES (?, NULL)',
                        [(f"/crops/o{i}.png",) for i in range(3)])
        con.commit()
    finally:
        con.close()
    return str(path)



# ---------------------------------------------------------------------------
# A stop hook that cannot be asked
# ---------------------------------------------------------------------------

def test_a_stop_hook_that_raises_stops_the_page(seeded_cellpose_model):
    """An unanswerable question is a stop, not a reason to keep working.

    The hook is a bound method of a QThread wrapper, and once Qt has deleted
    the C++ object behind it the call raises ``RuntimeError`` instead of
    returning a bool. Nobody is waiting for this crop either way, so the
    inference must not be entered.
    """
    def deleted_thread():
        raise RuntimeError("Internal C++ object already deleted")

    crop = Image.fromarray(_blob())

    with pytest.raises(OutlineCancelled):
        outline_image(base_img=crop, full_img=crop, outline_channels=["r"],
                      outline_method="cellpose", should_stop=deleted_thread)

    assert seeded_cellpose_model.calls == 0, "cellpose was entered anyway"

    # The same crop, asked of a caller that is still there, is outlined --
    # so the cancellation above came from the raise and not from the setup.
    drawn = np.asarray(outline_image(
        base_img=crop, full_img=crop, outline_channels=["r"],
        outline_method="cellpose", should_stop=lambda: False))

    assert seeded_cellpose_model.calls == 1
    assert (drawn[:, :, 0] == 255).any(), "no outline was drawn"


# ---------------------------------------------------------------------------
# Evicting from a cache that is no longer there
# ---------------------------------------------------------------------------

def test_a_dropped_plate_leaves_the_memory_policy_nothing_to_evict():
    """``forget_outline_masks`` nulls the caches; eviction must survive it.

    The global memory policy holds the record keys it was handed and comes
    back for them later. A plate change between those two moments used to
    leave it calling ``.pop`` on ``None``.
    """
    crop = Image.fromarray(_blob())
    outline_image(base_img=crop, full_img=crop, outline_channels=["r"])

    entries = cache_budget_entries()
    kinds = {kind for (kind, _key), _bytes, _used, _pinned in entries}
    assert kinds == {"mask", "edge"}, "the draw did not fill both caches"
    assert all(size > 0 for _key, size, _used, _pinned in entries)

    victim = entries[0][0]
    assert drop_cache_budget_entry(victim) is True
    assert victim not in [key for key, *_rest in cache_budget_entries()]
    # Still a live cache, just no longer holding that array.
    assert drop_cache_budget_entry(victim) is False

    forget_outline_masks()                       # the user changed plate

    assert drop_cache_budget_entry(victim) is False
    assert cache_budget_entries() == [], "a forgotten cache still reports rows"


# ---------------------------------------------------------------------------
# An object filter with no objects to filter
# ---------------------------------------------------------------------------

def test_an_object_filter_on_a_featureless_crop_draws_nothing():
    """A flat crop thresholds to an empty mask: zero labels, no means."""
    flat = np.full((24, 24, 3), 120, dtype=np.uint8)
    flat_img = Image.fromarray(flat)
    window = (1, 10000)                # a filter wide enough to keep anything

    blank = np.asarray(outline_image(
        base_img=flat_img, full_img=flat_img, outline_channels=["r"],
        object_size=window))

    assert blank[:, :, 0].max() == 0, "an outline was drawn around nothing"
    assert (blank[:, :, 1] == 120).all(), "an untouched plane was touched"

    # The same filter on a crop that HAS an object outlines it: a 12x12
    # square's inner boundary is its 44-pixel perimeter.
    crop = Image.fromarray(_blob())
    drawn = np.asarray(outline_image(
        base_img=crop, full_img=crop, outline_channels=["r"],
        object_size=window))

    assert int((drawn[:, :, 0] == 255).sum()) == 44


# ---------------------------------------------------------------------------
# Joins that already carry what the merge would have built
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The save worker
# ---------------------------------------------------------------------------

def test_a_stop_before_the_writer_ever_ran_keeps_the_edit(png_list_db):
    """Closing a screen whose writer never started must not lose keystrokes.

    There is no thread to join, so ``stop`` returns immediately -- but the
    batch it was holding is still owed to the database, and a later start
    has to commit it.
    """
    worker = SaveWorker(png_list_db, "annotate")
    worker.submit({"/crops/o0.png": 1})
    assert worker.pending_batches == 1

    worker.stop()

    assert worker.is_alive is False
    assert worker.pending_batches == 1, "the queued edit was discarded"
    assert class_counts(png_list_db, "annotate") == []

    worker.start()
    worker.stop()

    assert class_counts(png_list_db, "annotate") == [(1, 1)]
    assert worker.pending_batches == 0
    assert worker.last_error is None


def test_the_first_writer_failure_is_the_one_the_user_keeps(tmp_path):
    """A restart onto a vanished database must not overwrite the real reason.

    "The writer could not start" is true of the second attempt and useless:
    it says nothing about the annotations still sitting unsaved in memory.
    The message that names the failed transaction is the one that tells the
    user what to fix, so it is the one that survives.
    """
    project = tmp_path / "expt"
    project.mkdir()
    db = project / "measurements.db"
    con = sqlite3.connect(db)
    try:
        # png_list with NO annotate column: the UPDATE cannot be prepared.
        con.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        con.execute('INSERT INTO "png_list" VALUES (\'/crops/o0.png\')')
        con.commit()
    finally:
        con.close()

    worker = SaveWorker(str(db), "annotate")
    worker.start()
    worker.submit({"/crops/o0.png": 1})
    worker.stop()

    first = worker.last_error
    assert first is not None
    assert "no such column" in first
    assert "resolve the database problem" in first

    # The project is deleted under the writer and the user tries again.
    shutil.rmtree(project)
    worker.start()
    worker.stop()

    assert worker.last_error == first, "the actionable message was replaced"
    assert "could not start" not in worker.last_error

    # ... and "could not start" IS what a writer with no earlier failure says
    # about that same unopenable path, so the absence above is a choice.
    fresh = SaveWorker(str(db), "annotate")
    fresh.start()
    fresh.stop()

    assert fresh.last_error is not None
    assert "could not start" in fresh.last_error


# ---------------------------------------------------------------------------
# The two guards that cannot be reached
# ---------------------------------------------------------------------------
