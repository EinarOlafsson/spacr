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
* the two measurement joins (``fetch_filtered_paths`` and ``gate_paths``)
  meeting a frame that **already carries its ``prcfo`` key as a column**,
  or that already carries the annotation column, so neither may be reset or
  overwritten;
* the save worker being **stopped before it ever ran**, which must keep the
  queued edit rather than swallow it;
* a writer that fails a transaction and is then restarted onto a database
  that has vanished — the FIRST, actionable error is the one the user keeps.

Two guards in this module cannot be reached at all. They are not silenced;
the invariant that makes each of them dead is asserted instead, so the day
someone breaks the invariant a test fails here rather than a branch quietly
coming alive. See ``test_every_readable_filter_yields_at_least_one_token``
(``parse_image_type``'s empty-token guard) and
``test_the_annotation_column_survives_every_step_after_it_is_added``
(``fetch_filtered_paths``'s re-check of the column it just created).

Offscreen, offline, no real Cellpose checkpoint, no sleeps.
"""
from __future__ import annotations

import re
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr.qt import annotate_engine as ae
from spacr.qt.annotate_engine import (OutlineCancelled, SaveWorker,
                                      cache_budget_entries, class_counts,
                                      drop_cache_budget_entry,
                                      fetch_filtered_paths,
                                      forget_outline_masks, gate_paths,
                                      outline_image, parse_image_type)
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

    def eval(self, image, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
        # channel_axis IS NAMED AND IS READ. Absorbing it into **kwargs
        # leaves the double unable to tell a working call from the
        # `channel_axis=3` that raised on every real run, which is how that
        # bug survived fifteen tests. `check_cellpose_eval_call` raises on
        # an axis this image cannot take, and the sentinel default keeps
        # "omitted" distinguishable from "passed None".
        # require_channel_axis=False: the outline path passes a single
        # 2-D plane (annotate_engine.py:338), where Cellpose's own
        # auto-detect is correct and spaCR deliberately omits the
        # axis -- the same choice spacr.spacrops and spacr.submodules
        # make. The value is still checked when one IS passed, which
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


@pytest.fixture
def stub_joined_tables(monkeypatch):
    """Drive ``spacr.io``'s two readers, whose frames decide the join path.

    The shape of the joined frame is the whole subject here -- whether it
    carries ``prcfo`` as a column or as an index, and whether the annotation
    column is already on it -- and no on-disk schema can express the second
    of those, since the join builds it.
    """
    def install(joined, png_list=None):
        import spacr.io as io

        monkeypatch.setattr(io, "_read_and_join_tables",
                            lambda *_a, **_k: joined.copy())
        monkeypatch.setattr(
            io, "_read_db",
            lambda *_a, **_k: [(png_list if png_list is not None
                                else pd.DataFrame()).copy()])
    return install


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

def test_a_join_that_already_carries_its_key_is_not_reset(tmp_path,
                                                          stub_joined_tables):
    """``reset_index`` is for a frame INDEXED by prcfo, not one holding it.

    Resetting an index that is not prcfo would push a meaningless level in
    as a column and the merge would then find no key at all.
    """
    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    as_column = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                              "cell_area": [50.0, 500.0, 900.0]})
    crops = pd.DataFrame({
        "prcfo": ["o1", "o2", "o3"],
        "png_path": ["/c/o1.png", "/c/o2.png", "/c/o3.png"]}).set_index("prcfo")
    stub_joined_tables(as_column, crops)

    rows = fetch_filtered_paths(str(db), "annotate", ["cell_area"], [100.0],
                                ["higher"])

    assert [path for path, _annotation in rows] == ["/c/o2.png", "/c/o3.png"]
    assert all(annotation is None for _path, annotation in rows)


def test_an_annotation_already_on_the_join_is_returned_not_blanked(
        tmp_path, stub_joined_tables):
    """The column is CREATED when missing, never recreated over real values."""
    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    with_annotations = pd.DataFrame({
        "png_path": ["/c/o1.png", "/c/o2.png"],
        "cell_area": [500.0, 900.0],
        "annotate": [1, 2]})
    stub_joined_tables(with_annotations)

    rows = fetch_filtered_paths(str(db), "annotate", ["cell_area"], [10.0],
                               ["higher"])

    assert [tuple(row) for row in rows] == [("/c/o1.png", 1), ("/c/o2.png", 2)]

    # ... and the same frame WITHOUT the column comes back with it, empty.
    stub_joined_tables(with_annotations.drop(columns=["annotate"]))
    blank = fetch_filtered_paths(str(db), "annotate", ["cell_area"], [10.0],
                                 ["higher"])
    assert [path for path, _a in blank] == ["/c/o1.png", "/c/o2.png"]
    assert all(annotation is None for _p, annotation in blank)


def test_a_gated_join_that_already_carries_its_key_is_not_reset(
        tmp_path, stub_joined_tables):
    """The gate route merges crop paths on exactly as the threshold one does.

    A population gated on screen and one annotated from it have to be the
    same population, so the two joins must not diverge on frame shape.
    """
    from spacr.qt.widgets.gate_spec import ThresholdGate

    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    as_column = pd.DataFrame({"prcfo": ["o1", "o2", "o3"],
                              "cell_area": [50.0, 500.0, 900.0]})
    crops = pd.DataFrame({
        "prcfo": ["o1", "o2", "o3"],
        "png_path": ["/c/o1.png", "/c/o2.png", "/c/o3.png"]}).set_index("prcfo")
    stub_joined_tables(as_column, crops)

    gate = ThresholdGate(name="big cells", column="cell_area", low=100.0)

    assert gate_paths(str(db), [gate]) == ["/c/o2.png", "/c/o3.png"]


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

def test_every_readable_filter_yields_at_least_one_token():
    """Why ``parse_image_type``'s empty-token guard is dead code.

    ``parse_image_type`` returns early on an empty expression, so by the time
    it tokenises, ``text`` is non-empty AFTER ``str.strip()`` -- it holds at
    least one character that ``str.isspace()`` calls False. The tokeniser's
    pattern is ``\\(|\\)|[^\\s()]+``, whose third alternative matches every
    character that is neither whitespace nor a parenthesis and whose first
    two match the parentheses, so ``re.findall`` cannot return an empty list
    for such a string (``re``'s ``\\s`` and ``str.isspace`` are the same
    predicate -- both are ``Py_UNICODE_ISSPACE`` -- and that equality is
    asserted below over the whole code space). Every one of the tokeniser's
    four branches then appends at least one token.

    So ``if not tokens: return "", []`` cannot fire. What is pinned here is
    the invariant, not the guard: if the pattern or a branch ever stops
    producing a token, this fails instead of the guard silently coming alive
    and turning a typo into "no filter at all".
    """
    assert not [cp for cp in range(0x110000)
                if chr(cp).isspace() != bool(re.fullmatch(r"\s", chr(cp)))]

    awkward = ["a", "!", "!a", "(", ")", "()", "(a)", "a AND NOT b",
               " x ", "\x1cq", "((", "NOT", "-", "\U0001f600"]
    for expression in awkward:
        stripped = expression.strip()
        assert stripped, f"{expression!r} is not a case this guard sees"
        assert ae._tokenise_image_type(stripped), (
            f"{expression!r} tokenised to nothing")

    # And the guard's own contract for the input it really does get: an
    # expression that is only whitespace filters nothing.
    assert parse_image_type("   ") == ("", [])
    assert parse_image_type(None) == ("", [])
    assert parse_image_type("!a") == ("(NOT png_path LIKE ?)", ["%a%"])


def test_the_annotation_column_survives_every_step_after_it_is_added(
        tmp_path, stub_joined_tables):
    """Why ``fetch_filtered_paths``'s second column check is dead code.

    The function adds ``annotation_column`` to the frame when the join did
    not bring one, and then re-checks for it just before selecting. Nothing
    between the two can remove a column: ``_apply_threshold`` returns either
    the frame or a boolean row selection of it, and ``dropna(subset=...)``
    and the ``image_type`` mask are row selections too. So the second check
    cannot fail and its ``return []`` is unreachable.

    The steps in between are what this pins. Each is driven here against a
    frame that arrived WITHOUT the column, and the column is still there --
    carrying ``None`` -- on the far side of all three.
    """
    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    joined = pd.DataFrame({
        "png_path": ["/c/o1.png", "/c/o2.png", None, "/c/keep_o4.png"],
        "cell_area": [50.0, 500.0, 900.0, 900.0]})
    stub_joined_tables(joined)

    # threshold (drops o1), dropna on png_path (drops the None), and the
    # image_type mask (drops o2) all run between the add and the re-check.
    rows = fetch_filtered_paths(str(db), "annotate", ["cell_area"], [100.0],
                                ["higher"], image_type="keep")

    assert [tuple(row) for row in rows] == [("/c/keep_o4.png", None)]

    # Every row is a PAIR: the column exists even though the join had none.
    assert all(len(row) == 2 for row in rows)

    # ... and when the thresholds empty the frame entirely, the answer is
    # still a well-formed (empty) list rather than the guard's early return.
    stub_joined_tables(joined)
    assert fetch_filtered_paths(str(db), "annotate", ["cell_area"],
                                [10000.0], ["higher"]) == []
