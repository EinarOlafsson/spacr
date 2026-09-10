"""Store, promote and revert a hit attribution -- the write/undo lifecycle.

Instruction 60. ``store_attribution``, ``promote_calls`` and
``revert_promotion`` were uncovered, and they are the three functions in
:mod:`spacr.hit_attribution` that WRITE TO A USER'S DATABASE.

``revert_promotion``'s docstring makes a strong promise -- "restore exactly
the values replaced by promote_calls" -- and exactly is the word worth
testing. A revert that restores NULL where the user had typed an annotation
is data loss that looks like an undo working.

So the shape of this file is one round trip: annotate some rows by hand,
promote over them, check the promotion took, revert, and check the hand
annotations came back BYTE FOR BYTE including the ones that were already
there and the ones that were not.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.hit_attribution import (
    HitAttributionError, HitInvestigationResult, HitRunContext,
    promote_calls, revert_promotion, store_attribution,
)


KEYS = [f"plate1_r1_c1_f1_o{i}" for i in range(1, 5)]


@pytest.fixture
def database(tmp_path):
    """A measurements database with a png_list the promoter can write to."""
    path = tmp_path / "measurements.db"
    png = pd.DataFrame({
        "prcfo": KEYS,
        "png_path": [f"/crops/{k}.png" for k in KEYS],
        # Two rows the user already annotated by hand, two untouched.
        "test_annotation": ["keep-me", None, "also-mine", None],
    })
    with sqlite3.connect(str(path)) as db:
        png.to_sql("png_list", db, index=False)
    return str(path)


def _result(run_id="run-1", probabilities=(0.9, 0.8, 0.2, 0.1)):
    return HitInvestigationResult(
        attribution_run_id=run_id,
        context=HitRunContext(
            regression_results_folder="/screens/plate1/results",
            regression_run_sha256="0" * 64,
            gene="TGGT1_123456",
            phenotype="infection",
            effect=1.25,
            guides=("g1", "g2"),
            fdr=0.05,
            direction="positive",
        ),
        cells=pd.DataFrame({
            "prcfo": KEYS,
            "candidate_probability": list(probabilities),
        }),
        wells=pd.DataFrame({"prc": ["plate1_r1_c1"], "fraction": [0.5]}),
        enrichment={"odds_ratio": 3.0},
        feature_columns=["area"],
        split_level="well",
    )


# --------------------------------------------------------------------------- #
#  store_attribution
# --------------------------------------------------------------------------- #

def test_storing_an_investigation_writes_its_rows(database):
    stored = store_attribution(database, _result())
    assert stored == len(KEYS)


def test_storing_refuses_a_database_that_is_not_there(tmp_path):
    """A typo'd path must say so, not create a database and look successful."""
    missing = str(tmp_path / "nope.db")
    with pytest.raises(HitAttributionError) as caught:
        store_attribution(missing, _result())
    assert "no database" in str(caught.value)


def test_storing_refuses_duplicate_object_keys(database):
    """Two rows for one object means one of them is wrong, and storing both
    would make every count computed from this run wrong too."""
    result = _result()
    result.cells = pd.DataFrame({
        "prcfo": [KEYS[0], KEYS[0]],
        "candidate_probability": [0.9, 0.4],
    })
    with pytest.raises(HitAttributionError) as caught:
        store_attribution(database, result)
    assert "duplicate" in str(caught.value)


def test_storing_refuses_a_frame_missing_the_probability(database):
    result = _result()
    result.cells = pd.DataFrame({"prcfo": KEYS})
    with pytest.raises(HitAttributionError):
        store_attribution(database, result)


# --------------------------------------------------------------------------- #
#  promote_calls -- and the audit that makes it undoable
# --------------------------------------------------------------------------- #

def test_promoting_refuses_an_annotation_column_that_is_not_a_name(database):
    """The column name is interpolated into SQL. Anything but letters,
    numbers and underscores is refused rather than quoted-and-hoped."""
    store_attribution(database, _result())
    for bad in ("drop table png_list; --", "a b", "col-1", '"x"'):
        with pytest.raises(HitAttributionError):
            promote_calls(database, "run-1", bad)


def test_promoting_creates_the_column_when_it_is_new(database):
    store_attribution(database, _result())
    promotion = promote_calls(database, "run-1", "brand_new_column")

    assert promotion
    with sqlite3.connect(database) as db:
        columns = {row[1] for row in db.execute('PRAGMA table_info("png_list")')}
    assert "brand_new_column" in columns


def test_promoting_refuses_a_png_list_with_no_object_key(tmp_path):
    """Without prcfo there is nothing to promote ONTO."""
    path = tmp_path / "bad.db"
    with sqlite3.connect(str(path)) as db:
        pd.DataFrame({"png_path": ["/x.png"]}).to_sql("png_list", db,
                                                      index=False)
    with pytest.raises(HitAttributionError) as caught:
        promote_calls(str(path), "run-1", "annotation")
    assert "prcfo" in str(caught.value)


# --------------------------------------------------------------------------- #
#  The round trip. This is the one that matters.
# --------------------------------------------------------------------------- #

def _annotations(database, column="test_annotation"):
    with sqlite3.connect(database) as db:
        rows = db.execute(
            f'SELECT prcfo, "{column}" FROM png_list ORDER BY prcfo').fetchall()
    return dict(rows)


def test_a_revert_restores_exactly_what_was_there_before(database):
    """Including the NULLs.

    Two of these rows carried a hand annotation and two did not. A revert
    that writes back "" or leaves the promoted value on the previously-empty
    rows is data loss wearing an undo's clothes.
    """
    before = _annotations(database)
    assert before[KEYS[0]] == "keep-me"
    assert before[KEYS[1]] is None

    store_attribution(database, _result())
    promotion = promote_calls(database, "run-1", "test_annotation")

    during = _annotations(database)
    assert during != before, "the promotion must actually change something"

    reverted = revert_promotion(database, promotion)
    assert reverted > 0

    assert _annotations(database) == before


def test_reverting_an_unknown_promotion_changes_nothing(database):
    """Zero rows, not an exception: an undo of something that never happened
    is a no-op, and raising would make a double-click destructive."""
    before = _annotations(database)

    assert revert_promotion(database, "no-such-promotion-id") == 0
    assert _annotations(database) == before


def test_reverting_twice_is_harmless(database):
    """The audit is marked undone, so the second call finds nothing. A
    second revert must not re-apply anything."""
    store_attribution(database, _result())
    promotion = promote_calls(database, "run-1", "test_annotation")
    before = _annotations(database)

    first = revert_promotion(database, promotion)
    after_first = _annotations(database)
    second = revert_promotion(database, promotion)

    assert first > 0
    assert second == 0
    assert _annotations(database) == after_first
    assert after_first != before
