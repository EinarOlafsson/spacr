"""The Suggest button, against a database whose right answer we planted.

A suggestion engine can be wrong in a way that looks healthy: it emits labels
of the right shape, and a fixed fraction of them agree with the truth by
chance. So the fixture plants a measurement that SEPARATES the classes
perfectly, and the tests demand the suggestions come back matching it.

The rest of these tests are about what it must refuse to do. Losing a human's
annotation costs hours of work and is not noticed until a run comes out wrong,
so the refusals matter more than the accuracy.
"""
from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("xgboost")

from spacr import suggest

N_PER_CLASS = 40          # comfortably over MIN_PER_CLASS
N_UNANNOTATED = 30


def _build(tmp_path, annotate_both=True, n_annotated=N_PER_CLASS):
    """A measurements.db where `area` separates class 1 from class 2.

    :param tmp_path: pytest's directory.
    :param annotate_both: annotate both classes, or only class 1.
    :param n_annotated: how many of each class carry a label.
    :returns: ``(db_path, truth)`` where truth maps png_path to its class.
    """
    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, "
               "fieldID TEXT, object_label INTEGER, area REAL, perimeter REAL)")
    db.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, columnID TEXT,"
               " fieldID TEXT, cell_id TEXT, png_path TEXT, test INTEGER)")
    rng = np.random.default_rng(3)
    truth = {}
    label_id = 0

    def add(cls, annotated):
        nonlocal label_id
        label_id += 1
        # Class 1 is small, class 2 is large, with no overlap: separable.
        area = rng.normal(400 if cls == 1 else 1600, 40)
        db.execute("INSERT INTO cell VALUES ('p1','A','1','f1',?,?,?)",
                   (label_id, float(area), float(area) * 0.4))
        png = f"/crops/{label_id}.png"
        db.execute("INSERT INTO png_list VALUES "
                   "('p1','A','1','f1',?,?,?)",
                   (f"o{label_id}", png, cls if annotated else None))
        truth[png] = cls

    for _ in range(n_annotated):
        add(1, True)
        add(2, annotate_both)
    for _ in range(N_UNANNOTATED):
        add(1, False)
        add(2, False)
    db.commit(); db.close()
    return str(path), truth


def test_the_suggestions_match_the_class_we_planted(tmp_path):
    """With a measurement that separates the classes, it must find them."""
    db_path, truth = _build(tmp_path)
    result = suggest.suggest(db_path, "test")

    assert not result.frame.empty, result.note
    assert not result.was_one_class
    assert "area" in result.features

    wrong = [(p, s) for p, s in zip(result.frame["png_path"],
                                    result.frame["suggested"])
             if truth[p] != s]
    assert not wrong, f"{len(wrong)} of {len(result.frame)} suggested wrongly"


def test_they_come_back_sorted_so_doubt_sits_together(tmp_path):
    """No confidence floor, so the ORDER is what makes review efficient."""
    db_path, _truth = _build(tmp_path)
    frame = suggest.suggest(db_path, "test").frame
    confidences = list(frame["confidence"])
    assert confidences == sorted(confidences, reverse=True)


def test_a_suggestion_is_stored_as_its_own_value(tmp_path):
    """A suggested 1 is stored as 11, so nothing can mistake it for an answer."""
    db_path, _truth = _build(tmp_path)
    result = suggest.suggest(db_path, "test")
    written = suggest.write_suggestions(db_path, "test", result.frame)
    assert written > 0

    with sqlite3.connect(db_path) as db:
        values = [r[0] for r in db.execute(
            'SELECT test FROM png_list WHERE test IS NOT NULL')]
    assert set(values) <= {1, 2, 11, 12}
    assert any(v > suggest.SUGGESTION_OFFSET for v in values)


def test_a_suggestion_never_overwrites_an_annotation(tmp_path):
    """THE RULE THAT PROTECTS HOURS OF WORK.

    Every annotated row must hold exactly what it held before, whatever the
    model thought of it.
    """
    db_path, _truth = _build(tmp_path)
    with sqlite3.connect(db_path) as db:
        before = dict(db.execute(
            'SELECT png_path, test FROM png_list WHERE test IS NOT NULL'))

    result = suggest.suggest(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)

    with sqlite3.connect(db_path) as db:
        after = dict(db.execute(
            'SELECT png_path, test FROM png_list WHERE png_path IN '
            f"({','.join('?' * len(before))})", list(before)))
    assert after == before, "an annotation was overwritten by a suggestion"


def test_keeping_turns_suggestions_into_annotations_and_throwing_clears_them(
        tmp_path):
    """Both bulk actions touch ONLY suggestions."""
    db_path, _truth = _build(tmp_path)
    result = suggest.suggest(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)

    with sqlite3.connect(db_path) as db:
        annotated_before = db.execute(
            'SELECT COUNT(*) FROM png_list WHERE test IN (1,2)').fetchone()[0]

    kept = suggest.resolve_suggestions(db_path, "test", keep=True)
    assert kept > 0
    with sqlite3.connect(db_path) as db:
        suggestions_left = db.execute(
            'SELECT COUNT(*) FROM png_list WHERE test > 10').fetchone()[0]
        annotated_after = db.execute(
            'SELECT COUNT(*) FROM png_list WHERE test IN (1,2)').fetchone()[0]
    assert suggestions_left == 0
    assert annotated_after == annotated_before + kept

    # And throwing away puts them back to nothing.
    second = tmp_path / "again"
    second.mkdir()
    fresh, _ = _build(second)
    result2 = suggest.suggest(fresh, "test")
    suggest.write_suggestions(fresh, "test", result2.frame)
    thrown = suggest.resolve_suggestions(fresh, "test", keep=False)
    assert thrown > 0
    with sqlite3.connect(fresh) as db:
        assert db.execute(
            'SELECT COUNT(*) FROM png_list WHERE test > 10').fetchone()[0] == 0


def test_too_few_annotations_is_a_refusal_with_a_reason(tmp_path):
    """Below the floor it says what it needs instead of training on noise."""
    db_path, _truth = _build(tmp_path, n_annotated=5)
    result = suggest.suggest(db_path, "test")
    assert result.frame.empty
    assert "annotate more" in result.note or "are needed" in result.note


def test_one_annotated_class_is_labelled_as_a_ranking_not_a_verdict(tmp_path):
    """The drawn negatives are mostly-negative, and the caller must be told.

    A ranking presented as a classification invites a bulk accept, which is
    exactly the mistake this flag exists to prevent.
    """
    db_path, _truth = _build(tmp_path, annotate_both=False)
    result = suggest.suggest(db_path, "test")
    assert result.was_one_class is True
    assert "ranking" in result.note.lower()
    assert "mostly negative" in result.note.lower()
