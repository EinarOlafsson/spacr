"""Suggestions a reviewer can accept or throw away, without losing a label.

The model is not this module's job -- `active_learning.retrain_round` already
fits on every label, scores on a grouped held-out split, and writes per-class
probabilities into `png_list`. What is tested here is the part that is new:
turning those probabilities into values a reviewer acts on IN BULK, without
any chance of a human's annotation being overwritten by a machine's guess.

The refusals matter more than the ranking. A lost annotation costs hours and
is not noticed until a run comes out wrong.
"""
from __future__ import annotations

import sqlite3

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from spacr import suggest


def _db(tmp_path, *, scored=True, annotated=(1, 2), n_unlabelled=6):
    """A crop table with annotations and, optionally, retrain scores.

    :param tmp_path: pytest's directory.
    :param scored: write the ``al_prob_*`` columns a retrain would leave.
    :param annotated: class values to plant as real annotations.
    :param n_unlabelled: how many crops carry no annotation.
    :returns: ``(db_path, expected)`` where expected maps png_path to the
        class its scores favour.
    """
    from spacr.active_learning import ROUND_PRED_PREFIX

    path = tmp_path / "measurements.db"
    db = sqlite3.connect(path)
    cols = ["png_path TEXT", "test INTEGER"]
    if scored:
        cols += [f"{ROUND_PRED_PREFIX}0 REAL", f"{ROUND_PRED_PREFIX}1 REAL"]
    db.execute(f"CREATE TABLE png_list ({', '.join(cols)})")

    expected = {}
    for i, cls in enumerate(annotated):
        # A real annotation, with scores that DISAGREE with it on purpose:
        # nothing may rewrite these.
        values = [f"/crops/labelled_{i}.png", cls]
        if scored:
            values += [0.5, 0.5]
        db.execute(f"INSERT INTO png_list VALUES ({','.join('?' * len(values))})",
                   values)

    for i in range(n_unlabelled):
        png = f"/crops/blank_{i}.png"
        # Alternate which class the scores favour, with a confidence that
        # descends, so the sort has something to order.
        p1 = 0.95 - i * 0.05 if i % 2 == 0 else 0.05 + i * 0.05
        values = [png, None]
        if scored:
            values += [1.0 - p1, p1]
        db.execute(f"INSERT INTO png_list VALUES ({','.join('?' * len(values))})",
                   values)
        expected[png] = 2 if p1 > 0.5 else 1
    db.commit(); db.close()
    return str(path), expected


def test_the_suggestion_is_the_class_the_scores_favour(tmp_path):
    """argmax of what the retrain wrote, mapped back to the real class value."""
    db_path, expected = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")

    assert not result.frame.empty, result.note
    assert result.classes == [1, 2]
    got = dict(zip(result.frame["png_path"], result.frame["suggested"]))
    assert got == expected


def test_they_arrive_sorted_so_the_doubtful_ones_sit_together(tmp_path):
    """No confidence floor, so the ORDER is what makes review efficient."""
    db_path, _ = _db(tmp_path)
    frame = suggest.suggest_from_scores(db_path, "test").frame
    confidences = list(frame["confidence"])
    assert confidences == sorted(confidences, reverse=True)


def test_a_suggestion_is_stored_as_its_own_value(tmp_path):
    """A suggested 1 is stored as 11, so nothing mistakes it for an answer."""
    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    assert suggest.write_suggestions(db_path, "test", result.frame) > 0

    with sqlite3.connect(db_path) as db:
        values = {r[0] for r in db.execute(
            "SELECT test FROM png_list WHERE test IS NOT NULL")}
    assert values <= {1, 2, 11, 12}
    assert any(v > suggest.SUGGESTION_OFFSET for v in values)
    assert suggest.pending_suggestions(db_path, "test") > 0


def test_a_suggestion_never_overwrites_an_annotation(tmp_path):
    """THE RULE THAT PROTECTS HOURS OF WORK.

    The two annotated crops carry scores of 0.5/0.5, so the model has an
    opinion about them. It must not be written anywhere near them.
    """
    db_path, _ = _db(tmp_path)
    with sqlite3.connect(db_path) as db:
        before = dict(db.execute(
            "SELECT png_path, test FROM png_list WHERE test IS NOT NULL"))
    assert before, "the fixture planted no annotations"

    result = suggest.suggest_from_scores(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)

    with sqlite3.connect(db_path) as db:
        after = dict(db.execute(
            "SELECT png_path, test FROM png_list WHERE png_path IN "
            f"({','.join('?' * len(before))})", list(before)))
    assert after == before, "an annotation was overwritten by a suggestion"


def test_keeping_promotes_only_suggestions(tmp_path):
    """KEEP rewrites 11 -> 1 and leaves every real annotation alone."""
    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)

    with sqlite3.connect(db_path) as db:
        real_before = db.execute(
            "SELECT COUNT(*) FROM png_list WHERE test IN (1,2)").fetchone()[0]

    kept = suggest.resolve_suggestions(db_path, "test", keep=True)
    assert kept == len(result.frame)
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM png_list WHERE test > 10"
                          ).fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM png_list WHERE test IN (1,2)"
                          ).fetchone()[0] == real_before + kept


def test_throwing_away_clears_only_suggestions(tmp_path):
    """THROW sets them back to NULL and leaves annotations untouched."""
    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)

    with sqlite3.connect(db_path) as db:
        real_before = dict(db.execute(
            "SELECT png_path, test FROM png_list WHERE test IN (1,2)"))

    thrown = suggest.resolve_suggestions(db_path, "test", keep=False)
    assert thrown == len(result.frame)
    with sqlite3.connect(db_path) as db:
        assert suggest.pending_suggestions(db_path, "test") == 0
        real_after = dict(db.execute(
            "SELECT png_path, test FROM png_list WHERE test IN (1,2)"))
    assert real_after == real_before


def test_one_crop_can_be_resolved_without_touching_the_rest(tmp_path):
    """A reviewer accepts a screen and rejects three of it."""
    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    suggest.write_suggestions(db_path, "test", result.frame)
    one = [result.frame["png_path"].iloc[0]]

    assert suggest.resolve_suggestions(db_path, "test", keep=True,
                                       paths=one) == 1
    assert suggest.pending_suggestions(db_path, "test") == len(result.frame) - 1


def test_an_unscored_database_says_to_retrain_rather_than_guessing(tmp_path):
    """No scores is not "no suggestions"; it is "press Retrain first".

    Fitting a second model here would give a reviewer an opinion that differs
    from the one the queue is ranked by, with no way to tell which is right.
    """
    db_path, _ = _db(tmp_path, scored=False)
    result = suggest.suggest_from_scores(db_path, "test")
    assert result.frame.empty
    assert "Retrain" in result.note


# ---------------------------------------------------------------------------
# A suggestion is not a label, wherever the labels are read
# ---------------------------------------------------------------------------

def test_a_suggestion_is_never_read_back_as_a_class():
    """The offset marks a guess; nothing may fit it as an answer.

    `write_suggestions` stores a proposal in the SAME column as the answers,
    offset by `SUGGESTION_OFFSET` so the two cannot collide -- a suggested 1
    becomes 11. Anything that reads that column with a bare `IS NOT NULL`
    therefore sees classes 11 and 12 and treats them as labels a person
    wrote.

    THE COST IS NOT A WRONG COUNT. `retrain_round` would FIT them: a model
    trained on its own previous output, and a model card that does not say
    so, because from the fitter's side those rows are indistinguishable from
    human work. Guarding it in the GUI is not enough -- a script calling
    `retrain_round` directly was never guarded -- so the rule lives where the
    labels are read.
    """
    from spacr.active_learning import _is_suggestion

    assert not _is_suggestion(1)
    assert not _is_suggestion(2.0)          # 2.0 and 2 are one class
    assert not _is_suggestion(None)
    assert _is_suggestion(suggest.SUGGESTION_OFFSET + 1)
    assert _is_suggestion(suggest.SUGGESTION_OFFSET + 2)


def test_class_counts_reports_only_what_a_person_labelled(tmp_path):
    """The dialog's census may not invent classes out of the machine's guesses."""
    from spacr.qt.annotate_engine import class_counts

    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    assert suggest.write_suggestions(db_path, "test", result.frame) > 0

    with sqlite3.connect(db_path) as db:
        stored = {r[0] for r in db.execute(
            "SELECT test FROM png_list WHERE test IS NOT NULL")}
    assert any(v > suggest.SUGGESTION_OFFSET for v in stored), (
        "the fixture must actually contain suggestions or this proves nothing")

    counted = {value for value, _n in class_counts(db_path, "test")}
    assert counted == {1, 2}, counted


def test_a_real_class_at_the_offset_stops_suggest_rather_than_being_eaten(
        tmp_path):
    """379-C's collision, from the side that costs somebody their work.

    `write_suggestions` already refuses to SUGGEST a class at or above the
    offset. This is the other direction: a column that ALREADY holds one.

    A suggested 1 is stored as 11, `pending_suggestions` counts every value
    above the offset as outstanding, and `is_suggestion` reads one the same
    way -- so a person who annotated class 11 has a row the bulk KEEP would
    rewrite to a 1, silently, with no way to notice until a run came out
    wrong. PART 2's first rule is that a suggestion must never overwrite a
    human annotation, and this was the last route by which it still could.

    Refusing is the whole fix and is deliberately not a repair: whether the
    scheme should extend past two classes is 379-C, which is the
    maintainer's decision and is still open. What is not open is that it
    must never collide in silence.
    """
    db_path, _ = _db(tmp_path)
    result = suggest.suggest_from_scores(db_path, "test")
    assert not result.frame.empty, result.note

    # One human answer at the offset, planted after the model was scored so
    # it cannot be mistaken for a proposal this run produced.
    with sqlite3.connect(db_path) as db:
        db.execute("UPDATE png_list SET test = ? WHERE png_path = ("
                   "SELECT png_path FROM png_list WHERE test IS NOT NULL "
                   "LIMIT 1)", (suggest.SUGGESTION_OFFSET + 1,))

    with pytest.raises(ValueError) as caught:
        suggest.write_suggestions(db_path, "test", result.frame)
    message = str(caught.value)
    assert str(suggest.SUGGESTION_OFFSET) in message
    assert "379-C" in message, "the refusal has to say which decision it waits on"

    # AND IT WROTE NOTHING. A refusal that had already written half the rows
    # would be worse than the collision it is preventing.
    with sqlite3.connect(db_path) as db:
        stored = {r[0] for r in db.execute(
            "SELECT test FROM png_list WHERE test IS NOT NULL")}
    assert stored <= {1, 2, suggest.SUGGESTION_OFFSET + 1}, stored
