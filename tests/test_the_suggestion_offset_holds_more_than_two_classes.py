"""379-C, measured: the offset scheme is limited by class VALUE, not COUNT.

379-C is written as "extending the scheme past two classes is an open
decision", and `suggest.py` carries a guard whose message says the same. On
2026-09-10 the maintainer asked for it to be extended -- and the premise
turned out to be false, so this file is the measurement that corrects it.

`suggest_from_scores` takes a `classes` list and argmaxes across N score
columns; nothing in it counts to two. `SUGGESTION_OFFSET` is 10, so a
suggested 1 stores as 11 and a suggested 9 as 19, and neither collides with
any class value below 10.

    THREE TO NINE CLASSES ALREADY WORK. What does not work is a class VALUE
    of 10 or more, which is a different limit and a far less likely one --
    an annotation column holds small integers because a person types them.

These tests exist so that stays true. It was never covered, so it could
have stopped working at any time without anything failing.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr.suggest import SUGGESTION_OFFSET, suggest_from_scores


def _database(path, classes, unlabelled_scores):
    """A crop table with `classes` annotated and some scored unlabelled rows."""
    from spacr.active_learning import ROUND_PRED_PREFIX

    width = len(classes)
    columns = ", ".join(f"{ROUND_PRED_PREFIX}{i} REAL" for i in range(width))
    con = sqlite3.connect(path)
    con.execute(f"CREATE TABLE png_list (png_path TEXT, my_class INTEGER, "
                f"{columns})")
    rows = [(f"/crops/a{i}.png", value) + (0.0,) * width
            for i, value in enumerate(classes * 2)]
    rows += [(f"/crops/u{i}.png", None) + tuple(scores)
             for i, scores in enumerate(unlabelled_scores)]
    con.executemany(
        f"INSERT INTO png_list VALUES (?, ?, {', '.join('?' * width)})", rows)
    con.commit()
    con.close()


def test_three_classes_round_trip_through_the_offset(tmp_path):
    """The case 379-C says is undecided, and it already works.

    Not "should be safe" -- driven. Three classes in, three suggestions out,
    each stored above the offset and each decoding back to the class it
    came from.
    """
    db = str(tmp_path / "measurements.db")
    _database(db, [1, 2, 3], [
        (0.7, 0.2, 0.1),
        (0.1, 0.8, 0.1),
        (0.2, 0.1, 0.7),
    ])

    out = suggest_from_scores(db, "my_class")

    assert out.classes == [1, 2, 3]
    assert out.scored == 3
    assert sorted(out.frame["suggested"]) == [1, 2, 3]
    stored = sorted(int(v) for v in out.frame["stored"])
    assert stored == [11, 12, 13]
    assert all(v > SUGGESTION_OFFSET for v in stored)
    assert sorted(v - SUGGESTION_OFFSET for v in stored) == [1, 2, 3]


def test_nine_classes_are_the_edge_and_they_still_decode(tmp_path):
    """NINE, not two, is where the scheme actually ends.

    A suggested 9 stores as 19 and a real 9 is 9, so they are still
    distinguishable. Ten is where it would break, and the guard in
    `write_suggestions` refuses that rather than storing it.
    """
    db = str(tmp_path / "measurements.db")
    classes = list(range(1, 10))
    scores = []
    for index in range(len(classes)):
        row = [0.01] * len(classes)
        row[index] = 0.9
        scores.append(tuple(row))
    _database(db, classes, scores)

    out = suggest_from_scores(db, "my_class")

    assert out.classes == classes
    assert sorted(out.frame["suggested"]) == classes
    stored = sorted(int(v) for v in out.frame["stored"])
    assert stored == [c + SUGGESTION_OFFSET for c in classes]
    assert sorted(v - SUGGESTION_OFFSET for v in stored) == classes


def test_the_documented_limit_is_the_class_value_not_the_class_count():
    """Pinned as prose, because the instruction says otherwise.

    379-C's own words are "extending the scheme past two classes", and the
    guard's message repeats it. Anybody reading either would conclude a
    third class needs building. It does not; a class VALUE at or above the
    offset does.
    """
    assert SUGGESTION_OFFSET == 10, (
        "the offset moved -- the two tests above pin 1..9 as the workable "
        "range and both need re-deriving from the new value")
