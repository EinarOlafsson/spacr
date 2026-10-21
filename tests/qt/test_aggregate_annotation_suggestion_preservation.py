"""Re-thresholding preserves human answers, pending edits and recorded judgements."""

import numpy as np
import pytest

from spacr.qt.screens import annotate
from spacr.suggest import SUGGESTION_OFFSET


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = annotate.AnnotateScreen()
    qtbot.addWidget(widget)
    return widget


def test_rethresholding_updates_only_replaceable_unedited_slots_and_cached_rows(screen):
    screen._page_paths = [("human", 1), ("pending", None), ("new", None),
                          ("old", SUGGESTION_OFFSET + 2), ("untouched", None)]
    screen._pending_updates = {"pending": 2}
    screen._filtered_rows = list(screen._page_paths)
    screen._show_rethresholded({"human": 2, "pending": 1, "new": 2, "old": None})
    assert screen._page_paths == [("human", 1), ("pending", None),
                                  ("new", SUGGESTION_OFFSET + 2), ("old", None),
                                  ("untouched", None)]
    assert screen._filtered_rows == [("human", 1), ("pending", SUGGESTION_OFFSET + 1),
                                     ("new", SUGGESTION_OFFSET + 2), ("old", None),
                                     ("untouched", None)]
    screen._filtered_rows = None
    screen._show_rethresholded({"untouched": 1})
    assert screen._page_paths[-1] == ("untouched", SUGGESTION_OFFSET + 1)


def test_missing_or_malformed_slot_never_invents_a_class_or_judgement(screen):
    screen._page_paths = [("bad", "invalid"), ("unlabelled", None), ("human", 2)]
    assert screen._displayed_class(0) == "invalid"
    assert not screen._is_suggested_slot(0)
    assert screen._displayed_class(1) is None
    assert screen._displayed_class(2) == 2
    assert screen._verdict_value(-1) is None
    assert not screen._set_verdict(99, 1)
    assert screen._judgement_of(99) is None
    assert screen._judge_contribution(99) == (0, 0, 0)
    screen._suggest_cache = {"index": {"bad": 0}, "classes": [1, 2],
                             "matrix": np.array([[0.1, 0.9]])}
    assert screen._suggestion_tip(0) == ""
    screen._page_paths[0] = ("bad", SUGGESTION_OFFSET + 2)
    assert "0.90" in screen._suggestion_tip(0)
    screen._suggest_cache["matrix"] = None
    assert screen._suggestion_tip(0) == ""
    screen._suggest_cache["index"] = {}
    assert screen._suggestion_tip(0) == ""


def test_database_set_aside_preserves_sqlite_companions_and_reports_missing_source(tmp_path):
    database = tmp_path / "measurements.db"
    database.write_bytes(b"database")
    for suffix in ("-wal", "-shm"):
        database.with_name(database.name + suffix).write_bytes(suffix.encode())
    target = annotate._set_aside_example_database(database)
    assert target.read_bytes() == b"database"
    assert not database.exists()
    for suffix in ("-wal", "-shm"):
        assert target.with_name(target.name + suffix).read_bytes() == suffix.encode()
    assert annotate._set_aside_example_database(database) is None
