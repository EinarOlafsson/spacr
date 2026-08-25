"""The queue survives a bad file, a bad row, and a pipeline it cannot find.

A plate queue is the plan for an overnight run of twenty plates. The failure
that matters is not an exception -- it is a queue that comes back from disk
one item short, or a CSV row that silently becomes a plate pointing at
nothing. These pin the places where the queue must either keep going loudly
or refuse outright.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from spacr.qt import plate_queue as pq
from spacr.qt.plate_queue import (PlateQueue, QueueItem, Status,
                                  default_runner, import_plates_from_csv)


def test_a_queue_iterates_in_the_order_it_will_run(tmp_path: Path):
    """Iterating the queue yields its items in run order.

    The screen and ``run_queue`` both walk the queue directly; an iterator
    that returned a copy in another order, or the underlying dict, would run
    the plates in an order the user never saw.
    """
    queue = PlateQueue(path=tmp_path / "queue.json")
    queue.add(QueueItem.build("mask", {"src": "/plates/one"}))
    queue.add(QueueItem.build("mask", {"src": "/plates/two"}))

    assert [item.label for item in queue] == ["/plates/one", "/plates/two"]
    assert len(list(queue)) == len(queue)


def test_a_queue_that_cannot_be_written_says_so_and_keeps_the_plan(
        tmp_path: Path, caplog):
    """An unwritable snapshot warns instead of losing the in-memory queue.

    The snapshot exists so a crash does not lose the plan; a failure to take
    it must not itself destroy the plan the user just built, and must not
    pass silently either -- the user would restart and find the queue empty
    with no explanation.
    """
    queue = PlateQueue(path=tmp_path / "queue.json")
    queue._path = tmp_path / "no" / "such" / "dir" / "queue.json"

    with caplog.at_level(logging.WARNING, logger="spacr.qt.plate_queue"):
        queue.add(QueueItem.build("mask", {"src": "/plates/one"}))

    assert len(queue) == 1
    assert "failed to persist queue" in caplog.text


def test_one_malformed_entry_does_not_cost_the_other_plates(
        tmp_path: Path, caplog):
    """A queue file with a broken row loads the rows that are intact.

    Dropping the whole file would throw away nineteen good plates because of
    one; loading it silently would leave the user counting rows to find out.
    """
    path = tmp_path / "queue.json"
    path.write_text(json.dumps({"items": [
        {"id": "a1", "app_key": "mask", "settings": {"src": "/plates/one"}},
        {"app_key": "mask", "settings": {"src": "/plates/broken"}},
        {"id": "c3", "app_key": "mask", "settings": {"src": "/plates/two"}},
    ]}))

    with caplog.at_level(logging.INFO, logger="spacr.qt.plate_queue"):
        queue = PlateQueue(path=path)

    assert [item.id for item in queue] == ["a1", "c3"]
    assert "skipping malformed queue entry" in caplog.text


def test_a_csv_row_with_no_source_is_not_a_plate(tmp_path: Path):
    """Rows with a blank ``src`` are skipped rather than queued.

    A trailing blank line, or a row whose path cell was cleared, would
    otherwise become an item whose src is the base settings' -- running the
    same plate twice under two names.
    """
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text("src,cell_diameter\n"
                        "/plates/one,30\n"
                        "   ,40\n"
                        ",50\n")

    items = import_plates_from_csv(csv_path, {"src": "/base", "nucleus": 1})

    assert [i.settings["src"] for i in items] == ["/plates/one"]
    assert items[0].settings["cell_diameter"] == 30
    assert items[0].settings["nucleus"] == 1


def test_a_csv_says_no_as_well_as_yes(tmp_path: Path):
    """``false``/``no`` cells become False, not the truthy string.

    Every non-empty string is truthy, so a column left as text would turn
    "no" into "yes" for every plate in the batch -- the one class of CSV
    error that changes the science and never raises.
    """
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text("src,plot,save,verbose,timelapse\n"
                        "/plates/one,false,no,true,none\n")

    settings = import_plates_from_csv(csv_path, {})[0].settings

    assert settings["plot"] is False
    assert settings["save"] is False
    assert settings["verbose"] is True
    assert settings["timelapse"] is None


def test_the_default_runner_calls_the_pipeline_the_item_names(monkeypatch):
    """The runner resolves the app key and hands it exactly the settings.

    This is the only path between a queued plate and the code that processes
    it, so a runner that passed the item, or the wrong app's entry point,
    would run twenty plates through the wrong pipeline overnight.
    """
    from spacr.qt import bridge

    seen = {}
    monkeypatch.setattr(bridge, "resolve_pipeline_entry",
                        lambda key: (seen.__setitem__("key", key)
                                     or (lambda s: seen.__setitem__("settings", s))))

    item = QueueItem.build("mask", {"src": "/plates/one", "nucleus": 2})
    default_runner(item)

    assert seen["key"] == "mask"
    assert seen["settings"] == {"src": "/plates/one", "nucleus": 2}


def test_a_queue_item_for_an_unknown_app_refuses_to_run(monkeypatch):
    """An app key with no pipeline raises and names the key.

    Silently doing nothing would mark the plate SUCCESS in the journal with
    no output anywhere -- the user would find out days later.
    """
    from spacr.qt import bridge

    monkeypatch.setattr(bridge, "resolve_pipeline_entry", lambda key: None)
    item = QueueItem.build("annotate", {"src": "/plates/one"})

    with pytest.raises(RuntimeError, match="annotate"):
        default_runner(item)
