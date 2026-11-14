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

from spacr.qt.plate_queue import PlateQueue, QueueItem, default_runner


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
    ]}), encoding="utf-8")

    with caplog.at_level(logging.INFO, logger="spacr.qt.plate_queue"):
        queue = PlateQueue(path=path)

    assert [item.id for item in queue] == ["a1", "c3"]
    assert "skipping malformed queue entry" in caplog.text


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
