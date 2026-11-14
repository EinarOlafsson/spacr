"""The plate queue's run order, its crash-recovery snapshot, and CSV import.

A plate queue is the plan for an overnight run.  What matters is that it
iterates in the order it will run, that its snapshot is rewritten on every real
change (and only then), that a snapshot it cannot write is reported without
losing the plan in memory, and that a CSV of plates becomes exactly the
settings the user typed, with "no" meaning False.
"""
from __future__ import annotations

import json
import logging
import re

import pytest

from spacr.qt import plate_queue as pq
from spacr.qt.plate_queue import (PlateQueue, QueueItem, Status,
                                  import_plates_from_csv)


def test_queue_items_run_in_order_and_the_snapshot_follows_real_changes(
        tmp_path, caplog, subtests):
    """Build labels, iteration order, save-on-change and an unwritable snapshot."""
    with subtests.test(msg="build: label from src, explicit, or minted"):
        base = {"src": "/plates/one", "nested": [1]}
        from_src = QueueItem.build("mask", base)
        assert (from_src.app_key, from_src.label, from_src.status) == (
            "mask", "/plates/one", Status.QUEUED)
        assert re.fullmatch(r"[0-9a-f]{8}", from_src.id)
        assert from_src.settings == base and from_src.settings is not base
        assert QueueItem.build("measure", base, label="Plate A").label == \
            "Plate A"
        anonymous = QueueItem.build("mask", {"src": ""})
        assert anonymous.label == f"plate-{anonymous.id}"
        assert (from_src.start_ts, from_src.end_ts, from_src.error,
                from_src.run_dir) == (None, None, None, None)

    path = tmp_path / "queue.json"
    queue = PlateQueue(path=path)
    first = QueueItem.build("mask", {"src": "/plates/one"})
    second = QueueItem.build("mask", {"src": "/plates/two"})
    queue.add(first)
    queue.add(second)

    with subtests.test(msg="iteration is run order, and so is the snapshot"):
        assert [item.label for item in queue] == ["/plates/one",
                                                  "/plates/two"]
        assert list(queue) == queue.items() == [first, second]
        assert len(queue) == 2
        stored = json.loads(path.read_text(encoding="utf-8"))["items"]
        assert [(row["id"], row["label"], row["status"]) for row in stored] \
            == [(first.id, "/plates/one", "queued"),
                (second.id, "/plates/two", "queued")]
        reloaded = PlateQueue(path=path)
        assert [(i.id, i.label, i.status) for i in reloaded] == [
            (first.id, "/plates/one", Status.QUEUED),
            (second.id, "/plates/two", Status.QUEUED)]

    # Each patch, applied after the snapshot was deleted: a save recreates
    # the file, so its presence afterwards says whether update() wrote.
    patches = [
        ("same values", first.id, {"status": Status.QUEUED,
                                   "label": "/plates/one"}, False),
        ("a field the item does not have", first.id,
         {"not_a_queue_field": 17}, False),
        ("an unknown id", "no-such-id", {"status": Status.RUNNING}, False),
        ("a real change", first.id, {"status": Status.RUNNING}, True),
    ]
    for case, item_id, fields, writes in patches:
        with subtests.test(msg=f"update: {case}"):
            path.unlink(missing_ok=True)
            queue.update(item_id, **fields)
            assert path.exists() is writes
            assert not hasattr(first, "not_a_queue_field")
    assert first.status is Status.RUNNING
    assert [row["status"] for row in json.loads(
        path.read_text(encoding="utf-8"))["items"]] == ["running", "queued"]

    with subtests.test(msg="an unwritable snapshot warns and keeps the plan"):
        queue._path = tmp_path / "no" / "such" / "dir" / "queue.json"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spacr.qt.plate_queue"):
            queue.add(QueueItem.build("mask", {"src": "/plates/three"}))
        assert [item.label for item in queue] == [
            "/plates/one", "/plates/two", "/plates/three"]
        warnings = [r.getMessage() for r in caplog.records
                    if r.name == "spacr.qt.plate_queue"]
        assert len(warnings) == 1
        assert warnings[0].startswith("failed to persist queue: ")
        assert not queue._path.exists()


def test_import_plates_from_csv_coerces_each_cell_as_typed(tmp_path,
                                                           subtests):
    """Rows without a source are skipped; cells become int, float, bool, None or text."""
    csv_path = tmp_path / "plates.csv"
    csv_path.write_text(
        "src,cell_size,ratio,plot,save,verbose,overlay,timelapse,mode,note\n"
        "/plates/one,30,0.5,false,no,true,yes,none,null,two channels\n"
        "  /plates/two  ,-4,1e3,False,NO,TRUE,Yes,NULL,,\n"
        "   ,40,,,,,,,,\n"
        ",50,,,,,,,,\n"
        "/plates/short,25\n", encoding="utf-8")
    base = {"src": "/base", "nucleus": 1, "channels": [0, 1], "note": ""}

    items = import_plates_from_csv(csv_path, base, app_key="measure")

    assert [i.label for i in items] == ["/plates/one", "/plates/two",
                                        "/plates/short"]
    assert [i.app_key for i in items] == ["measure"] * 3
    expected = [
        {"src": "/plates/one", "nucleus": 1, "channels": [0, 1],
         "cell_size": 30, "ratio": 0.5, "plot": False, "save": False,
         "verbose": True, "overlay": True, "timelapse": None, "mode": None,
         "note": "two channels"},
        {"src": "/plates/two", "nucleus": 1, "channels": [0, 1],
         "cell_size": -4, "ratio": 1000.0, "plot": False, "save": False,
         "verbose": True, "overlay": True, "timelapse": None, "mode": None,
         "note": None},
        # A short row: DictReader hands None for the missing cells, which
        # stays None rather than becoming the string "None".
        {"src": "/plates/short", "nucleus": 1, "channels": [0, 1],
         "cell_size": 25, "ratio": None, "plot": None, "save": None,
         "verbose": None, "overlay": None, "timelapse": None, "mode": None,
         "note": None},
    ]
    for item, want in zip(items, expected):
        with subtests.test(msg=item.label):
            assert item.settings == want
            for key in ("cell_size", "ratio", "plot", "verbose"):
                assert type(item.settings[key]) is type(want[key])

    with subtests.test(msg="base settings are deep-copied per row"):
        assert items[0].settings["channels"] is not items[1].settings[
            "channels"]
        assert items[0].settings["channels"] is not base["channels"]
        assert base == {"src": "/base", "nucleus": 1, "channels": [0, 1],
                        "note": ""}

    with subtests.test(msg="a missing file is refused"):
        with pytest.raises(FileNotFoundError):
            import_plates_from_csv(tmp_path / "absent.csv", {})

    with subtests.test(msg="the default pipeline is mask"):
        assert [i.app_key for i in pq.import_plates_from_csv(
            csv_path, {})] == ["mask"] * 3
