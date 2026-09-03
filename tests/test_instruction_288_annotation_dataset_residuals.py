"""Direct checks for the current :mod:`spacr.annotation_dataset` residuals."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import spacr.annotation_dataset as module


def test_empty_selections_and_path_lists_stay_empty():
    """An empty request returns the public empty shapes, not invented rows."""
    assert module.filter_selection(None, {}) is None
    assert list(module.png_list_frame(None, []).columns) == list(
        module.PNG_LIST_COLUMNS
    )


def test_an_explicit_well_column_is_the_filter_key():
    """A stream that already names wells need not carry row/column columns."""
    selection = pd.DataFrame({"well": ["A01", "A02"], "objectID": [1, 2]})

    kept = module.filter_selection(selection, {"wells": ["A02"]})

    assert kept.to_dict("records") == [{"well": "A02", "objectID": 2}]


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_a_failed_reservation_rolls_back_and_preserves_the_cause(
    monkeypatch, rollback_fails
):
    """Reservation failure closes the connection even if rollback also fails."""
    class Connection:
        closed = False

        def execute(self, statement):
            if statement.startswith("CREATE TABLE"):
                raise RuntimeError("create failed")
            if statement == "ROLLBACK" and rollback_fails:
                raise sqlite3.Error("rollback failed")
            return self

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(module, "connect_database", lambda _path: connection)
    monkeypatch.setattr(module, "next_png_table", lambda _connection: "png_list")

    with pytest.raises(RuntimeError, match="create failed"):
        module.reserve_png_table("measurements.db")

    assert connection.closed


def test_a_failed_write_preserves_its_cause_when_rollback_also_fails(monkeypatch):
    """A secondary SQLite rollback error must not replace the write failure."""
    class Connection:
        closed = False

        def execute(self, statement):
            if statement == "ROLLBACK":
                raise sqlite3.Error("rollback failed")
            return self

        def executemany(self, _statement, _rows):
            raise RuntimeError("insert failed")

        def close(self):
            self.closed = True

    connection = Connection()
    monkeypatch.setattr(module, "connect_database", lambda _path: connection)

    with pytest.raises(RuntimeError, match="insert failed"):
        module.write_png_list(
            "measurements.db",
            pd.DataFrame(columns=module.PNG_LIST_COLUMNS),
            table="png_list",
        )

    assert connection.closed


def test_a_table_reservation_failure_returns_actionable_trouble(monkeypatch):
    """The generator names the database it could not reserve, then stops."""
    monkeypatch.setattr(
        module, "reserve_png_table", lambda _path: (_ for _ in ()).throw(
            OSError("read only")
        )
    )

    report = module.generate_annotation_dataset({"database": "broken.db"})

    assert report["written"] == 0 and report["table"] == ""
    assert "broken.db" in " ".join(report["trouble"])


def test_a_stream_that_writes_nothing_keeps_an_empty_reserved_record(
    monkeypatch, tmp_path
):
    """A claimed table is not misreported as a populated annotation set."""
    from spacr import stream_dataset

    selection = pd.DataFrame({
        "plateID": ["p1"],
        "rowID": ["r1"],
        "columnID": ["c1"],
        "fieldID": ["f1"],
        "objectID": [1],
    })
    monkeypatch.setattr(
        module,
        "reserve_png_table",
        lambda _path: pytest.fail("an explicit table must not be reserved again"),
    )
    monkeypatch.setattr(
        stream_dataset,
        "build_selection",
        lambda *_args, **_kwargs: (selection, "selection.csv"),
    )
    monkeypatch.setattr(
        stream_dataset,
        "stream",
        lambda *_args, **_kwargs: {
            "written": 0,
            "missing": 0,
            "fields": 1,
            "folders": [],
        },
    )

    report = module.generate_annotation_dataset({
        "src": str(tmp_path),
        "database": str(tmp_path / "measurements.db"),
        "table": "png_list",
    })

    assert report["table"] == ""
    assert report["trouble"] == [
        "nothing was written; the reserved table png_list is empty"
    ]


def test_database_object_reader_distinguishes_missing_files_and_other_errors(
    monkeypatch, tmp_path
):
    """An absent table is optional, but an unrelated reader error propagates."""
    assert module.read_objects_from_database(tmp_path / "absent.db", "cell") is None

    monkeypatch.setattr(module.os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(
        module,
        "read_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad rows")),
    )

    with pytest.raises(ValueError, match="bad rows"):
        module.read_objects_from_database("present.db", "cell")
