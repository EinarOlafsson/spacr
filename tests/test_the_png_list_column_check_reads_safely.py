"""The annotation-column check reads every plate's png_list without tripping.

Issue #186: a Classify run in annotation mode must say, before it starts,
when its label column is in no plate's ``png_list`` -- and must say nothing
when it cannot tell. A missing file, a file that is not SQLite, a file this
user may not open, or a database without ``png_list`` all mean "cannot
tell", so they are passed over instead of raising or inventing a problem.
"""

from __future__ import annotations

import os
import sqlite3

import pytest

from spacr import validate
from spacr.training_basis import _annotation_column_absent, _png_list_columns


def _database(path, columns):
    connection = sqlite3.connect(path)
    if columns:
        names = ", ".join(f'"{name}"' for name in columns)
        connection.execute(f"CREATE TABLE png_list ({names})")
    else:
        connection.execute("CREATE TABLE other (x)")
    connection.commit()
    connection.close()
    return str(path)


@pytest.fixture
def plates(tmp_path):
    """One database of each kind the check meets."""
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"this is not an SQLite file " * 64)
    locked = _database(tmp_path / "locked.db", ["png_path", "label"])
    os.chmod(locked, 0)
    yield {
        "missing": str(tmp_path / "missing.db"),
        "garbage": str(garbage),
        "locked": locked,
        "no_png_list": _database(tmp_path / "no_png_list.db", []),
        "without": _database(tmp_path / "without.db", ["png_path", "plate"]),
        "with": _database(tmp_path / "with.db", ["png_path", "label"]),
    }
    os.chmod(locked, 0o600)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root opens a file whatever its permissions")
def test_every_unreadable_database_reads_as_no_columns(plates):
    for kind in ("missing", "garbage", "locked", "no_png_list"):
        assert _png_list_columns(plates[kind]) is None, kind
    assert _png_list_columns(plates["with"]) == ["png_path", "label"]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root opens a file whatever its permissions")
def test_the_message_names_png_list_and_only_the_databases_it_could_read(plates):
    unreadable = [plates[k] for k in ("missing", "garbage", "locked", "no_png_list")]

    message = _annotation_column_absent(unreadable + [plates["without"]], "label")

    assert message.startswith(
        f"annotation_column='label' is not a column of png_list in {plates['without']}.")
    assert "png_list has: plate, png_path." in message
    assert "press Save annotations in Graph Builder" in message
    for path in unreadable:
        assert path not in message
    assert _annotation_column_absent(unreadable, "label") is None
    assert _annotation_column_absent([plates["without"], plates["with"]], "label") is None
    assert _annotation_column_absent([plates["without"]], "") is None


def _plate(tmp_path, name, columns):
    folder = tmp_path / name / "measurements"
    folder.mkdir(parents=True)
    _database(folder / "measurements.db", columns)
    return str(tmp_path / name)


@pytest.mark.parametrize(
    ("settings", "flagged"),
    [
        ({"dataset_mode": "annotation", "annotation_column": "label"}, True),
        ({"dataset_mode": "annotation", "annotation_column": "plate"}, False),
        ({"dataset_mode": "metadata", "annotation_column": "label"}, False),
        ({"dataset_mode": "not-a-basis", "annotation_column": "label"}, False),
    ],
)
def test_validation_reports_a_label_column_no_png_list_has(tmp_path, settings, flagged):
    settings = dict(settings, src=_plate(tmp_path, "plate1", ["png_path", "plate"]))

    problems = validate._check_annotation_column_in_png_list(settings)

    assert [p.setting for p in problems] == (["annotation_column"] if flagged else [])
    if flagged:
        assert "png_list" in problems[0].message
