"""What Run Compare does when the project's registry cannot be read at all.

A missing registry already has its own sentence. This is the other case: a
file that is there but is not a database, which is what a truncated copy or
an interrupted sync leaves behind.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.artifacts import ARTIFACTS_DB_NAME                    # noqa: E402
from spacr.qt.app import APPS, unregister_app                    # noqa: E402

_APPS_BEFORE = {row[0] for row in APPS}

from spacr.qt.screens import run_compare as screen               # noqa: E402

for _key in sorted({row[0] for row in APPS} - _APPS_BEFORE):
    unregister_app(_key)

pytestmark = pytest.mark.qt


@pytest.fixture()
def corrupt_project(tmp_path):
    """A project whose artifact registry is not a SQLite file."""
    root = tmp_path / "half-copied"
    root.mkdir()
    (root / ARTIFACTS_DB_NAME).write_bytes(b"SQLite format 3\x00 truncated")
    return str(root)


def test_a_registry_that_is_not_a_database_is_reported_not_raised(
        qapp, qtbot, corrupt_project):
    """The screen keeps working and says which project it could not read."""
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)

    runs = view.load_project(corrupt_project)

    assert runs == []
    assert view.comparison() is None
    assert "Could not read that project" in view.verdict_text()
    assert view.last_error, "the underlying message is kept for a caller"
    assert view._verdict.property("blocked") == "true"


def test_a_corrupt_registry_still_announces_the_project_and_empties_the_lists(
        qapp, qtbot, corrupt_project):
    """The load completes: both dropdowns are cleared and listeners hear it."""
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)

    with qtbot.waitSignal(view.project_loaded, timeout=1000) as caught:
        view.load_project(corrupt_project)

    assert caught.args == [corrupt_project]
    assert view._project_edit.text() == corrupt_project
    assert view._a_combo.count() == 0
    assert view._b_combo.count() == 0


def test_a_readable_project_loaded_after_a_corrupt_one_clears_the_error(
        qapp, qtbot, corrupt_project, tmp_path):
    """The stale failure message does not survive the next load."""
    view = screen.RunCompareScreen()
    qtbot.addWidget(view)
    view.load_project(corrupt_project)
    assert view.last_error

    empty = tmp_path / "never-run"
    empty.mkdir()
    view.load_project(str(empty))

    assert view.last_error == ""
    assert "no artifact registry" in view.verdict_text()
