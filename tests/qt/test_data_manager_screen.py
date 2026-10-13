"""The Data Manager screen, driven against a real project on disk.

The screen is a view over :mod:`spacr.data_manager` and owns exactly one
dangerous call, :meth:`~spacr.qt.screens.data_manager.DataManagerScreen.execute_prune`.
What is tested here is that it cannot be reached by accident: the delete
button is disabled without a plan, the confirmation dialog will not accept
until the box is ticked, and the list the dialog shows is the list the module
would delete.

Every test runs ``threaded=False`` so a scan is finished when the call
returns. Both paths run the same code and emit the same signals; the threaded
one is exercised separately, once, for its own reasons.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialogButtonBox               # noqa: E402

from spacr import data_manager as dm                          # noqa: E402
from spacr import ports                                       # noqa: E402
from spacr.qt.screens import data_manager as screen_module    # noqa: E402

from tests.test_data_manager import (build_project, du,        # noqa: E402
                                     register_pipeline)

pytestmark = pytest.mark.qt


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk."""
    root = str(tmp_path / "plate1")
    build_project(root)
    register_pipeline(root)
    return root


@pytest.fixture()
def screen(qtbot, project):
    """The screen, opened on the project, running inline."""
    widget = screen_module.DataManagerScreen(project=project, threaded=False)
    qtbot.addWidget(widget)
    return widget


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_the_screen_registers_itself_through_the_seam():
    """No row in ``app.py``; the import is what puts it in the registry."""
    from spacr.qt.app import APPS, SECTION_DATA, registered_factory
    row = next((r for r in APPS if r[0] == screen_module.APP_KEY), None)
    assert row is not None, "importing the module did not register the app"
    assert row[3] == SECTION_DATA
    assert registered_factory(screen_module.APP_KEY) is not None
    assert screen_module.register() is False, "register() is not idempotent"


def test_the_screen_styles_itself_through_the_theme_seam(qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names
    assert "DataManager" in widget_qss_names()
    assert "QFrame#DataManagerTotals" in stylesheet()


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

def test_opening_on_a_project_shows_its_size_broken_down_by_kind(screen,
                                                                 project):
    usage = screen.usage
    assert usage is not None
    assert usage.total_bytes == du(project)
    assert dm.human_bytes(usage.total_bytes) in screen.total_label.text()

    labels = {screen.usage_table.item(r, 0).text()
              for r in range(screen.usage_table.rowCount())}
    assert dm.KIND_LABELS[ports.MERGED_ARRAYS] in labels
    assert dm.KIND_LABELS[ports.CROPS] in labels
    assert dm.KIND_LABELS[ports.RAW_IMAGES] in labels


def test_the_unregistered_bytes_are_called_out(screen):
    assert "no registry record" in screen.note_label.text()


def test_a_project_that_is_not_a_folder_is_refused_without_a_crash(qtbot,
                                                                   tmp_path):
    widget = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(widget)
    widget._root = str(tmp_path / "does_not_exist")
    assert widget.scan() is False
    assert "Choose a project" in widget.note_label.text()


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def test_the_delete_button_is_dead_until_a_plan_exists(screen):
    assert not screen.delete_button.isEnabled()
    assert screen.confirm_and_prune() is False, (
        "the screen deleted with no plan in hand")


def test_planning_fills_both_tables_and_arms_the_button(screen, project):
    screen.plan_prune()
    plan = screen.plan
    assert plan is not None and plan.candidates
    assert screen.prune_table.rowCount() == len(plan.candidates)
    assert screen.kept_table.rowCount() == len(plan.kept)
    assert screen.delete_button.isEnabled()
    assert dm.human_bytes(plan.total_bytes) in screen.freed_label.text()

    reasons = {screen.kept_table.item(r, 2).text()
               for r in range(screen.kept_table.rowCount())}
    assert all(reason for reason in reasons), "a kept row explains nothing"


def test_changing_the_kinds_invalidates_the_plan_they_were_not_made_with(
        screen):
    screen.plan_prune()
    assert screen.plan is not None
    screen.kind_boxes[ports.MASKS].setChecked(False)
    assert screen.plan is None
    assert not screen.delete_button.isEnabled()
    assert screen.prune_table.rowCount() == 0


def test_the_originals_are_not_even_offered_as_a_checkbox(screen):
    for kind in dm.ORIGINAL_KINDS:
        assert kind not in screen.kind_boxes, (
            f"{kind} has a checkbox, implying a path that does not exist")


def test_the_protected_kinds_start_unticked_and_say_why(screen):
    for kind in dm.PROTECTED_KINDS:
        box = screen.kind_boxes[kind]
        assert not box.isChecked()
        assert "only if you mean it" in box.toolTip()


def test_the_confirmation_dialog_lists_every_file_and_starts_refusing(
        qtbot, screen):
    screen.plan_prune()
    plan = screen.plan
    dialog = screen_module.ConfirmDeleteDialog(plan, screen)
    qtbot.addWidget(dialog)

    ok = dialog.buttons.button(QDialogButtonBox.Ok)
    assert not ok.isEnabled(), "the delete button was armed before it was read"

    text = dialog.describe()
    files, _truncated = plan.file_list()
    assert files
    for path in files:
        assert path in text
    assert dm.human_bytes(plan.total_bytes) in text
    assert "cannot be undone" in text

    dialog.acknowledged.setChecked(True)
    assert ok.isEnabled()


def test_executing_the_plan_deletes_exactly_it_and_rescans(screen, project):
    screen.plan_prune()
    plan = screen.plan
    planned, _ = plan.file_list()
    before_files = {
        os.path.join(folder, name)
        for folder, _dirs, names in os.walk(project)
        for name in names
    }

    assert screen.execute_prune(plan) is True
    for path in planned:
        assert not os.path.exists(path)
    after_files = {
        os.path.join(folder, name)
        for folder, _dirs, names in os.walk(project)
        for name in names
    }
    assert before_files - after_files == set(planned)
    # Marking the registry rows is deliberately committed before deletion and
    # may allocate a new SQLite page. The exact net byte delta is therefore
    # not ``plan.total_bytes`` even though exactly the planned files went.
    # The originals are untouched.
    assert du(os.path.join(project, "orig")) > 0
    # And the screen re-measured itself.
    assert screen.usage.total_bytes == du(project)
    assert screen.plan is None
    assert not screen.delete_button.isEnabled()


def test_a_stale_plan_is_refused_by_the_module_not_by_the_screen(screen,
                                                                 project,
                                                                 qtbot):
    """Something wrote into the project after the plan was shown."""
    screen.plan_prune()
    plan = screen.plan
    with open(os.path.join(project, "merged", "late.npy"), "wb") as handle:
        handle.write(b"late")

    with qtbot.waitSignal(screen.job_finished, timeout=5000) as blocker:
        screen.execute_prune(plan)
    assert blocker.args == [False]
    assert "Nothing was deleted" in screen.note_label.text()
    for candidate in plan.candidates:
        assert os.path.exists(candidate.path)


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------

def test_archiving_needs_a_destination_before_it_will_plan(screen):
    assert not screen.archive_plan_button.isEnabled()
    assert screen.plan_archive() is False
    assert "destination" in screen.note_label.text()


def test_planning_an_archive_lists_what_would_move(screen, project, tmp_path):
    destination = str(tmp_path / "cold")
    screen.set_destination(destination)
    assert screen.archive_plan_button.isEnabled()
    screen.plan_archive()

    plan = screen._archive_plan
    assert plan is not None and plan.items
    assert screen.archive_table.rowCount() == len(plan.items)
    assert screen.archive_button.isEnabled()
    assert destination in screen.note_label.text()
    # Nothing has moved.
    assert os.path.isdir(os.path.join(project, "merged"))


def test_the_archive_moves_and_leaves_a_record(screen, project, tmp_path):
    destination = str(tmp_path / "cold")
    screen.set_destination(destination)
    screen.plan_archive()
    plan = screen._archive_plan

    result = dm.archive(plan, confirm=plan.token)
    assert os.path.isfile(result.manifest_path)
    assert os.path.isfile(result.ledger_path)
    assert os.path.isdir(os.path.join(destination, "merged"))
    assert not os.path.exists(os.path.join(project, "merged"))


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_a_threaded_scan_settles_on_the_gui_thread(qtbot, project):
    """The default path, once: the handler must not run in the worker.

    ``PipelineWorker.finished`` is emitted in the worker thread, so the
    completion handler is chained through a bound method of the widget. If
    that chaining were dropped, this test would fill a QTableWidget off the
    GUI thread — which is undefined behaviour rather than a clean failure,
    so what is asserted is the observable part: the tables are filled and
    the widget is still usable afterwards.
    """
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.job_finished, timeout=20000) as blocker:
        widget.set_project(project)
    assert blocker.args == [True]
    assert widget.usage is not None
    assert widget.usage.total_bytes == du(project)
    assert widget.usage_table.rowCount() > 0
    widget.close()


def test_a_threaded_prune_still_re_measures_the_project_it_changed(qtbot,
                                                                   project):
    """The handler starts the next job, so busy must clear before it runs.

    A completion handler that clears "busy" *after* calling its handler
    leaves the screen marked busy while the handler runs, and the rescan the
    prune handler starts is silently dropped — the numbers on screen would
    then be the ones from before the deletion.
    """
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.job_finished, timeout=20000):
        widget.set_project(project)
    with qtbot.waitSignal(widget.job_finished, timeout=20000):
        widget.plan_prune()
    plan = widget.plan
    assert plan is not None and plan.candidates

    with qtbot.waitSignal(widget.job_finished, timeout=20000):
        widget.execute_prune(plan)
    # The rescan the prune handler starts is a second job; wait for it too.
    qtbot.waitUntil(lambda: widget.usage.total_bytes == du(project),
                    timeout=20000)
    assert du(os.path.join(project, "orig")) > 0
    widget.close()
