"""Keeping a run's state: what the menu does, and what it says when it fails.

Every failure here is NAMED. A save that reports "2 of 5 saved" without
saying which three did not is a report a user cannot act on, so the folder
and the reason travel together from the loop that writes the bundles all the
way to the note under the table.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spacr import workspace  # noqa: E402
from spacr.qt.widgets.sweep_runs import (  # noqa: E402
    SweepRunsPanel, _has_workspace, _readable_size, describe_saved_states,
    save_run_states,
)


@pytest.fixture
def a_provider():
    """One panel offering some state, the way a real screen does."""
    workspace.clear_providers()
    workspace.register("volcano", lambda: {"threshold": 0.05})
    yield
    workspace.clear_providers()


def _run_folder(root, name, seed=1):
    """A folder shaped like a finished run: a coefficient table in it."""
    rng = np.random.default_rng(seed)
    folder = root / "results" / name
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "feature": [f"fraction:grna[{seed}_{i}]" for i in range(4)],
        "coefficient": rng.normal(0, 0.5, 4),
        "p_value": rng.uniform(size=4),
    }).to_csv(folder / "results.csv", index=False)
    return str(folder)


# --------------------------------------------------------------------------
# When the save cannot happen at all


def test_a_save_without_the_workspace_module_names_every_folder(tmp_path,
                                                                monkeypatch):
    """The bundle writer is imported at the point of use, so a build without
    it must refuse in words rather than raise out of a menu entry."""
    first = tmp_path / "ols_1"
    second = tmp_path / "ols_2"
    first.mkdir()
    second.mkdir()
    monkeypatch.setitem(sys.modules, "spacr.workspace", None)

    saved, failures = save_run_states([str(first), str(second)])

    assert saved == []
    assert [path for path, _why in failures] == [str(first), str(second)]
    assert all(why.startswith("workspace unavailable:")
               for _path, why in failures), failures


def test_one_run_that_will_not_write_does_not_stop_the_others(tmp_path,
                                                              monkeypatch,
                                                              a_provider):
    """Five selected runs and a full disk on the second is four saves and one
    named failure, not one exception and nothing saved."""
    good = tmp_path / "ols_good"
    bad = tmp_path / "ols_bad"
    good.mkdir()
    bad.mkdir()
    real = workspace.save_for_run

    def refuse(folder, settings=None):
        if os.path.basename(folder) == "ols_bad":
            raise OSError("No space left on device")
        return real(folder, settings)

    monkeypatch.setattr(workspace, "save_for_run", refuse)

    saved, failures = save_run_states([str(bad), str(good)])

    assert saved == [str(good)]
    assert failures == [(str(bad), "OSError: No space left on device")]
    assert _has_workspace(str(good))
    assert not _has_workspace(str(bad))


# --------------------------------------------------------------------------
# The note under the table


def test_the_note_counts_the_runs_it_saved():
    assert describe_saved_states(["/runs/ols_1"], []) == (
        "Saved the state of 1 run.")
    assert describe_saved_states(
        ["/runs/ols_1", "/runs/ols_2", "/runs/ols_3"], []) == (
        "Saved the state of 3 runs.")


def test_the_note_reports_the_saves_and_the_failures_together():
    """A user who selected four runs and got two bundles needs both halves in
    one sentence, and the failures by name."""
    said = describe_saved_states(
        ["/runs/ols_1", "/runs/ols_2"],
        [("/runs/ols_3", "the run folder is not on disk any more")])
    assert said.startswith("Saved the state of 2 runs.")
    assert "ols_3: the run folder is not on disk any more" in said


# --------------------------------------------------------------------------
# From the menu


def test_saving_from_the_menu_writes_the_bundle_and_says_so(qtbot, tmp_path,
                                                            a_provider):
    """The menu entry is the whole feature: the panel's state goes to disk
    beside the run, and the note says how many runs got one."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)
    record = panel.loaded_run()

    assert panel._apply_run_menu("save_state", [record]) is True
    assert _has_workspace(folder)
    assert workspace.load(folder)["sections"]["volcano"] == {"threshold": 0.05}
    assert panel._source_note == "Saved the state of 1 run."


def test_saving_a_run_does_not_move_the_loaded_mark(qtbot, tmp_path,
                                                    a_provider):
    """Keeping a run for later is not switching to it: moving the mark would
    drag every view on the screen with it."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    first = _run_folder(tmp_path, "ols_1", seed=1)
    second = _run_folder(tmp_path, "ols_2", seed=2)
    panel.load_run_from_disk(first)
    panel.load_run_from_disk(second)
    loaded_before = panel.loaded_run()["folder"]

    other = [row for row in panel._all_rows() if row["folder"] == first]
    assert panel._apply_run_menu("save_state", other) is True
    assert panel.loaded_run()["folder"] == loaded_before


def test_a_menu_save_that_wrote_nothing_reports_false_and_says_why(
        qtbot, tmp_path, monkeypatch):
    """No panel offered any state, so there is no bundle -- and the entry has
    to report that rather than claim a save."""
    workspace.clear_providers()
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    folder = _run_folder(tmp_path, "ols_1")
    panel.load_run_from_disk(folder)

    assert panel._apply_run_menu("save_state", [panel.loaded_run()]) is False
    assert "ols_1" in panel._source_note
    assert "nothing to save" in panel._source_note


def test_a_row_with_no_folder_behind_it_is_not_offered_to_the_writer(
        qtbot, tmp_path, a_provider):
    """A queued fit has no folder yet, and a save that passed '' along would
    create a directory named nothing."""
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    handle = panel.record_run("run A")
    record = dict(panel._recorded[handle])
    assert not record.get("folder")

    assert panel._apply_run_menu("save_state", [record]) is False
    assert panel._source_note == "No run was selected, so nothing was saved."


# --------------------------------------------------------------------------
# The size column


def test_a_folder_size_is_read_in_the_unit_a_person_decides_in():
    """31 MB, not 32,505,856: the column exists to answer "can I delete some
    of these", and bytes do not answer it."""
    assert _readable_size(0) == "0 B"
    assert _readable_size(999) == "999 B"
    assert _readable_size(2048) == "2.0 KB"
    assert _readable_size(32_505_856) == "31 MB"
    assert _readable_size(5 * 1024 ** 3) == "5.0 GB"
    assert _readable_size(4096 * 1024 ** 3) == "4096 GB"
    assert _readable_size(-17) == "0 B"
