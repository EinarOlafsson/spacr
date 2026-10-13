"""Right-click a run, or several, and keep their state for later.

Asked for on 2026-08-27. A bundle was only ever written automatically when a
run closed, gated by a preference, and the restore action was built only for
a single run -- so a user who wanted to keep a finished run's state could not
ask for it.
"""

import os

import pytest

pytest.importorskip("PySide6")

from spacr import workspace  # noqa: E402
from spacr.qt.widgets.sweep_runs import (  # noqa: E402
    PREFERRED_COLUMNS, RUN_SETTING_COLUMNS, _has_workspace, _permuted,
    _run_settings_row, describe_saved_states, save_run_states,
)


@pytest.fixture
def a_provider():
    """One panel offering some state, the way a real screen does."""
    workspace.clear_providers()
    workspace.register("volcano", lambda: {"threshold": 0.05})
    yield
    workspace.clear_providers()


# --- saving ----------------------------------------------------------------


def test_one_run_is_saved(tmp_path, a_provider):
    folder = tmp_path / "ols_1"
    folder.mkdir()
    saved, failures = save_run_states([str(folder)])
    assert saved == [str(folder)]
    assert failures == []


def test_several_runs_are_saved_together(tmp_path, a_provider):
    """The half the old menu could not do: restore is single-run because two
    workspaces cannot both be on screen, but saving several is not that."""
    folders = []
    for name in ("ols_1", "ols_2", "ols_3"):
        folder = tmp_path / name
        folder.mkdir()
        folders.append(str(folder))
    saved, failures = save_run_states(folders)
    assert len(saved) == 3
    assert failures == []


def test_what_it_writes_is_what_restore_looks_for(tmp_path, a_provider):
    """A save the existing restore cannot find would be no save at all."""
    folder = tmp_path / "ols_1"
    folder.mkdir()
    assert not _has_workspace(str(folder))
    save_run_states([str(folder)])
    assert _has_workspace(str(folder))


def test_the_state_itself_survives(tmp_path, a_provider):
    folder = tmp_path / "ols_1"
    folder.mkdir()
    save_run_states([str(folder)])
    assert workspace.load(str(folder))["sections"]["volcano"] == {
        "threshold": 0.05}


def test_it_saves_even_when_the_automatic_preference_is_off(tmp_path,
                                                            a_provider,
                                                            monkeypatch):
    """A menu item that silently does nothing is worse than one that is not
    there. The user asked; the preference governs the automatic save."""
    monkeypatch.setattr(workspace, "default_mode", lambda: "off")
    folder = tmp_path / "ols_1"
    folder.mkdir()
    saved, _failures = save_run_states([str(folder)])
    assert saved == [str(folder)]


def test_the_preference_is_not_rewritten(tmp_path, a_provider, monkeypatch):
    """Forcing the mode for one call must not turn the setting on."""
    seen = []
    monkeypatch.setattr(workspace, "set_default_mode",
                        lambda *a, **k: seen.append(a))
    folder = tmp_path / "ols_1"
    folder.mkdir()
    save_run_states([str(folder)])
    assert seen == []


# --- what it does when it cannot ------------------------------------------


def test_a_run_with_nothing_to_save_is_named(tmp_path):
    workspace.clear_providers()
    folder = tmp_path / "ols_1"
    folder.mkdir()
    saved, failures = save_run_states([str(folder)])
    assert saved == []
    assert len(failures) == 1
    assert "nothing to save" in failures[0][1]


def test_one_failure_does_not_stop_the_others(tmp_path, a_provider):
    """The rule the overlay loop and the segmentation QC already follow."""
    good = tmp_path / "ols_1"
    good.mkdir()
    saved, failures = save_run_states([str(good), str(tmp_path / "gone" / "x")])
    assert saved == [str(good)]
    assert len(failures) == 1


def test_a_blank_folder_is_skipped_not_failed(tmp_path, a_provider):
    saved, failures = save_run_states(["", "   "])
    assert saved == [] and failures == []


def test_the_note_names_the_failures(tmp_path):
    workspace.clear_providers()
    folder = tmp_path / "ols_1"
    folder.mkdir()
    saved, failures = save_run_states([str(folder)])
    said = describe_saved_states(saved, failures)
    assert "ols_1" in said


def test_a_long_failure_list_is_summarised(tmp_path):
    failures = [(f"/runs/ols_{i}", "nothing to save") for i in range(9)]
    said = describe_saved_states([], failures)
    assert "and 6 more" in said


def test_nothing_selected_says_so():
    assert "nothing was saved" in describe_saved_states([], [])


def test_saving_does_not_move_the_loaded_mark():
    """Keeping a run for later is not choosing it."""
    import inspect

    from spacr.qt.widgets import sweep_runs

    branch = inspect.getsource(sweep_runs.SweepRunsPanel._apply_run_menu)
    save_branch = branch.split('verb == "save_state"')[1].split("if verb ==")[0]
    assert "_loaded_key" not in save_branch


def test_the_menu_offers_it_for_one_and_for_several():
    import inspect

    from spacr.qt.widgets import sweep_runs

    built = inspect.getsource(sweep_runs.SweepRunsPanel._build_run_menu)
    assert '"save_state"' in built
    assert "Save the state of" in built


# --- the extra columns -----------------------------------------------------


def test_the_backend_and_the_shuffles_are_columns():
    assert "regression_backend" in PREFERRED_COLUMNS
    assert "guide_permutations" in PREFERRED_COLUMNS


def test_every_settings_column_is_one_the_ordering_knows():
    """A name the ordering does not know lands past the last sweep column,
    which is the far right of a twenty-column table: recorded, never seen."""
    assert set(RUN_SETTING_COLUMNS) <= set(PREFERRED_COLUMNS)


def test_a_permutation_run_reports_its_shuffles():
    row = _run_settings_row({"analysis_mode": "guide_permutation",
                             "guide_permutations": 200000})
    assert row["guide_permutations"] == 200000


def test_a_least_squares_run_reports_no_shuffles():
    """`guide_permutations` has a default, so every settings dict carries a
    number whether or not a shuffle ever happened."""
    row = _run_settings_row({"inference": "parametric",
                             "guide_permutations": 200000})
    assert "guide_permutations" not in row


def test_zero_shuffles_and_no_permutation_are_different_facts():
    row = _run_settings_row({"inference": "nonparametric",
                             "guide_permutations": 0})
    assert row["guide_permutations"] == 0


def test_the_backend_is_reported_either_way():
    for settings in ({"analysis_mode": "guide_permutation"},
                     {"inference": "parametric"}):
        settings["regression_backend"] = "statsmodels"
        assert _run_settings_row(settings)["regression_backend"] == \
            "statsmodels"


@pytest.mark.parametrize("settings,expected", [
    ({"analysis_mode": "guide_permutation"}, True),
    ({"inference": "nonparametric"}, True),
    ({"inference": "permutation"}, True),
    ({"inference": "parametric"}, False),
    ({}, False),
])
def test_what_counts_as_a_permutation_run(settings, expected):
    assert _permuted(settings) is expected
