"""The results table shows its columns and no rows, and says nothing at all.

Reported while running the regression on 2026-08-16:

    "I can see where the table data should go, i see the columns, but no data
     is loaded."

The panel itself loads fine -- 1,213 rows off the real folder -- so the
failure is somewhere on the path from a FINISHED RUN to the panel. That path
is three separate defects, and the one that matters most is that it is
INVISIBLE: every way it can fail returns False, and what the user gets is an
empty table with no reason attached.

1.  IT SAYS NOTHING. "No results table under /some/folder" does not say what
    it was looking for, how deep it looked, or how many it found, so there is
    nothing for the user to act on.
2.  FIRST IN A SORTED WALK IS NOT THIS RUN'S RESULTS. The search returned the
    first ``results.csv`` in a depth-first sorted walk, so ``glm/`` beats
    ``ols/`` on the alphabet and last month's run beats the one that just
    finished. The newest is the one the user is waiting for.
3.  IT ONLY EVER FIRES ON RUN COMPLETION. A user with results on disk and no
    run to start had no way to open them at all.

And it assumed one shape of coefficient table. The significance column is
``p_value`` on spaCR's own OLS output, ``P>|t|`` or ``P>|z|`` on anything that
came through a statsmodels summary, and for the penalised backends -- see
:data:`spacr.hits.NO_P_VALUE_TYPES` -- there is no p-value at all, only a
bootstrap selection frequency. Two of those three drew an empty p-value
histogram and said nothing about why.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("pyqtgraph")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture()
def results():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(size=n),
        "q_value": np.sort(rng.uniform(size=n)),
    })


def _panel(qtbot):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    return panel


# --------------------------------------------------------------------------- #
#  1. A failure the user can read
# --------------------------------------------------------------------------- #

def test_a_folder_with_no_results_says_what_it_looked_for(qtbot, tmp_path):
    """"No results table under X" leaves the user with nothing to do.

    Which folder, which filenames, and how deep -- otherwise the answer to
    "why is my table empty" is another guess.
    """
    panel = _panel(qtbot)
    assert panel.load(str(tmp_path)) is False

    said = panel.status_text()
    assert str(tmp_path) in said, said
    assert "results.csv" in said, (
        "the message never names the file it was hunting for: " + said)
    assert "deep" in said.lower() or "folder" in said.lower(), said


def test_a_folder_that_is_not_there_is_not_reported_as_an_empty_one(qtbot,
                                                                   tmp_path):
    """A path that does not exist and a path with nothing in it are two
    different problems and used to produce the same silence."""
    panel = _panel(qtbot)
    missing = tmp_path / "no_such_run"
    assert panel.load(str(missing)) is False
    said = panel.status_text().lower()
    assert "does not exist" in said, said


def test_loading_says_which_file_it_chose_and_how_many_rows(qtbot, tmp_path,
                                                            results):
    """A silent success is only better than a silent failure by luck."""
    run = tmp_path / "results" / "plate1_dv" / "ols" / "list"
    run.mkdir(parents=True)
    results.to_csv(run / "results.csv", index=False)

    panel = _panel(qtbot)
    assert panel.load(str(tmp_path)) is True

    said = panel.status_text()
    assert str(run / "results.csv") in said, said
    assert str(len(results)) in said, said


# --------------------------------------------------------------------------- #
#  2. The newest run, not the first letter of the alphabet
# --------------------------------------------------------------------------- #

def test_the_newest_results_win_not_the_first_in_a_sorted_walk(tmp_path,
                                                               results):
    """``glm`` sorts before ``ols``, so the run that just finished lost to
    one from last month and the user read the wrong screen."""
    from spacr.qt.widgets.regression_results import find_results_table

    old = tmp_path / "plate1_dv" / "glm" / "list"
    new = tmp_path / "plate1_dv" / "ols" / "list"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    results.to_csv(old / "results.csv", index=False)
    results.to_csv(new / "results.csv", index=False)
    os.utime(old / "results.csv", (1_000_000, 1_000_000))

    assert find_results_table(str(tmp_path)) == str(new / "results.csv")


def test_the_full_table_beats_its_own_gene_and_guide_splits(tmp_path, results):
    """Caught on the real screen. ``perform_regression`` writes results.csv,
    then results_gene.csv, then results_grna.csv into ONE folder, milliseconds
    apart -- so "the newest file" is the guide split, which is 823 rows of a
    1,213-row fit. The run folder's newest table dates the run; inside it the
    full coefficient table wins."""
    from spacr.qt.widgets.regression_results import find_results_table

    run = tmp_path / "plate1_dv" / "ols" / "list"
    run.mkdir(parents=True)
    results.to_csv(run / "results.csv", index=False)
    results.head(50).to_csv(run / "results_gene.csv", index=False)
    results.head(80).to_csv(run / "results_grna.csv", index=False)
    os.utime(run / "results.csv", (1_000_000, 1_000_000))
    os.utime(run / "results_gene.csv", (1_000_001, 1_000_001))
    os.utime(run / "results_grna.csv", (1_000_002, 1_000_002))

    assert find_results_table(str(tmp_path)) == str(run / "results.csv")


def test_a_newer_run_still_beats_an_older_run_with_a_full_table(tmp_path,
                                                                results):
    """The file-priority rule must not undo the newest-run rule: a stale
    ``results.csv`` next door does not outrank the run that just finished."""
    from spacr.qt.widgets.regression_results import find_results_table

    old = tmp_path / "glm" / "list"
    new = tmp_path / "ols" / "list"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    results.to_csv(old / "results.csv", index=False)
    results.to_csv(new / "results.csv", index=False)
    results.head(50).to_csv(new / "results_grna.csv", index=False)
    os.utime(old / "results.csv", (1_000_000, 1_000_000))

    assert find_results_table(str(tmp_path)) == str(new / "results.csv")


def test_every_candidate_is_reported_newest_first(tmp_path, results):
    """The panel says how many it found; that count has to come from
    somewhere the caller can also see."""
    from spacr.qt.widgets.regression_results import find_results_tables

    old = tmp_path / "a" / "list"
    new = tmp_path / "b" / "list"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    results.to_csv(old / "results.csv", index=False)
    results.to_csv(new / "results.csv", index=False)
    os.utime(old / "results.csv", (1_000_000, 1_000_000))

    found = find_results_tables(str(tmp_path))
    assert found[0] == str(new / "results.csv")
    assert str(old / "results.csv") in found


def test_the_panel_says_it_picked_the_newest_of_several(qtbot, tmp_path,
                                                        results):
    panel = _panel(qtbot)
    old = tmp_path / "glm" / "list"
    new = tmp_path / "ols" / "list"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    results.to_csv(old / "results.csv", index=False)
    results.to_csv(new / "results.csv", index=False)
    os.utime(old / "results.csv", (1_000_000, 1_000_000))

    assert panel.load(str(tmp_path)) is True
    said = panel.status_text()
    assert str(new / "results.csv") in said, said
    assert "newest" in said.lower(), said


# --------------------------------------------------------------------------- #
#  3. Reachable without a run
# --------------------------------------------------------------------------- #

def test_results_on_disk_can_be_opened_without_starting_a_run(qtbot, tmp_path,
                                                              results, monkeypatch):
    """The load only ever fired on successful run completion, so a user whose
    results were already on disk had no way in."""
    from PySide6.QtWidgets import QPushButton

    run = tmp_path / "ols" / "list"
    run.mkdir(parents=True)
    results.to_csv(run / "results.csv", index=False)

    panel = _panel(qtbot)
    buttons = [b for b in panel.findChildren(QPushButton)
               if "load results" in b.text().lower()]
    assert buttons, "no way to open a results folder by hand"

    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(tmp_path)))
    buttons[0].click()
    assert panel.table.table.rowCount() == len(results)


def test_cancelling_the_folder_chooser_changes_nothing(qtbot, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    panel = _panel(qtbot)
    before = panel.status_text()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: ""))
    assert panel.browse_for_results() is False
    assert panel.status_text() == before


# --------------------------------------------------------------------------- #
#  4. Every backend's coefficient table
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("column", ["p_value", "P>|t|", "P>|z|", "Pr(>|z|)",
                                    "pvalue", "p-value"])
def test_the_significance_column_is_found_however_it_is_spelled(qtbot, column):
    """``t value`` for OLS and ``z value`` for a GLM come with p columns
    spelled ``P>|t|`` and ``P>|z|``. Only ``p_value`` was recognised, so two
    of the three backends drew an empty histogram."""
    frame = pd.DataFrame({
        "feature": [f"g{i}" for i in range(30)],
        "coefficient": np.linspace(-1, 1, 30),
        column: np.linspace(0.001, 0.9, 30),
    })
    panel = _panel(qtbot)
    assert panel._p_column(frame) == column
    assert panel.set_frame(frame, source="results.csv")
    assert "No p-values" not in panel.p_values._status.text()


def test_a_table_with_a_statistic_but_no_p_value_still_sorts(qtbot):
    """A ``z value`` is not a p-value. The table must still show and still
    sort, and the panel must not pretend a p-value is there."""
    frame = pd.DataFrame({
        "feature": [f"g{i}" for i in range(30)],
        "coefficient": np.linspace(-1, 1, 30),
        "z value": np.linspace(-4, 4, 30),
    })
    panel = _panel(qtbot)
    assert panel._p_column(frame) is None
    assert panel.set_frame(frame, source="results.csv")
    assert panel.table.table.rowCount() == 30
    panel.table.table.sortItems(1)
    assert panel.table.table.rowCount() == 30
    said = panel.status_text().lower()
    assert "no p-value" in said, said


def test_a_penalised_backend_says_it_has_no_p_value(qtbot):
    """lasso and elasticnet rank by bootstrap SELECTION FREQUENCY and carry
    no frequentist p-value at all -- see spacr.hits.NO_P_VALUE_TYPES. An
    empty p-value histogram with no caption reads as a broken panel."""
    frame = pd.DataFrame({
        "feature": [f"g{i}" for i in range(30)],
        "coefficient": np.linspace(-1, 1, 30),
        "selection_frequency": np.linspace(0.0, 1.0, 30),
    })
    panel = _panel(qtbot)
    assert panel.set_frame(frame, source="results.csv")
    assert panel.table.table.rowCount() == 30

    said = panel.status_text().lower()
    assert "selection frequency" in said, said
    for plot in (panel.p_values, panel.qq):
        assert "selection frequency" in plot._status.text().lower(), \
            plot._status.text()
    assert panel.table._only_hits.isVisible() is False, (
        "'significant only' cuts on `value <= alpha`, which is backwards for "
        "a selection frequency and would hide every real selection")


def test_a_lasso_run_is_not_ranked_by_its_ols_p_value(qtbot, tmp_path):
    """spacr.ml writes an OLS-style ``p_value`` into a lasso results.csv --
    it is computed ignoring the penalty and means nothing. The run folder is
    named for the backend, so the panel can and must know."""
    from spacr.hits import NO_P_VALUE_TYPES

    assert "lasso" in NO_P_VALUE_TYPES
    frame = pd.DataFrame({
        "feature": [f"g{i}" for i in range(30)],
        "coefficient": np.linspace(-1, 1, 30),
        "p_value": np.linspace(0.001, 0.9, 30),
    })
    run = tmp_path / "plate1_dv" / "lasso" / "list"
    run.mkdir(parents=True)
    frame.to_csv(run / "results.csv", index=False)

    panel = _panel(qtbot)
    assert panel.load(str(tmp_path)) is True
    said = panel.status_text().lower()
    assert "lasso" in said, said
    assert "no p-value" in said or "not a p-value" in said, said


def test_the_ranking_column_is_reported_for_the_ordinary_case(qtbot, results):
    panel = _panel(qtbot)
    assert panel.set_frame(results, source="results.csv")
    assert panel.ranking() == ("p-value", "p_value")


def test_the_caller_can_put_its_own_reason_in_the_same_header(qtbot):
    """AppScreen decides which folder to search and can come up with none.
    That failure is the caller's, and it has to land where the user is
    already looking rather than in a debug log."""
    panel = _panel(qtbot)
    panel.say("The run named no output folder.", detail="src, count_data")
    assert panel.status_text() == "The run named no output folder."
    assert panel._source.toolTip() == "src, count_data"
