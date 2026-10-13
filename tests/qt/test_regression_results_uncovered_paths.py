"""What the results panel does when the parts it leans on are not there.

The panel reads its vocabulary from three other modules -- ``spacr.hits`` for
what tells a gene from its guides and where the covariates end,
``spacr.guide_concordance`` for per-gene guide agreement, and ``spacr.ml``
for the names a run writes its summary under -- and it reads each of them at
the point of use rather than at import, so a build missing one of them is a
panel with one answer missing rather than a screen that will not open.

These drive those fallbacks, the tab-raising route on a panel whose tab bar
has already gone, and the loader's own recovery when a job is refused, and
assert what the reader is left looking at in each case.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import regression_results as module          # noqa: E402
from spacr.qt.widgets.regression_results import (                  # noqa: E402
    RegressionResultsPanel, backend_of, find_summary_file, summary_text,
)


def _mixed(genes: int = 3) -> pd.DataFrame:
    """A ``level='both'`` table: an intercept, ``genes`` genes, 2 guides."""
    rows = [{"feature": "Intercept", "coefficient": 0.2, "p_value": 1e-9}]
    for gene in range(genes):
        rows.append({"feature": f"gene_fraction:gene[G{gene}]",
                     "coefficient": 0.5 + gene,
                     "p_value": 0.01 * (gene + 1)})
        for guide in (1, 2):
            rows.append({"feature": f"fraction:grna[G{gene}_{guide}]",
                         "coefficient": 0.4 + gene + 0.1 * guide,
                         "p_value": 0.02 * (gene + 1)})
    return pd.DataFrame(rows)


def _panel(qtbot, **kwargs) -> RegressionResultsPanel:
    panel = RegressionResultsPanel(**kwargs)
    qtbot.addWidget(panel)
    return panel


def _block(monkeypatch, name: str) -> None:
    """Make ``from <name> import ...`` raise, the way a partial build does."""
    monkeypatch.setitem(sys.modules, name, None)


# --------------------------------------------------------------- spacr.ml

def test_the_summary_a_run_wrote_is_not_found_without_the_writers_names(
        tmp_path, monkeypatch):
    """No ``spacr.ml`` means no list of summary names, so none is claimed."""
    (tmp_path / "model_summary.txt").write_text("Dep. Variable: y\n",
                                                encoding="utf-8")

    assert find_summary_file(tmp_path) == str(tmp_path / "model_summary.txt")

    _block(monkeypatch, "spacr.ml")
    assert find_summary_file(tmp_path) is None, (
        "the panel invented a summary filename rather than reading the list "
        "from the module that writes them")


def test_the_summary_tab_says_it_looked_for_a_summary_file_with_no_names(
        tmp_path, monkeypatch):
    """The tab still explains itself, generically, with no name list."""
    _block(monkeypatch, "spacr.ml")

    text = summary_text(None, path=str(tmp_path))

    assert text.startswith("No summary:")
    assert "looked for a summary file" in text, (
        f"the explanation should fall back to a generic name; got {text!r}")


# ------------------------------------------------------------- spacr.hits

def test_a_penalised_run_is_ranked_by_its_folder_only_while_hits_is_readable():
    """``backend_of`` is what stops a lasso table being ranked by a fake p."""
    table = pd.DataFrame({"feature": ["fraction:grna[g1_1]"],
                          "coefficient": [1.0], "p_value": [0.001]})
    source = "/runs/results/lasso_1/results.csv"

    assert backend_of(source) == "lasso"
    assert RegressionResultsPanel._rank_by(table, source) == (
        "selection-frequency", None)


def test_a_penalised_folder_is_unrecognised_without_the_backend_vocabulary(
        monkeypatch):
    """With ``spacr.hits`` gone the folder says nothing about the backend."""
    _block(monkeypatch, "spacr.hits")
    source = "/runs/results/lasso_1/results.csv"

    assert backend_of(source) is None
    assert backend_of("") is None
    kind, column = RegressionResultsPanel._rank_by(
        pd.DataFrame({"feature": ["fraction:grna[g1_1]"],
                      "coefficient": [1.0], "p_value": [0.001]}),
        source)
    assert (kind, column) == ("p-value", "p_value"), (
        "the table has to fall back to its own columns when the folder "
        "cannot be interpreted")


def test_a_mixed_table_gains_a_level_column_beside_its_feature_names():
    """The baseline for the fallback below: the column is normally added."""
    named = RegressionResultsPanel._name_the_levels(_mixed())

    assert list(named.columns)[:2] == ["feature", "level"]
    assert set(named["level"]) == {"nuisance", "gene", "grna"}


def test_a_mixed_table_is_passed_through_when_the_level_vocabulary_is_missing(
        monkeypatch):
    """No ``spacr.hits`` means no level column -- and no exception either."""
    frame = _mixed()
    _block(monkeypatch, "spacr.hits")

    named = RegressionResultsPanel._name_the_levels(frame)

    assert "level" not in named.columns
    assert list(named.columns) == list(frame.columns)
    assert len(named) == len(frame)


def test_the_level_filter_shows_every_row_when_the_split_cannot_be_asked_for(
        qtbot, monkeypatch):
    """A filter that cannot be computed shows the whole fit, not nothing."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    panel.set_level("grna")

    assert len(panel.filtered_frame()) == 6
    assert len(panel.results_frame()) == 10

    _block(monkeypatch, "spacr.hits")

    assert panel.level() == "grna"
    assert len(panel.filtered_frame()) == 10, (
        "an unanswerable filter must fall back to the whole table rather "
        "than dropping every row")


def test_the_tested_family_is_reported_as_unknown_rather_than_guessed(
        monkeypatch):
    """``None`` is the honest answer when the family rule cannot be read."""
    frame = _mixed()

    tested = RegressionResultsPanel._tested_mask(frame)
    assert list(tested) == [False] + [True] * 9

    _block(monkeypatch, "spacr.hits")
    assert RegressionResultsPanel._tested_mask(frame) is None, (
        "the panel re-derived the hypothesis family instead of admitting it "
        "could not ask for it")


def test_guide_support_rows_lose_their_key_when_gene_terms_cannot_be_resolved(
        qtbot, monkeypatch):
    """Without ``gene_of`` the support rows are still listed, just unlinked."""
    panel = _panel(qtbot)
    frame = _mixed()

    panel._draw_guide_support(frame)
    assert panel.support.table.rowCount() == 3
    assert panel.support.table.item(0, 0).text() == "gene_fraction:gene[G0]"

    _block(monkeypatch, "spacr.hits")
    panel._draw_guide_support(frame)

    assert panel.support.table.rowCount() == 3, (
        "the guide-support rows are computed without spacr.hits and must "
        "survive its absence")
    assert panel.support.table.item(0, 0).text() == "", (
        "a row that cannot be joined to a coefficient term must carry no "
        "term rather than a wrong one")


# --------------------------------------------------- spacr.guide_concordance

def test_the_run_is_still_drawn_when_guide_concordance_is_missing(
        qtbot, monkeypatch):
    """The Guide support tab is optional; the run behind it is not."""
    panel = _panel(qtbot)
    _block(monkeypatch, "spacr.guide_concordance")

    assert panel.set_frame(_mixed(4), source="run-one") is True
    assert panel.table.table.rowCount() == 8, (
        "the coefficient table lost its rows because an optional tab could "
        "not be built")


def test_a_repeated_feature_column_empties_the_guide_support_tab(qtbot):
    """A table with two ``feature`` columns is drawn as no support at all."""
    panel = _panel(qtbot)
    frame = _mixed()
    panel._draw_guide_support(frame)
    assert panel.support.table.rowCount() == 3

    doubled = pd.concat([frame, frame[["feature"]]], axis=1)
    panel._draw_guide_support(doubled)

    assert panel.support.table.rowCount() == 0, (
        "a table shape guide_support cannot read must clear the tab rather "
        "than leave the previous run's genes on it")


# ------------------------------------------------------------ tabs and loader

def test_a_live_tile_raises_its_tab_while_the_tab_bar_is_alive(qtbot):
    """The baseline for the destroyed-tab-bar case below."""
    panel = _panel(qtbot)

    assert panel.show_panel("qq") is True
    assert panel.tabs.tabText(panel.tabs.currentIndex()).startswith("Q-Q")
    assert panel.show_panel("not-a-panel") is False


def test_a_live_tile_press_is_refused_once_the_tab_bar_has_been_destroyed(
        qtbot):
    """A tile click arriving after the tabs went is answered, not raised."""
    from shiboken6 import Shiboken

    panel = _panel(qtbot)
    assert panel.show_panel("qq") is True

    Shiboken.delete(panel.tabs)

    assert panel.show_panel("qq") is False, (
        "show_panel let Qt's dead-object RuntimeError out into the click "
        "that delivered it")


def test_a_refused_load_leaves_the_button_ready_for_the_next_one(qtbot):
    """``start_load`` clears its own busy state when the job never starts.

    The loader is put in unthreaded mode so the refusal is synchronous, and
    its failure signal is deliberately left unconnected: what is under test
    is that ``start_load`` itself recovers, rather than relying on the
    ``job_failed`` route to undo the busy state for it. A panel that stays
    busy has a button that reads "Cancel load" for the rest of the session
    and refuses every later folder.
    """
    from spacr.qt.job_runner import JobRunner

    panel = _panel(qtbot)
    panel._load_jobs.shutdown()
    panel._load_jobs = JobRunner(panel, threaded=False, app_key="results")

    # An object with no filesystem path at all: the worker raises on it, so
    # the runner reports the job as never started.
    assert panel.start_load(object()) is False
    assert panel.is_loading() is False
    assert panel._load_button.text().startswith("Load results")


def test_start_load_refuses_a_second_read_while_one_is_in_flight(qtbot):
    """The busy flag is what stops two reads of the same folder."""
    panel = _panel(qtbot)
    panel._set_loading(True)

    assert panel.start_load("/anywhere") is False
    assert panel._load_button.text() == "Cancel load"


def test_start_load_says_so_when_it_is_handed_no_folder(qtbot):
    """An empty path is answered on the status line, not silently ignored."""
    panel = _panel(qtbot)

    assert panel.start_load("") is False
    assert "no folder to search" in panel.status_text()
    assert panel.is_loading() is False


# ------------------------------------------------- a run folder read from disk

def _write_run(folder, primary, gene=None):
    folder.mkdir(parents=True, exist_ok=True)
    primary.to_csv(folder / "results.csv", index=False)
    if gene is not None:
        gene.to_csv(folder / "results_gene.csv", index=False)
    return folder


def _guide_rows(n=3) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": [f"fraction:grna[G{i}_1]" for i in range(n)],
        "coefficient": [0.1 * (i + 1) for i in range(n)],
        "p_value": [0.01 * (i + 1) for i in range(n)]})


def _gene_rows(n=2) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(n)],
        "coefficient": [0.5 + i for i in range(n)],
        "p_value": [0.04 + 0.01 * i for i in range(n)]})


def test_an_older_run_gains_a_level_column_when_its_gene_rows_are_merged(
        tmp_path):
    """The primary table has no ``level``; the merge has to give it one.

    Without it the merged frame would say ``gene`` for the sibling's rows and
    nothing for the guides it already held, and the guide filter would then
    find no rows at all in the run's own primary table.
    """
    folder = _write_run(tmp_path / "guide_permutation_1", _guide_rows(),
                        _gene_rows())

    frame, found, merged = module.read_run_tables(
        module.find_results_tables(str(folder)))

    assert found.endswith("results.csv")
    assert [m.rsplit("/", 1)[-1] for m in merged] == ["results_gene.csv"]
    assert len(frame) == 5
    assert sorted(frame["level"].dropna().unique()) == ["gene", "grna"]
    assert (frame["level"] == "grna").sum() == 3


def test_a_primary_table_that_names_its_own_level_still_takes_the_gene_half(
        tmp_path):
    """``results.csv`` says ``grna`` for every row; the gene half is elsewhere.

    The level column is what makes the sibling worth reading -- it is how the
    merge knows this table holds no gene rows -- and it must not be rewritten
    on the way.
    """
    guides = _guide_rows()
    guides["level"] = "grna"
    folder = _write_run(tmp_path / "guide_permutation_2", guides, _gene_rows())

    frame, _found, merged = module.read_run_tables(
        module.find_results_tables(str(folder)))

    assert len(merged) == 1
    assert list(frame["level"]) == ["grna"] * 3 + ["gene"] * 2


def test_a_sibling_that_cannot_be_read_leaves_the_primary_table_intact(
        tmp_path):
    """A run killed mid-write leaves an empty sibling; the run still opens."""
    folder = _write_run(tmp_path / "run_1", _guide_rows())
    (folder / "results_gene.csv").write_text("", encoding="utf-8")

    frame, found, merged = module.read_run_tables(
        module.find_results_tables(str(folder)))

    assert merged == []
    assert len(frame) == 3
    assert found.endswith("results.csv")


def test_a_column_the_table_cannot_measure_is_kept_rather_than_dropped():
    """``for_table`` drops blank columns, and keeps what it cannot judge."""
    base = pd.DataFrame({"feature": ["a", "b"],
                         "coefficient": [1.0, 2.0],
                         "note": ["x", "y"],
                         "blank": [None, None]})

    assert list(module.for_table(base).columns) == [
        "feature", "coefficient", "note"]

    # A repeated column name makes ``frame[name]`` a frame rather than a
    # series, so the emptiness test cannot be made. The column is kept -- an
    # unjudged column is not an empty one -- and the blank column beside it is
    # still dropped.
    doubled = pd.concat([base, base[["note"]]], axis=1)
    kept = module.for_table(doubled)

    assert "blank" not in kept.columns
    assert "note" in kept.columns
    assert list(kept["feature"].iloc[:, 0] if kept["feature"].ndim > 1
                else kept["feature"]) == ["a", "b"]


# -------------------------------------------------------- loader housekeeping

def test_the_loading_state_survives_the_button_going_away_under_a_close(qtbot):
    """The busy flag must not depend on the button still existing."""
    from shiboken6 import Shiboken

    panel = _panel(qtbot)
    Shiboken.delete(panel._load_button)

    panel._set_loading(True)

    assert panel.is_loading() is True


def test_cancelling_a_load_answers_the_caller_waiting_on_it(qtbot):
    """A caller waiting on ``load_finished`` must not be left waiting."""
    panel = _panel(qtbot)
    panel._set_loading(True)

    with qtbot.waitSignal(panel.load_finished, timeout=1000) as blocker:
        assert panel.cancel_load() is True

    assert blocker.args == [False]
    assert panel.is_loading() is False
    assert "cancelled" in panel.status_text()


# ------------------------------------------------------ what the panel reports

def test_a_table_with_no_usable_p_values_says_so_without_calling_them_blups(
        qtbot):
    """"No p-value" and "these are BLUPs" are different findings."""
    panel = _panel(qtbot)
    frame = _guide_rows(4)
    frame["p_value"] = float("nan")

    panel._say_if_no_p_values(frame)

    assert "None of these 4 coefficient(s) carries a p value" in \
        panel.status_text()
    assert "BLUP" not in panel.status_text()


def test_a_mixed_model_table_is_named_as_blups_rather_than_missing_p_values(
        qtbot):
    """``term_type`` is what turns the silence into an explanation."""
    panel = _panel(qtbot)
    frame = _guide_rows(4)
    frame["p_value"] = float("nan")
    frame["term_type"] = "random_effect_blup"

    panel._say_if_no_p_values(frame)

    assert "BLUPs, not estimates" in panel.status_text()


def test_an_empty_table_reads_its_analysis_path_off_the_run_settings(qtbot):
    """With no columns to judge by, the settings are the only evidence left."""
    panel = _panel(qtbot)
    panel._frame = pd.DataFrame()

    panel._run_settings = {"analysis_mode": "guide_permutation"}
    assert panel._analysis_path() == "permutation"

    panel._run_settings = {"analysis_mode": "ols"}
    assert panel._analysis_path() == "fitted"

    panel._run_settings = None
    assert panel._analysis_path() == "fitted"


def test_a_guide_only_run_blames_its_level_column_for_the_empty_gene_view(
        qtbot):
    """A table that records the fit each row came from can say so exactly."""
    panel = _panel(qtbot)
    guides = _guide_rows(3)
    guides["level"] = "grna"
    panel.set_frame(guides, source="guides-only")

    panel.set_level("gene")

    note = panel.missing_level_note()
    assert "No gene-level coefficients in this run" in note
    assert "the table records the fit each coefficient came from" in note
    assert "3 guide-level coefficients are still here" in note


def test_a_gene_only_fit_says_which_level_it_was_fitted_at(qtbot):
    """The run settings name the fit that produced a gene-only table."""
    panel = _panel(qtbot)
    panel.set_frame(_gene_rows(2), source="gene-only")
    panel._run_settings = {"level": "gene"}

    panel.set_level("grna")

    note = panel.missing_level_note()
    assert "No guide-level coefficients in this run" in note
    assert "fitted at level='gene'" in note


def test_a_level_index_outside_the_menu_changes_nothing(qtbot):
    """``activated`` can arrive for a row the box no longer has."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    before = panel.level()

    panel._level_chosen(panel._level_box.count() + 5)

    assert panel.level() == before


def test_a_table_that_names_neither_genes_nor_guides_is_not_filtered(qtbot):
    """With no ``feature`` and no ``level``, there is no split to apply."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    panel.set_level("grna")

    assert panel._level_mask(pd.DataFrame({"coefficient": [1.0]})) is None


def test_a_clicked_key_counts_as_reachable_when_the_split_cannot_be_computed(
        qtbot, monkeypatch):
    """An unanswerable filter must not send the panel after a level change."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    panel.set_level("gene")

    assert panel._reachable("fraction:grna[G0_1]") is False

    _block(monkeypatch, "spacr.hits")
    assert panel._reachable("fraction:grna[G0_1]") is True


def test_a_gene_term_that_names_no_gene_contributes_no_key():
    """``gene_of`` answering ``None`` must not put a null key in the map."""
    frame = pd.DataFrame({
        "feature": ["gene_fraction:gene[]", "gene_fraction:gene[G1]"],
        "coefficient": [0.1, 0.2], "p_value": [0.5, 0.01]})

    terms = RegressionResultsPanel._gene_terms(frame)

    assert terms == {"G1": "gene_fraction:gene[G1]"}


# ------------------------------------------------ settings before a table

def test_new_run_settings_do_not_overwrite_a_summary_that_has_a_model(qtbot):
    """A live fit's summary is the run's own, not replaced by a reason."""
    panel = _panel(qtbot)
    panel._model = object()

    panel.set_run_settings({"level": "both"})

    assert panel._model is not None
    assert "No summary" not in panel._summary.toPlainText()


def test_a_baseline_chosen_before_a_table_arrives_is_remembered(qtbot):
    """The volcano's baseline menu works on an empty panel."""
    panel = _panel(qtbot)

    panel.set_baseline("median")

    assert panel.plot_state()["baseline"] == ("median", None)


def test_compartment_colouring_on_an_empty_panel_says_there_is_none(qtbot):
    """Every compartment choice is answerable before a run is loaded."""
    from spacr.localisation import ALL as ALL_COMPARTMENTS

    panel = _panel(qtbot)

    panel.set_compartment(None)
    assert panel.plot_state()["compartment"] is None

    panel.set_compartment(ALL_COMPARTMENTS)
    assert "nothing to colour by" in panel.status_text()


def test_an_unknown_colour_column_in_a_saved_state_leaves_the_combo_alone(
        qtbot):
    """A run's saved colouring may name a column the next table has not got."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    before = panel._colour_by.currentIndex()

    state = {"colour_by": "a-column-nothing-has"}
    assert panel.apply_plot_state(state) is True
    assert panel._colour_by.currentIndex() == before


def test_a_workspace_state_keeps_only_the_runs_it_can_read(qtbot):
    """Anything that is not a dict of plot state is not a remembered run."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")

    assert panel.apply_workspace_state(
        {"runs": {"good": {"level": "gene"}, "bad": "not a state"}}) is True
    assert panel.remembered_runs() == ("good",)

    assert panel.apply_workspace_state({"runs": "not a mapping"}) is True
    assert panel.remembered_runs() == ("good",)


def test_asking_for_adjusted_p_values_a_table_has_not_got_keeps_the_points(
        qtbot):
    """The volcano falls back to the raw column instead of drawing nothing."""
    panel = _panel(qtbot)
    panel.set_frame(_mixed(), source="run-one")
    key = "fraction:grna[G0_1]"
    assert panel.volcano.highlight_key(key) is True

    panel.set_p_value_kind("adjusted")

    assert panel.volcano.highlight_key(key) is True, (
        "the volcano was emptied because the table has no corrected column"
    )
    assert panel.plot_state()["p_value_kind"] == "adjusted"
