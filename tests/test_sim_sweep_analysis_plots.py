"""Sweep-result analysis figures in :mod:`spacr.sim`.

One test per figure builder: ``plot_simulations`` (PR-AUC grids),
``plot_feature_importance`` and ``calculate_permutation_importance`` (random
forest importances), ``plot_correlation_matrix`` and
``plot_partial_dependences``. Inputs are hand-built sweep tables whose target
is driven almost entirely by ``classifier_accuracy``, so the ranked labels,
bar widths, panel counts and saved files are exact.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spacr import sim as S

GROUPING_VARS = ['number_of_active_genes', 'number_of_control_genes',
                 'avg_reads_per_gene', 'classifier_accuracy', 'nr_plates',
                 'number_of_genes', 'avg_genes_per_well', 'avg_cells_per_well',
                 'sequencing_error', 'well_ineq_coeff', 'gene_ineq_coeff']


@pytest.fixture(autouse=True)
def _in_tmp_and_close_figures(tmp_path, monkeypatch):
    """Run from a temp folder: the figures land in the RELATIVE ``figures`` folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPACR_SIM_FIGURES", raising=False)
    yield
    plt.close("all")


def _sweep_results(n_reps=3, accuracies=(0.8, 0.95)):
    """A tidy sweep table: nr_plates varies within each accuracy condition."""
    rows = []
    for acc in accuracies:
        for plates in (1, 2, 4):
            for rep in range(n_reps):
                rows.append({
                    'number_of_active_genes': 10, 'number_of_control_genes': 5,
                    'avg_reads_per_gene': 100, 'classifier_accuracy': acc,
                    'nr_plates': plates, 'number_of_genes': 30,
                    'avg_genes_per_well': 5, 'avg_cells_per_well': 20,
                    'sequencing_error': 0.01, 'well_ineq_coeff': 1.2,
                    'gene_ineq_coeff': 1.2, 'prauc': 0.1 * plates + acc + 0.01 * rep})
    return pd.DataFrame(rows)


def _importance_df(n=60, seed=0):
    """prauc is driven almost entirely by classifier_accuracy."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({v: rng.uniform(1.0, 10.0, n) for v in GROUPING_VARS})
    df["prauc"] = 5.0 * df["classifier_accuracy"] + rng.normal(0, 0.001, n)
    return df


def _visible(fig):
    return [a for a in fig.axes if a.get_visible()]


def _means_by_plates(df):
    return list(np.round(df.groupby("nr_plates")["prauc"].mean().values, 6))


def test_plot_simulations_draws_one_panel_per_condition(subtests):
    """plot_simulations: one PR-AUC-vs-variable panel per varying condition, unused grid cells hidden."""
    df = _sweep_results()
    by_acc = {acc: _means_by_plates(df[df.classifier_accuracy == acc]) for acc in (0.8, 0.95)}
    cases = [
        # BUG (fixed): the clean branch read ``relevant_data`` before assigning it.
        ("default_clean_groups_varying", df, {}, [True, True],
         sorted(by_acc.values())),
        # constants add no splits even when kept
        ("clean_false_legend_grid", df, dict(clean=False, legend=True, grid=True),
         [True, True], sorted(by_acc.values())),
        ("single_panel_when_nothing_else_varies", df[df.classifier_accuracy == 0.8], {},
         [True], [by_acc[0.8]]),
        ("rotation_and_verbose", df, dict(x_rotation=90, verbose=True), [True, True],
         sorted(by_acc.values())),
        # 3 conditions land in a 2x2 grid and the 4th panel is switched off
        ("three_conditions_hide_fourth_cell", _sweep_results(accuracies=(0.7, 0.85, 0.95)),
         {}, [True, True, True, False], None),
    ]
    for case_id, frame, kwargs, visibility, means in cases:
        with subtests.test(msg=case_id):
            fig = S.plot_simulations(frame, "nr_plates", **kwargs)
            assert [a.get_visible() for a in fig.axes] == visibility
            for ax in _visible(fig):
                assert ax.get_xlabel() == "nr_plates"
                assert ax.get_ylabel() == "Precision-Recall AUC (PRAUC)"
                assert list(ax.lines[0].get_xdata()) == [1, 2, 4]
                assert ax.lines[0].get_label() == "Mean PRAUC"
                labels = ax.get_xticklabels()
                assert [t.get_text() for t in labels] == ["1", "2", "4"]
                assert {t.get_rotation() for t in labels} == {
                    float(kwargs.get("x_rotation", 45))}
                assert (ax.get_legend() is not None) == kwargs.get("legend", False)
                annotated = [t.get_text() for t in ax.texts]
                if kwargs.get("verbose"):
                    assert len(annotated) == 1 and annotated[0].startswith("classifier_accuracy: ")
                else:
                    assert annotated == []
            if means is not None:
                drawn = sorted(list(np.round(a.lines[0].get_ydata(), 6)) for a in _visible(fig))
                assert drawn == means

    with subtests.test(msg="missing_columns_refused"):
        with pytest.raises(ValueError, match="DataFrame must contain") as raised:
            S.plot_simulations(df.drop(columns=["sequencing_error", "prauc"]), "nr_plates")
        assert "sequencing_error" in str(raised.value) and "prauc" in str(raised.value)
        assert str(raised.value).endswith(" columns")


def _with_constant_plates():
    df = _importance_df()
    df["nr_plates"] = 3.0
    return df


def test_plot_feature_importance_ranks_the_features_it_fits(tmp_path, subtests):
    """plot_feature_importance: bars sorted by importance and labelled with their own feature."""
    cases = [
        # id, frame, kwargs, expected label set
        # BUG (fixed): tick labels were reversed relative to the bars.
        ("clean_all_features", _importance_df(), dict(clean=True), set(GROUPING_VARS)),
        ("exclude_a_list", _importance_df(), dict(exclude=["nr_plates", "sequencing_error"]),
         set(GROUPING_VARS) - {"nr_plates", "sequencing_error"}),
        ("exclude_one_name", _importance_df(), dict(exclude="nr_plates"),
         set(GROUPING_VARS) - {"nr_plates"}),
        ("clean_false_keeps_constant", _with_constant_plates(), dict(clean=False),
         set(GROUPING_VARS)),
        ("clean_drops_constant", _with_constant_plates(), dict(clean=True),
         set(GROUPING_VARS) - {"nr_plates"}),
    ]
    for case_id, frame, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            fig = S.plot_feature_importance(frame, target="prauc", **kwargs)
            ax = fig.axes[0]
            labels = [t.get_text() for t in ax.get_yticklabels()]
            widths = [p.get_width() for p in ax.patches]
            assert set(labels) == expected and len(labels) == len(expected)
            assert widths == sorted(widths, reverse=True)
            assert labels[0] == "classifier_accuracy"
            assert widths[0] > 0.9
            assert sum(widths) == pytest.approx(1.0)
            assert ax.get_xlabel() == "Feature Importance"
            assert ax.get_title() == "Feature Importances"
            if "nr_plates" in labels and case_id == "clean_false_keeps_constant":
                # a constant feature can never be split on
                assert widths[labels.index("nr_plates")] == pytest.approx(0.0)
            assert (tmp_path / "figures" / "feature_importance" / "1_figure.pdf").is_file()


def test_calculate_permutation_importance_labels_bars_from_the_features(tmp_path, subtests):
    """calculate_permutation_importance: ascending bars labelled from the feature list, excludes honoured."""
    target_first = _importance_df()[["prauc"] + GROUPING_VARS]
    cases = [
        # BUG (fixed): labels came from df.columns, which only matched by accident.
        ("target_first_column_order", target_first, dict(n_repeats=3), set(GROUPING_VARS)),
        # BUG (fixed): a list exclude also ran the single-name branch and raised.
        ("exclude_a_list", _importance_df(), dict(exclude=["nr_plates"], n_repeats=2),
         set(GROUPING_VARS) - {"nr_plates"}),
        ("exclude_one_name", _importance_df(), dict(exclude="sequencing_error", n_repeats=2),
         set(GROUPING_VARS) - {"sequencing_error"}),
        # BUG (fixed): an unknown name raised ValueError from features.remove().
        ("unknown_exclude_ignored", _importance_df(), dict(exclude="not_a_feature", n_repeats=2),
         set(GROUPING_VARS)),
        ("clean_false_keeps_constant", _with_constant_plates(), dict(n_repeats=2, clean=False),
         set(GROUPING_VARS)),
    ]
    for case_id, frame, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            fig = S.calculate_permutation_importance(frame, target="prauc", **kwargs)
            ax = fig.axes[0]
            labels = [t.get_text() for t in ax.get_yticklabels()]
            widths = [p.get_width() for p in ax.patches]
            assert set(labels) == expected and len(labels) == len(expected)
            assert "prauc" not in labels
            assert widths == sorted(widths)
            assert labels[-1] == "classifier_accuracy"
            assert widths[-1] == max(widths)
            assert ax.get_xlabel() == "Permutation Importance"
            assert (tmp_path / "figures" / "permutation_importance" / "1_figure.pdf").is_file()


def _correlation_df():
    df = _importance_df()
    df["optimal_threshold"] = np.linspace(0.2, 0.8, len(df))
    df["accuracy"] = np.linspace(0.5, 1.0, len(df))
    df["roc_auc"] = np.linspace(0.6, 0.99, len(df))
    df["genes_per_well_gini"] = np.linspace(0.1, 0.9, len(df))
    df["wells_per_gene_gini"] = np.linspace(0.2, 0.8, len(df))
    return df


METRICS = ['optimal_threshold', 'accuracy', 'prauc', 'roc_auc',
           'genes_per_well_gini', 'wells_per_gene_gini']


def test_plot_correlation_matrix_covers_the_sweep_and_metric_columns(tmp_path, subtests):
    """plot_correlation_matrix: one row/column per sweep + metric column, constants dropped only when cleaning."""
    constant = _correlation_df()
    constant["nr_plates"] = 3.0
    cases = [
        ("clean_annotated", _correlation_df(), dict(annot=True, clean=True),
         GROUPING_VARS + METRICS),
        ("clean_drops_constant", constant, dict(annot=False, clean=True),
         [c for c in GROUPING_VARS if c != "nr_plates"] + METRICS),
        ("clean_false_keeps_constant", constant, dict(annot=False, clean=False),
         GROUPING_VARS + METRICS),
    ]
    for case_id, frame, kwargs, columns in cases:
        with subtests.test(msg=case_id):
            fig = S.plot_correlation_matrix(frame, **kwargs)
            ax = fig.axes[0]
            ticks = [t.get_text() for t in ax.get_xticklabels()]
            assert ticks == columns
            # the upper triangle (incl. diagonal) is masked; annotations only when asked
            n = len(columns)
            assert len(ax.texts) == (n * (n - 1) // 2 if kwargs["annot"] else 0)
            assert (tmp_path / "figures" / "correlation_matrix" / "1_figure.pdf").is_file()


def test_plot_partial_dependences_gives_every_feature_its_own_panel(tmp_path, subtests):
    """plot_partial_dependences: a 4-column grid with one titled panel per feature, spare cells hidden."""
    three = _importance_df()
    keep = ['avg_reads_per_gene', 'classifier_accuracy', 'nr_plates']
    for col in GROUPING_VARS:
        if col not in keep:
            three[col] = 1.0
    cases = [
        ("clean_false_all_features", _importance_df(), False, GROUPING_VARS, 1),
        ("clean_multi_row_grid", _importance_df(), True, GROUPING_VARS, 1),
        # BUG (fixed): a single row of axes was handed whole to the first feature.
        ("clean_single_row", three, True, keep, 1),
    ]
    for case_id, frame, clean, features, hidden in cases:
        with subtests.test(msg=case_id):
            fig = S.plot_partial_dependences(frame, target="prauc", clean=clean)
            titled = [a.get_title() for a in _visible(fig) if a.get_title()]
            assert titled == features
            assert len([a for a in fig.axes if not a.get_visible()]) == hidden
            pdp = [a for a in _visible(fig) if a.get_ylabel() == "Partial dependence"]
            assert sorted(a.get_xlabel() for a in pdp) == sorted(features)
            assert fig._suptitle.get_text() == "Partial Dependence Plots"
            assert (tmp_path / "figures" / "partial_dependences" / "1_figure.pdf").is_file()
