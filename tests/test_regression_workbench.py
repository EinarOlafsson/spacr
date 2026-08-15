"""Tests for the regression workbench: corrections, inputs, diagnostics, plot.

Grouped by the promise each piece makes, because that is what a regression
here would break:

* every offered correction is actually implemented and behaves like its family;
* a path setting can be filled by picking or dropping files, repeatedly;
* the readable settings resolve to the historical ones without changing them;
* ``inference='auto'`` refuses a design that cannot identify its parameters;
* the design diagnostics detect that refusal for the right reason;
* the volcano renders every styling path and exports what is on screen.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from spacr import multiple_testing as mt
from spacr.regression_diagnostics import (
    collinear_guide_pairs,
    design_report,
    residual_report,
    variance_inflation_factors,
    write_diagnostic_suite,
)
from spacr.volcano_style import (
    VolcanoStyle,
    _resolve_effect_threshold,
    point_details,
    render_volcano,
)

pytestmark = pytest.mark.qt


def _tracked(qtbot, widget):
    """Hand a widget to qtbot so Qt destroys it with the test."""
    qtbot.addWidget(widget)
    return widget



# --------------------------------------------------------------- corrections


@pytest.mark.parametrize("method", list(mt.METHODS))
def test_every_offered_method_runs_and_is_monotone(method):
    """A method in the dropdown must exist, run, and order like its P values.

    The dropdown is built from ``METHODS``, so an entry with no working
    implementation is a control the user can select that then fails mid-run.
    """
    rng = np.random.default_rng(0)
    p = np.sort(np.concatenate([rng.uniform(0, 1, 200),
                                rng.uniform(0, 1e-5, 20)]))
    adjusted, rejected = mt.adjust_p_values(p, method=method, alpha=0.05)
    assert np.all(np.isfinite(adjusted))
    assert np.all((adjusted >= 0) & (adjusted <= 1))
    # Adjusted values never decrease as the raw P values increase.
    assert np.all(np.diff(adjusted) >= -1e-12)
    assert rejected.dtype == bool


def test_family_wise_methods_are_at_least_as_strict_as_fdr():
    """FWER control cannot make more discoveries than FDR control."""
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 1, 400), rng.uniform(0, 1e-4, 40)])
    fdr = mt.adjust_p_values(p, "fdr_bh")[1].sum()
    for method in ("bonferroni", "sidak", "holm", "holm_sidak",
                   "simes_hochberg", "hommel"):
        assert mt.adjust_p_values(p, method)[1].sum() <= fdr, method


def test_storey_is_never_more_conservative_than_benjamini_hochberg():
    """The whole point of estimating pi0 is that it cannot cost power."""
    rng = np.random.default_rng(2)
    p = np.concatenate([rng.uniform(0, 1, 200), rng.beta(0.3, 8, 300)])
    bh = mt.adjust_p_values(p, "fdr_bh")[0]
    storey = mt.adjust_p_values(p, "storey")[0]
    assert np.all(storey <= bh + 1e-12)
    # With 60% non-null, pi0 must fall well below 1 or it is not adaptive.
    assert mt.estimate_pi0(p) < 0.9


def test_missing_p_values_do_not_join_the_family():
    """A guide that could not be tested is not a test.

    Counting it inflates the family size and makes every real discovery less
    significant, which is a silent loss of power rather than an error.
    """
    p = np.array([0.001, np.nan, 0.5, 0.02])
    adjusted, rejected = mt.adjust_p_values(p, "fdr_bh")
    assert np.isnan(adjusted[1]) and not rejected[1]
    # Family size 3, so the smallest P value is multiplied by 3/1.
    assert adjusted[0] == pytest.approx(0.003)


@pytest.mark.parametrize("spelling,expected", [
    ("BH", "fdr_bh"), ("bh", "fdr_bh"), ("benjamini-hochberg", "fdr_bh"),
    ("b", "bonferroni"), ("Holm-Sidak", "holm_sidak"), ("ho", "hommel"),
    ("qvalue", "storey"), (None, "none"), ("", "none"), ("raw", "none"),
])
def test_method_aliases_resolve(spelling, expected):
    assert mt.canonical_method(spelling) == expected


def test_an_unknown_method_is_refused_with_the_inventory():
    with pytest.raises(ValueError, match="fdr_bh"):
        mt.adjust_p_values([0.1], "definitely-not-a-method")


def test_guide_permutation_reuses_the_shared_inventory():
    """One inventory, or the dropdown and the analysis can disagree."""
    from spacr.guide_permutation import MULTIPLE_TESTING_METHODS

    assert set(MULTIPLE_TESTING_METHODS) == set(mt.METHODS)


def test_adjusted_value_label_names_the_right_quantity():
    from spacr.guide_permutation import adjusted_value_label

    assert adjusted_value_label("fdr_bh") == "BH q"
    assert adjusted_value_label("storey") == "Storey q"
    assert adjusted_value_label("none") == "P"
    # A family-wise method adjusts a P value; it does not produce a q value.
    assert adjusted_value_label("holm") == "adjusted P"


# ----------------------------------------------------------- file list widget


@pytest.fixture()
def csv_folder(tmp_path):
    for name in ("plate1.csv", "plate2.csv", "plate3.csv"):
        (tmp_path / name).write_text("a,b\n1,2\n")
    (tmp_path / "picture.png").write_bytes(b"\x89PNG")
    return tmp_path


def test_file_list_appends_across_several_picks(qtbot, csv_folder):
    """Pressing Add files twice must add to the list, not replace it."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = _tracked(qtbot, FilePathListWidget(kind="csv"))
    widget.add_paths([str(csv_folder / "plate1.csv")])
    widget.add_paths([str(csv_folder / "plate2.csv")])
    assert len(widget.get_value()) == 2
    # And the same file twice is one file.
    assert widget.add_paths([str(csv_folder / "plate2.csv")]) == 0
    assert len(widget.get_value()) == 2


def test_file_list_ignores_the_legacy_placeholder(qtbot):
    """'list of paths' was the old default and is not a path."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    assert _tracked(qtbot, FilePathListWidget(value="list of paths")).get_value() == []


def test_file_list_expands_a_folder_by_kind(qtbot, csv_folder):
    """A folder dropped on a CSV setting must not contribute its PNGs."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = _tracked(qtbot, FilePathListWidget(kind="csv"))
    widget.add_paths([str(csv_folder)])
    names = [os.path.basename(p) for p in widget.get_value()]
    assert sorted(names) == ["plate1.csv", "plate2.csv", "plate3.csv"]


def test_file_list_keeps_a_missing_path_and_flags_it(qtbot):
    """A settings file may be written on one machine and run on another."""
    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = _tracked(qtbot, FilePathListWidget(kind="csv"))
    widget.add_paths(["/no/such/file.csv"])
    assert len(widget.get_value()) == 1
    assert "not found" in widget._hint.text()


def test_file_list_accepts_a_real_drop(qtbot, csv_folder):
    from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PySide6.QtGui import QDropEvent

    from spacr.qt.widgets.file_list import FilePathListWidget

    widget = _tracked(qtbot, FilePathListWidget(kind="csv"))
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(csv_folder / "plate1.csv"))])
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                       Qt.NoModifier)
    widget.dropEvent(event)
    assert event.isAccepted()
    assert len(widget.get_value()) == 1


def test_path_settings_render_as_file_pickers(qtbot):
    """The keys that name input files must get the picker, not a text box."""
    from spacr.qt.screens.settings_model import PATH_LIST_KEYS

    assert {"score_data", "count_data", "metadata_files"} <= set(PATH_LIST_KEYS)


# -------------------------------------------------------------- the settings


def test_inference_selects_the_analysis_mode():
    from spacr.settings import get_perform_regression_default_settings as defaults

    assert defaults({"inference": "parametric"})["analysis_mode"] == "regression"
    assert defaults({"inference": "nonparametric"})["analysis_mode"] == \
        "guide_permutation"
    # 'auto' leaves the decision for the run, which is the only place the
    # guides and wells can be counted.
    assert defaults({"inference": "auto"})["inference"] == "auto"


def test_the_default_does_not_move_an_existing_run_onto_a_new_estimand():
    """Defaulting to 'auto' would silently change what every old run does.

    A settings file written before this feature existed asks for the
    simultaneous fit. Re-running it must still fit that model -- a different
    estimand with different columns and different numbers, chosen by nobody,
    is worse than a design warning.
    """
    from spacr.settings import get_perform_regression_default_settings as defaults

    resolved = defaults({})
    assert resolved["inference"] == "parametric"
    assert resolved["analysis_mode"] == "regression"


def test_analysis_unit_cell_is_the_explicit_spelling_of_agg_type_none():
    from spacr.settings import get_perform_regression_default_settings as defaults

    assert defaults({"analysis_unit": "cell"})["agg_type"] is None
    assert defaults({"analysis_unit": "well"})["agg_type"] == "mean"


def test_regression_type_auto_becomes_the_historical_none():
    from spacr.settings import get_perform_regression_default_settings as defaults

    assert defaults({"regression_type": "auto"})["regression_type"] is None


def test_bad_inference_and_unit_are_refused_not_guessed():
    from spacr.settings import get_perform_regression_default_settings as defaults

    with pytest.raises(ValueError, match="inference"):
        defaults({"inference": "bayesian-ish"})
    with pytest.raises(ValueError, match="analysis_unit"):
        defaults({"analysis_unit": "plate"})


def test_path_defaults_are_empty_lists_not_a_placeholder_string():
    from spacr.settings import get_perform_regression_default_settings as defaults

    resolved = defaults({})
    assert resolved["score_data"] == [] and resolved["count_data"] == []


def test_dependent_variable_accepts_several_responses():
    from spacr.settings import expected_types

    assert list in (expected_types["dependent_variable"]
                    if isinstance(expected_types["dependent_variable"], tuple)
                    else (expected_types["dependent_variable"],))


# ------------------------------------------------------------ auto inference


def _design_frame(n_wells, n_guides, plates=4, guides_per_well=5):
    rng = np.random.default_rng(0)
    rows = []
    for well in range(n_wells):
        for guide in rng.choice(n_guides, size=min(guides_per_well, n_guides),
                                replace=False):
            rows.append({"prc": f"w{well}", "grna": f"g{guide}",
                         "plateID": f"plate{well % plates + 1}"})
    return pd.DataFrame(rows)


def test_auto_refuses_a_design_it_cannot_identify():
    """824 guides in 587 wells is the shape that produced the bad figure."""
    from spacr.ml import resolve_auto_inference

    mode, reason = resolve_auto_inference(
        _design_frame(587, 824), {"inference": "auto"})
    assert mode == "guide_permutation"
    assert "cannot identify" in reason


def test_auto_allows_a_design_with_room_to_spare():
    from spacr.ml import resolve_auto_inference

    mode, _reason = resolve_auto_inference(
        _design_frame(600, 40), {"inference": "auto"})
    assert mode == "regression"


def test_auto_never_overrides_an_explicit_choice():
    from spacr.ml import resolve_auto_inference

    mode, _reason = resolve_auto_inference(
        _design_frame(587, 824),
        {"inference": "parametric", "analysis_mode": "regression"})
    assert mode == "regression"


def test_an_unidentifiable_parametric_fit_is_warned_about_loudly():
    """The default runs what was asked for, but cannot be quiet about it."""
    from spacr.ml import _identifiability_warning

    warning = _identifiability_warning(_design_frame(587, 824), {})
    assert warning is not None
    assert "not identifiable" in warning
    # It has to name the numbers, or it is just noise the user learns to skip.
    assert "587" in warning and "nonparametric" in warning


def test_no_warning_when_the_design_is_fine():
    from spacr.ml import _identifiability_warning

    assert _identifiability_warning(_design_frame(600, 40), {}) is None


# -------------------------------------------------------------- diagnostics


def _fractions(n_wells, n_guides, guides_per_well=5, seed=0):
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_wells, n_guides))
    for well in range(n_wells):
        for guide in rng.choice(n_guides, size=guides_per_well, replace=False):
            matrix[well, guide] = rng.uniform(0.02, 0.4)
    return pd.DataFrame(matrix, columns=[f"g{i}" for i in range(n_guides)],
                        index=[f"w{i}" for i in range(n_wells)])


def test_design_report_calls_the_published_shape_unidentifiable():
    frame = _fractions(587, 824)
    block = pd.Series([f"plate{i % 4 + 1}" for i in range(587)],
                      index=frame.index)
    report = design_report(frame, block=block)
    assert report["identifiable"] is False
    assert report["non_identifiable_directions"] > 0
    assert report["wells_per_parameter"] < 1


def test_design_report_passes_a_healthy_design():
    report = design_report(_fractions(600, 40, guides_per_well=8))
    assert report["identifiable"] is True
    assert report["residual_degrees_of_freedom"] > 0


def test_collinear_pairs_finds_a_planted_duplicate():
    frame = _fractions(200, 30)
    frame["clone"] = frame["g0"]
    pairs = collinear_guide_pairs(frame, threshold=0.95)
    assert {"g0", "clone"} == set(pairs.iloc[0][["guide_a", "guide_b"]])


def test_collinear_pairs_returns_an_empty_typed_frame():
    """The healthy case must not raise -- it did, on a missing column."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.uniform(size=(100, 5)),
                         columns=[f"g{i}" for i in range(5)])
    pairs = collinear_guide_pairs(frame, threshold=0.99)
    assert pairs.empty
    assert list(pairs.columns) == ["guide_a", "guide_b", "correlation",
                                   "shared_wells"]


def test_vif_refuses_a_rank_deficient_design_instead_of_returning_infinities():
    with pytest.raises(ValueError, match="rank deficient"):
        variance_inflation_factors(_fractions(50, 200))


def test_vif_is_near_one_for_independent_predictors():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(500, 6)),
                         columns=[f"g{i}" for i in range(6)])
    assert variance_inflation_factors(frame)["vif"].max() < 1.5


def test_residual_report_detects_heteroscedasticity():
    rng = np.random.default_rng(5)
    x = rng.uniform(1, 10, 400)
    fitted = 2 * x
    # Spread grows with the fitted value, which is what the test must catch.
    observed = fitted + rng.normal(0, x, 400)
    report = residual_report(observed, fitted)
    assert report["heteroscedasticity_p_value"] < 0.05


def test_diagnostic_suite_writes_what_its_inputs_allow(tmp_path):
    frame = _fractions(120, 60)
    block = pd.Series([f"plate{i % 4 + 1}" for i in range(120)])
    rng = np.random.default_rng(6)
    p = np.concatenate([rng.uniform(0, 1, 55), rng.uniform(0, 1e-4, 5)])
    adjusted, _ = mt.adjust_p_values(p, "fdr_bh")
    written = write_diagnostic_suite(
        tmp_path, fractions=frame, block=block, p_values=p, adjusted=adjusted,
        formats=("png",))
    assert not [k for k in written if k.endswith("_error")]
    assert "design_diagnostics_png" in written
    assert "inference_diagnostics_png" in written
    assert os.path.exists(written["diagnostic_summary"])


def test_a_failing_diagnostic_is_recorded_not_raised(tmp_path):
    """A diagnostic must never take the analysis down with it."""
    written = write_diagnostic_suite(
        tmp_path, observed=[1.0, 2.0], fitted=["not", "numeric"],
        formats=("png",))
    assert any(key.endswith("_error") for key in written)


# ------------------------------------------------------------------- volcano


@pytest.fixture()
def volcano_results():
    rng = np.random.default_rng(7)
    n = 200
    frame = pd.DataFrame({
        "guide": [f"g{i}" for i in range(n)],
        "gene": [f"TGGT1_{200000 + i // 3}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(0, 0.08, n),
        "adjusted_p_value": np.clip(rng.beta(0.6, 6, n), 1e-8, 1),
        "wells_with_guide": rng.integers(1, 12, n),
        "compartment": rng.choice(["GRA", "ROP", "other"], n),
        "significant": False,
        "alpha": 0.05,
    })
    frame.loc[[3, 17], "standardized_marginal_effect"] = [0.25, 0.28]
    frame.loc[[3, 17], "adjusted_p_value"] = 0.004
    frame.loc[[3, 17], "significant"] = True
    return frame


@pytest.mark.parametrize("style", [
    VolcanoStyle(),
    VolcanoStyle(color_by="wells_with_guide", colormap="plasma", marker="D"),
    VolcanoStyle(color_by="compartment", shape_by="compartment",
                 colormap="tab10"),
    VolcanoStyle(split_axis=True, split_y_lims=((0, 1.2), (2.0, 2.6))),
    VolcanoStyle(y_scale="log", y_neg_log10=False, line_style=":",
                 line_width=2.2, grid_axis="both"),
    VolcanoStyle(font_family="serif", font_size=14, invert_x=True,
                 legend=False, hide_top_right_spines=False),
])
def test_every_styling_path_renders(volcano_results, style, tmp_path):
    path = tmp_path / "volcano.png"
    figure, panels = render_volcano(volcano_results, style, save_path=path)
    assert path.exists() and path.stat().st_size > 0
    assert len(panels) == (2 if style.split_axis else 1)
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize("method,multiplier,expected", [
    ("value", 1.0, 0.2),
    ("std", 0.0, 0.0),
])
def test_effect_threshold_rules(method, multiplier, expected):
    style = VolcanoStyle(threshold_method=method,
                         threshold_multiplier=multiplier,
                         effect_threshold=0.2)
    values = np.array([-0.3, -0.1, 0.0, 0.1, 0.3])
    assert _resolve_effect_threshold(values, style) == pytest.approx(expected)


def test_quantile_threshold_tracks_the_data():
    style = VolcanoStyle(threshold_method="quantile", threshold_multiplier=0.5)
    values = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    assert _resolve_effect_threshold(values, style) == pytest.approx(2.0)


def test_an_unknown_threshold_method_is_refused():
    style = VolcanoStyle(threshold_method="vibes")
    with pytest.raises(ValueError, match="threshold_method"):
        _resolve_effect_threshold(np.array([1.0, 2.0]), style)


def test_style_round_trips_through_json(tmp_path):
    style = VolcanoStyle(colormap="magma", marker="*", font_size=13.5,
                         annotations={"g3": "EAF1"})
    reloaded = VolcanoStyle.load(style.save(tmp_path / "style.json"))
    assert reloaded.colormap == "magma"
    assert reloaded.marker == "*"
    assert reloaded.annotations == {"g3": "EAF1"}


def test_style_ignores_settings_from_a_newer_version():
    """A style saved by a later spaCR must still load, minus what is new."""
    style = VolcanoStyle.from_dict({"colormap": "cividis",
                                    "a_setting_from_the_future": 42})
    assert style.colormap == "cividis"


def test_a_missing_column_is_named_in_the_error(volcano_results):
    with pytest.raises(ValueError, match="not_a_column"):
        render_volcano(volcano_results, VolcanoStyle(x_column="not_a_column"))


def test_point_details_reports_plotted_and_source_values(volcano_results):
    detail = point_details(volcano_results, 3, VolcanoStyle())
    assert detail["guide"] == "g3"
    assert detail["_plotted_x"] == pytest.approx(0.25)
    assert detail["_plotted_y"] == pytest.approx(-np.log10(0.004))


# ---------------------------------------------------------------- explorer


def test_explorer_click_finds_the_nearest_point(qtbot, volcano_results):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    axes = explorer._panels[0]
    index = explorer.nearest_point(0.25, -np.log10(0.004), axes)
    assert volcano_results.loc[index, "guide"] == "g3"
    # A click on empty space selects nothing rather than the nearest dot
    # somewhere off screen.
    assert explorer.nearest_point(1e6, 1e6, axes) is None


def test_explorer_select_shows_every_field(qtbot, volcano_results):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    seen = []
    explorer.point_selected.connect(seen.append)
    detail = explorer.select_point(17)
    assert detail["guide"] == "g17"
    assert seen and seen[0]["gene"] == detail["gene"]
    # Every source column, plus the two derived plotted coordinates.
    assert explorer._detail_table.rowCount() == len(volcano_results.columns) + 2


def test_explorer_controls_write_through_to_the_style(qtbot, volcano_results):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    explorer._controls["colormap"].setCurrentIndex(
        explorer._controls["colormap"].findData("plasma"))
    explorer._controls["font_size"].setValue(15.0)
    explorer._controls["marker"].setCurrentIndex(
        explorer._controls["marker"].findData("D"))
    assert explorer.style().colormap == "plasma"
    assert explorer.style().font_size == 15.0
    assert explorer.style().marker == "D"


def test_explorer_derives_split_limits_when_asked_to_split(qtbot,
                                                          volcano_results):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    explorer._controls["split_axis"].setChecked(True)
    lower, upper = explorer.style().split_y_lims
    assert lower[1] <= upper[0]


def test_explorer_merges_an_annotation_file_and_offers_its_columns(
        qtbot, volcano_results, tmp_path):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    annotation = pd.DataFrame({
        "gene": volcano_results["gene"].unique(),
        "expression": np.arange(volcano_results["gene"].nunique()),
    })
    path = tmp_path / "annotations.csv"
    annotation.to_csv(path, index=False)
    assert explorer.merge_annotation_file(path) == 1
    offered = [explorer._controls["color_by"].itemText(i)
               for i in range(explorer._controls["color_by"].count())]
    assert "expression" in offered


def test_explorer_refuses_an_unjoinable_annotation_file(qtbot, volcano_results,
                                                       tmp_path):
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    path = tmp_path / "unrelated.csv"
    pd.DataFrame({"something_else": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="shares no column"):
        explorer.merge_annotation_file(path)


@pytest.mark.parametrize("fmt", ["pdf", "png"])
def test_explorer_exports_by_rerendering(qtbot, volcano_results, tmp_path, fmt):
    """Export must be a re-render, not a screenshot of the widget."""
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    path = explorer.export(fmt, path=str(tmp_path / f"volcano.{fmt}"))
    assert os.path.exists(path) and os.path.getsize(path) > 1000
    if fmt == "pdf":
        # Vector, so the PDF header is there and the file is not a bitmap.
        with open(path, "rb") as handle:
            assert handle.read(4) == b"%PDF"


def test_explorer_survives_an_impossible_style(qtbot, volcano_results):
    """A bad setting must draw an explanation, not raise into the event loop."""
    from spacr.qt.widgets.volcano_explorer import VolcanoExplorer

    explorer = _tracked(qtbot, VolcanoExplorer(volcano_results))
    explorer._style.x_column = "does_not_exist"
    explorer.refresh()
    assert explorer._panels
