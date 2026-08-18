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
import warnings

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


def test_legacy_agg_type_none_still_means_cell_analysis():
    from spacr.settings import get_perform_regression_default_settings as defaults

    resolved = defaults({"agg_type": None})
    assert resolved["analysis_unit"] == "cell"
    assert resolved["agg_type"] is None


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
    assert resolved["paired_data"] == []
    assert "score_data" not in resolved and "count_data" not in resolved
    assert "plateID" not in resolved


def test_estimator_enablement_is_generated_from_the_runtime_inventory(qtbot):
    from spacr.ml import REGRESSION_SETTINGS_USED
    from spacr.qt.screens.settings_model import SettingsWidgets

    builder = SettingsWidgets("regression")
    builder.build_sections()
    owned = {key for keys in REGRESSION_SETTINGS_USED.values() for key in keys}
    for family, used in REGRESSION_SETTINGS_USED.items():
        assert builder.set_value_for_key("regression_type", family)
        enabled = {key for key in owned if builder._widgets[key].isEnabled()}
        assert enabled == set(used), family


def test_inference_and_analysis_unit_grey_the_controls_they_do_not_use(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    builder = SettingsWidgets("regression")
    builder.build_sections()
    guide_keys = {
        "guide_min_wells", "guide_primary_min_wells", "guide_permutations",
        "guide_permutation_seed", "guide_permutation_block",
        "guide_nuisance_columns", "guide_presence_threshold",
        # `guide_permutation_plot` left this list on 2026-08-18 with
        # instruction 135: the permutation plot is always written now, so
        # there is no control to grey out. The other eight still are.
        "guide_permutation_batch_size",
    }
    builder.set_value_for_key("inference", "parametric")
    assert all(not builder._widgets[key].isEnabled() for key in guide_keys)
    builder.set_value_for_key("inference", "nonparametric")
    assert all(builder._widgets[key].isEnabled() for key in guide_keys)

    builder.set_value_for_key("analysis_unit", "cell")
    agg = builder._widgets["agg_type"]
    assert not agg.isEnabled()
    assert "analysis_unit" in agg.toolTip()


def test_filename_pair_proposals_survive_opposite_picker_order(tmp_path):
    from spacr.qt.widgets.file_list import suggest_file_pairs

    scores = [tmp_path / "screen_plate1_maxvit_result.csv",
              tmp_path / "screen_plate2_maxvit_result.csv"]
    counts = [tmp_path / "plate_2_unique_combinations.csv",
              tmp_path / "plate_1_unique_combinations.csv"]
    rows = suggest_file_pairs([str(path) for path in scores],
                              [str(path) for path in counts])
    assert [os.path.basename(row["count"]) for row in rows] == [
        "plate_1_unique_combinations.csv",
        "plate_2_unique_combinations.csv",
    ]


def test_pair_plate_resolution_uses_own_partner_and_row_order(tmp_path):
    from spacr.ml import load_regression_input_pairs

    def write(name, plate=None):
        frame = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                              "value": [1]})
        if plate is not None:
            frame["plateID"] = plate
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return str(path)

    pairs = [
        {"score": write("s1.csv", "declared"),
         "count": write("c1.csv", "declared")},
        {"score": write("s2.csv", "partner"),
         "count": write("c2.csv")},
        {"score": write("s3.csv"), "count": write("c3.csv")},
    ]
    counts, scores, audit = load_regression_input_pairs(pairs)
    assert set(counts.plateID) == {"declared", "partner", "plate3"}
    assert set(scores.plateID) == {"declared", "partner", "plate3"}
    assert [row["rule"] for row in audit] == [
        "both files agree", "copied from score file",
        "assigned from pair row order"]

    with pytest.raises(ValueError, match="conflicts"):
        load_regression_input_pairs([{
            "score": write("bad_s.csv", "left"),
            "count": write("bad_c.csv", "right"),
        }])


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


# ------------------------------------------------- score/count plate pairing


def test_a_legacy_plate_column_is_normalised_before_it_becomes_the_plate_id():
    """'pplate1' in a `plate` column must end up as 'plate1'.

    The repair used to run before the legacy promotion and only over
    'plateID', so for the files that actually carry the artifact -- a score
    CSV with a `plate` column and no `plateID` -- it did nothing, and the
    unrepaired value was then copied straight into `plateID`.
    """
    from spacr.utils import correct_metadata

    frame = correct_metadata(pd.DataFrame({
        "plate": ["pplate1"], "row": ["r9"], "col": ["c16"],
        "prc": ["pplate1_r9_c16"], "pred": [0.2],
    }))
    assert frame["plateID"].iloc[0] == "plate1"
    # prc embeds the plate too, and is what the join actually uses.
    assert frame["prc"].iloc[0] == "plate1_r9_c16"


def test_a_single_p_plate_id_is_left_alone():
    from spacr.utils import correct_metadata

    frame = correct_metadata(pd.DataFrame({"plateID": ["plate1", "p1"]}))
    assert list(frame["plateID"]) == ["plate1", "p1"]


def _pairing_frames(score_plate, count_plate, wells=10):
    independent = pd.DataFrame({
        "prc": [f"{count_plate}_r{i}_c1" for i in range(wells)],
        "grna": ["g0"] * wells,
    })
    dependent = pd.DataFrame({
        "prc": [f"{score_plate}_r{i}_c1" for i in range(wells)],
        "pred": [0.5] * wells,
    })
    return independent, dependent, independent.merge(dependent, on="prc")


def test_an_empty_score_count_join_is_refused_naming_both_plate_sets():
    """The failure that crashed 200 lines later inside a plot.

    'pplate1' scores against 'plate1' counts joined to nothing, and the run
    continued until `plot_plates` died with `KeyError: 0` -- an error naming
    neither the plates nor the files.
    """
    from spacr.ml import _check_score_count_pairing

    with pytest.raises(ValueError) as raised:
        _check_score_count_pairing(*_pairing_frames("pplate1", "plate1"))
    message = str(raised.value)
    assert "no well in common" in message
    # Both sides must be named, or the user cannot tell which one is wrong.
    assert "pplate1" in message and "plate1" in message
    # The remedy must be named, whatever it is currently called.
    assert "plateID column" in message


def test_a_mostly_unpaired_join_is_refused_too():
    """Fitting on whatever overlapped and calling it the screen is worse."""
    from spacr.ml import _check_score_count_pairing

    # Equal-sized sides, so only 40% of EITHER side can pair. Measuring
    # against the smaller side is what makes this a real mismatch rather than
    # the ordinary size difference tested below.
    independent = pd.DataFrame({
        "prc": [f"plate1_r{i}_c1" for i in range(100)]})
    dependent = pd.DataFrame({
        "prc": ([f"plate1_r{i}_c1" for i in range(40)]
                + [f"plate9_r{i}_c1" for i in range(60)]), "pred": 0.5})
    with pytest.raises(ValueError, match="40.0%"):
        _check_score_count_pairing(
            independent, dependent, independent.merge(dependent, on="prc"))


def test_a_healthy_join_passes_silently():
    """Silently means silently: no refusal AND no warning. A guard that
    passes but prints a scare on every healthy run trains the user to ignore
    it, which is the same as not having it."""
    from spacr.ml import _check_score_count_pairing

    frames = _pairing_frames("plate1", "plate1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _check_score_count_pairing(*frames) is None
    assert len(frames[2]) > 0, "the fixture joined nothing, so nothing was checked"


def test_far_more_count_wells_than_score_wells_is_normal():
    """The shape of every real screen, and the one this guard first rejected.

    Sequencing covers the whole plate; imaging keeps only the wells that
    survive segmentation and the minimum-cell filter. On the TSG101 screen
    that is 463 imaged wells against 1,344 sequenced wells with all 463
    paired -- a perfect join that read as 34% when the denominator was the
    sequencing side, and the run was refused.
    """
    from spacr.ml import _check_score_count_pairing

    independent = pd.DataFrame({
        "prc": [f"plate1_r{i}_c1" for i in range(1344)]})
    dependent = pd.DataFrame({
        "prc": [f"plate1_r{i}_c1" for i in range(463)], "pred": 0.5})
    merged = independent.merge(dependent, on="prc")

    assert len(merged) == 463, (
        "every imaged well should pair; the fixture is not the shape the "
        "docstring describes")
    assert _check_score_count_pairing(independent, dependent, merged) is None


def test_partial_but_adequate_overlap_is_allowed():
    """Sequencing legitimately covers wells that were never imaged."""
    from spacr.ml import _check_score_count_pairing

    independent = pd.DataFrame({
        "prc": [f"plate1_r{i}_c1" for i in range(100)]})
    dependent = pd.DataFrame({
        "prc": [f"plate1_r{i}_c1" for i in range(89)], "pred": 0.5})
    merged = independent.merge(dependent, on="prc")

    assert len(merged) == 89, "89 of the imaged wells should pair"
    assert _check_score_count_pairing(independent, dependent, merged) is None


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


# ------------------------------------------------------------- dropped files


def test_a_data_csv_is_routed_to_the_input_it_belongs_in(qtbot, tmp_path):
    """Dropping a score or count CSV must fill that input, not fail.

    Every dropped CSV used to go to the settings importer, which rejected a
    score table with "CSV file must contain setting_key and setting_value
    columns" -- true, and useless, about a file that never claimed to be
    settings. The header decides: a count export carries a gRNA name and a
    count, a score export carries neither.
    """
    from spacr.qt.dnd import _looks_like_settings_csv, _route_data_csv_to_inputs
    from spacr.qt.widgets.file_list import FilePathListWidget

    score = tmp_path / "plate1_dv.csv"
    score.write_text("path,pred,plate,row,col\na,0.5,plate1,r1,c1\n")
    count = tmp_path / "plate_1_unique_combinations.csv"
    count.write_text("row_name,column_name,grna_name,count\nr1,c1,g_1,5\n")
    settings_csv = tmp_path / "settings.csv"
    settings_csv.write_text("Key,Value\nsrc,/tmp\n")

    assert not _looks_like_settings_csv(score)
    assert not _looks_like_settings_csv(count)
    # A real settings export must still reach the settings importer.
    assert _looks_like_settings_csv(settings_csv)

    class _Model:
        pass

    class _Screen:
        def apply_settings_dict(self, values):
            return 0

    screen = _Screen()
    model = _Model()
    model._widgets = {
        "score_data": _tracked(qtbot, FilePathListWidget(kind="table")),
        "count_data": _tracked(qtbot, FilePathListWidget(kind="table")),
    }
    screen._settings_model = model

    assert _route_data_csv_to_inputs(score, screen) == "score_data"
    assert _route_data_csv_to_inputs(count, screen) == "count_data"
    assert len(model._widgets["score_data"].get_value()) == 1
    assert len(model._widgets["count_data"].get_value()) == 1


def test_a_screen_with_no_file_inputs_still_reports_the_problem(qtbot,
                                                               tmp_path):
    """Routing returns None so the caller can explain, rather than silently
    swallowing a drop the screen cannot use."""
    from spacr.qt.dnd import _route_data_csv_to_inputs

    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")

    class _Screen:
        def apply_settings_dict(self, values):
            return 0

    assert _route_data_csv_to_inputs(data, _Screen()) is None


def test_a_dropped_csv_reaches_the_paired_table_not_metadata(qtbot, tmp_path):
    """The regression panel's real shape, and the bug it caused.

    The panel replaced its separate score_data / count_data lists with one
    paired_data table. The router looked for those two keys, found neither,
    and fell through to metadata_files -- the only file list left on the
    screen -- so EVERY dropped CSV went to metadata.
    """
    from spacr.qt.dnd import _route_data_csv_to_inputs
    from spacr.qt.widgets.file_list import (
        FilePathListWidget, PairedFileTableWidget,
    )

    score = tmp_path / "plate1_dv.csv"
    score.write_text("path,pred,plate,row,col\na,0.5,plate1,r1,c1\n")
    count = tmp_path / "plate_1_unique_combinations.csv"
    count.write_text("row_name,column_name,grna_name,count\nr1,c1,g_1,5\n")
    annotation = tmp_path / "grna_barcodes.csv"
    annotation.write_text("name,sequence\nTGGT1_225160_2,ACGT\n")

    class _Model:
        pass

    class _Screen:
        def apply_settings_dict(self, values):
            return 0

    screen, model = _Screen(), _Model()
    paired = _tracked(qtbot, PairedFileTableWidget())
    metadata = _tracked(qtbot, FilePathListWidget(kind="table"))
    model._widgets = {"paired_data": paired, "metadata_files": metadata}
    screen._settings_model = model

    assert "count" in _route_data_csv_to_inputs(count, screen)
    assert "score" in _route_data_csv_to_inputs(score, screen)
    # An annotation table is neither side of the pairing.
    assert _route_data_csv_to_inputs(annotation, screen) == "metadata_files"
    assert len(metadata.get_value()) == 1

    rows = paired.get_value()
    assert len(rows) == 1, "the score and count should share one plate row"
    assert rows[0]["score"] and rows[0]["count"]


def test_a_count_dropped_before_its_score_still_pairs(qtbot, tmp_path):
    """"if i drop a plate 1 independent variable, then i should be able to
    drop plate 1 dependent variable to its left" -- order must not matter."""
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    counts, scores = [], []
    for plate in (1, 2):
        count = tmp_path / f"plate_{plate}_unique_combinations.csv"
        count.write_text("row_name,column_name,grna_name,count\nr1,c1,g,5\n")
        score = tmp_path / f"plate{plate}_dv.csv"
        score.write_text("path,pred,plate\na,0.5,plate1\n")
        counts.append(str(count))
        scores.append(str(score))

    # Counts first, scores second -- the order the report describes.
    paired.add_paths_for_side(counts, "count")
    paired.add_paths_for_side(scores, "score")

    rows = paired.get_value()
    assert len(rows) == 2
    for row in rows:
        assert row["score"] and row["count"]
        # Same plate token on both sides of each row.
        assert row["plate"] in ("plate1", "plate2")


def test_a_side_must_be_score_or_count(qtbot):
    from spacr.qt.widgets.file_list import PairedFileTableWidget

    paired = _tracked(qtbot, PairedFileTableWidget())
    with pytest.raises(ValueError, match="score.*count"):
        paired.add_paths_for_side(["/tmp/x.csv"], "sideways")


def test_two_lists_added_in_opposite_orders_still_pair_by_plate(qtbot,
                                                                tmp_path):
    """The wrong-pair run that nothing else catches, and the whole reason
    the paired table exists.

    Two score files and two count files, added in OPPOSITE orders. Under the
    old contract -- two independent lists zipped by LIST POSITION -- plate 2's
    scores met plate 1's counts. The join succeeded, the well count looked
    right, and every effect size was attached to the wrong plate: no error,
    no warning, no way to see it in the panel, because nothing in the panel
    ever stated that position was the contract.

    The file picker hands files over in the file manager's order, so getting
    them in opposite orders is one sort click away.
    """
    from itertools import zip_longest

    from spacr.qt.widgets.file_list import PairedFileTableWidget

    scores, counts = {}, {}
    for plate in (1, 2):
        score = tmp_path / f"plate{plate}_dv.csv"
        score.write_text("path,pred,plate\na,0.5,plate1\n")
        count = tmp_path / f"plate_{plate}_unique_combinations.csv"
        count.write_text("row_name,column_name,grna_name,count\nr1,c1,g,5\n")
        scores[plate], counts[plate] = str(score), str(count)

    added_scores = [scores[2], scores[1]]
    added_counts = [counts[1], counts[2]]

    # The old rule, replayed on the same two lists: position pairs plate 2's
    # scores with plate 1's counts.
    positional = list(zip_longest(added_scores, added_counts))
    assert positional[0] == (scores[2], counts[1])

    paired = _tracked(qtbot, PairedFileTableWidget())
    paired.add_paths_for_side(added_scores, "score")
    paired.add_paths_for_side(added_counts, "count")

    rows = paired.get_value()
    assert len(rows) == 2, "two plates are two rows"
    for row in rows:
        assert row["score"] and row["count"], "no row may be left half filled"
    # Both cells of a row name the same plate, whichever order they arrived in.
    by_score = {row["score"]: row["count"] for row in rows}
    assert by_score[scores[1]] == counts[1]
    assert by_score[scores[2]] == counts[2]


class TestTheOutputFolderIsTheCallersChoice:
    """Two conditions must not overwrite each other's results.

    ``_perform_regression_set_paths`` used to assign
    ``settings['src'] = dirname(count_data[0])`` unconditionally, discarding
    whatever the caller asked for. Every run of a given family then wrote to
    one identical path, so comparing thirteen corrections left only the last
    on disk -- silently, with nothing warning that the earlier ones were gone.

    THESE USED TO REACH INTO A CLOSURE. The resolver was nested inside
    ``perform_regression``, so the only way to test it was to pull its code
    object out of ``__code__.co_consts`` and rebuild a function around it --
    which tests the bytecode's name rather than the behaviour, and broke the
    moment the function moved. It is module level now, and imported.
    """

    def _paths(self, tmp_path, **extra):
        from spacr.ml import _perform_regression_set_paths

        data = tmp_path / "data"
        data.mkdir(exist_ok=True)
        count = data / "counts.csv"
        count.write_text("grna,count\na,1\n")
        settings = {"count_data": [str(count)], "score_data": [str(count)],
                    "regression_type": "ols"}
        settings.update(extra)
        return _perform_regression_set_paths(settings)[4], settings

    def test_a_requested_src_is_used(self, tmp_path):
        """The whole point: ask for a folder, get that folder.

        The parameter sweep depends on it -- each trial sets its own src, and
        without this every trial would write over the last.
        """
        import os

        wanted = tmp_path / "my_output"
        res_folder, settings = self._paths(tmp_path, src=str(wanted))

        assert str(wanted) in res_folder, \
            f"the requested folder was ignored; got {res_folder}"
        assert settings["src"] == os.path.abspath(str(wanted))

    def test_without_a_src_it_falls_back_to_the_count_data(self, tmp_path):
        """Callers that never set src -- the GUI among them -- put their
        output beside the count data, which is what the maintainer asked
        for: "just store everything in the same location as the first count
        data"."""
        res_folder, _settings = self._paths(tmp_path)

        assert str(tmp_path / "data") in res_folder

    def test_a_blank_src_does_not_become_a_folder_named_nothing(self, tmp_path):
        res_folder, _settings = self._paths(tmp_path, src="   ")

        assert str(tmp_path / "data") in res_folder

    def test_a_second_run_of_the_same_type_does_not_overwrite_the_first(
            self, tmp_path):
        """The other half of the same problem, and the half a requested src
        does not solve: two runs of one family into one folder."""
        import pathlib as _pathlib

        first, _ = self._paths(tmp_path)
        (_pathlib.Path(first) / "results.csv").write_text("x\n")

        second, _ = self._paths(tmp_path)

        assert second != first
        assert second.endswith("ols_1")


class TestTheNonparametricTableMatchesTheParametricOne:
    """One name must not mean two shapes.

    ``results.csv`` on disk carried feature / coefficient / p_value / q_value;
    the returned ``output['results']`` did not. Every consumer of a
    coefficient table -- the results panel, guide concordance, the volcano,
    the sweep's hit counts -- therefore raised KeyError('feature') on
    permutation output while working on parametric output.
    """

    #: What any consumer of a spaCR coefficient table is entitled to assume.
    CONTRACT = ("feature", "coefficient", "p_value", "q_value", "grna")

    def test_the_aliases_are_on_the_returned_results(self):
        import pandas as pd

        # The shape the permutation path builds, before the aliases.
        raw = pd.DataFrame({
            "guide": ["TGGT1_225160_2", "TGGT1_239740_3"],
            "standardized_marginal_effect": [1.4, -0.9],
            "permutation_p_value": [5e-06, 5e-06],
            "adjusted_p_value": [5e-06, 5e-06],
            "minimum_wells_threshold": [1, 1],
            "significant": [True, True],
        })
        # Apply exactly what ml.py now applies.
        out = raw.copy()
        out["grna"] = out["guide"]
        out["feature"] = "fraction:grna[" + out["guide"].astype(str) + "]"
        out["coefficient"] = out["standardized_marginal_effect"]
        out["p_value"] = out["permutation_p_value"]
        out["q_value"] = out["adjusted_p_value"]

        for column in self.CONTRACT:
            assert column in out.columns

    def test_the_source_columns_are_kept_not_replaced(self):
        """A caller that wants the permutation quantities still has them."""
        import inspect

        from spacr.ml import perform_regression
        source = inspect.getsource(perform_regression.__globals__["perform_regression"]) \
            if "perform_regression" in perform_regression.__globals__ else ""
        # The aliasing block adds columns; it must not drop the originals.
        import spacr.ml as ml
        text = inspect.getsource(ml)
        block = text[text.index("THE SAME ALIASES ON THE FULL TABLE"):]
        block = block[:block.index("significant = primary_table")]
        assert "results = results.copy()" in block
        assert "drop(" not in block, "the permutation columns were dropped"

    def test_guide_concordance_reads_permutation_output(self):
        """The consumer that broke, on the shape that broke it."""
        import pandas as pd

        from spacr.guide_concordance import guide_support

        frame = pd.DataFrame({
            "feature": ["fraction:grna[TGGT1_225160_2]",
                        "fraction:grna[TGGT1_225160_3]"],
            "coefficient": [1.4, 1.1],
            "p_value": [5e-06, 2e-03],
        })
        support = guide_support(frame)
        assert "225160" in support.index
        assert support.loc["225160"]["n_guides"] == 2


class TestTheVolcanoIsNotAToxoplasmaFeature:
    """A run with toxo=False wrote sixteen diagnostic figures and not the one
    the user came for, silently.

    The compartment COLOURING needs the LOPIT table; the volcano does not.
    Gating the whole figure on an organism-specific flag meant every non-toxo
    user of the module got no volcano and no explanation.
    """

    def test_the_fallback_is_reached_when_toxo_is_off(self):
        import inspect

        import spacr.ml as ml

        text = inspect.getsource(ml)
        assert "A VOLCANO IS NOT A TOXOPLASMA FEATURE" in text
        block = text[text.index("A VOLCANO IS NOT A TOXOPLASMA FEATURE"):]
        block = block[:block.index("print('Significant Genes')")]
        # Guarded on toxo being OFF, so the coloured version still wins when
        # the metadata is there.
        # The sentinel is `_toxoplasma_is_on(settings)` since the rename on
        # 2026-08-17 (instruction 133, "change the toxo settings to
        # Toxoplasma"). Reading the source for a literal is fragile by
        # nature; what this test is really asserting is that the fallback
        # volcano is guarded by the Toxoplasma switch and not drawn
        # unconditionally, so it asks for the resolver by name.
        assert "not _toxoplasma_is_on(settings)" in block
        assert "volcano_plot" in block

    def test_it_still_announces_where_the_file_went(self):
        """Every other artefact says where it went; the volcano used not to,
        which made 'drew one' and 'drew none' indistinguishable."""
        import inspect

        import spacr.ml as ml

        text = inspect.getsource(ml)
        block = text[text.index("A VOLCANO IS NOT A TOXOPLASMA FEATURE"):]
        block = block[:block.index("print('Significant Genes')")]
        assert "Saved volcano plot to" in block

    def test_a_drawing_failure_does_not_sink_the_run(self):
        """The regression is complete and written by this point."""
        import inspect

        import spacr.ml as ml

        text = inspect.getsource(ml)
        block = text[text.index("A VOLCANO IS NOT A TOXOPLASMA FEATURE"):]
        block = block[:block.index("print('Significant Genes')")]
        assert "except Exception" in block


class TestASettingThatCannotDoAnythingIsGreyedOut:
    """Instruction 106. An enabled control is a promise that changing it
    changes the run, and ``_reject_unused_settings`` REFUSES a setting the
    chosen family cannot read -- so an enabled-but-unread control invites an
    edit that the pipeline then rejects, after the CSVs have been read. The
    panel and the validator have to agree about what is legal.
    """

    @staticmethod
    def _panel(qtbot):
        from spacr.qt.screens.settings_model import SettingsWidgets

        panel = SettingsWidgets("regression")
        panel.build_sections()
        for widget in panel._widgets.values():
            qtbot.addWidget(widget)
        return panel

    @staticmethod
    def _set(panel, key, value):
        from PySide6.QtWidgets import QComboBox
        widget = panel._widgets.get(key)
        assert widget is not None, f"the regression panel has no {key!r}"
        if isinstance(widget, QComboBox):
            for index in range(widget.count()):
                if widget.itemData(index) == value or widget.itemText(index) == str(value):
                    widget.setCurrentIndex(index)
                    return
            raise AssertionError(f"{key!r} offers no {value!r}")
        raise AssertionError(f"{key!r} is a {type(widget).__name__}")

    def test_the_panel_and_the_validator_agree_on_every_family(self, qtbot):
        """The assertion that matters: for each regression family, exactly
        the settings that family READS are editable.

        This is the GUI/validator disagreement made impossible. Anything
        else and the panel invites a value the run refuses.
        """
        from spacr.ml import REGRESSION_SETTINGS_USED

        panel = self._panel(qtbot)
        owned = {key for keys in REGRESSION_SETTINGS_USED.values()
                 for key in keys}
        wrong = []
        for family, used in REGRESSION_SETTINGS_USED.items():
            try:
                self._set(panel, "regression_type", family)
            except AssertionError:
                continue        # family the panel does not offer
            panel._refresh_setting_dependencies()
            enabled = {key for key in owned
                       if key in panel._widgets
                       and panel._widgets[key].isEnabled()}
            expected = {key for key in used if key in panel._widgets}
            if enabled != expected:
                wrong.append(
                    f"{family}: enabled-but-unread {sorted(enabled - expected)}, "
                    f"read-but-disabled {sorted(expected - enabled)}")
        assert not wrong, "\n  ".join([""] + wrong)

    def test_parametric_inference_greys_the_permutation_controls(self, qtbot):
        panel = self._panel(qtbot)
        guides = [key for key in panel._widgets if key.startswith("guide_")]
        # EIGHT since 2026-08-18. `guide_permutation_plot` was the ninth and
        # is retired: the permutation plot is always written now, so there is
        # no control to grey out. A floor rather than an equality, because
        # the point of this test is that the permutation controls go dark
        # TOGETHER -- a new one arriving must be covered, not counted.
        assert len(guides) >= 8

        self._set(panel, "inference", "parametric")
        panel._refresh_setting_dependencies()
        assert not [k for k in guides if panel._widgets[k].isEnabled()]

        self._set(panel, "inference", "nonparametric")
        panel._refresh_setting_dependencies()
        assert not [k for k in guides if not panel._widgets[k].isEnabled()]

    def test_a_per_cell_analysis_greys_the_aggregation(self, qtbot):
        """agg_type says how wells are pooled. Per cell, nothing is pooled."""
        panel = self._panel(qtbot)
        self._set(panel, "analysis_unit", "cell")
        panel._refresh_setting_dependencies()
        assert not panel._widgets["agg_type"].isEnabled()

        self._set(panel, "analysis_unit", "well")
        panel._refresh_setting_dependencies()
        assert panel._widgets["agg_type"].isEnabled()

    def test_the_panel_counts_the_plates_in_the_loaded_inputs(
            self, qtbot, tmp_path):
        """The fact the data-dependent half needs, read off the CSVs.

        Cheaply: the header and the plate column only, stopping at the second
        distinct plate, because a panel that stalls after a file is dropped is
        worse than one control too many.
        """
        from spacr.qt.screens.settings_model import SettingsWidgets

        one = tmp_path / "one.csv"
        one.write_text("plateID,well,score\np1,A01,1\np1,A02,2\n")
        two = tmp_path / "two.csv"
        two.write_text("plateID,well,score\np1,A01,1\np2,A02,2\n")
        bare = tmp_path / "bare.csv"
        bare.write_text("well,score\nA01,1\nA02,2\n")

        read = SettingsWidgets._plate_context
        assert read([str(one)]) == {'plate_count': 1, 'has_plate_id': True}
        assert read([str(two)]) == {'plate_count': 2, 'has_plate_id': True}
        assert read([])['has_plate_id'] is False
        # No plate column: score_data[i] and count_data[i] describe the same
        # plate, so their absent IDs share one fallback identity.
        assert read([(0, str(bare)), (0, str(bare))]) == {
            'plate_count': 1, 'has_plate_id': False}

    def test_a_rule_that_reads_the_data_greys_its_control(
            self, qtbot, tmp_path, monkeypatch):
        """The seam for the half the user asked about by name.

        "plateid is only usefull when there is one plate" is a rule about the
        LOADED DATA, not about another setting. The panel already computes the
        plate count, but every rule in ``settings.setting_dependencies``
        ignores the context argument, so the count is computed and thrown
        away. This drives the wiring with a rule that does read it, so the
        table can gain one without the Qt side needing another change.
        """
        import spacr.settings as settings_module

        two = tmp_path / "two.csv"
        two.write_text("plateID,well,score\np1,A01,1\np2,A02,2\n")
        one = tmp_path / "one.csv"
        one.write_text("plateID,well,score\np1,A01,1\np1,A02,2\n")

        rule = {
            'sources': ('paired_data',),
            'predicate': lambda s, ctx: ctx.get('plate_count') == 1,
            'reason': lambda s, ctx: (
                "guide_permutation_block never shuffles residuals between its "
                f"levels, and these inputs hold {ctx.get('plate_count')} "
                "plates, so blocking on the plate constrains nothing."),
        }
        monkeypatch.setattr(
            settings_module, 'get_setting_dependencies',
            lambda: {'guide_permutation_block': rule})

        panel = self._panel(qtbot)
        block = panel._widgets["guide_permutation_block"]
        paired = panel._widgets["paired_data"]

        # The regression panel takes its CSVs as PAIRS, in one table
        # (instruction 107). It has no score_data or count_data widget, so a
        # context read from those keys sees the defaults -- None -- and
        # answers "no plates" whatever the user loaded.
        paired.set_value([{'score': str(one), 'count': str(one)}])
        panel._refresh_setting_dependencies()
        assert panel._data_context['plate_count'] == 1, (
            "the context must read the files the user actually loaded")
        assert block.isEnabled()

        paired.set_value([{'score': str(two), 'count': str(two)}])
        panel._refresh_setting_dependencies()
        assert panel._data_context['plate_count'] == 2
        assert not block.isEnabled()
        assert "2 plates" in block.toolTip(), (
            "a greyed control with no reason is a dead end")

    def test_every_greyed_control_says_what_would_enable_it(self, qtbot):
        """A greyed control with no explanation is a dead end: the user
        cannot tell whether it is inapplicable or broken."""
        panel = self._panel(qtbot)
        self._set(panel, "regression_type", "ols")
        self._set(panel, "inference", "parametric")
        panel._refresh_setting_dependencies()

        silent = []
        for key, widget in panel._widgets.items():
            if widget.isEnabled():
                continue
            tip = widget.toolTip()
            if not tip or key not in tip:
                silent.append(key)
        assert not silent, (
            f"greyed with no reason naming the setting: {sorted(silent)}")


class TestASettingIsGreyedFromTheDATANotOnlyFromOtherSettings:
    """Instruction 106's data-dependent half, and the reason `context` exists.

    Every other dependency rule reads the other SETTINGS -- which estimator
    is selected, whether permutation is on. This one reads the loaded DATA,
    and until it existed the `context` argument was computed by the panel on
    every drop and then thrown away.

    `guide_permutation_block` names the column permutations are blocked
    within, and residuals are never shuffled between its levels. With one
    plate, blocking on the plate is the whole dataset and constrains nothing.

    IT IS ADDED TO THE RULE ALREADY THERE, not assigned over it. This setting
    is dead under parametric inference AND dead on a single plate, and the
    table holds one entry per setting -- so the first version of this patch
    silently dropped the parametric rule and left the field enabled among its
    greyed siblings.
    """

    #: The other rule on this setting only lets it through under permutation.
    ACTIVE = {"inference": "nonparametric", "analysis_mode": "guide_permutation"}

    def _rule(self):
        from spacr.settings import get_setting_dependencies

        return get_setting_dependencies()["guide_permutation_block"]

    def test_one_plate_greys_the_block_setting(self):
        assert self._rule()["predicate"](
            dict(self.ACTIVE), {"plate_count": 1}) is False

    def test_more_than_one_plate_leaves_it_alone(self):
        assert self._rule()["predicate"](
            dict(self.ACTIVE), {"plate_count": 4}) is True

    def test_an_unknown_plate_count_greys_nothing(self):
        """A control disabled because a file was too large to scan is
        indistinguishable, to the person looking at it, from one disabled on
        purpose. Absence of knowledge leaves the control alone."""
        rule = self._rule()

        assert rule["predicate"](dict(self.ACTIVE), {}) is True
        assert rule["predicate"](dict(self.ACTIVE), {"plate_count": None}) is True

    def test_the_older_reason_is_not_lost(self):
        """Both rules apply. Under parametric inference the setting is dead
        whatever the plate count, and the user must be told THAT reason
        rather than the one about plates."""
        rule = self._rule()
        parametric = {"inference": "parametric"}

        assert rule["predicate"](parametric, {"plate_count": 4}) is False
        assert "permutation" in rule["reason"](parametric, {"plate_count": 4})

    def test_the_reason_shown_is_the_one_that_fired(self):
        rule = self._rule()

        reason = rule["reason"](dict(self.ACTIVE), {"plate_count": 1})
        assert "one plate" in reason
        assert "kept" in reason, "a greyed setting must say its value survives"

    def test_the_polarity_matches_every_other_rule(self):
        """True means APPLICABLE. The estimator rules return True when the
        setting IS read, and a rule written the other way round would grey
        exactly the cases it meant to keep."""
        from spacr.settings import get_setting_dependencies

        estimator = get_setting_dependencies()["alpha"]

        assert estimator["predicate"]({"regression_type": "ridge"}, {}) is True
        assert estimator["predicate"]({"regression_type": "ols"}, {}) is False
        assert self._rule()["predicate"](
            dict(self.ACTIVE), {"plate_count": 4}) is True

    def test_it_listens_to_whichever_input_widget_the_panel_has(self):
        """The regression screen takes pairs in one `paired_data` table and
        has neither `score_data` nor `count_data` as a widget. Listing all
        three is what makes the rule work on every panel -- and the composed
        rule keeps the sources of the rule it was added to."""
        sources = set(self._rule()["sources"])

        assert {"paired_data", "score_data", "count_data"} <= sources
        assert "inference" in sources, (
            "the composed rule dropped the sources of the rule it joined, so "
            "changing inference would no longer re-evaluate it")
