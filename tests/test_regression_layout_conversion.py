import numpy as np
import pandas as pd
import pytest

from spacr.ml import _wide_fixed_effect_design, regression_model
from spacr.regression_layout import (
    infer_regression_layout,
    long_to_wide_regression_data,
    normalise_count_table_layout,
    wide_to_long_regression_data,
)
from spacr.settings import get_perform_regression_default_settings


def _long_fractions():
    rows = []
    for well, (g1, g2) in enumerate(
        [(0.1, 0.0), (0.3, 0.1), (0.6, 0.2), (0.8, 0.1)], start=1
    ):
        for guide, fraction, gene in (
            ("g1", g1, "gene1"), ("g2", g2, "gene2")
        ):
            rows.append(
                {
                    "prc": f"p1_r{well}_c1",
                    "plateID": "p1",
                    "rowID": f"r{well}",
                    "columnID": "c1",
                    "grna": guide,
                    "gene": gene,
                    "fraction": fraction,
                    "gene_fraction": fraction,
                    "score": 0.2 + 0.7 * g1,
                    "cell_count": 100 + well,
                }
            )
    return pd.DataFrame(rows)


def test_long_to_wide_and_back_preserves_nonzero_predictors():
    long = _long_fractions()
    wide = long_to_wide_regression_data(
        long,
        predictor_column="grna",
        value_column="fraction",
        metadata_columns=["score", "plateID", "rowID", "columnID"],
    )
    assert len(wide) == long["prc"].nunique() == 4
    assert {"g1", "g2", "score"}.issubset(wide.columns)
    restored = wide_to_long_regression_data(
        wide,
        predictor_columns=["g1", "g2"],
        id_columns=["prc", "score", "plateID", "rowID", "columnID"],
        predictor_name="grna",
        value_name="fraction",
        drop_zero=True,
    )
    expected = long.loc[long["fraction"].ne(0), ["prc", "grna", "fraction"]]
    observed = restored[["prc", "grna", "fraction"]]
    pd.testing.assert_frame_equal(
        observed.sort_values(["prc", "grna"]).reset_index(drop=True),
        expected.sort_values(["prc", "grna"]).reset_index(drop=True),
    )


def test_auto_layout_accepts_long_and_wide_count_tables():
    long = pd.DataFrame(
        {
            "plateID": ["p1", "p1"],
            "rowID": ["r1", "r1"],
            "columnID": ["c1", "c1"],
            "guide": ["g1", "g2"],
            "reads": [7, 3],
        }
    )
    normalized, resolved = normalise_count_table_layout(
        long, layout="auto", guide_column="guide", count_column="reads"
    )
    assert resolved == "long"
    assert {"grna", "count"}.issubset(normalized.columns)

    wide = pd.DataFrame(
        {
            "plateID": ["p1", "p1"],
            "rowID": ["r1", "r2"],
            "columnID": ["c1", "c1"],
            "g1": [7, 0],
            "g2": [3, 9],
        }
    )
    normalized, resolved = normalise_count_table_layout(wide, layout="auto")
    assert resolved == "wide"
    assert len(normalized) == 3
    assert set(normalized["grna"]) == {"g1", "g2"}


def test_auto_layout_accepts_the_legacy_grna_name_header():
    """The downloadable example uses the header process_reads already accepts."""
    legacy = pd.DataFrame(
        {
            "plateID": ["p1", "p1"],
            "rowID": ["r1", "r1"],
            "columnID": ["c1", "c1"],
            "grna_name": ["g1", "g2"],
            "count": [7, 3],
        }
    )

    normalized, resolved = normalise_count_table_layout(legacy, layout="auto")

    assert resolved == "long"
    assert normalized["grna"].tolist() == ["g1", "g2"]
    assert "grna_name" not in normalized.columns


def test_partial_long_signature_is_refused_instead_of_guessed_wide():
    frame = pd.DataFrame({"grna": ["g1"], "p1": [4]})
    try:
        infer_regression_layout(frame)
    except ValueError as error:
        assert "but not 'count'" in str(error)
    else:
        raise AssertionError("partial long table was silently treated as wide")


@pytest.mark.parametrize("count_column", ["column_name", "grna_name"])
@pytest.mark.parametrize("layout", ["auto", "long"])
def test_misconfigured_count_identifier_names_the_setting_and_actual_headers(
        tmp_path, monkeypatch, count_column, layout):
    from spacr.ml import load_regression_input_pairs
    from spacr.regression_failure import describe_failure
    from spacr.tabular import write_table

    monkeypatch.setenv("SPACR_LANGUAGE", "en")
    count_path = tmp_path / "counts.csv"
    score_path = tmp_path / "scores.csv"
    write_table(pd.DataFrame({
        "row_name": ["r1"], "column_name": ["c1"],
        "grna_name": ["g1"], "count": [7],
    }), count_path, canonicalise=False)
    write_table(pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r1"],
        "columnID": ["c1"], "score": [0.5],
    }), score_path)
    counts, _scores, _audit = load_regression_input_pairs([
        {"score": str(score_path), "count": str(count_path)},
    ])
    before = counts.copy(deep=True)
    assert {"rowID", "columnID", "grna_name", "count"} <= set(counts.columns)
    assert "column_name" not in counts.columns

    with pytest.raises(ValueError) as caught:
        normalise_count_table_layout(
            counts, layout=layout, count_column=count_column,
        )

    text = str(caught.value)
    assert f"count_value_column={count_column!r}" in text
    assert "count_grna_column='grna'" in text
    assert "numeric read-count column" in text
    assert "not a guide name or a row, column or plate identifier" in text
    available = text.split("Available columns after reading", 1)[1]
    actual_columns = ["grna" if name == "grna_name" else name
                      for name in counts.columns]
    assert all(repr(name) in available for name in actual_columns)
    assert repr(count_column) not in available
    assert "do not switch" in text and "bypass this check" in text
    assert isinstance(caught.value.__cause__, ValueError)
    pd.testing.assert_frame_equal(counts, before)

    report = describe_failure(caught.value, include_traceback=False)
    advice = report.split("WHAT TO CHANGE", 1)[1]
    assert "count_value_column" in advice and "numeric read counts" in advice
    assert "no recorded remedy" not in advice


def test_wide_fixed_effect_path_has_one_row_per_well_and_fits_ols():
    frame = _long_fractions()
    y, design, metadata = _wide_fixed_effect_design(
        frame,
        "score",
        level="grna",
        model_plate_position=False,
    )
    assert len(y) == len(design) == len(metadata) == 4
    assert list(design.columns) == [
        "Intercept", "fraction:grna[g1]", "fraction:grna[g2]"
    ]
    model = regression_model(design, y, regression_type="ols")
    assert np.isfinite(np.asarray(model.params, dtype=float)).all()


def test_regression_layout_settings_are_explicit_and_backwards_compatible():
    settings = get_perform_regression_default_settings({})
    assert settings["independent_variable_layout"] == "auto"
    assert settings["wide_predictor_columns"] == []
    assert settings["model_data_layout"] == "long"


@pytest.mark.parametrize("layout, expected_rows", [("long", 6), ("wide", 3)])
@pytest.mark.parametrize("level", ["grna", "gene"])
def test_fit_counts_measure_the_cleaned_design_not_input_or_coefficients(
        tmp_path, monkeypatch, layout, expected_rows, level):
    from spacr import ml, plot

    frame = _long_fractions()
    frame.loc[frame["prc"] == "p1_r4_c1", "score"] = np.nan
    monkeypatch.setattr(plot, "plot_histogram", lambda *args, **kwargs: None)
    model, coefficients, kind = ml.regression(
        frame, str(tmp_path / "screen.csv"), dependent_variable="score",
        regression_type="ridge", level=level, model_data_layout=layout,
        model_plate_position=False, qc=False, plot=False, draw_shared_panels=False)

    counts = coefficients.attrs["fit_design"]
    assert len(frame) == 8
    assert counts["n_rows_fitted"] == expected_rows
    assert counts["n_wells"] == 3
    assert counts["n_guides"] == 2 and counts["n_genes"] == 2
    assert counts["n_design_columns"] == model.n_features_in_
    assert counts["layout"] == layout
    assert kind == "ridge"


def test_fit_counts_follow_rows_patsy_drops_after_response_cleaning(tmp_path, monkeypatch):
    from spacr import ml, plot

    frame = _long_fractions()
    frame.loc[frame["prc"] == "p1_r4_c1", "score"] = np.nan
    frame.loc[frame["prc"] == "p1_r3_c1", "rowID"] = None
    monkeypatch.setattr(plot, "plot_histogram", lambda *args, **kwargs: None)
    _model, coefficients, _kind = ml.regression(
        frame, str(tmp_path / "screen.csv"), dependent_variable="score",
        regression_type="ridge", level="grna", model_data_layout="long",
        model_plate_position=True, qc=False, plot=False, draw_shared_panels=False)
    assert coefficients.attrs["fit_design"]["n_rows_fitted"] == 4
    assert coefficients.attrs["fit_design"]["n_wells"] == 2


def test_missing_count_column_reports_only_the_first_twenty_headers(monkeypatch):
    monkeypatch.setenv("SPACR_LANGUAGE", "en")
    headers = ["grna", *[f"public_header_{index:02d}" for index in range(19)],
               "private_extra_header", "another_extra_header"]
    frame = pd.DataFrame({name: ["g1" if name == "grna" else 1]
                          for name in headers})
    before = frame.copy(deep=True)

    with pytest.raises(ValueError) as caught:
        normalise_count_table_layout(
            frame, layout="long", count_column="missing_reads")

    message = str(caught.value)
    assert "Long independent-variable table lacks: missing_reads" in message
    assert "count_grna_column='grna'" in message
    assert "count_value_column='missing_reads'" in message
    preview = message.split("Available columns after reading (up to 20): ", 1)[1]
    preview = preview.split(". Select the actual read-count header", 1)[0]
    assert preview == ", ".join(repr(name) for name in headers[:20]) + ", ..."
    assert "private_extra_header" not in message
    assert "another_extra_header" not in message
    assert isinstance(caught.value.__cause__, ValueError)
    pd.testing.assert_frame_equal(frame, before)


def _full_composition_long_rows():
    return pd.DataFrame({
        "screenID": ["s1", "s1", "s1", "s1"],
        "plateID": ["p1", "p1", "p1", "p2"],
        "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4,
        "prc": ["p1_r1_c1"] * 3 + ["p2_r1_c1"],
        "grna": ["a1", "a2", "b1", "b1"],
        "gene": ["a", "a", "b", "b"],
        "fraction": [0.1, 0.2, 0.7, 1.0],
        "gene_fraction": [0.3, 0.3, 0.7, 1.0],
        "score": [3.0, 3.0, 3.0, 5.0],
        "cell_count": [100, 100, 100, 120],
    }, index=pd.Index([17, 3, 44, 9], name="observation"))


def test_long_gene_design_repeats_complete_well_composition_without_double_counting():
    from spacr.ml import _long_gene_fixed_effect_design, prepare_formula

    frame = _full_composition_long_rows()
    formula = prepare_formula("score", level="gene", model_plate_position=False)
    response, design = _long_gene_fixed_effect_design(frame, formula)
    pd.testing.assert_index_equal(design.index, frame.index)
    pd.testing.assert_frame_equal(response, frame[["score"]])
    np.testing.assert_allclose(
        design[["gene_fraction:gene[a]", "gene_fraction:gene[b]"]],
        [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.0, 1.0]],
    )
    np.testing.assert_array_equal(design["Intercept"], np.ones(len(frame)))
    pd.testing.assert_series_equal(frame["cell_count"],
                                   pd.Series([100, 100, 100, 120],
                                             index=frame.index, name="cell_count"))


def test_long_gene_composition_preserves_patsy_nuisance_and_dropped_row_alignment():
    from patsy import dmatrices
    from spacr.ml import _long_gene_fixed_effect_design

    frame = _full_composition_long_rows()
    frame["batch"] = ["x", "x", None, "y"]
    formula = "score ~ gene_fraction:gene + plateID + batch"
    expected_y, original = dmatrices(formula, data=frame, return_type="dataframe")
    response, design = _long_gene_fixed_effect_design(frame, formula)
    pd.testing.assert_frame_equal(response, expected_y)
    nuisance = [name for name in original if not name.startswith("gene_fraction:")]
    pd.testing.assert_frame_equal(design[nuisance], original[nuisance])
    np.testing.assert_allclose(
        design[["gene_fraction:gene[a]", "gene_fraction:gene[b]"]],
        [[0.3, 0.7], [0.3, 0.7], [0.0, 1.0]],
    )
    assert list(design.index) == [17, 3, 9]


def test_long_gene_composition_rejects_conflicting_repeated_gene_totals():
    from spacr.ml import _long_gene_fixed_effect_design

    frame = _full_composition_long_rows()
    frame.loc[3, "gene_fraction"] = 0.4
    with pytest.raises(ValueError, match="conflicting values"):
        _long_gene_fixed_effect_design(frame, "score ~ gene_fraction:gene")


def test_long_gene_composition_keeps_reused_wells_on_separate_screens():
    from spacr.ml import _long_gene_fixed_effect_design, check_and_clean_data

    frame = _full_composition_long_rows().iloc[:3].copy()
    second = frame.copy()
    second["screenID"] = "s2"
    second["fraction"] = [0.2, 0.3, 0.5]
    combined = pd.concat([frame, second], ignore_index=True)
    cleaned = check_and_clean_data(combined, "score")
    np.testing.assert_allclose(cleaned["gene_fraction"], [0.3, 0.3, 0.7, 0.5, 0.5, 0.5])
    response, design = _long_gene_fixed_effect_design(cleaned, "score ~ gene_fraction:gene")
    assert len(response) == len(design) == len(combined)
    np.testing.assert_allclose(
        design[["gene_fraction:gene[a]", "gene_fraction:gene[b]"]],
        [[0.3, 0.7]] * 3 + [[0.5, 0.5]] * 3,
    )


def test_long_gene_regression_passes_full_composition_to_the_real_estimator(tmp_path):
    from spacr import ml

    frame = _long_fractions()
    model, coefficients, kind = ml.regression(
        frame, str(tmp_path / "screen.csv"), dependent_variable="score",
        regression_type="ols", level="gene", model_data_layout="long",
        model_plate_position=False, qc=False, plot=False, draw_shared_panels=False)
    assert kind == "ols"
    assert coefficients.attrs["fit_design"]["n_rows_fitted"] == len(frame)
    gene_columns = [i for i, name in enumerate(model.model.exog_names)
                    if name.startswith("gene_fraction:gene[")]
    np.testing.assert_allclose(model.model.exog[:, gene_columns],
                               np.repeat([[0.1, 0.0], [0.3, 0.1],
                                          [0.6, 0.2], [0.8, 0.1]], 2, axis=0))


def test_long_gene_composition_accepts_a_prc_only_well_index():
    from spacr.ml import _long_gene_fixed_effect_design

    frame = _full_composition_long_rows().drop(
        columns=["screenID", "plateID", "rowID", "columnID"])
    response, design = _long_gene_fixed_effect_design(frame, "score ~ gene_fraction:gene")
    pd.testing.assert_index_equal(response.index, frame.index)
    np.testing.assert_allclose(
        design[["gene_fraction:gene[a]", "gene_fraction:gene[b]"]],
        [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7], [0.0, 1.0]],
    )


def test_long_gene_likelihood_weights_follow_retained_full_well_identity():
    from spacr.ml import _long_gene_fixed_effect_design

    first = _full_composition_long_rows()
    second = first.iloc[:3].copy()
    second['screenID'] = 's2'
    second['score'] = [np.nan, 2.0, 2.0]
    frame = pd.concat([first, second], ignore_index=True)
    frame.index = pd.Index([7, 7, 5, 9, 2, 2, 8], name='source_row')
    response, design = _long_gene_fixed_effect_design(frame, 'score ~ gene_fraction:gene')
    weights = design.attrs['well_likelihood_weights']
    np.testing.assert_allclose(weights, [1/3, 1/3, 1/3, 1, 1/2, 1/2])
    assert design.attrs["likelihood_well_ids"] == (
        (("s1", "p1", "r1", "c1", "p1_r1_c1"),) * 3
        + (("s1", "p2", "r1", "c1", "p2_r1_c1"),)
        + (("s2", "p1", "r1", "c1", "p1_r1_c1"),) * 2)
    pd.testing.assert_index_equal(response.index, frame.index.take([0, 1, 2, 3, 5, 6]))
    np.testing.assert_array_equal(response.to_numpy().ravel(), [3, 3, 3, 5, 2, 2])
