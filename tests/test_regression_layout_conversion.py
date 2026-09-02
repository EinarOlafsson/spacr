import numpy as np
import pandas as pd

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
