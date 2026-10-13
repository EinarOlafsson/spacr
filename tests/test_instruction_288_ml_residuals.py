"""Small, direct checks for instruction 288's final :mod:`spacr.ml` arcs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import ml


def _fractions() -> pd.DataFrame:
    rows = []
    for well in range(1, 9):
        plate = "p1" if well <= 4 else "p2"
        screen = "s1" if well % 2 else "s2"
        for guide, gene, fraction in (
            ("g1", "gene1", well / 20),
            ("g2", "gene2", (9 - well) / 20),
        ):
            rows.append(
                {
                    "prc": f"{plate}_r{well}_c1",
                    "plateID": plate,
                    "rowID": f"r{well}",
                    "columnID": "c1",
                    "screenID": screen,
                    "grna": guide,
                    "gene": gene,
                    "fraction": fraction,
                    "gene_fraction": fraction,
                    "predictions": 0.1 + 0.8 * well / 8,
                    "cell_count": 100 + well,
                }
            )
    return pd.DataFrame(rows)


def test_the_slow_backend_message_names_an_unavailable_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        ml, "backend_status",
        lambda backend, family: {"enabled": False, "reason": "no CUDA device"},
    )

    ml._say_what_a_mixed_fit_will_cost("statsmodels", [1, 2])

    assert "not usable here: no CUDA device" in capsys.readouterr().out


def test_wide_gene_design_carries_metadata_layout_and_zero_intercept():
    y, design, metadata = ml._wide_fixed_effect_design(
        _fractions(), "predictions", level="gene",
        model_plate_position=True, block_screen=True, intercept="zero",
    )

    assert len(y) == len(design) == len(metadata) == 8
    assert "Intercept" not in design
    assert {"gene_fraction:gene[gene1]", "gene_fraction:gene[gene2]"} <= set(design)
    assert any(name.startswith("plateID[") for name in design)
    assert any(name.startswith("rowID[") for name in design)
    assert any(name.startswith("screenID[") for name in design)
    assert "cell_count" in metadata


def test_wide_design_rejects_an_unknown_level_and_intercept():
    frame = _fractions()
    with pytest.raises(ValueError, match="wide model design needs one level"):
        ml._wide_fixed_effect_design(frame, "predictions", level="both")
    with pytest.raises(ValueError, match="intercept='mystery'"):
        ml._wide_fixed_effect_design(
            frame, "predictions", level="grna",
            model_plate_position=False, intercept="mystery",
        )


def test_wide_design_refuses_a_pivot_that_lost_a_predictor(monkeypatch):
    import spacr.regression_layout as layout

    frame = _fractions()
    real = layout.long_to_wide_regression_data

    def lose_one(*args, **kwargs):
        return real(*args, **kwargs).drop(columns=["g2"])

    monkeypatch.setattr(layout, "long_to_wide_regression_data", lose_one)
    with pytest.raises(AssertionError, match="lost predictor columns: g2"):
        ml._wide_fixed_effect_design(
            frame, "predictions", level="grna",
            model_plate_position=False,
        )


def test_wide_design_requires_each_requested_layout_column():
    frame = _fractions().drop(columns=["plateID"])
    with pytest.raises(ValueError, match="needs nuisance column 'plateID'"):
        ml._wide_fixed_effect_design(
            frame, "predictions", level="grna", model_plate_position=True,
        )


def test_wide_design_namespaces_predictors_away_from_layout_terms():
    frame = _fractions().drop(columns=["cell_count"])
    frame.loc[frame["grna"].eq("g2"), "grna"] = "plateID[T.p2]"

    _y, design, _metadata = ml._wide_fixed_effect_design(
        frame, "predictions", level="grna", model_plate_position=True,
    )

    assert not design.columns.duplicated().any()
    assert "fraction:grna[plateID[T.p2]]" in design
    assert "plateID[T.p2]" in design


def test_regression_rejects_an_unknown_model_layout_before_fitting():
    with pytest.raises(ValueError, match="choose 'long' or 'wide'"):
        ml.regression(
            _fractions(), "counts.csv", regression_type="ols",
            dependent_variable="predictions", model_data_layout="diagonal",
            dst=None, qc=False, plot=False,
        )


def test_fixed_regression_reaches_the_wide_design(capsys):
    model, coefficients, kind = ml.regression(
        _fractions(), "counts.csv", regression_type="ols",
        dependent_variable="predictions", model_data_layout="wide",
        model_plate_position=False, controls=[], dst=None, qc=False, plot=False,
        draw_shared_panels=False,
    )

    assert kind == "ols" and model is not None and not coefficients.empty
    assert "Model data pivoted to 8 independent well rows" in capsys.readouterr().out


def test_mixed_regression_says_wide_was_normalised_back_to_long(
        monkeypatch, capsys):
    coefficients = pd.DataFrame(
        {"feature": ["gene_fraction:gene[gene1]"],
         "coefficient": [0.5], "p_value": [0.01], "term_type": [ml.TERM_FIXED]}
    )
    monkeypatch.setattr(
        ml, "fit_mixed_model",
        lambda *args, **kwargs: (object(), coefficients.copy()),
    )

    _model, observed, kind = ml.regression(
        _fractions(), "counts.csv", regression_type="mixed",
        dependent_variable="predictions", model_data_layout="wide",
        controls=[], dst=None, qc=False, plot=False,
    )

    assert kind == "mixed" and observed["level"].eq("gene").all()
    assert "normalized back to long" in capsys.readouterr().out


def _diagnostic_long() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prc": ["p1_r1_c1", "p1_r1_c1", "p1_r2_c1", "p1_r2_c1"],
            "grna": ["g1", "g2", "g1", "g2"],
            "fraction": [0.7, 0.3, 0.2, 0.8],
            "plateID": ["p1", "p1", "p1", "p1"],
        }
    )


def test_diagnostic_design_keeps_non_frames_and_validates_long_values():
    marker = np.array([[1.0]])
    returned, block = ml._diagnostic_screen_design(marker, {})
    assert returned is marker and block is None

    missing = _diagnostic_long()
    missing.loc[0, "grna"] = None
    with pytest.raises(ValueError, match="identifiers must not contain missing"):
        ml._diagnostic_screen_design(missing, {})

    infinite = _diagnostic_long()
    infinite.loc[0, "fraction"] = np.inf
    with pytest.raises(ValueError, match="fractions must be finite"):
        ml._diagnostic_screen_design(infinite, {})

    negative = _diagnostic_long()
    negative.loc[0, "fraction"] = -0.1
    with pytest.raises(ValueError, match="fractions must be non-negative"):
        ml._diagnostic_screen_design(negative, {})


def test_diagnostic_design_validates_and_omits_block_labels():
    conflicting = _diagnostic_long()
    conflicting.loc[1, "plateID"] = "p2"
    with pytest.raises(ValueError, match="not constant within well"):
        ml._diagnostic_screen_design(conflicting, {})

    missing = _diagnostic_long()
    missing.loc[missing["prc"].eq("p1_r2_c1"), "plateID"] = np.nan
    with pytest.raises(ValueError, match="block labels are missing"):
        ml._diagnostic_screen_design(missing, {})

    without_block = _diagnostic_long().drop(columns=["plateID"])
    wide, block = ml._diagnostic_screen_design(without_block, {})
    assert wide.shape == (2, 2) and block is None


def test_process_reads_refuses_to_exclude_the_entire_table():
    counts = pd.DataFrame(
        {"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
         "grna": ["g1"], "count": [10]}
    )
    with pytest.raises(ValueError, match="removed all 1 raw count rows"):
        ml.process_reads(
            counts, fraction_threshold=None, plate=None, exclude_grnas=["g1"],
        )


def test_process_reads_records_an_exclusion_that_matches_nothing(capsys):
    counts = pd.DataFrame(
        {"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
         "grna": ["g1"], "count": [10]}
    )
    record = {}
    result = ml.process_reads(
        counts, fraction_threshold=None, plate=None,
        exclude_grnas=["not-there"], record=record,
    )
    assert len(result) == 1 and record["exclude_grnas"] == 0
    assert "match no guide or gene" in capsys.readouterr().out


def test_perform_regression_threads_exclusions_to_the_raw_count_reader(
        tmp_path, monkeypatch):
    from tests.test_cov_ml_perform_regression import (
        base_settings, write_counts, write_metadata, write_scores,
    )

    score_dir = tmp_path / "scores"
    count_dir = tmp_path / "counts"
    score_dir.mkdir()
    count_dir.mkdir()
    screen = {
        "root": tmp_path,
        "score": write_scores(score_dir / "xgb_scores.csv"),
        "count": write_counts(count_dir / "counts.csv"),
        "meta": write_metadata(tmp_path / "TGME49_Summary.csv"),
    }
    requested = ["TGGT1_111111_3"]
    seen = []
    real = ml.process_reads

    def recording_reader(*args, **kwargs):
        seen.append(kwargs.get("exclude_grnas"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ml, "process_reads", recording_reader)
    output = ml.perform_regression(
        base_settings(screen, exclude_grnas=requested, toxo=False)
    )

    assert seen == [requested]
    assert output["res_folder"]
