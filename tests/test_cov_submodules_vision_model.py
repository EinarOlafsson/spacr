"""Corners of the legacy vision-model explainer ``spacr.submodules.interperate_vision_model``.

Two disagreeing key columns warn, and the nested radar-plot and grouping
helpers keep their defaults.  The join and the explainers are tested in
``tests/test_submodules_vision_model.py``.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Imported up front so the slow torch/cellpose import chain behind
# spacr.submodules is paid at collection time, not charged to one test.
import spacr.io  # noqa: E402
import spacr.submodules  # noqa: E402


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    plt.close("all")
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# synthetic data shaped like io._read_and_merge_data output
# ---------------------------------------------------------------------------

def _grid(n):
    """``n`` unique (plate, row, column, field) metadata tuples."""
    return [("plate1", f"r{i % 3 + 1}", f"c{i % 4 + 1}", f"f{i}") for i in range(n)]


def _metadata(g):
    """The five join columns; object_label carries the 'oN' prefix form."""
    return {
        "object_label": [f"o{i}" for i in range(1, len(g) + 1)],
        "plateID": [x[0] for x in g],
        "rowID": [x[1] for x in g],
        "column_name": [x[2] for x in g],
        "fieldID": [x[3] for x in g],
    }


def _measurement_frame(n, seed=0, zero_pathogen_row=None):
    """Full four-compartment frame: every compartment owns an ``_area`` column
    (so the ratio expansion has partners), two channel intensities, one
    two-channel colocalization on a ``cells_`` prefix, and one feature that
    belongs to no compartment at all."""
    rng = np.random.default_rng(seed)
    g = _grid(n)
    signal = rng.normal(size=n)
    cols = _metadata(g)
    cols.update({
        "cell_area": rng.uniform(1000.0, 4000.0, n),
        "nucleus_area": rng.uniform(100.0, 900.0, n),
        "pathogen_area": rng.uniform(10.0, 90.0, n),
        "cytoplasm_area": rng.uniform(500.0, 3000.0, n),
        "cell_channel_0_mean_intensity": 1500.0 + 400.0 * signal,
        "nucleus_channel_1_mean_intensity": rng.uniform(200.0, 900.0, n),
        "cells_channel_2_channel_3_colocalization": rng.uniform(0.0, 1.0, n),
        "field_focus_score": rng.uniform(0.0, 1.0, n),
    })
    df = pd.DataFrame(cols)
    if zero_pathogen_row is not None:
        df.loc[zero_pathogen_row, "pathogen_area"] = 0.0
    return df, g, (signal > 0).astype(int)


def _write_scores(path, g, labels, extra=None, score_column="score"):
    """Write the per-object prediction CSV that gets merged onto the frame."""
    data = {
        "plateID": [x[0] for x in g],
        "fieldID": [x[3] for x in g],
        "object": np.arange(1, len(g) + 1),
        score_column: labels,
    }
    data.update(extra if extra is not None else {
        "rowID": [x[1] for x in g],
        "column_name": [x[2] for x in g],
        "object_label": np.arange(1, len(g) + 1),
    })
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _nested(name):
    """Rebuild a closure-free nested helper of ``interperate_vision_model``.

    ``create_extended_radar_plot`` has both of its call sites commented out
    (submodules.py lines 1616-1617) and the two ``... is None`` default
    branches are unreachable because the call sites always pass the argument;
    rebuilding the real code object is the only way to execute that product
    code. The helpers close over nothing, so module globals are enough.
    """
    from spacr.submodules import interperate_vision_model

    code = next(c for c in interperate_vision_model.__code__.co_consts
                if isinstance(c, types.CodeType) and c.co_name == name)
    assert code.co_freevars == (), f"{name} unexpectedly closes over state"
    return types.FunctionType(code, spacr.submodules.__dict__, name)


# ===========================================================================
# read_and_preprocess_data — merge, aliases, ratio expansion
# ===========================================================================

def test_two_disagreeing_key_columns_warn_rather_than_one_winning(
        tmp_path, monkeypatch):
    """The half the old rule hid (145).

    A scores CSV carrying BOTH `row` and `row_name`, disagreeing, has two
    opinions about which row an object came from. The old behaviour picked
    one by declaration order and said nothing. spaCR now says which it kept
    AND HOW MANY ROWS DIFFER -- because "they disagree" is not actionable,
    "3 of 40,000 differ" is a typo, and "24 of 24" is the wrong file.
    """
    import warnings

    from spacr.tabular import read_table

    df, g, _sig = _measurement_frame(24, seed=2)
    labels = np.arange(len(g)) % 2
    scores = _write_scores(tmp_path / "scores_c.csv", g, labels, extra={
        "row": ["WRONG"] * len(g),
        "row_name": [x[1] for x in g],
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        frame = read_table(scores, report=None)

    said = " ".join(str(w.message) for w in caught)
    assert "rowID" in said
    assert "row_name" in said
    assert f"{len(g)} of {len(g)} rows differ" in said
    assert "rowID" in frame.columns


# ===========================================================================
# nested helpers that the call sites can never reach
# ===========================================================================

def test_radar_plot_helper_draws_a_closed_polar_loop():
    """``create_extended_radar_plot`` builds a filled polar axes whose ticks
    are the labels it was handed. Both call sites are commented out, so the
    helper is rebuilt from its code object."""
    radar = _nested("create_extended_radar_plot")

    values = [0.4, 0.3, 0.2]
    radar(values, ["cell", "nucleus", "pathogen"], "SHAP by compartment")

    axes = [ax for num in plt.get_fignums() for ax in plt.figure(num).axes]
    assert len(axes) == 1
    ax = axes[0]
    assert ax.name == "polar"
    assert ax.get_title() == "SHAP by compartment"
    assert [t.get_text() for t in ax.get_xticklabels()] == \
        ["cell", "nucleus", "pathogen"]
    # the loop is closed: 4 points for 3 labels, first == last
    (line,) = ax.lines
    xs, ys = line.get_xydata().T
    assert len(ys) == 4
    assert ys[0] == pytest.approx(ys[-1] if len(set(values)) > 1 else values[0])
    assert list(ys[:3]) == pytest.approx(values)
    assert xs[0] == pytest.approx(0.0)
    assert len(ax.patches) == 1, "the area under the curve is filled"


def test_comparison_columns_default_to_the_four_spacr_compartments():
    """``generate_comparison_columns`` defaults to cell/nucleus/pathogen/
    cytoplasm when no compartment list is given (the call site always passes
    one, so the default is rebuilt from the code object)."""
    gen = _nested("generate_comparison_columns")

    df = pd.DataFrame({"cell_area": [10.0, 20.0],
                       "nucleus_area": [5.0, 5.0],
                       "pathogen_area": [2.0, 0.0]})

    out, comparisons = gen(df, None)

    assert comparisons == {
        "cell_area": ["nucleus_area", "pathogen_area"],
        "nucleus_area": ["cell_area", "pathogen_area"],
        "pathogen_area": ["cell_area", "nucleus_area"],
    }
    assert out["nucleus_cell_area"].tolist() == [0.5, 0.25]
    assert out["cell_pathogen_area"].tolist() == [5.0, 0.0]  # 20/0 -> inf -> 0
    assert out["pathogen_nucleus_area"].tolist() == [0.4, 0.0]
    assert np.isfinite(out.to_numpy()).all()


def test_group_feature_class_defaults_to_the_four_compartments():
    """``group_feature_class`` falls back to the standard compartment list and
    drops features that match none of them."""
    group = _nested("group_feature_class")

    imp = pd.DataFrame({
        "feature": ["cell_area", "nucleus_cell_area", "pathogen_x", "zzz"],
        "importance": [0.4, 0.3, 0.2, 0.1],
    })

    summed = group(imp.copy(), None, "compartment", False)

    assert summed.set_index("compartment")["compartment_importance_sum"].to_dict() \
        == pytest.approx({"cell": 0.4, "cell-nucleus": 0.3, "pathogen": 0.2})
    assert "zzz" not in summed["compartment"].tolist()
    assert summed["compartment_importance_sum"].sum() == pytest.approx(0.9)
