"""Instruction 156, driven through the REAL Summary tab.

GREEN TESTS DO NOT MEAN THE FEATURE WORKS. The model-layer contract is pinned
in ``tests/test_every_regression_mode_gets_a_summary.py``; this file presses
the thing the maintainer presses. It loads a run FOLDER through
``RegressionResultsPanel.load`` -- what "Load run..." does -- selects the
Summary tab and reads the text that is actually on the widget.

The reported failure was a nonparametric mixed run whose Summary tab said "No
summary: this run came back without a fitted model, so there is none to
summarise". That is the assertion below: not that a helper returns a string,
but that the tab does not show that sentence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.regression_summary import write_run_summary


GUIDES = [f"{gene}_{i}" for gene in ("000000", "220950", "gene3", "gene4")
          for i in (1, 2, 3)]


def _fitted_table(n_wells=24, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        prc = f"plate{w % 2 + 1}_r{w // 4 + 1:02d}_c{w % 4 + 1:02d}"
        cells = int(rng.integers(100, 400))
        response = float(rng.normal(0.5, 0.2))
        raw = rng.random(3) + 0.2
        for guide, one in zip(rng.choice(GUIDES, size=3, replace=False),
                              raw / raw.sum()):
            rows.append({
                "plateID": prc.split("_")[0], "rowID": prc.split("_")[1],
                "columnID": prc.split("_")[2], "prc": prc,
                "grna": guide, "gene": guide.rsplit("_", 1)[0],
                "fraction": float(one), "cell_count": cells,
                "pred": response,
            })
    return pd.DataFrame(rows)


def _permutation_results(permutations=1000, seed=2):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(0, 1, len(GUIDES)), 1.0 / (permutations + 1), 1.0)
    frame = pd.DataFrame({
        "outcome": "pred", "guide": GUIDES, "grna": GUIDES,
        "feature": [f"fraction:grna[{g}]" for g in GUIDES],
        "wells_with_guide": 6,
        "standardized_marginal_effect": rng.normal(0, 1, len(GUIDES)),
        "permutations": permutations, "permutation_p_value": p,
        "block_column": "plateID", "nuisance_columns": "",
        "minimum_wells_threshold": 1, "multiple_testing_method": "fdr_bh",
    })
    frame["coefficient"] = frame["standardized_marginal_effect"]
    frame["p_value"] = frame["permutation_p_value"]
    frame["adjusted_p_value"] = np.clip(frame["p_value"] * 2, 0, 1)
    frame["q_value"] = frame["adjusted_p_value"]
    frame["significant"] = frame["q_value"] < 0.05
    frame["condition"] = ["nc"] * 3 + ["pc"] * 3 + ["other"] * 6
    frame["effect_size_threshold"] = 1.2
    frame["effect_size_rule"] = "3x std of 3 controls = 1.2"
    frame["passes_effect_size"] = frame["coefficient"].abs() >= 1.2
    return frame


def _settings():
    return {
        "regression_type": "mixed", "inference": "nonparametric",
        "analysis_mode": "guide_permutation", "dependent_variable": "pred",
        "analysis_unit": "well", "agg_type": "mean", "transform": None,
        "multiple_testing_method": "fdr_bh", "fdr_alpha": 0.05,
        "fraction_threshold": 0.01, "min_cell_count": 25, "level": "both",
        "guide_permutations": 1000, "guide_permutation_block": "plateID",
        "positive_control": "220950", "negative_control": "000000",
        "controls": ["000000_1", "000000_2", "000000_3"],
    }


@pytest.fixture
def run_folder(tmp_path):
    """A nonparametric run on disk, complete with its summary."""
    folder = tmp_path / "guide_permutation"
    folder.mkdir()
    _fitted_table().to_csv(folder / "regression_data.csv", index=False)
    _permutation_results().to_csv(folder / "results.csv", index=False)
    write_run_summary(str(folder), model=None, settings=_settings(),
                      coef_df=_permutation_results(),
                      regression_type="mixed")
    return folder


def _summary_tab_text(panel, qtbot):
    index = [i for i in range(panel.tabs.count())
             if panel.tabs.tabText(i) == "Summary"]
    assert index, "the panel has no Summary tab"
    panel.tabs.setCurrentIndex(index[0])
    qtbot.wait(1)
    return panel.tabs.widget(index[0]).toPlainText()


def test_a_nonparametric_run_loaded_from_disk_shows_a_full_summary(
        qtbot, run_folder):
    """The reported failure, pressed through the widget."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(run_folder))

    text = _summary_tab_text(panel, qtbot)
    assert "No summary" not in text
    assert "came back without a fitted model" not in text
    assert "spaCR RUN SUMMARY" in text


def test_the_tab_carries_every_section_of_the_contract(qtbot, run_folder):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel
    from spacr.regression_summary import SECTIONS

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(run_folder))
    text = _summary_tab_text(panel, qtbot)
    for _name, title in SECTIONS:
        assert title in text, f"the Summary tab is missing {title!r}"


def test_the_tab_says_r2_does_not_exist_and_lists_the_assumptions(qtbot,
                                                                  run_folder):
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(run_folder))
    text = _summary_tab_text(panel, qtbot)
    assert "R2 DOES NOT EXIST" in text
    # Five assumptions, each NOT ASSUMED rather than blank -- an empty block
    # would make the safer method look like the less informative one.
    assert text.count("NOT ASSUMED") == 5
    assert "1/(1,000+1)" in text


def test_the_tab_says_the_text_was_read_from_the_run(qtbot, run_folder):
    """Instruction 153's line survives: a reader is told it is the run's own."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(run_folder))
    text = _summary_tab_text(panel, qtbot)
    assert text.splitlines()[0].startswith("Read from ")
    assert "model_summary.txt" in text.splitlines()[0]


def test_a_run_folder_with_no_summary_still_says_a_true_thing(qtbot, tmp_path):
    """The absence must still be explained, not blank -- 153's rule, kept."""
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    folder = tmp_path / "ols_9"
    folder.mkdir()
    _permutation_results().to_csv(folder / "results.csv", index=False)
    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.load(str(folder))
    text = _summary_tab_text(panel, qtbot)
    assert text.startswith("No summary")
    assert "model_summary.txt" in text
