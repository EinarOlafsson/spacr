"""The figures a run writes to disk are the figures the grid shows.

Reported 2026-08-17: "there are a tone of qc graphs that get saved but are
not shown in the program", and "with BH and non parametric there are several
graphs that are saved that are not visualized in the software".

BOTH ARE ONE `os.listdir`. `_load_trial_figures` listed the run folder
FLAT, and almost everything a run writes is in a subfolder: ~19 panels under
`regression_qc/`, the permutation plots under their own, and a summary plot
per measurement under `results/<name>/`. A flat listing saw one file.

Also here: raw vs adjusted p-values on the volcano's y-axis, offered only
when there is a correction to switch to.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt


def _figure(path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [1, 0])
    figure.savefig(path)
    plt.close(figure)


@pytest.fixture
def run_folder(tmp_path):
    """Shaped like a real run: one figure at the top, the rest nested."""
    _figure(tmp_path / "regression_figure.pdf")
    for index in range(19):
        _figure(tmp_path / "regression_qc" / f"panel_{index:02d}.pdf")
    for name in ("cell_count", "wells_per_gene", "gene_per_well"):
        _figure(tmp_path / "results" / name / f"{name}_jitter_bar.pdf")
    _figure(tmp_path / "guide_permutation" / "null_distribution.pdf")
    return tmp_path


def test_every_saved_figure_is_found(qtbot, run_folder):
    """24 files: 1 sheet + 19 QC + 3 summaries + 1 permutation. A flat
    listing found 1."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert screen._load_trial_figures(str(run_folder)) == 24


def _titles_handed_to_the_grid(screen, folder):
    """The captions the loader passes to the grid.

    Captured at the SEAM rather than read back off a cell: the cell keeps no
    title attribute, and two earlier versions of these tests guessed one and
    failed for no reason. What the loader hands over is the observable thing
    and the thing the complaint is about.
    """
    captured = {}
    original = screen._figure_grid.set_figures

    def spy(pixmaps, titles=None, sections=None):
        captured["titles"] = list(titles or [])
        return original(pixmaps, titles, sections)

    screen._figure_grid.set_figures = spy
    try:
        screen._load_trial_figures(folder)
    finally:
        screen._figure_grid.set_figures = original
    return captured.get("titles", [])


def test_the_qc_panels_specifically_are_found(qtbot, run_folder):
    """The complaint names them: "a tone of qc graphs that get saved but are
    not shown"."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    titles = _titles_handed_to_the_grid(screen, str(run_folder))
    joined = " ".join(titles)
    assert "panel_00" in joined, titles[:6]
    assert "regression_qc" in joined, titles[:6]


def test_the_subfolder_is_part_of_the_caption(qtbot, run_folder):
    """"residuals" under regression_qc/ and "residuals" under results/ are
    two different pictures; a grid captioning both the same is one you
    cannot navigate."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    titles = _titles_handed_to_the_grid(screen, str(run_folder))
    assert any(" / " in title for title in titles), titles[:6]


def test_a_flat_folder_still_works(qtbot, tmp_path):
    """The recursion must not break the case it started as."""
    from spacr.qt.screens.app_screen import AppScreen

    for name in ("a.pdf", "b.pdf"):
        _figure(tmp_path / name)
    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert screen._load_trial_figures(str(tmp_path)) == 2


def test_a_folder_with_no_figures_is_zero_not_an_error(qtbot, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    (tmp_path / "results.csv").write_text("a\n")
    screen = AppScreen("regression")
    qtbot.addWidget(screen)

    assert screen._load_trial_figures(str(tmp_path)) == 0


# --------------------------------------------------------------------------- #
#  Raw vs adjusted
# --------------------------------------------------------------------------- #

def _frame(method="fdr_bh", seed=0, n=400):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": rng.normal(0, .5, n),
        "p_value": rng.uniform(size=n),
        "condition": list(rng.choice(["nc", "other"], n, p=[.06, .94]))})
    frame["q_value"] = (frame["p_value"] if method == "none"
                        else rng.uniform(size=n))
    frame["multiple_testing_method"] = method
    return frame


def _panel(qtbot, frame):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(frame)
    return panel


def test_a_corrected_run_offers_both(qtbot):
    panel = _panel(qtbot, _frame("fdr_bh"))

    text = " ".join(a.text() for a in panel.volcano.build_style_menu().actions())
    assert "raw p-value" in text
    assert "adjusted (fdr_bh)" in text


def test_the_correction_is_named_not_just_called_adjusted(qtbot):
    """"adjusted" alone does not say by what, and thirteen methods are
    offered."""
    panel = _panel(qtbot, _frame("bonferroni"))

    text = " ".join(a.text() for a in panel.volcano.build_style_menu().actions())
    assert "bonferroni" in text


def test_an_uncorrected_run_offers_no_toggle(qtbot):
    """`multiple_testing_method='none'` writes a q_value EQUAL to the raw p.
    A menu entry promising "adjusted" there offers a number that is not
    there."""
    panel = _panel(qtbot, _frame("none"))

    text = " ".join(a.text() for a in panel.volcano.build_style_menu().actions())
    assert "adjusted" not in text


def test_and_it_says_why_there_is_no_toggle(qtbot):
    panel = _panel(qtbot, _frame("none"))

    assert "equals the raw p-value" in panel._p_value_note


def test_switching_changes_the_axis(qtbot):
    panel = _panel(qtbot, _frame("fdr_bh"))
    before = [xy[1] for xy in panel.volcano._row_xy.values()]
    panel.set_p_value_kind("adjusted")
    after = [xy[1] for xy in panel.volcano._row_xy.values()]

    assert not np.allclose(sorted(before), sorted(after))


def test_raw_is_the_default(qtbot):
    """The panel must not silently open on a different axis from the one the
    summary and the table describe."""
    panel = _panel(qtbot, _frame("fdr_bh"))

    assert panel._p_value_kind == "raw"
