"""Exporting a figure exports the numbers behind it.

Asked for on 2026-08-16: "the user should be able to determine the intercept
and modify as much as possible regarding the plots so they can right click
and export and use directly in a publication", and "add if a graph is
exported its data is also exported with the filename of the graph and a stats
table is generated with the correct stats".

    volcano.pdf          the figure
    volcano.csv          the rows it actually drew
    volcano_stats.csv    the test, its assumptions, and its result
    volcano_legend.txt   the sentence for the caption

One basename, so "where do these numbers come from" is answered by the folder
rather than by the analyst's memory.

THE DATA IS WHAT WAS DRAWN, not what the panel was handed. A volcano is given
1,213 coefficients and draws 1,212 -- the nuisance terms are not hypotheses --
and a CSV whose row count disagrees with the n printed on the picture is
worse than no CSV at all, because the CSV is what a reader believes.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


def _results(n=300, seed=0):
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, .4, n)
    effect[:10] += 3.0
    p = rng.uniform(size=n)
    p[:10] = 1e-8
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i // 3}_{i % 3}]" for i in range(n)],
        "coefficient": effect,
        "p_value": p,
        "q_value": np.minimum(p * 3, 1.0),
        "grna": [f"{i // 3}_{i % 3}" for i in range(n)],
        "gene": [None] * n,
        "condition": list(rng.choice(["nc", "pc", "other"], n,
                                     p=[.08, .04, .88])),
    })
    intercept = pd.DataFrame([{
        "feature": "Intercept", "coefficient": .19, "p_value": 3e-46,
        "q_value": np.nan, "grna": None, "gene": None, "condition": "other"}])
    return pd.concat([intercept, frame], ignore_index=True)


def _export(key, tmp_path, frame=None):
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import save_figure_as

    figure, panel = build_panel(key, frame if frame is not None else _results())
    target = tmp_path / f"{key}.pdf"
    save_figure_as(None, figure, str(target))
    plt.close(figure)
    return panel, sorted(p.name for p in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
#  Three files, one basename
# --------------------------------------------------------------------------- #

def test_the_data_leaves_with_the_figure(tmp_path):
    _panel, written = _export("volcano", tmp_path)

    assert "volcano.pdf" in written
    assert "volcano.csv" in written


def test_the_csv_holds_what_was_drawn_not_what_was_handed_in(tmp_path):
    """A row count that disagrees with the n on the picture is worse than no
    CSV, because the CSV is what a reader believes."""
    frame = _results()
    _panel, _written = _export("volcano", tmp_path, frame)

    exported = pd.read_csv(tmp_path / "volcano.csv")

    assert len(exported) == len(frame) - 1, (
        "the intercept was exported; it is not a hypothesis and is not on "
        "the plot")
    assert "Intercept" not in set(exported["feature"])


def test_regression_zip_keeps_replay_source_and_separate_drawn_rows(tmp_path):
    """The publication rows differ from the complete recreation input."""
    from spacr.figures import build_panel
    from spacr.figures.bundle import _save_zip

    source = _results()
    figure, panel = build_panel("volcano", source)
    archive_path = _save_zip(figure, str(tmp_path / "volcano.zip"),
                             formats=["png"])
    plt.close(figure)

    with zipfile.ZipFile(archive_path) as archive:
        recorded = pd.read_csv(io.BytesIO(archive.read("data.csv")))
        drawn = pd.read_csv(io.BytesIO(archive.read("drawn_data.csv")))
        spec = json.loads(archive.read("spec.json"))
        archive.extractall(tmp_path / "recreated")
    assert len(recorded) == len(source)
    assert recorded["feature"].tolist() == source["feature"].tolist()
    assert drawn["feature"].tolist() == panel.data["feature"].tolist()
    assert "Intercept" not in set(drawn["feature"])
    assert (spec["kind"], spec["regression"]) == ("regression_panel", "volcano")
    result = subprocess.run(
        [sys.executable, "recreate_figure.py"], cwd=tmp_path / "recreated",
        capture_output=True, text=True, timeout=30,
        env=dict(os.environ, MPLBACKEND="Agg"),
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert (tmp_path / "recreated" / "recreated.png").is_file()


def test_retyping_a_regression_panel_replaces_the_published_rows(tmp_path):
    """A later graph type cannot export the stale volcano selection."""
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import _retype, export_sidecars

    source = _results()
    figure, _panel = build_panel("volcano", source)
    assert _retype(figure, "hist") is True
    assert figure._spacr_drawn_data is None
    export_sidecars(figure, tmp_path / "hist.png")
    histogram_rows = pd.read_csv(tmp_path / "hist.csv")
    assert histogram_rows["feature"].tolist() == source["feature"].tolist()

    assert _retype(figure, "regression_panel") is True
    export_sidecars(figure, tmp_path / "volcano.png")
    volcano_rows = pd.read_csv(tmp_path / "volcano.csv")
    assert "Intercept" not in set(volcano_rows["feature"])
    assert len(volcano_rows) == len(source) - 1
    plt.close(figure)


def test_multi_panel_recreation_does_not_publish_only_its_last_panel():
    """A sheet has several drawn subsets but one complete replay source."""
    from spacr.figures.bundle import _draw
    from spacr.figures.sheet import build_sheet

    source = _results()
    sheet = build_sheet(source, order=("volcano", "qq"))
    figure = sheet.figure
    assert len(sheet.panels) == 2
    _draw(figure, source, figure._spacr_spec)
    assert figure._spacr_drawn_data is None
    assert len(figure.axes) == len(sheet.panels)
    plt.close(figure)


def test_invalid_drawn_row_metadata_falls_back_to_replay_source(tmp_path):
    """An unrelated private attribute cannot silently erase the CSV."""
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import export_sidecars

    source = _results()
    figure, _panel = build_panel("volcano", source)
    figure._spacr_drawn_data = object()
    export_sidecars(figure, tmp_path / "fallback.png")
    exported = pd.read_csv(tmp_path / "fallback.csv")
    assert exported["feature"].tolist() == source["feature"].tolist()
    plt.close(figure)


def test_failed_retype_keeps_the_last_plotted_rows(tmp_path, monkeypatch):
    """A refused redraw must not silently switch its CSV to the input."""
    from spacr.figures import build_panel, bundle
    from spacr.qt.widgets.figure_settings import _retype, export_sidecars

    figure, panel = build_panel("volcano", _results())

    def unavailable(*_args, **_kwargs):
        """Simulate a renderer refusing the requested graph type."""
        figure._spacr_drawn_data = None
        raise ValueError("renderer unavailable")

    monkeypatch.setattr(bundle, "_draw", unavailable)
    assert _retype(figure, "hist") is False
    export_sidecars(figure, tmp_path / "last.png")
    exported = pd.read_csv(tmp_path / "last.csv")
    assert exported["feature"].tolist() == panel.data["feature"].tolist()
    plt.close(figure)


def test_a_comparison_panel_exports_its_statistics(tmp_path):
    _panel, written = _export("controls", tmp_path)

    assert "controls_stats.csv" in written
    table = pd.read_csv(tmp_path / "controls_stats.csv")
    assert len(table) >= 1
    for column in ("test", "groups", "n", "p_value", "effect_size",
                   "why_this_test"):
        assert column in table.columns, column


def test_the_legend_leaves_with_it(tmp_path):
    """A journal figure without its legend is half a figure."""
    _panel, written = _export("volcano", tmp_path)

    assert "volcano_legend.txt" in written
    text = (tmp_path / "volcano_legend.txt").read_text()
    assert "tested coefficients" in text


def test_a_panel_that_is_not_a_comparison_gets_no_stats_file(tmp_path):
    """A Q-Q is not two groups, and inventing a test for it would be worse
    than offering none."""
    _panel, written = _export("qq", tmp_path)

    assert "qq.pdf" in written
    assert not any(name.endswith("_stats.csv") for name in written)


# --------------------------------------------------------------------------- #
#  The statistics are the right ones
# --------------------------------------------------------------------------- #

def test_every_pair_is_tested_and_corrected_across_them(tmp_path):
    """Six pairwise tests at 0.05 is a 26% chance of one false positive, and
    the individual p-values give no hint of it."""
    _panel, _written = _export("controls", tmp_path)
    table = pd.read_csv(tmp_path / "controls_stats.csv")

    assert len(table) >= 3, "not every pair of control classes was compared"
    assert (table["p_adjusted"] >= table["p_value"] - 1e-12).all()
    assert set(table["correction"].dropna()) <= {"fdr_bh"}


def test_the_table_says_why_it_chose_that_test(tmp_path):
    _panel, _written = _export("controls", tmp_path)
    table = pd.read_csv(tmp_path / "controls_stats.csv")

    reasons = " ".join(table["why_this_test"].astype(str))
    assert "normal" in reasons or "variance" in reasons


def test_the_unit_of_replication_is_named(tmp_path):
    """A test across the wrong unit returns p < 1e-10 on noise."""
    _panel, _written = _export("controls", tmp_path)
    table = pd.read_csv(tmp_path / "controls_stats.csv")

    assert set(table["unit"]) == {"coefficient"}


# --------------------------------------------------------------------------- #
#  Nothing here may cost a figure
# --------------------------------------------------------------------------- #

def test_a_figure_with_no_data_attached_still_exports(tmp_path):
    """Any old matplotlib figure must still save."""
    from spacr.qt.widgets.figure_settings import save_figure_as

    figure = plt.figure()
    figure.add_subplot(111).plot([0, 1], [1, 0])
    target = tmp_path / "plain.png"
    try:
        assert save_figure_as(None, figure, str(target)) == str(target)
    finally:
        plt.close(figure)
    assert target.stat().st_size > 0
    assert not (tmp_path / "plain.csv").exists()


def test_an_unwritable_sidecar_does_not_lose_the_figure(tmp_path,
                                                        monkeypatch):
    """The export already did the useful part."""
    from spacr.figures import build_panel
    from spacr.qt.widgets.figure_settings import save_figure_as

    figure, _panel = build_panel("volcano", _results())

    def _explode(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _explode)
    target = tmp_path / "volcano.pdf"
    try:
        assert save_figure_as(None, figure, str(target)) == str(target)
    finally:
        plt.close(figure)
    assert target.stat().st_size > 0


def test_a_group_too_small_to_test_is_skipped_not_faked(tmp_path):
    """A comparison that could not be made is not a comparison with an
    unknown answer."""
    frame = _results()
    frame.loc[frame["condition"] == "pc", "condition"] = "other"
    frame.loc[frame.index[:1], "condition"] = "pc"      # one positive control

    _panel, written = _export("controls", tmp_path, frame)

    if "controls_stats.csv" in written:
        table = pd.read_csv(tmp_path / "controls_stats.csv")
        assert not (table["groups"].str.contains("positive")).any(), (
            "a group of one was tested")
