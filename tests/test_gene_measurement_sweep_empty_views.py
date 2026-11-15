"""Guide-to-gene naming, the sweep, and the pictures that refuse to draw nothing.

Units of :mod:`spacr.gene_measurement_sweep`, each walked over a case table:

* ``gene_of_guide`` reads the gene out of every guide spelling in use.
* ``sweep`` -- too few wells give an empty table, a real screen gives exact
  survivors, and every "nothing to draw" exit of the views built on it returns
  ``None`` instead of an empty axis that reads as a measured absence.
* ``plot_guide_concordance`` draws one row per gene with two or more agreeing
  or disagreeing guides, and returns ``None`` when there is nothing to compare.
"""
from __future__ import annotations

import types

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr.gene_measurement_sweep import (SweepResult, gene_of_guide,  # noqa: E402
                                          plot_circularity, plot_gene_profile,
                                          plot_guide_concordance,
                                          plot_measurement_hits, plot_sweep,
                                          sweep)


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture()
def screen():
    """80 wells on two plates; gene 111's two guides drive two measurements."""
    rng = np.random.default_rng(3)
    n = 80
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    driver = np.zeros(n)
    driver[rng.choice(n, 30, replace=False)] = rng.uniform(0.3, 0.6, 30)
    quiet = rng.uniform(0.1, 0.2, n)
    wells = pd.DataFrame({
        "cell_area": driver * 8.0 + rng.normal(0, 0.2, n),
        "nucleus_eccentricity": driver * 6.0 + rng.normal(0, 0.3, n),
        "pathogen_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({
        "TGGT1_111_1": driver,
        "TGGT1_111_2": driver * 0.9 + 0.01,
        "TGGT1_222_1": quiet,
    }, index=index)
    plates = [i.split("_")[0] for i in index]
    return wells, fractions, plates


def test_gene_of_guide(subtests):
    """Every guide spelling resolves to its gene, or to None when unnamed."""
    cases = [
        ("empty", "", None, None),
        ("None", None, None, None),
        ("blank", "   ", None, None),
        ("Toxoplasma", "TGGT1_233460_1", None, "233460"),
        ("Plasmodium, not listed anywhere", "PF3D7_0100100_1", None, "0100100"),
        ("human gene with an underscore", "HGNC_RAB11_A_1", None, "RAB11_A"),
        ("known prefix stripped only once", "TGGT1_ROP18_kinase_1", None,
         "ROP18_kinase"),
        ("known prefix, no guide suffix", "TGGT1_225160", None, "225160"),
        ("measured prefix", "HGNC_RAB11_A_1", "HGNC", "RAB11_A"),
        ("measured known prefix", "TGGT1_225160_2", "TGGT1", "225160"),
        ("measured prefix, one part left", "XYZ_A", "XYZ", "A"),
        ("two parts", "ABC_1", None, "ABC"),
        ("one part", "solo", None, "solo"),
        ("bracketed design term", "fraction:grna[TGGT1_225160_1]", None,
         "TGGT1_225160"),
    ]
    for case_id, guide, prefix, expected in cases:
        with subtests.test(msg=case_id):
            assert gene_of_guide(guide, prefix=prefix) == expected


def test_sweep_and_its_empty_views(subtests, screen):
    """Exact sweep answers; every view asked for nothing returns None."""
    with subtests.test(msg="two wells are not a screen"):
        result = sweep(pd.DataFrame({"cell_area": [1.0, 2.0]}, index=["w1", "w2"]),
                       pd.DataFrame({"TGGT1_111_1": [0.1, 0.9]}, index=["w1", "w2"]))
        assert (result.n_wells, result.n_blocks) == (2, 0)
        assert list(result.table.columns) == [
            "guide", "measurement", "effect", "p", "q", "circularity", "n_wells"]
        assert len(result.table) == 0

    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates)
    with subtests.test(msg="a real screen"):
        assert (result.n_wells, result.n_blocks) == (80, 2)
        assert result.circularity_known is False
        assert len(result.table) == 9
        assert result.table["circularity"].isna().all()
        survivors = result.survivors(alpha=0.05)
        assert sorted(zip(survivors["guide"], survivors["measurement"])) == [
            ("TGGT1_111_1", "cell_area"), ("TGGT1_111_1", "nucleus_eccentricity"),
            ("TGGT1_111_2", "cell_area"), ("TGGT1_111_2", "nucleus_eccentricity")]
        assert dict(zip(result.table["guide"], result.table["n_wells"])) == {
            "TGGT1_111_1": 30, "TGGT1_111_2": 80, "TGGT1_222_1": 80}

    empties = [
        ("heatmap of the top zero", lambda: plot_sweep(result, top=0)),
        ("gene profile of zero measurements",
         lambda: plot_gene_profile(result, "TGGT1_111_1", top=0)),
        ("measurement hit count of zero", lambda: plot_measurement_hits(result, top=0)),
    ]
    for case_id, draw in empties:
        with subtests.test(msg=case_id):
            assert draw() is None
    with subtests.test(msg="the same heatmap with rows is drawn"):
        assert isinstance(plot_sweep(result), matplotlib.figure.Figure)

    with subtests.test(msg="circularity with no survivor is not drawn"):
        scores = np.random.default_rng(5).uniform(0, 1, len(wells.index))
        scored = sweep(wells, fractions, blocks=plates, scores=scores)
        assert scored.circularity_known is True
        assert scored.table["circularity"].notna().all()
        assert plot_circularity(scored, alpha=0.0) is None


def _table(**columns) -> pd.DataFrame:
    """A guide-level sweep table, defaults filled for every column a view reads."""
    n = len(next(iter(columns.values())))
    base = {"level": ["guide"] * n, "guide": [f"g{i}" for i in range(n)],
            "measurement": ["cell_area"] * n, "effect": [0.5] * n,
            "p": [0.001] * n, "q": [0.001] * n}
    base.update(columns)
    return pd.DataFrame(base)


def _result(table):
    return SweepResult(table=table, effects=pd.DataFrame(), n_wells=40, n_blocks=2)


def _concordance_rows():
    rows = []
    for gene, signs in (("AGREE", [1.0, 1.0]), ("SPLIT", [1.0, -1.0]),
                        ("ALSO", [-1.0, -1.0])):
        for index, sign in enumerate(signs):
            for measurement in ("area", "perimeter"):
                rows.append({"gene": gene, "guide": f"{gene}_{index}",
                             "measurement": measurement,
                             "effect": sign * (index + 1), "q": 0.001})
    return pd.DataFrame(rows)


def test_plot_guide_concordance(subtests, tmp_path):
    """One labelled row per multi-guide gene at its agreement; else None."""
    two_guides = ["TGGT1_111_1", "TGGT1_111_2"]
    nothing = [
        ("only gene-level rows", _table(level=["gene", "gene"], guide=two_guides)),
        ("no guide belongs to a gene", _table(guide=["", "   "])),
        ("one guide per gene", _table(guide=["TGGT1_111_1", "TGGT1_222_1"])),
        ("nothing past the correction", _table(guide=two_guides, q=[0.4, 0.6])),
        ("one signed effect per pair", _table(
            guide=two_guides, measurement=["cell_area", "nucleus_area"])),
        ("zero effects carry no sign", _table(guide=two_guides, effect=[0.0, 0.0])),
    ]
    for case_id, table in nothing:
        with subtests.test(msg=case_id):
            assert plot_guide_concordance(_result(table)) is None

    def _drawn(figure):
        axes = figure.axes[0]
        labels = [t.get_text() for t in axes.get_yticklabels()]
        means = {}
        for line in axes.lines:
            y = line.get_ydata()
            if len(y) == 2 and y[1] - y[0] == pytest.approx(0.56):
                gene = labels[int(round((y[0] + y[1]) / 2))].split("  ")[0]
                means[gene] = float(line.get_xdata()[0])
        return axes, labels, means

    with subtests.test(msg="three genes, every one drawn at its agreement"):
        table = _concordance_rows()
        figure = plot_guide_concordance(types.SimpleNamespace(table=table),
                                        alpha=0.05)
        axes, labels, means = _drawn(figure)
        assert sorted(labels) == ["AGREE  (2)", "ALSO  (2)", "SPLIT  (2)"]
        assert labels[-1] == "SPLIT  (2)"
        assert means == {"AGREE": 1.0, "ALSO": 1.0, "SPLIT": 0.0}
        assert axes.get_title() == "do a gene's own guides agree?"
        assert axes.get_xlim() == pytest.approx((-0.03, 1.05))
        points = [c for c in axes.collections if len(c.get_offsets())]
        assert sorted(len(c.get_offsets()) for c in points) == [2, 2, 2]
    with subtests.test(msg="top and title"):
        figure = plot_guide_concordance(types.SimpleNamespace(table=table),
                                        top=1, title="mine")
        axes, labels, _means = _drawn(figure)
        assert len(labels) == 1 and labels[0] != "SPLIT  (2)"
        assert axes.get_title() == "mine"
    with subtests.test(msg="a figure is written to the path given"):
        path = tmp_path / "concordance.png"
        plot_guide_concordance(types.SimpleNamespace(table=table), path=str(path))
        assert path.stat().st_size > 0
