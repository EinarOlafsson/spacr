"""The screen's own figures, drawn and then inspected: ``spacr.toxo``.

The six figures this module builds are the ones the maintainer looks at -- the
volcano, the GT1 phenotype curve, the ME49 transcription heatmap and the three
GO-enrichment panels -- so they are where a restyle is most visible and where
breaking it costs most.

THESE TESTS DRAW. Every assertion is about an artist that ended up on an axis,
because that is the only thing a reader sees; asserting that the module
imports ``figures.style`` would pass on a panel that imported it and then drew
27 hues anyway.

Four things are checked, and each is a failure this repository has already
had:

1. **Everything is grey except what the sentence is about.** The volcano used
   to paint every point one of 27 LOPIT compartment colours -- eight of them
   the same slategray -- and ``spacr/localisation.py`` records what that cost:
   40 ms of a 49 ms redraw, and no claim. The coloured marks now have to be a
   minority.
2. **The globals come back.** ``custom_volcano_plot`` set ``font.size`` to 18
   with a bare ``rcParams.update``, so every figure the session drew afterwards
   -- in any module -- came out at the volcano's size.
3. **The ink follows the theme**, rather than a hard-coded near-black that is
   invisible on spaCR's dark ground.
4. **The numbers did not move.** A restyle that silently changes a statistic
   is the worst outcome available, so the hit rule, the ranking, the
   normalisation and the Fisher enrichment are all asserted against
   independently computed values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from spacr import toxo as T  # noqa: E402
from spacr.figures.style import (ROLES, TYPE_SCALE, Palette,  # noqa: E402
                                 resolve_ink, theme_target)


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


#: The hues the volcano used to paint its 27 compartments with. Not one of
#: them is in the published palette, and none may survive.
OLD_VOLCANO_HUES = ("darkviolet", "teal", "pink", "slategray", "turquoise",
                    "orange", "green", "blue", "red")

#: The teal curve and purple highlights of the old phenotype plot.
OLD_PHENOTYPE_HUES = ((0 / 255, 155 / 255, 155 / 255),
                      (155 / 255, 55 / 255, 155 / 255))

_MEAN_COL = "T.gondii GT1 CRISPR Phenotype - Mean Phenotype"
_SE_COL = "T.gondii GT1 CRISPR Phenotype - Standard Error"


# --------------------------------------------------------------------------- #
#  Fixtures. Small, synthetic, and with a KNOWN answer, so "the numbers did
#  not move" is checkable rather than merely asserted.
# --------------------------------------------------------------------------- #

def _screen(n=40, seed=3):
    """A gene-level regression result with a handful of real calls in it.

    p values are laid out so that exactly the first ``n // 8`` rows clear
    0.05, which makes the highlight a minority by construction -- the thing
    rule 1 is about.
    """
    rng = np.random.default_rng(seed)
    p = np.full(n, 0.5)
    p[: n // 8] = np.linspace(1e-6, 0.01, n // 8)
    coefficient = rng.normal(0, 0.3, n)
    # Half the calls positive, half negative, so both roles are exercised.
    coefficient[: n // 16] = np.abs(coefficient[: n // 16]) + 0.1
    coefficient[n // 16: n // 8] = -np.abs(coefficient[n // 16: n // 8]) - 0.1
    return pd.DataFrame({
        "feature": [f"gene[{220000 + i}_1]" for i in range(n)],
        "coefficient": coefficient,
        "p_value": p,
    })


def _lopit(n=40):
    places = ["cytosol", "rhoptries 1", "dense granules", "Golgi"]
    return pd.DataFrame({
        "gene_nr": [str(220000 + i) for i in range(n)],
        "tagm_location": [places[i % len(places)] for i in range(n)],
    })


def _gene_table(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Gene ID": [f"TGGT1_{200000 + i}" for i in range(n)],
        _MEAN_COL: rng.normal(0, 1, n),
        _SE_COL: rng.random(n) * 0.2,
        "extra_metric": rng.random(n),
    })


def _facecolours(axis):
    """Every point's face colour on ``axis``, as hex, one entry per mark."""
    out = []
    for collection in axis.collections:
        colours = collection.get_facecolors()
        offsets = np.asarray(collection.get_offsets())
        if len(colours) == 0:
            continue
        for index in range(len(offsets)):
            out.append(to_hex(colours[index % len(colours)]))
    return out


# --------------------------------------------------------------------------- #
#  Rule 2, first, because it is the one that leaks into every other test.
# --------------------------------------------------------------------------- #

def test_no_toxo_figure_leaves_its_style_on_the_globals():
    """Draw all six, then compare matplotlib's globals to what they were.

    A single global write here would restyle every figure drawn afterwards in
    the session, in every other module, until the process exits. It has
    already happened: 18 pt from the volcano reached everything.
    """
    before = dict(matplotlib.rcParams)

    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)
    T.custom_volcano_plot(data, metadata, figsize=6, y_lims=[[0, 2], [3, 8]])
    genes = _gene_table()
    T.plot_gene_phenotypes(genes, ["TGGT1_200001", "TGGT1_200005"])
    T.plot_gene_heatmaps(genes, ["200001", "200005"],
                         [_MEAN_COL, "extra_metric"], normalize=True)

    after = dict(matplotlib.rcParams)
    changed = {key: (before[key], after[key]) for key in before
               if repr(before[key]) != repr(after.get(key))}
    assert not changed, f"spacr.toxo leaked rcParams: {sorted(changed)}"


def test_the_go_panels_leave_the_globals_alone_too(tmp_path):
    """The enrichment figures are the other three, and take a CSV path."""
    before = dict(matplotlib.rcParams)
    _run_go(tmp_path)
    after = dict(matplotlib.rcParams)
    changed = [key for key in before
               if repr(before[key]) != repr(after.get(key))]
    assert not changed, f"go_term_enrichment_by_column leaked: {sorted(changed)}"


# --------------------------------------------------------------------------- #
#  Rule 1: the volcano.
# --------------------------------------------------------------------------- #

def test_the_volcano_is_grey_except_for_the_genes_it_called():
    """Three colours at most, and the coloured marks are the minority.

    The old figure gave every point a compartment colour, so 100% of the
    marks were "highlighted" and the picture had no claim.
    """
    data, metadata = _screen(n=40), _lopit(40)
    hits = T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)

    axis = plt.gcf().axes[0]
    colours = _facecolours(axis)
    assert len(colours) == 40, "a point went missing"

    grey = to_hex(ROLES["data"])
    up, down = to_hex(ROLES["up"]), to_hex(ROLES["down"])
    assert set(colours) <= {grey, up, down}
    coloured = [c for c in colours if c != grey]
    assert len(coloured) == len(hits) == 5
    assert len(coloured) < len(colours) / 2, (
        "more than half the marks carry colour, so the figure has no claim")
    # Both directions really were exercised, so the two roles are not the
    # same role under two names.
    assert set(coloured) == {up, down}

    for gone in OLD_VOLCANO_HUES:
        assert to_hex(gone) not in colours


def test_the_called_genes_are_green_up_and_rust_down():
    """Direction is the argument, so direction is what the hue carries."""
    data, metadata = _screen(n=40), _lopit(40)
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)
    axis = plt.gcf().axes[0]

    for collection in axis.collections:
        offsets = np.asarray(collection.get_offsets())
        if not len(offsets):
            continue
        colour = to_hex(collection.get_facecolors()[0])
        if colour == to_hex(ROLES["up"]):
            assert (offsets[:, 0] > 0).all()
        elif colour == to_hex(ROLES["down"]):
            assert (offsets[:, 0] <= 0).all()


def test_the_volcanos_legend_is_two_lines_of_text_not_twenty_seven_swatches():
    """A legend is an index. It used to be taller than the plot."""
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)
    axis = plt.gcf().axes[0]

    assert axis.get_legend() is None, (
        "the framed swatch legend is back; it is what squeezed the data into "
        "a strip beside it")
    legend_texts = [t.get_text() for t in axis.texts if "called" in t.get_text()]
    assert sorted(legend_texts) == ["called, negative (3)",
                                    "called, positive (2)"]
    # ...and it is coloured to match the marks it indexes.
    by_text = {t.get_text(): to_hex(t.get_color()) for t in axis.texts}
    assert by_text["called, positive (2)"] == to_hex(ROLES["up"])
    assert by_text["called, negative (3)"] == to_hex(ROLES["down"])


def test_a_compartment_can_still_be_asked_for_one_at_a_time():
    """The localisation question survives; the 27-hue answer does not."""
    data, metadata = _screen(n=40), _lopit(40)
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0,
                          highlight_location="rhoptries 1")
    axis = plt.gcf().axes[0]
    colours = _facecolours(axis)

    blue = to_hex(ROLES["highlight"])
    assert colours.count(blue) == 10, "one point per rhoptries-1 gene"
    assert len(set(colours)) <= 4
    assert any(t.get_text() == "rhoptries 1" for t in axis.texts)


def test_asking_for_no_compartment_colours_none_of_them():
    """``highlight_location=None`` and an empty list both mean 'not this'."""
    data, metadata = _screen(), _lopit()
    for wanted in (None, [], "", ["", None]):
        plt.close("all")
        T.custom_volcano_plot(data, metadata, figsize=6, threshold=0,
                              highlight_location=wanted)
        colours = set(_facecolours(plt.gcf().axes[0]))
        assert to_hex(ROLES["highlight"]) not in colours, wanted


def test_the_volcanos_threshold_lines_are_thin_dashed_and_grey():
    """A reference is not a result. They were solid black."""
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0.2)
    axis = plt.gcf().axes[0]

    reference = to_hex(ROLES["reference"])
    assert axis.lines, "the thresholds did not draw"
    for line in axis.lines:
        assert to_hex(line.get_color()) == reference
        assert line.get_linestyle() != "-"
        assert line.get_linewidth() <= 1.0
    # Both coefficient cuts and the p = 0.05 line.
    verticals = sorted(round(float(line.get_xdata()[0]), 3)
                       for line in axis.lines
                       if len(set(line.get_xdata())) == 1)
    assert verticals == [-0.2, 0.2]


def test_a_zero_threshold_draws_the_zero_line_once():
    """``threshold=0`` used to draw two identical lines on top of each other."""
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)
    axis = plt.gcf().axes[0]
    verticals = [line for line in axis.lines
                 if len(set(line.get_xdata())) == 1]
    assert len(verticals) == 1
    assert float(verticals[0].get_xdata()[0]) == 0.0


def test_the_broken_axis_still_splits_the_points_between_its_panels():
    """The broken-axis path is a second figure layout, and gets the same rules."""
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0,
                          y_lims=[[0, 2], [3, 8]])
    figure = plt.gcf()
    assert len(figure.axes) == 2
    lower, upper = figure.axes[1], figure.axes[0]

    below = np.concatenate([np.asarray(c.get_offsets())[:, 1]
                            for c in lower.collections
                            if len(c.get_offsets())])
    above = np.concatenate([np.asarray(c.get_offsets())[:, 1]
                            for c in upper.collections
                            if len(c.get_offsets())])
    assert (below <= 3).all() and (above > 3).all()
    assert len(below) + len(above) == 40
    # The break is a gap, so the spine that would draw through it is off.
    assert not upper.spines["bottom"].get_visible()
    for axis in (lower, upper):
        assert not axis.spines["top"].get_visible()
        assert not axis.spines["right"].get_visible()


def test_the_volcanos_ink_follows_the_theme():
    """Rule 3: a hard-coded near-black is invisible on the dark ground."""
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0)
    axis = plt.gcf().axes[0]
    ink = resolve_ink(theme_target())
    assert to_hex(axis.xaxis.label.get_color()) == to_hex(ink)
    assert to_hex(axis.spines["left"].get_edgecolor()) == to_hex(ink)
    assert not axis.xaxis.get_gridlines()[0].get_visible()


def test_the_type_scales_with_the_canvas_and_keeps_the_published_ratios():
    """7 pt on a 20-inch poster is a footnote; the skill states RATIOS."""
    data, metadata = _screen(), _lopit()
    sizes = {}
    for width in (6, 20):
        plt.close("all")
        T.custom_volcano_plot(data, metadata, figsize=width, threshold=0)
        sizes[width] = plt.gcf().axes[0].xaxis.label.get_fontsize()

    # Small figure: the published absolute size, unscaled.
    assert sizes[6] == pytest.approx(TYPE_SCALE["label"])
    # 20 inches is 2.82x the 180 mm page the scale was measured on.
    assert sizes[20] == pytest.approx(TYPE_SCALE["label"] * 20 / 7.09, rel=1e-6)
    assert sizes[20] > sizes[6]


def test_the_hit_list_is_the_rule_the_picture_draws():
    """The restyle must not move a number.

    The hit rule, computed here from the raw frame, has to be exactly what
    the function returns and exactly what it colours.
    """
    data, metadata = _screen(n=40), _lopit(40)
    expected = [f"{220000 + i}_1" for i in range(40)
                if data["p_value"][i] <= 0.05 and abs(data["coefficient"][i]) >= 0.15]

    hits = T.custom_volcano_plot(data, metadata, figsize=6, threshold=0.15)
    assert hits == expected

    coloured = [c for c in _facecolours(plt.gcf().axes[0])
                if c != to_hex(ROLES["data"])]
    assert len(coloured) == len(expected)


def test_the_fast_path_still_returns_the_same_list_without_drawing():
    """``draw=False`` is `perform_regression`'s path and must not diverge."""
    data, metadata = _screen(), _lopit()
    plt.close("all")
    quiet = T.custom_volcano_plot(data, metadata, figsize=6, threshold=0.15,
                                  draw=False)
    assert plt.get_fignums() == []
    drawn = T.custom_volcano_plot(data, metadata, figsize=6, threshold=0.15)
    assert quiet == drawn


def test_the_volcano_saves_through_the_format_preference(tmp_path, monkeypatch):
    """Rule 4: no figure writes its own extension."""
    seen = {}
    real = T.save_figure

    def spy(fig, path, **kwargs):
        seen["path"] = path
        return real(fig, path, **kwargs)

    monkeypatch.setattr(T, "save_figure", spy)
    data, metadata = _screen(), _lopit()
    T.custom_volcano_plot(data, metadata, figsize=6, threshold=0,
                          save_path=str(tmp_path / "v.pdf"))
    assert seen["path"] == str(tmp_path / "v.pdf")
    assert (tmp_path / "v.pdf").exists()


# --------------------------------------------------------------------------- #
#  The GT1 phenotype curve.
# --------------------------------------------------------------------------- #

def test_the_phenotype_curve_is_grey_and_only_the_named_genes_are_blue():
    genes = _gene_table()
    picked = ["TGGT1_200001", "TGGT1_200005"]
    T.plot_gene_phenotypes(genes, picked)
    axis = plt.gcf().axes[0]

    line = axis.lines[0]
    assert to_hex(line.get_color()) == to_hex(Palette.GREY_DARK)
    for old in OLD_PHENOTYPE_HUES:
        assert to_hex(line.get_color()) != to_hex(old)

    # The SE band takes the line's own hue at the one alpha the published
    # figures use on a curve.
    band = axis.collections[0]
    assert to_hex(band.get_facecolor()[0]) == to_hex(Palette.GREY_DARK)
    assert float(band.get_alpha()) == pytest.approx(0.25)

    highlights = [c for c in axis.collections
                  if str(c.get_label()).startswith("Highlighted Gene: ")]
    assert len(highlights) == 2
    for scatter in highlights:
        assert to_hex(scatter.get_facecolor()[0]) == to_hex(ROLES["highlight"])


def test_the_phenotype_curve_still_ranks_the_genes_it_used_to():
    """The restyle must not move the curve. Ranks recomputed here."""
    genes = _gene_table()
    T.plot_gene_phenotypes(genes, ["TGGT1_200001"])
    axis = plt.gcf().axes[0]
    x, y = axis.lines[0].get_xdata(), axis.lines[0].get_ydata()

    assert list(x) == list(range(1, len(genes) + 1))
    assert np.asarray(y) == pytest.approx(np.sort(genes[_MEAN_COL].to_numpy()))
    assert axis.get_xlabel() == "Rank"
    assert axis.get_ylabel() == "Mean Phenotype"


def test_the_phenotype_plot_does_not_retype_the_callers_table():
    """``data.loc[:, col] = pd.to_numeric(...)`` wrote through to the caller.

    ``ml.perform_regression`` reads the GT1 table once and hands the same
    frame to this function, so the coercion landed on a table it goes on to
    use. Found while restyling; fixed here.
    """
    genes = _gene_table()
    before = {c: genes[c].dtype for c in genes.columns}
    columns = list(genes.columns)

    T.plot_gene_phenotypes(genes, ["TGGT1_200001"])

    assert list(genes.columns) == columns, "a working column leaked to the caller"
    assert {c: genes[c].dtype for c in genes.columns} == before


# --------------------------------------------------------------------------- #
#  The ME49 transcription heatmap.
# --------------------------------------------------------------------------- #

def test_the_transcription_heatmap_uses_the_single_hue_ramp():
    """viridis is a rainbow standing in for one ordered score."""
    genes = _gene_table()
    T.plot_gene_heatmaps(genes, ["200001", "200005"],
                         [_MEAN_COL, "extra_metric"], normalize=True)
    figure = plt.gcf()
    image = figure.axes[0].collections[0]
    assert image.cmap.name == Palette.SEQUENTIAL
    assert image.cmap.name != "viridis"
    # No white rules between the cells: the rule is no gridlines, ever.
    assert float(np.ravel(image.get_linewidths())[0]) == 0.0


def test_the_heatmap_normalisation_is_the_one_it_always_was():
    """Every gene's row scaled to [0, 1]; recomputed here, not read back."""
    genes = _gene_table()
    picked = ["200001", "200005"]
    columns = [_MEAN_COL, "extra_metric"]
    T.plot_gene_heatmaps(genes, picked, columns, normalize=True)

    drawn = plt.gcf().axes[0].collections[0].get_array().data.reshape(-1)
    rows = genes[genes["Gene ID"].str.split("_").str[1].isin(picked)]
    expected = []
    for _, row in rows.iterrows():
        values = row[columns].astype(float).to_numpy()
        expected.extend((values - values.min()) / (values.max() - values.min()))
    assert np.sort(drawn) == pytest.approx(np.sort(np.asarray(expected)))


def test_the_heatmap_does_not_add_a_column_to_the_callers_table():
    """``data['x'] = ...`` was a new column on the frame ``ml`` passes in."""
    genes = _gene_table()
    columns = list(genes.columns)
    T.plot_gene_heatmaps(genes, ["200001"], [_MEAN_COL], normalize=False)
    assert list(genes.columns) == columns


# --------------------------------------------------------------------------- #
#  GO enrichment.
# --------------------------------------------------------------------------- #

def _run_go(tmp_path, n=40):
    """Run the enrichment on a table where one term is genuinely enriched.

    The first ten genes are the hits: eight are ``metabolism``, which is over-
    represented, and one each of ``signaling`` and ``transport``, which are
    depleted. ``binding`` never appears among the hits, so its enrichment is 0
    and the function drops it -- that filter is the function's own and the
    restyle did not touch it, so the assertions below expect three points and
    not four.
    """
    gene_nrs = [str(220000 + i) for i in range(n)]
    terms = ["metabolism", "signaling", "transport", "binding"]
    assigned = []
    for i in range(n):
        if i < 8:
            assigned.append(terms[0])
        elif i < 10:
            assigned.append(terms[1 + (i - 8)])
        else:
            assigned.append(terms[1 + (i % 3)])
    metadata = pd.DataFrame({
        "Gene ID": [f"TGGT1_{g}" for g in gene_nrs],
        "GO": assigned,
    })
    path = tmp_path / "go.csv"
    metadata.to_csv(path, index=False)
    hits = pd.DataFrame({"n_gene": gene_nrs[:10]})
    T.go_term_enrichment_by_column(hits, str(path), go_term_columns=["GO"])
    return metadata


def _expected_enrichment(metadata, n=40, hits=10):
    """Enrichment per term, recomputed from the table rather than read back.

    Only the terms the function keeps -- enrichment strictly above zero.
    """
    counts = metadata["GO"].value_counts()
    hit_counts = metadata.iloc[:hits]["GO"].value_counts()
    scores = {term: (hit_counts.get(term, 0) / hits) / (counts[term] / n)
              for term in counts.index}
    return {term: score for term, score in scores.items() if score > 0.0}


def test_the_enrichment_panels_are_grey_with_the_called_terms_blue(tmp_path):
    """``hue='GO Term'`` gave every term of the ontology its own hue."""
    _run_go(tmp_path)
    figures = [plt.figure(i) for i in plt.get_fignums()]
    assert len(figures) == 2

    for figure in figures:
        axis = figure.axes[0]
        colours = set(_facecolours(axis))
        assert colours <= {to_hex(ROLES["data"]), to_hex(ROLES["highlight"])}, (
            f"a hue outside the palette survived: {sorted(colours)}")
        assert axis.get_legend() is None


def test_the_enrichment_scores_are_the_numbers_fisher_gives(tmp_path):
    """The restyle touched the drawing, not the statistic."""
    metadata = _run_go(tmp_path, n=40)
    expected = _expected_enrichment(metadata)
    # The fixture is only a real test if the terms differ.
    assert expected["metabolism"] > 1.0
    assert all(v < 1.0 for k, v in expected.items() if k != "metabolism")

    axis = plt.figure(plt.get_fignums()[0]).axes[0]
    drawn = np.concatenate([np.asarray(c.get_offsets())[:, 0]
                            for c in axis.collections])
    assert sorted(drawn) == pytest.approx(sorted(expected.values()))
    assert axis.get_xlabel() == "Enrichment Score"
    assert axis.get_ylabel() == "-log10(P-value)"


def test_the_combined_panel_still_names_every_term(tmp_path):
    """The names moved from a legend column onto the points."""
    metadata = _run_go(tmp_path)
    combined = plt.figure(plt.get_fignums()[1]).axes[0]
    labelled = {t.get_text() for t in combined.texts}
    assert set(_expected_enrichment(metadata)) <= labelled
    # Every point is named, and nothing else is: the panel carries no legend
    # to be confused with the labels.
    assert combined.get_legend() is None
