"""The two well-level distributions a run writes, converted to the house style.

``fraction_histogram.pdf`` and ``log_pred_histogram.pdf`` were the last figures
the regression pipeline drew in the old idiom: ``spacr.plot.plot_histogram``,
which is a 10x10 inch canvas, a saturated teal ``sns.histplot`` at 60% alpha,
the sentence title "Histogram of fraction", the y-axis "Frequency" and a
closing ``plt.show()``. Every one of those is something
``.claude/skills/apicomplexan-figures`` says not to do, so each is a test here.

Two of these tests are not about style at all, and they are the ones worth
keeping:

* :func:`test_the_response_is_counted_once_per_well` — the old histogram was
  handed one row per guide-in-well and the response is a property of the WELL.
  On the tsg101 screen that is 1,945 rows carrying 610 distinct values, so the
  figure counted a 15-guide well fifteen times and stated an n three times the
  number of independent observations.

* :func:`test_a_guide_is_compared_with_its_own_wells_equal_share` — a raw
  fraction cannot answer "is the library evenly represented", because a
  two-guide well splits near 1/2 and a fifteen-guide well near 1/15. Pooling
  them measures how many guides landed per well.

The numbers in the real-data tests at the bottom come from
``tsg101_screen/test/results/plate1_dv/ols/list/regression_data.csv`` and are
pinned, so a change to how a share or a moment is computed has to be argued
for rather than absorbed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgba  # noqa: E402

from spacr.figures import distributions as D  # noqa: E402
from spacr.figures.style import ROLES, WEIGHTS  # noqa: E402

#: The real screen. Every pinned number below was measured from it.
REAL = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/test/results"
        "/plate1_dv/ols/list/regression_data.csv")


def _wells(n_wells=40, seed=0):
    """A well-level frame with the shape the pipeline actually hands over.

    One row per guide-in-well, a varying number of guides per well including
    wells with exactly one, and a response that is constant within a well —
    which is the property the de-duplication depends on and therefore the
    property a fixture must reproduce.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        k = int(rng.integers(1, 6))
        response = float(rng.lognormal(-2.0, 0.4))
        for guide in range(k):
            rows.append({"prc": f"plate1_r{well // 8 + 1}_c{well % 8 + 1}",
                         "grna": f"g{well}_{guide}",
                         "gene": f"gene{well % 7}",
                         "fraction": float(rng.uniform(0.02, 0.5)),
                         "pred": np.expm1(response),
                         "log_pred": response})
    return pd.DataFrame(rows)


def _draw(key, frame=None, **kwargs):
    """``(figure, ax, panel)`` for one panel, drawn for print."""
    frame = _wells() if frame is None else frame
    figure, panel = D.build_panel(key, frame, target="print", **kwargs)
    return figure, figure.axes[0], panel


# --------------------------------------------------------------------------- #
#  The style is scoped. This is the one that would cost the most.
# --------------------------------------------------------------------------- #

def test_drawing_a_distribution_does_not_leak_into_the_process():
    """spaCR draws from a long-lived GUI. A style applied by writing rcParams
    restyles every figure drawn afterwards, in every module, until the process
    exits — which is exactly what the old ``plot_histogram`` path risked and
    what the context manager exists to prevent."""
    before = dict(plt.rcParams)
    for key, kwargs in (("guide_fraction", {}),
                        ("response", {"column": "log_pred"})):
        figure, _ax, _panel = _draw(key, **kwargs)
        plt.close(figure)
    changed = {k for k in before if str(before[k]) != str(plt.rcParams[k])}
    assert not changed, f"these rcParams were left changed: {sorted(changed)}"


def test_the_module_never_writes_rcparams_or_calls_show():
    """Parsed, not grepped: the module documents both prohibitions in prose,
    so a text search matches its own explanation and passes whatever the code
    does."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(D))
    updates, shows = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Attribute) and func.attr == "update"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "rcParams"):
            updates.append(node.lineno)
        if isinstance(func, ast.Attribute) and func.attr == "show":
            shows.append(node.lineno)
    assert not updates, f"rcParams written globally at lines {updates}"
    assert not shows, (
        f"plt.show() at lines {shows}; the old plot_histogram called it and "
        f"popped a window out of headless and GUI-driven runs")


# --------------------------------------------------------------------------- #
#  The rules the skill states, one test each
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_the_fill_is_the_pale_house_fill_and_is_solid(key, kwargs):
    """"Density/histogram fills: solid but pale (CORAL), not a translucent
    saturated colour." The old one was #009B9B teal at alpha 0.6.

    THE TRANSPARENCY IS CHECKED ON THE RGBA, not on ``get_alpha()``. seaborn
    bakes its ``alpha=`` into the facecolour and leaves the alpha property
    None, so ``assert bar.get_alpha() in (None, 1.0)`` passes on the exact
    figure this panel replaces.
    """
    figure, ax, _panel = _draw(key, **kwargs)
    try:
        bars = [p for p in ax.patches if p.get_height() > 0]
        assert bars, "the panel drew no bars"
        for bar in bars:
            assert to_hex(bar.get_facecolor()).lower() == ROLES["fill"].lower()
            assert bar.get_alpha() in (None, 1.0)
            assert to_rgba(bar.get_facecolor())[3] == 1.0
            assert bar.get_edgecolor()[3] == 0.0, "bars are drawn without edges"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_there_are_no_gridlines_and_no_sentence_title(key, kwargs):
    """"No gridlines. Ever." and "No panel titles as sentences" — the old one
    was titled "Histogram of fraction"."""
    figure, ax, _panel = _draw(key, **kwargs)
    try:
        assert ax.get_title() == ""
        assert not any(line.get_visible()
                       for line in ax.get_xgridlines() + ax.get_ygridlines())
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_the_reference_is_grey_thin_and_dashed(key, kwargs):
    """A reference is not a result. Both panels carry one — the equal-share
    line and the fitted family's own normal — and neither may out-weigh the
    bars it is drawn over."""
    figure, ax, _panel = _draw(key, **kwargs)
    try:
        lines = [line for line in ax.lines if line.get_visible()]
        assert lines, "the panel drew no reference at all"
        for line in lines:
            assert to_hex(line.get_color()).lower() == \
                ROLES["reference"].lower()
            assert line.get_linewidth() <= WEIGHTS["reference"]
            assert line.get_linestyle() not in ("-", "solid")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_the_in_panel_note_has_no_frame_and_states_the_n(key, kwargs):
    """"no frame, no box" — the old QC idiom put its stats in a white rounded
    box. And the n is stated in-panel, which is the skill's rule for a
    distribution and the thing a reader cannot recover from the bars."""
    figure, ax, _panel = _draw(key, **kwargs)
    try:
        notes = [t for t in ax.texts if "n = " in t.get_text()]
        assert len(notes) == 1, [t.get_text() for t in ax.texts]
        assert notes[0].get_bbox_patch() is None
        assert not ax.get_legend(), "a framed legend is not the house style"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_the_axis_labels_are_lower_case_and_spelled_out(key, kwargs):
    """"Axis labels are lower-case" — and never "Frequency", which names the
    mark instead of what was counted."""
    figure, ax, _panel = _draw(key, **kwargs)
    try:
        for label in (ax.get_xlabel(), ax.get_ylabel()):
            assert label, "an axis went out unlabelled"
            assert label == label.lower(), label
            assert label != "frequency"
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The panel says something. "Distribution of X" is what the axis already says.
# --------------------------------------------------------------------------- #

def test_the_fraction_caption_answers_whether_the_library_is_even():
    figure, _ax, panel = _draw("guide_fraction")
    try:
        assert "Gini" in panel.caption
        assert "equal" in panel.caption
        # A caption that only names its variable has wasted the panel.
        assert len(panel.caption.split()) > 25, panel.caption
    finally:
        plt.close(figure)


def test_the_response_caption_answers_whether_the_assumption_is_plausible():
    figure, _ax, panel = _draw("response", column="log_pred")
    try:
        assert "Skewness" in panel.caption
        assert "kurtosis" in panel.caption
        # The convention is stated, so no reader is handed a bare adjective.
        assert "near-symmetric" in panel.caption
        # And the panel must not overclaim: OLS assumes normal RESIDUALS.
        assert "RESIDUALS" in panel.caption
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  Evenness: a guide is measured against its OWN well
# --------------------------------------------------------------------------- #

def test_a_guide_is_compared_with_its_own_wells_equal_share():
    """The arithmetic, on a frame small enough to check by hand.

    Well A holds 0.1 and 0.3, so an equal split is 0.2 and the two guides sit
    at 0.5x and 1.5x. Well B holds 0.4 and 0.4 and both sit at exactly 1x —
    the same *raw* 0.4 that is 2x in well A. A histogram of the raw column
    cannot tell those apart, which is the whole reason for the division.
    """
    frame = pd.DataFrame({
        "prc": ["A", "A", "B", "B"],
        "fraction": [0.1, 0.3, 0.4, 0.4]})
    values, dropped_wells, dropped_rows = D.relative_representation(
        frame, "fraction", "prc")
    assert dropped_wells == 0 and dropped_rows == 0
    assert np.allclose(sorted(values), [0.5, 1.0, 1.0, 1.5])


def test_a_single_guide_well_is_excluded_rather_than_spiked_onto_the_line():
    """One guide is 1.0x its own equal share by construction. Left in, those
    rows pile pure arithmetic onto the exact x the reference line stands at —
    the worst place in the panel for an artefact."""
    frame = pd.DataFrame({
        "prc": ["A", "A", "B"],
        "fraction": [0.1, 0.3, 0.4]})
    values, dropped_wells, dropped_rows = D.relative_representation(
        frame, "fraction", "prc")
    assert dropped_wells == 1 and dropped_rows == 1
    assert np.allclose(sorted(values), [0.5, 1.5])
    assert 1.0 not in set(values)


def test_a_well_that_sums_to_nothing_is_dropped_not_divided_by_zero():
    frame = pd.DataFrame({"prc": ["A", "A", "B", "B"],
                          "fraction": [0.0, 0.0, 0.25, 0.75]})
    values, dropped_wells, _rows = D.relative_representation(
        frame, "fraction", "prc")
    assert dropped_wells == 1
    assert np.all(np.isfinite(values))
    assert np.allclose(sorted(values), [0.5, 1.5])


def test_the_equal_share_line_stands_at_exactly_one_and_carries_a_tick():
    """The reference is only legitimate because the x is a ratio, and a
    reader can only read it if 1 is on the axis.

    The tick thinning used to slice from the low end, which starts at
    whatever the smallest guide happened to be: on the real screen that
    produced ticks at 0.0625, 0.25 and 4 with the reference line standing over
    an unlabelled position.
    """
    figure, ax, _panel = _draw("guide_fraction")
    try:
        positions = [line.get_xdata()[0] for line in ax.lines]
        assert 1.0 in positions, positions
        assert "1" in [t.get_text() for t in ax.get_xticklabels()]
        assert ax.get_xscale() == "log"
    finally:
        plt.close(figure)


def test_an_unknown_share_does_not_inflate_the_rest_of_its_well():
    """A NaN share is not a guide with no reads — it is a guide whose share is
    unknown, and it must not shrink the equal split for its well-mates.

    ``size`` counts the unknown row, ``sum`` does not, so well A's equal split
    came out 0.4/3 instead of 0.4/2 and its two real guides were reported at
    0.75x and 2.25x rather than 0.5x and 1.5x. Silently: the unknown row is
    dropped and the dropped-row count still reads 1.
    """
    frame = pd.DataFrame({"prc": ["A", "A", "A"],
                          "fraction": [0.1, 0.3, np.nan]})
    values, dropped_wells, dropped_rows = D.relative_representation(
        frame, "fraction", "prc")
    assert dropped_wells == 0 and dropped_rows == 1
    assert np.allclose(sorted(values), [0.5, 1.5])


def test_a_well_with_one_real_guide_and_one_unknown_is_not_kept_at_two_x():
    """The same arithmetic artefact single-guide wells are excluded for, and
    in the worst possible place: it lands at exactly 2.0x, which is the "at
    least twice equal" cut the panel reports as a number."""
    frame = pd.DataFrame({"prc": ["B", "B", "C", "C"],
                          "fraction": [0.4, np.nan, 0.25, 0.75]})
    values, dropped_wells, dropped_rows = D.relative_representation(
        frame, "fraction", "prc")
    assert dropped_wells == 1 and dropped_rows == 2
    assert 2.0 not in set(values)
    assert np.allclose(sorted(values), [0.5, 1.5])


def test_a_tiny_share_is_labelled_rather_than_rounded_away_to_nothing():
    """Every tick says where it is, to within a couple of percent.

    The labels were formatted with a fixed ``.4f`` and then stripped of
    trailing zeros, which is only honest down to 2^-6: 2^-9 read "0.002" (2.4%
    out), 2^-12 read "0.0002" (18% out) and 2^-15 rounded to "0.0000", which
    the strip turned into "0." — on a wide enough axis, twice, so two
    different ticks carried the same label. A raw share of 1e-4 is ordinary
    for a deeply sequenced library, so this is the normal axis, not a corner.
    """
    frame = pd.DataFrame({"prc": np.repeat([f"w{i}" for i in range(10)], 4),
                          "fraction": np.geomspace(2e-5, 0.5, 40)})
    figure, ax, panel = _draw("guide_fraction", frame, relative=False)
    try:
        assert panel.drawn is True
        labelled = [(position, text.get_text()) for position, text
                    in zip(ax.get_xticks(), ax.get_xticklabels())
                    if text.get_text()]
        assert labelled
        assert len({text for _p, text in labelled}) == len(labelled), labelled
        for position, text in labelled:
            assert float(text) == pytest.approx(position, rel=0.02), \
                f"the tick at {position!r} is labelled {text!r}"
    finally:
        plt.close(figure)


def test_gini_is_zero_when_every_guide_is_equal_and_climbs_with_inequality():
    assert D.gini(np.full(50, 0.2)) == pytest.approx(0.0, abs=1e-12)
    even = D.gini(np.array([1.0, 1.0, 1.0, 1.0]))
    skewed = D.gini(np.array([1.0, 1.0, 1.0, 9.0]))
    hogged = D.gini(np.array([0.0, 0.0, 0.0, 1.0]))
    assert even < skewed < hogged
    assert hogged == pytest.approx(0.75)


def test_gini_of_nothing_is_nan_rather_than_a_perfectly_even_zero():
    """An evenness of 0.0 reads as "perfectly even", which is the opposite of
    "there was nothing to measure"."""
    assert np.isnan(D.gini([]))
    assert np.isnan(D.gini([0.0, 0.0, 0.0]))
    assert np.isnan(D.gini([1.0, -1.0]))


# --------------------------------------------------------------------------- #
#  The response is counted once per well
# --------------------------------------------------------------------------- #

def test_the_response_is_counted_once_per_well():
    """The old histogram counted a well once per guide it retained."""
    frame = pd.DataFrame({"prc": ["A", "A", "A", "B"],
                          "log_pred": [0.2, 0.2, 0.2, 0.5]})
    values, deduplicated = D.one_value_per_well(frame, "log_pred", "prc")
    assert deduplicated is True
    assert sorted(values) == [0.2, 0.5]


def test_a_categorical_well_column_with_unused_categories_counts_correctly():
    """The shape the pipeline ACTUALLY hands over.

    ``ml.check_and_clean_data`` converts `prc` to a categorical before the
    call site, and patsy then drops rows, leaving categories with no rows
    behind. A groupby over a categorical defaults to ``observed=False``, which
    emits one group per CATEGORY — so the response would be counted over every
    well the plate ever had rather than the wells that were fitted. Today the
    empty groups come back NaN and get filtered downstream, which is luck
    rather than design, and pandas 3 flips the default anyway.
    """
    frame = pd.DataFrame({
        "prc": pd.Categorical(np.repeat(["w0", "w1", "w2"], 4),
                              categories=[f"w{i}" for i in range(12)]),
        "log_pred": np.repeat([0.1, 0.2, 0.3], 4),
        "fraction": np.linspace(0.05, 0.4, 12)})
    assert len(frame["prc"].cat.categories) == 12    # 9 of them unused

    values, deduplicated = D.one_value_per_well(frame, "log_pred", "prc")
    assert deduplicated is True
    assert sorted(values) == [0.1, 0.2, 0.3]

    relative, dropped_wells, _rows = D.relative_representation(
        frame, "fraction", "prc")
    assert relative.size == 12 and dropped_wells == 0
    # Each well's four shares must divide by that well's own mean, so every
    # well's shares average to exactly 1.
    assert relative.reshape(3, 4).mean(axis=1) == pytest.approx(1.0)


def test_no_pandas_groupby_in_this_module_leaves_observed_to_the_default():
    """Parsed rather than trusted: the default differs between pandas 2 and 3,
    so an unpinned groupby here means the panel counts differently on the
    maintainer's machine than on CI."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(D))
    unpinned = [node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "groupby"
                and "observed" not in {kw.arg for kw in node.keywords}]
    assert not unpinned, f"groupby without observed= at lines {unpinned}"


def test_a_response_that_genuinely_varies_within_a_well_is_left_alone():
    """Checked rather than assumed: collapsing a per-guide response would
    throw away the data instead of a duplicate."""
    frame = pd.DataFrame({"prc": ["A", "A", "B"],
                          "log_pred": [0.2, 0.4, 0.5]})
    values, deduplicated = D.one_value_per_well(frame, "log_pred", "prc")
    assert deduplicated is False
    assert sorted(values) == [0.2, 0.4, 0.5]


def test_the_panel_states_wells_when_it_deduplicated_and_says_so():
    """Ten wells of three guides each: the y-axis counts wells, the note
    states wells, and the caption says the 30 rows were 10 observations."""
    frame = pd.DataFrame({
        "prc": np.repeat([f"w{i}" for i in range(10)], 3),
        "log_pred": np.repeat(np.linspace(0.1, 0.5, 10), 3),
        "fraction": np.linspace(0.05, 0.4, 30)})
    figure, ax, panel = _draw("response", frame, column="log_pred")
    try:
        assert panel.drawn is True
        assert ax.get_ylabel() == "wells"
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "n = 10 wells" in note
        assert "30 guide-level rows collapse to 10" in panel.caption
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The shape statistics a reader acts on
# --------------------------------------------------------------------------- #

def test_shape_of_calls_a_normal_sample_near_symmetric():
    sample = np.random.default_rng(3).normal(0, 1, 4000)
    shape = D.shape_of(sample)
    assert abs(shape["skew"]) < 0.5
    assert shape["verdict"] == "near-symmetric"
    assert shape["n"] == 4000


def test_shape_of_calls_a_lognormal_sample_strongly_skewed_right():
    sample = np.random.default_rng(4).lognormal(0, 1, 4000)
    shape = D.shape_of(sample)
    assert shape["skew"] > 1
    assert shape["verdict"] == "strongly skewed right"


def test_shape_of_names_the_symmetric_heavy_tailed_case_separately():
    """Symmetric and heavy-tailed is a different failure from skewed: a normal
    fitted to it has the right centre and the wrong tail probabilities, which
    is precisely what a p-value is computed from."""
    rng = np.random.default_rng(5)
    sample = rng.standard_t(3, 6000)
    shape = D.shape_of(sample)
    assert abs(shape["skew"]) < 0.5 and shape["excess_kurtosis"] > 1
    assert shape["verdict"] == "symmetric, heavy-tailed"


def test_shape_of_a_constant_is_not_measurable_rather_than_zero():
    shape = D.shape_of(np.full(20, 3.0))
    assert shape["verdict"] == "not measurable"
    assert np.isnan(shape["skew"])


def test_the_before_transform_skew_is_stated_only_when_the_raw_column_is_there():
    """And on the pipeline's own frame it is NOT there, which is worth pinning.

    ``ml.check_and_clean_data`` keeps ``['fraction', dependent_variable]`` plus
    the identifiers, so a run fitted on ``log_pred`` hands this module a frame
    with no ``pred`` column and the panel cannot say what the transform bought.
    It must then say nothing rather than a number it did not measure — the
    line appears only for a caller drawing from ``regression_data.csv``, which
    still carries both.
    """
    frame = _wells()
    figure, ax, _panel = _draw("response", frame, column="log_pred")
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "skew before log:" in note
    finally:
        plt.close(figure)

    figure, ax, _panel = _draw("response", frame.drop(columns=["pred"]),
                               column="log_pred")
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "before log" not in note, note
        assert "skew = " in note
    finally:
        plt.close(figure)


def test_the_normal_reference_is_only_drawn_for_the_family_that_was_fitted():
    """A normal over a Poisson fit is a curve no part of the model assumed,
    and a reader would take the mismatch for a finding."""
    frame = _wells()
    figure, ax, panel = _draw("response", frame, column="log_pred",
                              family="poisson")
    try:
        assert not ax.lines
        assert "no normal is drawn" in panel.caption
        assert "poisson" in panel.caption
    finally:
        plt.close(figure)
    figure, ax, panel = _draw("response", frame, column="log_pred")
    try:
        assert len(ax.lines) == 1
        assert "normal of the same mean" in panel.caption
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  A panel that cannot be drawn says why
# --------------------------------------------------------------------------- #

def test_a_frame_with_no_fraction_column_says_so_rather_than_drawing_empty():
    figure, ax, panel = _draw("guide_fraction",
                              _wells().drop(columns=["fraction"]))
    try:
        assert panel.drawn is False
        assert "fraction" in panel.reason
        assert not ax.patches
    finally:
        plt.close(figure)


def test_without_a_well_column_the_relative_panel_refuses_and_explains():
    """It refuses rather than quietly falling back, because the fallback
    answers a different question and would look identical."""
    figure, _ax, panel = _draw("guide_fraction",
                               _wells().drop(columns=["prc"]))
    try:
        assert panel.drawn is False
        assert "equal share" in panel.reason
    finally:
        plt.close(figure)


def test_the_raw_fallback_does_not_claim_the_equal_share_construction():
    """Asked for explicitly, the raw view draws — and its caption says the
    spread mixes evenness with guides-per-well, which the relative view does
    not have to say."""
    figure, _ax, panel = _draw("guide_fraction",
                               _wells().drop(columns=["prc"]), relative=False)
    try:
        assert panel.drawn is True
        assert "divided by an equal split" not in panel.caption
        assert "mixes uneven representation with how many guides" \
            in panel.caption
    finally:
        plt.close(figure)


def test_the_raw_fallbacks_in_panel_note_does_not_say_equal_either():
    """On the raw axis there is no "equal" for a guide to be twice of, so
    "≥ 2× equal" there would attach a meaning the panel did not measure."""
    figure, ax, _panel = _draw("guide_fraction",
                               _wells().drop(columns=["prc"]), relative=False)
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "equal" not in note, note
        assert "Gini" in note and "median" in note
    finally:
        plt.close(figure)


def test_a_library_where_every_share_is_identical_still_draws():
    """Degenerate but not impossible, and ``geomspace`` over a zero-width
    range returns identical edges, which matplotlib bins into nothing."""
    frame = pd.DataFrame({"prc": np.repeat(["A", "B", "C", "D", "E"], 4),
                          "fraction": np.full(20, 0.25)})
    figure, ax, panel = _draw("guide_fraction", frame)
    try:
        assert panel.drawn is True
        assert sum(p.get_height() for p in ax.patches) == 20
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "Gini = 0.00" in note, note
    finally:
        plt.close(figure)


@pytest.mark.parametrize("key,kwargs", [("guide_fraction", {}),
                                        ("response", {"column": "log_pred"})])
def test_too_few_values_is_a_stated_reason_not_a_rug(key, kwargs):
    frame = _wells().head(3)
    figure, _ax, panel = _draw(key, frame, **kwargs)
    try:
        assert panel.drawn is False
        assert str(D.MIN_VALUES) in panel.reason or "usable" in panel.reason
    finally:
        plt.close(figure)


def test_a_response_column_that_cannot_be_identified_is_named_as_the_reason():
    frame = pd.DataFrame({"prc": ["A", "B"], "note": ["x", "y"]})
    figure, _ax, panel = _draw("response", frame)
    try:
        assert panel.drawn is False
        assert "response column" in panel.reason
    finally:
        plt.close(figure)


def test_the_response_column_is_unambiguous_when_the_frame_has_one_number():
    """The shape ``dmatrices`` hands back: a single-column response frame."""
    frame = pd.DataFrame({"log_pred": np.linspace(0.1, 0.4, 30)})
    assert D.response_column(frame) == "log_pred"
    assert D.response_column(_wells(), "pred") == "pred"
    assert D.response_column(_wells()) == "log_pred"


# --------------------------------------------------------------------------- #
#  The files a run writes keep their names
# --------------------------------------------------------------------------- #

def test_saving_writes_the_same_two_filenames_the_pipeline_always_wrote(
        tmp_path):
    """The grid view, the queue and ``test_cov_ml_regression_core`` all find
    these figures by name; a restyle that renames them is a restyle that
    loses them."""
    written = D.save_distributions(_wells(), str(tmp_path),
                                   response_variable="log_pred",
                                   target="print")
    assert set(written) == {"guide_fraction", "response"}
    assert (tmp_path / "fraction_histogram.pdf").stat().st_size > 0
    assert (tmp_path / "log_pred_histogram.pdf").stat().st_size > 0
    assert not plt.get_fignums(), "save_distributions leaked an open figure"


def test_a_saved_distribution_is_inked_for_the_page_not_for_the_gui_theme(
        tmp_path, monkeypatch):
    """A file is read on a page, so a saver resolves its ink for one.

    ``theme_target()`` answers what the GUI theme is doing and returns
    ``'screen'`` for every user who has not explicitly set a white figure
    background — ``INK_SCREEN`` (#E8EDEE) on a transparent PDF, which is
    near-white axes and labels on a white page. ``regression_qc`` pinned
    ``_REPORT_TARGET = 'print'`` for exactly this reason and
    ``ml._save_regression_figure`` passes ``'print'`` for the sheet these two
    figures sit beside on the grid; a saver that asked the theme instead would
    put two invisible tiles in the middle of a legible run.
    """
    import spacr.plot as P
    from spacr.figures.style import INK_PRINT

    monkeypatch.setattr(D, "theme_target", lambda: "screen")
    inks = []
    real_save = P.save_figure

    def capture(figure, path, **kwargs):
        inks.append(figure.axes[0].xaxis.label.get_color())
        return real_save(figure, path, **kwargs)

    monkeypatch.setattr(P, "save_figure", capture)
    written = D.save_distributions(_wells(), str(tmp_path),
                                   response_variable="log_pred")
    assert set(written) == {"guide_fraction", "response"}
    assert inks and set(inks) == {INK_PRINT}, inks


def test_saving_skips_a_panel_it_cannot_draw_rather_than_writing_a_blank(
        tmp_path, capsys):
    """An empty figure in a results folder is worse than a missing one,
    because the grid view will show it."""
    frame = _wells().drop(columns=["fraction"])
    written = D.save_distributions(frame, str(tmp_path),
                                   response_variable="log_pred",
                                   target="print")
    assert set(written) == {"response"}
    assert not (tmp_path / "fraction_histogram.pdf").exists()
    assert "Skipped guide_fraction" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
#  These are not coefficient panels, and must not join that registry
# --------------------------------------------------------------------------- #

def test_these_panels_are_not_in_the_coefficient_registry():
    """``panels.REGISTRY`` is iterated over the coefficient table, which has
    neither a fraction column nor a response. Merging these in would make
    ``build_sheet`` report two permanently skipped panels on every run."""
    from spacr.figures import panels

    assert not set(D.REGISTRY) & set(panels.REGISTRY)
    assert D.ORDER == ("guide_fraction", "response")


# --------------------------------------------------------------------------- #
#  The real screen
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real_frame():
    import os

    if not os.path.exists(REAL):
        pytest.skip(f"the real screen is not on this machine: {REAL}")
    return pd.read_csv(REAL)


def test_the_real_screen_draws_both_panels_with_the_numbers_pinned(real_frame):
    """1,945 guide-in-well rows over 610 wells, 93 of them single-guide."""
    assert len(real_frame) == 1945
    assert real_frame["prc"].nunique() == 610

    figure, ax, panel = _draw("guide_fraction", real_frame)
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "n = 1,852 guides" in note          # 1945 - 93 single-guide
        assert "Gini = 0.32" in note
        assert "93" in panel.caption               # the excluded wells
    finally:
        plt.close(figure)


def test_the_real_screens_response_is_610_wells_not_1945_rows(real_frame):
    """The number the old figure got wrong, on the data it got it wrong on."""
    figure, ax, panel = _draw("response", real_frame, column="log_pred")
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "n = 610 wells" in note
        assert "skew = +1.83" in note
        assert "strongly skewed right" in note
        assert "1,945 guide-level rows collapse to 610" in panel.caption
    finally:
        plt.close(figure)


def test_the_real_screens_log_transform_reduced_the_skew_but_did_not_fix_it(
        real_frame):
    """The one number that says whether the transform was worth having.

    log1p took the response from 2.43 to 1.83, so it helped and did not
    deliver symmetry — the panel says both, which is what decides whether the
    q-q and residual panels need reading before the hits are believed.
    """
    raw = D.shape_of(real_frame.groupby("prc")["pred"].first())
    logged = D.shape_of(real_frame.groupby("prc")["log_pred"].first())
    assert raw["skew"] == pytest.approx(2.43, abs=0.01)
    assert logged["skew"] == pytest.approx(1.83, abs=0.01)
    assert logged["verdict"] == "strongly skewed right"

    figure, ax, _panel = _draw("response", real_frame, column="log_pred")
    try:
        note = [t.get_text() for t in ax.texts if "n = " in t.get_text()][0]
        assert "skew before log: +2.43" in note
    finally:
        plt.close(figure)
