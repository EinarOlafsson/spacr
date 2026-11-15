"""Table-driven tests of ``spacr.volcano_style.render_volcano``: every style option has to reach the picture.

The style object is what the figure menu writes into, so a field the
renderer ignores is a control that does nothing.  One table walks the
series, reference lines, point labels and axis labels a style produces on
synthetic coefficient tables; a second test reads the axis options, the
ink, the broken-axis label placement and the refusal off the drawn figure.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from spacr import volcano_style as vs  # noqa: E402
from spacr.localisation import table  # noqa: E402
from spacr.volcano_style import VolcanoStyle, render_volcano  # noqa: E402

ALPHA_RULE = ((0.0, 1.0), (1.301, 1.301))
ZERO_RULE = ((0.0, 0.0), (0.0, 1.0))
X_LABEL = "Standardized marginal effect"
Y_LABEL = "-log10(adjusted_p_value)"


@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    plt.close("all")


def _screen(n=40, seed=0):
    """A coefficient table with four real hits and a two-plate column."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 0.4, n)
    p = rng.uniform(0.2, 1.0, n)
    effect[:4] = [2.4, -2.1, 1.9, -1.7]
    p[:4] = [1e-6, 1e-5, 1e-4, 1e-4]
    return pd.DataFrame({
        "standardized_marginal_effect": effect,
        "adjusted_p_value": p,
        "guide": [f"g{i:03d}" for i in range(n)],
        "plate": ["p1" if i % 2 else "p2" for i in range(n)],
    })


def _scored(n=24, seed=3):
    """A coefficient table with three real hits and a numeric covariate."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0.0, 0.5, n)
    p = rng.uniform(0.2, 1.0, n)
    effect[:3] = [2.2, -2.0, 1.8]
    p[:3] = [1e-7, 1e-6, 1e-5]
    return pd.DataFrame({
        "standardized_marginal_effect": effect,
        "adjusted_p_value": p,
        "guide": [f"g{i:03d}" for i in range(n)],
        "score": np.linspace(0.0, 1.0, n),
    })


def _no_hits():
    frame = _screen()
    frame["adjusted_p_value"] = 0.9
    return frame


def _microneme_genes():
    genes = [gene for gene, place in table().items()
             if place == "micronemes"][:6]
    return pd.DataFrame({
        "gene": genes,
        "standardized_marginal_effect": np.linspace(-1.0, 1.0, len(genes)),
        "adjusted_p_value": [0.5] * len(genes),
    })


def _rules(axis):
    """Each drawn line as rounded ``((x0, x1), (y0, y1))``."""
    return [tuple(tuple(round(float(v), 4) for v in data)
                  for data in (line.get_xdata(), line.get_ydata()))
            for line in axis.lines]


def test_render_volcano_draws_the_series_rules_and_labels_its_style_asks_for(
        monkeypatch, subtests):
    """Empty groups and unmatched categories add no series; rules switch off; labels need their column."""
    nan_plates = np.array(["p1", "p2", np.nan, "p1"] * 10, dtype=object)
    effect_rules = [((-1.0, -1.0), (0.0, 1.0)), ((1.0, 1.0), (0.0, 1.0))]
    named_hits = dict(annotate_significant=True, effect_threshold=1.0,
                      threshold_multiplier=1.0)
    cases = [
        # id, frame, style, colour override, series, rules, texts, y label
        ("default", _screen, VolcanoStyle(), None,
         [("not significant", 36), ("significant", 4)],
         [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("no-hits-no-significant-series", _no_hits, VolcanoStyle(), None,
         [("not significant", 40)], [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("hits-named", _screen, VolcanoStyle(**named_hits), None,
         [("not significant", 36), ("significant", 4)],
         [ALPHA_RULE, ZERO_RULE] + effect_rules,
         ["g000", "g001", "g002", "g003"], Y_LABEL),
        ("label-column-missing", lambda: _screen().drop(columns=["guide"]),
         VolcanoStyle(annotate_significant=True), None,
         [("not significant", 36), ("significant", 4)],
         [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("colour-by-plate", _screen, VolcanoStyle(color_by="plate"), None,
         [("p2", 20), ("p1", 20)], [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("colour-category-matching-no-row", _screen,
         VolcanoStyle(color_by="plate"), (nan_plates, True),
         [("p1", 20), ("p2", 10)], [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("compartment-on-a-nameless-table",
         lambda: _scored().drop(columns=["guide"]),
         VolcanoStyle(localizations=("micronemes",)), None,
         [("not significant", 21), ("significant", 3)],
         [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("compartment-on-named-genes", _microneme_genes,
         VolcanoStyle(localizations=("micronemes",)), None,
         [("micronemes", 6)], [ALPHA_RULE, ZERO_RULE], [], Y_LABEL),
        ("reference-rules-off", _scored,
         VolcanoStyle(show_alpha_line=False, show_zero_line=False), None,
         [("not significant", 21), ("significant", 3)], [], [], Y_LABEL),
        ("named-y-label-wins", _scored,
         VolcanoStyle(y_label="Significance (-log10 q)"), None,
         [("not significant", 21), ("significant", 3)],
         [ALPHA_RULE, ZERO_RULE], [], "Significance (-log10 q)"),
    ]
    for case_id, frame, style, colours, series, rules, texts, y_label in \
            cases:
        with subtests.test(msg=case_id):
            data = frame()
            with monkeypatch.context() as patch:
                if colours is not None:
                    patch.setattr(vs, "_colour_values",
                                  lambda frame, style, c=colours: c)
                figure, panels = render_volcano(data, style)
            assert len(panels) == 1 and figure.axes == [panels[0]]
            axis = panels[0]
            assert [(c.get_label(), len(c.get_offsets()))
                    for c in axis.collections] == series
            assert _rules(axis) == rules
            assert sorted(t.get_text() for t in axis.texts) == texts
            assert axis.get_xlabel() == X_LABEL
            assert axis.get_ylabel() == y_label
            if case_id == "compartment-on-a-nameless-table":
                assert vs.point_localizations(data, style) is None
            plt.close(figure)


def test_render_volcano_applies_axis_options_ink_and_broken_axis_labels(
        subtests):
    """Scales, limits, inversion, grid, ground, title, fonts and ink are read back off the drawn axes."""
    with subtests.test(msg="every-axis-option"):
        figure, (axis,) = render_volcano(_screen(), VolcanoStyle(
            x_scale="symlog", x_lim=(-3.0, 3.0), y_lim=(0.0, 8.0),
            invert_y=True, grid=False, background_color="#F0F0F0",
            title="tsg101 screen", effect_threshold=1.0,
            threshold_multiplier=1.0, show_effect_lines=True))
        assert axis.get_xscale() == "symlog"
        assert axis.get_xlim() == (-3.0, 3.0)
        assert axis.get_ylim() == (8.0, 0.0)
        assert not axis.xaxis._major_tick_kw.get("gridOn", False)
        assert axis.get_facecolor() == to_rgba("#F0F0F0")
        assert axis.get_title() == "tsg101 screen"
        assert _rules(axis) == [ALPHA_RULE, ZERO_RULE,
                                ((-1.0, -1.0), (0.0, 1.0)),
                                ((1.0, 1.0), (0.0, 1.0))]

    with subtests.test(msg="axis-labels-own-font-size"):
        figure, (axis,) = render_volcano(
            _screen(), VolcanoStyle(font_size=31, label_font_size=17))
        assert (axis.xaxis.label.get_fontsize(),
                axis.yaxis.label.get_fontsize()) == (17.0, 17.0)

    for case_id, ink, expected in [
            ("axis-colour-paints-the-ink", "#FF0000", "#FF0000"),
            ("empty-axis-colour-keeps-ambient-ink", "", None)]:
        with subtests.test(msg=case_id):
            spine_ambient = to_rgba(matplotlib.rcParams["axes.edgecolor"])
            label_ambient = to_rgba(matplotlib.rcParams["axes.labelcolor"])
            figure, panels = render_volcano(_scored(), VolcanoStyle(
                color_by="score", show_colorbar=True, axis_color=ink))
            assert len(figure.axes) == 2
            bar = figure.axes[-1]
            assert bar.get_ylabel() == "score"
            assert np.allclose(panels[0].collections[0].get_array(),
                               np.linspace(0.0, 1.0, 24))
            spine = panels[0].spines["left"].get_edgecolor()
            label = to_rgba(bar.yaxis.label.get_color())
            if expected is None:
                assert (spine, label) == (spine_ambient, label_ambient)
            else:
                assert (spine, label) == (to_rgba(expected), to_rgba(expected))

    with subtests.test(msg="broken-axis-labels-follow-their-points"):
        frame = pd.DataFrame({
            "standardized_marginal_effect": [-1.0, 0.0, 1.0],
            "y": [8.0, 4.0, 0.5],
            "guide": ["hit", "gap", "low"],
        })
        figure, panels = render_volcano(frame, VolcanoStyle(
            y_column="y", y_neg_log10=False, split_axis=True,
            split_y_lims=((0.0, 2.0), (5.0, 10.0)),
            annotations={"hit": "HIT", "gap": "GAP", "low": "LOW"}))
        assert len(panels) == 2
        anchors = [[(t.get_text(), tuple(float(v) for v in t.xy))
                    for t in panel.texts] for panel in panels]
        # y=4 sits in the break, nearest the upper panel: clamped to its edge.
        assert anchors == [[("HIT", (-1.0, 8.0)), ("GAP", (0.0, 5.0))],
                           [("LOW", (1.0, 0.5))]]
        figure.canvas.draw()
        gap = panels[0].texts[1].get_window_extent(
            figure.canvas.get_renderer())
        upper = panels[0].get_window_extent()
        assert upper.y0 <= gap.y0 and gap.y1 <= upper.y1

    with subtests.test(msg="missing-y-column-is-named"):
        with pytest.raises(ValueError) as raised:
            render_volcano(_screen().drop(columns=["adjusted_p_value"]),
                           VolcanoStyle())
        assert str(raised.value) == (
            "y_column='adjusted_p_value' is not a column of the results "
            "(['guide', 'plate', 'standardized_marginal_effect']).")
