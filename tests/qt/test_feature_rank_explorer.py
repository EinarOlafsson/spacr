"""Feature ranking scores and the Feature Explorer panel that shows them.

The ranking must keep "cannot be computed" (NaN) apart from "explains nothing"
(0.0), print only the statistics that are finite, and let a sparsely measured
column honestly make the shuffled null unbeatable.  The explorer must draw
only features that have finite values, restore the user's split column across
a reload, mark the rows the null did not clear and the shape-only rows the
statistic cannot rank, coalesce a burst of control changes into one ranking,
and close cleanly whether or not its canvas can cancel a pending draw.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every Qt case builds its own panel.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets import feature_rank as FR  # noqa: E402
from spacr.qt.widgets.feature_rank import (  # noqa: E402
    AUC,
    ExplorerResult,
    ExplorerSpec,
    FeatureScore,
    mutual_info_of,
    rank_features,
)

pytestmark = pytest.mark.qt


@pytest.fixture
def two_conditions() -> pd.DataFrame:
    """Sixteen objects, eight per condition; ``cell_area`` separates fully."""
    return pd.DataFrame({
        "condition": ["ctrl"] * 8 + ["trt"] * 8,
        "cell_area": [100.0 + 10.0 * i for i in range(8)]
                     + [200.0 + 10.0 * i for i in range(8)],
    })


@pytest.fixture
def planted() -> pd.DataFrame:
    """``perfect`` (separation 1.0) then ``partial`` (0.75), by hand."""
    return pd.DataFrame({
        "cls": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "perfect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "partial": [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0],
    })


@pytest.fixture
def make_panel(qtbot):
    """A fresh Feature Explorer per case."""
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel

    def make():
        widget = FeatureExplorerPanel()
        qtbot.addWidget(widget)
        return widget
    return make


def _score(feature: str, **over) -> FeatureScore:
    fields = dict(feature=feature, statistic=AUC, score=0.9, auc=0.95,
                  cohen_d=1.5, ks=0.6, mutual_info=0.3, higher_in="b",
                  against="a", n_by_class={"a": 4, "b": 4})
    fields.update(over)
    return FeatureScore(**fields)


def _result(scores, *, spec=None, label="cls", **extra) -> ExplorerResult:
    spec = spec or ExplorerSpec(label=label)
    return ExplorerResult(spec=spec, label=label, classes=("a", "b"),
                          scores=tuple(scores), n_rows=8,
                          n_considered=len(scores), **extra)


def test_mutual_info_of_keeps_not_measurable_apart_from_explains_nothing(
        monkeypatch, subtests):
    """No counts is NaN, a one-class split is 0.0, a real split is in (0, 1]."""
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.5, 1.5, 2.5, 3.5])

    def no_counts(values, bins=None, **kw):
        return np.zeros(len(bins) - 1, dtype=int), bins

    def one_sided():
        state = {"first": True}

        def histogram(values, bins=None, **kw):
            counts = np.zeros(len(bins) - 1, dtype=int)
            if state["first"]:
                counts[0] = 4
                state["first"] = False
            return counts, bins
        return histogram

    with subtests.test(msg="a binning that captures nothing is NaN"):
        with monkeypatch.context() as patch:
            patch.setattr(FR.np, "histogram", no_counts)
            assert np.isnan(mutual_info_of(a, b, bins=4))

    with subtests.test(msg="a class split with no entropy explains nothing"):
        with monkeypatch.context() as patch:
            patch.setattr(FR.np, "histogram", one_sided())
            assert mutual_info_of(a, b, bins=4) == 0.0

    with subtests.test(msg="two separated classes explain everything"):
        assert mutual_info_of(np.zeros(50), np.ones(50), bins=4) == \
            pytest.approx(1.0)

    with subtests.test(msg="identical classes explain nothing"):
        same = np.arange(50.0)
        assert mutual_info_of(same, same.copy(), bins=4) == pytest.approx(0.0)

    with subtests.test(msg="an empty group is NaN, one pooled value is 0.0"):
        assert np.isnan(mutual_info_of(np.array([]), b, bins=4))
        assert np.isnan(mutual_info_of(a, np.array([]), bins=4))
        assert mutual_info_of(np.full(5, 2.0), np.full(5, 2.0), bins=4) == 0.0


def test_rank_features_captions_and_null_say_only_what_was_measured(
        two_conditions, subtests):
    """Captions print finite statistics only; a sparse column sets the null to 1."""
    real = rank_features(two_conditions,
                         ExplorerSpec(label="condition")).score_for("cell_area")
    nan = float("nan")
    shape_tail = ("distributional shape differs without a location shift; a "
                  "rank statistic does not detect this pattern")
    cases = [
        ("nothing finite", dict(score=nan, auc=nan, cohen_d=nan, ks=nan,
                                mutual_info=nan), "cell_area: nan"),
        ("only a spread statistic", dict(score=0.05, auc=nan, cohen_d=nan,
                                         ks=0.42),
         "cell_area: 0.050 · KS 0.420"),
        ("AUC and KS, shape not shift", dict(score=0.05, auc=0.52,
                                             cohen_d=nan, ks=0.42),
         "cell_area: 0.050 · AUC 0.520, higher in trt · KS 0.420 · "
         + shape_tail),
    ]
    with subtests.test(msg="a real row says all three statistics"):
        said = real.describe()
        assert said.startswith("cell_area: 1.000")
        assert "AUC 1.000, higher in trt" in said
        assert "KS 1.000" in said
        assert "d +" in said
    for case_id, over, expected in cases:
        with subtests.test(msg=case_id):
            assert replace(real, **over).describe() == expected

    with subtests.test(msg="a column measured on two objects beats the null"):
        spec = ExplorerSpec(label="condition", n_permutations=12, seed=3)
        alone = rank_features(two_conditions, spec)
        assert alone.null_threshold is not None
        assert alone.null_threshold < 1.0
        assert [s.feature for s in alone.above_null()] == ["cell_area"]
        frame = two_conditions.copy()
        sparse = [np.nan] * 16
        sparse[0], sparse[15] = 5.0, 9.0
        frame["rare_stain"] = sparse
        result = rank_features(frame, spec)
        assert result.n_considered == 2
        assert result.skipped == {}
        assert result.null_threshold == 1.0
        assert result.above_null() == ()
        assert "shuffling the labels reaches 1.000" in result.summary()
        assert "0 feature(s) beat it" in result.summary()


def test_the_explorer_draws_only_features_that_have_finite_values(
        make_panel, subtests):
    """One strip per kept feature; an all-NaN column gets an empty, untitled one."""
    frame = pd.DataFrame({"label": ["a", "b"] * 4,
                          "area": np.arange(8.0),
                          "blank": [np.nan] * 8})
    cases = [
        # id, features ranked, expected (has bars, title contains) per strip
        ("an empty ranking clears the canvas", (), []),
        ("a feature with values is drawn", ("area",), [(True, "area")]),
        ("a feature with no finite value is left undrawn", ("blank",),
         [(False, None)]),
        ("drawn and undrawn side by side", ("area", "blank"),
         [(True, "area"), (False, None)]),
    ]
    for case_id, features, strips in cases:
        with subtests.test(msg=case_id):
            panel = make_panel()
            panel.set_frame(frame)
            spec = ExplorerSpec(label="label")
            panel._spec = spec
            panel._draw(_result([_score(f) for f in features], spec=spec,
                                label="label"))
            axes = panel._figure.get_axes()
            assert len(axes) == len(strips)
            for ax, (bars, title) in zip(axes, strips):
                assert bool(ax.patches) is bars
                if title is None:
                    assert ax.get_title(loc="left") == ""
                else:
                    assert title in ax.get_title(loc="left")

    with subtests.test(msg="a table whose only feature has no finite value"):
        panel = make_panel()
        panel.set_frame(pd.DataFrame({"label": ["a", "b"] * 4,
                                      "blank": [np.nan] * 8}))
        assert panel.result is None
        assert panel.summary().startswith(
            "none of the 1 features could be scored against 'label'. blank: ")
        spec = ExplorerSpec(label="label")
        panel._spec = spec
        panel._draw(_result([_score("blank")], spec=spec, label="label"))
        axes = panel._figure.get_axes()
        assert len(axes) == 1
        assert not axes[0].patches
        assert axes[0].get_title(loc="left") == ""

    with subtests.test(msg="the panel reports the spec its controls hold"):
        panel = make_panel()
        panel.set_spec(ExplorerSpec(label="label", top=3, bins=8))
        assert panel.spec.top == 3
        assert panel.spec.bins == 8


class _PlainCanvas:
    """A canvas with no owned draw timer: nothing to cancel."""


class _CancellingCanvas(_PlainCanvas):
    def __init__(self, log):
        self._log = log

    def cancel_pending_draw(self):
        self._log.append("cancelled")


def test_the_explorer_panel_keeps_its_rows_honest(
        make_panel, planted, qtbot, subtests):
    """Choices survive reloads, rows are painted by what they mean, work is coalesced."""
    from spacr.qt.theme import active_palette
    from spacr.qt.widgets.feature_explorer import DEBOUNCE_MS

    palette = active_palette()

    with subtests.test(msg="a split column chosen before a reload is kept"):
        panel = make_panel()
        wider = planted.assign(plate=["p1", "p1", "p2", "p2"] * 2)
        panel.set_frame(wider)
        panel._label.setCurrentText("plate")
        assert panel.rank_now().label == "plate"
        panel.set_frame(wider.copy())
        assert panel._label.currentText() == "plate"
        assert panel.result.label == "plate"
        panel.set_frame(planted)
        assert panel._label.currentText() == "cls"
        assert panel.result.label == "cls"

    with subtests.test(msg="rows the null did not clear are greyed"):
        panel = make_panel()
        panel.set_frame(planted)
        panel._fill_table(_result(
            [_score("perfect", score=0.9), _score("noise", score=0.2)],
            null_threshold=0.5))
        above = panel.table.item(0, 0).foreground().color().name().lower()
        below = panel.table.item(1, 0).foreground().color().name().lower()
        assert below == palette["fg_muted"].lower()
        assert above != below

    with subtests.test(msg="a shape-only feature is marked though it ranks low"):
        panel = make_panel()
        panel.set_frame(planted)
        shape = _score("shape", score=0.0, auc=0.5, ks=0.5)
        assert shape.is_shape_not_shift
        panel._fill_table(_result([shape, _score("plain", score=0.4)]))
        marked = panel.table.item(0, 0)
        plain = panel.table.item(1, 0)
        warning = palette["warning"].lower()
        assert marked.foreground().color().name().lower() == warning
        assert plain.foreground().color().name().lower() != warning
        assert "shape differs without a location shift" in marked.toolTip()

    with subtests.test(msg="closing cancels a pending draw, either canvas"):
        log: list = []
        for canvas, expected_log in ((_CancellingCanvas(log), ["cancelled"]),
                                     (_PlainCanvas(), ["cancelled"])):
            panel = make_panel()
            panel.set_frame(planted)
            panel._canvas = canvas
            panel._schedule()
            assert panel._debounce.isActive()
            assert panel.close() is True
            assert log == expected_log
            assert not panel._debounce.isActive()

    with subtests.test(msg="a burst of control changes ranks once, the last"):
        panel = make_panel()
        panel.set_frame(planted)
        seen: list = []
        panel.ranked.connect(lambda result: seen.append(result.spec.top))
        for value in (5, 4, 3, 2, 1):
            panel._top.setValue(value)
        assert panel._debounce.isActive()
        assert seen == []
        qtbot.wait(DEBOUNCE_MS * 3)
        assert seen == [1]
        assert panel.table.rowCount() == 1
        assert panel.table.item(0, 0).text() == "perfect"
