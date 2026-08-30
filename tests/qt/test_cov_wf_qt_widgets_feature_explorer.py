"""The feature explorer's edge paths: no table, no kept feature, no canvas.

Three moments the panel has to survive without a traceback and without
quietly showing the wrong thing:

* the table it was pointed at is taken away again (a file closed, a filter
  that emptied the source), so the class picker must empty and the panel must
  say it has nothing rather than offering the previous table's columns;
* a ranking comes back having kept nothing, so the list must clear instead of
  leaving the previous table's rows under a summary describing a new one;
* the panel is closed while a redraw is pending, on a canvas that does not
  own the deferred-draw timer the packaged one owns.

Every check drives the working case in the same test as the empty one, so an
"it is empty" assertion cannot pass by exercising nothing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.feature_rank import (
    AUC, ExplorerError, ExplorerResult, ExplorerSpec, FeatureScore,
    STATISTICS,
)


@pytest.fixture
def planted() -> pd.DataFrame:
    """Two features whose separation is worked out by hand.

    ``perfect`` puts every ``b`` above every ``a`` (AUC 1, separation 1.0);
    ``partial`` overlaps by two objects (AUC 14/16, separation 0.75). The
    ranking is therefore perfect, then partial, and a row order is an
    assertion rather than an accident of column order.
    """
    return pd.DataFrame({
        "cls": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "perfect": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "partial": [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0],
    })


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel

    widget = FeatureExplorerPanel()
    qtbot.addWidget(widget)
    return widget


def _score(feature: str, **over) -> FeatureScore:
    fields = dict(feature=feature, statistic=AUC, score=0.9, auc=0.95,
                  cohen_d=1.5, ks=0.6, mutual_info=0.3, higher_in="b",
                  against="a", n_by_class={"a": 4, "b": 4})
    fields.update(over)
    return FeatureScore(**fields)


def _result(scores, *, spec=None) -> ExplorerResult:
    spec = spec or ExplorerSpec(label="cls")
    return ExplorerResult(spec=spec, label="cls", classes=("a", "b"),
                          scores=tuple(scores), n_rows=8,
                          n_considered=len(scores))


# ---------------------------------------------------------------------------
# The table is taken away again
# ---------------------------------------------------------------------------

def test_taking_the_table_away_empties_the_class_picker_and_says_so(
        panel, planted):
    """A closed file must not leave its column names in the Split-by picker.

    The picker is the one control that decides what a separation is computed
    against. If a table is unloaded and its columns stay on offer, the next
    user action ranks a column that is not in any loaded table — the panel
    would either raise from inside the ranking or, worse, silently re-rank
    the table it was supposed to have forgotten. So: with a frame the picker
    offers that frame's class column and the summary describes the ranking;
    with the frame taken away the picker is empty and the panel says it has
    no table, and asking it to rank refuses instead of guessing.
    """
    panel.set_frame(planted)
    offered = [panel._label.itemText(i) for i in range(panel._label.count())]
    assert offered == ["cls"]
    assert panel._label.currentText() == "cls"
    assert "2 features over 8 objects" in panel.summary()
    assert [s.feature for s in panel.result.scores] == ["perfect", "partial"]

    panel.set_frame(None)

    assert panel._label.count() == 0
    assert panel._label.currentText() == ""
    assert panel.summary() == "no table loaded"
    assert panel.rank_now() is None
    # The statistic's blind spot is still on screen: the panel having no
    # table does not mean it stops saying what the ranking cannot see.
    assert "cannot see" in panel._blind.text()


def test_a_column_chosen_before_the_reload_is_chosen_again_after_it(
        panel, planted):
    """Reloading a table must not silently re-point the ranking.

    ``set_frame`` refills the picker from the new table, and a user who chose
    to split by ``plate`` and then reloaded would otherwise land back on
    whichever column sorts first — a different ranking under the same
    heading. The choice is restored when the new table still has that column,
    and only then; a column the new table lost cannot be kept, and the panel
    falls back to the first one it does have.
    """
    wider = planted.assign(plate=["p1", "p1", "p2", "p2"] * 2)
    panel.set_frame(wider)
    panel._label.setCurrentText("plate")
    # Changing a control only schedules the re-rank; the panel coalesces a
    # burst of them, so ask for the ranking the user is waiting on.
    assert panel.rank_now().label == "plate"

    panel.set_frame(wider.copy())
    assert panel._label.currentText() == "plate"
    assert panel.result.label == "plate"

    # The same reload against a table without that column cannot restore it.
    panel.set_frame(planted)
    assert panel._label.currentText() == "cls"
    assert panel.result.label == "cls"


# ---------------------------------------------------------------------------
# A ranking that kept nothing
# ---------------------------------------------------------------------------

def test_a_ranking_that_kept_nothing_clears_the_rows_and_selects_none(panel):
    """An empty ranking must not leave the previous ranking's rows selected.

    Filling the table also selects row 0, which emits ``feature_selected``
    and drives everything downstream — the plot beside it, the linked views
    on the screen. If a result with no kept features left the old rows in
    place, the panel would keep announcing a feature that is not in the
    current ranking, and every linked view would be showing a column the
    summary no longer mentions. So a filled table selects its best row and
    announces it, and an emptied one selects nothing and announces nothing.
    """
    announced: list = []
    panel.feature_selected.connect(announced.append)
    assert panel.table.rowCount() == 0          # nothing ranked yet

    panel._fill_table(_result([_score("perfect"), _score("partial",
                                                         score=0.75)]))
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "perfect"
    assert panel.table.item(0, 5).text() == "4"          # min n travels along
    assert panel.table.currentRow() == 0
    assert panel.selected_feature() == "perfect"
    assert announced[-1] == "perfect"
    filled_announcements = len(announced)

    panel._fill_table(_result([]))

    assert panel.table.rowCount() == 0
    assert panel.table.currentRow() == -1
    assert panel.selected_feature() == ""
    assert len(announced) == filled_announcements


def test_the_null_greys_the_rows_it_did_not_clear(panel, planted):
    """A feature that does not beat its own shuffled null must look different.

    The whole point of the shuffle test is that a separation of 0.6 over
    forty features is what forty noise features reach by chance. If the rows
    below the threshold were painted the same as the rows above it, the
    number would be computed, printed in the summary, and then thrown away by
    the one part of the screen the user actually reads.
    """
    from spacr.qt.theme import active_palette

    palette = active_palette()
    panel.set_frame(planted)
    result = ExplorerResult(
        spec=ExplorerSpec(label="cls"), label="cls", classes=("a", "b"),
        scores=(_score("perfect", score=0.9), _score("noise", score=0.2)),
        n_rows=8, n_considered=2, null_threshold=0.5)

    panel._fill_table(result)

    above = panel.table.item(0, 0).foreground().color().name()
    below = panel.table.item(1, 0).foreground().color().name()
    assert below.lower() == palette["fg_muted"].lower()
    assert above.lower() != below.lower()


def test_a_shape_only_feature_is_marked_even_though_it_ranks_low(panel,
                                                                 planted):
    """AUC's blind spot has to be visible on the row it applies to.

    A feature whose classes differ in spread rather than in level scores near
    0.5 and sinks to the bottom of an AUC ranking. Marking it is the only
    thing standing between the user and a real difference that the chosen
    statistic is structurally unable to rank.
    """
    from spacr.qt.theme import active_palette

    palette = active_palette()
    panel.set_frame(planted)

    # AUC exactly at the coin flip with the CDFs half a step apart: the
    # module's own definition of "shape, not shift".
    shape = _score("shape", score=0.0, auc=0.5, ks=0.5)
    assert shape.is_shape_not_shift
    panel._fill_table(_result([shape, _score("plain", score=0.4)]))

    marked = panel.table.item(0, 0)
    plain = panel.table.item(1, 0)
    assert marked.foreground().color().name().lower() == \
        palette["warning"].lower()
    assert plain.foreground().color().name().lower() != \
        palette["warning"].lower()
    assert "shape differs without a location shift" in marked.toolTip()


def test_a_feature_with_no_finite_value_is_listed_but_not_drawn(panel):
    """A column of NaN has no bin edges, so drawing it would be a lie.

    The strip would come out as an empty axis under a title asserting a
    score, which is the one thing worse than not drawing it: the table still
    lists the feature, so nothing is hidden, but the picture only shows what
    the data can support.
    """
    frame = pd.DataFrame({"cls": ["a", "b"] * 4,
                          "area": np.arange(8.0),
                          "blank": [np.nan] * 8})
    panel.set_frame(frame)
    spec = ExplorerSpec(label="cls")
    panel._spec = spec

    panel._draw(_result([_score("area"), _score("blank")], spec=spec))

    axes = panel._figure.get_axes()
    assert len(axes) == 2
    assert axes[0].patches and "area" in axes[0].get_title(loc="left")
    assert not axes[1].patches
    assert axes[1].get_title(loc="left") == ""


# ---------------------------------------------------------------------------
# Closing while a draw is pending
# ---------------------------------------------------------------------------

class _PlainCanvas:
    """A canvas with no owned draw timer — nothing to cancel."""


class _CancellingCanvas(_PlainCanvas):
    def __init__(self, log):
        self._log = log

    def cancel_pending_draw(self):
        self._log.append("cancelled")


def test_closing_cancels_a_pending_draw_and_survives_a_canvas_that_cannot(
        qtbot, planted):
    """Closing must stop the debounce whether or not the canvas can help.

    The packaged canvas owns its deferred-draw timer so it can be cancelled
    on close; an unowned timer firing after Qt has deleted the widget is a
    segfault, which is why the cancel exists at all. But the panel must not
    require it: a canvas swapped in by an embedding screen, or a plain
    matplotlib canvas, has no such method, and ``close()`` raising
    ``AttributeError`` on the way out would take the window down with it.
    Both cases stop the re-ranking timer, which is the panel's own pending
    work and the thing that would otherwise re-rank a closed panel.
    """
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel

    log: list = []

    able = FeatureExplorerPanel()
    qtbot.addWidget(able)
    able.set_frame(planted)
    able._canvas = _CancellingCanvas(log)
    able._schedule()
    assert able._debounce.isActive()
    assert able.close() is True
    assert log == ["cancelled"]
    assert not able._debounce.isActive()

    unable = FeatureExplorerPanel()
    qtbot.addWidget(unable)
    unable.set_frame(planted)
    unable._canvas = _PlainCanvas()
    unable._schedule()
    assert unable._debounce.isActive()
    assert unable.close() is True
    # Nothing new to cancel, and the panel's own pending work stopped anyway.
    assert log == ["cancelled"]
    assert not unable._debounce.isActive()


def test_the_debounce_coalesces_a_burst_of_control_changes(panel, planted,
                                                           qtbot):
    """Re-ranking on every keystroke would re-read the whole table each time.

    ``Top`` is a spin box: holding its arrow walks it through a dozen values,
    and each one is a full pass over every continuous column. The debounce
    turns that burst into one ranking — the last one — which is why the timer
    is single-shot and why ``rank_now`` stops it before ranking.
    """
    from spacr.qt.widgets.feature_explorer import DEBOUNCE_MS

    panel.set_frame(planted)
    seen: list = []
    panel.ranked.connect(lambda result: seen.append(result.spec.top))
    seen.clear()

    for value in (5, 4, 3, 2, 1):
        panel._top.setValue(value)
    assert panel._debounce.isActive()
    assert seen == []

    qtbot.wait(DEBOUNCE_MS * 3)

    assert seen == [1]
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "perfect"


def test_the_statistic_picker_offers_exactly_the_known_statistics(panel):
    """A spec can only name a statistic the picker carries, and vice versa.

    ``set_spec`` restores a saved analysis by looking its statistic up in the
    picker. The lookup only ever succeeds because both sides are built from
    the same ``STATISTICS`` tuple — the picker in the constructor, the spec
    in its own validation — so a statistic added to one and not the other
    would silently restore the wrong ranking for every saved analysis.
    """
    carried = [panel._statistic.itemData(i)
               for i in range(panel._statistic.count())]
    assert tuple(carried) == tuple(STATISTICS)
    for statistic in STATISTICS:
        panel.set_spec(ExplorerSpec(label="cls", statistic=statistic, top=7))
        assert panel._statistic.currentData() == statistic
        assert panel._top.value() == 7
    # And the other direction: a statistic the picker does not carry cannot
    # be written into a spec in the first place, which is why the lookup in
    # `set_spec` never comes back empty.
    with pytest.raises(ExplorerError, match="unknown separation statistic"):
        ExplorerSpec(label="cls", statistic="ttest")
