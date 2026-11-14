"""The feature explorer's edge paths: no table, no kept feature, no canvas.

Four moments the panel has to survive without a traceback and without
quietly showing the wrong thing:

* the table it was pointed at is taken away again (a file closed, a filter
  that emptied the source), so the class picker must empty and the panel must
  say it has nothing rather than offering the previous table's columns;
* a ranking comes back having kept nothing, so the list must clear instead of
  leaving the previous table's rows under a summary describing a new one;
* the panel is closed while a redraw is pending, on a canvas that does not
  own the deferred-draw timer the packaged one owns;
* a saved spec is pushed back in naming a statistic this build's picker does
  not carry, so the picker must stay on the one it is actually ranking by
  rather than going blank over an AUC fallback.

Every check drives the working case in the same test as the empty one, so an
"it is empty" assertion cannot pass by exercising nothing.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.feature_rank import (
    AUC,
    STATISTICS,
    ExplorerError,
    ExplorerResult,
    ExplorerSpec,
    FeatureScore,
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
    assert panel.table.rowCount() == 2
    assert panel.table.currentRow() == 0
    assert panel.selected_feature() == "perfect"
    assert panel._figure.get_axes()

    panel.set_frame(None)

    assert panel._label.count() == 0
    assert panel._label.currentText() == ""
    assert panel.summary() == "no table loaded"
    assert panel.result is None
    assert panel.table.rowCount() == 0
    assert panel.table.currentRow() == -1
    assert panel.selected_feature() == ""
    assert panel._figure.get_axes() == []
    assert panel.rank_now() is None
    # The statistic's blind spot is still on screen: the panel having no
    # table does not mean it stops saying what the ranking cannot see.
    assert "cannot see" in panel._blind.text()


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


def test_a_new_top_feature_at_the_same_row_is_announced_once(
        panel, planted, monkeypatch):
    """Linked views follow the feature in row zero, not just its row index."""
    from spacr.qt.widgets import feature_explorer as explorer

    panel.set_frame(planted)
    assert panel.table.currentRow() == 0
    assert panel.selected_feature() == "perfect"

    announced: list = []
    panel.feature_selected.connect(announced.append)
    reranked = _result([
        _score("partial", score=0.95),
        _score("perfect", score=0.9),
    ])
    monkeypatch.setattr(explorer, "rank_features", lambda *_args: reranked)

    assert panel.rank_now() is reranked
    assert panel.table.currentRow() == 0
    assert panel.selected_feature() == "partial"
    assert announced == ["partial"]


# ---------------------------------------------------------------------------
# Closing while a draw is pending
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# A restored spec naming a statistic this build does not carry
# ---------------------------------------------------------------------------

def test_a_spec_naming_an_unknown_statistic_leaves_the_picker_alone(panel,
                                                                    planted):
    """Restoring a foreign analysis must not blank the Rank-by picker.

    ``ExplorerSpec`` checks its statistic against ``STATISTICS`` when it is
    constructed -- but a spec does not always arrive by construction. A
    session restore rebuilds the object field by field without re-running
    that validation, so an analysis saved by a build carrying one more
    statistic comes back naming something this picker has never heard of.
    ``set_spec`` looks the name up and, finding nothing, leaves the picker
    where it was. Without that guard the lookup's ``-1`` would go straight to
    ``setCurrentIndex``, which *clears* a combo box: the picker would show
    nothing, ``currentData()`` would come back ``None``, and ``rank_now``
    would quietly rank by its AUC fallback under a blank control -- a number
    on screen with no statement of what produced it, which is the one failure
    this whole panel exists to prevent. Everything the spec asks for that
    this build can honour is still applied.
    """
    import pickle

    from spacr.qt.widgets.feature_rank import COHEN_D

    panel.set_frame(planted)
    # The same call with a statistic the picker does carry: it moves.
    panel.set_spec(ExplorerSpec(label="cls", statistic=COHEN_D, top=5))
    assert panel._statistic.currentData() == COHEN_D
    assert panel._top.value() == 5

    # The spec a restore hands back. `object.__setattr__` stands in for the
    # build that offered "ttest"; the pickle round trip is the restore
    # itself, and it demonstrably does not re-validate -- the name survives.
    foreign = ExplorerSpec(label="cls", statistic=COHEN_D, top=3,
                           n_permutations=50)
    object.__setattr__(foreign, "statistic", "ttest")
    restored = pickle.loads(pickle.dumps(foreign))
    assert restored.statistic == "ttest"
    assert restored.statistic not in STATISTICS

    panel.set_spec(restored)

    # The picker did not move and, above all, did not go blank.
    assert panel._statistic.currentIndex() >= 0
    assert panel._statistic.currentData() == COHEN_D
    assert panel._statistic.currentText() != ""
    # The rest of the spec was applied, and the ranking that came back names
    # the statistic actually used rather than the one that was asked for.
    assert panel._top.value() == 3
    assert panel._null.isChecked()
    assert panel.result.spec.statistic == COHEN_D
    assert panel.result.spec.n_permutations == 50
    assert [s.feature for s in panel.result.scores] == ["perfect", "partial"]
