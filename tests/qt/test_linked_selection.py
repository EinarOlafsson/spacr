"""Tests for the process-wide linked selection.

The behaviours worth pinning are the ones that would make two open views
disagree about what the user is looking at, the one that would make a lasso
destructive, and — for the routing half — the ones that would make a click
silently do nothing.
"""
from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtWidgets import QWidget

from spacr.qt import linked_selection as linked_selection_module
from spacr.qt.linked_selection import (
    DEFAULT_OPEN_KIND,
    LinkedSelection,
    LinkedView,
    NoObjectOpener,
    has_object_opener,
    linked_selection,
    object_opener_kinds,
    open_objects,
    open_request,
    register_object_opener,
    unregister_object_opener,
)
from spacr.selection import (
    CategoryFilter,
    DataFilter,
    ObjectRequest,
    RangeFilter,
    Selection,
)


@pytest.fixture
def link() -> LinkedSelection:
    """A fresh instance — never the process-wide one, which other tests share."""
    return LinkedSelection()


def _frame(n: int = 6, *, timelapse: bool = False) -> pd.DataFrame:
    df = pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{i % 2 + 1}" for i in range(n)],
        "columnID": [f"c{i % 3 + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": [10.0 * (i + 1) for i in range(n)],
    })
    if timelapse:
        df["timeID"] = list(range(1, n + 1))
    return df


def test_the_accessor_is_a_singleton():
    assert linked_selection() is linked_selection()


def test_it_starts_with_no_filter_and_no_selection(link):
    assert link.filter.is_empty
    assert not link.selection.is_active


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_setting_a_filter_emits_only_filter_changed(link, qtbot):
    """The two signals cost different amounts to honour, so they stay apart.

    A filter change makes a view re-query and re-lay-out; a selection change
    usually only repaints. One combined signal would make every lasso reload a
    million-row table.
    """
    seen = {"filter": 0, "selection": 0}
    link.filter_changed.connect(lambda: seen.__setitem__("filter",
                                                         seen["filter"] + 1))
    link.selection_changed.connect(
        lambda: seen.__setitem__("selection", seen["selection"] + 1))

    link.set_filter(DataFilter().add(RangeFilter("area", low=20.0)))
    assert seen == {"filter": 1, "selection": 0}


def test_setting_a_selection_emits_only_selection_changed(link):
    seen = {"filter": 0, "selection": 0}
    link.filter_changed.connect(lambda: seen.__setitem__("filter",
                                                         seen["filter"] + 1))
    link.selection_changed.connect(
        lambda: seen.__setitem__("selection", seen["selection"] + 1))

    link.select_frame(_frame(3), source="umap")
    assert seen == {"filter": 0, "selection": 1}


def test_an_identical_looking_filter_still_emits(link):
    """No equality short-circuit, deliberately.

    A caller that mutated a DataFilter in place and handed the same object
    back would compare equal to itself and emit nothing, leaving views showing
    a population that no longer matches the controls.
    """
    f = DataFilter().add(RangeFilter("area", low=20.0))
    link.set_filter(f)

    fired = []
    link.filter_changed.connect(lambda: fired.append(1))
    f.add(RangeFilter("area", low=50.0))     # mutated in place
    link.set_filter(f)                        # same object back
    assert fired == [1], "a re-set must emit even when the object is unchanged"


def test_clearing_emits_too(link):
    link.set_filter(DataFilter().add(RangeFilter("area", low=20.0)))
    link.select_frame(_frame(2), source="plate")

    fired = {"filter": 0, "selection": 0}
    link.filter_changed.connect(
        lambda: fired.__setitem__("filter", fired["filter"] + 1))
    link.selection_changed.connect(
        lambda: fired.__setitem__("selection", fired["selection"] + 1))

    link.clear_filter()
    link.clear_selection()
    assert fired == {"filter": 1, "selection": 1}
    assert link.filter.is_empty
    assert not link.selection.is_active


# ---------------------------------------------------------------------------
# What a view actually asks for
# ---------------------------------------------------------------------------

def test_visible_applies_the_filter(link):
    df = _frame(6)
    link.set_filter(DataFilter().add(RangeFilter("area", low=40.0)))
    out = link.visible(df)
    assert len(out) == 3
    assert (out["area"] >= 40.0).all()


def test_visible_does_not_apply_the_selection(link):
    """A selection HIGHLIGHTS; it must never hide.

    A view that dropped unselected rows would make the lasso destructive — you
    could not see what you had excluded, or undo it by lassoing more.
    """
    df = _frame(6)
    link.select_frame(df.iloc[[0]], source="umap")
    assert len(link.visible(df)) == 6


def test_a_filter_and_a_selection_compose_without_interfering(link):
    df = _frame(6)
    link.set_filter(DataFilter().add(CategoryFilter("rowID", ("r1",))))
    link.select_frame(df.iloc[[0, 1]], source="umap")

    shown = link.visible(df)
    assert set(shown["rowID"]) == {"r1"}
    # The selection still resolves against the narrowed frame.
    assert link.selection.mask_for(shown).sum() == 1


def test_the_selection_carries_its_source(link):
    """So a view can ignore the echo of its own selection.

    Without it every lasso costs a repaint in the view that drew it, and a
    view that normalises what it publishes can loop.
    """
    link.select_frame(_frame(2), source="plate_view")
    assert link.selection.source == "plate_view"


def test_set_selection_accepts_a_prebuilt_selection(link):
    sel = Selection.from_frame(_frame(3), source="db_browser")
    link.set_selection(sel)
    assert link.selection is sel


# ---------------------------------------------------------------------------
# Routing: "show me exactly these objects"
# ---------------------------------------------------------------------------

class _Recorder:
    """An opener that records what it was handed and reports what it opened."""

    def __init__(self, result="opened"):
        self.result = result
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.result


@pytest.fixture
def process_link(monkeypatch) -> LinkedSelection:
    """Stand a fresh instance in as the process-wide one.

    The module-level routing functions go through `linked_selection()`, and a
    test that registered an opener on the real one would leak it into every
    later test in the session.
    """
    fresh = LinkedSelection()
    monkeypatch.setattr(linked_selection_module, "_LINKED", fresh)
    return fresh


def test_an_opener_receives_the_keys_the_caller_named(link):
    """The whole seam: a caller hands rows, the destination gets object keys.

    The opener is reached without either side importing the other, and it may
    assume normalised keys — it never sees the caller's DataFrame.
    """
    opener = _Recorder(result="annotate-screen")
    link.register_object_opener("annotate", opener)

    df = _frame(4)
    returned = link.open_objects(df.iloc[[2, 0]], reason="misclassified",
                                 source="classifier_evaluation")

    assert returned == "annotate-screen", "the opener's return is passed back"
    request, = opener.requests
    assert list(request.keys) == ["p1_r1_c3_f1_3", "p1_r1_c1_f1_1"], \
        "the caller's order survives the trip — worst errors first"
    assert request.reason == "misclassified"
    assert request.source == "classifier_evaluation"
    assert request.kind == "annotate"


def test_an_unregistered_kind_raises_rather_than_doing_nothing(link):
    """A silent no-op here is a button that looks broken, not a missing screen."""
    with pytest.raises(NoObjectOpener, match="annotate"):
        link.open_objects(_frame(1), reason="clicked")


def test_the_error_names_what_is_registered_so_a_typo_is_visible(link):
    link.register_object_opener("annotate", _Recorder())
    with pytest.raises(NoObjectOpener, match="registered: annotate"):
        link.open_objects(_frame(1), reason="clicked", kind="annotote")


def test_a_caller_can_ask_first_instead_of_catching(link):
    """So an unavailable destination is a greyed menu entry, not an exception."""
    assert not link.has_object_opener("annotate")
    link.register_object_opener("annotate", _Recorder())
    assert link.has_object_opener("annotate")
    assert link.object_opener_kinds() == ("annotate",)


def test_opening_announces_itself_only_after_it_worked(link):
    """Other views follow `objects_opened`; they must not chase a failed jump."""
    seen = []
    link.objects_opened.connect(seen.append)

    def _explode(request):
        raise RuntimeError("no crops on disk")

    link.register_object_opener("annotate", _explode)
    with pytest.raises(RuntimeError, match="no crops"):
        link.open_objects(_frame(1), reason="clicked")
    assert seen == [], "a failed open must not be announced as an open"

    link.register_object_opener("annotate", _Recorder())
    link.open_objects(_frame(1), reason="clicked")
    assert len(seen) == 1 and seen[0].reason == "clicked"


def test_opening_objects_does_not_move_the_shared_selection(link):
    """Opening a subset must not wipe the lasso it was opened from.

    Highlighting them everywhere is a separate act, available to the receiver
    as `request.as_selection()`.
    """
    link.register_object_opener("annotate", _Recorder())
    link.select_frame(_frame(6), source="umap")
    before = link.selection

    link.open_objects(_frame(1), reason="clicked", source="umap")
    assert link.selection is before
    assert len(link.selection) == 6


def test_registering_the_same_kind_twice_replaces_and_hands_back(link):
    """Re-opening a screen makes the new one the live destination."""
    first, second = _Recorder("first"), _Recorder("second")
    assert link.register_object_opener("annotate", first) is None
    assert link.register_object_opener("annotate", second) is first
    assert link.open_objects(_frame(1), reason="clicked") == "second"


def test_a_closing_screen_cannot_unregister_the_one_that_replaced_it(link):
    """The first screen's closeEvent runs after the second has registered.

    An unconditional withdrawal there leaves the live screen unreachable —
    the click stops working and nothing says why.
    """
    first, second = _Recorder("first"), _Recorder("second")
    link.register_object_opener("annotate", first)
    link.register_object_opener("annotate", second)

    assert link.unregister_object_opener("annotate", first) is False
    assert link.open_objects(_frame(1), reason="clicked") == "second"
    assert link.unregister_object_opener("annotate", second) is True
    assert not link.has_object_opener("annotate")


def test_unregistering_something_that_was_never_there_is_false_not_an_error(link):
    assert link.unregister_object_opener("annotate") is False


def test_unregistering_without_naming_the_opener_still_works(link):
    link.register_object_opener("annotate", _Recorder())
    assert link.unregister_object_opener("annotate") is True


def test_a_bad_registration_is_refused_where_it_is_made(link):
    """Not at the click three screens later that would have used it."""
    with pytest.raises(ValueError, match="non-blank kind"):
        link.register_object_opener("  ", _Recorder())
    with pytest.raises(TypeError, match="not callable"):
        link.register_object_opener("annotate", "not a function")


def test_a_prebuilt_request_can_be_routed_later(link):
    """Built where the data is, off the event loop; routed on the GUI thread."""
    opener = _Recorder()
    link.register_object_opener(DEFAULT_OPEN_KIND, opener)

    request = ObjectRequest(keys=_frame(2), reason="built in a worker")
    link.open_request(request)

    routed, = opener.requests
    assert routed.kind == DEFAULT_OPEN_KIND, \
        "the router stamps the kind it dispatched to"
    assert list(routed.keys) == list(request.keys)
    assert routed.reason == "built in a worker"


def test_an_empty_request_still_reaches_the_destination(link):
    """A confusion-matrix cell with no errors is an answer, not an exception."""
    opener = _Recorder()
    link.register_object_opener("annotate", opener)
    link.open_objects([], reason="no errors in this cell")
    assert len(opener.requests[0]) == 0


def test_context_reaches_the_destination(link):
    """Per-key scores are how "high-confidence errors first" gets drawn."""
    opener = _Recorder()
    link.register_object_opener("annotate", opener)
    link.open_objects(["p1_r1_c1_f1_1"], reason="worst",
                      context={"scores": {"p1_r1_c1_f1_1": 0.98}})
    assert opener.requests[0].context["scores"]["p1_r1_c1_f1_1"] == 0.98


def test_the_module_level_functions_route_through_the_process_wide_link(
        process_link):
    """What the other screens actually import, end to end."""
    opener = _Recorder("annotate-screen")
    assert register_object_opener("annotate", opener) is None
    assert has_object_opener("annotate")
    assert object_opener_kinds() == ("annotate",)

    assert open_objects(_frame(1), reason="clicked", source="umap") == \
        "annotate-screen"
    open_request(ObjectRequest(keys=_frame(1), reason="second"))
    assert [r.reason for r in opener.requests] == ["clicked", "second"]

    assert unregister_object_opener("annotate", opener) is True
    assert not has_object_opener("annotate")
    with pytest.raises(NoObjectOpener):
        open_objects(_frame(1), reason="after the screen closed")


# ---------------------------------------------------------------------------
# LinkedView — the three lines a view writes to join in
# ---------------------------------------------------------------------------

class _View(LinkedView, QWidget):
    """A view that opts in and records what it was told."""

    def __init__(self, link, source, *, echo=False):
        super().__init__()
        self.filters = []
        self.selections = []
        self.link_selection(source, link=link, echo=echo)

    def on_linked_filter_changed(self, data_filter):
        self.filters.append(data_filter)

    def on_linked_selection_changed(self, selection):
        self.selections.append(selection)


class _SilentView(LinkedView, QWidget):
    """A view that overrides nothing — the default hooks must not crash."""


def test_a_view_hears_the_shared_filter_and_gets_handed_it(link, qtbot):
    view = _View(link, "umap")
    qtbot.addWidget(view)

    data_filter = DataFilter().add(RangeFilter("area", low=40.0))
    link.set_filter(data_filter)

    assert view.filters == [data_filter], \
        "the hook is handed the filter, so the view need not go and fetch it"


def test_a_view_hears_other_views_selections(link, qtbot):
    view = _View(link, "plate_view")
    qtbot.addWidget(view)

    link.select_frame(_frame(2), source="umap")

    selection, = view.selections
    assert selection.source == "umap"
    assert list(selection.keys) == ["p1_r1_c1_f1_1", "p1_r2_c2_f1_2"]


def test_a_view_does_not_hear_the_echo_of_its_own_selection(link, qtbot):
    """Otherwise every lasso costs the drawing view a repaint of its own work,
    and a view that normalises what it publishes oscillates."""
    view = _View(link, "umap")
    qtbot.addWidget(view)

    view.publish_selection(_frame(2))
    assert view.selections == []

    link.select_frame(_frame(1), source="plate_view")
    assert len(view.selections) == 1


def test_a_view_can_ask_to_hear_its_own(link, qtbot):
    """For a view that redraws from the shared state rather than locally."""
    view = _View(link, "umap", echo=True)
    qtbot.addWidget(view)

    view.publish_selection(_frame(2))
    assert [s.source for s in view.selections] == ["umap"]


def test_an_unnamed_view_hears_everything_including_unnamed_selections(
        link, qtbot):
    """With no name there is no echo to suppress, and suppressing on "" would
    silence every view that has not named itself."""
    view = _View(link, "")
    qtbot.addWidget(view)

    link.set_selection(Selection.from_frame(_frame(1)))
    assert len(view.selections) == 1


def test_publishing_stamps_the_view_that_published(link, qtbot):
    view = _View(link, "umap")
    qtbot.addWidget(view)

    returned = view.publish_selection(_frame(2))
    assert link.selection.source == "umap"
    assert returned is link.selection


def test_a_view_can_publish_bare_keys_and_a_frame_alike(link, qtbot):
    view = _View(link, "umap")
    qtbot.addWidget(view)

    view.publish_selection("p1_r1_c1_f1_1")
    assert list(link.selection.keys) == ["p1_r1_c1_f1_1"]

    view.publish_selection(_frame(2, timelapse=True), timelapse=True)
    assert list(link.selection.keys) == ["p1_r1_c1_f1_1_1", "p1_r2_c2_f1_2_2"]


def test_a_view_clears_to_the_resting_state_not_to_an_empty_selection(
        link, qtbot):
    view = _View(link, "umap")
    qtbot.addWidget(view)

    view.publish_selection(_frame(2))
    view.clear_linked_selection()
    assert not link.selection.is_active


def test_a_view_can_narrow_the_population_for_everyone(link, qtbot):
    publisher = _View(link, "data_filter_panel")
    subscriber = _View(link, "plate_view")
    qtbot.addWidget(publisher)
    qtbot.addWidget(subscriber)

    publisher.publish_filter(DataFilter().add(CategoryFilter("rowID", ("r1",))))
    assert len(subscriber.filters) == 1
    assert len(subscriber.linked_visible(_frame(6))) == 3


def test_the_mixins_visible_still_refuses_to_hide_a_selection(link, qtbot):
    """The same rule as `LinkedSelection.visible`, restated through the mixin.

    A view that dropped unselected rows would make the lasso destructive: you
    could not see what you had excluded, or undo it by lassoing more.
    """
    view = _View(link, "umap")
    qtbot.addWidget(view)

    view.publish_selection(_frame(6).iloc[[0]])
    assert len(view.linked_visible(_frame(6))) == 6


def test_a_view_opens_objects_as_itself(link, qtbot):
    """The source is filled in from the view, so the destination can say who."""
    opener = _Recorder()
    link.register_object_opener("annotate", opener)
    view = _View(link, "umap")
    qtbot.addWidget(view)

    view.open_objects(_frame(1), reason="clicked a point",
                      context={"crop": "cell"})

    request, = opener.requests
    assert request.source == "umap"
    assert request.reason == "clicked a point"
    assert request.context["crop"] == "cell"


def test_linking_twice_does_not_double_the_callbacks(link, qtbot):
    """A screen that re-links on reload would repaint twice per lasso forever."""
    view = _View(link, "umap")
    qtbot.addWidget(view)
    view.link_selection("umap", link=link)

    link.set_filter(DataFilter())
    link.select_frame(_frame(1), source="plate_view")
    assert len(view.filters) == 1
    assert len(view.selections) == 1


def test_unlinking_stops_the_callbacks(link, qtbot):
    view = _View(link, "umap")
    qtbot.addWidget(view)
    view.unlink_selection()

    link.set_filter(DataFilter())
    link.select_frame(_frame(1), source="plate_view")
    assert view.filters == [] and view.selections == []
    assert not view.is_linked


def test_unlinking_twice_is_quiet(link, qtbot, capfd):
    """Qt does not raise on a disconnect that finds nothing — it prints
    `libpyside: Failed to disconnect` to stderr, where no `except` can reach
    it. A screen closed twice, which Qt does on teardown, would print one
    every time and train the reader to ignore that warning."""
    view = _View(link, "umap")
    qtbot.addWidget(view)
    capfd.readouterr()

    view.unlink_selection()
    view.unlink_selection()

    assert "Failed to disconnect" not in capfd.readouterr().err


def test_a_view_that_never_linked_still_reaches_the_process_wide_link(
        process_link, qtbot):
    """Publishing without subscribing is half the contract, and legal."""
    view = _SilentView()
    qtbot.addWidget(view)

    assert not view.is_linked
    assert view.link is process_link
    view.publish_selection(_frame(1))
    assert len(process_link.selection) == 1


def test_a_view_that_overrides_nothing_survives_both_signals(link, qtbot):
    """The default hooks are no-ops, so a view can subscribe for one and
    ignore the other without writing an empty method."""
    view = _SilentView()
    qtbot.addWidget(view)
    view.link_selection("silent", link=link)

    link.set_filter(DataFilter().add(RangeFilter("area", low=1.0)))
    link.select_frame(_frame(1), source="umap")
    assert view.is_linked
