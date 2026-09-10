"""The last arc in ten small Qt widgets.

Every module here is above 98.6%, so what is left is one or two branches
apiece: the case a widget only meets when something it usually has is
absent. Three of them turn out to be guards a line above has already made
true; those carry the proof and a test that pins the invariant instead of
a contortion.

What is pinned, module by module:

``umap_explorer``
    The body splitter's divider always carries its "drag me" tooltip
    (proved: the handle exists because both panes are already in), and a
    close event on an explorer whose canvas has already gone still
    reaches ``QWidget.closeEvent``.
``foldable``
    ``make_foldable`` only wires ``remember`` when there is a key to
    remember under, so the ``if key:`` inside it cannot be false
    (proved); an unkeyed panel still reports its folds to ``on_change``.
``dna_rain_settings``
    The popover placed against an anchor with no screen behind it.
``channel_picker``
    A toggle arriving from something that is not a ``Toggle`` falls
    through to the change signal instead of being put back.
``animation_zoom``
    A well so small the padding closes its inner rectangle: the ring is
    then the whole rounded box.
``section``
    The header's mark row is built once and reused.
``activity_spinner``
    A delay that fires after the work already finished shows nothing.
``figure_grid``
    A relayout over a grid holding a spacer.
``grouped_plot``
    ``export_settings`` on a plot that holds no spec yet.
``umap_search_viewer``
    A gallery row whose stored index is not a usable trial index.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QLabel, QListWidgetItem,
                               QSpacerItem, QWidget)


# ---------------------------------------------------------------------------
# umap_explorer
# ---------------------------------------------------------------------------

class TestTheUmapDividerAndItsTeardown:
    """``if handle is not None`` and ``if getattr(self, "_canvas", None)``."""

    def test_the_body_divider_says_what_dragging_it_does(self, qapp):
        """``QSplitter.handle(1)`` is never None here, and this is why.

        The tooltip is set immediately after both panes have been added
        to ``self._body_splitter`` -- the chart on one side, the sidebar
        wrapper on the other. A ``QSplitter`` with N widgets owns N
        handles, index 0 being the unusable one before the first widget,
        so ``handle(1)`` is the divider between the two panes and exists
        as soon as the second ``addWidget`` returns. The ``is not None``
        beside it is a re-check of what those two calls guarantee.

        What matters is the tooltip itself: a 1 px line with no hover
        text is indistinguishable from the edge of the chart, which is
        the whole reason the line is there.
        """
        from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

        panel = ImageUmapExplorer()
        try:
            splitter = panel._body_splitter
            assert splitter.count() >= 2, (
                "the divider only exists because both panes are in; "
                f"got {splitter.count()} pane(s)")
            handle = splitter.handle(1)
            assert handle is not None
            assert "Drag" in handle.toolTip(), (
                f"the divider must announce itself; got {handle.toolTip()!r}")
        finally:
            panel.close()
            panel.deleteLater()

    def test_closing_an_explorer_whose_canvas_is_gone_still_closes(self, qapp):
        """Teardown order is not ours to choose.

        ``closeEvent`` cancels the canvas's pending ``draw_idle`` timer
        before Qt deletes the C++ side. An explorer torn down in a
        different order -- the canvas already dropped -- must still close
        rather than raise inside a Qt event handler.
        """
        from PySide6.QtGui import QCloseEvent

        from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

        panel = ImageUmapExplorer()
        try:
            assert getattr(panel, "_canvas", None) is not None, \
                "a live explorer has a canvas; that is the contrast"
            # The teardown case: the attribute has already been dropped.
            del panel._canvas
            event = QCloseEvent()
            panel.closeEvent(event)
            assert event.isAccepted(), \
                "QWidget.closeEvent must still have run"
        finally:
            panel.deleteLater()


# ---------------------------------------------------------------------------
# foldable
# ---------------------------------------------------------------------------

class TestAFoldIsOnlyRememberedWhenThereIsAKey:
    """``if key:`` inside ``make_foldable``'s ``remember``.

    ``remember`` is installed as ``on_change=remember if key else
    on_change`` -- the closure is only ever reachable when ``key`` is a
    non-empty string, because otherwise the caller's own ``on_change``
    (or ``None``) is wired instead and ``remember`` is never called at
    all. The ``if key:`` inside it therefore cannot be false: it is a
    re-check of the condition that decided whether this function would
    be called.

    Both halves of that decision are pinned below.
    """

    @staticmethod
    def _panel(qapp, key, seen):
        from spacr.qt.widgets.foldable import make_foldable

        heading = QLabel("Advanced")
        body = QWidget()
        folder = make_foldable(heading, body, name="advanced",
                               persist_key=key, on_change=seen.append)
        return heading, body, folder

    def test_a_keyed_panel_stores_the_fold_and_tells_the_caller(
            self, qapp, monkeypatch):
        import spacr.qt.preferences as prefs

        stored = {}
        monkeypatch.setattr(prefs, "get_folded_panels", lambda: {})
        monkeypatch.setattr(prefs, "set_folded_panel",
                            lambda key, shut: stored.__setitem__(key, shut))

        seen = []
        _heading, _body, folder = self._panel(qapp, "mask/advanced", seen)
        folder.set_shut(True)

        assert stored == {"mask/advanced": True}, (
            "a keyed panel writes its fold to the preferences; got "
            f"{stored}")
        assert seen == [True], "and still tells the caller"

    def test_an_unkeyed_panel_reports_but_stores_nothing(self, qapp,
                                                          monkeypatch):
        """The other half: no key, so ``remember`` is not even installed.

        A bare panel in a test must not write to the real preferences --
        it would fold a panel on the user's next launch -- but the
        caller's own callback still has to fire.
        """
        import spacr.qt.preferences as prefs

        stored = {}
        monkeypatch.setattr(prefs, "set_folded_panel",
                            lambda key, shut: stored.__setitem__(key, shut))

        seen = []
        _heading, _body, folder = self._panel(qapp, "", seen)
        folder.set_shut(True)

        assert seen == [True], "the caller is told either way"
        assert stored == {}, (
            "with no key nothing may be written; the same call with a key "
            "writes, which the test above shows")


# ---------------------------------------------------------------------------
# dna_rain_settings
# ---------------------------------------------------------------------------

class TestThePopoverWithNoScreenBehindIt:
    """``if screen is not None:`` in ``_position_near``."""

    @staticmethod
    def _popover(qapp):
        from spacr.qt.widgets.dna_rain_settings import (DnaRainSettingsBar,
                                                        DnaRainSettingsPopover)

        bar = DnaRainSettingsBar()
        return DnaRainSettingsPopover(bar)

    def test_without_a_screen_the_popover_is_placed_unclamped(self, qapp,
                                                              monkeypatch):
        """A widget on no screen at all still gets a position.

        ``QGuiApplication.screenAt`` can answer None, and on a headless
        run ``primaryScreen`` can too. The popover must still move to the
        anchor rather than sit at (0, 0) on top of the window.
        """
        from PySide6.QtGui import QGuiApplication

        from spacr.qt.widgets import dna_rain_settings as drs

        monkeypatch.setattr(drs.QGuiApplication, "screenAt",
                            staticmethod(lambda point: None))
        monkeypatch.setattr(drs.QGuiApplication, "primaryScreen",
                            staticmethod(lambda: None))

        anchor = QWidget()
        anchor.setFixedSize(80, 24)
        anchor.move(500, 400)
        popover = self._popover(qapp)
        popover.resize(200, 120)

        popover._position_near(anchor)

        # Directly above the anchor, centred on it, with the gap.
        assert popover.y() < anchor.mapToGlobal(anchor.rect().topLeft()).y()
        assert popover.x() == (
            anchor.mapToGlobal(anchor.rect().topLeft()).x()
            + anchor.width() // 2 - popover.width() // 2), (
            "with no screen the raw centred position is used, unclamped")
        assert QGuiApplication is not None
        popover.deleteLater()
        anchor.deleteLater()

    def test_with_a_screen_the_position_is_clamped_into_it(self, qapp):
        """The contrast: a real screen, and the popover is kept on it."""
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:                       # pragma: no cover - headless
            pytest.skip("no screen on this platform")
        area = screen.availableGeometry()

        anchor = QWidget()
        anchor.setFixedSize(80, 24)
        anchor.move(area.left(), area.top())     # hard against the corner
        popover = self._popover(qapp)
        popover.resize(200, 120)

        popover._position_near(anchor)

        assert popover.x() >= area.left(), (
            "a popover anchored at the left edge is pushed back on screen; "
            f"{popover.x()} < {area.left()}")
        assert popover.y() >= area.top()
        popover.deleteLater()
        anchor.deleteLater()


# ---------------------------------------------------------------------------
# channel_picker
# ---------------------------------------------------------------------------

class TestTheLastChannelIsPutBackOnlyWhenItCanBe:
    """``if isinstance(box, Toggle):`` in ``ChannelPicker._on_toggled``."""

    def test_unchecking_the_last_channel_puts_it_back(self, qapp):
        from spacr.qt.widgets.channel_picker import ChannelPicker

        picker = ChannelPicker("r", allow_none=False)
        heard = []
        picker.changed.connect(heard.append)

        picker._boxes["r"].setChecked(False)

        assert picker.value() == "r", (
            "a display that must keep a channel puts the last one back; got "
            f"{picker.value()!r}")
        assert heard == [], \
            "and does not announce a value the user never chose"
        picker.deleteLater()

    def test_a_toggle_from_nowhere_falls_through_to_the_signal(self, qapp):
        """The slot invoked directly -- ``sender()`` is then not a Toggle.

        Reached through the slot rather than a click because that is the
        only way ``sender()`` is anything but one of the picker's own
        boxes: Qt sets it from the emitting object, and outside a signal
        it is None. The guard exists because ``sender()`` is typed as
        ``QObject``, and this is the branch it takes when the cast fails.
        """
        from spacr.qt.widgets.channel_picker import ChannelPicker

        picker = ChannelPicker("", allow_none=False)
        heard = []
        picker.changed.connect(heard.append)

        picker._on_toggled(False)

        assert heard == [""], (
            "with no Toggle to put back the picker announces the value it "
            f"actually has; got {heard}")
        picker.deleteLater()


# ---------------------------------------------------------------------------
# animation_zoom
# ---------------------------------------------------------------------------

class TestAWellTooSmallToHaveAnInside:
    """``if inner_box[2] > inner_box[0] and inner_box[3] > inner_box[1]:``"""

    def test_a_box_narrower_than_the_padding_has_no_hole(self):
        from spacr.qt.widgets.animation_zoom import field_ring_mask

        # 4 px wide against the module's padding: left + pad is already
        # past right - pad, so there is no inner rectangle to subtract.
        ring = field_ring_mask(32, box=(14.0, 14.0, 18.0, 18.0), radius=1.0)

        assert ring.dtype == bool and ring.shape == (32, 32)
        assert ring.any(), "the outer rounded box is still drawn"
        # With no hole, the ring IS the filled outer box: its centre is set.
        assert bool(ring[16, 16]), (
            "with no inner rectangle to remove, the middle of the well is "
            "part of the ring")

    def test_a_normal_well_is_a_ring_with_a_hole(self):
        """The contrast: a box wide enough for an inside, which is empty."""
        from spacr.qt.widgets.animation_zoom import field_ring_mask

        ring = field_ring_mask(64, box=(8.0, 8.0, 56.0, 56.0), radius=6.0)

        assert ring.any()
        assert not bool(ring[32, 32]), (
            "a well with room for an inside has a hole in the middle")


# ---------------------------------------------------------------------------
# section
# ---------------------------------------------------------------------------

class TestTheHeaderMarkRowIsBuiltOnce:
    """``if row is None:`` in ``Section._source_row``."""

    def test_a_second_ask_reuses_the_layout_the_first_one_made(self, qapp):
        """Only reachable from inside: ``set_source_app`` asks exactly once.

        The public method builds the badge and asks for the row in the
        same branch, and never again once the badge exists -- so the
        "already built" side of the guard has to be driven by asking a
        second time directly. What it pins is that the header keeps ONE
        layout: a second one would re-parent the badge and leave the
        heading text under two competing layouts.
        """
        from spacr.qt.widgets.section import Section

        section = Section("Advanced")
        try:
            made = section.set_source_app("mask", "Generate masks")
            if not made:
                pytest.skip("no module icon is installed for 'mask'")
            first = section._header.layout()
            assert first is not None, "the mark row was built"

            again = section._source_row()

            assert again is first, (
                "the header must keep the one layout it has, not build a "
                "second over the heading text")
            assert section.source_mark() is not None
        finally:
            section.deleteLater()


# ---------------------------------------------------------------------------
# activity_spinner
# ---------------------------------------------------------------------------

class TestADelayThatFiresAfterTheWorkFinished:
    """``if self.is_busy():`` in ``_on_delay_elapsed``."""

    def test_work_that_finished_during_the_wait_shows_nothing(self, qapp):
        """The whole "not a prediction" claim, driven.

        The delay is armed while a job runs; the job finishes at 1.9 s;
        the timer fires at 2.0 s. The question is asked about the
        present, so nothing appears.
        """
        from spacr.qt.widgets.activity_spinner import ActivitySpinner

        spinner = ActivitySpinner()
        try:
            spinner.set_delay_ms(50_000)          # long enough never to fire
            spinner.set_busy(True)
            assert spinner.is_waiting(), "the delay must be armed"

            spinner.set_busy(False)               # the job finished first
            spinner._on_delay_elapsed()           # ...then the timer fired

            assert not spinner.is_spinning(), (
                "a job that finished before the delay elapsed must not put a "
                "spinner on screen")
            assert not spinner.isVisible()
        finally:
            spinner.deleteLater()

    def test_work_still_running_when_the_delay_fires_does_show(self, qapp):
        """The contrast that makes the absence above mean something."""
        from spacr.qt.widgets.activity_spinner import ActivitySpinner

        spinner = ActivitySpinner()
        try:
            spinner.set_delay_ms(50_000)
            spinner.set_busy(True)

            spinner._on_delay_elapsed()

            assert spinner.is_spinning(), \
                "work that is still running gets its spinner"
        finally:
            spinner.deleteLater()


# ---------------------------------------------------------------------------
# figure_grid
# ---------------------------------------------------------------------------

class TestARelayoutOverSomethingThatIsNotAWidget:
    """``if widget is not None:`` while draining the grid."""

    def test_a_spacer_in_the_grid_is_dropped_without_a_crash(self, qapp,
                                                             tmp_path):
        """``QLayout.takeAt`` hands back layout ITEMS, not widgets.

        ``relayout`` adds only labels, so the spacer has to be put in
        directly -- but the guard is not decoration: calling
        ``setParent`` on the ``None`` a spacer's ``widget()`` returns is
        an ``AttributeError`` inside a Qt event handler, and the grid is
        drained on every resize.
        """
        from PySide6.QtGui import QImage

        from spacr.qt.widgets.figure_grid import SearchFigureGrid

        png = tmp_path / "trial.png"
        QImage(8, 8, QImage.Format_RGB32).save(str(png))

        grid = SearchFigureGrid(["alpha"])
        try:
            grid.add_figure(str(png), {"alpha": 1})
            grid.relayout()
            assert grid.count() == 1

            grid._grid.addItem(QSpacerItem(4, 4))
            before = grid._grid.count()
            assert before > 1, "the spacer really is in the grid"

            grid.relayout()

            assert grid.count() == 1, "the figure survives the relayout"
            assert grid._grid.count() == 1, (
                "the spacer was drained and only the figure's label was put "
                f"back; got {grid._grid.count()} item(s)")
        finally:
            grid.deleteLater()


# ---------------------------------------------------------------------------
# grouped_plot
# ---------------------------------------------------------------------------

class TestExportingAPlotThatHoldsNothingYet:
    """``if self.spec is not None:`` in ``GroupedPlot.export_settings``."""

    def test_an_empty_plot_exports_only_the_base_settings(self, qapp):
        from spacr.qt.widgets.grouped_plot import GroupedPlot

        plot = GroupedPlot()
        try:
            out = plot.export_settings()

            assert isinstance(out, dict)
            for key in ("kind", "group", "value", "unit", "shape"):
                assert key not in out, (
                    f"a plot with no spec cannot describe its {key}; got "
                    f"{out}")
        finally:
            plot.deleteLater()

    def test_a_plot_with_a_spec_exports_what_it_draws(self, qapp):
        """The contrast: the same call, with data behind it."""
        from spacr.qt.widgets.grouped_plot import GroupedPlot, PlotSpec

        import pandas as pd

        frame = pd.DataFrame({"treatment": ["a", "a", "b", "b"],
                              "area": [1.0, 2.0, 3.0, 4.0]})
        spec = PlotSpec(frame=frame, group="treatment", value="area",
                        unit="cell")
        plot = GroupedPlot()
        try:
            plot.show_spec(spec)
            out = plot.export_settings()

            assert out["group"] == "treatment"
            assert out["value"] == "area"
            assert out["unit"] == "cell"
            assert out["kind"], "the drawn kind is exported"
            assert out["shape"]
        finally:
            plot.deleteLater()


# ---------------------------------------------------------------------------
# umap_search_viewer
# ---------------------------------------------------------------------------

class TestAGalleryRowThatPointsNowhere:
    """``if isinstance(index, int) and 0 <= index < len(self._trials):``"""

    class _Trial:
        """A search trial as the gallery reads one."""

        def __init__(self, name):
            self.params = {"n_neighbors": 15}
            self.score = 0.5
            self.name = name
            self.extra_metrics = {
                "embedding": np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.9]]),
                "backend": "cpu",
            }

    def test_a_row_with_no_usable_index_chooses_nothing(self, qapp):
        """A list item can outlive the trials it was built from.

        The gallery is refilled from a new search while a click is in
        flight; the item still carries the old row number. Emitting on it
        would load somebody else's embedding into the viewer.
        """
        from spacr.qt.widgets.umap_search_viewer import UmapGalleryDialog

        dialog = UmapGalleryDialog([self._Trial("a")])
        try:
            chosen = []
            dialog.trial_chosen.connect(chosen.append)

            stale = QListWidgetItem("a row from a previous search")
            stale.setData(Qt.UserRole, 7)          # past the end
            dialog._choose(stale)
            assert chosen == [], "an out-of-range row must choose nothing"

            # The same call with a usable index does emit, which is what
            # makes the empty list above a real absence.
            good = QListWidgetItem("row 0")
            good.setData(Qt.UserRole, 0)
            dialog._choose(good)
            assert len(chosen) == 1 and chosen[0].name == "a"
        finally:
            dialog.deleteLater()
