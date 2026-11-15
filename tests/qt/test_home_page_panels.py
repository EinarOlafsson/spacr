"""The Home page, its running banner, totals and news panels, and the Navigator field, with a piece missing.

Home is a dashboard over other state -- the run journal, the theme
preferences, the logo artwork, the app registry, a running job -- and it is
the first screen the application shows.  Every case here is one of those
readings failing or one control acting on an edge: the page still opens with
its tabs and masthead, a quit prompt stops exactly what was chosen, totals
read as exact zeroes rather than a missing panel, and the news heading fits
the viewport it actually has.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case builds its own widget.
"""
from __future__ import annotations

import builtins
import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt, QVariantAnimation  # noqa: E402
from PySide6.QtWidgets import QComboBox, QFrame, QLabel, QWidget  # noqa: E402
from shiboken6 import isValid  # noqa: E402

import spacr.qt.bridge as bridge  # noqa: E402
import spacr.qt.preferences as preferences  # noqa: E402
import spacr.qt.shutdown as shutdown  # noqa: E402
from spacr import run_journal  # noqa: E402
from spacr.qt.app import SECTION_CORE, _icon_for_app  # noqa: E402
from spacr.qt.widgets import home as home_mod  # noqa: E402
from spacr.qt.widgets.ambient import AmbientWidget  # noqa: E402
from spacr.qt.widgets.home import (AppTile, HomePage, RecentRunsPanel,  # noqa: E402
                                   RunningBanner, TotalsPanel)
from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile  # noqa: E402,F401
from tests.qt.test_startup_ai_reporting_choice import (  # noqa: E402
    private_ai_profile as private_ai_profile,  # noqa: F401
)

pytestmark = pytest.mark.qt

APPS = [
    ("mask", "Mask", "Segment a plate", "Core"),
    ("measure", "Measure", "Measure objects", "Core"),
    ("classify", "Classify", "Train a classifier", "Analysis"),
]


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    """Point the run journal at an empty private folder."""
    root = tmp_path / "journal"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    return root


def _block_imports_of(mp, name_fragment):
    """Make ``import <name_fragment>...`` fail inside the patch context."""
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name_fragment in name:
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    mp.setattr(builtins, "__import__", _guarded)


def test_the_home_page_opens_with_tabs_and_masthead_whatever_is_missing(
        qtbot, monkeypatch, qt_theme_applied, _empty_journal, subtests):
    """HomePage builds its tabs, hero and tiles with an unreadable theme, no logo, dead apps or no ambient."""
    def build(apps=APPS, **kwargs):
        page = HomePage(apps, lambda key: None, **kwargs)
        qtbot.addWidget(page)
        return page

    with subtests.test(msg="unreadable_theme_still_builds_tabs"):
        calls = []

        def fails():
            calls.append(True)
            raise OSError("preferences are unreadable")

        with monkeypatch.context() as mp:
            mp.setattr(preferences, "resolve_effective_theme", fails)
            page = build()
        assert calls
        assert page._tabs.count() > 0

    with subtests.test(msg="hero_labels_reachable_by_object_name"):
        hero = build().findChild(QWidget, "Hero")
        assert hero is not None
        assert {label.text() for label in hero.findChildren(QLabel)} >= {"spaCR"}

    with subtests.test(msg="no_logo_costs_only_the_mark"):
        assert build().findChild(QLabel, "HeroMark") is not None
        with monkeypatch.context() as mp:
            mp.setattr(home_mod, "_find_logo_pixmap", lambda: None)
            without = build()
        assert without.findChild(QLabel, "HeroMark") is None
        assert without._hero_mark is None
        assert "spaCR" in [w.text() for w in without.findChildren(QLabel)]

    with subtests.test(msg="group_of_only_dead_apps_loses_its_tab"):
        page = build(categories=[("Real", ["mask", "gone"]),
                                 ("Ghosts", ["gone", "also_gone"])])
        titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
        assert any(t.startswith("Real  (1)") for t in titles), titles
        assert not any(t.startswith("Ghosts") for t in titles), titles

    with subtests.test(msg="unfittable_name_elides_within_the_tile_cap"):
        from spacr.qt.preferences import scaled_px

        long_name = "Extremely Long Hypothetical Module Name For Testing"
        page = HomePage([("mask", long_name, "d", SECTION_CORE)], _icon_for_app)
        qtbot.addWidget(page)
        page.resize(1400, 600)
        page.show()
        qtbot.waitExposed(page)
        tile = next(t for t in page.findChildren(AppTile) if t.isVisible())
        label = tile.name_label
        assert label.is_elided()
        assert "…" in label.text() and label.text() != long_name
        assert label.full_text() == long_name == label.toolTip()
        assert tile.width() <= scaled_px(HomePage.TILE_MAX_W)

    with subtests.test(msg="discard_with_the_ambient_module_absent_sweeps_nothing"):
        page = build()
        leftover = AmbientWidget(page)
        with monkeypatch.context() as mp:
            _block_imports_of(mp, "ambient")
            page._discard_ambient(leftover)
        assert leftover.parent() is page

    with subtests.test(msg="a_stubborn_leftover_does_not_stop_the_sweep"):
        page = build()

        class _Stubborn(AmbientWidget):
            def set_animating(self, on):
                raise RuntimeError("already deleted")

            def setParent(self, parent):
                raise RuntimeError("already deleted")

        stubborn = _Stubborn(page)
        ordinary = AmbientWidget(page)
        page._discard_ambient(stubborn)
        assert ordinary.parent() is None


class _Gate:
    @staticmethod
    def is_paused():
        return False


class _Handle:
    """A run handle with the surface Home reads."""

    def __init__(self, app_key="mask", running=True):
        self.app_key = app_key
        self.thread = None
        self.worker = None
        self.cancels = []
        self.supports_pause = False
        self.gate = _Gate()
        self.progress = None
        self.last_line = ""
        self._running = running

    def is_running(self):
        return self._running

    def request_cancel(self, why):
        self.cancels.append(why)

    def fraction(self):
        return None

    def elapsed(self):
        return 12.0


def test_the_running_banner_stops_exactly_what_the_quit_prompt_chose(qtbot, monkeypatch, subtests):
    """RunningBanner._on_quit and _terminate cancel, force or do nothing, surviving dead handles and threads."""
    def banner():
        widget = RunningBanner(lambda key: None, {"mask": "Mask"})
        qtbot.addWidget(widget)
        return widget

    with subtests.test(msg="no_job_asks_nothing"):
        widget = banner()
        asked = []
        with monkeypatch.context() as mp:
            mp.setattr(shutdown, "ask_how_to_quit",
                       lambda parent, what, detail: asked.append(what))
            widget.bind(None)
            widget._on_quit()
        assert asked == []

    choices = [("cancel_stops_nothing", shutdown.CANCEL, []),
               ("force_stops_outright", shutdown.FORCE, ["force quit from the Home screen"]),
               ("graceful_asks_and_keeps_watching", shutdown.GRACEFUL, ["quit from the Home screen"])]
    for case_id, choice, cancels in choices:
        with subtests.test(msg=case_id):
            widget = banner()
            handle = _Handle()
            widget.bind(handle)
            with monkeypatch.context() as mp:
                mp.setattr(shutdown, "ask_how_to_quit", lambda parent, what, detail: choice)
                widget._on_quit()
            assert handle.cancels == cancels
            if choice == shutdown.GRACEFUL:
                assert widget._quit_watcher.parent() is widget
                widget._quit_watcher.stop()

    with subtests.test(msg="terminate_with_no_thread"):
        handle = _Handle()
        RunningBanner._terminate(handle)
        assert handle.cancels == ["force quit from the Home screen"]

    with subtests.test(msg="terminate_a_handle_refusing_to_cancel_still_drains"):
        class _Refusing(_Handle):
            def request_cancel(self, why):
                raise RuntimeError("the handle is already retired")

        drained = []
        handle = _Refusing()
        handle.thread = object()
        with monkeypatch.context() as mp:
            mp.setattr(bridge, "drain_thread",
                       lambda thread, worker=None, timeout_ms=3000: drained.append(thread))
            RunningBanner._terminate(handle)
        assert drained == [handle.thread]

    with subtests.test(msg="terminate_a_thread_already_gone"):
        def gone(thread, worker=None, timeout_ms=3000):
            raise RuntimeError("Internal C++ object already deleted.")

        handle = _Handle()
        handle.thread = object()
        with monkeypatch.context() as mp:
            mp.setattr(bridge, "drain_thread", gone)
            RunningBanner._terminate(handle)
        assert handle.cancels == ["force quit from the Home screen"]


def _classic_state(mp, folder):
    """Private preferences and a two-run journal; returns the manifest bytes."""
    folder.mkdir()
    settings = QSettings(str(folder / "classic-home.ini"), QSettings.IniFormat)
    mp.setattr(preferences, "_settings", lambda: settings)
    runs = folder / "runs"
    runs.mkdir()
    mp.setattr(run_journal, "runs_root", lambda: runs)
    manifests = []
    for index, module in enumerate(("mask", "measure")):
        run = runs / f"2020-01-01_12000{index}_{module}"
        run.mkdir()
        path = run / "manifest.json"
        path.write_text(json.dumps({"app_key": module, "status": "success",
                                    "start_utc": f"2020-01-01T12:00:0{index}+00:00",
                                    "elapsed_s": 3}), encoding="utf-8")
        manifests.append(path)
    return {path: path.read_bytes() for path in manifests}


def _unavailable(*args, **kwargs):
    raise OSError("optional dashboard service unavailable")


ZERO_TOTALS = dict(total_runs=0, mask_runs=0, measure_runs=0, models_recorded=0)


def test_the_totals_panel_counts_the_journal_or_reports_exact_zeroes(qtbot, monkeypatch, tmp_path, subtests):
    """TotalsPanel reads the journal, resets only its watermark, and gives zeroes when a service fails."""
    with subtests.test(msg="reset_moves_only_watermarks"):
        with monkeypatch.context() as mp:
            journal = _classic_state(mp, tmp_path / "reset")
            recent = RecentRunsPanel(known_keys={"mask": "Segment", "measure": "Measure"},
                                     read_now=False)
            totals = TotalsPanel(read_now=False)
            qtbot.addWidget(recent)
            qtbot.addWidget(totals)
            assert totals.read() == dict(total_runs=2, mask_runs=1, measure_runs=1,
                                         classify_runs=0, models_recorded=0)
            recent.refresh()
            totals.refresh()
            assert len(recent.read()) == 2
            cleared, reset = [], []
            recent.cleared.connect(lambda: cleared.append(True))
            totals.reset_requested.connect(lambda: reset.append(True))
            recent._clear.click()
            totals._reset.click()
            assert cleared == [True] and reset == [True]
            assert preferences.get_dashboard_watermark("runs")
            assert preferences.get_dashboard_watermark("totals")
            assert recent.read() == []
            assert totals.read()["total_runs"] == 0
            assert not recent._clear.isEnabled() and not totals._reset.isEnabled()
        assert all(path.read_bytes() == raw for path, raw in journal.items())

    with subtests.test(msg="watermark_service_failure_counts_everything"):
        with monkeypatch.context() as mp:
            journal = _classic_state(mp, tmp_path / "watermark")
            panel = TotalsPanel(read_now=False)
            qtbot.addWidget(panel)
            mp.setattr(preferences, "get_dashboard_watermark", _unavailable)
            assert panel.read() == dict(total_runs=2, mask_runs=1, measure_runs=1,
                                        classify_runs=0, models_recorded=0)
        assert all(path.read_bytes() == raw for path, raw in journal.items())

    with subtests.test(msg="journal_service_failure_reads_zeroes"):
        with monkeypatch.context() as mp:
            journal = _classic_state(mp, tmp_path / "journal")
            panel = TotalsPanel(read_now=False)
            qtbot.addWidget(panel)
            mp.setattr(run_journal, "journal_totals", _unavailable)
            assert panel.read() == ZERO_TOTALS
            panel.refresh()
            assert not panel._reset.isEnabled()
        assert all(path.read_bytes() == raw for path, raw in journal.items())

    with subtests.test(msg="unreadable_journal_at_construction_reads_zeroes"):
        with monkeypatch.context() as mp:
            mp.setattr(run_journal, "journal_totals", _unavailable)
            panel = TotalsPanel()
            qtbot.addWidget(panel)
            assert panel.read() == ZERO_TOTALS


def test_the_news_heading_fits_the_viewport_it_has(qtbot, monkeypatch, tmp_path, subtests):
    """NewsPanel wraps its title to the viewport width, bounds its date, and declines unusable viewports."""
    settings = QSettings(str(tmp_path / "classic-home.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)

    def panel_with(name, date, tag):
        with monkeypatch.context() as mp:
            mp.setattr(home_mod.NewsPanel, "read_releases", staticmethod(lambda: [
                {"tag": tag, "name": name, "published": date, "body": ""}]))
            panel = home_mod.NewsPanel()
        qtbot.addWidget(panel)
        return panel

    with subtests.test(msg="title_wraps_while_a_long_date_scrolls"):
        name = ("A readable release heading with enough words to require several "
                "lines on a narrow panel")
        date = "Wednesday September 30 2026 23:59:59 UTC, localized publication date"
        panel = panel_with(name, date, "v2")
        title, stamp = panel._headline_labels
        title.setFrameStyle(QFrame.Box)
        title.setMargin(4)
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        panel.resize(580, 360)
        panel.show()
        qtbot.waitExposed(panel)
        panel._news_title_fit._fit()
        wide_height = title.property("minimumSize").height()
        panel.resize(280, 360)
        qtbot.waitUntil(lambda: panel._notes.viewport().property("size").width() < 400)
        panel._news_title_fit._fit()
        width = panel._notes.viewport().property("size").width()
        assert title.property("minimumSize").width() == width
        assert title.property("maximumSize").width() == width
        assert title.property("minimumSize").height() > wide_height
        assert title.text() == name and title.accessibleName() == name
        assert stamp.text() == date and not stamp.wordWrap()
        assert stamp.property("maximumSize").width() > width

    with subtests.test(msg="bottom_aligned_framed_heading_and_short_date"):
        title_text = "A release heading with enough words to wrap inside the framed news panel"
        date_text = "2026-10-28"
        panel = panel_with(title_text, date_text, "v3")
        title, stamp = panel._headline_labels
        title.setFrameStyle(QFrame.Box)
        title.setMargin(4)
        title.setIndent(-1)
        title.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        panel.resize(300, 380)
        panel.show()
        qtbot.waitExposed(panel)
        qtbot.waitUntil(lambda: title.property("minimumSize").height()
                        >= title.heightForWidth(title.property("size").width()) > 0)
        width = panel._notes.viewport().property("size").width()
        assert title.wordWrap() and title.indent() == -1
        assert title.property("minimumSize").width() == width
        assert title.property("maximumSize").width() == width
        assert title.text() == title.accessibleName() == title_text
        assert stamp.text() == date_text and not stamp.wordWrap()
        assert stamp.property("maximumSize").width() == width

    for state in ("zero_allocation", "retired_viewport"):
        with subtests.test(msg=f"fitter_declines_{state}"):
            panel = home_mod.NewsPanel()
            qtbot.addWidget(panel)
            fitter = panel._news_title_fit
            titles = tuple(panel._headline_labels)
            if state == "retired_viewport":
                previous = fitter._viewport
                content = panel._notes.takeWidget()
                panel._notes.setViewport(QWidget())
                panel._notes.setWidget(content)
                assert not isValid(previous)
            else:
                fitter._viewport.resize(0, 0)
                assert fitter._viewport.property("size").width() == 0
            fitter._fit()
            assert tuple(panel._headline_labels) == titles
            assert isValid(panel) and all(isValid(title) for title in titles)


LONG_CHOICE = "A choice whose text is long enough to need several lines when narrow"


def test_the_navigator_protocol_field_places_label_and_choice(qtbot, subtests):
    """_NavigatorProtocolField asks for no room when empty and places its label beside the choice when wide."""
    def field_with_parts():
        label = QLabel("Sequencer")
        label.setWordWrap(True)
        choice = QComboBox()
        choice.addItems(["short", LONG_CHOICE])
        field = home_mod._NavigatorProtocolField()
        qtbot.addWidget(field)
        field.set_parts(label, choice)
        return field, label, choice

    with subtests.test(msg="no_parts_asks_for_no_room"):
        field = home_mod._NavigatorProtocolField()
        qtbot.addWidget(field)
        assert field.beside(100) is True
        assert field.hasHeightForWidth() is True
        assert field.heightForWidth(100) == 0
        assert (field.sizeHint().width(), field.sizeHint().height()) == (0, 0)
        assert field.minimumSizeHint().height() == 0
        field._place(100)
        field.resize(80, 20)

    with subtests.test(msg="wide_field_puts_the_label_beside_the_choice"):
        field, label, choice = field_with_parts()
        wide = field.sizeHint().width() + 40
        assert field.beside(wide) is True
        field.resize(wide, field.heightForWidth(wide))
        field.show()
        assert label.geometry().right() < choice.geometry().left()

    with subtests.test(msg="changing_the_choice_places_the_parts_again"):
        field, _label, choice = field_with_parts()
        field.resize(field.sizeHint().width() + 40, 80)
        field.show()
        choice.setCurrentIndex(1)
        assert field.heightForWidth(field.width()) > 0


def test_the_easy_home_field_finishes_growth_when_motion_stops(qtbot, monkeypatch, protocol_profile, subtests):  # noqa: F811
    """_EasyHomePage's field keeps arrival progress with motion on and finishes growth with motion off or hidden."""
    from tests.qt.test_739_navigator_protocols import _choose

    page = home_mod._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1366, 700)
    page.show()
    qtbot.waitExposed(page)
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, "sequencing")

    with subtests.test(msg="motion_on_leaves_the_arrival_alone"):
        controls = page.field.protocol_controls
        controls._progress = 0.4
        with monkeypatch.context() as mp:
            mp.setattr(page.field, "_motion_enabled", lambda: True)
            page.field._sync_run_state()
        assert controls._progress == 0.4

    with subtests.test(msg="motion_off_without_controls_finishes_growth"):
        page.field.protocol_controls = None
        running = QVariantAnimation(page.field)
        running.setDuration(60000)
        running.start()
        page.field._animations.append(running)
        assert running.state() == QVariantAnimation.Running
        with monkeypatch.context() as mp:
            mp.setattr(page.field, "_motion_enabled", lambda: False)
            page.field._sync_run_state()
        assert running.state() == QVariantAnimation.Stopped
        assert page.field._animations == []
        for card in page.field.cards:
            if card.parentWidget() is page.field and page.field._suggestions_active:
                assert card._growing is False
                assert card.isEnabled()
                assert card.geometry() == card._final_rect

    with subtests.test(msg="hiding_a_field_without_controls"):
        page.field.protocol_controls = None
        page.field.hide()
        assert not page.field.isVisible()
