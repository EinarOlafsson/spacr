"""Pre-run panels — the paths where the data, the screen or the job is wrong.

``tests/qt/test_prerun.py`` proves the two panels work on real masks and
real screens, and that neither can touch Run. This file is the other half:
what happens when the folder is a placeholder, the scorecard is unreadable,
the worker will not start, the settings model refuses to be read, or the
screen has nowhere to put the panel.

The rule these all serve is one sentence from the module: *a screen that
opens without the banner is always better than a screen that does not
open*. So every failure here is asserted to end in a hidden panel or a
sentence in the panel's own status label -- never in an exception reaching
the caller.

The worker jobs are run on the calling thread through ``_start_job``'s own
signature, which is also the only way their bodies are ever measured: a
closure that only ever runs inside a ``QThread`` is invisible to coverage.
"""
from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel, QLineEdit, QHBoxLayout, QVBoxLayout, QWidget

from spacr.qt import prerun


# ---------------------------------------------------------------------------
# Stand-ins for the parts of an AppScreen these panels reach for
# ---------------------------------------------------------------------------

class _Model:
    """The two attributes ``prerun`` reads off ``screen._settings_model``."""

    def __init__(self, widgets=None, settings=None, collect_raises=False):
        self._widgets = dict(widgets or {})
        self._settings = dict(settings or {})
        self._collect_raises = collect_raises
        self.written = {}

    def collect(self):
        if self._collect_raises:
            raise RuntimeError("the settings model is mid-rebuild")
        return dict(self._settings)

    def set_value_for_key(self, key, value):
        self.written[key] = value
        return True


class _Screen(QWidget):
    """A screen shaped like ``AppScreen`` for the parts prerun touches."""

    def __init__(self, widgets=None, settings=None, collect_raises=False,
                 with_anchors=True):
        super().__init__()
        self._settings_model = _Model(widgets, settings, collect_raises)
        self._applied = []
        if with_anchors:
            self._runtime_wrap = QWidget(self)
            QVBoxLayout(self._runtime_wrap)
            self._actions_row = QWidget(self._runtime_wrap)
            QHBoxLayout(self._actions_row)
            self._runtime_wrap.layout().addWidget(self._actions_row)

    def _apply_value(self, widget, value):
        self._applied.append((widget, value))
        widget.setText(str(value))


def _src_field(text=""):
    field = QLineEdit()
    field.setText(text)
    return field


@pytest.fixture
def a_screen(qtbot):
    def build(**kwargs):
        screen = _Screen(**kwargs)
        qtbot.addWidget(screen)
        return screen
    return build


# ---------------------------------------------------------------------------
# _ShowFilter
# ---------------------------------------------------------------------------

def test_a_show_event_reaches_the_callback_and_is_not_consumed(qtbot):
    """Built once, shown many times -- Show is when a re-mask is noticed."""
    seen = []
    watched = QWidget()
    qtbot.addWidget(watched)
    watched.installEventFilter(prerun._ShowFilter(lambda: seen.append(1),
                                                  watched))

    watched.show()
    qtbot.waitExposed(watched)

    assert seen


def test_a_callback_that_raises_does_not_swallow_the_show_event(qtbot,
                                                                caplog):
    watched = QWidget()
    qtbot.addWidget(watched)

    def explode():
        raise RuntimeError("the refresh blew up")

    filt = prerun._ShowFilter(explode, watched)
    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        consumed = filt.eventFilter(watched, QEvent(QEvent.Show))

    assert consumed is False
    assert "pre-run refresh failed on show" in caplog.text


def test_any_other_event_passes_straight_through(qtbot):
    seen = []
    watched = QWidget()
    qtbot.addWidget(watched)
    filt = prerun._ShowFilter(lambda: seen.append(1), watched)

    assert filt.eventFilter(watched, QEvent(QEvent.Hide)) is False
    assert seen == []


# ---------------------------------------------------------------------------
# Reading the src field
# ---------------------------------------------------------------------------


def test_a_list_of_plates_is_passed_through_whole(a_screen):
    """Both readers take a list, so several plates in one run stay a list."""
    class Several:
        def get_value(self):
            return ["/data/plate1", "  /data/plate2  ", "path", ""]

    screen = a_screen(widgets={"src": Several()})
    assert prerun._src_of(screen) == ["/data/plate1", "/data/plate2"]
    assert prerun._has_src(["/data/plate1"]) is True
    assert prerun._has_src([]) is False


def test_a_placeholder_is_not_a_source(a_screen, qtbot):
    for placeholder in sorted(prerun._PLACEHOLDERS):
        screen = a_screen(widgets={"src": _src_field(placeholder)})
        assert prerun._src_of(screen) == ""
        assert prerun._has_src(prerun._src_of(screen)) is False


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _JobMixin
# ---------------------------------------------------------------------------

class _Jobs(prerun._JobMixin):
    def __init__(self):
        self._init_jobs()


def test_a_second_job_while_one_is_in_flight_is_refused():
    jobs = _Jobs()
    jobs._busy = True
    assert jobs._start_job(lambda box: None, {}, None, "seg_qc") is False


def test_no_worker_thread_available_is_a_refusal_not_a_crash(monkeypatch,
                                                             caplog):
    import builtins

    real_import = builtins.__import__

    def without_bridge(name, *args, **kwargs):
        if name.endswith("bridge"):
            raise ImportError("no bridge in this build")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_bridge)
    jobs = _Jobs()

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert jobs._start_job(lambda box: None, {}, None, "seg_qc") is False
    assert jobs.busy is False


def test_a_thread_that_cannot_be_built_is_a_refusal(monkeypatch, caplog):
    from spacr.qt import bridge

    def explode(*_args, **_kwargs):
        raise RuntimeError("no QApplication on this thread")

    monkeypatch.setattr(bridge, "make_thread", explode)
    jobs = _Jobs()

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert jobs._start_job(lambda box: None, {}, None, "seg_qc") is False
    assert "could not build the worker thread" in caplog.text
    assert jobs.busy is False


def test_a_completion_handler_that_raises_still_clears_the_busy_flag(caplog):
    """Otherwise one bad redraw leaves the panel unable to run again."""
    jobs = _Jobs()
    jobs._busy = True
    jobs._box = {"value": 1}
    jobs._on_done = lambda box: (_ for _ in ()).throw(
        RuntimeError("the redraw failed"))

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        jobs._job_settled(True)

    assert jobs.busy is False
    assert "pre-run job completion failed" in caplog.text


def test_a_failed_job_hands_the_error_to_the_handler():
    jobs = _Jobs()
    jobs._busy = True
    jobs._box = {"error": "the masks could not be opened"}
    seen = []
    jobs._on_done = seen.append

    jobs._job_settled(False)

    assert seen == [{"error": "the masks could not be opened"}]


# ---------------------------------------------------------------------------
# SegQCBanner
# ---------------------------------------------------------------------------

class _Digest:
    """A seg-QC digest, shaped the way the banner reads one."""

    def __init__(self, verdict="warn", findings=(), stale=False,
                 scorecards=()):
        self.verdict = verdict
        self.findings = list(findings)
        self.stale = stale
        self.scorecards = list(scorecards)
        self.blocks_run = False
        self.headline = f"{verdict} headline"
        self.subhead = "two plates, eight wells"
        self.plates = ("plate1",)


class _Finding:
    def __init__(self, severity="warn", headline="h", detail="d", fix="f"):
        self.severity = severity
        self.headline = headline
        self.detail = detail
        self.fix = fix


@pytest.fixture
def banner(qtbot, a_screen):
    def build(src="", reader=None, **kwargs):
        screen = a_screen(widgets={"src": _src_field(src)}, **kwargs)
        # threaded=False: the read runs inline, emitting the same
        # signals in the same order, so these tests can assert on the
        # drawn banner without spinning an event loop. The interface
        # never builds one this way -- see tests/qt/
        # test_the_measure_banner_never_waits_on_a_filesystem.py.
        widget = prerun.SegQCBanner(screen, reader=reader,
                                    threaded=False)
        qtbot.addWidget(widget)
        return widget
    return build


def test_a_screen_that_cannot_be_watched_still_gets_its_banner(qtbot,
                                                               caplog):
    """A plain object has no installEventFilter; the banner is built anyway."""
    holder = QWidget()
    qtbot.addWidget(holder)

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        widget = prerun.SegQCBanner(object(), parent=holder)

    assert "could not watch the Measure screen" in caplog.text
    assert widget.digest is None


def test_a_src_field_that_cannot_be_followed_still_builds(qtbot, a_screen,
                                                          caplog):
    class Unfollowable:
        class textChanged:
            @staticmethod
            def connect(_slot):
                raise RuntimeError("that signal is gone")

        def text(self):
            return ""

    screen = a_screen(widgets={"src": Unfollowable()})
    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        widget = prerun.SegQCBanner(screen)
    qtbot.addWidget(widget)

    assert "could not follow the src field" in caplog.text


def test_returning_to_the_screen_schedules_a_refresh(banner):
    widget = banner()
    widget._on_screen_shown()
    assert widget._timer.isActive() is True


def test_drawing_before_a_digest_arrives_does_nothing(banner):
    widget = banner()
    widget._digest = None
    widget._draw()                       # must not raise
    assert widget._title.text() == "Segmentation QC"


def test_the_findings_are_collapsed_then_expanded_on_request(banner):
    widget = banner(src="/data/plate1",
                    reader=lambda src: _Digest(
                        "warn", [_Finding(headline=f"finding {i}",
                                          detail=f"detail {i}",
                                          fix="Do this. " + "And then look "
                                              "at everything else. " * 12)
                                 for i in range(5)]))
    widget.refresh()

    collapsed = [lbl.text() for lbl in widget.findChildren(QLabel)]
    assert sum(1 for t in collapsed if t.startswith("• finding")) == 2
    assert widget._btn_more.text() == "Show all 5 findings"
    # Collapsed shows only the actionable head of a long fix, and no detail.
    assert any(t == "→ Do this. …" for t in collapsed)
    assert not any(t.startswith("detail") for t in collapsed)

    widget._on_toggle_findings()

    expanded = [lbl.text() for lbl in widget.findChildren(QLabel)]
    assert sum(1 for t in expanded if t.startswith("• finding")) == 5
    assert sum(1 for t in expanded if t.startswith("detail")) == 5
    assert any(t.endswith("everything else. ") for t in expanded)
    assert widget._btn_more.text() == "Show less"

    widget._on_toggle_findings()
    assert widget._btn_more.text() == "Show all 5 findings"


def test_toggling_before_a_digest_arrives_does_nothing(banner):
    widget = banner()
    widget._on_toggle_findings()
    assert widget._expanded is True


# -- copy -------------------------------------------------------------------

def test_copying_before_a_digest_arrives_does_nothing(banner):
    """Nothing means the clipboard is left alone.

    A copy button that runs before there is a report and writes an empty
    string has thrown away whatever the user was carrying -- silently, and
    without raising.
    """
    from PySide6.QtWidgets import QApplication

    QApplication.clipboard().setText("something the user copied earlier")
    widget = banner()
    widget._on_copy_clicked()
    assert QApplication.clipboard().text() == "something the user copied earlier"


# -- scoring ----------------------------------------------------------------


def test_the_scoring_job_really_scores_the_masks_under_src(banner, tmp_path,
                                                           qtbot):
    """The one expensive path, run here on the calling thread."""
    from spacr import seg_qc

    root = tmp_path / "plate1"
    folder = root / "norm_channel_stack" / "cell_mask_stack"
    folder.mkdir(parents=True)
    for well in ("A01", "A02", "B01", "B02"):
        labels = np.zeros((64, 64), np.uint16)
        for index, (row, col) in enumerate(
                [(10, 10), (10, 40), (40, 10), (40, 40)], start=1):
            labels[row:row + 8, col:col + 8] = index
        np.save(folder / f"plate1_{well}_1.npy", labels)

    widget = banner(src=str(root))
    ran = {}

    def run_here(fn, box, on_done, key):
        fn(box)
        ran["box"] = box
        on_done(box)
        return True

    widget._start_job = run_here
    widget._on_score_clicked()

    assert ran["box"]["digest"] is not None
    assert widget.digest is ran["box"]["digest"]
    assert widget.digest.verdict in {"ok", "warn", "fail"}
    assert seg_qc.find_scorecards(str(root))


def test_a_scoring_pass_that_produced_nothing_says_so(banner):
    widget = banner(src="/data/plate1")
    widget._on_scored({"error": "the masks could not be opened"})
    assert widget._title.text() == (
        "Segmentation QC — could not score these masks")
    assert widget._btn_score.isEnabled() is True


# ---------------------------------------------------------------------------
# DiameterPanel
# ---------------------------------------------------------------------------

def _estimate(object_type="cell", diameter=24.0, usable=True):
    from spacr.diameter import DiameterEstimate

    if usable:
        return DiameterEstimate(
            object_type=object_type, diameter=diameter, low=diameter - 4,
            high=diameter + 4, n_objects=120, n_fields=5,
            method="threshold_otsu", confidence="high",
            note="measured from five fields")
    return DiameterEstimate(
        object_type=object_type, diameter=float("nan"), low=float("nan"),
        high=float("nan"), n_objects=0, n_fields=0, method="none",
        confidence="low", note="nothing was found in these fields")


@pytest.fixture
def diameter(qtbot, a_screen):
    def build(src="", settings=None, estimator=None, widgets=None, **kwargs):
        fields = {"src": _src_field(src)}
        for obj in ("cell", "nucleus", "pathogen"):
            fields[f"{obj}_diameter"] = _src_field("")
        fields.update(widgets or {})
        screen = a_screen(widgets=fields, settings=settings, **kwargs)
        panel = prerun.DiameterPanel(screen, estimator=estimator)
        qtbot.addWidget(panel)
        return panel
    return build


def test_a_screen_with_no_settings_model_reads_no_settings(qtbot):
    holder = QWidget()
    qtbot.addWidget(holder)
    panel = prerun.DiameterPanel(object(), parent=holder)
    assert panel._settings() == {}


def test_settings_that_cannot_be_collected_read_as_none(diameter, caplog):
    panel = diameter(collect_raises=True)
    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert panel._settings() == {}
    assert "could not read the mask settings" in caplog.text


def test_the_injected_estimator_is_the_one_the_job_calls(diameter, tmp_path):
    seen = {}

    def estimator(src, channels, *, n_fields, metadata_type, custom_regex):
        seen.update(src=src, channels=dict(channels), n_fields=n_fields)
        return {"cell": _estimate()}

    panel = diameter(src=str(tmp_path), settings={"cell_channel": 0},
                     estimator=estimator)
    panel._start_job = lambda fn, box, on_done, key: (fn(box), on_done(box),
                                                      True)[-1]

    panel._on_measure_clicked()

    assert seen["src"] == str(tmp_path)
    assert seen["channels"] == {"cell": 0}
    assert panel.estimates["cell"].diameter == pytest.approx(24.0)


def test_the_rows_are_redrawn_rather_than_stacked(diameter, tmp_path):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate()}})
    first = panel._rows_layout.count()

    panel._on_estimated({"estimates": {"cell": _estimate(diameter=30.0)}})

    assert panel._rows_layout.count() == first == 1
    assert "30.0 px" in "\n".join(lbl.text()
                                 for lbl in panel.findChildren(QLabel))


def test_an_unusable_estimate_is_shown_as_no_estimate_and_offers_no_button(
        diameter, tmp_path):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(usable=False)}})

    texts = "\n".join(lbl.text() for lbl in panel.findChildren(QLabel))
    assert "cell: no estimate" in texts
    assert panel._btn_use_all.isVisible() is False
    assert panel.apply("cell") is False


# -- applying ---------------------------------------------------------------

def test_applying_an_object_that_was_never_measured_writes_nothing(diameter):
    panel = diameter()
    assert panel.apply("cell") is False


def test_a_measured_diameter_is_written_as_a_whole_number(diameter,
                                                          tmp_path):
    """``expected_types`` declares these keys int; a float arrives as "24.0"."""
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(diameter=23.6)}})

    assert panel.apply("cell") is True
    assert panel._screen._settings_model.written == {"cell_diameter": 24}
    assert "cell_diameter set to 24 px." == panel._status.text()


def test_a_model_that_refuses_the_write_falls_back_to_the_widget(diameter,
                                                                  tmp_path,
                                                                  caplog):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(diameter=24.0)}})
    model = panel._screen._settings_model
    model.set_value_for_key = lambda key, value: (_ for _ in ()).throw(
        RuntimeError("that key is locked"))

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert panel.apply("cell") is True

    widget = panel._screen._settings_model._widgets["cell_diameter"]
    assert widget.text() == "24"
    assert "could not write cell_diameter" in caplog.text


def test_a_model_that_declines_without_raising_also_falls_back(diameter,
                                                               tmp_path):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(diameter=24.0)}})
    panel._screen._settings_model.set_value_for_key = lambda key, value: False

    assert panel.apply("cell") is True
    assert panel._screen._applied


def test_a_widget_that_also_refuses_leaves_the_setting_alone(diameter,
                                                             tmp_path,
                                                             caplog):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(diameter=24.0)}})
    panel._screen._settings_model.set_value_for_key = lambda key, value: False
    panel._screen._apply_value = lambda widget, value: (_ for _ in ()).throw(
        RuntimeError("that widget is gone"))

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert panel.apply("cell") is False


def test_use_all_writes_every_usable_estimate_and_names_them(diameter,
                                                             tmp_path):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {
        "cell": _estimate("cell", 24.0),
        "nucleus": _estimate("nucleus", 12.0),
        "pathogen": _estimate("pathogen", usable=False)}})

    panel._on_use_all_clicked()

    written = panel._screen._settings_model.written
    assert written == {"cell_diameter": 24, "nucleus_diameter": 12}
    assert panel._status.text() == "Set cell_diameter, nucleus_diameter."


def test_use_all_with_nothing_usable_says_nothing_new(diameter, tmp_path):
    panel = diameter(src=str(tmp_path))
    panel._on_estimated({"estimates": {"cell": _estimate(usable=False)}})
    before = panel._status.text()

    panel._on_use_all_clicked()

    assert panel._status.text() == before


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def test_a_screen_with_no_anchors_cannot_carry_a_panel(qtbot):
    screen = _Screen(with_anchors=False)
    qtbot.addWidget(screen)
    assert prerun._insert_before_dimension_switches(screen, QWidget()) is False


def test_an_action_row_with_no_layout_cannot_carry_a_panel(qtbot):
    screen = _Screen()
    qtbot.addWidget(screen)
    screen._actions_row = QWidget(screen)          # no layout on it
    assert prerun._insert_before_dimension_switches(screen, QWidget()) is False


def test_the_qc_switch_goes_left_of_the_dimension_switches(qtbot):
    screen = _Screen(widgets={"src": _src_field("")})
    qtbot.addWidget(screen)
    row = screen._actions_row.layout()
    run, z_switch, t_switch = QWidget(), QWidget(), QWidget()
    for widget in (run, z_switch, t_switch):
        row.addWidget(widget)
    screen.dimension_switch = {"z": z_switch, "t": t_switch}.get

    banner = prerun.install_qc_banner(screen, reader=lambda src: _Digest())

    assert banner is not None
    assert row.indexOf(screen._seg_qc_toggle) == row.indexOf(z_switch) - 1
    assert row.indexOf(banner) < 0
    assert prerun._qc_dialog(screen).isAncestorOf(banner)
    # Installing twice hands back the one that is already there.
    assert prerun.install_qc_banner(screen) is banner


def test_a_banner_that_cannot_be_placed_is_cleaned_up(qtbot):
    screen = _Screen(widgets={"src": _src_field("")}, with_anchors=False)
    qtbot.addWidget(screen)
    assert prerun.install_qc_banner(screen) is None
    assert prerun.qc_banner(screen) is None


def test_a_banner_that_raises_while_being_built_costs_only_the_banner(
        qtbot, monkeypatch, caplog):
    screen = _Screen(widgets={"src": _src_field("")})
    qtbot.addWidget(screen)

    def explode(*_args, **_kwargs):
        raise RuntimeError("the banner could not be built")

    monkeypatch.setattr(prerun, "SegQCBanner", explode)

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert prerun.install_qc_banner(screen) is None
    assert "could not install the segmentation-QC banner" in caplog.text


def test_a_second_popup_shows_the_last_measurement_again(qtbot):
    """Closing the popup is not a reason to measure twice."""
    screen = _Screen(widgets={"src": _src_field(""),
                              "cell_diameter": _src_field("")})
    qtbot.addWidget(screen)
    first = prerun.diameter_dialog(screen)
    qtbot.addWidget(first)
    first.panel._on_estimated({"estimates": {"cell": _estimate()}})

    second = prerun.diameter_dialog(screen)
    qtbot.addWidget(second)

    assert set(second.panel.estimates) == {"cell"}
    assert second.panel._btn_use_all.isHidden() is False
    assert second.panel._rows_box.isHidden() is False


def test_install_puts_the_banner_on_measure_and_nothing_on_mask(qtbot):
    measure = _Screen(widgets={"src": _src_field("")})
    mask = _Screen(widgets={"src": _src_field(""),
                            "cell_diameter": _src_field("")})
    for screen in (measure, mask):
        qtbot.addWidget(screen)
    measure.app_key = prerun.QC_APP
    mask.app_key = prerun.DIAMETER_APP

    prerun.install(measure)
    prerun.install(mask)

    assert prerun.qc_banner(measure) is not None
    assert prerun.qc_banner(mask) is None
    for screen in (measure, mask):
        assert screen.findChildren(prerun.DiameterPanel) == []


def test_the_mask_screen_is_found_from_any_widget_inside_it(qtbot):
    mask = _Screen(widgets={"src": _src_field(""),
                            "cell_diameter": _src_field("")})
    qtbot.addWidget(mask)
    mask.app_key = prerun.DIAMETER_APP
    inner = QWidget(QWidget(mask))

    assert prerun.diameter_screen_of(inner) is mask
    assert prerun.diameter_screen_of(mask) is mask
    assert prerun.diameter_screen_of(None) is None
    mask.app_key = prerun.QC_APP
    assert prerun.diameter_screen_of(inner) is None


# ---------------------------------------------------------------------------
# The factory seam
# ---------------------------------------------------------------------------


@pytest.fixture
def factories_restored():
    """``APP_FACTORIES`` is process-global; hand it back afterwards."""
    from spacr.qt.app import APP_FACTORIES

    saved = dict(APP_FACTORIES)
    saved_inner = dict(prerun._INNER)
    yield APP_FACTORIES
    APP_FACTORIES.clear()
    APP_FACTORIES.update(saved)
    prerun._INNER.clear()
    prerun._INNER.update(saved_inner)


def test_registering_twice_is_a_no_op(factories_restored):
    prerun.unregister()
    assert prerun.register() is True
    assert prerun.register() is False
    assert factories_restored[prerun.QC_APP] is prerun._prerun_screen


def test_a_screen_built_here_is_wired_to_its_host_the_way_the_window_would(
        factories_restored, qtbot, monkeypatch):
    from spacr.qt import chaining

    seen = []

    class Host:
        pass

    host = Host()
    name = next(iter(chaining.HOST_CONNECTIONS))
    slot = chaining.HOST_CONNECTIONS[name]
    setattr(host, slot, lambda *args: seen.append(args))

    prerun._INNER.pop(prerun.QC_APP, None)
    screen = prerun._base_screen(prerun.QC_APP, host)
    qtbot.addWidget(screen)

    signal = getattr(screen, name, None)
    assert signal is not None
    assert screen.app_key == prerun.QC_APP


def test_wiring_that_fails_still_returns_a_screen(factories_restored, qtbot,
                                                  monkeypatch, caplog):
    from spacr.qt import chaining

    monkeypatch.setattr(chaining, "install_chaining",
                        lambda screen: (_ for _ in ()).throw(
                            RuntimeError("no strip today")))
    prerun._INNER.pop(prerun.QC_APP, None)

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        screen = prerun._base_screen(prerun.QC_APP, None)
    qtbot.addWidget(screen)

    assert screen is not None
    assert "could not wire measure" in caplog.text


# ---------------------------------------------------------------------------
# Edges the coverage ratchet found untested (dispatch 36739819315)
# ---------------------------------------------------------------------------


def test_a_rescore_with_nothing_to_score_still_rereads(banner, monkeypatch):
    widget = banner(src="")
    reread = []
    monkeypatch.setattr(widget, "refresh", lambda: reread.append(True))
    widget._score_again = True
    widget._refresh_again = True
    widget._pending_work()
    assert widget.busy is False
    assert reread == [True]
    assert widget._score_again is False and widget._refresh_again is False


def test_restyling_a_widget_without_a_style_is_a_no_op(banner):
    class _Styleless:
        def style(self):
            return None

    assert banner()._restyle(_Styleless()) is None


def test_the_qc_popup_follows_its_switch_and_survives_a_refused_glass(
        qtbot, a_screen, monkeypatch):
    from PySide6.QtWidgets import QCheckBox

    from spacr.qt.widgets import glass

    assert prerun._on_qc_toggled(a_screen(), True) is None
    screen = a_screen(widgets={"src": _src_field("")})
    widget = prerun.SegQCBanner(screen, threaded=False)
    dialog = prerun._SegQCDialog(screen, widget)
    screen._seg_qc_dialog = dialog
    screen._seg_qc_toggle = QCheckBox(screen)
    prerun._on_qc_toggled(screen, False)
    assert dialog.isHidden()

    def refuse(_dialog):
        raise RuntimeError("no glass here")

    monkeypatch.setattr(glass, "glass", refuse)
    prerun._on_qc_toggled(screen, True)
    assert dialog.isVisible()
    screen._seg_qc_toggle.setChecked(False)
    dialog.close()
    assert not dialog.isVisible()
    assert screen._seg_qc_toggle.isChecked() is False
