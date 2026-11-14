"""Table-driven tests of the pre-run panels in ``spacr.qt.prerun``.

One general test per unit: reading a settings widget (shared with the
chaining strip's reader), cutting a long fix to its first sentence, calling a
screen factory with the arguments it declares, the segmentation-QC banner's
read cache, drawing, copying and scoring, the diameter panel's measuring and
reporting, the diameter popup, and registering the Measure factory.  Every
failure is asserted to end in a hidden panel, an exact status line or an exact
log line -- never in an exception reaching the caller.  Offscreen Qt, worker
jobs replaced by inline calls through ``_start_job``'s own signature.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import (QHBoxLayout, QLineEdit, QVBoxLayout,  # noqa: E402
                               QWidget)

from spacr.qt import chaining as qt_chaining  # noqa: E402
from spacr.qt import prerun  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Stand-ins for the parts of an AppScreen these panels reach for
# ---------------------------------------------------------------------------

class _Model:
    """The attributes ``prerun`` reads off ``screen._settings_model``."""

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

    def __init__(self, widgets=None, settings=None, collect_raises=False):
        super().__init__()
        self._settings_model = _Model(widgets, settings, collect_raises)
        self._applied = []
        self._runtime_wrap = QWidget(self)
        QVBoxLayout(self._runtime_wrap)
        self._actions_row = QWidget(self._runtime_wrap)
        QHBoxLayout(self._actions_row)
        self._runtime_wrap.layout().addWidget(self._actions_row)

    def _apply_value(self, widget, value):
        self._applied.append((widget, value))
        widget.setText(str(value))


def _field(text=""):
    field = QLineEdit()
    field.setText(text)
    return field


@pytest.fixture
def a_screen(qtbot):
    """Build a stand-in screen qtbot cleans up."""
    def build(**kwargs):
        screen = _Screen(**kwargs)
        qtbot.addWidget(screen)
        return screen
    return build


class _Digest:
    """A seg-QC digest, shaped the way the banner reads one."""

    def __init__(self, verdict="warn", findings=(), stale=False):
        self.verdict = verdict
        self.findings = list(findings)
        self.stale = stale
        self.scorecards = []
        self.blocks_run = False
        self.headline = f"{verdict} headline"
        self.subhead = "two plates, eight wells"
        self.plates = ("plate1",)


@pytest.fixture
def banner(qtbot, a_screen):
    """A banner whose read runs inline (``threaded=False``)."""
    def build(src="", reader=None, **kwargs):
        screen = a_screen(widgets={"src": _field(src)}, **kwargs)
        widget = prerun.SegQCBanner(screen, reader=reader, threaded=False)
        qtbot.addWidget(widget)
        return widget
    return build


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
    """A diameter panel on a screen with src and the three diameter fields."""
    def build(src="", settings=None, estimator=None, **kwargs):
        fields = {"src": _field(src)}
        for obj in ("cell", "nucleus", "pathogen"):
            fields[f"{obj}_diameter"] = _field("")
        screen = a_screen(widgets=fields, settings=settings, **kwargs)
        panel = prerun.DiameterPanel(screen, estimator=estimator)
        qtbot.addWidget(panel)
        return panel
    return build


def _messages(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "spacr.qt.prerun"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_both_settings_readers_prefer_get_value_then_text_then_nothing(
        subtests, qtbot):
    """``get_value`` wins over ``text``; a raising reader or no reader reads as None."""
    class Both:
        def get_value(self):
            return "/data/plate1"

        def text(self):
            raise AssertionError("text() was preferred over get_value()")

    class Listed:
        def get_value(self):
            return ["/plate/a"]

    class GetterRaises:
        def get_value(self):
            raise RuntimeError("half-built")

        def text(self):
            return "never read"

    class TextRaises:
        def text(self):
            raise RuntimeError("deleted underneath us")

    line = _field("/data/plate2")
    qtbot.addWidget(line)
    cases = [
        ("get_value is preferred over text", Both(), "/data/plate1"),
        ("a list value is returned whole", Listed(), ["/plate/a"]),
        ("a plain line edit is read through text", line, "/data/plate2"),
        ("a raising getter is None even with a text", GetterRaises(), None),
        ("a raising text call is None", TextRaises(), None),
        ("a QObject with neither is None", QObject(), None),
        ("a plain object is None", object(), None),
        ("no widget at all is None", None, None),
    ]
    for module in (prerun, qt_chaining):
        for case_id, widget, expected in cases:
            with subtests.test(msg=f"{module.__name__}: {case_id}"):
                assert module._widget_value(widget) == expected


def test_a_long_fix_is_cut_at_its_first_sentence_or_a_word(subtests):
    """Short text is kept whole; long text is cut at ``. `` within the limit, else at a word."""
    long_tail = ("Then look at the flow threshold and at the channel order "
                 "and at everything else. " * 4)
    cases = [
        ("a short fix is shown whole", "Re-run masking.", {},
         "Re-run masking."),
        ("surrounding blanks are stripped", "  Re-run masking.  ", {},
         "Re-run masking."),
        ("None reads as an empty fix", None, {}, ""),
        ("text exactly at the limit is whole", "a" * 170, {}, "a" * 170),
        ("a long fix is cut after its first sentence",
         "Lower the cell diameter. " + long_tail, {},
         "Lower the cell diameter. …"),
        ("no sentence break is cut at the last whole word", "word " * 80,
         {"limit": 40}, " ".join(["word"] * 8) + " …"),
        ("a sentence break past the limit is ignored",
         "x" * 200 + ". tail", {}, "x" * 170 + " …"),
        ("a break at position 0 is not a sentence", ". " + "ab " * 100, {},
         ". " + "ab " * 55 + "ab …"),
    ]
    for case_id, text, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert prerun._first_sentence(text, **kwargs) == expected


def test_a_screen_factory_gets_exactly_the_arguments_it_declares(subtests):
    """``_call`` passes ``app_key``/``host`` by signature; an unreadable signature is called bare."""
    def takes_kwargs(**kwargs):
        return kwargs

    def takes_key(app_key):
        return {"app_key": app_key}

    def takes_host(host=None):
        return {"host": host}

    def takes_both(app_key, host):
        return (app_key, host)

    cases = [
        ("an unreadable signature is called bare", dict, {}),
        ("**kwargs is given both", takes_kwargs,
         {"app_key": "measure", "host": "host"}),
        ("neither declared is given neither", lambda: "screen", "screen"),
        ("only app_key declared", takes_key, {"app_key": "measure"}),
        ("only host declared", takes_host, {"host": "host"}),
        ("both declared", takes_both, ("measure", "host")),
    ]
    for case_id, factory, expected in cases:
        with subtests.test(msg=case_id):
            assert prerun._call(factory, "measure", "host") == expected


# ---------------------------------------------------------------------------
# SegQCBanner
# ---------------------------------------------------------------------------

def test_the_qc_fingerprint_is_paths_times_and_sizes_or_none(
        subtests, banner, monkeypatch, tmp_path):
    """A card listing that fails, or a card that vanished, gives None and forces a read."""
    from spacr import seg_qc

    card = tmp_path / "segmentation_qc_cell.csv"
    card.write_text("well,n\nA01,10\n", encoding="utf-8")
    info = card.stat()

    def unlistable(src):
        raise OSError("no such plate")

    cases = [
        ("an unlistable folder", unlistable, None),
        ("a card that vanished before its stat",
         lambda src: [str(tmp_path / "gone.csv")], None),
        ("no cards at all", lambda src: [], ()),
        ("one card", lambda src: [str(card)],
         ((str(card), info.st_mtime_ns, info.st_size),)),
    ]
    widget = banner(src=str(tmp_path))
    for case_id, finder, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.setattr(seg_qc, "find_scorecards", finder)
            assert widget._fingerprint(str(tmp_path)) == expected


def test_the_qc_banner_reads_the_verdict_only_when_there_is_something_new(
        subtests, banner, monkeypatch, tmp_path, caplog):
    """No source or an unreadable verdict hides the banner; an unchanged card is not re-read."""
    from spacr import seg_qc

    card = tmp_path / "segmentation_qc_cell.csv"
    card.write_text("well,n\nA01,10\n", encoding="utf-8")
    monkeypatch.setattr(seg_qc, "find_scorecards", lambda src: [str(card)])

    def unreadable(_src):
        raise OSError("the scorecard folder is unreadable")

    with subtests.test(msg="no source or a placeholder hides the banner"):
        for placeholder in sorted(prerun._PLACEHOLDERS):
            widget = banner(src=placeholder,
                            reader=lambda src: pytest.fail("read with no src"))
            seen = []
            widget.refreshed.connect(seen.append)
            widget.refresh()
            assert (widget.digest, widget.isHidden(), seen) == (None, True,
                                                                [""])

    with subtests.test(msg="a verdict that cannot be read hides the banner"):
        widget = banner(src=str(tmp_path), reader=unreadable)
        seen = []
        widget.refreshed.connect(seen.append)
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
            widget.refresh()
        assert (widget.digest, widget.isHidden(), seen) == (None, True, [""])
        assert _messages(caplog) == ["could not read the segmentation verdict"]

    with subtests.test(msg="an unchanged card is read once, a rewritten one again"):
        reads = []

        def reader(src):
            reads.append(src)
            return _Digest("ok" if len(reads) == 1 else "fail")

        widget = banner(src=str(tmp_path), reader=reader)
        seen = []
        widget.refreshed.connect(seen.append)
        widget.refresh()
        widget.refresh()
        assert reads == [str(tmp_path)]
        assert seen == ["ok", "ok"]
        assert widget.isHidden() is False
        assert widget._title.text() == "Segmentation QC — passed"

        card.write_text("well,n\nA01,10\nA02,12\n", encoding="utf-8")
        widget.refresh()
        assert reads == [str(tmp_path)] * 2
        assert seen == ["ok", "ok", "fail"]
        assert widget._title.text() == "Segmentation QC — failed"


def test_the_qc_banner_draws_each_verdict_and_copies_the_whole_report(
        subtests, banner, monkeypatch, caplog):
    """Titles, buttons and findings follow the verdict; Copy puts the report on the clipboard or logs."""
    from PySide6.QtWidgets import QApplication
    from spacr import seg_qc

    cases = [
        # (verdict, stale, title, score button, copy enabled)
        ("ok", False, "Segmentation QC — passed", "Score again", True),
        ("warn", False, "Segmentation QC — look at this first", "Score again",
         True),
        ("fail", False, "Segmentation QC — failed", "Score again", True),
        ("missing", False, "Segmentation QC — not run", "Score the masks now",
         False),
        ("error", False, "Segmentation QC — unreadable", "Score again", True),
        ("odd", False, "Segmentation QC", "Score again", True),
        ("ok", True, "Segmentation QC — passed (out of date)",
         "Score the masks now", True),
    ]
    for verdict, stale, title, score, copy in cases:
        with subtests.test(msg=f"draw {verdict} stale={stale}"):
            widget = banner(src="/data/plate1",
                            reader=lambda src, v=verdict, s=stale: _Digest(
                                v, stale=s))
            widget.refresh()
            assert widget._title.text() == title
            assert widget._headline.text() == f"{verdict} headline"
            assert widget._btn_score.text() == score
            assert widget._btn_copy.isEnabled() is copy
            assert widget._findings_box.isHidden() is True
            assert widget._btn_more.isHidden() is True

    with subtests.test(msg="the whole report reaches the clipboard"), \
            monkeypatch.context() as m:
        m.setattr(seg_qc, "format_digest", lambda digest: "the whole report")
        widget = banner(src="/data/plate1", reader=lambda src: _Digest("fail"))
        widget.refresh()
        QApplication.clipboard().setText("before")
        widget._on_copy_clicked()
        assert QApplication.clipboard().text() == "the whole report"

    with subtests.test(msg="a report that cannot be formatted is logged"), \
            monkeypatch.context() as m:
        def explode(digest):
            raise ValueError("that digest has no plates")

        m.setattr(seg_qc, "format_digest", explode)
        widget = banner(src="/data/plate1", reader=lambda src: _Digest("fail"))
        widget.refresh()
        QApplication.clipboard().setText("before")
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
            widget._on_copy_clicked()
        assert _messages(caplog) == ["could not copy the segmentation report"]
        assert QApplication.clipboard().text() == "before"


def test_scoring_the_masks_starts_one_job_or_puts_the_button_back(
        subtests, banner, tmp_path):
    """No source, a running pass, a queued click, unreadable settings and a refused job each end exactly."""
    started = []

    def accept(fn, box, on_done, key):
        started.append((box, key))
        return True

    with subtests.test(msg="no source does nothing"):
        widget = banner(src="")
        widget._score_again = True
        widget._start_job = accept
        widget._on_score_clicked()
        assert (widget.busy, widget._btn_score.isEnabled(),
                widget._score_again, started) == (False, True, False, [])

    with subtests.test(msg="a pass already running swallows the click"):
        widget = banner(src="/data/plate1")
        widget._busy = True
        widget._on_score_clicked()
        assert widget._title.text() == "Segmentation QC"
        assert widget._score_again is False
        assert widget._btn_score.isEnabled() is True

    with subtests.test(msg="a click during a read is queued"):
        widget = banner(src="/data/plate1")
        widget._busy = True
        widget._reading = True
        widget._on_score_clicked()
        assert widget._score_again is True
        assert widget._btn_score.isEnabled() is False
        assert widget._title.text() == "Segmentation QC — scoring the masks…"

    with subtests.test(msg="settings that cannot be collected cost the thresholds"):
        started.clear()
        widget = banner(src=str(tmp_path), collect_raises=True)
        widget._start_job = accept
        widget._on_score_clicked()
        assert started == [({"src": str(tmp_path), "settings": {}}, "seg_qc")]
        assert widget._btn_score.isEnabled() is False
        assert widget._title.text() == "Segmentation QC — scoring the masks…"

    with subtests.test(msg="collected settings are handed to the job"):
        started.clear()
        widget = banner(src=str(tmp_path), settings={"min_cells": 3})
        widget._start_job = accept
        widget._on_score_clicked()
        assert started == [({"src": str(tmp_path),
                             "settings": {"min_cells": 3}}, "seg_qc")]

    with subtests.test(msg="a job that will not start puts the button back"):
        widget = banner(src=str(tmp_path), reader=lambda src: _Digest("ok"))
        widget.refresh()
        widget._start_job = lambda *args, **kwargs: False
        widget._on_score_clicked()
        assert widget._btn_score.isEnabled() is True
        assert widget._title.text() == "Segmentation QC — passed"


# ---------------------------------------------------------------------------
# DiameterPanel
# ---------------------------------------------------------------------------

def test_measuring_hands_the_source_channels_and_field_count_to_one_job(
        subtests, diameter, tmp_path):
    """Missing source or channels say what to fill; the job box carries exact values."""
    no_channel = ("No object channel is set. Fill in cell_channel, "
                  "nucleus_channel or pathogen_channel — they are 0-based "
                  "indices into the sorted channel IDs — and measure again.")
    src = str(tmp_path)

    def box_for(settings):
        return {"src": src, "channels": {"cell": 0}, "n_fields": 5,
                "metadata_type": "cellvoyager", "custom_regex": None,
                "estimator": None, **settings}

    cases = [
        # (id, src, settings, busy, start ok, status, box, button enabled)
        ("no source asks for one", "", {"cell_channel": 0}, False, True,
         "Point src at a plate folder first — there is nothing to measure yet.",
         None, True),
        ("measuring while measuring does nothing", src, {"cell_channel": 0},
         True, True, "", None, True),
        ("no object channel names the three settings", src, {}, False, True,
         no_channel, None, True),
        ("the source, channels and settings are handed over", src,
         {"cell_channel": 1, "nucleus_channel": 0,
          "diameter_estimate_n_fields": 3, "metadata_type": "cq1",
          "custom_regex": r"(?P<well>\w+)"}, False, True,
         "Measuring cell, nucleus across 3 field(s)…",
         box_for({"channels": {"cell": 1, "nucleus": 0}, "n_fields": 3,
                  "metadata_type": "cq1", "custom_regex": r"(?P<well>\w+)"}),
         False),
        ("a field count that is not a number falls back to five", src,
         {"cell_channel": 0, "diameter_estimate_n_fields": "as many as you like"},
         False, True, "Measuring cell across 5 field(s)…", box_for({}), False),
        ("a field count of zero falls back to five", src,
         {"cell_channel": 0, "diameter_estimate_n_fields": 0}, False, True,
         "Measuring cell across 5 field(s)…", box_for({}), False),
        ("a negative field count is raised to one", src,
         {"cell_channel": 0, "diameter_estimate_n_fields": -2}, False, True,
         "Measuring cell across 1 field(s)…", box_for({"n_fields": 1}), False),
        ("a job that will not start says so", src, {"cell_channel": 0}, False,
         False, "Could not start the measurement.", box_for({}), True),
    ]
    for case_id, where, settings, busy, ok, status, box, enabled in cases:
        with subtests.test(msg=case_id):
            panel = diameter(src=where, settings=settings)
            panel._busy = busy
            seen = []
            panel._start_job = (lambda fn, b, on_done, key, ok=ok:
                                seen.append((dict(b), key)) or ok)
            panel._on_measure_clicked()
            assert panel._status.text() == status
            assert panel._status.isHidden() is (status == "")
            assert seen == ([] if box is None else [(box, "diameter")])
            assert panel._btn_measure.isEnabled() is enabled


def test_a_measurement_runs_the_injected_or_the_real_estimator(diameter,
                                                               subtests,
                                                               tmp_path):
    """The job calls the injected estimator with exact arguments, else ``spacr.diameter``."""
    calls = []

    def estimator(src, channels, *, n_fields, metadata_type, custom_regex):
        calls.append((src, dict(channels), n_fields, metadata_type,
                      custom_regex))
        return {"cell": _estimate()}

    def run_inline(panel):
        box = {}
        panel._start_job = lambda fn, b, on_done, key: (
            fn(b), box.update(b), on_done(b), True)[-1]
        panel._on_measure_clicked()
        return box

    with subtests.test(msg="the injected estimator"):
        panel = diameter(src=str(tmp_path), settings={"cell_channel": 0},
                         estimator=estimator)
        run_inline(panel)
        assert calls == [(str(tmp_path), {"cell": 0}, 5, "cellvoyager", None)]
        assert panel.estimates["cell"].diameter == pytest.approx(24.0)

    with subtests.test(msg="the real estimator finds nothing to sample"):
        panel = diameter(src=str(tmp_path), settings={"cell_channel": 0})
        box = run_inline(panel)
        estimate = box["estimates"]["cell"]
        assert estimate.usable is False
        assert "nothing to sample" in estimate.note
        assert panel.estimates["cell"] is estimate


def test_a_finished_measurement_reports_what_it_found_or_why_not(
        subtests, diameter, qtbot, tmp_path):
    """Errors become one exact status line; estimates are shown, kept on the screen and emitted."""
    with subtests.test(msg="an error says why"):
        panel = diameter(src=str(tmp_path))
        heard = []
        panel.estimated.connect(heard.append)
        panel._btn_measure.setEnabled(False)
        panel._on_estimated({"error": "no readable field under that folder"})
        assert panel._status.text() == (
            "Could not measure a diameter from these images: no readable "
            "field under that folder.")
        assert heard == [[]]
        assert panel._btn_measure.isEnabled() is True

    with subtests.test(msg="a failure without a reason still says something"):
        panel = diameter(src=str(tmp_path))
        panel._on_estimated({})
        assert panel._status.text() == (
            "Could not measure a diameter from these images: the estimator "
            "failed.")

    with subtests.test(msg="estimates are kept on the screen that measured"):
        panel = diameter(src=str(tmp_path))
        heard = []
        panel.estimated.connect(heard.append)
        panel._say("Measuring cell across 5 field(s)…")
        estimates = {"cell": _estimate(), "nucleus": _estimate(usable=False)}
        panel._on_estimated({"estimates": estimates})
        assert heard == [["cell"]]
        assert panel._screen._diameter_estimates == estimates
        assert panel._status.isHidden() is True
        assert panel._btn_use_all.isHidden() is False

    with subtests.test(msg="a screen that cannot keep them still shows them"):
        holder = QWidget()
        qtbot.addWidget(holder)
        panel = prerun.DiameterPanel(object(), parent=holder)
        heard = []
        panel.estimated.connect(heard.append)
        panel._on_estimated({"estimates": {"cell": _estimate()}})
        assert heard == [["cell"]]
        assert panel.estimates["cell"].diameter == pytest.approx(24.0)


def test_the_diameter_popup_exists_only_for_a_screen_with_a_diameter_field(
        subtests, qtbot, monkeypatch, caplog):
    """No diameter field gives None, a panel that raises gives None and a log; else a popup."""
    with subtests.test(msg="no diameter field"):
        screen = _Screen(widgets={"src": _field("")})
        qtbot.addWidget(screen)
        assert prerun.diameter_dialog(screen) is None

    with subtests.test(msg="the popup carries the panel for its screen"):
        screen = _Screen(widgets={"src": _field(""),
                                  "nucleus_diameter": _field("")})
        qtbot.addWidget(screen)
        dialog = prerun.diameter_dialog(screen)
        qtbot.addWidget(dialog)
        assert type(dialog) is prerun.DiameterDialog
        assert type(dialog.panel) is prerun.DiameterPanel
        assert dialog.panel._screen is screen
        assert dialog.panel.window() is dialog
        assert dialog.objectName() == "DiameterDialog"
        assert dialog.close_button.objectName() != "DangerButton"
        assert screen.findChildren(prerun.DiameterPanel) == []

    with subtests.test(msg="a popup that raises costs only the popup"), \
            monkeypatch.context() as m:
        screen = _Screen(widgets={"src": _field(""),
                                  "cell_diameter": _field("")})
        qtbot.addWidget(screen)

        def explode(*_a, **_k):
            raise RuntimeError("no")

        m.setattr(prerun, "DiameterPanel", explode)
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
            assert prerun.diameter_dialog(screen) is None
        assert _messages(caplog) == ["could not build the diameter popup for ?"]


# ---------------------------------------------------------------------------
# Registering the Measure factory
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


def test_registering_wraps_and_unregistering_restores_the_measure_factory(
        subtests, factories_restored, qtbot, monkeypatch, caplog):
    """The displaced factory builds the screen; stylesheet failures are logged, never fatal."""
    from spacr.qt import theme

    def refuse(*_a, **_k):
        raise RuntimeError("the theme is mid-reload")

    built = []

    def inner(app_key=None, host=None):
        built.append((app_key, host))
        screen = _Screen(widgets={"src": _field("")})
        qtbot.addWidget(screen)
        screen.app_key = app_key
        return screen

    with subtests.test(msg="a stylesheet that will not register"), \
            monkeypatch.context() as m:
        prerun.unregister()
        m.setattr(theme, "register_widget_qss", refuse)
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
            assert prerun.register() is True
        assert _messages(caplog) == [
            "could not register the pre-run stylesheet"]
        assert factories_restored[prerun.QC_APP] is prerun._prerun_screen
        assert prerun.register() is False

    with subtests.test(msg="the displaced factory builds the screen"):
        prerun.unregister()
        factories_restored[prerun.QC_APP] = inner
        assert prerun.register() is True
        assert prerun._INNER[prerun.QC_APP] is inner
        screen = factories_restored[prerun.QC_APP](app_key=prerun.QC_APP,
                                                   host="window")
        assert built == [(prerun.QC_APP, "window")]
        assert prerun.qc_banner(screen) is not None

    with subtests.test(msg="a stylesheet that will not unregister"), \
            monkeypatch.context() as m:
        m.setattr(theme, "unregister_widget_qss", refuse)
        caplog.clear()
        with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
            assert prerun.unregister() == 1
        assert _messages(caplog) == ["could not remove the pre-run stylesheet"]
        assert factories_restored[prerun.QC_APP] is inner
        assert prerun.QC_APP not in prerun._INNER
        assert prerun.unregister() == 0

    with subtests.test(msg="a key with nothing displaced is removed"):
        factories_restored.pop(prerun.QC_APP, None)
        assert prerun.register() is True
        assert prerun.unregister() == 1
        assert prerun.QC_APP not in factories_restored
