"""The pre-run panels when the parts around them misbehave.

Everything here is a failure the two panels are supposed to absorb: a
teardown callback that throws, a QC helper that cannot resolve its targets,
a field browser whose C++ half has already gone, a platform with no
clipboard, a settings model that predates ``set_value_for_key``. The rule
each one serves is the module's own: a screen that opens without the banner
is better than a screen that does not open, and nothing advisory may reach
an exception the caller has to handle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QDialog,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from spacr import seg_qc  # noqa: E402
from spacr.qt import prerun  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Stand-ins for the parts of an AppScreen these panels reach for
# ---------------------------------------------------------------------------

class _Model:
    """A settings model with the modern ``set_value_for_key`` contract."""

    def __init__(self, widgets=None, settings=None):
        self._widgets = dict(widgets or {})
        self._settings = dict(settings or {})
        self.written = {}

    def collect(self):
        return dict(self._settings)

    def set_value_for_key(self, key, value):
        self.written[key] = value
        return True


class _ModelWithoutSetter:
    """A settings model that only exposes its widgets.

    ``apply`` has a second route for exactly this: write through the
    screen's ``_apply_value`` instead of through the model.
    """

    def __init__(self, widgets=None, settings=None):
        self._widgets = dict(widgets or {})
        self._settings = dict(settings or {})

    def collect(self):
        return dict(self._settings)


class _Screen(QWidget):
    """A screen shaped like ``AppScreen`` for the parts prerun touches."""

    def __init__(self, model=None, with_applier=True):
        super().__init__()
        self._thread = None
        if model is not None:
            self._settings_model = model
        self._applied = []
        if with_applier:
            self._apply_value = self._record_applied

    def _record_applied(self, widget, value):
        self._applied.append((widget, value))
        widget.setText(str(value))


class _Watched(QWidget):
    """A widget that keeps the show and hide events actually delivered to it.

    A consumed event is the only thing this can see that ``isVisible`` cannot:
    ``hide()`` has already flipped the visibility flag by the time the Hide
    event is sent, so a filter that swallowed it would leave a hidden widget
    that never ran its own ``hideEvent``.
    """

    def __init__(self):
        super().__init__()
        self.delivered = []

    def showEvent(self, event):                 # noqa: N802 - Qt override
        self.delivered.append("show")
        super().showEvent(event)

    def hideEvent(self, event):                 # noqa: N802 - Qt override
        self.delivered.append("hide")
        super().hideEvent(event)


def _field(text=""):
    line = QLineEdit()
    line.setText(text)
    return line


def _stale_targets():
    """Browser targets left over from a plate the banner has moved on from."""
    from spacr.qt.widgets.qc_field_browser import QCFieldTarget

    return (QCFieldTarget(field="A01_f01", plate_root="/gone/plate0",
                          merged_dir="/gone/plate0/merged"),)


def _estimate(object_type="cell", diameter=24.0):
    from spacr.diameter import DiameterEstimate

    return DiameterEstimate(
        object_type=object_type, diameter=diameter, low=diameter - 4,
        high=diameter + 4, n_objects=120, n_fields=5,
        method="threshold_otsu", confidence="high",
        note="measured from five fields")


# ---------------------------------------------------------------------------
# A digest with a flagged finding, so the field links are drawn
# ---------------------------------------------------------------------------

def _write_field(plate: Path, field: str) -> None:
    merged = plate / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:16, :20]
    array = np.zeros((16, 20, 4), dtype=np.uint16)
    array[..., 0] = xx * 100
    array[..., 1] = yy * 120
    cell = np.zeros((16, 20), dtype=np.uint16)
    cell[4:12, 5:15] = 1
    array[..., 2] = cell
    array[..., 3] = cell
    np.save(merged / f"{field}.npy", array)
    (merged / ".spacr_plane_layout.json").write_text(json.dumps({
        "version": 1,
        "intensity_channels": ["DNA", "green"],
        "mask_plane_order": ["cell"],
        "mask_dims": {"cell": 2},
    }), encoding="utf-8")
    stack = plate / "norm_channel_stack" / "cell_mask_stack"
    stack.mkdir(parents=True, exist_ok=True)
    np.save(stack / f"{field}.npy", cell)


def _flagged_digest(plate: Path):
    """A two-field ``fail`` digest whose finding names both fields."""
    first, second = "plate1_A01_1", "plate1_A02_1"
    _write_field(plate, first)
    _write_field(plate, second)
    qc_dir = plate / "qc"
    qc_dir.mkdir(exist_ok=True)
    rows = [
        seg_qc.FieldQC(first, "cell", 1, ["under_segmented"], {}, "fail",
                       "One cell region covers most of the field."),
        seg_qc.FieldQC(second, "cell", 2, ["tiny_objects"], {}, "warn",
                       "Most cell regions are unusually small."),
    ]
    card = seg_qc.Scorecard(
        str(qc_dir / "segmentation_qc_cell.csv"), "cell", rows)
    finding = seg_qc.Finding(
        severity="fail", kind="flag", flag="under_segmented",
        headline="Two fields need visual review.", plate="plate1",
        object_type="cell", fields=(first, second), n_fields=2,
        detail="Both fields carry the same flag.",
        fix="Lower the cell diameter and score again.")
    return seg_qc.QCDigest(
        root=str(plate), verdict="fail", headline=finding.headline,
        scorecards=[card], findings=[finding])


@pytest.fixture
def banner(qtbot, tmp_path):
    """A drawn banner on a minimal screen, with its digest already read."""
    screen = _Screen(_Model(widgets={"src": _field(str(tmp_path))}))
    qtbot.addWidget(screen)
    made = prerun.SegQCBanner(screen)
    made._digest = _flagged_digest(tmp_path / "plate1")
    made._draw()
    return made


# ---------------------------------------------------------------------------
# _ShowFilter — the hide half
# ---------------------------------------------------------------------------

def test_a_hide_callback_that_raises_does_not_consume_the_hide_event(
        qtbot, caplog):
    """The teardown half of the filter is as fenced as the refresh half."""
    shown, hidden = [], []

    def _blow_up():
        hidden.append(1)
        raise RuntimeError("the field browser was already gone")

    watched = _Watched()
    qtbot.addWidget(watched)
    watched.installEventFilter(prerun._ShowFilter(
        lambda: shown.append(1), watched, on_hide=_blow_up))

    watched.show()
    qtbot.waitExposed(watched)
    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        watched.hide()

    assert hidden, "the hide callback never ran"
    assert "pre-run cleanup failed on hide" in caplog.text
    assert watched.isVisible() is False
    # THE EVENT WENT ON TO THE WIDGET. A filter that returned True after the
    # raising callback would swallow it, and the widget would be hidden
    # without ever having run its own teardown.
    assert watched.delivered == ["show", "hide"], "the Hide event was consumed"
    # The filter is still live: a second Show still refreshes, and that one
    # reaches the widget too.
    watched.show()
    qtbot.waitExposed(watched)
    assert len(shown) == 2
    assert watched.delivered == ["show", "hide", "show"]


# ---------------------------------------------------------------------------
# _JobMixin
# ---------------------------------------------------------------------------

def test_a_settled_job_with_no_completion_callback_still_clears_busy(banner):
    """A stray ``finished`` must leave the panel usable, not wedged busy."""
    banner._busy = True
    banner._on_done = None

    banner._job_settled(True)

    assert banner.busy is False


# ---------------------------------------------------------------------------
# Findings, and the QC helpers behind their links
# ---------------------------------------------------------------------------

def test_clearing_the_findings_drops_spacers_as_well_as_widgets(banner):
    """``takeAt`` yields layout items, and not every item owns a widget."""
    banner._findings_layout.addStretch(1)
    assert banner._findings_layout.count() > len(banner._digest.findings)

    banner._draw_findings(banner._digest)

    assert banner._findings_layout.count() == len(banner._digest.findings)


def test_browser_targets_that_cannot_be_resolved_do_not_cost_the_findings(
        qtbot, tmp_path, caplog, monkeypatch):
    """The findings are the point; the links into the browser are a bonus."""
    from spacr.qt.widgets import qc_field_browser

    def _refuse(_digest):
        raise RuntimeError("the merged plane layout is unreadable")

    monkeypatch.setattr(qc_field_browser, "targets_from_digest", _refuse)

    screen = _Screen(_Model(widgets={"src": _field(str(tmp_path))}))
    qtbot.addWidget(screen)
    made = prerun.SegQCBanner(screen)
    digest = _flagged_digest(tmp_path / "plate1")
    # Targets an earlier, readable digest left behind. They name fields of a
    # run this banner is no longer showing, so a redraw that cannot resolve
    # new ones has to DROP them: keeping them would leave links that open the
    # browser at the wrong plate.
    made._field_targets = _stale_targets()

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        made._digest = digest
        made._draw()

    assert made._field_targets == ()
    assert "could not resolve segmentation-QC browser targets" in caplog.text
    headlines = [lbl.text() for lbl in made.findChildren(QLabel)]
    assert any("Two fields need visual review." in text for text in headlines)
    assert made.findChildren(QLabel, "QCFieldLinks") == []


def test_field_links_that_cannot_be_resolved_leave_the_finding_readable(
        qtbot, tmp_path, caplog, monkeypatch):
    from spacr.qt.widgets import qc_field_browser

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("this finding names no field")

    monkeypatch.setattr(qc_field_browser, "finding_targets", _refuse)

    screen = _Screen(_Model(widgets={"src": _field(str(tmp_path))}))
    qtbot.addWidget(screen)
    made = prerun.SegQCBanner(screen)

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        made._digest = _flagged_digest(tmp_path / "plate1")
        made._draw()

    assert "could not resolve segmentation-QC field links" in caplog.text
    assert made.findChildren(QLabel, "QCFieldLinks") == []
    texts = [lbl.text() for lbl in made.findChildren(QLabel)]
    assert any("Two fields need visual review." in text for text in texts)


# ---------------------------------------------------------------------------
# Is a run in flight?
# ---------------------------------------------------------------------------

def test_a_deleted_worker_thread_reads_as_no_run_in_flight(banner):
    """The screen keeps ``_thread`` after Qt has already freed the object."""
    dead = QThread()
    shiboken6.delete(dead)
    banner._screen._thread = dead

    assert banner._measure_run_active() is False


def test_a_live_worker_thread_reads_as_a_run_in_flight(banner, qtbot):
    thread = QThread()
    try:
        thread.start()
        qtbot.waitUntil(thread.isRunning, timeout=5000)
        banner._screen._thread = thread
        assert banner._measure_run_active() is True
    finally:
        thread.quit()
        thread.wait(5000)


# ---------------------------------------------------------------------------
# Opening the field browser
# ---------------------------------------------------------------------------

class _RecordingBrowser(QDialog):
    """A stand-in for ``QCFieldBrowser`` that records how it was opened."""

    def __init__(self, targets, *, initial_field="", initial_plate_root="",
                 run_active=None, parent=None):
        super().__init__(parent)
        self.targets = tuple(targets)
        self.initial_field = initial_field
        self.initial_plate_root = initial_plate_root
        self.run_active = run_active


def test_a_link_with_no_target_behind_it_opens_nothing(banner, caplog):
    """The link map is keyed by href; an href it does not know opens no window."""
    # The digest is present and the targets are resolved, so the ONLY reason
    # to open nothing is the missing target itself.
    assert banner._digest is not None
    assert banner._field_targets

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        banner._on_field_link(None)

    assert banner._field_browser is None
    assert banner.findChildren(QDialog) == []
    # An href with no target behind it is an ordinary state, not a fault: the
    # handler returns before it can reach for a field a missing target has.
    assert "could not open the segmentation-QC field browser" not in caplog.text


def test_a_browser_whose_window_has_gone_is_replaced_rather_than_reused(
        banner, monkeypatch):
    """``open_at`` on a closed dialog raises; the click must still land."""
    from spacr.qt.widgets import qc_field_browser
    monkeypatch.setattr(qc_field_browser, "QCFieldBrowser", _RecordingBrowser)

    class _Gone:
        def open_at(self, _field, _plate_root):
            raise RuntimeError("Internal C++ object already deleted")

    banner._field_browser = _Gone()
    target = qc_field_browser.targets_from_digest(banner._digest)[1]

    banner._on_field_link(target)

    made = banner._field_browser
    assert isinstance(made, _RecordingBrowser)
    assert made.initial_field == target.field
    assert made.initial_plate_root == target.plate_root
    assert made.isVisible() is True


def test_with_no_test_factory_the_banner_opens_the_modules_own_browser(
        banner, monkeypatch):
    """No ``_field_browser_factory`` is set in production; the default is used."""
    from spacr.qt.widgets import qc_field_browser
    monkeypatch.setattr(qc_field_browser, "QCFieldBrowser", _RecordingBrowser)

    target = qc_field_browser.targets_from_digest(banner._digest)[0]
    # Targets are resolved on demand when the draw could not cache them.
    banner._field_targets = ()

    banner._on_field_link(target)

    made = banner._field_browser
    assert isinstance(made, _RecordingBrowser)
    assert [t.field for t in made.targets] == [
        "plate1_A01_1", "plate1_A02_1"], "the targets were not re-resolved"
    assert made.run_active == banner._measure_run_active


def test_a_browser_that_will_not_open_leaves_the_banner_intact(
        banner, caplog, monkeypatch):
    from spacr.qt.widgets import qc_field_browser

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("no display for the field browser")

    monkeypatch.setattr(qc_field_browser, "QCFieldBrowser", _refuse)
    target = qc_field_browser.targets_from_digest(banner._digest)[0]

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        banner._on_field_link(target)

    assert banner._field_browser is None
    assert "could not open the segmentation-QC field browser" in caplog.text
    assert banner.isEnabled() is True


def test_closing_a_browser_whose_window_has_gone_still_forgets_it(banner):
    """Leaving Measure closes the browser; a dead one must not raise."""
    class _Gone:
        def close(self):
            raise RuntimeError("Internal C++ object already deleted")

    banner._field_browser = _Gone()

    banner._close_field_browser()

    assert banner._field_browser is None


# ---------------------------------------------------------------------------
# Copying the report
# ---------------------------------------------------------------------------

def test_a_platform_with_no_clipboard_copies_nothing_and_says_nothing(
        banner, monkeypatch, caplog):
    """``QGuiApplication.clipboard`` is None where the platform has none."""
    from PySide6.QtWidgets import QApplication

    class _NoClipboard:
        @staticmethod
        def clipboard():
            return None

    held = "whatever the user had copied before"
    QApplication.clipboard().setText(held)
    monkeypatch.setattr(prerun, "QApplication", _NoClipboard)

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        banner._on_copy_clicked()

    # Nothing was written, and nothing went wrong not writing it: a platform
    # with no clipboard is a platform, not a failure to report.
    assert QApplication.clipboard().text() == held
    assert "could not copy the segmentation report" not in caplog.text


def test_copying_puts_the_whole_report_on_the_clipboard(banner):
    from PySide6.QtWidgets import QApplication

    QApplication.clipboard().setText("")
    banner._on_copy_clicked()

    text = QApplication.clipboard().text()
    assert "Two fields need visual review." in text


# ---------------------------------------------------------------------------
# DiameterPanel — clearing rows, and the two write routes
# ---------------------------------------------------------------------------

@pytest.fixture
def panel_without_setter(qtbot):
    """A panel over a settings model that has no ``set_value_for_key``."""
    widgets = {"src": _field(""), "cell_diameter": _field("")}
    screen = _Screen(_ModelWithoutSetter(widgets=widgets))
    qtbot.addWidget(screen)
    made = prerun.DiameterPanel(screen)
    made._estimates = {"cell": _estimate()}
    return made


def test_clearing_the_rows_drops_spacers_as_well_as_widgets(
        qtbot, panel_without_setter):
    panel = panel_without_setter
    panel._draw_rows()
    rows = panel._rows_layout.count()
    panel._rows_layout.addStretch(1)
    assert panel._rows_layout.count() == rows + 1

    panel._draw_rows()

    assert panel._rows_layout.count() == rows


def test_a_model_without_a_setter_is_written_through_the_screens_applier(
        panel_without_setter):
    """The fallback route exists so an older screen is still writable."""
    panel = panel_without_setter
    field = panel._screen._settings_model._widgets["cell_diameter"]

    assert panel.apply("cell") is True

    assert field.text() == "24"
    assert panel._screen._applied == [(field, 24)]
    assert "cell_diameter set to 24 px." in panel._status.text()


def test_a_screen_with_no_way_to_write_reports_the_failure_rather_than_lying(
        qtbot, caplog):
    """Neither route available: ``apply`` says False and writes nothing.

    The field to write is there; what is missing is anything to write it
    with. ``_apply_value`` is an attribute like any other, so a screen can
    carry one that is not a function -- and the panel has to read that as
    "no route" rather than calling it and reporting the crash.
    """
    field = _field("")
    screen = _Screen(_ModelWithoutSetter(widgets={"src": _field(""),
                                                  "cell_diameter": field}),
                     with_applier=False)
    screen._apply_value = "not a function"
    qtbot.addWidget(screen)
    panel = prerun.DiameterPanel(screen)
    panel._estimates = {"cell": _estimate()}

    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        assert panel.apply("cell") is False

    assert panel._screen._applied == []
    assert field.text() == ""
    assert panel._status.text() == ""
    # NOT ATTEMPTED, not attempted-and-failed: calling the non-function would
    # log a write failure for a write the panel never had a way to make.
    assert "could not write cell_diameter" not in caplog.text


def test_use_all_reports_nothing_when_nothing_could_be_written(qtbot):
    screen = _Screen(_ModelWithoutSetter(widgets={"src": _field("")}),
                     with_applier=False)
    qtbot.addWidget(screen)
    panel = prerun.DiameterPanel(screen)
    panel._estimates = {"cell": _estimate()}

    panel._on_use_all_clicked()

    assert panel._status.text() == ""


# ---------------------------------------------------------------------------
# Decoration is never load-bearing
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_make_a_widget_transparent_returns_it_anyway(
        qtbot, caplog, monkeypatch):
    """``_transparent`` is used inline; it must always hand the widget back."""
    from spacr.qt import theme

    def _refuse(_widget):
        raise RuntimeError("the palette is not resolved yet")

    monkeypatch.setattr(theme, "make_transparent", _refuse)
    widget = QWidget()
    qtbot.addWidget(widget)

    with caplog.at_level("DEBUG", logger="spacr.qt.prerun"):
        assert prerun._transparent(widget) is widget

    assert "could not make" in caplog.text


def test_a_panel_still_builds_when_the_theme_will_not_make_it_transparent(
        qtbot, monkeypatch):
    """The findings box is built through ``_transparent``; it must survive."""
    from spacr.qt import theme

    def _refuse(_widget):
        raise RuntimeError("the palette is not resolved yet")

    monkeypatch.setattr(theme, "make_transparent", _refuse)
    screen = _Screen(_Model(widgets={"src": _field("")}))
    qtbot.addWidget(screen)

    made = prerun.SegQCBanner(screen)

    assert made._findings_box is not None
    assert made.objectName() == prerun.QC_OBJECT_NAME


def test_a_theme_registry_that_refuses_the_stylesheet_still_imports(
        caplog, monkeypatch):
    """A stylesheet that cannot be registered costs the panel its background.

    The module body is executed again under a throwaway name — deliberately
    kept out of ``sys.modules``, so the real ``SegQCBanner`` class every
    live screen holds is not swapped out from under it.
    """
    import importlib.util

    from spacr.qt import theme

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("the widget QSS registry is closed")

    monkeypatch.setattr(theme, "register_widget_qss", _refuse)

    spec = importlib.util.spec_from_file_location(
        "spacr.qt._prerun_refused_qss", prerun.__file__)
    module = importlib.util.module_from_spec(spec)
    with caplog.at_level("ERROR", logger="spacr.qt.prerun"):
        spec.loader.exec_module(module)

    assert "could not register the pre-run stylesheet at import" in caplog.text
    assert module.QC_OBJECT_NAME == prerun.QC_OBJECT_NAME
    assert issubclass(module.SegQCBanner, QWidget)


class _OpenableBrowser(_RecordingBrowser):
    """A browser that can be told to jump to a field without being rebuilt."""

    accepts = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opened_at = None

    def open_at(self, field, plate_root=""):
        self.opened_at = (field, plate_root)
        return self.accepts


def test_a_second_link_reuses_the_open_browser_instead_of_a_second_window(
        banner, monkeypatch):
    """One window, moved — two windows onto the same plate is a bug report."""
    from spacr.qt.widgets import qc_field_browser
    monkeypatch.setattr(qc_field_browser, "QCFieldBrowser", _OpenableBrowser)

    targets = qc_field_browser.targets_from_digest(banner._digest)
    banner._on_field_link(targets[0])
    first = banner._field_browser
    assert isinstance(first, _OpenableBrowser)
    first.hide()

    banner._on_field_link(targets[1])

    assert banner._field_browser is first, "a second window was opened"
    assert first.opened_at == (targets[1].field, targets[1].plate_root)
    assert first.isVisible() is True


def test_a_browser_that_cannot_reach_the_field_is_rebuilt_around_it(
        banner, monkeypatch):
    """``open_at`` answering False means this window does not hold that field."""
    from spacr.qt.widgets import qc_field_browser
    monkeypatch.setattr(qc_field_browser, "QCFieldBrowser", _OpenableBrowser)

    targets = qc_field_browser.targets_from_digest(banner._digest)
    banner._on_field_link(targets[0])
    first = banner._field_browser
    first.accepts = False

    banner._on_field_link(targets[1])

    second = banner._field_browser
    assert second is not first, "a browser that refused the field was reused"
    assert second.initial_field == targets[1].field
