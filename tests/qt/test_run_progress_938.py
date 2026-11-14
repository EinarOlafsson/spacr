"""Item 938: run progress shows percent done and time left, on the module and on Home.

Covers the bridge's parsing of ``Time_left`` and of multi-stage progress,
the estimator (warm-up, smoothing, pause) and the text the module bar and
the Home banner show.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QLabel, QProgressBar  # noqa: E402

from spacr.qt import bridge as B  # noqa: E402
from spacr.qt import run_progress as RP  # noqa: E402

pytestmark = pytest.mark.qt


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def _line(done, total, label, minutes=None):
    tail = "None" if minutes is None else (
        f"Time/image: 1.000sec, Time_left: {minutes:.3f} min.")
    return f"Progress: {done}/{total}, operation_type: {label}, {tail}"


@pytest.fixture(autouse=True)
def _english(monkeypatch):
    from spacr.qt import i18n
    monkeypatch.setattr(i18n, "current_language", lambda: "en")


@pytest.fixture(autouse=True)
def _isolated_worker_journal(monkeypatch, tmp_path):
    from spacr import run_journal
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)


@pytest.fixture
def handle():
    worker = B.PipelineWorker(lambda settings: None, {})
    made = B.RunHandle("measure", worker, QThread())
    yield made
    made.worker = None
    made.thread = None


def test_print_progress_output_parses_with_its_time_left(capsys):
    from spacr.utils import print_progress

    print_progress(3, 12, n_jobs=1, time_ls=[2.0, 2.0, 2.0],
                   operation_type="Measure and Crop")
    sample, = RP.parse_progress(capsys.readouterr().out)

    assert (sample.done, sample.total) == (3, 12)
    assert sample.label == "Measure and Crop"
    assert sample.seconds_left == pytest.approx(18.0, abs=0.1)


def test_a_line_without_time_left_or_before_the_first_unit_has_none():
    no_time, first, bare = RP.parse_progress(
        _line(4, 10, "Normalizing") + "\n" + _line(0, 10, "x", minutes=0.0)
        + "\nProgress: 7/9 (fit)\n")

    assert no_time.seconds_left is None and no_time.label == "Normalizing"
    assert first.seconds_left is None
    assert (bare.done, bare.total, bare.label) == (7, 9, None)


def test_mask_gpu_lines_are_labelled_and_text_without_progress_is_skipped():
    gpu, = RP.parse_progress(
        "loading model\n[mask GPUs] cell Progress: 4/16 archives, failed 0"
        " | GPU 0 busy 2/8\n")

    assert (gpu.done, gpu.total, gpu.label) == (4, 16, RP.MASK_GPU_LABEL)
    assert RP.parse_progress("") == []


def test_no_estimate_before_warm_up_then_one_after_thirty_seconds():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    clock.tick(10)
    record.observe(RP.ProgressSample(1, 100, "a"))
    assert record.fraction() == pytest.approx(0.01)
    assert record.seconds_left() is None
    assert RP.format_time_left(record.seconds_left()) == "estimating…"

    clock.tick(20)
    record.observe(RP.ProgressSample(3, 100, "a"))
    assert record.warmed_up()
    assert record.seconds_left() == pytest.approx(970.0)


def test_five_percent_ends_the_warm_up_before_thirty_seconds():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    clock.tick(5)
    record.observe(RP.ProgressSample(10, 100, "a"))

    assert record.warmed_up()
    assert record.seconds_left() == pytest.approx(45.0)


def test_the_rate_is_smoothed_so_one_fast_step_moves_it_only_partly():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    for done in range(1, 6):
        clock.tick(10)
        record.observe(RP.ProgressSample(done * 2, 100, "a"))
    steady = record.seconds_left()
    assert steady == pytest.approx(90 * 5.0)

    clock.tick(10)
    record.observe(RP.ProgressSample(30, 100, "a"))
    jumped = record.seconds_left()
    newest_step_alone = 0.70 / (0.20 / 10.0)
    old_rate_alone = 0.70 / 0.002

    assert newest_step_alone < jumped < old_rate_alone


def test_paused_time_does_not_count_and_the_rate_carries_on():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    clock.tick(10)
    record.observe(RP.ProgressSample(10, 100, "a"))
    before = record.seconds_left()

    record.pause()
    clock.tick(500)
    assert record.is_paused()
    assert record.seconds_left() == pytest.approx(before)
    assert record.active_seconds() == pytest.approx(10)

    record.resume()
    clock.tick(10)
    record.observe(RP.ProgressSample(20, 100, "a"))
    assert record.active_seconds() == pytest.approx(20)
    assert record.seconds_left() == pytest.approx(80.0)


def test_worker_time_left_is_used_and_counts_down():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    clock.tick(5)
    record.observe(RP.ProgressSample(10, 100, "a", seconds_left=600.0))
    clock.tick(60)

    assert record.seconds_left() == pytest.approx(540.0)


def test_declared_stages_weight_one_overall_bar_that_never_goes_back():
    clock = _Clock()
    record = RP.RunProgress("measure", clock=clock)
    assert record.declares_stages

    record.observe(RP.ProgressSample(5, 10, "Merging Arrays"))
    assert record.fraction() == pytest.approx(0.05)
    record.observe(RP.ProgressSample(0, 10, "Measure and Crop"))
    assert record.fraction() == pytest.approx(0.10)
    record.observe(RP.ProgressSample(5, 10, "Measure and Crop"))
    assert record.fraction() == pytest.approx(0.55)
    record.observe(RP.ProgressSample(1, 10, "something nested"))
    assert record.fraction() == pytest.approx(0.55)
    record.observe(RP.ProgressSample(4, 10, "Measure and Crop"))
    assert record.fraction() == pytest.approx(0.55)
    assert record.stage_label == "Measure and Crop"


def test_worker_time_left_counts_only_on_the_last_declared_stage():
    clock = _Clock()
    record = RP.RunProgress("measure", clock=clock)
    clock.tick(40)
    record.observe(RP.ProgressSample(5, 10, "Merging Arrays",
                                     seconds_left=7.0))
    assert record._worker_left is None
    record.observe(RP.ProgressSample(5, 10, "Measure and Crop",
                                     seconds_left=300.0))
    assert record.seconds_left() == pytest.approx(300.0)


def test_mask_stages_take_role_lines_and_skip_nested_merge_lines():
    record = RP.RunProgress("mask", clock=_Clock())
    record.observe(RP.ProgressSample(1, 3, "cell_mask_gen"))
    after_role = record.fraction()
    record.observe(RP.ProgressSample(9, 10, "merge_nucleus"))

    assert record.fraction() == after_role
    assert 0.2 < after_role < 0.6


def test_an_undeclared_module_restarts_per_stage_and_names_it():
    clock = _Clock()
    record = RP.RunProgress("", clock=clock)
    clock.tick(5)
    record.observe(RP.ProgressSample(10, 10, "first"))
    record.observe(RP.ProgressSample(1, 10, "second"))

    assert record.fraction() == pytest.approx(0.1)
    assert record.stage_label == "second"
    assert record.seconds_left() is None


def test_custom_stage_weights_are_normalised_and_bad_totals_ignored():
    record = RP.RunProgress("", clock=_Clock(),
                            stages=((("a",), 1.0), (("b",), 3.0)))
    record.observe(RP.ProgressSample(1, 0, "a"))
    assert record.fraction() is None
    record.observe(RP.ProgressSample(1, 1, "b"))
    assert record.fraction() == 1.0
    assert record.seconds_left() == 0.0


@pytest.mark.parametrize("seconds, text", [
    (None, "estimating…"),
    (20, "less than a minute left"),
    (12 * 60 + 10, "about 12 min left"),
    (2 * 3600 + 9 * 60, "about 2 h 10 min left"),
    (3 * 3600 + 60, "about 3 h left"),
])
def test_time_left_is_rounded_for_display(seconds, text):
    assert RP.format_time_left(seconds) == text


def test_the_handle_parses_time_left_and_multi_stage_lines(handle):
    handle._on_line(_line(5, 10, "Merging Arrays") + "\n"
                    + _line(5, 10, "Measure and Crop", minutes=3.0) + "\n")

    assert handle.progress == (5, 10)
    assert handle.fraction() == pytest.approx(0.55)
    assert handle.seconds_left() == pytest.approx(180.0, abs=2.0)


def test_a_paused_gate_freezes_the_handles_estimate(handle):
    clock = _Clock()
    handle.run_progress = RP.RunProgress("", clock=clock)
    clock.tick(5)
    handle._on_line(_line(10, 100, "a") + "\n")
    before = handle.seconds_left()
    handle.gate.pause()
    handle.fraction()
    clock.tick(300)

    assert handle.seconds_left() == pytest.approx(before)
    handle.gate.resume()
    handle.fraction()
    clock.tick(1)
    assert handle.seconds_left() == pytest.approx(before - 1)


def test_an_external_job_reports_through_the_same_record():
    job = B._ExternalJobHandle("install")
    assert job.fraction() is None
    job.report(3, 4, "unpacking")

    assert job.fraction() == pytest.approx(0.75)
    assert job.seconds_left() is None or job.seconds_left() >= 0


def test_the_home_banner_shows_percent_and_time_left(qapp, handle):
    from spacr.qt.widgets.home import RunningBanner

    clock = _Clock()
    handle.run_progress = RP.RunProgress("measure", clock=clock)
    clock.tick(40)
    handle._on_line(_line(5, 10, "Measure and Crop", minutes=12.0) + "\n")
    banner = RunningBanner(lambda key: None, {"measure": "Measure"})
    banner.bind(handle)

    assert banner._bar.maximum() == 100
    assert banner._bar.value() == 55
    assert banner._bar.text() == "55 %"
    assert "about 12 min left" in banner._sub.text()
    assert "5 of 10" in banner._sub.text()
    banner.deleteLater()


def test_the_home_banner_keeps_a_busy_bar_without_progress(qapp, handle):
    from spacr.qt.widgets.home import RunningBanner

    banner = RunningBanner(lambda key: None, {})
    banner.bind(handle)

    assert banner._bar.maximum() == 0
    assert "left" not in banner._sub.text()
    banner.deleteLater()


def test_the_module_bar_shows_the_same_text_as_the_banner(qapp, handle):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.home import RunningBanner

    clock = _Clock()
    handle.run_progress = RP.RunProgress("measure", clock=clock)
    clock.tick(40)
    handle._on_line(_line(5, 10, "Measure and Crop", minutes=12.0) + "\n")
    screen = SimpleNamespace(_progress=QProgressBar(), _progress_eta=QLabel(),
                             _progress_handle=handle)
    AppScreen._refresh_run_progress(screen)
    banner = RunningBanner(lambda key: None, {})
    banner.bind(handle)

    assert screen._progress.text() == banner._bar.text() == "55 %"
    assert screen._progress_eta.text() == "about 12 min left"
    assert not screen._progress_eta.isHidden()
    assert screen._progress_eta.text() in banner._sub.text()
    banner.deleteLater()


def test_the_module_bar_follows_the_registered_run_and_stops(qapp, monkeypatch):
    from PySide6.QtCore import QTimer

    from spacr.qt.screens.app_screen import AppScreen

    reg = B.RunRegistry()
    monkeypatch.setattr(B, "_REGISTRY", reg)
    worker = B.PipelineWorker(lambda settings: None, {})
    run = reg.register(B.RunHandle("measure", worker, QThread()))
    screen = SimpleNamespace(_progress=QProgressBar(), _progress_eta=QLabel(),
                             _progress_handle=None, _progress_timer=QTimer())
    screen._refresh_run_progress = lambda: AppScreen._refresh_run_progress(
        screen)

    AppScreen._follow_run_progress(screen, worker)
    assert screen._progress_handle is run
    assert screen._progress_timer.isActive()
    worker.line_ready.emit(_line(5, 10, "Measure and Crop") + "\n")
    qapp.processEvents()
    assert screen._progress.value() == 55

    AppScreen._stop_following_run_progress(screen)
    assert screen._progress_handle is None
    assert not screen._progress_timer.isActive()
    assert screen._progress.maximum() == 0
    run.worker = None
    run.thread = None


def test_a_progress_tuple_set_by_its_owner_still_reads_as_a_fraction(handle):
    handle.progress = (1, 4)

    assert handle.fraction() == pytest.approx(0.25)
    assert handle.seconds_left() is None
