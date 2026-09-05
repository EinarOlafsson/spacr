"""Opening Measure never stats the user's src on the GUI thread.

THE FREEZE, 2026-09-04. `install_qc_banner` builds a `SegQCBanner` while
`MainWindow._build_screen` constructs the Measure screen, and finished by
calling `banner.refresh()` synchronously. That went:

    SegQCBanner.refresh
      -> _fingerprint(src)
        -> spacr.seg_qc.find_scorecards(src)
          -> qc_roots  -> os.path.isdir(<user's src>)
                       -> os.listdir(<user's src>) per plate child
                       -> os.path.isdir(<child>/qc)
          -> os.listdir(<root>/qc), os.path.isfile per card
        -> os.stat(card)                              per card
      -> _read(src) -> spacr.seg_qc.read_digest       opens and parses each CSV

and `src` is whatever folder the user last worked in -- a module screen opens
on it. One of the maintainer's was an ``autofs`` mount whose share was
asleep, and a single `os.path.exists` under it had NOT RETURNED AFTER TWENTY
SECONDS: the stat is what triggers the automount. So the whole interface was
frozen for as long as that took, every time Measure was opened, and it left
no traceback because a stalled event loop is not a crash. It reached the
maintainer as "opening map barcodes crashes spacr", hover flicker, and
glimpses of other screens showing through Home.

The same three GUI-thread entries all landed on that one method: screen
construction, the `_ShowFilter` Show event on every return to the screen, and
the 450 ms debounce armed by typing in the src field.

WHAT IS ASSERTED HERE is only the property the freeze violated -- that the
call returns -- plus the thing that makes the fix worth having: the verdict
still arrives, a moment later, with every word of it intact.
"""
from __future__ import annotations

import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from spacr.qt import prerun

#: Longer than any human would call responsive, far shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 6.0

#: What "did not block" means. Generous enough to survive a loaded CI box,
#: small enough that it cannot pass by accident when SLOW_S is being waited.
FAST_S = 1.0

#: Lets a parked worker go at teardown. A QThread destroyed while it is still
#: running ABORTS the process -- no exception, no traceback -- so a test that
#: deliberately strands a worker in a sleeping mount has to be able to wake it
#: again, and cannot simply walk away from it.
_RELEASE = threading.Event()


class _Model:
    """The one attribute `prerun._widgets` reads off a settings model."""

    def __init__(self, widgets):
        self._widgets = dict(widgets)

    def collect(self):
        return {}


class _Screen(QWidget):
    """A screen shaped like `AppScreen` for the parts the banner touches."""

    app_key = prerun.QC_APP

    def __init__(self, src):
        super().__init__()
        field = QLineEdit()
        field.setText(str(src))
        self._settings_model = _Model({"src": field})
        self._runtime_wrap = QWidget(self)
        QVBoxLayout(self._runtime_wrap)
        self._actions_row = QWidget(self._runtime_wrap)
        self._runtime_wrap.layout().addWidget(self._actions_row)


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Make every scorecard lookup take :data:`SLOW_S`, as a cold mount does.

    `find_scorecards` is the first filesystem call the refresh makes and the
    one that reaches the user's own path, so blocking it stands in for the
    whole `qc_roots` -> `os.stat` -> `read_digest` sequence below it.
    """
    from spacr import seg_qc

    def asleep(*_args, **_kwargs):
        # `wait`, not `sleep`: see :data:`_RELEASE`.
        _RELEASE.wait(SLOW_S)
        raise AssertionError("the GUI thread waited for the scorecards")

    monkeypatch.setattr(seg_qc, "find_scorecards", asleep)
    monkeypatch.setattr(seg_qc, "read_digest", asleep)
    return asleep


@pytest.fixture
def banners(qtbot):
    """Build banners on stand-in screens; leave none with a worker in flight.

    The drain is not tidiness. These tests park a worker inside a sleeping
    mount on purpose, and Qt aborts the process when a running QThread is
    destroyed, so the parked worker has to be released and joined before the
    widget that owns it goes.
    """
    made = []

    def build(src, **kwargs):
        screen = _Screen(src)
        qtbot.addWidget(screen)
        banner = prerun.SegQCBanner(screen, **kwargs)
        made.append(banner)
        return banner

    build.made = made
    yield build
    _RELEASE.set()
    app = QApplication.instance()
    for banner in made:
        deadline = time.monotonic() + 30.0
        while banner.busy and time.monotonic() < deadline:
            # The completion is a QUEUED metacall -- `_job_settled` is a bound
            # method of a GUI-thread object, which is the whole point -- so it
            # only lands if somebody spins the loop. Nothing else will here:
            # the test has already ended.
            app.processEvents()
            time.sleep(0.02)
    _RELEASE.clear()


def test_refresh_returns_before_the_scorecards_answer(banners, sleeping_mount,
                                                      tmp_path):
    """The property the freeze violated: asking does not mean waiting."""
    banner = banners(tmp_path)

    started = time.monotonic()
    banner.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"refresh() took {elapsed:.1f}s -- it is reading the scorecards on "
        "the GUI thread again, which is the freeze")


def test_opening_the_measure_screen_returns_before_the_mount_wakes(
        qtbot, banners, sleeping_mount, tmp_path):
    """`install_qc_banner` is called inside `MainWindow._build_screen`.

    That build does not yield, so every millisecond spent here is a
    millisecond the application is unpaintable. It used to spend the mount's
    full wake-up time.
    """
    screen = _Screen(tmp_path)
    qtbot.addWidget(screen)

    started = time.monotonic()
    banner = prerun.install_qc_banner(screen)
    elapsed = time.monotonic() - started

    assert banner is not None
    banners.made.append(banner)
    assert elapsed < FAST_S, (
        f"building the Measure screen took {elapsed:.1f}s -- installing the "
        "banner is statting the user's src folder again")


def test_a_show_and_a_burst_of_typing_never_block_either(banners,
                                                         sleeping_mount,
                                                         tmp_path):
    """The other two GUI-thread entries reach the same method.

    Returning to the screen fires `_ShowFilter`, and every keystroke in the
    src field arms the debounce; both used to land on the blocking read.
    """
    banner = banners(tmp_path)
    field = banner._screen._settings_model._widgets["src"]

    started = time.monotonic()
    banner._on_screen_shown()
    for index in range(20):
        field.setText(f"{tmp_path}/plate{index}")
    banner.refresh()
    banner.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < FAST_S, (
        f"a show plus twenty keystrokes took {elapsed:.1f}s on the GUI "
        "thread")


def test_the_verdict_still_arrives_it_is_just_a_moment_later(qtbot, banners,
                                                             tmp_path):
    """The fix is worthless if the banner stops saying what it said.

    Read on a real worker thread this time -- no `threaded=False` -- so this
    is the path the interface actually takes.
    """
    plate = tmp_path / "plate1"
    (plate / "qc").mkdir(parents=True)
    card = plate / "qc" / "segmentation_qc_cell.csv"
    card.write_text("well,n_objects\nA01,120\n", encoding="utf-8")

    banner = banners(plate)

    with qtbot.waitSignal(banner.refreshed, timeout=60000) as caught:
        banner.refresh()

    assert banner.digest is not None, "the verdict never arrived"
    assert caught.args[0] == banner.digest.verdict
    assert not banner.isHidden()


def test_the_read_is_never_announced_as_a_run(qtbot, banners, monkeypatch,
                                              tmp_path):
    """Housekeeping the user did not start must not claim a run banner.

    Home shows a banner for every visible job in the run registry. A CSV read
    fired by opening a screen is not a run, and one that says it is would
    flash "measure - running" at the user for as long as the screen is open.
    """
    seen = {}
    real = prerun.SegQCBanner._start_job

    def _spy(self, fn, box, on_done, app_key, **kwargs):
        seen.update(kwargs)
        return real(self, fn, box, on_done, app_key, **kwargs)

    monkeypatch.setattr(prerun.SegQCBanner, "_start_job", _spy)

    banner = banners(tmp_path)
    with qtbot.waitSignal(banner.refreshed, timeout=60000):
        banner.refresh()

    assert seen.get("user_visible") is False
    # `make_thread` imports matplotlib.pyplot on the CALLING thread before the
    # first capturing job, and the caller here is the GUI thread.
    assert seen.get("capture_figures") is False


class _Digest:
    """A scored digest, shaped the way `SegQCBanner._draw` reads one."""

    verdict = "fail"
    stale = False
    headline = "two wells segmented no cells at all"
    subhead = ""
    root = ""
    scorecards = ()
    findings = ()


def test_score_the_masks_now_still_works_while_the_read_is_in_flight(
        qtbot, banners, sleeping_mount, monkeypatch, tmp_path):
    """The button must not become a silent no-op for the length of a stat.

    THE SECOND HALF OF THE FREEZE. `_JobMixin` holds ONE job slot, and every
    caller is refused while it is taken. That was harmless while the only
    thing in it was a scoring pass the user had started -- refusing a second
    click on "Score the masks now" is the pass already running.

    Moving the read onto a worker put housekeeping in that slot: a read runs
    on every screen open, on every return to the screen, and 450 ms after
    every keystroke in src. `_on_score_clicked` began with

        if not _has_src(src) or self.busy:
            return

    so during a read the button did nothing at all -- no message, no
    disabled state, no scoring -- for as long as the filesystem took, which
    on the mount that started this was twenty seconds. A fix for a freeze
    that costs a button is not a fix.

    The click is remembered and run the instant the read lets go. Note the
    read here FAILS: a queued click must survive a worker that raised, not
    just one that succeeded.
    """
    from spacr import seg_qc

    scored = []

    def _score(src, thresholds=None):
        scored.append(str(src))
        return _Digest()

    monkeypatch.setattr(seg_qc, "score_digest", _score)
    monkeypatch.setattr(seg_qc, "thresholds_from_settings", lambda _s: {})

    banner = banners(tmp_path)
    banner.refresh()
    assert banner.busy, "the read should be parked in the sleeping mount"

    banner._btn_score.click()

    assert not banner._btn_score.isEnabled(), (
        "the click was swallowed -- the button is a silent no-op again")
    assert banner._title.text() == "Segmentation QC — scoring the masks…"
    assert not scored, "the scoring pass must wait for the slot, not race it"

    _RELEASE.set()
    qtbot.waitUntil(lambda: bool(scored), timeout=60000)
    qtbot.waitUntil(lambda: not banner.busy, timeout=60000)

    assert scored == [str(tmp_path)]
    assert banner.digest is not None
    assert banner._btn_score.isEnabled()


def test_a_read_that_lands_after_src_is_cleared_is_thrown_away(
        qtbot, banners, monkeypatch, tmp_path):
    """A worker cannot be interrupted, so its answer has to be checked.

    `JobRunner.cancel` and `_JobMixin` alike can stop nothing that is already
    inside a `stat`. Clearing the src field hides the banner and forgets the
    digest -- and the read still in flight then landed, called `show()`, and
    put the previous plate's verdict back on a screen naming no source at
    all. Every request bumps a generation; an answer stamped with an older
    one is dropped.
    """
    from spacr import seg_qc

    plate = tmp_path / "plate1"
    (plate / "qc").mkdir(parents=True)
    (plate / "qc" / "segmentation_qc_cell.csv").write_text(
        "well,n_objects\nA01,120\n", encoding="utf-8")

    gate = threading.Event()
    real = seg_qc.find_scorecards

    def held(src):
        gate.wait(SLOW_S)
        return real(src)

    monkeypatch.setattr(seg_qc, "find_scorecards", held)

    banner = banners(plate)
    field = banner._screen._settings_model._widgets["src"]
    banner.refresh()
    assert banner.busy

    # The user clears the field while the read is still inside the mount.
    field.setText("")
    banner.refresh()
    assert banner.isHidden()
    assert banner.digest is None

    gate.set()
    qtbot.waitUntil(lambda: not banner.busy, timeout=60000)

    assert banner.isHidden(), (
        "a read of the cleared source un-hid the banner")
    assert banner.digest is None, (
        "the banner is showing a verdict for a source nobody named")


def test_a_burst_while_a_read_is_parked_is_recorded_once_not_re_asked(
        qtbot, banners, monkeypatch, tmp_path):
    """Coalesced the way `ChainingBar._refresh` coalesces: one flag, one catch-up.

    Twenty keystrokes are twenty requests for one answer. Starting a job each
    would ask a sleeping mount the same question twenty times; dropping them
    outright loses the last one, which is the only one that matters. So the
    request is REMEMBERED, and `_pending_work` runs exactly one catch-up when
    the slot frees -- driven by the job landing, not by a clock.

    The first pass re-armed the 450 ms debounce instead. That loses nothing,
    and it is not the defect this file was flagged for; what it does is turn
    every stalled read into a poll. `refresh` re-entered every 450 ms for the
    twenty seconds a cold `autofs` mount took, and each entry walked the
    settings model for `src` again to reach the same conclusion it had
    already reached. The catch-up also arrived up to a debounce late, because
    it waited for the next tick rather than for the answer.
    """
    from spacr.qt import prerun as _prerun

    plates = []
    for name in ("plate_a", "plate_b"):
        plate = tmp_path / name
        (plate / "qc").mkdir(parents=True)
        (plate / "qc" / "segmentation_qc_cell.csv").write_text(
            "well,n_objects\nA01,120\n", encoding="utf-8")
        plates.append(plate)

    gate = threading.Event()
    reached = threading.Event()
    asked = []
    entered = []
    real_fingerprint = _prerun.SegQCBanner._fingerprint
    real_refresh = _prerun.SegQCBanner.refresh

    def counted(self, src):
        """Record the source each read job asks about; hold the first."""
        asked.append(str(src))
        if len(asked) == 1:
            reached.set()
            gate.wait(SLOW_S)
        return real_fingerprint(self, src)

    def counted_refresh(self):
        """Count every entry into `refresh`, whoever asked for it."""
        entered.append(1)
        return real_refresh(self)

    monkeypatch.setattr(_prerun.SegQCBanner, "_fingerprint", counted)
    # Patched BEFORE the banner is built: `_timer.timeout` binds `refresh`
    # at construction, so a later patch would never reach the timer's slot --
    # which is the caller this test is counting.
    monkeypatch.setattr(_prerun.SegQCBanner, "refresh", counted_refresh)

    banner = banners(plates[0])
    field = banner._screen._settings_model._widgets["src"]
    banner.refresh()
    # The burst has to arrive while the read is genuinely on the disk. The
    # worker is a real thread, so "after refresh() returned" is not that:
    # without this wait the test races the scheduler and asserts against a
    # job that has not started.
    assert reached.wait(SLOW_S), "the read never reached the scorecards"
    assert banner.busy

    # Signals blocked so the 450 ms debounce is not armed by the edit either:
    # what is under test is what the twenty requests below cost, and a live
    # textChanged would arm the very timer this is measuring.
    field.blockSignals(True)
    field.setText(str(plates[1]))
    field.blockSignals(False)
    for _ in range(20):
        banner.refresh()

    assert asked == [str(plates[0])], "a job per request reached the disk"
    assert len(entered) == 21, "the burst itself was miscounted"

    # Three debounce periods with the read still parked. Nothing may re-ask.
    qtbot.wait(3 * 450 + 200)
    assert len(entered) == 21, (
        f"the parked read was re-asked {len(entered) - 21} time(s) on a "
        "timer -- the request is meant to be recorded, not polled")

    gate.set()
    qtbot.waitUntil(lambda: len(asked) >= 2, timeout=60000)
    qtbot.waitUntil(lambda: not banner.busy, timeout=60000)

    assert asked == [str(plates[0]), str(plates[1])], (
        "the catch-up must run exactly once, and must ask about the source "
        "the field names now")
    assert banner.digest is not None, "the last request was dropped"
