"""Opening Annotate does not wait on the folder it was last opened on.

THE FREEZE, 2026-09-04. `AnnotateScreen.__init__` ended with

    if self._suggested_source and os.path.isdir(self._suggested_source):
        self._src_label.setText(f"Suggested (last used): ...")

and `_suggested_source` is `prefs.get_last_source("annotate")` -- the folder
the USER last worked in, read back out of QSettings. The screen is built
synchronously on the GUI thread (`MainWindow._build_screen`; Qt forbids
building widgets anywhere else), so that stat ran with the interface frozen
and nothing yet drawn.

Measured on the maintainer's machine: one `os.path.isdir` on a path under
`/nas_mnt` -- an `autofs` mount with `timeout=600` whose share was asleep --
had NOT RETURNED AFTER TWENTY SECONDS, because the stat is what triggers the
automount. Anyone whose last annotation session was on the NAS opened
Annotate and got a dead application, no traceback, nothing in the logs: a
stalled event loop is not a crash. It was reported as "opening map barcodes
crashes spacr", plus hover flicker and glimpses of other screens, all of
which are one blocked thread.

The subtitle is a HINT. It is not worth a millisecond of the interface, so
it now comes from `spacr.qt.path_probe` -- cached, answered off the GUI
thread -- and is filled in when the probe lands.
"""
from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe

#: Long enough that a GUI thread which waited for it is unmistakably caught,
#: short enough that the test itself stays runnable. The real measurement was
#: twenty seconds and still counting.
SLOW_S = 8.0

#: What the maintainer's QSettings actually held.
ASLEEP = "/nas_mnt/data/annotate/plate_1"


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    """Each test asks its questions for the first time, as a cold start does."""
    path_probe.forget()
    yield
    path_probe.forget()


@pytest.fixture
def sleeping_mount(monkeypatch):
    """Make exactly :data:`ASLEEP` behave like an autofs share that is out.

    Only that one path, and every other path answered by the real call: a
    blanket patch would also stall the icon and stylesheet lookups that
    ordinary construction makes, and this test is about ONE path -- the one
    the user supplied.
    """
    real_isdir = os.path.isdir
    real_exists = os.path.exists

    def isdir(path, *args, **kwargs):
        if str(path) == ASLEEP:
            time.sleep(SLOW_S)
            return True
        return real_isdir(path, *args, **kwargs)

    def exists(path, *args, **kwargs):
        if str(path) == ASLEEP:
            time.sleep(SLOW_S)
            return True
        return real_exists(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "isdir", isdir)
    monkeypatch.setattr(os.path, "exists", exists)


def _screen(qtbot):
    from spacr.qt.screens.annotate import AnnotateScreen

    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    return screen


def test_opening_annotate_does_not_wait_for_a_sleeping_last_source(
        qtbot, qt_theme_applied, sleeping_mount):
    """The regression itself: construction returns while the mount is out."""
    from spacr.qt import prefs

    prefs.set_last_source("annotate", ASLEEP)

    started = time.monotonic()
    screen = _screen(qtbot)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"building AnnotateScreen took {elapsed:.1f}s with the remembered "
        "source asleep -- it is stat-ing the user's path on the GUI thread "
        "again, which is the freeze")
    # And the screen is a whole screen, not a half-built one that merely
    # returned quickly.
    assert screen._suggested_source == ASLEEP
    assert screen._src_label is not None


def test_the_suggestion_still_appears_once_the_probe_answers(
        qtbot, qt_theme_applied, tmp_path):
    """Nothing the user was shown is lost -- it arrives a moment later.

    An unknown path is reported ABSENT, so the placeholder stands for an
    instant. This is the half that puts the suggestion back.
    """
    from spacr.qt import prefs

    last = tmp_path / "expt"
    last.mkdir()
    prefs.set_last_source("annotate", str(last))

    screen = _screen(qtbot)

    qtbot.waitUntil(
        lambda: "Suggested (last used)" in screen._src_label.text(),
        timeout=5000)
    assert str(last) in screen._src_label.text()
    assert screen._src_label.property("i18nSkipText") is True


def test_a_source_that_is_gone_leaves_the_placeholder_standing(
        qtbot, qt_theme_applied, tmp_path):
    """The optimism has a limit: a folder that is not there is not offered."""
    from spacr.qt import prefs

    gone = tmp_path / "deleted"
    prefs.set_last_source("annotate", str(gone))

    screen = _screen(qtbot)

    qtbot.waitUntil(
        lambda: path_probe.known(str(gone), want_dir=True) is False,
        timeout=5000)
    qtbot.wait(50)          # let any queued emission be delivered
    assert "Suggested" not in screen._src_label.text()


def test_the_picker_is_not_started_in_a_folder_that_is_gone(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """QFileDialog stats AND lists its starting directory, on the GUI thread.

    Handing it a remembered path on a sleeping mount is the same freeze one
    click further on, so a folder the probe has answered for is the only one
    offered.
    """
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt import prefs

    gone = tmp_path / "deleted"
    prefs.set_last_source("annotate", str(gone))
    screen = _screen(qtbot)
    qtbot.waitUntil(
        lambda: path_probe.known(str(gone), want_dir=True) is False,
        timeout=5000)

    seen = []

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    screen._on_pick_source()

    assert seen == [os.getcwd()], (
        f"the picker was pointed at {seen!r} -- a folder the probe already "
        "knows is not there")


def test_closing_releases_the_screen_from_a_process_wide_signal(
        qtbot, qt_theme_applied, tmp_path):
    """`path_probe.probes` outlives every screen; a live connection is a crash.

    Not a leak so much as an abort: the emission is delivered to a Python
    wrapper whose C++ half Qt has already destroyed.
    """
    from spacr.qt import prefs

    last = tmp_path / "expt2"
    last.mkdir()
    prefs.set_last_source("annotate", str(last))
    screen = _screen(qtbot)
    assert screen._path_probe_landed is not None

    screen.close()
    assert screen._path_probe_landed is None
    # Nothing left to deliver to: emitting for the same path is a no-op.
    path_probe.probes.answered.emit(str(last), True)


# ---------------------------------------------------------------------------
# The sibling sites: everywhere else this screen hands a user's path to
# something that stats it on the GUI thread.
# ---------------------------------------------------------------------------

def test_the_screen_subscribes_to_the_probe_before_it_asks_it_anything(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Order matters, and getting it wrong loses the answer silently.

    `path_probe.isdir` QUEUES the check; a worker that finishes quickly emits
    `answered` from its own thread while `__init__` is still running. A Qt
    signal emitted with nothing connected to it is dropped, not buffered --
    so asking before subscribing can lose the only answer this path is ever
    given, and the suggestion is then missing for the life of the screen.
    Pinned as an order because the race itself is microseconds wide and would
    make a flaky test.
    """
    from spacr.qt import prefs
    from spacr.qt.screens.annotate import AnnotateScreen

    last = tmp_path / "ordered"
    last.mkdir()
    prefs.set_last_source("annotate", str(last))

    order = []
    real_follow = AnnotateScreen._follow_path_probes
    real_apply = AnnotateScreen._apply_suggested_source

    def follow(self):
        order.append("subscribe")
        return real_follow(self)

    def apply_it(self):
        order.append("ask")
        return real_apply(self)

    monkeypatch.setattr(AnnotateScreen, "_follow_path_probes", follow)
    monkeypatch.setattr(AnnotateScreen, "_apply_suggested_source", apply_it)

    screen = _screen(qtbot)

    assert order[:2] == ["subscribe", "ask"], (
        f"the screen asked the probe before subscribing to it ({order!r}) -- "
        "a fast answer is emitted to nobody and the suggestion never appears")
    assert screen._path_probe_landed is not None


def test_the_picker_does_not_start_in_an_open_source_it_cannot_vouch_for(
        qtbot, qt_theme_applied, monkeypatch, sleeping_mount):
    """The other branch of the same button, left blocking by the first fix.

    Once a source is open, `_settings.src` is what the picker started in --
    unprobed, and just as much a path the user supplied as the remembered
    one. An `autofs` mount that has gone back to sleep since the source was
    opened is the ordinary way to reach this.
    """
    from PySide6.QtWidgets import QFileDialog

    screen = _screen(qtbot)
    screen._settings.src = ASLEEP

    seen = []

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    started = time.monotonic()
    screen._on_pick_source()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"opening the source picker took {elapsed:.1f}s -- the open source "
        "is being stat-ed on the GUI thread")
    assert seen == [os.getcwd()]


class _NoWorker:
    """Stands in for `SaveWorker`: started and stopped, never saves."""

    last_error = None
    last_save_ts = None
    busy = False
    pending_batches = 0

    def start(self):
        """Do nothing, successfully."""

    def stop(self, wait=False):
        """Do nothing, successfully."""


def test_opening_a_source_teaches_the_probe_so_the_picker_can_go_back(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """Nothing is lost: the picker still opens in the folder being annotated.

    `_open_source` queues the `isdir` answer while the mount is demonstrably
    awake -- the picker has just listed the folder -- so `_starting_folder`
    has it cached by the time anyone presses Source again.
    """
    from spacr.qt import path_probe as probe
    from spacr.qt.screens.annotate import AnnotateScreen

    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "measurements" / "measurements.db").touch()

    screen = _screen(qtbot)
    # Everything `_open_source` does after pointing itself at the folder is
    # database work this test is not about.
    monkeypatch.setattr(AnnotateScreen, "_refresh_total",
                        lambda self, then=None: None)
    monkeypatch.setattr(AnnotateScreen, "_refresh_round_state",
                        lambda self: None)
    monkeypatch.setattr("spacr.qt.screens.annotate.ensure_annotation_column",
                        lambda *a, **k: None)
    monkeypatch.setattr("spacr.qt.screens.annotate.SaveWorker",
                        lambda *a, **k: _NoWorker())

    screen._open_source(str(src))

    qtbot.waitUntil(
        lambda: probe.known(str(src), want_dir=True) is True, timeout=5000)
    assert screen._starting_folder() == str(src)


# ---------------------------------------------------------------------------
# Settings ▸ Source folder ▸ Browse… -- the same dialog, one click further in
# ---------------------------------------------------------------------------

def test_browsing_from_the_settings_dialog_does_not_wait_on_the_field(
        qtbot, qt_theme_applied, monkeypatch, sleeping_mount):
    """`Browse…` hands the field's text to QFileDialog, which lists it.

    The field is prefilled with the open source and can be typed into, so it
    holds a path the user supplied by two routes. This was the sibling site
    the first pass left behind: the screen's own picker was fixed and this
    one, one dialog in, was not.
    """
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt import annotate_engine as engine
    from spacr.qt.screens import annotate as annotate_mod

    settings = engine.AnnotateSettings()
    settings.src = ASLEEP
    dialog = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dialog)

    seen = []

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    started = time.monotonic()
    dialog._pick_src()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"pressing Browse took {elapsed:.1f}s -- the settings dialog is "
        "stat-ing the source field on the GUI thread")
    assert seen == [os.getcwd()]
    assert dialog._src_edit.text() == ASLEEP, "cancelling emptied the field"


def test_browsing_still_starts_in_a_folder_that_is_really_there(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """The behaviour is kept, not dropped: the answer just arrives first.

    The probe is queued when the dialog is built, so a folder that exists has
    answered long before anybody can press the button.
    """
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt import annotate_engine as engine
    from spacr.qt import path_probe as probe
    from spacr.qt.screens import annotate as annotate_mod

    folder = tmp_path / "plate"
    folder.mkdir()
    settings = engine.AnnotateSettings()
    settings.src = str(folder)
    dialog = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dialog)

    qtbot.waitUntil(
        lambda: probe.known(str(folder), want_dir=True) is True, timeout=5000)

    seen = []
    chosen = tmp_path / "picked"
    chosen.mkdir()

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return str(chosen)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    dialog._pick_src()

    assert seen == [str(folder)]
    assert dialog._src_edit.text() == str(chosen)
    # And the folder just chosen is learned while the mount is awake, so the
    # next press of Browse starts there instead of at the working directory.
    qtbot.waitUntil(
        lambda: probe.known(str(chosen), want_dir=True) is True, timeout=5000)


def test_a_path_typed_into_the_field_is_probed_when_the_editing_ends(
        qtbot, qt_theme_applied, tmp_path):
    """Typed paths get their head start too, without a probe per keystroke.

    `editingFinished` fires on focus-out -- which for a mouse is the PRESS on
    Browse, a moment before its `clicked` -- so a local folder typed by hand
    is normally answered in time to be offered. One probe per editing
    session, not one per keystroke: `path_probe`'s workers are few on purpose
    and a sleeping mount parks each one it is given for five seconds.
    """
    from spacr.qt import annotate_engine as engine
    from spacr.qt import path_probe as probe
    from spacr.qt.screens import annotate as annotate_mod

    typed = tmp_path / "typed"
    typed.mkdir()
    dialog = annotate_mod._SettingsDialog(engine.AnnotateSettings())
    qtbot.addWidget(dialog)

    assert probe.known(str(typed), want_dir=True) is None
    dialog._src_edit.setText(f"  {typed}  ")
    dialog._src_edit.editingFinished.emit()

    qtbot.waitUntil(
        lambda: probe.known(str(typed), want_dir=True) is True, timeout=5000)


# ---------------------------------------------------------------------------
# The probe's answer is not always an answer
# ---------------------------------------------------------------------------

def test_a_probe_answer_the_stat_never_gave_is_not_good_enough_to_open_in(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """`path_probe` reports an unanswered stat as PRESENT once it gives up.

    `_stat_with_timeout` waits `PROBE_TIMEOUT_S` for the stat and then says
    True. That is the right way round for the question it was written for --
    drawing a remembered path red because a mount is slow is worse than
    leaving it black -- and it is the wrong way round for this one, because
    `QFileDialog` then runs the very stat that never came back.

    So gating the picker on the probe's own answer does not remove the freeze,
    it postpones it by the length of the timeout: five seconds after the probe
    gave up, the gate opens and the window locks for as long as the mount
    takes. The gate has to be the stricter question -- did a real stat come
    back? -- and that is `_vouched_dir`.
    """
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt import prefs

    # Short, so the probe gives up inside the test rather than in five
    # seconds' time. Read at call time by `_stat_with_timeout`, by
    # `_on_probe_answered` and by `_vouched_dir`, so one patch moves all three.
    monkeypatch.setattr(path_probe, "PROBE_TIMEOUT_S", 0.2)

    asleep = tmp_path / "asleep"
    asleep.mkdir()                      # it IS there; it just cannot say so
    real_isdir = os.path.isdir

    def isdir(path, *args, **kwargs):
        if str(path) == str(asleep):
            time.sleep(3.0)
            return True
        return real_isdir(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "isdir", isdir)
    prefs.set_last_source("annotate", str(tmp_path / "never-existed"))

    screen = _screen(qtbot)
    screen._settings.src = str(asleep)

    seen = []

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    # The first press asks the question. Nothing is known yet, so the picker
    # starts at the working directory -- which the first pass already got
    # right, and is not what this test is about.
    screen._on_pick_source()

    # Let the probe give up and cache its optimistic "present", and let the
    # answer be delivered.
    qtbot.waitUntil(
        lambda: path_probe.known(str(asleep), want_dir=True) is True,
        timeout=5000)
    qtbot.wait(50)

    started = time.monotonic()
    screen._on_pick_source()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"opening the source picker took {elapsed:.1f}s")
    assert seen == [os.getcwd(), os.getcwd()], (
        f"the picker was pointed at {seen[-1]!r} -- a folder whose stat has "
        "still not come back. `path_probe` only assumed it was there when it "
        "stopped waiting, and the dialog would have done the waiting instead")


def test_the_settings_browse_button_will_not_open_in_an_assumed_folder(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """The same hole, one dialog in. `Browse…` reads the same probe cache."""
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt import annotate_engine as engine
    from spacr.qt.screens import annotate as annotate_mod

    monkeypatch.setattr(path_probe, "PROBE_TIMEOUT_S", 0.2)

    asleep = tmp_path / "asleep-field"
    asleep.mkdir()
    real_isdir = os.path.isdir

    def isdir(path, *args, **kwargs):
        if str(path) == str(asleep):
            time.sleep(3.0)
            return True
        return real_isdir(path, *args, **kwargs)

    monkeypatch.setattr(os.path, "isdir", isdir)

    settings = engine.AnnotateSettings()
    settings.src = str(asleep)
    dialog = annotate_mod._SettingsDialog(settings)
    qtbot.addWidget(dialog)

    qtbot.waitUntil(
        lambda: path_probe.known(str(asleep), want_dir=True) is True,
        timeout=5000)
    qtbot.wait(50)

    seen = []

    def fake_dialog(_parent, _caption, directory, *args, **kwargs):
        seen.append(directory)
        return ""

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", fake_dialog)
    started = time.monotonic()
    dialog._pick_src()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"pressing Browse took {elapsed:.1f}s"
    assert seen == [os.getcwd()], (
        f"Browse was pointed at {seen!r} -- a folder the probe only assumed "
        "was there because its stat never returned")
    assert dialog._src_edit.text() == str(asleep), "cancelling emptied the field"


def test_a_folder_another_screen_probed_first_is_still_offered(
        qtbot, qt_theme_applied, tmp_path):
    """Strictness must not cost the head start it exists to protect.

    `ChainingBar.search_roots` probes `prefs.get_last_source("annotate")` --
    exactly this screen's remembered source -- before Annotate is ever built,
    so the answer is routinely in the cache already, with nothing to say
    whether a stat returned it or the timeout invented it. Inheriting it would
    reopen the hole; refusing it outright would send every picker to the
    working directory. The question is asked again instead, once, under this
    module's own clock.
    """
    from spacr.qt import prefs

    plate = tmp_path / "plate"
    plate.mkdir()
    # Somebody else asks first, exactly as the chaining strip does.
    path_probe.isdir(str(plate))
    qtbot.waitUntil(
        lambda: path_probe.known(str(plate), want_dir=True) is True,
        timeout=5000)

    prefs.set_last_source("annotate", str(plate))
    screen = _screen(qtbot)

    qtbot.waitUntil(
        lambda: screen._starting_folder() == str(plate), timeout=5000)
    # And the suggestion still reaches the subtitle.
    qtbot.waitUntil(
        lambda: "Suggested (last used)" in screen._src_label.text(),
        timeout=5000)
