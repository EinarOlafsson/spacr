"""The spaceout backdrop must never block the GUI thread on a heavy import.

The reported symptom was an asymmetry: opening a module under ``spaceout``
put the compositor's "is not responding" dialog up, while the same module
under ``spacr`` opened cleanly.  Only ``spaceout`` builds the fractal
backdrop, and only that backdrop takes ``HEAVY_IMPORT_LOCK`` -- the lock
the pipeline preloader holds for a whole module import, 2.3 s for each of
the two that pull torch.  Taking it with a plain ``with lock:`` on the GUI
thread is a priority inversion: a background task with no deadline holding
up the one thread that has one.

Measured, on this machine, against a 2,000 ms hold:

    before   2,130 ms of blocked GUI thread, backdrop built
    after      142 ms, backdrop refused; the retry builds it in 43 ms

``AppScreen._heavy_lock_is_free`` cannot close this on its own.  It is a
check, not a reservation, and the preloader re-takes the lock between two
imports -- so landing in the gap is the ordinary case for a click made
while the preloader runs, not a rare one.
"""
from __future__ import annotations

import inspect
import threading
import time

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The refusal, and what it must not be mistaken for
# ---------------------------------------------------------------------------

class TestTheRefusalIsNotAGpuFailure:

    def test_it_is_not_a_gpu_backend_error(self):
        """The distinction the CPU fallback turns on.

        ``create_fractal_widget`` answers a ``GpuBackendError`` with the
        CPU renderer -- the one instruction 315 recorded as eating twenty
        cores.  A busy lock is not a GPU failure: the context was never
        attempted and this machine's shaders compile fine.  Making the
        refusal a subclass would trade a 0.3 s wait for that renderer,
        every time a module is opened during startup.
        """
        from spacr.qt.widgets.fractal_travel import (GpuBackendError,
                                                     _HeavyImportInProgress)

        assert not issubclass(_HeavyImportInProgress, GpuBackendError)
        assert issubclass(_HeavyImportInProgress, RuntimeError)

    def test_the_auto_fallback_lets_it_past_instead_of_drawing_on_the_cpu(
            self, monkeypatch):
        """THE ARC: ``except _HeavyImportInProgress: raise``.

        Driven rather than read, because the handler sits above a bare
        ``except Exception`` that would otherwise swallow it -- which is
        exactly what it did before this branch existed.
        """
        from spacr.qt.widgets import fractal_travel as F

        built = []

        def refuse(*_args, **_kwargs):
            raise F._HeavyImportInProgress("busy")

        monkeypatch.setattr(F, "gpu_is_available", lambda: True)
        monkeypatch.setattr(F, "_make_gpu_widget", refuse)
        monkeypatch.setattr(F, "_make_cpu_widget",
                            lambda *a, **k: built.append("cpu"))

        with pytest.raises(F._HeavyImportInProgress):
            F.create_fractal_widget(F.Settings(backend="auto"))

        assert built == [], (
            "a busy lock fell through to the CPU renderer, so a click made "
            "during startup now costs the twenty-core fallback")

    def test_an_ordinary_gpu_failure_still_falls_back(self, monkeypatch):
        """The other half, so the re-raise cannot be read as "never fall
        back": a machine whose shaders will not compile still gets an
        animation."""
        from spacr.qt.widgets import fractal_travel as F

        built = []

        def fail(*_args, **_kwargs):
            raise F.GpuBackendError("no context")

        monkeypatch.setattr(F, "gpu_is_available", lambda: True)
        monkeypatch.setattr(F, "_make_gpu_widget", fail)
        monkeypatch.setattr(F, "_make_cpu_widget",
                            lambda *a, **k: built.append("cpu") or "widget")

        assert F.create_fractal_widget(F.Settings(backend="auto")) == "widget"
        assert built == ["cpu"]


# ---------------------------------------------------------------------------
# The wait itself
# ---------------------------------------------------------------------------

class TestTheWaitIsBounded:

    def test_the_constructor_no_longer_takes_the_lock_open_endedly(self):
        """THE PIN, for a constructor that needs a GL context to run.

        The body cannot be driven without one, so what is held here is
        the shape: a bounded acquire and a refusal, never ``with lock:``.
        The bound is what keeps the GUI thread under any compositor's
        threshold while the preloader finishes.
        """
        from spacr.qt.widgets import fractal_travel as F

        source = inspect.getsource(F._make_gpu_widget)
        assert "lock.acquire(timeout=_HEAVY_LOCK_WAIT)" in source
        assert "raise _HeavyImportInProgress(" in source
        # CODE ONLY. The comment above the acquire quotes the old form to
        # say what it cost, so matching the raw source would fail on the
        # explanation rather than on the thing explained.
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))
        assert "with lock:" not in code, (
            "the GL context is built under an unbounded acquire again, "
            "which is the multi-second GUI freeze this test exists for")
        assert "finally:" in source and "lock.release()" in source, (
            "the lock is no longer released on the failure path")

    def test_the_bound_is_short_enough_to_be_invisible(self):
        from spacr.qt.widgets.fractal_travel import _HEAVY_LOCK_WAIT

        assert 0 < _HEAVY_LOCK_WAIT <= 0.25, (
            "the bounded wait has grown past a quarter second, which is "
            "long enough to see and long enough to stack up over retries")


# ---------------------------------------------------------------------------
# The two helpers the callers share
# ---------------------------------------------------------------------------

class TestTheLockPeek:

    def test_a_free_lock_answers_yes_and_is_not_held_afterwards(self):
        from spacr.qt.app import HEAVY_IMPORT_LOCK
        from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

        assert _the_heavy_import_lock_is_free() is True
        assert HEAVY_IMPORT_LOCK.acquire(blocking=False), (
            "the peek kept the lock, so it is a reservation and not a peek")
        HEAVY_IMPORT_LOCK.release()

    def test_a_held_lock_answers_no(self):
        from spacr.qt.app import HEAVY_IMPORT_LOCK
        from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

        held = threading.Event()
        release = threading.Event()

        def hold():
            with HEAVY_IMPORT_LOCK:
                held.set()
                release.wait(5)

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        try:
            assert held.wait(5)
            assert _the_heavy_import_lock_is_free() is False
        finally:
            release.set()
            worker.join(5)


class TestTellingARefusalFromAFailure:

    def test_the_refusal_is_recognised(self):
        from spacr.qt.widgets.ambient import _the_backdrop_wants_a_retry
        from spacr.qt.widgets.fractal_travel import _HeavyImportInProgress

        assert _the_backdrop_wants_a_retry(_HeavyImportInProgress("busy"))

    def test_a_real_failure_is_not(self):
        """The half that matters for the console: a broken backdrop must
        still be reported once, not retried every 120 ms forever."""
        from spacr.qt.widgets.ambient import _the_backdrop_wants_a_retry
        from spacr.qt.widgets.fractal_travel import GpuBackendError

        assert not _the_backdrop_wants_a_retry(GpuBackendError("no context"))
        assert not _the_backdrop_wants_a_retry(RuntimeError("anything else"))
        assert not _the_backdrop_wants_a_retry(ImportError("no module"))


# ---------------------------------------------------------------------------
# The callers come back
# ---------------------------------------------------------------------------

class _Scheduled:
    """Captures ``QTimer.singleShot`` so a retry can be seen and run."""

    def __init__(self, monkeypatch):
        from PySide6.QtCore import QTimer

        self.calls = []
        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: self.calls.append((ms, fn))))

    def run_one(self):
        _ms, fn = self.calls.pop(0)
        fn()


class _FakeAppScreen:
    """Enough of ``AppScreen`` for ``_install_ambient`` to run against."""

    def __init__(self):
        self._backdrops_ready = True
        self._ambient = None
        self._ambient_applied = None
        self.cleared = 0
        self.synced = 0
        self.orphans = 0

    def _heavy_lock_is_free(self):
        return True

    def _clear_page_surfaces(self):
        self.cleared += 1

    def _sync_page_palette(self):
        self.synced += 1

    def _discard_orphan_ambient(self):
        self.orphans += 1

    def _install_ambient(self):
        from spacr.qt.screens.app_screen import AppScreen

        return AppScreen._install_ambient(self)


class TestAModuleScreenComesBack:

    def test_a_refusal_is_forgotten_and_retried(self, monkeypatch):
        """THE ARC, and the reason it is not enough to swallow the error.

        ``_ambient_applied`` is set BEFORE the install so a failing
        machine is not asked twice on every palette event.  A refusal
        recorded there would mean a screen opened while the preloader ran
        stays undecorated for the life of the session -- so the refusal
        clears it again, which is the same thing the peek does for the
        case it can see.
        """
        from spacr.qt.widgets import ambient as A
        from spacr.qt.widgets.fractal_travel import _HeavyImportInProgress

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                _HeavyImportInProgress("busy")))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        screen = _FakeAppScreen()
        screen._install_ambient()

        assert screen._ambient is None
        assert screen._ambient_applied is None, (
            "the refusal was recorded as an attempt, so this screen will "
            "never try again and stays undecorated for the session")
        assert [ms for ms, _ in scheduled.calls] == [120]

    def test_the_retry_installs_it_once_the_lock_frees(self, monkeypatch):
        from spacr.qt.widgets import ambient as A
        from spacr.qt.widgets.fractal_travel import _HeavyImportInProgress

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        attempts = []

        def busy_once(*_args, **_kwargs):
            attempts.append("try")
            if len(attempts) == 1:
                raise _HeavyImportInProgress("busy")
            return "the backdrop"

        monkeypatch.setattr(A, "install_ambient", busy_once)

        screen = _FakeAppScreen()
        screen._install_ambient()
        assert screen._ambient is None

        scheduled.run_one()

        assert attempts == ["try", "try"]
        assert screen._ambient == "the backdrop"
        assert screen.cleared == 1 and screen.synced == 1

    def test_a_real_failure_is_not_retried(self, monkeypatch):
        """The other half: a machine that cannot draw it is asked once."""
        from spacr.qt.widgets import ambient as A

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no context")))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        screen = _FakeAppScreen()
        screen._install_ambient()

        assert scheduled.calls == []
        assert screen._ambient_applied is not None, (
            "a real failure was forgotten, so it will be re-attempted on "
            "every palette event")


class _FakeHome:
    def __init__(self):
        self._ambient = None
        self.cleared = 0
        self.discarded = []

    def _clear_page_surfaces(self):
        self.cleared += 1

    def _discard_ambient(self, widget):
        self.discarded.append(widget)

    @staticmethod
    def _ambient_backdrop():
        return None

    def _install_ambient(self):
        from spacr.qt.widgets.home import HomePage

        return HomePage._install_ambient(self)


class TestHomeComesBackToo:

    def test_home_does_not_even_try_while_the_lock_is_held(self,
                                                           monkeypatch):
        """THE ARC: the peek Home did not have.

        Home is built at startup, which is exactly when the preloader is
        importing torch.  Without the peek every Home build paid the
        bounded wait and then failed.
        """
        from spacr.qt.widgets import ambient as A
        from spacr.qt.widgets import home as H

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: False)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: pytest.fail(
                                "Home built the backdrop while the heavy "
                                "import lock was held"))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        assert H is not None

        home = _FakeHome()
        home._install_ambient()

        assert [ms for ms, _ in scheduled.calls] == [120]
        assert home._ambient is None

    def test_a_second_install_does_not_stack_a_backdrop(self, monkeypatch):
        """A retry that arrives after a rebuild already installed one must
        not leave the first parented, ticking and invisible behind it."""
        from spacr.qt.widgets import ambient as A

        _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: pytest.fail(
                                "a second backdrop was installed over one "
                                "that was already there"))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        home = _FakeHome()
        home._ambient = "already there"
        home._install_ambient()

        assert home._ambient == "already there"


# ---------------------------------------------------------------------------
# The screens that build their own
# ---------------------------------------------------------------------------

class TestAScreenThatIsNotAnAppScreen:

    def test_the_backdrop_install_is_its_own_step(self):
        """THE PIN, for the split in ``_theme_screen``.

        The retry re-enters the backdrop install alone.  Re-running
        ``_theme_screen`` would re-apply the stylesheet to the whole tree
        every 120 ms while the preloader runs, which is the repolish
        storm this file's neighbours were written about.
        """
        from spacr.qt import app as APP

        theming = inspect.getsource(APP.MainWindow._theme_screen)
        assert "self._install_screen_backdrop(screen, key)" in theming
        assert "install_ambient(" not in theming

        installing = inspect.getsource(APP.MainWindow._install_screen_backdrop)
        assert "_the_heavy_import_lock_is_free()" in installing
        assert "_the_backdrop_wants_a_retry" in installing

    def test_the_retry_checks_the_screen_is_still_alive(self):
        """A module can be closed inside the 120 ms, and calling a method
        on a freed QWidget is the "Internal C++ object already deleted"
        storm this file's neighbours have had before."""
        from spacr.qt import app as APP

        retry = inspect.getsource(APP.MainWindow._retry_screen_backdrop)
        assert "from shiboken6 import isValid" in retry
        assert "if not isValid(screen):" in retry
        assert "QTimer.singleShot(120, again)" in retry


# ---------------------------------------------------------------------------
# The asymmetry the maintainer reported
# ---------------------------------------------------------------------------

class TestOnlySpaceoutEverTookThisLock:

    def test_the_ordinary_ambient_backdrop_takes_no_lock(self):
        """Why ``spacr`` opened the same module cleanly.

        ``AmbientWidget`` is pure Python and Qt painting -- no GL context,
        so nothing to serialise against a CUDA import.  If it ever grows
        one, this test is where the asymmetry stops being the explanation.
        """
        from spacr.qt.widgets.ambient import AmbientWidget

        source = inspect.getsource(AmbientWidget.__init__)
        assert "HEAVY_IMPORT_LOCK" not in source
        assert "_heavy_import_lock" not in source

    def test_the_fractal_is_only_reached_under_spaceout(self, monkeypatch):
        from spacr.qt import theme
        from spacr.qt.widgets import ambient as A

        monkeypatch.setattr(theme, "spaceout_enabled", lambda: False)
        assert A._the_spaceout_fractal(object()) is None


class TestTheMeasurementIsRecorded:

    def test_the_numbers_that_justify_the_bound_are_written_down(self):
        """The before/after is the evidence, and a bound with no measured
        cost beside it is a magic number the next reader will change."""
        source = inspect.getsource(
            __import__("spacr.qt.widgets.fractal_travel",
                       fromlist=["_"]))
        assert "2,130 ms" in source or "2130 ms" in source
        assert "priority inversion" in source


class TestHomesOwnHandler:

    def test_a_refusal_that_gets_past_the_peek_is_retried(self, monkeypatch):
        """THE ARC: the ``except`` half of Home's retry.

        The peek covers the case it can see; this covers the one it
        cannot. The preloader re-takes the lock between two imports, so a
        peek that answered yes is routinely followed by an install that
        is refused -- and Home, unlike a module screen, is built once and
        rebuilt only on a theme change, so a refusal it forgot would cost
        the backdrop for the whole session.
        """
        from spacr.qt.widgets import ambient as A
        from spacr.qt.widgets.fractal_travel import _HeavyImportInProgress

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                _HeavyImportInProgress("busy")))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        home = _FakeHome()
        home._install_ambient()

        assert home._ambient is None
        assert [ms for ms, _ in scheduled.calls] == [120]

    def test_a_real_failure_is_reported_once_and_not_retried(self,
                                                             monkeypatch):
        """THE OTHER ARC, and the one that keeps the console usable: a
        machine that cannot draw the backdrop must not be asked again
        every 120 ms for the life of the session."""
        from spacr.qt.widgets import ambient as A

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no context")))
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)

        home = _FakeHome()
        home._install_ambient()

        assert scheduled.calls == []
        assert home._ambient is None


class TestTheClassifierCannotBeTheThingThatIsMissing:

    def test_no_backdrop_module_means_nothing_could_have_raised_it(
            self, monkeypatch):
        """THE ARC: ``_the_backdrop_wants_a_retry``'s own import guard.

        It is asked to classify a failure, and "the backdrop module is
        absent" is one of the failures it classifies -- so it has to
        survive being unable to import the class it compares against.
        Answering ``False`` is right: with no class, nothing can be an
        instance of it, and a retry would be scheduled forever against a
        module that is not there.
        """
        import sys

        from spacr.qt.widgets.ambient import _the_backdrop_wants_a_retry

        monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fractal_travel",
                            None)

        assert _the_backdrop_wants_a_retry(RuntimeError("anything")) is False

    def test_the_peek_answers_yes_when_there_is_no_lock_to_ask(
            self, monkeypatch):
        """A machine without the backdrop module behaves exactly as it did
        before the feature existed: the screen builds, undecorated."""
        import sys

        from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

        monkeypatch.setitem(sys.modules, "spacr.qt.widgets.fractal_travel",
                            None)

        assert _the_heavy_import_lock_is_free() is True

    def test_home_survives_the_ambient_module_being_gone(self, monkeypatch):
        """THE ARC: Home's own defensive import, inside the handler.

        The install raises because the module cannot be imported, and the
        handler then reaches for the same module to ask what kind of
        failure it was. Without its own guard that is a second
        ImportError, escaping a handler whose entire contract is that a
        backdrop can never stop Home being built.
        """
        import sys

        from spacr.qt.widgets.home import HomePage

        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", None)

        home = _FakeHome()
        HomePage._install_ambient(home)

        assert home._ambient is None


class TestTheRemainingHalves:

    def test_a_lock_that_answers_none_is_treated_as_free(self, monkeypatch):
        """THE ARC: ``if lock is None``.

        Distinct from the import failing. ``_heavy_import_lock`` answers
        None when the widget module is present but the application around
        it is not -- the backdrop is usable on its own, and a lock that
        does not exist cannot be held, so the answer is yes.
        """
        from spacr.qt.widgets import fractal_travel as F
        from spacr.qt.widgets.ambient import _the_heavy_import_lock_is_free

        monkeypatch.setattr(F, "_heavy_import_lock", lambda: None)

        assert _the_heavy_import_lock_is_free() is True

    def test_home_builds_nothing_when_the_animation_is_switched_off(
            self, monkeypatch):
        """THE ARC: the preference read, which comes BEFORE anything is
        constructed.

        Off means not built, not built-and-hidden: the construction is
        itself the cost the toggle exists to avoid on a machine running
        Cellpose on the GPU.
        """
        from spacr.qt.widgets import ambient as A

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: False)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free",
                            lambda: pytest.fail(
                                "the lock was peeked at for a backdrop that "
                                "is switched off"))
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: pytest.fail(
                                "a backdrop was built while the animation "
                                "preference was off"))

        home = _FakeHome()
        home._install_ambient()

        assert home._ambient is None
        assert scheduled.calls == []

    def test_a_free_lock_lets_home_install_and_clear_its_surfaces(
            self, monkeypatch):
        """THE ARC: the ordinary success path, and the ordering on it.

        The surfaces are cleared only AFTER the install, so a screen
        whose install failed is left opaque and normal rather than
        transparent with nothing behind it.
        """
        from spacr.qt.widgets import ambient as A

        scheduled = _Scheduled(monkeypatch)
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: "the backdrop")

        home = _FakeHome()
        home._install_ambient()

        assert home._ambient == "the backdrop"
        assert home.cleared == 1
        assert home.discarded == []
        assert scheduled.calls == []


class _FakeWindow:
    """Enough of ``MainWindow`` to drive the two backdrop methods."""

    def __init__(self):
        self.retries = []

    def _install_screen_backdrop(self, screen, key):
        from spacr.qt.app import MainWindow

        return MainWindow._install_screen_backdrop(self, screen, key)

    def _retry_screen_backdrop(self, screen, key):
        self.retries.append(key)


class TestTheScreensThatBuildTheirOwn:
    """``MainWindow._install_screen_backdrop``, driven rather than read.

    These screens took the same freeze as ``AppScreen`` and had none of
    its care: no peek, no retry, and the install inlined in
    ``_theme_screen`` where a retry would have re-applied the stylesheet
    to the whole tree.
    """

    def test_a_switched_off_animation_is_not_built_and_not_retried(
            self, monkeypatch):
        from spacr.qt.widgets import ambient as A

        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: False)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: pytest.fail(
                                "a backdrop was built while the animation "
                                "preference was off"))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert window.retries == []

    def test_a_held_lock_defers_instead_of_building(self, monkeypatch):
        """THE ARC the peek exists for, on the path that never had one."""
        from spacr.qt.widgets import ambient as A

        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: False)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: pytest.fail(
                                "the backdrop was built while the heavy "
                                "import lock was held"))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert window.retries == ["some_screen"]

    def test_a_free_lock_installs_it(self, monkeypatch):
        from spacr.qt.widgets import ambient as A

        built = []
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: built.append("built"))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert built == ["built"]
        assert window.retries == []

    def test_a_refusal_past_the_peek_is_retried_not_logged(self,
                                                           monkeypatch):
        """THE ARC: the ``except`` half, which is the one the peek cannot
        cover -- and it must not reach ``LOG.exception``, or an ordinary
        click during startup puts a traceback in the console."""
        from spacr.qt import app as APP
        from spacr.qt.widgets import ambient as A
        from spacr.qt.widgets.fractal_travel import _HeavyImportInProgress

        logged = []
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                _HeavyImportInProgress("busy")))
        monkeypatch.setattr(APP.LOG, "exception",
                            lambda *a, **k: logged.append(a))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert window.retries == ["some_screen"]
        assert logged == [], (
            "a busy lock was logged as an exception, so opening a module "
            "during startup now puts a traceback in the console")

    def test_a_real_failure_is_logged_once_and_not_retried(self,
                                                           monkeypatch):
        from spacr.qt import app as APP
        from spacr.qt.widgets import ambient as A

        logged = []
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setattr(A, "_the_heavy_import_lock_is_free", lambda: True)
        monkeypatch.setattr(A, "install_ambient",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("no context")))
        monkeypatch.setattr(APP.LOG, "exception",
                            lambda *a, **k: logged.append(a))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert window.retries == []
        assert len(logged) == 1

    def test_the_classifier_being_gone_falls_back_to_logging(self,
                                                             monkeypatch):
        """THE ARC: ``_the_backdrop_wants_a_retry = None``.

        The handler absorbs a missing ambient module, so it cannot lean
        on that module to classify the failure. With no classifier the
        honest answer is the old one -- log it once -- rather than
        retrying forever against something that is not there.
        """
        import sys

        from spacr.qt import app as APP

        logged = []
        monkeypatch.setattr("spacr.qt.preferences.get_ambient_enabled",
                            lambda: True)
        monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient", None)
        monkeypatch.setattr(APP.LOG, "exception",
                            lambda *a, **k: logged.append(a))

        window = _FakeWindow()
        window._install_screen_backdrop(object(), "some_screen")

        assert window.retries == []
        assert len(logged) == 1


class TestTheRetryChecksTheScreenIsStillThere:

    def test_a_screen_that_was_closed_is_not_called_back(self, monkeypatch,
                                                         qtbot):
        """THE ARC: ``if not isValid(screen)``.

        A module can be closed inside the 120 ms, and calling a method on
        a freed QWidget is the "Internal C++ object already deleted"
        storm this file's neighbours have had before.
        """
        import shiboken6
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QWidget

        from spacr.qt.app import MainWindow

        pending = []
        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: pending.append((ms, fn))))

        window = _FakeWindow()
        installs = []
        window._install_screen_backdrop = lambda s, k: installs.append(k)

        screen = QWidget()
        MainWindow._retry_screen_backdrop(window, screen, "some_screen")
        assert [ms for ms, _ in pending] == [120]

        screen.deleteLater()
        shiboken6.delete(screen)
        assert not shiboken6.isValid(screen)

        pending[0][1]()
        assert installs == [], (
            "the retry called back into a screen Qt had already freed")

    def test_a_screen_that_is_still_open_is_called_back(self, monkeypatch,
                                                        qtbot):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QWidget

        from spacr.qt.app import MainWindow

        pending = []
        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: pending.append((ms, fn))))

        window = _FakeWindow()
        installs = []
        window._install_screen_backdrop = lambda s, k: installs.append(k)

        screen = QWidget()
        qtbot.addWidget(screen)
        MainWindow._retry_screen_backdrop(window, screen, "some_screen")
        pending[0][1]()

        assert installs == ["some_screen"]

    def test_no_shiboken_means_no_callback_rather_than_a_guess(
            self, monkeypatch, qtbot):
        """THE ARC: ``except Exception: return`` around the liveness check.

        shiboken6 is PySide6's own binding runtime, so it is present
        wherever the widget it would be asked about is -- which is why
        this cannot run in a real launch. It is written down rather than
        assumed because the alternative, calling back without checking,
        is the one outcome the check exists to prevent: with no way to
        ask whether the screen is alive, not calling is the safe answer
        and a missing backdrop is the whole cost.
        """
        import sys

        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QWidget

        from spacr.qt.app import MainWindow

        pending = []
        monkeypatch.setattr(
            QTimer, "singleShot",
            staticmethod(lambda ms, fn: pending.append((ms, fn))))

        window = _FakeWindow()
        installs = []
        window._install_screen_backdrop = lambda s, k: installs.append(k)

        screen = QWidget()
        qtbot.addWidget(screen)
        MainWindow._retry_screen_backdrop(window, screen, "some_screen")

        monkeypatch.setitem(sys.modules, "shiboken6", None)
        pending[0][1]()

        assert installs == [], (
            "the retry called back without being able to check that the "
            "screen was still alive")
