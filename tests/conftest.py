"""
Shared pytest fixtures + synthetic-data builders for the spacr test suite.

Everything in here is DETERMINISTIC (fixed seeds) so failures can be
reproduced from a git hash alone. Fixtures are session-scoped where the
generated object is read-only; per-test fixtures reset writable state.

Fixtures provided:
    tmp_project_dir    per-test temp dir that gets wiped after the test
    rng                numpy Generator seeded to 0
    synth_image_2d     2-D uint16 grayscale "microscopy" image
    synth_image_3d     3-D uint16 image (Z, H, W)
    synth_image_stack  4-D uint16 stack (T, C, H, W)
    synth_mask_2d      2-D int label mask with N connected blobs
    synth_masks_multi  dict of cell/nucleus/pathogen label masks
    synth_measurements pandas DataFrame with typical spacr columns
    synth_sqlite_db    file-backed sqlite with a minimal spacr schema

Two fixtures are gone with the Tkinter interface. `dark_style` returned
``spacr.gui_elements.set_dark_style(...)``, whose module is deleted, and
`tk_root` handed out a hidden ``tkinter.Tk`` that only Tk widgets ever
needed. Nothing spaCR ships draws through Tk any more, so a test that wants
a live widget builds a Qt one -- see `tests/qt/` and the `qtbot` fixture
pytest-qt provides.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# The suite is not allowed to take the machine down
# ---------------------------------------------------------------------------
#
# THIS HAS HAPPENED TWICE, on 2026-09-04. The first time VS Code died. The
# second time the kernel's OOM killer took gnome-shell with it and logged the
# maintainer out mid-session. A single pytest process had reached 92 GB.
#
# Two earlier attempts were the wrong shape:
#
#   * a daemon polling /proc/meminfo every three seconds and killing the
#     largest offender. Its log shows it firing five times -- 92.0, 90.7,
#     89.1, 75.1, 14.6 GB -- and losing anyway. A process climbing to ninety
#     gigabytes outruns a three-second poll.
#   * `tools/run_capped.sh`, a cgroup cap, which works perfectly and which
#     nothing obliges anyone to use. Every run that skipped it was unguarded.
#
# So the limit lives HERE, where it binds however pytest was started -- by a
# person, by CI, or by an agent that had never heard of the wrapper. A thread
# watches this process's own RSS and takes the process out at the ceiling.
#
# `os._exit` on purpose. A MemoryError raised into arbitrary test code is
# caught by arbitrary test code; an exception cannot be relied on to end a
# process that is already thrashing. This leaves a message on stderr saying
# exactly what happened, so the next reader is not left guessing at an exit
# code the way this one was.
_MEMORY_CEILING_GB = float(os.environ.get("SPACR_TEST_MEMORY_GB", "12"))


def _stop_before_the_machine_does() -> None:
    """End this pytest if its own RSS passes the ceiling."""
    import threading

    if _MEMORY_CEILING_GB <= 0:            # explicitly disabled
        return

    def watch() -> None:
        import time
        page = os.sysconf("SC_PAGE_SIZE")
        statm = f"/proc/{os.getpid()}/statm"
        ceiling = _MEMORY_CEILING_GB * 1024 ** 3
        while True:
            time.sleep(2.0)
            try:
                with open(statm, encoding="ascii") as handle:
                    rss = int(handle.read().split()[1]) * page
            except (OSError, IndexError, ValueError):
                return
            if rss < ceiling:
                continue
            # WRITTEN TO FD 2 DIRECTLY, not through `sys.stderr`. pytest
            # replaces the stream to capture output, and a message written
            # into a capture buffer that is never drained -- because the
            # process is about to end -- is a message nobody reads. Verified:
            # the first version of this guard exited 3 and left an empty log.
            message = (
                f"\n\nspaCR test guard: this pytest reached "
                f"{rss / 1024 ** 3:.1f} GB, over the "
                f"{_MEMORY_CEILING_GB:.0f} GB ceiling, and is being ended "
                f"before it takes the machine with it.\n"
                f"Raise it deliberately with SPACR_TEST_MEMORY_GB=<n> if a "
                f"run genuinely needs more.\n\n")
            try:
                os.write(2, message.encode("utf-8", "replace"))
            except OSError:
                pass
            os._exit(3)

    threading.Thread(target=watch, daemon=True,
                     name="spacr-test-memory-guard").start()


_stop_before_the_machine_does()

# A pytest process must never be able to mutate the public GitHub tracker.
#
# ``PYTEST_CURRENT_TEST`` is phase-local: pytest removes it between tests and
# before session teardown.  It is therefore not a process-lifetime safety
# boundary, and it does not reliably reach children launched outside a test
# call phase.  This sentinel is installed while the root conftest is imported,
# before collection, and is inherited by every ordinary subprocess.  Do not
# remove it in a fixture -- fixture teardown was the hole that let real issue
# comments escape in the first place.
os.environ["SPACR_PYTEST_SESSION"] = "1"
# Older tests used this process-wide escape hatch.  A child inherited it and
# could use the developer's real ``gh`` credential, so it is intentionally
# inert now and cleared before any test module is imported.
os.environ.pop("SPACR_ALLOW_GITHUB_WRITES", None)

# Make the in-tree spacr importable without an editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Headless matplotlib for CI / test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

# THE SUITE DOES NOT WRITE INTO THE USER'S OWN LOG.
#
# `spacr.logging_util` logs to ~/.spacr/logs/spacr.log, and running the tests
# filled that file with tracebacks the tests THREW ON PURPOSE --
# ConnectionError("no dns"), MemoryError("the merged array will not fit"),
# ValueError("unreadable names") -- interleaved with real pipeline output.
#
# That is not untidiness. On 2026-09-01 a Measure run failed one field of
# fifty-two and its message said the traceback was in that log; finding it
# meant reading past a screenful of deliberate test failures, and a user
# reading their own crash report has no way to tell which lines are theirs.
#
# Set here, at import, rather than in a fixture: `log_dir()` is read the first
# time anything configures logging, which can happen while a test module is
# being imported -- before any fixture has run.
os.environ.setdefault(
    "SPACR_LOG_DIR",
    os.path.join(tempfile.gettempdir(), "spacr-test-logs"))

# ---------------------------------------------------------------------------
# Pre-empt display-touching imports. Three packages open the X display at
# IMPORT time and throw Xlib.error.DisplayConnectionError in a display-less
# subprocess run:
#   * mouseinfo (transitive via pyautogui)
#   * pyautogui itself (Linux backend probes the display)
#   * screeninfo.get_monitors
# The spacr modules that pulled them in at module load -- gui.py,
# gui_utils.py, gui_elements.py -- are deleted, so nothing spaCR ships
# reaches them now. The stubs stay because a test module, or a dependency
# one of them imports, can still name any of the three, and a no-op module
# is cheaper than an import that has to be guarded at every call site.
# ---------------------------------------------------------------------------
import types as _types


def _install_gui_stubs():
    def _no_op(*args, **kwargs):
        return None

    if "mouseinfo" not in sys.modules:
        mi = _types.ModuleType("mouseinfo")
        mi.position = lambda: (0, 0)
        mi.size = lambda: (0, 0)
        mi.MOUSE_LEFT = mi.MOUSE_RIGHT = mi.MOUSE_MIDDLE = None
        mi.PRIMARY = "primary"
        mi._display = None
        class MouseInfoException(Exception):  # noqa: N801 - upstream name
            pass
        mi.MouseInfoException = MouseInfoException
        sys.modules["mouseinfo"] = mi

    if "pyautogui" not in sys.modules:
        try:
            import pyautogui  # noqa: F401 - real import if available
        except Exception:
            pa = _types.ModuleType("pyautogui")
            for name in ("position", "size", "click", "moveTo", "moveRel",
                         "typewrite", "hotkey", "press", "screenshot",
                         "onScreen", "keyDown", "keyUp"):
                setattr(pa, name, _no_op)
            pa.FAILSAFE = True
            pa.PAUSE = 0.0
            sys.modules["pyautogui"] = pa

    if "screeninfo" not in sys.modules:
        try:
            import screeninfo  # noqa: F401
            # Verify get_monitors works; if not, patch it.
            try:
                screeninfo.get_monitors()
            except Exception:
                _fake = _types.SimpleNamespace(x=0, y=0, width=1920, height=1080,
                                               name="stub", is_primary=True)
                screeninfo.get_monitors = lambda: [_fake]
        except Exception:
            si = _types.ModuleType("screeninfo")
            _fake = _types.SimpleNamespace(x=0, y=0, width=1920, height=1080,
                                           name="stub", is_primary=True)
            si.get_monitors = lambda: [_fake]
            sys.modules["screeninfo"] = si


_install_gui_stubs()

# Fail collection if Python resolved ``spacr`` from another checkout.  The
# assertion is intentionally based on this file's location so it works in a
# developer clone, a git worktree, and GitHub Actions alike.
import spacr as _spacr_under_test

_EXPECTED_PACKAGE_ROOT = (_REPO_ROOT / "spacr").resolve()
_IMPORTED_PACKAGE_ROOT = Path(_spacr_under_test.__file__).resolve().parent
if _IMPORTED_PACKAGE_ROOT != _EXPECTED_PACKAGE_ROOT:
    raise RuntimeError(
        "pytest imported spaCR from the wrong checkout: "
        f"{_IMPORTED_PACKAGE_ROOT} (expected {_EXPECTED_PACKAGE_ROOT})"
    )


# ---------------------------------------------------------------------------
# QSettings sandbox
#
# `QSettings("spacr", "qt")` — the two-argument constructor every spacr Qt
# module uses — is built with `QSettings::NativeFormat`, ALWAYS. It ignores
# `QSettings.setDefaultFormat(IniFormat)`, so the redirect that several test
# modules reach for
#
#     QSettings.setDefaultFormat(QSettings.IniFormat)
#     QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tmp_path)
#
# isolates nothing at all: the object still resolves to the developer's real
# `~/.config/spacr/qt.conf`, and the `.clear()` those fixtures follow it with
# DELETES THE USER'S PREFERENCES — 53 times in one full run. The same hole let
# `spacr.qt.prefs` write pytest tmp paths into the real
# `~/.config/Olafsson Lab/spaCR.conf`, where a later test could read them back:
# cross-test pollution AND collateral damage from one bug.
#
# The fix is to redirect every (format, scope) pair, including NativeFormat,
# at a throwaway sandbox before a single test module is imported, and then to
# re-point it at a PER-TEST directory so no two tests ever share a store. A
# teardown guard fails the offending test loudly if its QSettings ever
# resolved outside the sandbox or the pytest temp tree.
# ---------------------------------------------------------------------------

import atexit as _atexit
import faulthandler as _faulthandler
import hashlib as _hashlib
import shutil as _shutil
import threading as _threading

#: Throwaway root that stands in for the user's config directory.
# RESOLVED, and that is the whole fix for macOS and Windows.
# `_inside_allowed_root` resolves the path it is checking, so the roots it
# checks against have to be resolved too or they can never match. On macOS
# `tempfile.mkdtemp()` returns `/var/folders/...` and `/var` is a symlink to
# `/private/var`, so the probe resolved to `/private/var/...` and compared
# against `/var/...`; on Windows the same thing happens with 8.3 short names
# (`RUNNER~1`). Both platforms reported "QSettings escaped the test sandbox"
# for settings that were sitting inside it. Linux never saw it because
# nothing there is a symlink.
_QSETTINGS_SANDBOX = Path(
    tempfile.mkdtemp(prefix="spacr-qsettings-")).resolve()
_atexit.register(_shutil.rmtree, str(_QSETTINGS_SANDBOX), True)

#: `(exists, size, mtime_ns)` for the real files a leak would damage, taken
#: once before the redirect goes in. Empty when PySide6 is not installed.
_QSETTINGS_REAL_STATE: dict = {}

#: Directories a QSettings file is allowed to resolve under. The sandbox plus
#: pytest's own basetemp, because a test module may legitimately point its own
#: settings at its ``tmp_path``.
_QSETTINGS_ALLOWED_ROOTS: list = [_QSETTINGS_SANDBOX]

_QSETTINGS_ACTIVE = False

#: Whether `setPath` actually moves the two-argument constructor on this
#: platform. False on macOS and Windows -- see `_install_qsettings_sandbox`.
_QSETTINGS_PATHS_REDIRECTABLE = True


def _qsettings_module():
    """Return ``PySide6.QtCore.QSettings``, or None when PySide6 is absent."""
    try:
        from PySide6.QtCore import QSettings
    except Exception:
        return None
    return QSettings


def _redirect_qsettings(target) -> None:
    """Point every QSettings (format, scope) pair at ``target``.

    NativeFormat is the one that matters — it is what the two-argument
    constructor uses — but all four combinations are redirected so no
    constructor spelling can escape.
    """
    settings = _qsettings_module()
    if settings is None:
        return
    # NOTE, because this is the second time it has been "fixed" wrongly:
    # `setDefaultFormat(IniFormat)` does NOT help. The block at the top of
    # this section already says why -- `QSettings(org, app)` is built with
    # NativeFormat ALWAYS and ignores the default format. Adding that call
    # here changed nothing on macOS and cost a CI round to find out.
    for fmt in (settings.NativeFormat, settings.IniFormat):
        for scope in (settings.UserScope, settings.SystemScope):
            settings.setPath(fmt, scope, str(target))


def _qsettings_probe_paths() -> list:
    """Where the org/app pairs spacr uses would land right now."""
    settings = _qsettings_module()
    if settings is None:
        return []
    pairs = (("spacr", "qt"), ("Olafsson Lab", "spaCR"))
    out = []
    for org, app in pairs:
        try:
            out.append(settings(org, app).fileName())
        except Exception:
            continue
    return out


def _stat_signature(path) -> tuple:
    try:
        info = os.stat(path)
    except OSError:
        return (False, 0, 0)
    return (True, info.st_size, info.st_mtime_ns)


#: Plugin name for the collection-node canonicaliser defined further down.
_ONE_NODE_PER_DIRECTORY = "spacr-one-node-per-directory"


def pytest_configure(config):
    """Sandbox QSettings, and pin one collection node per directory.

    Collection imports test modules, and a module-level ``QSettings(...)``
    would otherwise hit the real store, so this runs in ``pytest_configure``
    rather than in a fixture. The directory-node plugin is registered here
    for the same reason -- it has to be in place before the first directory
    is collected, and a conftest hook only reaches nodes at or below its own
    directory, which is one level too late to keep ``tests`` itself stable.
    """
    # SettingWithCopyWarning, ONLY WHERE IT STILL EXISTS. Writing through a
    # slice is a real bug and this suite promotes it to an error -- but
    # pandas 3 DELETED the class, because copy-on-write made the warning
    # unnecessary, and a `filterwarnings` line in pytest.ini naming a class
    # that is gone is an AttributeError during collection: the whole suite
    # fails to start, before a single test runs. Registered here instead, so
    # the guard holds on pandas 2 and simply does not apply on pandas 3,
    # where the fault it guards against cannot happen.
    try:
        from pandas.errors import SettingWithCopyWarning
    except ImportError:
        pass
    else:
        config.addinivalue_line(
            "filterwarnings", "error::pandas.errors.SettingWithCopyWarning")

    if not config.pluginmanager.has_plugin(_ONE_NODE_PER_DIRECTORY):
        config.pluginmanager.register(_OneNodePerDirectory(),
                                      _ONE_NODE_PER_DIRECTORY)

    global _QSETTINGS_ACTIVE
    if _qsettings_module() is None:
        return
    # Snapshot the real files FIRST — the guard compares against this.
    for real in _qsettings_probe_paths():
        _QSETTINGS_REAL_STATE[real] = _stat_signature(real)
        parent = os.path.dirname(real)
        _QSETTINGS_REAL_STATE[parent] = _stat_signature(parent)
    _redirect_qsettings(_QSETTINGS_SANDBOX)
    _QSETTINGS_ACTIVE = True
    # Can NativeFormat be redirected on this platform at all? Asked by
    # trying it rather than by naming operating systems, because the answer
    # is a property of the Qt backend and not of the OS badge.
    #
    # On Linux NativeFormat IS IniFormat and `setPath` moves it. On macOS it
    # is CFPreferences -- a plist under ~/Library/Preferences -- and on
    # Windows it is the registry; neither is a directory, so nothing can
    # move them and the probe below resolves to
    # `/Users/runner/Library/Preferences/com.spacr.qt.plist` or
    # `\HKEY_CURRENT_USER\Software\spacr\qt` however the sandbox is set.
    #
    # Where it cannot be redirected, the "did the path stay inside the
    # sandbox" check is structurally unanswerable and is turned off. The
    # check that MATTERS is kept on every platform: whether the real store
    # was modified. That is the actual harm -- deleting a developer's
    # preferences -- and `_QSETTINGS_REAL_STATE` detects it directly rather
    # than by inference from a path.
    global _QSETTINGS_PATHS_REDIRECTABLE
    probes = _qsettings_probe_paths()
    _QSETTINGS_PATHS_REDIRECTABLE = bool(probes) and all(
        _inside_allowed_root(p) for p in probes)

    try:
        basetemp = config._tmp_path_factory.getbasetemp()
    except Exception:
        basetemp = None
    if basetemp is not None:
        _QSETTINGS_ALLOWED_ROOTS.append(Path(str(basetemp)).resolve())
    # pytest's basetemp lives under a per-user root that also holds the
    # previous runs' directories; allow the whole tree so a module fixture
    # pointing at its own tmp_path is fine.
    _QSETTINGS_ALLOWED_ROOTS.append(
        Path(tempfile.gettempdir()).resolve() / f"pytest-of-{_current_user()}")


def _current_user() -> str:
    for key in ("USER", "USERNAME", "LOGNAME"):
        value = os.environ.get(key)
        if value:
            return value
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return "unknown"


def _inside_allowed_root(path) -> bool:
    try:
        resolved = Path(path).resolve()
    except Exception:
        return False
    for root in _QSETTINGS_ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


@pytest.fixture(autouse=True)
def _the_widget_tree_does_not_outgrow_the_session(_isolated_qsettings_store):
    """Deliver owner-requested Qt deletions for every Qt test boundary.

    ``tests/qt/conftest.py`` already delivers pending ``deleteLater`` calls
    at each test boundary — but a conftest
    only reaches its own directory, and the ``qt`` marker is much wider than
    that directory. Well over a hundred modules directly under ``tests/``
    carry ``@pytest.mark.qt``, build real widgets, and ran with none of that
    housekeeping, so their widgets accumulated for the whole job.

    That is what a Qt shard ends holding. Measured over thirty of those
    modules in one process, 330 tests:

        without this    peak 5,945 top-level windows / 39,713 widgets,
                        3,161 windows still standing at the end
        with it         peak   797 top-level windows /  5,709 widgets,
                          158 windows still standing at the end

    The cost of carrying that tree is paid by every test after it, because a
    palette or style change visits every live widget — and at the end it is
    paid once more by the process, which destroys the tree one object at a
    time before it can print a summary.

    Free for tests that are not Qt tests at all. PySide6 is never imported
    here: a run that has not already loaded it has no widgets to flush, so
    the fixture reads one entry in ``sys.modules`` and yields.

    Nothing is reached across. ``sendPostedEvents`` delivers only deletions
    their owners already requested, at SETUP where the previous test's
    teardown is complete. Do not run Python's cycle collector here: a wrapper
    can be unreachable while its C++ QThread is still running, and CI proved
    that collecting such a live Qt heap can segfault inside ``gc.collect``.
    Qt objects must instead be registered with ``qtbot`` or explicitly call
    ``deleteLater``; this boundary only completes that ownership protocol.

    Ordered behind the QSettings sandbox, and depending on it by name rather
    than by where it sits in this file, because destroying a widget can run
    a ``closeEvent`` that writes a preference. Whatever those writes land in
    has to be a sandbox already.
    """
    module = sys.modules.get("PySide6.QtWidgets")
    if module is not None:
        try:
            from PySide6.QtCore import QEvent

            app = module.QApplication.instance()
            if app is not None:
                module.QApplication.sendPostedEvents(
                    None, QEvent.DeferredDelete)
        except Exception:                                        # noqa: BLE001
            pass
    yield


@pytest.fixture(autouse=True)
def _no_provider_stream_outlives_a_test():
    """End any AI provider subprocess a test leaves being read.

    The reader thread BLOCKS on the child's stdout, so it does not notice a
    flag; only ending the child lets that read return. A thread still
    blocked when the session's collection pass runs takes the whole process
    with it -- Qt aborts as soon as the running QThread's wrapper is
    collected, and an abort kills every remaining test in the run, not one.

    Each file passed on its own, which is exactly why this belongs here: the
    leak and the collection that turns it fatal were in different files.
    """
    yield
    try:
        from spacr.qt.ai.providers import terminate_all_streams
    except Exception:                                            # noqa: BLE001
        return
    try:
        terminate_all_streams()
    except Exception:                                            # noqa: BLE001
        pass


#: Sandbox for everything the app keeps under `~/.spacr`. Session-wide and
#: created once, like the QSettings one above.
_DOT_SPACR_SANDBOX = Path(
    tempfile.mkdtemp(prefix="spacr-dot-spacr-")).resolve()
_atexit.register(_shutil.rmtree, str(_DOT_SPACR_SANDBOX), True)


@pytest.fixture(autouse=True)
def _isolated_dot_spacr_store(monkeypatch):
    """Keep the run journal and the plate queue out of the real `~/.spacr`.

    THIS IS NOT HYGIENE, IT IS A BUG THAT SHIPPED. Measured on the
    maintainer's machine 2026-09-03: `~/.spacr/runs` held 11,046 run
    folders and 7,323 of them were named `__job` or `___job` -- the app_key
    a test fixture opens a run with, written the same afternoon. So Home's
    Totals panel read "11,027 runs" to somebody who had done about a dozen,
    Recent runs listed four `_job` rows that navigate to a module that does
    not exist, and `~/.spacr/queue.json` held seven queued plates pointing
    at `/tmp/x`. Every one of those was a test's, and all three were
    reported as application bugs because from the outside that is exactly
    what they look like.

    Redirected at the FUNCTION that resolves the path rather than by moving
    `HOME`, because moving `HOME` for the session moves conda's, matplotlib's
    and Qt's caches too, and this suite is not the place to find out what
    that breaks. Both modules call their resolver on every use -- checked --
    so nothing captures the real path at import time.

    A test that wants its own directory still monkeypatches these itself and
    wins, because `monkeypatch` is LIFO: `test_home_v2._queue_at` already
    does exactly that and keeps working.
    """
    root = _DOT_SPACR_SANDBOX / "runs"
    root.mkdir(parents=True, exist_ok=True)
    try:
        from spacr import run_journal
    except Exception:                                            # noqa: BLE001
        pass
    else:
        monkeypatch.setattr(run_journal, "runs_root", lambda: root,
                            raising=False)
    try:
        from spacr.qt import plate_queue
    except Exception:                                            # noqa: BLE001
        pass
    else:
        monkeypatch.setattr(
            plate_queue, "_queue_path",
            lambda: _DOT_SPACR_SANDBOX / "queue.json", raising=False)
    yield


@pytest.fixture(autouse=True)
def _isolated_qsettings_store(request):
    """Give every test its own QSettings directory, and prove it stayed there.

    Autouse and declared in the ROOT conftest so it is set up before any
    per-module preference fixture and torn down after all of them — which is
    what makes the teardown assertion meaningful.
    """
    if not _QSETTINGS_ACTIVE:
        yield
        return
    digest = _hashlib.sha1(
        request.node.nodeid.encode("utf-8", "replace")).hexdigest()[:16]
    per_test = _QSETTINGS_SANDBOX / "per-test" / digest
    _redirect_qsettings(per_test)
    try:
        yield
    finally:
        escaped = ([p for p in _qsettings_probe_paths()
                    if not _inside_allowed_root(p)]
                   if _QSETTINGS_PATHS_REDIRECTABLE else [])
        damaged = [p for p, was in _QSETTINGS_REAL_STATE.items()
                   if _stat_signature(p) != was]
        # Always restore the sandbox, even on failure, so one leaky module
        # cannot drag the rest of the session down with it. Re-baselining the
        # real files keeps the blame on the FIRST test that touched them
        # instead of failing every test that runs after it.
        _redirect_qsettings(_QSETTINGS_SANDBOX)
        for path in damaged:
            _QSETTINGS_REAL_STATE[path] = _stat_signature(path)
        if escaped:
            raise AssertionError(
                "QSettings escaped the test sandbox.\n"
                f"  resolved outside the sandbox: {escaped}\n"
                f"  real files touched: {damaged}\n"
                "Use `QSettings(str(tmp_path / 'x.ini'), QSettings.IniFormat)` "
                "or monkeypatch the module's `_settings` factory. "
                "`QSettings.setPath(IniFormat, ...)` does NOT redirect "
                "`QSettings(org, app)` — that constructor is NativeFormat.")
        if damaged:
            # A changed real file with the sandbox still correctly in place
            # means something OUTSIDE this process wrote it — a second pytest
            # run, or the app itself. Warn rather than fail: an unrelated
            # process must not be able to turn this suite red.
            import warnings
            warnings.warn(
                "the real QSettings files changed while a test ran, but this "
                f"process was correctly sandboxed: {damaged}. Another process "
                "is writing them.", RuntimeWarning, stacklevel=1)


# ---------------------------------------------------------------------------
# CI suite classification
# ---------------------------------------------------------------------------

_INTEGRATION_MODULE_TOKENS = (
    "e2e",
    "integration",
    "pipeline",
    "real_data",
    "real_dataset",
)


def _automatic_ci_markers(path):
    """Return path-derived suite markers used to partition CI.

    Resource and duration markers stay explicit on tests because they encode
    runtime requirements. Qt ownership and integration scope are structural:
    every module below ``tests/qt`` is a Qt test, while modules whose names
    advertise an end-to-end, pipeline, integration, or real-data boundary are
    integration tests. Keeping these two rules here prevents a newly added Qt
    or end-to-end module from silently leaking into the fast suite.
    """
    test_path = Path(str(path))
    markers = set()
    if "tests" in test_path.parts and "qt" in test_path.parts:
        markers.add("qt")
    if any(token in test_path.stem for token in _INTEGRATION_MODULE_TOKENS):
        markers.add("integration")
    return markers


def _ci_file_shard(path, count):
    """Return a stable zero-based CI shard for one test module path."""
    test_path = Path(str(path))
    try:
        label = test_path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        label = test_path.as_posix()
    digest = _hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(count)


#: How much of the card a `@pytest.mark.gpu` test needs before it is worth
#: starting, in MiB. Generous: the tiles and batches in this suite are tiny,
#: so this is a check that the card is USABLE rather than a measurement of
#: any particular test. Override with SPACR_PYTEST_GPU_ROOM_MB.
GPU_ROOM_MB = int(os.environ.get("SPACR_PYTEST_GPU_ROOM_MB", "1500"))


def _no_room_on_the_gpu():
    """Why a GPU test cannot run right now, or '' when it can.

    THE CARD IS SHARED. Another session's training run holding 21 GiB of 24
    is the ordinary state of this machine, and nothing here may do anything
    about it -- a test does not get to kill somebody's training. Nine tests
    across four files failed with `torch.OutOfMemoryError: Tried to allocate
    20.00 MiB` for exactly that reason, which is red for a condition that
    has nothing to do with spaCR.

    Checked ONCE PER TEST rather than at collection, because the card fills
    and empties while a long suite runs.
    """
    try:
        import torch
    except Exception:                                        # noqa: BLE001
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:                                        # noqa: BLE001
        # An older driver with no mem_get_info: let the test try, and let a
        # real OOM be a real failure. Guessing would be worse.
        return ""
    free_mb, total_mb = free / (1024 * 1024), total / (1024 * 1024)
    if free_mb >= GPU_ROOM_MB:
        return ""
    return (f"the GPU is busy: {free_mb:.0f} MiB free of {total_mb:.0f}, "
            f"and this needs about {GPU_ROOM_MB}")


def pytest_runtest_setup(item):
    """Skip a GPU-marked test when the shared card has no room for it.

    ONE PLACE RATHER THAN PER FILE. Every test that needs the card already
    carries `@pytest.mark.gpu`, and a guard written into each file is a
    guard the next file will not have.
    """
    if item.get_closest_marker("gpu") is None:
        return
    trouble = _no_room_on_the_gpu()
    if trouble:
        pytest.skip(trouble)


# ---------------------------------------------------------------------------
# The run ends, even when shutting down does not
# ---------------------------------------------------------------------------
#
# A pytest run can finish every test and still never report. Once the session
# is over pytest's own ``--timeout`` is gone -- it is a per-test guard -- so a
# process that will not exit stops the run somewhere after the last test and
# before the summary line, with no test to blame and no output to read. Under
# ``-n`` it is worse: the controller waits on workers that have already
# written their results and are burning a core apiece, so the run costs hours
# and produces nothing.
#
# Python joins every NON-DAEMON thread before it finalises, which is what
# turns one forgotten thread into an unbounded wait, and Qt adds its own ways
# to stall on the way out. Neither can be fixed by guessing, so what is
# installed here is the thing that makes the next one findable: the threads
# that will hold the interpreter open are named while the run can still print,
# and a watchdog turns an endless shutdown into a stack dump for every thread
# followed by a non-zero exit.
#
# THE BUDGET STARTS THE INSTANT THE LAST TEST ENDS, from the innermost
# ``pytest_runtestloop`` wrapper, and that timing is the point.
# ``pytest_sessionfinish`` is one hook with many implementations and this
# file's runs last of them, so arming there leaves the whole of teardown up to
# that moment unguarded -- and teardown is where the expensive work is.
# pytest-cov writes the run's coverage data from the tail of its OWN
# ``pytest_runtestloop`` wrapper, before any ``sessionfinish`` runs at all, and
# a distributed worker reports itself finished from the tail of xdist's
# ``sessionfinish`` wrapper. A run that stalls anywhere in there has already
# written its data and has not yet printed a summary, which is precisely the
# shape this guard exists for, and a watchdog armed at the end of
# ``sessionfinish`` is on the far side of it.
#
# Armed from the end of the test loop instead, the budget covers every
# teardown after the last test: the rest of the loop's wrappers, every
# ``sessionfinish``, ``pytest_unconfigure``, ``atexit`` and interpreter
# finalisation. ``pytest_sessionfinish`` then RE-ARMS, so a shutdown that is
# honestly slow is measured from the point the reporting plugins have finished
# rather than from the last test.

SHUTDOWN_WATCHDOG_ENV = "SPACR_PYTEST_SHUTDOWN_WATCHDOG_S"

#: Seconds the interpreter may take to shut down after the last test before
#: every thread's stack is dumped and the process is killed. Generous, so an
#: honestly slow teardown is never mistaken for a stall; ``0`` turns it off.
SHUTDOWN_WATCHDOG_S = float(os.environ.get(SHUTDOWN_WATCHDOG_ENV, "300"))

REPORT_WATCHDOG_ENV = "SPACR_PYTEST_REPORT_WATCHDOG_S"

#: Seconds the phase between the last test and the summary line may take.
#: Longer than the interpreter's own shutdown budget because that phase holds
#: the one operation here that is legitimately slow -- combining and reporting
#: a distributed run's coverage, which is minutes of real work on a project
#: this size. Killing that would break the runs this guard exists to protect,
#: so it is bounded rather than trusted. Turning the shutdown watchdog off
#: turns this off with it, since the default is derived from it.
REPORT_WATCHDOG_S = float(
    os.environ.get(REPORT_WATCHDOG_ENV, SHUTDOWN_WATCHDOG_S * 4))


def teardown_budget(config):
    """How long this process may take between its last test and its summary.

    A distributed WORKER gets the short budget, because none of the slow work
    the long one exists for happens in one: a collocated worker saves its
    coverage data and stops, and the combining and reporting are the
    controller's job. A worker that goes quiet for the shutdown budget is
    stuck, and it is the process most worth shooting quickly -- the whole run
    waits on it, and until it says something there is nothing to read.
    """
    return (SHUTDOWN_WATCHDOG_S if hasattr(config, "workerinput")
            else REPORT_WATCHDOG_S)


def threads_that_outlive_the_session():
    """Every non-daemon thread that will hold the interpreter open at exit.

    Daemon threads are deliberately not reported: they cannot delay
    finalisation, and this suite ends with a handful of them on every run, so
    listing them would bury the one thread that matters.
    """
    return [thread for thread in _threading.enumerate()
            if thread is not _threading.main_thread()
            and thread.is_alive() and not thread.daemon]


def _threads_that_outlive_the_session_report(threads):
    """What to print about threads that will delay the interpreter's exit."""
    listed = "\n".join(
        f"    {thread.name!r} -> {getattr(thread, '_target', None)!r}"
        for thread in threads)
    return (
        f"{len(threads)} non-daemon thread(s) are still running now the "
        f"session is over:\n{listed}\n"
        "Python joins each of them before it finalises, so the run cannot "
        "report until they end. Whatever started one owes it a stop.")


#: How many retained widgets are worth mentioning. A Qt run ALWAYS ends with a
#: live QApplication -- pytest-qt's is session-scoped -- so saying so every
#: time would train the reader to skip the one report that matters. A healthy
#: run of this suite ends in the tens, and its measured worst accumulation was
#: five figures, so the number below separates "normal" from "the tree is the
#: reason this process is still working".
RETAINED_WIDGETS_WORTH_SAYING = 5000


def qt_things_that_outlive_the_session():
    """Everything Qt owns that can keep a finished run from exiting.

    ``threading.enumerate`` cannot see any of it, which is why a wedged run
    used to be reported as "no non-daemon threads" and nothing else. A
    ``QThread`` that has executed Python appears there as a DAEMON ``Dummy-N``
    and is filtered out with the harmless ones; a ``QApplication`` is not a
    thread at all. Both decide how a finished run ends anyway -- Qt waits on a
    running QThread while the application is torn down, and teardown costs
    what the retained tree is big -- so each is named with the number that
    makes it actionable rather than left to be guessed at.

    Read-only by construction. Nothing is stopped, closed or deleted: the one
    fixture in this suite's history that reached across live widgets during
    teardown crashed the run three different ways, and a diagnostic that can
    do that is worse than no diagnostic. Every probe is guarded, so a run
    without PySide6, or one whose Qt state is already half gone, reports what
    it can and stays quiet about the rest.
    """
    said = []

    app = None
    try:
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
    except Exception:                                            # noqa: BLE001
        app = None
    if app is not None:
        try:
            widgets = len(app.allWidgets())
            windows = len(app.topLevelWidgets())
        except Exception:                                        # noqa: BLE001
            widgets = windows = -1
        if widgets >= RETAINED_WIDGETS_WORTH_SAYING:
            said.append(
                f"    a QApplication is still alive holding {widgets} "
                f"widget(s) and {windows} top-level window(s); destroying "
                f"that tree one object at a time is the last thing the "
                f"process does")

    try:
        from spacr.qt import bridge
        handles = bridge.registry().active()
        parked = len(bridge._PARKED_THREADS)
    except Exception:                                            # noqa: BLE001
        handles, parked = [], 0
    for handle in handles:
        said.append(f"    a registered job is still running: "
                    f"{getattr(handle, 'app_key', 'job')!r}")
    if parked:
        said.append(f"    {parked} QThread(s) are parked -- they outlived the "
                    f"widget that owned them and were never seen to stop")

    try:
        from spacr.qt import job_runner
        runners = [runner for runner in list(job_runner._LIVE_RUNNERS)
                   if runner.is_busy()]
    except Exception:                                            # noqa: BLE001
        runners = []
    if runners:
        said.append(f"    {len(runners)} JobRunner(s) are still busy; each "
                    f"owns a QThread that shutdown() was never called on")

    try:
        import multiprocessing
        children = multiprocessing.active_children()
    except Exception:                                            # noqa: BLE001
        children = []
    for child in children:
        said.append(f"    a child process is still alive: {child.name!r} "
                    f"(pid {child.pid}); Python joins it at exit")

    return said


def _qt_things_report(said):
    """What to print about the Qt state that will delay the process's exit."""
    listed = "\n".join(said)
    return (
        f"{len(said)} thing(s) other than a Python thread can hold this "
        f"process open now the session is over:\n{listed}\n"
        "None of it is joined by Python, so none of it is named by the thread "
        "report above; each is still a reason a finished run burns a core "
        "instead of printing a summary.")


def arm_shutdown_watchdog(seconds):
    """Dump every thread's stack and kill the process if shutdown stalls.

    Returns whether the watchdog was armed. ``sys.__stderr__`` rather than
    ``sys.stderr`` because faulthandler writes through a file DESCRIPTOR and
    the replacement a distributed run installs has none.
    """
    if seconds <= 0:
        return False
    stream = sys.__stderr__ or sys.stderr
    try:
        _faulthandler.dump_traceback_later(seconds, exit=True, file=stream)
    except (AttributeError, ValueError, OSError):
        return False
    return True


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_runtestloop(session):
    """Start the shutdown budget the moment the last test is over.

    Innermost of the loop's wrappers on purpose, so the code after the
    ``yield`` runs before any other plugin's teardown does -- including the
    coverage write, which is the last thing a wedged run is known to have
    finished. Nothing is reported here: at this point the reporting plugins
    have not run and the threads that matter may still be retiring normally.
    Reporting is ``pytest_sessionfinish``'s job, and it re-arms the watchdog
    when it is done.
    """
    yield
    arm_shutdown_watchdog(teardown_budget(session.config))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Name what would hold the run open, then bound how long it may.

    Last of its kind on purpose: the threads worth reporting are the ones
    still alive after every other plugin has had its teardown, and the
    watchdog re-armed here must cover whatever happens after this.
    """
    config = session.config
    # A distributed worker has a terminal reporter that prints nowhere the
    # user will see, and a wedged WORKER is the case this exists for, so its
    # own stderr -- which the controller relays -- is the only channel that
    # reaches anybody.
    distributed = hasattr(config, "workerinput")
    reporter = (None if distributed
                else config.pluginmanager.get_plugin("terminalreporter"))

    def _say(message):
        if reporter is not None:
            reporter.write_line(message, yellow=True)
        else:
            print(message, file=sys.stderr, flush=True)

    lingering = threads_that_outlive_the_session()
    if lingering:
        _say(_threads_that_outlive_the_session_report(lingering))
    held = qt_things_that_outlive_the_session()
    if held:
        _say(_qt_things_report(held))
    if not arm_shutdown_watchdog(SHUTDOWN_WATCHDOG_S) \
            and SHUTDOWN_WATCHDOG_S > 0:
        # Said out loud rather than swallowed: a watchdog nobody armed is
        # exactly as useful as no watchdog, and the run it was meant to bound
        # would otherwise stop with no explanation at all.
        _say(f"the shutdown watchdog could not be armed; set "
             f"{SHUTDOWN_WATCHDOG_ENV}=0 to stop trying")


# ---------------------------------------------------------------------------
# One collection node per directory
# ---------------------------------------------------------------------------
#
# A conftest's fixtures are scoped to the DIRECTORY the conftest sits in.
# Every supported pytest 8.x release records that scope as
# ``FixtureDef.baseid`` and matches it against ancestor nodeids. pytest 8.0
# passes a nodeid string into ``getfixturedefs``; pytest 8.4 passes the Node,
# but the visibility rule on either side of that private-API change is the
# same stable nodeid ancestry.
#
# ``Session.collect`` re-collects a directory WITHOUT de-duplicating it when a
# bare FILE path is named on the command line ("for backward compat, files
# given directly multiple times on the command line should not be
# deduplicated"). Re-collecting a directory builds fresh child nodes, so a
# second Directory node appears for a directory that was already collected --
# and the conftest is not parsed a second time, because that parse is deferred
# to the FIRST Directory collection and consumed there.
#
# Duplicate Directory objects therefore do not by themselves hide fixtures on
# supported pytest. They are still canonicalised because collection hooks and
# plugins may keep node-local state, and one path answering with two nodes is
# an unstable tree whose behaviour can otherwise depend on argument order.
#
#     pytest tests/qt/test_a.py tests/test_b.py tests/qt/test_c.py
#
# collects tests/qt, then re-collects tests for the bare middle file, then
# reaches test_c through the SECOND tests/qt node. The invariant below folds
# those duplicate directory nodes together while deliberately leaving file
# duplication alone.
#
# Making a directory answer with ONE node for the whole session removes the
# hazard at its source: the conftest parse and the tests underneath it then
# refer to the same object no matter how many times collection walks the
# directory. Files are deliberately left alone, so naming a file twice still
# runs it twice.

_DIRECTORY_NODES_ATTR = "_spacr_directory_nodes"


def canonical_directory_children(registry, children):
    """Return ``children`` with every directory replaced by its first node.

    ``registry`` maps a directory path to the node this session already uses
    for it and is filled in as new directories are met. Non-directory children
    are passed through untouched, so a file named twice on the command line
    still collects twice.
    """
    canonical = []
    replaced = False
    for child in children:
        if isinstance(child, pytest.Directory):
            first = registry.setdefault(child.path, child)
            if first is not child:
                child = first
                replaced = True
        canonical.append(child)
    return canonical if replaced else children


class _OneNodePerDirectory:
    """Keep a directory's collection node stable for the whole session."""

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector):
        report = yield
        if isinstance(collector, pytest.Directory) and report.result:
            session = collector.session
            registry = getattr(session, _DIRECTORY_NODES_ATTR, None)
            if registry is None:
                registry = {}
                setattr(session, _DIRECTORY_NODES_ATTR, registry)
            registry.setdefault(collector.path, collector)
            report.result = canonical_directory_children(
                registry, report.result)
        return report


def _directory_fixture_nodeid(fixturedef):
    """Return the directory nodeid for a fixture defined by a conftest.

    Supported pytest 8.x records the stable, path-like ``baseid``. Accept a
    node-bearing representation defensively as well, but do not mistake a
    fixture defined in a test module for a directory fixture: only functions
    whose source really is a ``conftest.py`` qualify through ``baseid``.
    """
    node = getattr(fixturedef, "node", None)
    if isinstance(node, pytest.Directory):
        return node.nodeid

    baseid = getattr(fixturedef, "baseid", None)
    function = getattr(fixturedef, "func", None)
    code = getattr(function, "__code__", None)
    source = getattr(code, "co_filename", "")
    if baseid and Path(source).name == "conftest.py":
        return baseid
    return None


def directory_fixture_expectations(fixture_manager):
    """Map a directory's nodeid to the fixture names its conftest defines.

    Read back off the fixtures pytest actually registered rather than by
    importing conftests, so a fixture added tomorrow is covered without this
    being edited.

    A pytest that no longer keeps its registry where this reads it answers
    with nothing, so an upgrade cannot stop the suite collecting. It stops
    being SILENT one line down: the test that pins what this finds in a live
    session fails, which is the right place for "the check needs updating" to
    show up.
    """
    registry = getattr(fixture_manager, "_arg2fixturedefs", None)
    if not registry:
        return {}
    expectations = {}
    for argname, fixturedefs in registry.items():
        for fixturedef in fixturedefs:
            nodeid = _directory_fixture_nodeid(fixturedef)
            if nodeid is not None:
                expectations.setdefault(nodeid, set()).add(argname)
    return expectations


def lost_directory_conftest_fixtures(fixture_manager, items, expectations):
    """Return the conftest fixtures their own tests can no longer request.

    One entry per ``(directory nodeid, fixture name, witness test)``. Empty is
    the healthy answer: a fixture defined in ``tests/qt/conftest.py`` must be
    resolvable from every test collected under ``tests/qt``.

    Checked once per distinct chain of collection nodes rather than once per
    test, because every test sharing a chain shares its fixture visibility.
    """
    if not expectations:
        return []
    lost = []
    checked = set()
    for item in items:
        chain = item.listchain()
        signature = tuple(id(node) for node in chain[:-1])
        if signature in checked:
            continue
        checked.add(signature)
        # pytest 8.4 changed this private API's second argument from a nodeid
        # string to the requesting Node. Support the whole declared pytest
        # 8.x range without catching lookup failures that should remain
        # visible to the suite.
        requester = (item if pytest.version_tuple[:2] >= (8, 4)
                     else item.nodeid)
        for node in chain:
            for argname in sorted(expectations.get(node.nodeid, ())):
                if not fixture_manager.getfixturedefs(argname, requester):
                    lost.append((node.nodeid, argname, item.nodeid))
    return lost


def directory_conftest_parse_nodes(fixture_manager, items=None):
    """Map each conftest directory nodeid to its collection-tree node.

    A fixture definition may expose its collection node directly; supported
    pytest 8.x instead exposes the stable ``FixtureDef.baseid``. In that case,
    recover a representative node with the same nodeid from collected item
    chains. This preserves the duplicate-node diagnosis without claiming that
    fixture visibility itself depends on node identity.

    A pytest that keeps its registry somewhere else answers with nothing, for
    the same reason ``directory_fixture_expectations`` does.
    """
    registry = getattr(fixture_manager, "_arg2fixturedefs", None)
    if not registry:
        return {}
    parse_nodes = {}
    nodeids = set()
    for fixturedefs in registry.values():
        for fixturedef in fixturedefs:
            node = getattr(fixturedef, "node", None)
            if isinstance(node, pytest.Directory):
                parse_nodes.setdefault(node.nodeid, node)
            nodeid = _directory_fixture_nodeid(fixturedef)
            if nodeid is not None:
                nodeids.add(nodeid)
    if items is None:
        session = getattr(fixture_manager, "session", None)
        items = getattr(session, "items", ())
    for item in items or ():
        for node in item.listchain():
            if isinstance(node, pytest.Directory) and node.nodeid in nodeids:
                parse_nodes.setdefault(node.nodeid, node)
    return parse_nodes


def directories_collected_twice(items, parse_nodes=None):
    """Return the directory nodeids this run holds more than one node for.

    Two distinct node objects carrying one nodeid in the collected tree is
    the direct evidence. ``parse_nodes`` adds the other half: a directory
    node distinct from the representative conftest-directory node was
    collected twice even when the tests hanging off the first node are all
    gone -- a ``-k`` selection or a shard can remove that half of the evidence.

    ``None`` means there was nothing to look at: no collection node in the
    whole list was a directory, so no duplicate was ruled either in or out.
    An empty set is the opposite answer -- directories were examined and none
    of them was collected twice.
    """
    seen = {}
    twice = set()
    for item in items:
        for node in item.listchain():
            if not isinstance(node, pytest.Directory):
                continue
            first = seen.setdefault(node.nodeid, node)
            if first is not node:
                twice.add(node.nodeid)
            parsed = (parse_nodes or {}).get(node.nodeid)
            if parsed is not None and parsed is not node:
                twice.add(node.nodeid)
    if not seen:
        return None
    return twice


#: Whether the running pytest hides a directory conftest's fixtures when
#: that directory is collected twice.
#:
#: THE TWO MODELS DIFFER AND THE MESSAGE HAS TO SAY WHICH ONE IT IS IN.
#: pytest 8.x matches ``FixtureDef.baseid`` against ancestor nodeids, so the
#: duplicate node is a violated invariant that does NOT by itself hide
#: anything. pytest 9 resolves against the collected node, so the duplicate
#: IS the disappearance. Telling a reader the wrong one sends them looking
#: for a second fault that is not there -- or past the only one that is.
DUPLICATE_NODE_HIDES_FIXTURES = pytest.version_tuple[0] >= 9

_ORDERING_MODEL = (
    "On this pytest ({version}) a duplicated directory node IS the cause: "
    "fixtures are resolved against the collected node, so the copy that "
    "carries them is not the copy the tests hang under. Repairing the "
    "duplicate restores them."
    if DUPLICATE_NODE_HIDES_FIXTURES else
    "On this pytest ({version}) fixtures match by stable baseid, so the "
    "duplicate node alone does not explain their disappearance; it is "
    "nevertheless a second violated invariant that must be repaired or "
    "ruled out before diagnosing plugin state."
).format(version=pytest.__version__)

_ORDERING_CAUSE = (
    "This run has a COLLECTION ORDERING fault alongside the missing "
    "fixture: a directory whose conftest defines fixtures was collected "
    f"twice. {_ORDERING_MODEL} The duplicate is triggered "
    "by interleaving files from different directories, e.g. "
    "'pytest tests/qt/test_a.py tests/test_b.py tests/qt/test_c.py'.\n"
    "\n"
    "tests/conftest.py keeps one collection node per directory to "
    "prevent this; if you are reading this message, that guard no longer "
    "covers the case at hand. Run the directories as separate "
    "invocations until it does -- the alternative is a run whose summary "
    "line reads as a pass.")

_EVICTED_CAUSE = (
    "The conftest was EVICTED, not accompanied by a duplicate directory. "
    "Each directory listed above was collected exactly ONCE in this run, so "
    "the collection-tree invariant has been ruled out. The conftest was "
    "imported and its fixtures were registered, and then they stopped being "
    "reachable: something may have cleared it from sys.modules, reloaded it, "
    "or unregistered the plugin. Look at what the files in this run do to "
    "module state, and run them one file at a time to find which.")


def _conftest_fixtures_went_missing(lost, collected_twice=None):
    """The message a lost conftest gets, instead of a missing-fixture error.

    ``collected_twice`` is the set of directory nodeids the run holds more
    than one collection node for, and it picks which CAUSE is named. A
    duplicated directory adds the ordering diagnosis; a directory collected
    once does not. ``None`` means nobody looked, and the message keeps the
    conservative ordering wording used for that evidence-free case.
    """
    listed = "\n".join(
        f"    {argname!r} comes from the conftest in {directory}, and "
        f"{witness} cannot request it"
        for directory, argname, witness in lost[:10])
    more = f"\n    ... and {len(lost) - 10} more" if len(lost) > 10 else ""
    header = (f"{len(lost)} conftest fixture(s) are not visible to the tests "
              f"they belong to:\n{listed}{more}\n")

    directories = sorted({directory for directory, _, _ in lost})
    if collected_twice is None:
        out_ordered, evicted = directories, []
    else:
        out_ordered = [d for d in directories if d in collected_twice]
        evicted = [d for d in directories if d not in collected_twice]

    causes = []
    if out_ordered and collected_twice is None:
        causes.append(_ORDERING_CAUSE)
    elif out_ordered:
        causes.append(f"Collected twice: {', '.join(out_ordered)}.\n"
                      f"{_ORDERING_CAUSE}")
    if evicted:
        causes.append(f"Collected once: {', '.join(evicted)}.\n"
                      f"{_EVICTED_CAUSE}")
    return "\n\n".join([header, *causes])


def _check_directory_conftest_fixtures(session, items):
    """Fail the run loudly when a conftest's fixtures went missing.

    The collected tree is asked whether the directory really was collected
    twice, so the message names the cause it can show rather than the cause
    this was first found by.

    That question is only asked once something is already lost, so a healthy
    run pays for the lookup of the fixtures and nothing else.
    """
    fixture_manager = getattr(session, "_fixturemanager", None)
    if fixture_manager is None:
        return
    expectations = directory_fixture_expectations(fixture_manager)
    lost = lost_directory_conftest_fixtures(fixture_manager, items,
                                            expectations)
    if not lost:
        return
    collected_twice = directories_collected_twice(
        items, directory_conftest_parse_nodes(fixture_manager, items))
    raise pytest.UsageError(
        _conftest_fixtures_went_missing(lost, collected_twice))


def pytest_collection_modifyitems(session, config, items):
    """Guard conftest visibility, then apply CI markers and the file shard.

    The visibility check runs against the WHOLE collected list, before the
    shard below throws most of it away, so a lost conftest is reported from
    any shard rather than only from the one that happened to keep a witness.
    """
    _check_directory_conftest_fixtures(session, items)

    for item in items:
        for marker in _automatic_ci_markers(item.path):
            item.add_marker(getattr(pytest.mark, marker))

    count = int(os.environ.get("SPACR_PYTEST_FILE_SHARD_COUNT", "1"))
    index = int(os.environ.get("SPACR_PYTEST_FILE_SHARD_INDEX", "0"))
    if count < 1 or index < 0 or index >= count:
        raise pytest.UsageError(
            "SPACR_PYTEST_FILE_SHARD_INDEX must be within "
            f"[0, {count}); got {index}"
        )
    if count == 1:
        return

    selected = [item for item in items
                if _ci_file_shard(item.path, count) == index]
    deselected = [item for item in items
                  if _ci_file_shard(item.path, count) != index]
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


# Try to import matplotlib once with the Agg backend fixed. If unavailable,
# individual tests that need it will skip themselves.
try:  # pragma: no cover - import side effect only
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Basic infra fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Deterministic numpy Generator."""
    return np.random.default_rng(0)


@pytest.fixture
def tmp_project_dir(tmp_path):
    """A fresh temp directory laid out like a spacr project."""
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    (tmp_path / "measurements").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

def _place_blobs(shape, n_blobs, rng, radius_range=(6, 14), max_intensity=60000):
    """Draw n_blobs bright circular blobs on a dark background."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros(shape, dtype=np.uint16)
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        intensity = int(rng.integers(int(max_intensity * 0.4), max_intensity))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        img[mask] = np.maximum(img[mask], intensity)
    # Add a bit of gaussian background noise so np.min != np.max in flat regions.
    img = img + rng.integers(50, 200, size=shape, dtype=np.uint16)
    return img.astype(np.uint16)


@pytest.fixture
def synth_image_2d(rng):
    """256x256 uint16 grayscale image with ~8 bright blobs on dark background."""
    return _place_blobs((256, 256), n_blobs=8, rng=rng)


@pytest.fixture
def synth_image_3d(rng):
    """3-D image (Z=5, H=128, W=128) uint16."""
    return np.stack([_place_blobs((128, 128), n_blobs=6, rng=rng) for _ in range(5)])


@pytest.fixture
def synth_image_stack(rng):
    """4-D (T=3, C=2, H=128, W=128) uint16 timelapse-ish stack."""
    return np.stack(
        [
            np.stack([_place_blobs((128, 128), n_blobs=5, rng=rng) for _ in range(2)])
            for _ in range(3)
        ]
    )


# ---------------------------------------------------------------------------
# Synthetic label-mask fixtures
# ---------------------------------------------------------------------------

def _labeled_blobs(shape, n_blobs, rng, radius_range=(8, 16)):
    """Return an int32 label image where each blob has a unique id starting at 1."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    lbl = np.zeros(shape, dtype=np.int32)
    next_id = 1
    for _ in range(n_blobs):
        cy = int(rng.integers(20, h - 20))
        cx = int(rng.integers(20, w - 20))
        r = int(rng.integers(*radius_range))
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        # Skip if it would overlap an existing label (keep them disjoint).
        if lbl[mask].max() != 0:
            continue
        lbl[mask] = next_id
        next_id += 1
    return lbl


@pytest.fixture
def synth_mask_2d(rng):
    """256x256 int32 label mask, 6 disjoint blobs (ids 1..N)."""
    return _labeled_blobs((256, 256), n_blobs=6, rng=rng)


@pytest.fixture
def synth_masks_multi(rng):
    """Dict of aligned cell/nucleus/pathogen label masks for one 256x256 field."""
    cell = _labeled_blobs((256, 256), n_blobs=5, rng=rng, radius_range=(20, 30))
    # Nucleus sits inside cells; smaller radius, centered on cell centroids.
    nucleus = np.zeros_like(cell)
    from scipy.ndimage import center_of_mass
    for cell_id in np.unique(cell):
        if cell_id == 0:
            continue
        cy, cx = center_of_mass(cell == cell_id)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= 6 ** 2
        nucleus[m] = cell_id
    # Pathogens: 0-2 small blobs scattered inside random cells.
    pathogen = np.zeros_like(cell)
    next_id = 1
    for _ in range(int(rng.integers(0, 3))):
        cell_ids = [i for i in np.unique(cell) if i != 0]
        if not cell_ids:
            break
        cid = int(rng.choice(cell_ids))
        cy, cx = center_of_mass(cell == cid)
        yy, xx = np.mgrid[: cell.shape[0], : cell.shape[1]]
        offset_y = int(rng.integers(-10, 11))
        offset_x = int(rng.integers(-10, 11))
        m = (yy - (cy + offset_y)) ** 2 + (xx - (cx + offset_x)) ** 2 <= 3 ** 2
        m = m & (cell == cid)  # keep pathogen inside its cell
        if m.any():
            pathogen[m] = next_id
            next_id += 1
    return {"cell": cell, "nucleus": nucleus, "pathogen": pathogen}


# ---------------------------------------------------------------------------
# Synthetic DataFrames & sqlite
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_measurements(rng):
    """A DataFrame with typical spacr measurement columns for 40 objects."""
    n = 40
    plates = ["plate1"] * n
    rows = rng.integers(1, 9, size=n)  # A..H analog
    cols = rng.integers(1, 13, size=n)
    wells = [f"{chr(ord('A')+r-1)}{c:02d}" for r, c in zip(rows, cols)]
    fields = rng.integers(1, 4, size=n)
    prcs = [f"{p}_{w}_{f}" for p, w, f in zip(plates, wells, fields)]
    return pd.DataFrame(
        {
            "plate": plates,
            "row": rows,
            "column": cols,
            "well": wells,
            "field": fields,
            "prc": prcs,
            "object_label": np.arange(1, n + 1),
            "cell_area": rng.uniform(200, 4000, size=n),
            "cell_channel_0_mean_intensity": rng.uniform(500, 40000, size=n),
            "cell_channel_1_mean_intensity": rng.uniform(500, 40000, size=n),
            "nucleus_area": rng.uniform(80, 900, size=n),
            "pathogen_count": rng.integers(0, 5, size=n),
        }
    )


@pytest.fixture
def synth_sqlite_db(tmp_path, synth_measurements):
    """A file-backed sqlite database with a minimal spacr-ish schema."""
    db_path = tmp_path / "measurements.db"
    con = sqlite3.connect(db_path)
    try:
        synth_measurements.to_sql("cell", con, index=False)
        # A dummy annotation table many spacr helpers assume exists.
        anno = pd.DataFrame(
            {
                "prc": synth_measurements["prc"].unique(),
                "annotation": 0,
            }
        )
        anno.to_sql("png_list", con, index=False)
    finally:
        con.close()
    return db_path


# ---------------------------------------------------------------------------
# Cellpose mock contract
# ---------------------------------------------------------------------------
#
# Every ``CellposeModel`` stand-in in this suite used to be spelled
# ``def eval(self, x=None, **kwargs)``. A ``**kwargs`` mock accepts ANY
# argument, so it cannot tell a legal call from an illegal one — which is how
# spaCR's hard-coded ``channel_axis=3`` survived fifteen green tests while
# raising ``IndexError: tuple index out of range`` on every real run.
#
# The cure is not to re-assert a literal value (that just moves the guess into
# the test and drifts as spaCR's shapes change). It is to hand the exact
# ``(x, channel_axis, z_axis, do_3D)`` combination to the same validator the
# real ``CellposeModel.eval`` runs first — ``cellpose.transforms.convert_image``
# — which is pure numpy, CPU-only and offline. If spaCR ever passes an axis
# Cellpose rejects, the mock now fails with the production error.


class _MissingChannelAxis:
    """Sentinel: ``eval()`` was called with no ``channel_axis`` at all."""

    __slots__ = ()

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "<no channel_axis passed>"


#: Default for a mock's ``channel_axis`` parameter. Distinct from ``None``,
#: which is a *legal* value meaning "auto-detect the channel axis".
MISSING_CHANNEL_AXIS = _MissingChannelAxis()


def check_cellpose_eval_call(x, channel_axis=MISSING_CHANNEL_AXIS, *,
                             z_axis=None, do_3D=False, stitch_threshold=0.0,
                             require_channel_axis=True):
    """Reject an ``eval()`` call the real ``CellposeModel.eval`` would reject.

    Mirrors Cellpose 4's own dispatch: a list (or 5-D array) is evaluated one
    element at a time with the same axis kwargs, and every leaf goes through
    ``transforms.convert_image(x, channel_axis=..., z_axis=..., do_3D=...)``.

    :param x: whatever spaCR passed as the image argument.
    :param channel_axis: the value spaCR passed, or
        :data:`MISSING_CHANNEL_AXIS` when it passed none.
    :param require_channel_axis: assert the caller named ``channel_axis``
        explicitly. True for the call sites that do (``spacr.object``,
        ``spacr.pipeline_v2``, ``spacr.spacr_cellpose``) so the value stays
        under contract; False where spaCR deliberately leaves Cellpose to
        auto-detect (``spacr.spacrops``, ``spacr.submodules``).
    :returns: list of converted images, in call order — the mock can size its
        canned masks from these rather than re-deriving the shape.
    :raises AssertionError: when ``channel_axis`` was required and omitted.
    :raises ValueError, IndexError: whatever Cellpose itself would raise.
    """
    if require_channel_axis:
        assert channel_axis is not MISSING_CHANNEL_AXIS, (
            "model.eval() was called without channel_axis. spaCR must pass it "
            "explicitly at this call site so its value stays covered by this "
            "contract; a mock that defaults it silently accepts the "
            "channel_axis=3 that broke every real run."
        )
    axis = None if channel_axis is MISSING_CHANNEL_AXIS else channel_axis

    from cellpose import transforms

    # Cellpose recurses over a list before it converts anything, so a batch is
    # only legal if every element is.
    if isinstance(x, (list, tuple)):
        images = list(x)
    else:
        arr = np.asarray(x)
        images = list(arr) if arr.squeeze().ndim == 5 else [arr]

    converted = []
    for image in images:
        converted.append(transforms.convert_image(
            np.asarray(image), channel_axis=axis, z_axis=z_axis,
            do_3D=(do_3D or stitch_threshold > 0)))
    return converted


# ---------------------------------------------------------------------------
# Yokogawa microscopy fixtures — CellVoyager (default) and CQ1 filename styles
# ---------------------------------------------------------------------------
#
# CellVoyager filename regex (see spacr/utils.py::_get_regex):
#     {plateID}_{wellID}_T{timeID}F{fieldID}L{laserID}A{AID}Z{sliceID}C{chanID}.tif
#
# CQ1 filename regex:
#     W{wellID}F{fieldID}T{timeID}Z{sliceID}C{chanID}.tif
#     wellID is an integer 1..384 that spacr converts to A01..P24.

def _write_tif(path, arr):
    """Save an image as a real TIFF (tifffile if available, else Pillow)."""
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
        return
    except Exception:
        pass
    from PIL import Image
    Image.fromarray(arr).save(str(path))


def _make_field(rng, shape=(128, 128)):
    """Small deterministic uint16 field image with a few blobs."""
    return _place_blobs(shape, n_blobs=int(rng.integers(2, 6)), rng=rng)


@pytest.fixture
def yokogawa_cellvoyager_dir(tmp_path, rng):
    """
    A temp directory of TIFFs following the Yokogawa CellVoyager naming.

    Layout (deterministic):
      * 1 plate  ('plate1')
      * 2 wells  ('A01', 'A02')
      * 2 fields (F001, F002)
      * 2 channels (C01, C02)
      * 1 z-slice, 1 timepoint, 1 laser, 1 action
    -> 8 TIFFs total.

    Yields the directory path plus a manifest so tests can assert what was
    written.
    """
    src = tmp_path / "cellvoyager"
    src.mkdir()
    manifest = []
    for well in ("A01", "A02"):
        for field in ("001", "002"):
            for chan in ("01", "02"):
                fname = f"plate1_{well}_T0001F{field}L01A01Z01C{chan}.tif"
                img = _make_field(rng)
                _write_tif(src / fname, img)
                manifest.append(
                    {"plate": "plate1", "well": well, "field": field,
                     "channel": chan, "path": str(src / fname)}
                )
    return {"src": src, "manifest": manifest,
            "metadata_type": "cellvoyager",
            "n_wells": 2, "n_fields": 2, "n_channels": 2}


@pytest.fixture
def yokogawa_cq1_dir(tmp_path, rng):
    """
    A temp directory of TIFFs following the Yokogawa CQ1 naming.

    Uses integer well IDs (1..384) that spacr converts to A01..P24 via
    utils._convert_cq1_well_id. Here we use W1 (=A01) and W25 (=B01).
    """
    src = tmp_path / "cq1"
    src.mkdir()
    manifest = []
    for well_id, expected_well in ((1, "A01"), (25, "B01")):
        for field in ("001", "002"):
            for chan in ("1", "2"):
                fname = f"W{well_id}F{field}T0001Z01C{chan}.tif"
                img = _make_field(rng)
                _write_tif(src / fname, img)
                manifest.append(
                    {"well_id": well_id, "well": expected_well, "field": field,
                     "channel": chan, "path": str(src / fname)}
                )
    return {"src": src, "manifest": manifest,
            "metadata_type": "cq1",
            "n_wells": 2, "n_fields": 2, "n_channels": 2}


# ---------------------------------------------------------------------------
# Illumina sequencing fixtures — 3-barcode reads matching spacr's default
# regex.
# ---------------------------------------------------------------------------
#
# spacr's default barcode regex is:
#   ^(?P<column>.{8})TGCTG.*TAAAC(?P<grna>.{20,21})AACTT.*AGAAG(?P<row>.{8}).*
#
# so each read starts with an 8bp column barcode, then a constant TGCTG
# spacer, then some fill, then TAAAC + a 20-21bp gRNA, then AACTT..AGAAG,
# then an 8bp row barcode, then anything.
#
# The barcode reference is emitted as FASTA (one entry per barcode) AND
# CSV (with 'sequence' / 'name' columns), since spacr.sequencing itself
# consumes the CSV form via map_sequences_to_names().

def _rand_bases(rng, n):
    return "".join(rng.choice(list("ACGT"), size=n))


def _fastq_record(read_id, seq, qual_char="I"):
    qual = qual_char * len(seq)
    return f"@{read_id}\n{seq}\n+\n{qual}\n"


@pytest.fixture
def synth_barcodes(tmp_path, rng):
    """
    Build 3 barcode reference tables (columns, rows, gRNAs), each in BOTH
    FASTA and CSV form, and hand back the file paths + the raw sequences
    for use by test-read generators.

    Sizes: 4 columns, 4 rows, 6 gRNAs (small so the test suite stays fast).
    """
    N_COLUMNS = 4
    N_ROWS = 4
    N_GRNAS = 6

    # Deterministic barcode sequences.
    columns = {f"col{i+1}": _rand_bases(rng, 8) for i in range(N_COLUMNS)}
    rows = {f"row{i+1}": _rand_bases(rng, 8) for i in range(N_ROWS)}
    grnas = {f"grna{i+1}": _rand_bases(rng, 20) for i in range(N_GRNAS)}

    out_dir = tmp_path / "barcodes"
    out_dir.mkdir()

    def _write_fasta(path, name_to_seq):
        with open(path, "w") as f:
            for name, seq in name_to_seq.items():
                f.write(f">{name}\n{seq}\n")

    def _write_csv(path, name_to_seq):
        # spacr.sequencing.map_sequences_to_names expects 'sequence','name' columns.
        with open(path, "w") as f:
            f.write("sequence,name\n")
            for name, seq in name_to_seq.items():
                f.write(f"{seq},{name}\n")

    paths = {}
    for label, table in (("column", columns), ("row", rows), ("grna", grnas)):
        fasta = out_dir / f"{label}_barcodes.fasta"
        csv = out_dir / f"{label}_barcodes.csv"
        _write_fasta(fasta, table)
        _write_csv(csv, table)
        paths[f"{label}_fasta"] = str(fasta)
        paths[f"{label}_csv"] = str(csv)

    return {"columns": columns, "rows": rows, "grnas": grnas,
            "paths": paths, "dir": out_dir}


@pytest.fixture
def synth_illumina_reads(tmp_path, rng, synth_barcodes):
    """
    Build a paired-end Illumina FASTQ.gz pair whose R1 reads carry one
    column + one gRNA + one row barcode each, in the layout the default
    spacr regex expects. R2 mirrors R1 in this fixture.

    Yields:
      dict with 'r1_path', 'r2_path' (both .fastq.gz), 'n_reads',
      and 'truth' — a list of dicts telling the test which barcodes were
      injected into each read so tests can validate detection.
    """
    import gzip

    N_READS = 40
    truth = []
    col_seqs = list(synth_barcodes["columns"].items())
    row_seqs = list(synth_barcodes["rows"].items())
    grna_seqs = list(synth_barcodes["grnas"].items())

    lines_r1 = []
    lines_r2 = []
    for i in range(N_READS):
        col_name, col_seq = col_seqs[int(rng.integers(0, len(col_seqs)))]
        row_name, row_seq = row_seqs[int(rng.integers(0, len(row_seqs)))]
        grna_name, grna_seq = grna_seqs[int(rng.integers(0, len(grna_seqs)))]

        # Build a read exactly matching:
        #   {col:8}TGCTG{fill}TAAAC{grna:20-21}AACTT{fill}AGAAG{row:8}{trailing}
        # spacr's regex uses .* for the two fill regions.
        # THE ANCHOR'S OWN MIDDLE, not six random bases.
        #
        # `target_sequence` defaults to 'TGCTGTTTCCAGCATAGCTCTTAAAC', and
        # spaCR scans every read for an EXACT match of it -- reads without
        # one are skipped entirely. A random fill here put six arbitrary
        # bases where that constant belongs, so no read carried the anchor,
        # the module's own end-to-end test mapped 0 of 40 reads, and it
        # passed: it asserted only that the call returned.
        #
        # 'TGCTG' + this + 'TAAAC' is exactly the default anchor.
        fill1 = "TTTCCAGCATAGCTCT"
        fill2 = _rand_bases(rng, 6)
        trailing = _rand_bases(rng, 8)

        seq = (
            col_seq +
            "TGCTG" + fill1 +
            "TAAAC" + grna_seq +
            "AACTT" + fill2 +
            "AGAAG" + row_seq + trailing
        )

        # For paired-end Illumina, R2 comes off the opposite strand — the
        # simplest realistic fixture: R2 is the reverse complement of R1.
        # spacr.sequencing.paired_find_sequence_in_chunk_reads applies
        # reverse_complement(R2) before searching, so after that step R2
        # should equal R1 again and the target anchor is findable in both.
        def _rc(s):
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            return "".join(comp[b] for b in reversed(s))

        read_id = f"SIM:1:FCXX:1:1101:{i}:1"
        lines_r1.append(_fastq_record(read_id, seq))
        lines_r2.append(_fastq_record(read_id, _rc(seq)))
        truth.append({
            "read_id": read_id, "seq": seq,
            "column": col_name, "row": row_name, "grna": grna_name,
        })

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    r1 = seq_dir / "sample_R1.fastq.gz"
    r2 = seq_dir / "sample_R2.fastq.gz"
    with gzip.open(r1, "wt") as fh:
        fh.write("".join(lines_r1))
    with gzip.open(r2, "wt") as fh:
        fh.write("".join(lines_r2))

    return {
        "r1_path": str(r1),
        "r2_path": str(r2),
        "n_reads": N_READS,
        "truth": truth,
        "src": str(seq_dir),
    }


# ---------------------------------------------------------------------------
# Hugging Face-backed fixtures: real Yokogawa CellVoyager images + spacr's
# canonical settings CSVs.
# ---------------------------------------------------------------------------
#
# These pull a small deterministic slice of two public HF datasets:
#   einarolafsson/toxo_mito       real 4-channel CellVoyager microscopy
#   einarolafsson/spacr_settings  the reference settings CSVs
#
# Tests that use them are marked @pytest.mark.network and probe the endpoint
# automatically. They skip in offline environments and run whenever the
# Hugging Face service is reachable. Fixtures are session-scoped since the
# payload is stable.
#
# Only 4 TIFFs (one plate/well/field, four channels) are pulled — enough
# to exercise the metadata extractor, the settings loader, and one mask
# generation pass, without downloading the full 210-file dataset.
#
# NONE of these fixtures may turn a spaCR failure into a skip. Only the two
# things this machine genuinely might not have — the network and the optional
# packages — are allowed to skip, and each is caught by its own exception type
# so a bug cannot borrow the excuse. ``hf_hub_download`` signals every fetch
# problem it has (HTTP status, DNS, timeout, missing entry, unwritable cache)
# as an ``OSError`` subclass: ``HfHubHTTPError`` -> ``requests.HTTPError`` ->
# ``RequestException`` -> ``OSError``. A ``TypeError`` from calling it wrong,
# or an ``AssertionError`` from spaCR, is not an OSError and now fails loudly.


@pytest.fixture(scope="session")
def hf_toxo_mito_field(tmp_path_factory):
    """Download one field (4 channels) from einarolafsson/toxo_mito.

    Returns a dict with:
      * src: path to the local directory containing the TIFFs
      * files: list of absolute file paths (4 TIFFs, one per channel)
      * plate/well/field: the metadata slice picked
    """
    from tests.resource_capabilities import endpoint_available
    if not endpoint_available():
        pytest.skip("network / huggingface.co unreachable")
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download

    dst = tmp_path_factory.mktemp("hf_toxo_mito")
    target_files = [
        "plate1/plate1_E01_T0001F001L01A01Z01C02.tif",
        "plate1/plate1_E01_T0001F001L01A02Z01C01.tif",
        "plate1/plate1_E01_T0001F001L01A02Z01C04.tif",
        "plate1/plate1_E01_T0001F001L01A03Z01C03.tif",
    ]
    local_paths = []
    for rel in target_files:
        try:
            p = hf_hub_download(
                repo_id="einarolafsson/toxo_mito",
                filename=rel,
                repo_type="dataset",
                local_dir=str(dst),
            )
        except OSError as e:  # pragma: no cover - network path
            pytest.skip(f"HF download failed for {rel}: {e}")
        local_paths.append(p)
    return {
        "src": str(dst / "plate1"),
        "files": local_paths,
        "plate": "plate1",
        "well": "E01",
        "field": "001",
    }


@pytest.fixture(scope="session")
def spacr_pipeline_run(tmp_path_factory, hf_toxo_mito_multi_fields):
    """
    Run the full `preprocess_generate_masks` pipeline ONCE per test session
    on a copy of the HF toxo_mito data, then hand the working directory
    (with all generated folders + masks + measurements) to every
    downstream test that wants to inspect it.

    Marked as skip if GPU / cellpose / HF is unavailable — see the tests
    that use this fixture; they carry the @pytest.mark.slow +
    @pytest.mark.gpu + @pytest.mark.network markers so the whole thing
    only runs when explicitly opted in.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available for full pipeline test")
    pytest.importorskip("cellpose")

    import shutil
    work = tmp_path_factory.mktemp("spacr_pipeline")
    # Copy the HF TIFFs into a flat working dir (the pipeline expects a
    # flat directory of raw images).
    for src_path in hf_toxo_mito_multi_fields["files"]:
        shutil.copy(src_path, work / os.path.basename(src_path))

    from spacr.core import preprocess_generate_masks
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks(None)
    settings.update({
        "src": str(work),
        "metadata_type": "cellvoyager",
        "batch_size": 100,
        # channels are 0-indexed into the merged stack.
        "channels": [0, 1, 2, 3],
        # toxo_mito: C01=nucleus, C02=cell, C03=pathogen (0-indexed).
        "nucleus_channel": 0, "cell_channel": 1, "pathogen_channel": 2,
        "organelle_channel": None,
        "plot": False, "verbose": False, "test_mode": False, "timelapse": False,
        "n_jobs": 1, "adjust_cells": False, "delete_intermediate": False,
        # The e2e tests inspect the intermediate stack/ and masks/ folders.
        # Those are deleted once merged/ is built unless keep_intermediate is
        # set (default False since 1.4.6), so opt in here.
        "keep_intermediate": True, "keep_original_images": True,
    })

    # No try/skip. Detecting that the mask pipeline stopped working on real
    # microscopy data is the entire purpose of this fixture; catching its
    # failure and reporting a skip made "the pipeline is broken" and "this box
    # has no GPU" look identical, and only the second one is an excuse. The
    # GPU / cellpose / network preconditions are already checked above.
    preprocess_generate_masks(settings)

    return {"src": str(work), "settings": settings,
            "n_input_fields": len(hf_toxo_mito_multi_fields["fields"])}


@pytest.fixture(scope="session")
def spacr_measure_run(spacr_pipeline_run):
    """Run measure_crop on the shared pipeline output ONCE per test session
    and hand back the measurements DB path + the src directory.

    Session-scoped so multiple test modules (pipeline_e2e,
    pipeline_training_analysis, ...) can inspect the same DB / PNG
    outputs without re-running the (~2 minute) mask + measure work per
    module.
    """
    from spacr.measure import measure_crop
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings(None)
    settings.update({
        "src": spacr_pipeline_run["src"],
        # After preprocess_generate_masks the merged stack is
        # [C0=nucleus_intensity, C1=cell_intensity, C2=pathogen_intensity,
        #  C3=organelle_intensity(unused), C4=cell_mask, C5=nucleus_mask,
        #  C6=pathogen_mask].
        "channels": [0, 1, 2, 3],
        "cell_chann_dim": 1, "nucleus_chann_dim": 0, "pathogen_chann_dim": 2,
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "cytoplasm": True,
        "n_jobs": 1, "batch_size": 8, "verbose": False,
        # save_png=True so downstream tests can chain into generate_dataset
        # and apply_model on the resulting per-object crops.
        "plot": False, "save_png": True, "save_arrays": False,
    })
    # No try/skip: same reasoning as spacr_pipeline_run. measure_crop failing
    # on the output spaCR itself just produced IS the regression this fixture
    # exists to surface.
    measure_crop(settings)
    return {
        "src": spacr_pipeline_run["src"],
        "db_path": os.path.join(
            spacr_pipeline_run["src"], "measurements", "measurements.db"
        ),
    }


@pytest.fixture(scope="session")
def hf_toxo_mito_multi_fields(tmp_path_factory):
    """Download several fields (each 4 channels) from einarolafsson/toxo_mito
    into a NEW temp directory — the full mask pipeline needs enough FOVs
    to form a valid batch and populate the channel folders.

    Returns:
      dict with src pointing at the plate directory that contains the flat
      list of Yokogawa CellVoyager TIFFs (as the pipeline expects) plus the
      manifest of what was downloaded.
    """
    from tests.resource_capabilities import (
        cuda_available,
        endpoint_available,
        package_available,
    )
    if not cuda_available():
        pytest.skip("no CUDA available for full pipeline test")
    if not package_available("cellpose"):
        pytest.skip("cellpose unavailable")
    if not endpoint_available():
        pytest.skip("network / huggingface.co unreachable")
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download

    dst = tmp_path_factory.mktemp("hf_toxo_mito_multi")
    # 3 fields × 4 channels = 12 TIFFs — enough to satisfy batch checks
    # while staying under ~10 MB of download.
    fields = ("001", "009", "010")
    channel_layout = (
        ("A01Z01C02",),
        ("A02Z01C01",),
        ("A02Z01C04",),
        ("A03Z01C03",),
    )
    local_paths = []
    for f in fields:
        for (chan,) in channel_layout:
            rel = f"plate1/plate1_E01_T0001F{f}L01{chan}.tif"
            try:
                p = hf_hub_download(
                    repo_id="einarolafsson/toxo_mito",
                    filename=rel,
                    repo_type="dataset",
                    local_dir=str(dst),
                )
            except OSError as e:  # pragma: no cover
                pytest.skip(f"HF download failed for {rel}: {e}")
            local_paths.append(p)
    return {
        "src": str(dst / "plate1"),
        "files": local_paths,
        "plate": "plate1",
        "well": "E01",
        "fields": fields,
    }


@pytest.fixture(scope="session")
def hf_spacr_settings(tmp_path_factory):
    """Download the two reference settings CSVs from einarolafsson/spacr_settings."""
    from tests.resource_capabilities import endpoint_available
    if not endpoint_available():
        pytest.skip("network / huggingface.co unreachable")
    hf_hub_download = pytest.importorskip("huggingface_hub").hf_hub_download

    dst = tmp_path_factory.mktemp("hf_spacr_settings")
    paths = {}
    for name in ("gen_masks_settings.csv", "crop_measure_settings.csv"):
        try:
            p = hf_hub_download(
                repo_id="einarolafsson/spacr_settings",
                filename=name,
                repo_type="dataset",
                local_dir=str(dst),
            )
        except OSError as e:  # pragma: no cover
            pytest.skip(f"HF download failed for {name}: {e}")
        paths[name] = p
    return paths


@pytest.fixture(scope="session")
def _the_real_accelerator():
    """Probe this machine ONCE for the whole session.

    Probing torch is not free, and the answer cannot change while the
    suite runs. Resolving once here is what lets the per-test fixture
    below restore a WARM cache rather than an empty one.
    """
    try:
        from spacr import accelerator
    except Exception:               # accelerator unimportable in this env
        return None
    try:
        return accelerator.resolve()
    except Exception:
        return None


@pytest.fixture(autouse=True)
def _the_accelerator_verdict_does_not_leak_between_tests(
        _the_real_accelerator):
    """Put ``spacr.accelerator._CACHED`` back after every test.

    ``resolve()`` caches the machine's accelerator the first time it is
    asked, which is right in production -- probing torch is not free and
    the answer cannot change mid-run.

    In a test process it is a trap. A test that makes ``torch.cuda`` raise
    to prove the CPU fallback works leaves "this machine has no GPU"
    CACHED, and monkeypatch undoes the torch patch but knows nothing about
    the cache. Every later test in that process then sees a machine with
    no GPU.

    That is exactly how
    tests/qt/test_a_preview_without_torch_still_segments.py failed: the
    second test passed alone and failed after the first, and the failure
    looked like a bug in the preview's device choice rather than a
    neighbouring test's leftovers.

    RESTORES THE REAL VERDICT, NOT WHATEVER WAS THERE BEFORE. Putting
    back the pre-test value would mean putting back ``None`` for the first
    test that runs, and every test after it would re-probe torch -- slow,
    and on a machine with a flaky driver, differently flaky. Restoring the
    session's own answer keeps the cache warm and still lets no fake
    machine escape the test that built it.

    Autouse and unconditional: any test may poison the cache, so every
    test is protected rather than the handful known to need it.
    """
    try:
        from spacr import accelerator
    except Exception:
        yield
        return
    try:
        yield
    finally:
        accelerator._CACHED = _the_real_accelerator
