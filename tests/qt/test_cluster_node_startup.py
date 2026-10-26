"""Starting the GUI on a cluster node: missing X11 libraries and a shared home.

Reported from an interactive node on a Slurm cluster, home directory shared
by every node, no root:

* Qt aborted with "From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed",
  a core dump and nothing to act on;
* "another spaCR window is already running" named a process on another node;
* "did not shut down cleanly 5 times running" counted those aborts, and other
  nodes' sessions, against this one.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("PySide6")

import spacr.qt as qt  # noqa: E402
from spacr import doctor  # noqa: E402
from spacr.qt import crash_recovery as cr  # noqa: E402
from tests.qt.test_multi_instance_lock_and_jobs_dock import (  # noqa: E402
    _hold,
    _release,
)

CURSOR = "libxcb-cursor.so.0"


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A Linux X11 session whose system has none of Qt's X11 libraries."""
    environ = {"DISPLAY": "localhost:10.0", "PATH": os.environ.get("PATH", "")}
    qt_dir = tmp_path / "PySide6" / "Qt" / "lib"
    qt_dir.mkdir(parents=True)
    monkeypatch.setattr(qt, "_qt_library_dirs", lambda: [str(qt_dir)])
    monkeypatch.setattr(qt, "_loader_cache", lambda: frozenset())
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    return environ, tmp_path


def _conda_env(root, *names):
    """An environment folder whose ``lib`` holds ``names``."""
    lib = root / "lib"
    lib.mkdir(parents=True)
    for name in names:
        (lib / name).write_bytes(b"")
    return lib


def _no_exec(*args):
    raise AssertionError(f"relaunched although it must not: {args}")


# ---------------------------------------------------------------------------
# the library check
# ---------------------------------------------------------------------------

def test_a_missing_library_prints_what_to_install_and_exits(node, capsys):
    """A clean message and exit code 1, never Qt's abort."""
    environ, _root = node

    assert qt._check_the_x11_libraries(environ, linux=True, execv=_no_exec) == 1

    err = capsys.readouterr().err
    assert CURSOR in err
    assert "mamba install -c conda-forge xcb-util-cursor" in err
    assert "no root needed" in err
    assert "sudo apt install libxcb-cursor0" in err
    assert "Aborted" not in err and "Traceback" not in err


def test_run_stops_before_qt_when_a_library_is_missing(node, monkeypatch, capsys):
    """`spacr` returns 1 without ever constructing a QApplication."""
    import spacr.qt.app as app_mod

    environ, _root = node
    for key in ("QT_QPA_PLATFORM", "WAYLAND_DISPLAY", "CONDA_PREFIX",
                qt._X11_RELAUNCH_FLAG):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISPLAY", environ["DISPLAY"])
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "execv", _no_exec)
    monkeypatch.setattr(app_mod, "launch",
                        lambda argv: pytest.fail("Qt was started anyway"))

    assert qt.run([]) == 1
    assert CURSOR in capsys.readouterr().err


def test_every_missing_library_is_named_with_its_package(node, capsys):
    """The other xcb libraries clusters lack get the same treatment."""
    environ, _root = node

    missing = qt._missing_x11_libraries(environ)

    assert [row[0] for row in missing] == [row[0] for row in qt._X11_LIBRARIES]
    message = qt._x11_library_message(missing)
    for soname, conda, apt in qt._X11_LIBRARIES:
        assert soname in message and conda in message and apt in message
    mamba = next(line for line in message.splitlines() if "mamba install" in line)
    assert mamba.split().count("libxkbcommon") == 1, "one package is named once"


def test_libraries_the_system_or_qt_provides_are_not_missing(node, tmp_path):
    environ, _root = node
    beside_qt = qt._qt_library_dirs()[0]
    open(os.path.join(beside_qt, CURSOR), "wb").close()
    elsewhere = tmp_path / "opt"
    elsewhere.mkdir()
    (elsewhere / "libxcb-icccm.so.4").write_bytes(b"")
    environ["LD_LIBRARY_PATH"] = f"{elsewhere}:"
    cache = frozenset({"libxcb-image.so.0"})

    names = [row[0] for row in qt._missing_x11_libraries(environ, cache=cache)]

    assert CURSOR not in names
    assert "libxcb-icccm.so.4" not in names
    assert "libxcb-image.so.0" not in names
    assert "libxcb-keysyms.so.1" in names


def test_without_a_loader_cache_the_system_folders_are_searched(
        node, tmp_path, monkeypatch):
    environ, _root = node
    monkeypatch.setattr(qt, "_loader_cache", lambda: None)
    system = tmp_path / "usr-lib"
    system.mkdir()
    for row in qt._X11_LIBRARIES[1:]:
        (system / row[0]).write_bytes(b"")

    missing = qt._missing_x11_libraries(environ, qt_dirs=[],
                                        system_dirs=(str(system),))

    assert missing == [qt._X11_LIBRARIES[0]]


def test_the_check_is_only_for_qts_x11_plugin(node):
    environ, _root = node
    assert qt._check_the_x11_libraries(environ, linux=False, execv=_no_exec) == 0
    for platform in ("offscreen", "wayland", "vnc", "minimal"):
        environ["QT_QPA_PLATFORM"] = platform
        assert qt._check_the_x11_libraries(environ, linux=True,
                                           execv=_no_exec) == 0
    environ["QT_QPA_PLATFORM"] = "xcb;wayland"
    assert qt._qt_will_use_xcb(environ, linux=True) is True
    assert qt._qt_will_use_xcb({"WAYLAND_DISPLAY": "wayland-0"}, linux=True) is False
    assert qt._qt_will_use_xcb({}, linux=True) is False
    assert qt._qt_will_use_xcb({"DISPLAY": ":0", "WAYLAND_DISPLAY": "w"},
                               linux=True) is True


def test_a_failing_check_never_blocks_the_launch(node, monkeypatch):
    environ, _root = node

    def broken(environ=None):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(qt, "_missing_x11_libraries", broken)
    assert qt._check_the_x11_libraries(environ, linux=True, execv=_no_exec) == 0


def test_the_loader_cache_is_read_from_ldconfig(monkeypatch):
    import subprocess

    listing = ("12 libs found in cache `/etc/ld.so.cache'\n"
               "\tlibxcb-cursor.so.0 (libc6,x86-64) => /usr/lib/libxcb-cursor.so.0\n"
               "\tlibc.so.6 (libc6,x86-64) => /lib/libc.so.6\n")
    monkeypatch.setattr(os.path, "isfile", lambda path: path == "/sbin/ldconfig")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, stdout=listing))
    assert qt._loader_cache() == frozenset({"libxcb-cursor.so.0", "libc.so.6"})

    monkeypatch.setattr(os.path, "isfile", lambda path: False)
    assert qt._loader_cache() is None


# ---------------------------------------------------------------------------
# the relaunch with the conda environment's lib folder
# ---------------------------------------------------------------------------

def test_a_library_in_the_conda_env_relaunches_once_with_it_prepended(node):
    environ, root = node
    lib = _conda_env(root / "env", CURSOR)
    environ["CONDA_PREFIX"] = str(root / "env")
    environ["LD_LIBRARY_PATH"] = "/opt/site/lib"
    calls = []

    result = qt._check_the_x11_libraries(
        environ, linux=True, execv=lambda *a: calls.append(a))

    assert result == 0
    assert len(calls) == 1
    program, arguments = calls[0]
    assert program == sys.executable and arguments[0] == sys.executable
    assert environ["LD_LIBRARY_PATH"] == f"{lib}:/opt/site/lib"
    assert environ[qt._X11_RELAUNCH_FLAG] == "1"


def test_the_relaunched_copy_never_relaunches_again(node, capsys):
    """No loop: the copy reports what is still missing and exits."""
    environ, root = node
    lib = _conda_env(root / "env", CURSOR)
    environ["CONDA_PREFIX"] = str(root / "env")
    calls = []

    qt._check_the_x11_libraries(environ, linux=True,
                                execv=lambda *a: calls.append(a))
    assert len(calls) == 1
    assert environ["LD_LIBRARY_PATH"] == str(lib)

    second = qt._check_the_x11_libraries(environ, linux=True,
                                         execv=lambda *a: calls.append(a))

    assert len(calls) == 1
    assert second == 1
    assert environ["LD_LIBRARY_PATH"].count(str(lib)) == 1
    first_line = capsys.readouterr().err.splitlines()[0]
    assert CURSOR not in first_line
    assert "libxcb-icccm.so.4" in first_line


def test_an_env_without_the_library_does_not_relaunch(node, capsys):
    environ, root = node
    _conda_env(root / "env")
    environ["CONDA_PREFIX"] = str(root / "env")

    assert qt._check_the_x11_libraries(environ, linux=True, execv=_no_exec) == 1
    assert "LD_LIBRARY_PATH" not in environ


def test_the_running_conda_python_counts_as_the_environment(node, monkeypatch):
    environ, root = node
    prefix = root / "base"
    _conda_env(prefix, CURSOR)
    (prefix / "conda-meta").mkdir()
    monkeypatch.setattr(sys, "prefix", str(prefix))

    assert qt._conda_library_dirs(environ) == [str(prefix / "lib")]


def test_a_failed_relaunch_falls_back_to_the_message(node, capsys):
    environ, root = node
    _conda_env(root / "env", CURSOR)
    environ["CONDA_PREFIX"] = str(root / "env")

    def refuse(*args):
        raise OSError("exec format error")

    assert qt._check_the_x11_libraries(environ, linux=True, execv=refuse) == 1
    assert CURSOR in capsys.readouterr().err
    assert "LD_LIBRARY_PATH" not in environ
    assert qt._X11_RELAUNCH_FLAG not in environ


# ---------------------------------------------------------------------------
# spacr-doctor
# ---------------------------------------------------------------------------

def test_doctor_names_the_missing_libraries(node, monkeypatch):
    environ, _root = node
    monkeypatch.setattr(qt, "_qt_will_use_xcb", lambda *a, **k: True)
    monkeypatch.setattr(qt, "_missing_x11_libraries",
                        lambda *a, **k: [qt._X11_LIBRARIES[0]])
    monkeypatch.setattr(qt, "_conda_library_dirs", lambda *a, **k: [])

    row = doctor._check_x11_libraries(doctor.Context())

    assert row.status == doctor.WARN
    assert CURSOR in row.message
    assert "mamba install -c conda-forge xcb-util-cursor" in row.fix
    assert "sudo apt install libxcb-cursor0" in row.fix


def test_doctor_passes_when_the_conda_env_holds_them(node, monkeypatch):
    environ, root = node
    lib = _conda_env(root / "env", CURSOR)
    monkeypatch.setattr(qt, "_qt_will_use_xcb", lambda *a, **k: True)
    monkeypatch.setattr(qt, "_missing_x11_libraries",
                        lambda *a, **k: [qt._X11_LIBRARIES[0]])
    monkeypatch.setattr(qt, "_conda_library_dirs", lambda *a, **k: [str(lib)])

    row = doctor._check_x11_libraries(doctor.Context())

    assert row.status == doctor.PASS and "LD_LIBRARY_PATH" in row.message


def test_doctor_adds_no_row_off_x11(monkeypatch):
    monkeypatch.setattr(qt, "_qt_will_use_xcb", lambda *a, **k: False)
    assert doctor._check_x11_libraries(doctor.Context()) is None
    monkeypatch.setattr(qt, "_qt_will_use_xcb", lambda *a, **k: True)
    monkeypatch.setattr(qt, "_missing_x11_libraries", lambda *a, **k: [])
    assert doctor._check_x11_libraries(doctor.Context()).status == doctor.PASS


# ---------------------------------------------------------------------------
# the single-window lock on a shared home
# ---------------------------------------------------------------------------

@pytest.fixture
def locks(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_LOCK_DIR", str(tmp_path / "locks"))
    cr._release_all_locks()
    yield tmp_path
    cr._release_all_locks()


def test_the_instance_lock_is_named_after_this_host():
    assert cr._INSTANCE_LOCK_NAME == f"instance-{cr._HOST_TAG}.lock"
    assert cr._MARKER == f"running-{cr._HOST_TAG}.marker"
    assert cr._COUNTER == f"unclean-exits-{cr._HOST_TAG}"


def test_host_names_are_made_safe_for_a_file_name(monkeypatch):
    import socket

    monkeypatch.setattr(socket, "gethostname", lambda: "gl3001.arc ts/umich")
    assert cr._host_tag() == "gl3001.arc_ts_umich"
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    assert cr._host_tag() == "localhost"

    def broken():
        raise OSError("no name")

    monkeypatch.setattr(socket, "gethostname", broken)
    assert cr._host_tag() == "localhost"
    assert cr._held_on_another_host({"host": "gl3002"}) is False


def test_a_window_on_another_node_does_not_block(locks):
    """Another node's live window holds its own lock, never this node's."""
    other = os.path.join(cr._locks_folder(), "instance-gl3002.lock")
    proc = _hold(other)
    try:
        assert cr._claim_the_instance() == {}
        assert cr._OTHER_INSTANCE == {}
    finally:
        _release(proc)


def test_a_lock_recorded_by_another_host_does_not_block(locks, monkeypatch,
                                                        caplog):
    """Even at this node's own path, a holder on another host is ignored."""
    monkeypatch.setattr(cr, "_try_lock", lambda path: (
        False, {"pid": 1336294, "host": "gl3002.arc-ts.umich.edu",
                "app": "spacr"}))

    with caplog.at_level("INFO", logger="spacr.qt.crash_recovery"):
        assert cr._claim_the_instance() == {}

    assert cr._OTHER_INSTANCE == {}
    assert "already running" not in caplog.text


def test_a_stale_pid_on_this_host_does_not_block(locks):
    _hold(os.path.join(cr._locks_folder(), cr._INSTANCE_LOCK_NAME), mode="die")
    assert cr._claim_the_instance() == {}


def test_a_live_window_on_this_host_still_counts(locks, caplog):
    proc = _hold(os.path.join(cr._locks_folder(), cr._INSTANCE_LOCK_NAME))
    try:
        with caplog.at_level("WARNING", logger="spacr.qt.crash_recovery"):
            holder = cr._claim_the_instance()
        assert holder["pid"] == proc.pid
        assert "already running" in caplog.text
    finally:
        _release(proc)


# ---------------------------------------------------------------------------
# the unclean-shutdown counter
# ---------------------------------------------------------------------------

def test_other_hosts_unclean_exits_are_not_counted_here(tmp_path, locks,
                                                        monkeypatch):
    monkeypatch.setattr(cr, "_folder", lambda: str(tmp_path))
    (tmp_path / "running-gl3002.marker").write_text("77", encoding="utf-8")
    (tmp_path / "unclean-exits-gl3002").write_text("5", encoding="utf-8")
    (tmp_path / "running.marker").write_text("77", encoding="utf-8")
    (tmp_path / "unclean-exits").write_text("5", encoding="utf-8")

    assert cr.note_that_a_launch_began() == 0
    assert cr.should_start_without_the_backdrop() is False

    cr.note_a_clean_shutdown()
    assert (tmp_path / "unclean-exits-gl3002").read_text() == "5"
    assert (tmp_path / "running-gl3002.marker").exists()


def test_this_hosts_unclean_exits_still_count(tmp_path, locks, monkeypatch):
    monkeypatch.setattr(cr, "_folder", lambda: str(tmp_path))
    (tmp_path / cr._MARKER).write_text("77", encoding="utf-8")
    (tmp_path / cr._COUNTER).write_text("1", encoding="utf-8")

    assert cr.note_that_a_launch_began() == 2
    assert cr.should_start_without_the_backdrop() is True
    cr.note_a_clean_shutdown()


def test_a_launch_that_never_reaches_qt_is_not_counted(tmp_path, monkeypatch):
    """Qt's platform plugin failing is not the crash the counter is for."""
    import spacr.qt.app as app_mod

    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    began = []
    monkeypatch.setattr(cr, "note_that_a_launch_began",
                        lambda: began.append("counted") or 0)

    def no_platform(argv):
        raise RuntimeError("no Qt platform plugin could be initialized")

    no_platform.setAttribute = lambda *args: None
    monkeypatch.setattr(app_mod, "QApplication", no_platform)

    with pytest.raises(RuntimeError, match="platform plugin"):
        app_mod.launch(["--no-setup"])

    assert began == []
