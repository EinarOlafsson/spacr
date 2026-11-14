"""Small shell registries that run before or without a window.

* :func:`spacr.qt.menus.name_the_macos_application_menu` over a scripted
  Objective-C runtime (no Mac needed): it writes exactly ``CFBundleName`` on
  the one supported shape and refuses every other one quietly;
* :func:`spacr.qt.maturity.apply`, the stage table writer, which never
  demotes and never lets one failed retirement stop the rest;
* :mod:`spacr.qt.prefs`, the recent-folder store and the background writer's
  flush.
"""
from __future__ import annotations

import logging
import sys
import threading
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings                             # noqa: E402

from spacr.qt import maturity, prefs                             # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# macOS application-menu name
# ---------------------------------------------------------------------------

class _Callable:
    """A ctypes function stand-in which accepts ``restype``/``argtypes``."""

    def __init__(self, callback):
        self._callback = callback
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._callback(*args)


class _ObjectiveCRuntime:
    """The tiny Objective-C surface the bundle-name helper uses."""

    bundle_class = 101
    string_class = 102

    def __init__(self, *, bundle=201, info=202,
                 class_name=b"__NSCFDictionary", failed_string=None):
        self.bundle = bundle
        self.info = info
        self.class_name = class_name
        self.failed_string = failed_string
        self.assignments = []
        self.objc_getClass = _Callable(
            lambda name: {b"NSBundle": self.bundle_class,
                          b"NSString": self.string_class}[name])
        self.sel_registerName = _Callable(lambda selector: selector)
        self.object_getClassName = _Callable(lambda info: self.class_name)
        self.objc_msgSend = object()

    def send(self, receiver, selector, *args):
        if receiver == self.bundle_class and selector == b"mainBundle":
            return self.bundle
        if receiver == self.bundle and selector == b"infoDictionary":
            return self.info
        if (receiver == self.string_class
                and selector == b"stringWithUTF8String:"):
            return 0 if args[0] == self.failed_string else ("NSString", args[0])
        if receiver == self.info and selector == b"setObject:forKey:":
            self.assignments.append(args)
            return 1
        raise AssertionError((receiver, selector, args))


def test_the_macos_menu_name_is_written_only_to_a_mutable_bundle(
        monkeypatch, subtests):
    """Only an unbundled darwin launch with a mutable info dictionary is renamed."""
    import ctypes
    import ctypes.util

    from spacr.qt import menus

    def _load_error(path):
        raise OSError(f"cannot load {path}")

    cases = (
        # (case, runtime kwargs, platform, opt-out env, find_library, loader,
        #  expected result, expected assignments)
        ("mutable dictionary", {}, "darwin", None, True, None, True,
         [(("NSString", b"spaCR"), ("NSString", b"CFBundleName"))]),
        ("no main bundle", {"bundle": 0}, "darwin", None, True, None, False, []),
        ("no info dictionary", {"info": 0}, "darwin", None, True, None, False,
         []),
        ("immutable dictionary", {"class_name": b"__NSDictionaryI"}, "darwin",
         None, True, None, False, []),
        ("name not encodable", {"failed_string": b"spaCR"}, "darwin", None,
         True, None, False, []),
        ("key not encodable", {"failed_string": b"CFBundleName"}, "darwin",
         None, True, None, False, []),
        ("opted out", {}, "darwin", "1", True, None, False, []),
        ("opt-out of 0 still runs", {}, "darwin", "0", True, None, True,
         [(("NSString", b"spaCR"), ("NSString", b"CFBundleName"))]),
        ("no objc library", {}, "darwin", None, False, None, False, []),
        ("loader error", {}, "darwin", None, True, _load_error, False, []),
        ("not darwin", {}, "linux", None, True, None, False, []),
    )
    for (case, kwargs, platform, opt_out, found, loader, expected,
         assignments) in cases:
        with subtests.test(msg=case):
            runtime = _ObjectiveCRuntime(**kwargs)
            with monkeypatch.context() as patch:
                patch.setattr(menus.sys, "platform", platform)
                if opt_out is None:
                    patch.delenv(menus.BUNDLE_NAME_OPT_OUT, raising=False)
                else:
                    patch.setenv(menus.BUNDLE_NAME_OPT_OUT, opt_out)
                patch.setattr(ctypes.util, "find_library",
                              lambda name: f"/fake/{name}" if found else None)
                patch.setattr(
                    ctypes.cdll, "LoadLibrary", loader or (
                        lambda path: runtime if path.endswith("/objc")
                        else object()))
                patch.setattr(ctypes, "cast",
                              lambda pointer, signature: runtime.send)
                assert menus.name_the_macos_application_menu("spaCR") is expected
            assert runtime.assignments == assignments


# ---------------------------------------------------------------------------
# Maturity stages
# ---------------------------------------------------------------------------

def test_maturity_apply_promotes_fills_gaps_and_never_demotes(
        monkeypatch, caplog, subtests):
    """Unassessed keys become alpha, stated stages stay, failures are only logged."""
    from spacr.qt import app as app_mod

    promoted = next(iter(maturity.PROMOTIONS))
    promoted_stage = maturity.PROMOTIONS[promoted][0]
    with subtests.test(msg="a stage set further along is not demoted"):
        stages = {promoted: "stable"}
        assert maturity.apply(stages, keys=()) == []
        assert stages == {promoted: "stable"}
    with subtests.test(msg="an alpha stage is promoted"):
        stages = {promoted: "alpha"}
        expected = [] if promoted_stage == "alpha" else [promoted]
        assert maturity.apply(stages, keys=()) == expected
        assert stages.get(promoted, "stable") == promoted_stage
    with subtests.test(msg="an empty string is not a stated stage"):
        stages = {"half_written": ""}
        assert maturity.apply(stages, keys=("half_written",)) == ["half_written"]
        assert stages["half_written"] == maturity.UNASSESSED_STAGE == "alpha"
    with subtests.test(msg="a stated stage is kept"):
        stages = {"authored": "beta"}
        assert maturity.apply(stages, keys=("authored",)) == []
        assert stages["authored"] == "beta"
    with subtests.test(msg="a failed retirement does not stop the fill"):
        with monkeypatch.context() as patch:
            patch.setattr(maturity, "RETIREMENTS", {"gone": "superseded"})
            patch.setattr(app_mod, "unregister_app",
                          lambda key: (_ for _ in ()).throw(
                              KeyError("never registered")))
            stages = {}
            with caplog.at_level(logging.DEBUG, logger="spacr.qt.maturity"):
                assert maturity.apply(stages, keys=("brand_new",)) == [
                    "brand_new"]
        assert "could not retire 'gone'" in caplog.text
        assert stages["brand_new"] == maturity.UNASSESSED_STAGE
    with subtests.test(msg="no stage table without the app registry"):
        caplog.clear()
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "spacr.qt.app", None)
            with caplog.at_level(logging.DEBUG, logger="spacr.qt.maturity"):
                assert maturity.apply() == []
        assert "the app registry is not importable" in caplog.text


# ---------------------------------------------------------------------------
# Recent folders and the background writer
# ---------------------------------------------------------------------------

def test_recent_sources_read_string_and_list_forms(tmp_path, monkeypatch,
                                                   subtests):
    """Either stored shape is the same list, blanks dropped and the limit honoured."""
    ini = tmp_path / "recent.ini"
    monkeypatch.setattr(prefs, "_s",
                        lambda: QSettings(str(ini), QSettings.IniFormat))
    cases = (
        # (case, stored, limit, expected)
        ("unset", None, 8, []),
        ("list", ["/a", "/b", "/c"], 8, ["/a", "/b", "/c"]),
        ("list with blank, limited", ["/a", "", "/b", "/c"], 2, ["/a", "/b"]),
        ("joined string", "/plates/a\n/plates/b\n\n/plates/c", 8,
         ["/plates/a", "/plates/b", "/plates/c"]),
        ("joined string limited", "/plates/a\n/plates/b\n\n/plates/c", 2,
         ["/plates/a", "/plates/b"]),
        ("one entry string", "/only", 8, ["/only"]),
    )
    for case, stored, limit, expected in cases:
        with subtests.test(msg=case):
            store = prefs._s()
            store.remove("recent/annotate/list")
            if stored is not None:
                store.setValue("recent/annotate/list", stored)
            store.sync()
            assert prefs.get_recent_sources("annotate", limit=limit) == expected


@pytest.fixture
def private_writer(monkeypatch):
    """Private pending table and state for the preference writer."""
    pending = {}
    state = {"busy": False, "thread": None}
    monkeypatch.setattr(prefs, "_STORES", threading.Condition())
    monkeypatch.setattr(prefs, "_PENDING_STORES", pending)
    monkeypatch.setattr(prefs, "_STORE_STATE", state)
    return pending, state


def test_a_flush_stores_what_waits_unless_the_writer_is_busy(
        monkeypatch, private_writer, subtests):
    """No or a live idle writer: pending values are written here; busy: left queued."""
    pending, state = private_writer
    written = []
    monkeypatch.setattr(prefs, "_write_stores", written.append)
    alive = types.SimpleNamespace(is_alive=lambda: True)
    dead = types.SimpleNamespace(is_alive=lambda: False)
    cases = (
        # (case, thread, busy, timeout, result, written, left pending)
        ("no writer thread", None, False, 0.5, True, [{("x.ini", 0, "k"): 5}],
         {}),
        ("dead writer", dead, False, 0.5, True, [{("x.ini", 0, "k"): 5}], {}),
        ("live writer timed out", alive, False, 0.05, True,
         [{("x.ini", 0, "k"): 5}], {}),
        ("busy writer", alive, True, 0.0, False, [],
         {("x.ini", 0, "k"): 5}),
    )
    for case, thread, busy, timeout, result, expected, left in cases:
        with subtests.test(msg=case):
            written.clear()
            pending.clear()
            pending[("x.ini", 0, "k")] = 5
            state.update(busy=busy, thread=thread)
            assert prefs._flush_stores(timeout=timeout) is result
            assert written == expected
            assert pending == left
    with subtests.test(msg="nothing pending"):
        written.clear()
        pending.clear()
        state.update(busy=False, thread=None)
        assert prefs._flush_stores(timeout=0.0) is True
        assert written == []
