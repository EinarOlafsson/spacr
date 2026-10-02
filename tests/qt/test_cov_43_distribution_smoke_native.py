"""The installed-application smoke's provenance and Mac menu legs, off a Mac.

The distribution smoke refuses a run that is not a frozen CPU artifact, and
on macOS it drives the real Cocoa application menu through the Objective-C
runtime. Neither can happen on a Linux test runner, so these tests stand in
for exactly what those legs read -- the environment, ``sys.frozen``, the
Objective-C message call, the Preferences dialog -- and check what each leg
refuses and records.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import startup_benchmark as benchmark  # noqa: E402

Controller = benchmark._DistributionSmokeController

pytestmark = pytest.mark.qt


def _controller(tmp_path, **extra):
    """A stand-in with the attributes the smoke's methods read."""
    calls = []
    fake = SimpleNamespace(
        output=tmp_path / "receipt.json", record={"status": "running"},
        phase="running", calls=calls,
        timer=SimpleNamespace(stop=lambda: calls.append("timer stopped")),
        app=SimpleNamespace(exit=lambda code: calls.append(("exit", code)),
                            platformName=lambda: "xcb",
                            activeModalWidget=lambda: None))
    fake._write = lambda: Controller._write(fake)
    fake._finish = lambda code: Controller._finish(fake, code)
    fake._pipeline_failed = lambda error: Controller._pipeline_failed(fake, error)
    fake._installed_root = Controller._installed_root
    fake._require_installed_origin = Controller._require_installed_origin
    fake.__dict__.update(extra)
    return fake


def test_a_failed_pipeline_is_written_and_leaves_the_event_loop(tmp_path):
    fake = _controller(tmp_path)
    Controller._pipeline_failed(fake, "the worker died")
    written = json.loads(fake.output.read_text())
    assert written["status"] == "failed" and written["error"] == "the worker died"
    assert fake.phase == "finished"
    assert fake.calls == ["timer stopped", ("exit", 1)]


def _frozen_cpu(monkeypatch, tmp_path):
    """Make this interpreter look like a frozen CPU artifact rooted at /."""
    import torch

    monkeypatch.setenv("SPACR_DEVICE", "cpu")
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPACR_DISTRIBUTION_KIND", "frozen")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.setattr(torch.version, "cuda", None)
    roots = []
    monkeypatch.setattr(Controller, "_installed_root", staticmethod(
        lambda executable, bundle, platform:
        roots.append(bundle) or Path(executable.anchor)))
    monkeypatch.setattr(Controller, "_require_installed_origin",
                        staticmethod(lambda path, root: None))
    return roots


def test_a_frozen_cpu_artifact_records_its_provenance(monkeypatch, tmp_path):
    roots = _frozen_cpu(monkeypatch, tmp_path)
    monkeypatch.setenv("SPACR_ACCEPTANCE_SOURCE_COMMIT", "abc123")
    fake = _controller(tmp_path)
    Controller._provenance(fake)
    record = fake.record
    assert record["device"] == "cpu" and record["frozen"] is True
    assert record["kind"] == "frozen" and record["source_commit"] == "abc123"
    assert record["packaged_resources_verified"] is True
    assert set(record["import_origins"]) == {"spacr", "numpy", "PySide6", "torch"}
    assert record["qt_platform"] == "xcb"
    assert [str(path) for path in roots] == [str(tmp_path)]


def test_an_offscreen_artifact_is_not_a_native_acceptance(monkeypatch, tmp_path):
    _frozen_cpu(monkeypatch, tmp_path)
    fake = _controller(tmp_path)
    fake.app.platformName = lambda: "offscreen"
    with pytest.raises(RuntimeError, match="requires the native Qt platform"):
        Controller._provenance(fake)
    assert fake.record["qt_platform"] == "offscreen"


@pytest.mark.parametrize("change,match", [
    ("device", "explicitly select CPU"),
    ("accelerator", "Accelerator visibility must be empty"),
    ("kind", "not the requested artifact family"),
    ("unfrozen", "not the requested artifact family"),
    ("version", "metadata does not match"),
    ("cuda", "contains CUDA torch"),
])
def test_an_artifact_that_is_not_a_frozen_cpu_build_is_refused(
        monkeypatch, tmp_path, change, match):
    import torch

    import spacr.version

    _frozen_cpu(monkeypatch, tmp_path)
    if change == "device":
        monkeypatch.setenv("SPACR_DEVICE", "cuda")
    elif change == "accelerator":
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    elif change == "kind":
        monkeypatch.setenv("SPACR_DISTRIBUTION_KIND", "wheel")
    elif change == "unfrozen":
        monkeypatch.setattr(sys, "frozen", False)
    elif change == "version":
        monkeypatch.setattr(spacr.version, "get_version", lambda: "0.0.0")
    else:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(torch.version, "cuda", "12.4")
    with pytest.raises(RuntimeError, match=match):
        Controller._provenance(_controller(tmp_path))


def test_the_objective_c_call_goes_through_the_runtime_it_names(monkeypatch):
    seen = []
    signature = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p,
                                 ctypes.c_void_p, ctypes.c_long)

    def send(receiver, selector, value):
        seen.append((receiver, selector, value))
        return value * 2

    keep = signature(send)

    class _Runtime:
        def __init__(self, path):
            seen.append(path)
            self.sel_registerName = lambda name: 77 if name == b"count:" else 0
            self.objc_msgSend = keep

    monkeypatch.setattr(ctypes, "CDLL", _Runtime)
    answer = Controller._cocoa_message(5, "count:", ctypes.c_long,
                                       (ctypes.c_long, 21))
    assert answer == 42
    assert seen == ["/usr/lib/libobjc.A.dylib", (5, 77, 21)]


def test_the_native_menu_check_needs_a_mac(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="requires macOS Cocoa"):
        Controller._start_native_menu_check(_controller(tmp_path))
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="requires macOS Cocoa"):
        Controller._start_native_menu_check(_controller(tmp_path))


class _Cocoa:
    """The few Objective-C objects the menu check reads, by integer id."""

    COMMAND = 1 << 20

    def __init__(self, items, bar_items=1, submenu=True):
        self.items = items
        self.bar_items = bar_items
        self.submenu = submenu
        self.strings = {}
        self.updated = False

    def _string(self, text):
        handle = 1000 + len(self.strings)
        self.strings[handle] = text
        return handle

    def __call__(self, receiver, selector, result_type, *arguments):
        if selector == "sharedApplication":
            return 1
        if selector == "mainMenu":
            return 2 if self.bar_items is not None else 0
        if receiver == 2 and selector == "numberOfItems":
            return self.bar_items
        if receiver == 2 and selector == "itemAtIndex:":
            return 3
        if receiver == 3 and selector == "submenu":
            return 4 if self.submenu else 0
        if receiver == 3 and selector == "title":
            return self._string("spaCR")
        if receiver == 4 and selector == "update":
            self.updated = True
            return None
        if receiver == 4 and selector == "numberOfItems":
            return len(self.items)
        if receiver == 4 and selector == "itemAtIndex:":
            return 100 + arguments[0][1]
        if selector == "UTF8String":
            text = self.strings[receiver]
            return text.encode("utf-8") if text else None
        item = self.items[receiver - 100]
        if selector == "title":
            return self._string(item[0])
        if selector == "keyEquivalent":
            return self._string(item[1]) if item[1] is not None else 0
        if selector == "keyEquivalentModifierMask":
            return item[2]
        if selector == "isEnabled":
            return item[3]
        if selector == "isHidden":
            return item[4]
        raise AssertionError(selector)


def _on_cocoa(monkeypatch, tmp_path, cocoa, native=True):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(ctypes, "CDLL", lambda path: SimpleNamespace(
        objc_getClass=lambda name: 9))
    scheduled = []
    monkeypatch.setattr(benchmark, "QTimer", SimpleNamespace(
        singleShot=lambda delay, call: scheduled.append(call)))
    fake = _controller(
        tmp_path, _cocoa_message=cocoa,
        window=SimpleNamespace(menuBar=lambda: SimpleNamespace(
            isNativeMenuBar=lambda: native)))
    fake.app.platformName = lambda: "cocoa"
    fake._invoke_native_preferences = "invoke preferences"
    return fake, scheduled


def test_the_system_menu_names_its_one_preferences_and_one_quit(
        monkeypatch, tmp_path):
    command = _Cocoa.COMMAND
    cocoa = _Cocoa([("About spaCR", "", 0, True, False),
                    ("Settings…", ",", command, True, False),
                    ("Hide", "h", command, True, True),
                    ("Separator", None, 0, False, False),
                    ("Quit spaCR", "q", command, True, False)])
    fake, scheduled = _on_cocoa(monkeypatch, tmp_path, cocoa)
    Controller._start_native_menu_check(fake)
    menu = fake.record["native_menu"]
    assert menu["mode"] == "system-menu-on-cocoa"
    assert menu["application_menu"] == "spaCR"
    assert menu["selected"] == {"preferences": 1, "quit": 4}
    assert menu["items"][3]["key"] == ""
    assert cocoa.updated and fake.phase == "native-menu-opening"
    assert fake._native_actions == {"preferences": 1, "quit": 4}
    assert scheduled == ["invoke preferences"]
    assert json.loads(fake.output.read_text())["native_menu"]["selected"]


@pytest.mark.parametrize("cocoa,match", [
    (_Cocoa([], bar_items=None), "has no application menu"),
    (_Cocoa([], bar_items=0), "has no application menu"),
    (_Cocoa([], submenu=False), "submenu is absent"),
    (_Cocoa([("Quit", "q", _Cocoa.COMMAND, True, False)]),
     "lacks a unique enabled preferences"),
    (_Cocoa([("Settings", ",", _Cocoa.COMMAND, True, False),
             ("Quit", "q", 0, True, False)]),
     "lacks a unique enabled quit"),
])
def test_a_system_menu_without_its_items_is_refused(
        monkeypatch, tmp_path, cocoa, match):
    fake, scheduled = _on_cocoa(monkeypatch, tmp_path, cocoa)
    with pytest.raises(RuntimeError, match=match):
        Controller._start_native_menu_check(fake)
    assert not scheduled


def test_an_in_window_bar_on_a_mac_is_checked_in_the_window(
        monkeypatch, tmp_path):
    fake, _scheduled = _on_cocoa(monkeypatch, tmp_path, _Cocoa([]),
                                 native=False)
    fake._start_window_menu_check = lambda: fake.calls.append("window menu")
    Controller._start_native_menu_check(fake)
    assert fake.calls == ["window menu"]


@pytest.mark.parametrize("mode", ["system-menu-on-cocoa",
                                  "in-window-on-cocoa"])
def test_preferences_must_open_the_verified_dialog(tmp_path, mode):
    clicked = []
    fake = _controller(tmp_path, phase="native-menu-opening",
                       record={"native_menu": {"mode": mode}},
                       _native_menu=4, _native_actions={"preferences": 1})

    def opened(*args):
        clicked.append(args)
        fake.phase = "native-menu-closing"

    fake._cocoa_message = opened
    fake._click_window_menu_action = opened
    Controller._invoke_native_preferences(fake)
    assert fake.phase == "native-menu-ready-to-quit"
    assert fake.record["native_menu"]["preferences_closed"] is True
    expected = (("preferences",) if mode == "in-window-on-cocoa"
                else (4, "performActionForItemAtIndex:", None,
                      (ctypes.c_long, 1)))
    assert clicked == [expected]


def test_preferences_that_open_nothing_fail_the_smoke(tmp_path):
    fake = _controller(tmp_path, record={"native_menu": {}},
                       _native_menu=4, _native_actions={"preferences": 1},
                       _cocoa_message=lambda *args: None)
    Controller._invoke_native_preferences(fake)
    assert fake.record["status"] == "failed"
    assert "did not open the verified dialog" in fake.record["error"]
    assert ("exit", 1) in fake.calls


class _Dialog:
    def __init__(self, saved=True, visible=True):
        self.saved, self.visible, self.rejected = saved, visible, False

    def isVisible(self):
        return self.visible

    def grab(self):
        return SimpleNamespace(save=lambda path: self.saved)

    def reject(self):
        self.rejected = True


def _polling(monkeypatch, tmp_path, phase, dialog=None, mode="x"):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_preferences_window_class",
                        lambda: _Dialog)
    fake = _controller(tmp_path, phase=phase,
                       record={"native_menu": {"mode": mode}},
                       _native_menu_started=benchmark.time.monotonic(),
                       _native_menu=4, _native_actions={"quit": 7})
    fake.app.activeModalWidget = lambda: dialog
    return fake


def test_the_native_menu_has_a_deadline(monkeypatch, tmp_path):
    fake = _polling(monkeypatch, tmp_path, "native-menu-opening")
    fake._native_menu_started -= 31
    with pytest.raises(RuntimeError, match="exceeded its deadline"):
        Controller._poll_native_menu_check(fake)


def test_preferences_are_waited_for_then_photographed_and_closed(
        monkeypatch, tmp_path):
    fake = _polling(monkeypatch, tmp_path, "native-menu-opening")
    Controller._poll_native_menu_check(fake)
    assert fake.phase == "native-menu-opening"
    dialog = _Dialog()
    fake.app.activeModalWidget = lambda: dialog
    Controller._poll_native_menu_check(fake)
    assert dialog.rejected and fake.phase == "native-menu-closing"
    assert fake.record["native_menu"]["preferences_opened"] is True
    assert fake.record["native_menu"]["dialog_class"] == "_Dialog"


@pytest.mark.parametrize("dialog,match", [
    (object(), "opened a different dialog"),
    (_Dialog(visible=False), "opened a different dialog"),
    (_Dialog(saved=False), "Preferences screenshot"),
])
def test_a_wrong_or_unrecorded_preferences_dialog_is_refused(
        monkeypatch, tmp_path, dialog, match):
    fake = _polling(monkeypatch, tmp_path, "native-menu-opening", dialog)
    with pytest.raises(RuntimeError, match=match):
        Controller._poll_native_menu_check(fake)


def test_quit_waits_for_preferences_to_close(monkeypatch, tmp_path):
    fake = _polling(monkeypatch, tmp_path, "native-menu-ready-to-quit",
                    _Dialog())
    with pytest.raises(RuntimeError, match="did not close before native Quit"):
        Controller._poll_native_menu_check(fake)


@pytest.mark.parametrize("mode", ["system-menu-on-cocoa",
                                  "in-window-on-cocoa"])
def test_quit_goes_through_the_menu_it_was_verified_in(
        monkeypatch, tmp_path, mode):
    fake = _polling(monkeypatch, tmp_path, "native-menu-ready-to-quit",
                    mode=mode)
    sent = []
    fake._cocoa_message = lambda *args: sent.append(args)
    fake._click_window_menu_action = lambda label: sent.append(label)
    Controller._poll_native_menu_check(fake)
    assert fake.phase == "native-menu-quitting"
    assert fake.record["native_menu"]["quit_dispatched"] is True
    assert sent == (["quit"] if mode == "in-window-on-cocoa" else
                    [(4, "performActionForItemAtIndex:", None,
                      (ctypes.c_long, 7))])
    assert json.loads(fake.output.read_text())["native_menu"][
        "quit_dispatched"] is True


def test_another_menu_phase_waits(monkeypatch, tmp_path):
    fake = _polling(monkeypatch, tmp_path, "native-menu-closing")
    Controller._poll_native_menu_check(fake)
    assert fake.phase == "native-menu-closing"


class _Bar:
    def __init__(self, entry, inside=True):
        from PySide6.QtCore import QRect

        self.entry = entry
        self.bounds = QRect(0, 0, 200, 30) if inside else QRect(0, 0, 1, 1)

    def actionGeometry(self, action):
        return self.entry

    def rect(self):
        return self.bounds


class _Menu:
    def __init__(self, visible=True, action_rect=None, saved=True):
        from PySide6.QtCore import QRect

        self.visible = visible
        self.action_rect = action_rect or QRect(0, 10, 80, 20)
        self.saved = saved

    def menuAction(self):
        return "menu action"

    def isVisible(self):
        return self.visible

    def actionGeometry(self, action):
        return self.action_rect

    def rect(self):
        from PySide6.QtCore import QRect

        return QRect(0, 0, 100, 100)

    def grab(self):
        return SimpleNamespace(save=lambda path: self.saved)


def _window_menu(monkeypatch, tmp_path, bar, menu):
    from PySide6.QtTest import QTest

    clicks = []
    monkeypatch.setattr(QTest, "mouseClick",
                        lambda widget, button, pos=None: clicks.append(widget))
    fake = _controller(tmp_path, window=SimpleNamespace(menuBar=lambda: bar),
                       _window_menu=menu,
                       _window_menu_actions={"quit": "quit action"})
    return fake, clicks


def test_a_window_menu_action_is_clicked_where_it_is_drawn(
        monkeypatch, tmp_path):
    from PySide6.QtCore import QRect

    bar, menu = _Bar(QRect(10, 5, 40, 20)), _Menu()
    fake, clicks = _window_menu(monkeypatch, tmp_path, bar, menu)
    Controller._click_window_menu_action(fake, "quit")
    assert clicks == [bar, menu]


@pytest.mark.parametrize("defect,match", [
    ("empty-entry", "outside its bar"),
    ("entry-outside", "outside its bar"),
    ("menu-closed", "did not open spaCR"),
    ("action-outside", "quit menu action is outside its popup"),
    ("unsaved", "Mac menu screenshot"),
])
def test_a_window_menu_that_cannot_be_clicked_is_refused(
        monkeypatch, tmp_path, defect, match):
    from PySide6.QtCore import QRect

    entry = QRect() if defect == "empty-entry" else QRect(10, 5, 40, 20)
    bar = _Bar(entry, inside=defect != "entry-outside")
    menu = _Menu(visible=defect != "menu-closed",
                 action_rect=QRect(0, 500, 80, 20)
                 if defect == "action-outside" else None,
                 saved=defect != "unsaved")
    fake, _clicks = _window_menu(monkeypatch, tmp_path, bar, menu)
    with pytest.raises(RuntimeError, match=match):
        Controller._click_window_menu_action(fake, "quit")


def test_a_registry_row_wins_over_its_own_fold_child():
    clicked = []

    class _Button:
        def __init__(self, fold_child):
            self.fold_child = fold_child

        def property(self, name):
            return {"navKey": "measure", "isFoldChild": self.fold_child}[name]

        def isEnabled(self):
            return True

        def click(self):
            clicked.append(self)

    row, child = _Button(False), _Button(True)
    fake = SimpleNamespace(
        _finished=False, phase="modules", index=0, keys=("measure",),
        _arm_timeout=lambda *a: None,
        window=SimpleNamespace(_sidebar=SimpleNamespace(_items=[child, row])))
    benchmark.BenchmarkController._advance(fake)
    assert clicked == [row] and fake._door == "sidebar"


def test_a_linux_artifact_is_rooted_beside_its_executable(tmp_path):
    executable = tmp_path / "spacr"
    executable.write_text("")
    assert Controller._installed_root(executable, tmp_path, "linux") == tmp_path


def _resources(tmp_path, *, policy="{}", icon=True, font=b""):
    """A copy of spaCR's installed resources with one thing wrong."""
    import shutil

    import spacr

    real = Path(spacr.__file__).parent / "resources"
    package = tmp_path / "installed" / "spacr"
    (package / "resources" / "icons").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "resources" / "layout_policy.json").write_text(policy)
    if icon:
        shutil.copy(real / "icons" / "measure.png",
                    package / "resources" / "icons" / "measure.png")
    fonts = package / "resources" / "font" / "open_sans"
    fonts.mkdir(parents=True)
    (fonts / "OpenSans-VariableFont_wdth,wght.ttf").write_bytes(font)
    return package / "__init__.py"


@pytest.mark.parametrize("defect,match", [
    ("policy", "layout policy or Measure icon is missing"),
    ("icon", "layout policy or Measure icon is missing"),
    ("font", "Open Sans font is missing"),
])
def test_missing_installed_resources_are_refused(
        monkeypatch, tmp_path, defect, match):
    import spacr

    _frozen_cpu(monkeypatch, tmp_path)
    fake_init = _resources(
        tmp_path, policy="{}" if defect == "policy" else '{"a": 1}',
        icon=defect != "icon", font=b"" if defect == "font" else b"font")
    monkeypatch.setattr(spacr, "__file__", str(fake_init))
    with pytest.raises(RuntimeError, match=match):
        Controller._provenance(_controller(tmp_path))


def _advancing(tmp_path, phase, **extra):
    failures = []
    fake = _controller(tmp_path, phase=phase,
                       started=benchmark.time.monotonic(),
                       root=tmp_path / "experiment",
                       window=SimpleNamespace(findChildren=lambda kind: []),
                       **extra)
    fake._pipeline_failed = failures.append
    return fake, failures


def test_the_smoke_has_a_ten_minute_deadline(tmp_path):
    fake, failures = _advancing(tmp_path, "launch")
    fake.started -= 601
    Controller._advance(fake)
    assert failures == ["Distribution smoke exceeded its 600-second deadline"]


def test_only_a_visible_tour_is_skipped_and_a_hidden_window_waits(tmp_path):
    skipped = []
    shown = SimpleNamespace(isVisible=lambda: True, _skip_btn=SimpleNamespace(
        click=lambda: skipped.append("skipped")))
    hidden = SimpleNamespace(isVisible=lambda: False)
    fake, failures = _advancing(tmp_path, "launch")
    fake.window = SimpleNamespace(findChildren=lambda kind: [hidden, shown],
                                  isVisible=lambda: False)
    fake._provenance = lambda: skipped.append("provenance")
    Controller._advance(fake)
    assert skipped == ["skipped", "provenance"]
    assert fake.record["tours_skipped"] == 1
    assert fake.phase == "launch" and not failures


def _screen(enabled=True, **extra):
    return SimpleNamespace(_btn_run=SimpleNamespace(isEnabled=lambda: enabled),
                           **extra)


def test_the_screen_phase_waits_for_an_enabled_run_and_no_modal(tmp_path):
    fake, failures = _advancing(tmp_path, "screen")
    fake.window._screens = {}
    Controller._advance(fake)
    fake.window._screens = {"measure": _screen(enabled=False)}
    Controller._advance(fake)
    assert fake.phase == "screen" and not failures
    fake.window._screens = {"measure": _screen()}
    fake.app.activeModalWidget = lambda: object()
    Controller._advance(fake)
    assert failures == [
        "An unexpected modal dialog blocks the installed application"]


@pytest.mark.parametrize("change,message", [
    ("src", "did not receive the smoke input"),
    ("jobs", "did not retain its one-worker limit"),
    ("crops", "unexpectedly requires crop confirmation"),
    ("worker", "did not start a pipeline worker"),
])
def test_a_configured_screen_that_drifted_is_refused(tmp_path, change, message):
    collected = {"src": "/smoke" if change != "src" else "/elsewhere",
                 "n_jobs": 4 if change == "jobs" else 1}
    screen = _screen(
        _settings_model=SimpleNamespace(collect=lambda: dict(collected)),
        _crop_choice_warnings=lambda settings: ["crop"] if change == "crops"
        else [],
        _worker=None)
    screen._btn_run.click = lambda: None
    fake, failures = _advancing(tmp_path, "configured", screen=screen,
                                _expected_src="/smoke")
    Controller._advance(fake)
    assert len(failures) == 1 and message in failures[0]


def test_a_configured_screen_waits_for_its_run_button(tmp_path):
    fake, failures = _advancing(tmp_path, "configured",
                                screen=_screen(enabled=False))
    Controller._advance(fake)
    assert fake.phase == "configured" and not failures


@pytest.mark.parametrize("phase,method", [
    ("settling-layout", "_poll_layout_check"),
    ("checking-layout-reachability", "_poll_layout_reachability"),
    ("native-menu-opening", "_poll_native_menu_check"),
    ("finished", None),
])
def test_each_later_phase_goes_to_its_own_poll(tmp_path, phase, method):
    polled = []
    fake, failures = _advancing(tmp_path, phase)
    for name in ("_poll_layout_check", "_poll_layout_reachability",
                 "_poll_native_menu_check"):
        setattr(fake, name, lambda name=name: polled.append(name))
    Controller._advance(fake)
    assert polled == ([method] if method else []) and not failures


def test_an_unsaved_screenshot_is_an_error(tmp_path):
    fake = _controller(tmp_path, window=SimpleNamespace(
        grab=lambda: SimpleNamespace(save=lambda path: False)))
    with pytest.raises(RuntimeError, match="native application screenshot"):
        Controller._save_layout_image(fake, "shot.png")


def test_an_editor_two_levels_inside_a_spin_box_is_its_own(qtbot):
    from PySide6.QtWidgets import QLineEdit, QSpinBox, QWidget

    window = QWidget()
    qtbot.addWidget(window)
    window.setFixedSize(600, 400)
    pane = QWidget(window)
    pane.setObjectName("Console")
    pane.setGeometry(0, 0, 600, 400)
    spin = QSpinBox(pane)
    spin.setGeometry(10, 10, 120, 30)
    holder = QWidget(spin)
    editor = QLineEdit(holder)
    editor.setObjectName("nested_editor")
    loose = QWidget(pane)
    loose.setGeometry(10, 60, 200, 40)
    standalone = QLineEdit(loose)
    standalone.setObjectName("loose_editor")
    standalone.setGeometry(0, 0, 150, 30)
    window.show()
    qtbot.waitUntil(window.isVisible)
    runtime = SimpleNamespace(
        count=lambda: 1, widget=lambda index: pane,
        _pane_of=lambda widget: None, sizes=lambda: [400])
    fake = SimpleNamespace(window=window, screen=SimpleNamespace(
        _runtime_splitter=runtime, _settings_panel=pane,
        _body_splitter=SimpleNamespace(sizes=lambda: [600])))
    measured = Controller._layout_snapshot(fake)
    rows = {control["object_name"]: control
            for control in measured["controls"]}
    assert rows["loose_editor"]["embedded_editor"] is False
    if "nested_editor" in rows:
        assert rows["nested_editor"]["embedded_editor"] is True


def _splitter(*panes):
    return SimpleNamespace(
        count=lambda: len(panes), widget=lambda index: panes[index],
        _pane_of=lambda pane: SimpleNamespace(name=pane.name))


class _Pane:
    def __init__(self, name, children):
        self.name, self.children = name, children

    def findChildren(self, kind):
        return list(self.children)


class _Child:
    def __init__(self, name="row", visible=True):
        self.name, self.visible = name, visible

    def objectName(self):
        return self.name

    def isVisibleTo(self, pane):
        return self.visible


def test_a_measured_control_is_found_again_only_if_it_is_the_same():
    child = _Child()
    record = {"pane": "System", "position": 0, "class": "_Child",
              "object_name": "row"}
    fake = SimpleNamespace(screen=SimpleNamespace(_runtime_splitter=_splitter(
        _Pane("Console", []), _Pane("System", [child]))))
    assert Controller._control_for_layout_record(fake, record) is child
    assert Controller._control_for_layout_record(
        fake, dict(record, position=3)) is None
    assert Controller._control_for_layout_record(
        fake, dict(record, pane="Actions")) is None
    renamed = SimpleNamespace(screen=SimpleNamespace(_runtime_splitter=_splitter(
        _Pane("System", [_Child("other")]), _Pane("System", [child]))))
    assert Controller._control_for_layout_record(renamed, record) is child


def test_the_reachability_check_skips_non_acceptance_controls_and_lost_documents(
        monkeypatch):
    from PySide6.QtWidgets import QScrollArea

    viewport = QScrollArea()
    layout = _contained_layout_with(
        {"pane": "Console", "acceptance_control": False, "clipped": True,
         "undersized": False},
        {"pane": "Console", "clipped": False, "undersized": False,
         "document_edges": {"start": [0, 0, 1, 1], "end": [0, 9, 1, 1]}})
    calls = []
    fake = SimpleNamespace(
        record={"layout": {"samples": [layout]}},
        screen=SimpleNamespace(_runtime_viewport=viewport),
        _layout_previous=layout, _layout_stable=5,
        _layout_started=benchmark.time.monotonic(),
        _layout_snapshot=lambda: layout,
        _control_for_layout_record=lambda control: None,
        _save_layout_image=lambda name: [1, 1],
        _write=lambda: calls.append("written"),
        _finish_layout_check=lambda: calls.append("finished"))
    Controller._poll_layout_check(fake)
    assert fake.phase == "checking-layout-reachability"
    assert [target.get("document_edge") for target in fake._layout_targets] == [
        None, None, None, "start", "end"]
    assert all(target.get("acceptance_control", True)
               for target in fake._layout_targets)
    assert fake._layout_document_cursors == []
    viewport.deleteLater()


def _contained_layout_with(*extra):
    names = ("Console", "System", "Actions")
    return {"window_size": [1280, 720],
            "panes": [{"name": name} for name in names],
            "controls": [{"pane": name, "clipped": False, "undersized": False}
                         for name in names] + list(extra)}


def test_reachability_has_a_deadline_and_names_a_vanished_control():
    calls = []
    record = {"reachability": []}
    fake = SimpleNamespace(
        record={"layout": record},
        _layout_reachability_started=benchmark.time.monotonic() - 121,
        _finish_layout_check=lambda: calls.append("finished"))
    Controller._poll_layout_reachability(fake)
    assert record["status"] == "failed" and record["reachability_timeout"]
    assert calls == ["finished"]
    target = {"pane": "Console", "position": 0}
    outside = object()
    fake = SimpleNamespace(
        record={"layout": {"reachability": []}},
        _layout_reachability_started=benchmark.time.monotonic(),
        _layout_scroll=SimpleNamespace(isAncestorOf=lambda widget: False),
        _layout_target=None, _layout_restoring=False,
        _layout_targets=[target, dict(target, position=1)],
        _control_for_layout_record=lambda before: (
            None if before["position"] == 0 else outside),
        _write=lambda: calls.append("written"))
    Controller._poll_layout_reachability(fake)
    Controller._poll_layout_reachability(fake)
    rows = fake.record["layout"]["reachability"]
    assert [row["before"]["position"] for row in rows] == [0, 1]
    assert all(row["reachable"] is False for row in rows)
    assert all(row["error"] == "Observed control disappeared" for row in rows)


def test_a_passed_layout_hands_on_to_the_native_menu_when_asked(
        monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_NATIVE_MENU_SMOKE", "1")
    fake = _controller(tmp_path, record={"layout": {"status": "passed"}})
    fake._start_native_menu_check = lambda: fake.calls.append("menu check")
    Controller._finish_layout_check(fake)
    assert fake.calls == ["menu check"]
    assert fake.record["visual_layout_status"] == "passed"


def test_a_mac_window_bar_without_a_spacr_menu_is_refused(tmp_path):
    bar = SimpleNamespace(
        isVisible=lambda: True,
        visibleRegion=lambda: SimpleNamespace(isEmpty=lambda: False),
        findChildren=lambda kind: [])
    fake = _controller(tmp_path, window=SimpleNamespace(menuBar=lambda: bar))
    with pytest.raises(RuntimeError, match="does not have one spaCR menu"):
        Controller._start_window_menu_check(fake)


def test_a_quit_after_the_smoke_finished_changes_nothing(tmp_path):
    fake = _controller(tmp_path, phase="finished",
                       record={"status": "passed"})
    Controller._quitting(fake)
    assert fake.record == {"status": "passed"}
    assert not fake.output.exists()
