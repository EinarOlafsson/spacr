"""Every bundled tutorial script, driven for real against a MainWindow.

The point of these tests is the failure mode the tutorials are most
exposed to: a script names a widget that has been renamed, moved or
deleted, the engine's broad ``except`` swallows it, and the video ships
with the cursor pointing at nothing. Two bugs of exactly that shape were
found writing this file — see the module docstring in
``spacr/qt/tutorial/scripts.py`` and the regression tests at the bottom.

So each script is *run*: the steps are built, each step's action is
fired in order, and after each one the engine is asked to resolve that
step's target and highlight. Every declared target must come back as a
live widget with a real rectangle.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from spacr.qt.tutorial.scripts import AVAILABLE_TUTORIALS   # noqa: E402


# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def main_window(qt_theme_applied):
    """One MainWindow for the whole module.

    Building it costs ~11 s under the themed stylesheet, and every test
    here navigates itself to the screen it cares about, so a shared
    window is both faster and closer to how a real tutorial run works
    (one window, many modules opened in sequence).
    """
    from spacr.qt.first_run import mark_tour_seen
    mark_tour_seen()
    from spacr.qt.app import MainWindow
    win = MainWindow()
    win.resize(1200, 800)
    win.show()
    qt_theme_applied.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qt_theme_applied.processEvents()


@pytest.fixture
def home_in_tmp(tmp_path, monkeypatch):
    """Point Path.home() at a scratch dir so _tutorial_scratch and the
    demo generators never touch the developer's real home."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    return fake_home


class _Probe:
    """A Director bound to a window but with the render pipeline unused —
    we only want its real target/highlight resolution logic."""

    def __init__(self, window, tmp_path):
        from spacr.qt.tutorial.engine import Director
        from tests.qt.test_tutorial_director import FakeNarrator
        self.d = Director(window, [], out_dir=tmp_path / "probe-out",
                            narrator=FakeNarrator())

    def widget_of(self, step):
        raw = (step.target[0] if isinstance(step.target, tuple)
                 else step.target)
        return self.d._deref(raw)

    def close(self):
        shutil.rmtree(self.d._workdir, ignore_errors=True)


def _drive(window, steps, probe, app):
    """Walk a script the way Director._run_capture does: resolve the
    step's target, fire its action, let Qt settle. Returns one record per
    step."""
    records = []
    for i, step in enumerate(steps):
        widget = probe.widget_of(step)
        point = probe.d._resolve_target(step)
        rect = probe.d._resolve_highlight_rect(step)
        records.append({"i": i, "step": step, "widget": widget,
                          "point": point, "rect": rect})
        if step.action is not None:
            step.action()
        app.processEvents()
    return records


# ---------------------------------------------------------------------------
# build_steps dispatch
# ---------------------------------------------------------------------------

def test_available_tutorials_is_the_exact_dispatch_table():
    assert AVAILABLE_TUTORIALS == ["home", "mask", "measure", "crop",
                                     "classify", "timelapse"]
    assert len(set(AVAILABLE_TUTORIALS)) == len(AVAILABLE_TUTORIALS)


def test_build_steps_rejects_unknown_keys_and_names_the_valid_ones():
    from spacr.qt.tutorial.scripts import build_steps
    with pytest.raises(ValueError) as exc:
        build_steps("nonexistent-tutorial-name", window=None)
    msg = str(exc.value)
    assert "nonexistent-tutorial-name" in msg
    for name in AVAILABLE_TUTORIALS:
        assert name in msg


@pytest.mark.parametrize("app_key", AVAILABLE_TUTORIALS)
def test_every_script_is_well_formed(app_key, main_window, home_in_tmp):
    """Narration is real prose, holds are sane, and no step declares a
    target slot that is already dead at build time."""
    from spacr.qt.tutorial.engine import Step
    from spacr.qt.tutorial.scripts import build_steps

    steps = build_steps(app_key, main_window)
    assert steps, f"{app_key} produced no steps"
    assert all(isinstance(s, Step) for s in steps)

    narrations = [s.narration for s in steps]
    for text in narrations:
        assert isinstance(text, str)
        assert len(text.strip()) >= 30, f"stub narration: {text!r}"
        assert text.strip()[-1] in ".!?", f"unfinished narration: {text!r}"
    assert len(set(narrations)) == len(narrations), "duplicated narration"

    for s in steps:
        assert isinstance(s.hold_ms, int) and 0 <= s.hold_ms <= 5000
        assert s.action is None or callable(s.action)
        assert isinstance(s.dim_background, bool)
        assert isinstance(s.live_capture, bool)
        if s.target is not None:
            widget = (s.target[0] if isinstance(s.target, tuple)
                        else s.target)
            # Regression guard: a target slot that is None at build time
            # can never resolve — that is the eager-evaluation bug.
            assert widget is not None, (
                f"{app_key} step {narrations[steps.index(s)][:40]!r} has a "
                "target that was already None when the script was built")
        if s.highlight is not None:
            assert callable(s.highlight) or hasattr(s.highlight, "rect")

    # Every script contains at least one real click. Passive target and
    # highlight steps must not display the click point.
    assert any(s.target is not None and s.show_pointer for s in steps)
    assert all(s.target is not None for s in steps if s.show_pointer)


@pytest.mark.parametrize("app_key", AVAILABLE_TUTORIALS)
def test_every_script_target_resolves_to_a_live_widget(app_key, main_window,
                                                         home_in_tmp,
                                                         tmp_path,
                                                         qt_theme_applied,
                                                         caplog):
    """Drive the whole script and prove every declared target lands on a
    real, sized widget inside the window — and that the engine never had
    to log a "did not resolve" warning to get there."""
    from PySide6.QtWidgets import QWidget
    from spacr.qt.tutorial.scripts import build_steps

    probe = _Probe(main_window, tmp_path)
    steps = build_steps(app_key, main_window)
    with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
        records = _drive(main_window, steps, probe, qt_theme_applied)

    targeted = [r for r in records if r["step"].target is not None]
    assert targeted, f"{app_key} targets nothing"
    for rec in targeted:
        head = rec["step"].narration[:45]
        assert isinstance(rec["widget"], QWidget), (
            f"{app_key} step {rec['i']} ({head!r}) target resolved to "
            f"{rec['widget']!r}, not a widget")
        assert rec["widget"].width() > 0 and rec["widget"].height() > 0, (
            f"{app_key} step {rec['i']} ({head!r}) targets a zero-sized "
            "widget")
        assert rec["point"] is not None, (
            f"{app_key} step {rec['i']} ({head!r}) target did not resolve "
            "to a point")
        x, y = rec["point"]
        assert math.isfinite(x) and math.isfinite(y)

    for rec in [r for r in records if r["step"].highlight is not None]:
        assert rec["rect"] is not None, (
            f"{app_key} step {rec['i']} highlight did not resolve")
        _hx, _hy, hw, hh = rec["rect"]
        assert hw > 0 and hh > 0

    unresolved = [r.message for r in caplog.records
                    if "did not resolve" in r.message
                    or "no sidebar row" in r.message
                    or "no Demos menu" in r.message]
    assert not unresolved, unresolved
    probe.close()


@pytest.mark.parametrize("app_key", AVAILABLE_TUTORIALS)
def test_every_script_lands_on_the_screen_it_narrates(app_key, main_window,
                                                        home_in_tmp,
                                                        tmp_path,
                                                        qt_theme_applied):
    """After the script runs, the stack is showing the module the
    tutorial is about — not whatever happened to be open."""
    from spacr.qt.tutorial.scripts import build_steps

    expected = {"home": "mask",          # home ends by opening Mask
                 "mask": "mask",
                 "measure": "measure",
                 "crop": "measure",       # crop is a measure output
                 "classify": "annotate",  # classify starts in annotate
                 "timelapse": "timelapse"}[app_key]

    probe = _Probe(main_window, tmp_path)
    steps = build_steps(app_key, main_window)
    _drive(main_window, steps, probe, qt_theme_applied)

    assert expected in main_window._screens
    assert main_window._stack.currentWidget() is main_window._screens[expected]
    probe.close()


# ---------------------------------------------------------------------------
# The regressions: targets that only exist once an earlier step has run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", ["mask", "measure", "crop", "classify",
                                       "timelapse"])
def test_deferred_targets_are_dead_before_their_step_and_live_after(
        app_key, main_window, home_in_tmp, tmp_path, qt_theme_applied):
    """Regression for the eager-evaluation bug.

    Every module script names a settings panel / console / Run button
    that only exists after the demo-loading step has run. Those slots
    used to be evaluated while the Step list was being built, so they
    were permanently None and five of the six bundled tutorials drew no
    highlight and moved the cursor to nothing.

    Here: at build time the deferred slots resolve to None; once the
    script has been driven, they resolve to real widgets.
    """
    from PySide6.QtWidgets import QWidget
    from spacr.qt.tutorial.scripts import build_steps

    probe = _Probe(main_window, tmp_path)
    steps = build_steps(app_key, main_window)

    deferred = [s for s in steps
                  if s.target is not None
                  and callable(s.target[0] if isinstance(s.target, tuple)
                                 else s.target)]
    assert deferred, f"{app_key} has no deferred targets to check"
    # Not yet: nothing has navigated or loaded a demo.
    assert all(probe.widget_of(s) is None for s in deferred)

    _drive(main_window, steps, probe, qt_theme_applied)

    # Now every one of them points at something real.
    for s in deferred:
        widget = probe.widget_of(s)
        assert isinstance(widget, QWidget), (
            f"{app_key}: {s.narration[:45]!r} still resolves to {widget!r}")
        assert widget.width() > 0
    probe.close()


def test_run_step_targets_run_not_run_preview(main_window, home_in_tmp,
                                                tmp_path, qt_theme_applied):
    """Regression: _find_button matched on prefix only, and the Mask
    screen's child order puts "Run preview" ahead of "Run" — so the step
    narrating the actual run highlighted the preview button."""
    from spacr.qt.tutorial.scripts import build_steps, _find_button

    probe = _Probe(main_window, tmp_path)
    steps = build_steps("mask", main_window)
    _drive(main_window, steps, probe, qt_theme_applied)

    screen = main_window._screens["mask"]
    # The ambiguity is real on this screen…
    labels = {b.text().strip() for b in screen.findChildren(
        __import__("PySide6.QtWidgets", fromlist=["QPushButton"]).QPushButton)}
    assert {"Run", "Run preview"} <= labels

    # …and the helper resolves it exactly.
    assert _find_button(screen, "Run").text().strip() == "Run"
    assert _find_button(screen, "Run preview").text().strip() == "Run preview"

    run_steps = [s for s in steps if "When you hit Run" in s.narration]
    assert len(run_steps) == 1, "the mask script should have one Run step"
    for s in run_steps:
        assert s.highlight is not None
        assert probe.d._deref(s.highlight).text().strip() == "Run"
    probe.close()


def test_settings_and_console_steps_point_at_different_panels(
        main_window, home_in_tmp, tmp_path, qt_theme_applied):
    """The mask script narrates the settings panel and then the console.
    Before the _settings_panel fix both steps resolved into the console
    box, so the cursor never moved between them."""
    from spacr.qt.tutorial.scripts import build_steps

    probe = _Probe(main_window, tmp_path)
    steps = build_steps("mask", main_window)
    records = _drive(main_window, steps, probe, qt_theme_applied)

    settings_rec = next(r for r in records
                          if "settings panel on the left" in r["step"].narration)
    console_rec = next(r for r in records
                         if "console on the right" in r["step"].narration)

    assert settings_rec["widget"] is not None
    assert console_rec["widget"] is not None
    assert settings_rec["widget"] is not console_rec["widget"]
    assert not console_rec["widget"].isAncestorOf(settings_rec["widget"])
    # Narration says left/right, so the resolved points must obey that.
    assert settings_rec["point"][0] < console_rec["point"][0]
    probe.close()


@pytest.mark.parametrize("app_key,nav_key", [
    ("home", "mask"), ("mask", "mask"), ("measure", "measure"),
    ("crop", "measure"), ("classify", "annotate"),
    ("timelapse", "timelapse"),
])
def test_sidebar_steps_point_at_the_row_they_narrate(app_key, nav_key,
                                                       main_window,
                                                       home_in_tmp,
                                                       tmp_path,
                                                       monkeypatch):
    """Regression: the sidebar targets were hard-coded pixel offsets.
    (100, 250) named "the mask module" but landed on Map Barcodes, and
    (100, 300) named "annotate" and landed in the gap between two
    sections. They now resolve through the row's navKey property."""
    from spacr.qt.tutorial import engine
    from spacr.qt.tutorial.scripts import build_steps

    # 1:1 window->frame scaling so resolved points are directly
    # comparable with the widget geometry.
    monkeypatch.setattr(engine, "VIDEO_SIZE",
                          (main_window.width(), main_window.height()))
    probe = _Probe(main_window, tmp_path)
    steps = build_steps(app_key, main_window)
    sidebar_steps = [s for s in steps
                       if probe.widget_of(s) is not None
                       and probe.widget_of(s).property("navKey") is not None]
    assert sidebar_steps, f"{app_key} never points at a sidebar row"
    keys = {probe.widget_of(s).property("navKey") for s in sidebar_steps}
    assert nav_key in keys, f"{app_key} points at {keys}, expected {nav_key}"

    # And the resolved point is genuinely inside that row.
    for s in sidebar_steps:
        widget = probe.widget_of(s)
        if widget.property("navKey") != nav_key:
            continue
        point = probe.d._resolve_target(s)
        top_left = widget.mapTo(main_window, widget.rect().topLeft())
        assert top_left.x() <= point[0] <= top_left.x() + widget.width()
        assert top_left.y() <= point[1] <= top_left.y() + widget.height()
    probe.close()


def test_demos_menu_step_points_at_the_demos_menu(main_window, home_in_tmp,
                                                    tmp_path):
    """Regression: the Demos step used the literal point (170, 15), which
    is past the end of the menu bar's items — the cursor landed on blank
    chrome. It now comes from the menu's own action geometry."""
    from spacr.qt.tutorial.scripts import build_steps, _menu_target

    probe = _Probe(main_window, tmp_path)
    steps = build_steps("home", main_window)
    menu_steps = [s for s in steps
                    if isinstance(s.target, tuple)
                    and s.target[0] is main_window.menuBar()]
    assert len(menu_steps) == 1
    _widget, offset = menu_steps[0].target
    assert offset is not None

    mb = main_window.menuBar()
    demos = next(a for a in mb.actions()
                   if a.text().replace("&", "") == "Demos")
    rect = mb.actionGeometry(demos)
    assert rect.contains(*offset), (
        f"menu target {offset} is outside the Demos item {rect}")

    # And the helper degrades loudly, not silently, for a missing menu.
    fallback = _menu_target(main_window, "NoSuchMenu")
    assert fallback[0] is mb and fallback[1] is None
    probe.close()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def test_go_home_and_nav_to_actually_navigate(main_window, qt_theme_applied):
    from spacr.qt.tutorial.scripts import _go_home, _nav_to

    _nav_to(main_window, "measure")()
    qt_theme_applied.processEvents()
    assert main_window._stack.currentWidget() is main_window._screens["measure"]

    _go_home(main_window)()
    qt_theme_applied.processEvents()
    assert main_window._stack.currentWidget() is main_window._startup

    _nav_to(main_window, "mask")()
    assert main_window._stack.currentWidget() is main_window._screens["mask"]


def test_nav_to_returns_a_fresh_callable_per_key(main_window):
    from spacr.qt.tutorial.scripts import _nav_to
    a = _nav_to(main_window, "mask")
    b = _nav_to(main_window, "measure")
    assert a is not b and callable(a) and callable(b)


@pytest.mark.parametrize("demo_key,target_app", [
    ("mask", "mask"), ("measure", "measure"), ("crop", "measure"),
    ("classify", "annotate"), ("timelapse", "timelapse"),
])
def test_load_demo_generates_data_and_opens_the_target_app(
        demo_key, target_app, main_window, tmp_path, qt_theme_applied):
    """_load_demo is the file-dialog-free twin of the Demos menu: it must
    write a real dataset into the scratch root and leave the matching
    screen on screen."""
    from spacr.qt.tutorial.scripts import _load_demo

    root = tmp_path / "scratch"
    root.mkdir()
    _load_demo(main_window, demo_key, str(root))()
    qt_theme_applied.processEvents()

    dst = root / demo_key
    assert dst.is_dir(), "the demo destination was not created"
    assert any(dst.rglob("*")), f"{demo_key} demo wrote no files"
    assert main_window._stack.currentWidget() is (
        main_window._screens[target_app])


def test_load_demo_uses_the_windows_own_demo_targets(main_window,
                                                       tmp_path):
    """Every tutorial that loads a demo must name a key MainWindow knows,
    otherwise _load_demo raises KeyError mid-render."""
    for demo_key in ("mask", "measure", "crop", "classify", "timelapse"):
        assert demo_key in main_window.DEMO_TARGETS


def test_sidebar_button_resolves_by_nav_key(main_window):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.tutorial.scripts import _sidebar_button
    for key in ("mask", "measure", "annotate", "timelapse", "__home__"):
        btn = _sidebar_button(main_window, key)
        assert isinstance(btn, QPushButton)
        assert btn.property("navKey") == key


def test_sidebar_button_falls_back_to_the_label_then_warns(main_window,
                                                             caplog):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.tutorial.scripts import _sidebar_button
    # Labels are indented ("  Mask"); the helper strips before comparing.
    btn = _sidebar_button(main_window, "Mask")
    assert isinstance(btn, QPushButton)
    assert btn.property("navKey") == "mask"

    with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
        fallback = _sidebar_button(main_window, "no-such-app")
    assert fallback is main_window._sidebar
    assert any("no sidebar row" in r.message for r in caplog.records)


def test_menu_bar_helper_returns_the_windows_menu_bar(main_window):
    from spacr.qt.tutorial.scripts import _menu_bar
    assert _menu_bar(main_window) is main_window.menuBar()


def test_open_demos_menu_returns_the_menu_and_never_pops_it_up(main_window,
                                                                 caplog):
    """It must resolve the menu (so a rename is detectable) without
    actually popping it up — a live popup would grab input for the rest
    of the render."""
    from PySide6.QtGui import QAction
    from spacr.qt.tutorial.scripts import _open_demos_menu
    act = _open_demos_menu(main_window)
    assert isinstance(act, QAction)
    assert act.text().replace("&", "") == "Demos"
    assert act in main_window.menuBar().actions()
    # Resolving must not have opened anything.
    from PySide6.QtWidgets import QApplication
    assert QApplication.activePopupWidget() is None

    class NoMenus:
        def menuBar(self):
            class MB:
                def actions(self):
                    return []
            return MB()

    with caplog.at_level("WARNING", logger="spacr.qt.tutorial"):
        assert _open_demos_menu(NoMenus()) is None
    assert any("no Demos menu" in r.message for r in caplog.records)


def test_find_button_prefers_exact_then_prefix_then_none(qtbot,
                                                           qt_theme_applied):
    from PySide6.QtWidgets import QPushButton, QWidget
    from spacr.qt.tutorial.scripts import _find_button

    host = QWidget()
    qtbot.addWidget(host)
    # Prefix match comes first in child order — exact must still win.
    preview = QPushButton("Run preview", host)
    exact = QPushButton("Run", host)
    train = QPushButton("Train CV", host)

    assert _find_button(host, "Run") is exact
    assert _find_button(host, "run") is exact          # case-insensitive
    assert _find_button(host, "Run pre") is preview     # prefix fallback
    assert _find_button(host, "Train") is train         # only prefix exists
    assert _find_button(host, "Publish") is None
    assert _find_button(None, "Run") is None


@pytest.mark.parametrize("app_key", ["mask", "measure", "timelapse"])
def test_settings_panel_is_the_left_column_not_the_console_scroll(
        app_key, main_window, qt_theme_applied):
    """Regression: _settings_panel returned the first QScrollArea, and
    child order puts the console's own "ConsoleScroll" first — so the
    step narrating "the settings panel on the left" pointed the cursor at
    the console on the right."""
    from PySide6.QtWidgets import QScrollArea
    from spacr.qt.tutorial.scripts import _console_panel, _settings_panel

    main_window._on_nav_selected(app_key)
    qt_theme_applied.processEvents()
    screen = main_window._screens[app_key]

    panel = _settings_panel(screen)
    console = _console_panel(screen)
    assert isinstance(panel, QScrollArea)
    assert panel in screen.findChildren(QScrollArea)
    assert console is not None and console is screen._console

    # The console really does contain a scroll area that must lose.
    console_scrolls = console.findChildren(QScrollArea)
    assert console_scrolls, "the console no longer has a scroll area"
    assert panel not in console_scrolls
    assert not console.isAncestorOf(panel)

    # And the winner is genuinely the left column.
    panel_x = panel.mapTo(screen, panel.rect().topLeft()).x()
    console_x = console.mapTo(screen, console.rect().topLeft()).x()
    assert panel_x < console_x
    assert panel.width() > 0 and panel.height() > 0


def test_panel_lookups_degrade_to_none(main_window, qt_theme_applied):
    from PySide6.QtWidgets import QWidget
    from spacr.qt.tutorial.scripts import _console_panel, _settings_panel

    assert _settings_panel(None) is None
    assert _console_panel(None) is None
    # A screen with neither yields None rather than raising.
    bare = QWidget()
    assert _settings_panel(bare) is None
    assert _console_panel(bare) is None


def test_tutorial_scratch_is_per_tutorial_and_created(home_in_tmp):
    from spacr.qt.tutorial.scripts import _tutorial_scratch
    a = _tutorial_scratch("mask")
    b = _tutorial_scratch("measure")
    assert Path(a).is_dir() and Path(b).is_dir()
    assert a != b
    assert Path(a) == home_in_tmp / ".spacr" / "tutorial-scratch" / "mask"
    # Idempotent — building the same script twice must not raise.
    assert _tutorial_scratch("mask") == a


# ---------------------------------------------------------------------------
# Whole-script render through the real engine
# ---------------------------------------------------------------------------

def test_engine_drives_a_whole_script_and_completes(main_window,
                                                      home_in_tmp,
                                                      tmp_path,
                                                      monkeypatch):
    """The full narrate → capture → mux → SRT pipeline over the real
    'mask' script and a real MainWindow, at a tiny frame size."""
    import subprocess
    from spacr.qt.tutorial import engine
    from spacr.qt.tutorial.scripts import build_steps
    from tests.qt.test_tutorial_director import FakeNarrator

    monkeypatch.setattr(engine, "VIDEO_SIZE", (240, 160))
    monkeypatch.setattr(engine.subprocess, "run",
                          lambda cmd, **kw: subprocess.CompletedProcess(
                              cmd, 0, b"", b""))

    steps = build_steps("mask", main_window)
    d = engine.Director(main_window, steps, out_dir=tmp_path / "out",
                          narrator=FakeNarrator(seconds=0.05), fps=2)
    workdir = d._workdir
    result = d.render("mask")

    expected_frames = sum(
        max(1, math.ceil((0.05 + s.hold_ms / 1000.0) * 2)) for s in steps)
    assert result.frames == expected_frames
    assert result.duration_s == pytest.approx(
        sum(0.05 + s.hold_ms / 1000.0 for s in steps), abs=1e-3)
    assert result.srt.exists()
    srt_text = result.srt.read_text()
    for step in steps:
        assert step.narration in srt_text
    assert srt_text.count("-->") == len(steps)
    assert not workdir.exists(), "scratch dir should be cleaned up"

    # The script really drove the UI: the mask screen exists and is
    # showing, and the demo it loaded is on disk.
    assert main_window._stack.currentWidget() is main_window._screens["mask"]
    scratch = home_in_tmp / ".spacr" / "tutorial-scratch" / "mask" / "mask"
    assert scratch.is_dir() and any(scratch.rglob("*"))


def test_render_tutorial_boots_a_window_and_returns_paths(tmp_path,
                                                            home_in_tmp,
                                                            monkeypatch,
                                                            qt_theme_applied):
    """render_tutorial() is the public entry point: it must build its own
    MainWindow, run the named script through it and hand back the output
    paths."""
    import subprocess
    from spacr.qt.tutorial import engine
    from tests.qt.test_tutorial_director import FakeNarrator

    monkeypatch.setattr(engine, "VIDEO_SIZE", (240, 160))
    monkeypatch.setattr(engine.subprocess, "run",
                          lambda cmd, **kw: subprocess.CompletedProcess(
                              cmd, 0, b"", b""))

    built = {}

    class TinyNarrator(FakeNarrator):
        def __init__(self, voice_model=None, length_scale=1.0):
            super().__init__(seconds=0.02)
            built["voice_model"] = voice_model
            built["length_scale"] = length_scale

    monkeypatch.setattr(engine, "Narrator", TinyNarrator)

    out = tmp_path / "videos"
    result = engine.render_tutorial("home", out_dir=out,
                                      voice_model=Path("/voices/x.onnx"),
                                      length_scale=0.8)

    assert built == {"voice_model": Path("/voices/x.onnx"),
                       "length_scale": 0.8}
    assert result.mp4 == out / "home.mp4"
    assert result.srt == out / "home.srt"
    assert result.srt.exists()
    assert out.is_dir()
    assert result.frames > 0
    # The home script's five narrations all made it into the sidecar.
    assert result.srt.read_text().count("-->") == 5
    assert "Welcome to spaCR" in result.srt.read_text()


def test_render_tutorial_rejects_an_unknown_app_before_booting(
        tmp_path, monkeypatch, qt_theme_applied):
    """A typo must be reported without paying for a MainWindow — building
    one costs ~10 s."""
    from spacr.qt import app as app_mod
    from spacr.qt.tutorial import engine

    def explode(*a, **kw):
        raise AssertionError("MainWindow must not be built for a bad key")

    monkeypatch.setattr(app_mod, "MainWindow", explode)
    with pytest.raises(ValueError, match="unknown tutorial") as exc:
        engine.render_tutorial("not-a-tutorial", out_dir=tmp_path / "o")
    for name in AVAILABLE_TUTORIALS:
        assert name in str(exc.value)


def test_load_demo_skips_applying_when_the_screen_never_appears(tmp_path):
    """Defensive branch, exercised for real.

    With the real MainWindow, `_on_nav_selected` always registers the
    screen, so the `widget is not None` guard never fires. It exists for
    a screen whose construction failed — modelled here by a window whose
    navigation registers nothing. The demo must still be generated, and
    the apply step must be skipped rather than raising on None.
    """
    from spacr.qt.tutorial.scripts import _load_demo

    class Window:
        DEMO_TARGETS = {"ghost": ("ghost_app", "generate_ghost_demo")}

        def __init__(self, register):
            self._register = register
            self._screens = {}
            self.navigated = []
            self.generated = []
            self.applied = []

        def _run_demo_generator(self, demo_key, dst):
            (Path(dst) / "field_1.tif").write_text("pixels")
            self.generated.append((demo_key, dst))
            return f"layout-of-{demo_key}"

        def _on_nav_selected(self, key):
            self.navigated.append(key)
            if self._register:
                self._screens[key] = f"screen-{key}"

        def _apply_demo_to_screen(self, widget, layout):
            self.applied.append((widget, layout))

    root = tmp_path / "scratch"
    root.mkdir()

    missing = Window(register=False)
    _load_demo(missing, "ghost", str(root))()
    assert (root / "ghost" / "field_1.tif").read_text() == "pixels"
    assert missing.generated == [("ghost", str(root / "ghost"))]
    assert missing.navigated == ["ghost_app"]
    assert missing.applied == [], "nothing to apply to, so nothing applied"

    present = Window(register=True)
    _load_demo(present, "ghost", str(root))()
    assert present.applied == [("screen-ghost_app", "layout-of-ghost")]
