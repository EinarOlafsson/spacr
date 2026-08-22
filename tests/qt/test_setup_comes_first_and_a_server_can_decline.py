"""The setup screen opens before the main window, and a server skips it.

    "the first time spacr runs this should come before the main window. and
     there should be a terminal version that skipps it, for server
     compatibility."

WHY THE ORDER MATTERS AND IS NOT COSMETIC. The questions the screen asks
are the language, the theme and the font scale -- the three things the main
window is BUILT from. Asked afterwards, the user watched a window in the
wrong language appear, answered, and then watched it restyle itself
underneath them. Asked first, the window is built once, correctly.

WHY THE SKIP MATTERS. The screen is modal, and it is now the first thing a
launch draws. A batch job on a server that inherits a profile which has
never answered would block on an invisible modal dialog until somebody
killed it, and the only symptom would be a run that never started.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# The order
# ---------------------------------------------------------------------------
def _launch_source():
    from spacr.qt.app import launch

    return inspect.getsource(launch)


def test_the_setup_screen_is_opened_before_the_main_window_is_built():
    """READ FROM THE SOURCE, because the alternative is running `launch`,
    and `launch` builds a QApplication, installs a crash handler and opens
    a real main window. The requirement is literally an ordering of two
    statements, so the two statements are what is checked."""
    source = _launch_source()
    setup = source.index("open_setup_if_needed(")
    window = source.index("MainWindow(initial_app=")
    assert setup < window, "the setup screen still opens after the window"


def test_nothing_opens_the_setup_screen_a_second_time_after_the_window():
    """The move is a MOVE, not a copy: two calls would ask twice on the
    first launch of a version."""
    assert _launch_source().count("open_setup_if_needed(") == 1


def test_the_answers_are_applied_before_the_window_is_built():
    """A language chosen in the screen has to reach the window that is
    about to be built, or the screen has asked for nothing."""
    source = _launch_source()
    applied = source.rindex("apply_preferences_to_app(app)")
    window = source.index("MainWindow(initial_app=")
    assert applied < window


def test_the_screen_is_opened_with_no_parent():
    """There is no window yet to parent it to; passing the one that does
    not exist is how this reorder would fail silently."""
    assert "open_setup_if_needed(None)" in _launch_source()


# ---------------------------------------------------------------------------
# A launch can decline
# ---------------------------------------------------------------------------
class TestDeclining:
    @pytest.mark.parametrize("flag", ["--no-setup", "--skip-setup",
                                      "--headless-setup"])
    def test_the_flag_is_recognised(self, flag):
        from spacr.qt.setup_screen import take_the_setup_flags

        kept, asked = take_the_setup_flags([flag])
        assert asked is True
        assert kept == []

    def test_the_flag_is_consumed_not_ignored(self):
        """`launch` reads argv[0] as the module to open into, so a flag
        left in place would be looked up as a module name and open
        nothing."""
        from spacr.qt.setup_screen import take_the_setup_flags

        kept, asked = take_the_setup_flags(["--no-setup", "classify"])
        assert kept == ["classify"]
        assert asked is True

    def test_an_ordinary_launch_keeps_its_argument(self):
        from spacr.qt.setup_screen import take_the_setup_flags

        assert take_the_setup_flags(["classify"]) == (["classify"], False)

    def test_no_arguments_at_all(self):
        from spacr.qt.setup_screen import take_the_setup_flags

        assert take_the_setup_flags([]) == ([], False)
        assert take_the_setup_flags(None) == ([], False)

    def test_launch_takes_the_flags_before_it_reads_the_module_name(self):
        source = _launch_source()
        assert source.index("take_the_setup_flags(") < source.index(
            "initial_app = argv[0]")

    @pytest.mark.parametrize("said", ["1", "true", "TRUE", "yes", "on"])
    def test_the_environment_variable_says_yes(self, said):
        from spacr.qt.setup_screen import skipped_on_purpose

        assert skipped_on_purpose({"SPACR_NO_SETUP": said}) is True

    @pytest.mark.parametrize("said", ["0", "false", "no", "off"])
    def test_the_environment_variable_says_no_and_is_believed(self, said):
        """An explicit no OUTRANKS the platform guess -- somebody running
        offscreen deliberately to configure a profile has said what they
        want, and a guess should not overrule it."""
        from spacr.qt.setup_screen import skipped_on_purpose

        assert skipped_on_purpose({"SPACR_NO_SETUP": said,
                                   "QT_QPA_PLATFORM": "offscreen"}) is False

    @pytest.mark.parametrize("platform", ["offscreen", "minimal", "vnc"])
    def test_a_platform_with_nobody_looking_declines_by_itself(self,
                                                               platform):
        from spacr.qt.setup_screen import skipped_on_purpose

        assert skipped_on_purpose({"QT_QPA_PLATFORM": platform}) is True

    @pytest.mark.parametrize("platform", ["xcb", "wayland", "cocoa", ""])
    def test_a_real_display_does_not_decline(self, platform):
        from spacr.qt.setup_screen import skipped_on_purpose

        assert skipped_on_purpose({"QT_QPA_PLATFORM": platform}) is False


# ---------------------------------------------------------------------------
# The two questions are kept apart
# ---------------------------------------------------------------------------
class TestWhoDecides:
    def test_being_due_and_being_able_to_ask_are_different_questions(self):
        """`should_open` answers "has this profile answered this version",
        and nothing else. Folding the platform into it would make every
        headless test in this suite unable to see the screen at all --
        which is how this was first written, and it took four of them
        down."""
        from spacr.qt.setup_screen import should_open

        source = inspect.getsource(should_open)
        assert "skipped_on_purpose" not in source

    def test_the_opener_asks_both(self):
        from spacr.qt.widgets.setup_slides import open_setup_if_needed

        source = inspect.getsource(open_setup_if_needed)
        assert "skipped_on_purpose" in source
        assert "should_open" in source

    def test_a_declining_launch_gets_no_dialog_even_when_it_is_due(self,
                                                                   tmp_path,
                                                                   monkeypatch):
        """The one that matters: due, and still not shown."""
        from spacr.qt import setup_screen
        from spacr.qt.widgets import setup_slides

        monkeypatch.setattr(setup_screen, "should_open", lambda *a: True)
        monkeypatch.setattr(setup_slides, "SetupSlides", _never_built)
        monkeypatch.setenv("SPACR_NO_SETUP", "1")
        assert setup_slides.open_setup_if_needed(None) is None


class TestTheTerminalCommand:
    """`spacr-server` -- the same GUI with the screen never offered."""

    def test_it_is_installed_as_its_own_command(self):
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        entries = (root / "setup.py").read_text()
        assert re.search(r"spacr-server\s*=\s*spacr\.qt:run_without_setup",
                         entries)

    def test_it_declines_on_the_callers_behalf(self):
        import inspect

        from spacr.qt import run_without_setup

        assert "--no-setup" in inspect.getsource(run_without_setup)

    def test_it_still_forwards_the_module_to_open_into(self):
        """`spacr-server mask` has to reach the mask screen; a wrapper that
        dropped the argument would open the home screen instead."""
        seen = {}

        import spacr.qt as package

        original = package.run
        package.run = lambda argv=None: seen.setdefault("argv", argv) or 0
        try:
            package.run_without_setup(["mask"])
        finally:
            package.run = original
        assert seen["argv"] == ["--no-setup", "mask"]


def _never_built(*_args, **_kwargs):        # pragma: no cover - the point
    raise AssertionError("the setup screen was built on a declining launch")
