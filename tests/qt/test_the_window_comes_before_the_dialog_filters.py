"""The three dialog filters go on after the window, not before it.

`detach_all_dialogs`, `install_glass_everywhere` and
`install_dialog_translation` each install an APPLICATION-WIDE event filter:
a Python callable Qt runs for every event delivered to every object in the
process. Building the main window delivers tens of thousands of them, and
all three filters do the same thing with every one -- ask whether the
object is a QDialog and answer no. That is a large and measurable share of
the time between typing `spacr` and seeing a window, spent on a question
whose answer cannot change anything until a dialog exists.

NOTHING IS DEFERRED PAST THE POINT IT COULD MATTER. A dialog can only be
opened by the event loop, and `win.show()` is still before it. The one
exception is the first-run setup screen, which opens inside `launch` -- and
:class:`TestTheSetupScreenNeedsNoneOfThem` is why that screen loses nothing.

The saving is asserted here as a FACT rather than as a duration -- the
filters are absent while the window is built, present before the loop --
because a duration measured on a shared machine is a coin flip, while the
absence is what makes the window cheaper and it names the regression if it
comes back.

`launch` no longer imports matplotlib to name a backend either, which the
last test in this file holds it to.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# The installer itself
# ---------------------------------------------------------------------------
def _forget_the_filters(app) -> None:
    """Put the three filters back to "never installed" for this process."""
    from spacr.qt import dialogs
    from spacr.qt.widgets import glass

    glass.uninstall_glass_everywhere(app)
    if dialogs._DETACHER is not None:
        try:
            app.removeEventFilter(dialogs._DETACHER)
        except Exception:                                    # noqa: BLE001
            pass
    dialogs._DETACHER = None
    dialogs._DETACHED_APP = None
    existing = getattr(app, "_spacr_dialog_i18n_filter", None)
    if existing is not None:
        try:
            app.removeEventFilter(existing)
        except Exception:                                    # noqa: BLE001
            pass
        app._spacr_dialog_i18n_filter = None


def _what_is_installed(app) -> dict:
    from spacr.qt import dialogs
    from spacr.qt.widgets import glass

    return {
        "detach_all_dialogs": dialogs._DETACHER is not None,
        "install_glass_everywhere": glass._INSTALLED is not None,
        "install_dialog_translation":
            getattr(app, "_spacr_dialog_i18n_filter", None) is not None,
    }


@pytest.fixture
def clean_filters(qapp):
    """A QApplication with none of the three filters on it.

    PUT BACK WHATEVER WAS THERE. The application is shared for the whole
    session and the suite runs in a random order, so a file that leaves the
    filters off decides how every dialog examined after it behaves -- which
    is the reason `uninstall_glass_everywhere` exists, applied to all three.
    """
    import importlib

    from spacr.qt.app import _DIALOG_FILTERS

    was_there = _what_is_installed(qapp)
    # Preserve the real installers before the test gets a chance to
    # monkeypatch one of them.  A pre-existing filter must be restored during
    # teardown even when the test deliberately makes its public installer
    # raise; otherwise fixture finalization calls the test double and turns a
    # passing assertion into an order-dependent teardown error.
    installers = {
        function_name: getattr(importlib.import_module(module_name),
                               function_name)
        for module_name, function_name in _DIALOG_FILTERS
    }
    _forget_the_filters(qapp)
    try:
        yield qapp
    finally:
        _forget_the_filters(qapp)
        for _module_name, function_name in _DIALOG_FILTERS:
            if was_there.get(function_name):
                installers[function_name](qapp)


def test_the_installer_puts_all_three_on(clean_filters):
    from spacr.qt.app import install_the_dialog_filters

    assert _what_is_installed(clean_filters) == {
        "detach_all_dialogs": False,
        "install_glass_everywhere": False,
        "install_dialog_translation": False,
    }

    named = install_the_dialog_filters(clean_filters)

    assert set(named) == {"detach_all_dialogs", "install_glass_everywhere",
                          "install_dialog_translation"}
    assert all(_what_is_installed(clean_filters).values())


def test_installing_twice_leaves_one_filter_of_each_kind(clean_filters):
    """`launch` calls it once, but a second QApplication in one process --
    which the test suite makes -- must not end up with six filters."""
    from spacr.qt import dialogs
    from spacr.qt.app import install_the_dialog_filters
    from spacr.qt.widgets import glass

    install_the_dialog_filters(clean_filters)
    first = (dialogs._DETACHER, glass._INSTALLED,
             clean_filters._spacr_dialog_i18n_filter)

    install_the_dialog_filters(clean_filters)

    assert (dialogs._DETACHER, glass._INSTALLED,
            clean_filters._spacr_dialog_i18n_filter) == first


def test_one_filter_that_will_not_install_costs_only_itself(
        clean_filters, monkeypatch):
    """A launch is not worth losing to a filter."""
    from spacr.qt import dialogs
    from spacr.qt.app import install_the_dialog_filters

    def _explode(app):
        raise RuntimeError("no detaching today")

    monkeypatch.setattr(dialogs, "detach_all_dialogs", _explode)

    named = install_the_dialog_filters(clean_filters)

    assert "detach_all_dialogs" not in named
    standing = _what_is_installed(clean_filters)
    assert standing["install_glass_everywhere"] is True
    assert standing["install_dialog_translation"] is True


def test_the_filters_that_moved_are_the_dialog_ones_and_only_those(
        clean_filters):
    """The Qt catalogs are NOT in the deferred set: `install_qt_translations`
    is read while the window's own menus and message boxes are built, so it
    stays where it was."""
    from spacr.qt.app import _DIALOG_FILTERS

    assert [name for _module, name in _DIALOG_FILTERS] == [
        "detach_all_dialogs",
        "install_glass_everywhere",
        "install_dialog_translation",
    ]


def test_a_dialog_opened_afterwards_still_gets_the_card(clean_filters):
    """Deferring must not cost the thing the filter exists for."""
    from PySide6.QtWidgets import QDialog

    from spacr.qt.app import install_the_dialog_filters
    from spacr.qt.widgets.glass import GLASSED

    install_the_dialog_filters(clean_filters)

    dialog = QDialog()
    try:
        dialog.show()
        clean_filters.processEvents()
        assert dialog.property(GLASSED), (
            "a dialog opened after the window was shown lost its card")
    finally:
        dialog.close()
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# The one dialog that opens before the event loop
# ---------------------------------------------------------------------------
class TestTheSetupScreenNeedsNoneOfThem:
    """The first-run slides are built inside `launch`, before `win.show()`.

    They are the only dialog that opens while the three filters are not yet
    installed. Each thing a filter would have done to them is checked to be
    something they already do for themselves, and then the whole screen is
    built both ways and compared.
    """

    @pytest.fixture
    def slides(self, qapp):
        from spacr.qt.widgets.setup_slides import SetupSlides

        screen = SetupSlides(None)
        yield screen
        screen.close()
        screen.deleteLater()

    def test_the_glass_filter_would_have_left_it_alone(self, slides):
        """It builds its own card, and a second one covered the first one's
        buttons -- so `wants_glass` says no whether the filter is there or
        not."""
        from spacr.qt.widgets.glass import wants_glass

        assert wants_glass(slides) is False

    def test_it_detaches_itself(self, slides):
        """What `detach_all_dialogs` would have given it: a frameless window
        of its own rather than one bolted to a parent."""
        from PySide6.QtCore import Qt

        assert slides.parent() is None
        assert bool(slides.windowFlags()
                    & Qt.WindowType.FramelessWindowHint)

    def test_it_comes_out_the_same_with_the_filters_and_without(
            self, clean_filters):
        """The whole claim, measured rather than reasoned about.

        The screen is built and shown twice in one process -- once with the
        three filters on the application and once with none of them -- and
        the two are compared on everything a filter could have changed: the
        card, the resizing pass, the window flags and the size it opens at.

        A PLAIN DIALOG IS BUILT ALONGSIDE IT EACH TIME, because "nothing
        changed" is also what a test proves when the filters were never
        really installed. The control has to change, or the comparison is
        worth nothing.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QAbstractScrollArea, QDialog,
                                       QLineEdit, QVBoxLayout)

        from spacr.qt import dialogs
        from spacr.qt.app import install_the_dialog_filters
        from spacr.qt.widgets.glass import GLASSED
        from spacr.qt.widgets.setup_slides import SetupSlides

        def look(filters_on):
            if filters_on:
                install_the_dialog_filters(clean_filters)
            screen = SetupSlides(None)
            control = QDialog()
            QVBoxLayout(control).addWidget(QLineEdit())
            try:
                screen.show()
                control.show()
                clean_filters.processEvents()
                return {
                    "glassed": bool(screen.property(GLASSED)),
                    "resized": bool(screen.property(dialogs.RESIZABLE)),
                    "frameless": bool(screen.windowFlags()
                                      & Qt.WindowType.FramelessWindowHint),
                    "scroll areas": len(
                        screen.findChildren(QAbstractScrollArea)),
                    "size": (screen.width(), screen.height()),
                }, {
                    "glassed": bool(control.property(GLASSED)),
                    "resized": bool(control.property(dialogs.RESIZABLE)),
                }
            finally:
                for widget in (screen, control):
                    widget.close()
                    widget.deleteLater()
                clean_filters.processEvents()

        bare, bare_control = look(False)
        _forget_the_filters(clean_filters)
        filtered, filtered_control = look(True)

        assert bare_control == {"glassed": False, "resized": False}
        assert filtered_control == {"glassed": True, "resized": True}, (
            "the control dialog was untouched, so this comparison proves "
            "nothing about the filters")
        assert bare == filtered

    def test_it_retranslates_itself(self, slides, monkeypatch):
        """What `install_dialog_translation` would have given it: the same
        catalog walk over the same tree, which it runs itself."""
        from spacr.qt import i18n

        walked = []
        monkeypatch.setattr(i18n, "retranslate_widget_tree",
                            lambda widget: walked.append(widget))

        slides.retranslate()

        assert walked == [slides]


# ---------------------------------------------------------------------------
# The launch itself, cold, in a process of its own
# ---------------------------------------------------------------------------
#: Runs the REAL entry point and stops it at the event loop, recording what
#: was installed when the window construction started and what was installed
#: by the time `exec` was reached. In a child process because `launch` builds
#: a QApplication of its own, and because the matplotlib question is only
#: meaningful in an interpreter that has not already imported it.
_PROBE = r'''
import json, os, sys
import PySide6.QtWidgets as W

seen = {}


def _look(when):
    from PySide6.QtWidgets import QApplication
    from spacr.qt import dialogs
    from spacr.qt.widgets import glass

    app = QApplication.instance()
    seen[when] = {
        "detach_all_dialogs": dialogs._DETACHER is not None,
        "install_glass_everywhere": glass._INSTALLED is not None,
        "install_dialog_translation":
            getattr(app, "_spacr_dialog_i18n_filter", None) is not None,
    }


def _stop(self, *a, **k):
    _look("at the event loop")
    seen["matplotlib imported"] = "matplotlib" in sys.modules
    seen["MPLBACKEND"] = os.environ.get("MPLBACKEND")
    return 0


W.QApplication.exec = _stop
W.QApplication.exec_ = _stop

import spacr.qt
import spacr.qt.app as app_module
import spacr.qt.preferences as preferences

# HOW MANY TIMES THE THEME IS RESOLVED AND SET ON THE APPLICATION. Twice:
# once before the window is built, once to fold in the QSS blocks the
# window's own modules registered on their way in. A third if the setup
# screen actually asked something -- `pretend the setup screen opened` is
# the argument that makes it do so.
_real_apply = preferences.apply_preferences_to_app
seen["preferences applied"] = 0


def _counted(app):
    seen["preferences applied"] += 1
    return _real_apply(app)


preferences.apply_preferences_to_app = _counted

# AND HOW MANY BLOCKS ASKED FOR ONE. Counting both is what tells the two
# apart: a restyle per registering module and a restyle for all of them
# together look identical if you only count the restyles.
import spacr.qt.theme as _theme

_real_register = _theme.register_widget_qss
seen["blocks registered as the window is built"] = 0
_registering = {"window": False}


def _counted_register(name, fn, replace=False):
    if _registering["window"]:
        seen["blocks registered as the window is built"] += 1
    return _real_register(name, fn, replace=replace)


_theme.register_widget_qss = _counted_register

if "pretend the setup screen opened" in sys.argv:
    import spacr.qt.widgets.setup_slides as slides

    slides.open_setup_if_needed = lambda parent=None: object()

_real_init = app_module.MainWindow.__init__


def _init(self, *args, **kwargs):
    _look("as the window is built")
    _registering["window"] = True
    try:
        return _real_init(self, *args, **kwargs)
    finally:
        _registering["window"] = False


app_module.MainWindow.__init__ = _init
seen["exit"] = spacr.qt.run([])
print("PROBE_JSON" + json.dumps(seen))
'''


def _cold_launch(*arguments):
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.pop("MPLBACKEND", None)
    done = subprocess.run([sys.executable, "-c", _PROBE, *arguments],
                          capture_output=True, text=True, timeout=600,
                          env=environment)
    line = next((l for l in done.stdout.splitlines()
                 if l.startswith("PROBE_JSON")), "")
    if not line:
        pytest.fail("the cold launch produced no reading:\n"
                    + "\n".join((done.stderr or "").splitlines()[-12:]))
    return json.loads(line[len("PROBE_JSON"):])


@pytest.fixture(scope="module")
def cold_launch():
    return _cold_launch()


@pytest.mark.qt
def test_the_window_is_built_with_no_dialog_filter_on_the_application(
        cold_launch):
    """The measurement this change exists for, asserted as a fact rather
    than as a duration: a duration on a shared machine is a coin flip, and
    the filter being absent is what makes the window cheaper."""
    assert cold_launch["as the window is built"] == {
        "detach_all_dialogs": False,
        "install_glass_everywhere": False,
        "install_dialog_translation": False,
    }


@pytest.mark.qt
def test_all_three_are_on_before_the_event_loop_starts(cold_launch):
    """Deferred, not dropped. Nothing may reach `exec` without them."""
    assert cold_launch["at the event loop"] == {
        "detach_all_dialogs": True,
        "install_glass_everywhere": True,
        "install_dialog_translation": True,
    }


@pytest.mark.qt
def test_a_launch_does_not_import_matplotlib_to_choose_its_backend(
        cold_launch):
    """`matplotlib.use` needs a matplotlib; MPLBACKEND does not, and
    matplotlib reads it for itself whenever it does load."""
    assert cold_launch["MPLBACKEND"] == "Agg"
    assert cold_launch["matplotlib imported"] is False


@pytest.mark.qt
def test_the_launch_still_reaches_the_event_loop(cold_launch):
    """The other three assertions would all pass on a launch that crashed
    before it got anywhere."""
    assert cold_launch["exit"] == 0


@pytest.mark.qt
def test_the_theme_is_resolved_twice_and_not_once_per_screen_module(
        cold_launch):
    """The restyle count must not grow with the number of screens imported.

    Two are owed. One before the window is built, so its widgets are
    constructed against the right palette and font metrics. One after, to
    fold in the QSS blocks the window's own modules registered on their way
    in -- those blocks did not exist when the first sheet was composed.

    A third `apply_preferences_to_app` exists to take the answers the setup
    screen collected, and on every launch after the first there are none.

    What this pins is the SHAPE, not the number. `register_widget_qss` used
    to re-apply the whole application stylesheet the moment a block was
    registered, which is right for one screen arriving alone and quadratic
    for a run of them: building the window imports four such modules, so a
    cold launch composed the sheet five times and threw four away. Counting
    the blocks alongside the restyles is what tells the two arrangements
    apart -- they are indistinguishable if you only count restyles.
    """
    registered = cold_launch["blocks registered as the window is built"]

    assert registered >= 2, (
        "no module registered a block while the window was built, so this "
        "measurement cannot say anything about batching them")
    assert cold_launch["preferences applied"] == 2, (
        f"{registered} blocks were registered while the window was built "
        f"and the stylesheet was composed "
        f"{cold_launch['preferences applied']} times; a restyle per "
        f"registering module is the fault this guards")


@pytest.mark.qt
def test_the_answers_are_still_applied_when_the_screen_did_open():
    """And the launch that DOES ask still builds its window from what it
    was told -- otherwise the saving above would be a bug."""
    asked = _cold_launch("pretend the setup screen opened")

    assert asked["preferences applied"] == 3
    assert asked["exit"] == 0
