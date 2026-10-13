"""The per-root QSS overlay: installing it, folding it in, taking it off.

A registered widget block is appended to ONE screen's stylesheet rather
than the application's, because `QApplication.setStyleSheet` re-polishes
every live widget and doing that per screen made later module opens
progressively slower. The suffix is remembered on the root so it can be
removed again without disturbing what the screen set itself.

Everything uncovered here is what happens when that bookkeeping meets a
half-built or half-destroyed application: no QApplication at all, a
widget whose C++ half went while Qt was draining its queue, a
preference store that will not answer, and the no-op case where the
stylesheet already says what it should.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import theme as T


pytestmark = pytest.mark.qt

ATTR = T._LOCAL_WIDGET_QSS_ATTRIBUTE


@pytest.fixture()
def registry_sandbox():
    """Restore the block registry, WITHOUT suppressing the installer.

    `test_registration_seams.qss_sandbox` replaces
    `ensure_widget_qss_applied` with a stub, which is right for tests
    about the registry and wrong here -- the installer is the thing
    under test.
    """
    saved = dict(T._WIDGET_QSS)
    try:
        yield
    finally:
        T._WIDGET_QSS.clear()
        T._WIDGET_QSS.update(saved)


class TestReadingTheLivePreferences:
    """`_live_widget_qss_context(app)` -- cached on the app, else read."""

    def test_a_cached_context_on_the_application_is_used_as_it_is(self):
        class _App:
            pass

        app = _App()
        setattr(app, T._WIDGET_QSS_CONTEXT_ATTRIBUTE, ("light", 2.0, 0.5))
        assert T._live_widget_qss_context(app) == ("light", 2.0, 0.5)

    def test_a_cached_value_of_the_wrong_shape_is_ignored(self):
        """A two-tuple is not a context; reading it would unpack wrong."""
        class _App:
            pass

        app = _App()
        setattr(app, T._WIDGET_QSS_CONTEXT_ATTRIBUTE, ("light", 2.0))
        theme, scale, _opacity = T._live_widget_qss_context(app)
        assert isinstance(theme, str) and isinstance(scale, float)

    def test_a_preference_store_that_will_not_answer_falls_back(self,
                                                                monkeypatch):
        """THE GUARD. Theming must not be what stops the GUI starting.

        The fallback is a real, usable context -- dark, unscaled, the
        theme's own scrim -- not None, because every caller composes a
        stylesheet from it immediately.
        """
        import builtins

        class _App:
            pass

        real = builtins.__import__

        def refuse(name, g=None, l=None, fromlist=(), level=0):
            if "preferences" in name:
                raise ImportError("the preference store is unavailable")
            return real(name, g, l, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert T._live_widget_qss_context(_App()) == ("dark", 1.0, None)

    def test_the_ordinary_path_answers_from_the_preferences(self):
        class _App:
            pass

        theme, scale, _opacity = T._live_widget_qss_context(_App())
        assert isinstance(theme, str) and theme
        assert isinstance(scale, float)


class TestFoldingTheSuffixIntoAStylesheet:

    def test_a_root_with_no_overlay_is_returned_unchanged(self, qtbot):
        root = QWidget()
        qtbot.addWidget(root)
        assert T.preserve_widget_qss_overlay(
            root, "QLabel { color: red; }") == "QLabel { color: red; }"

    def test_a_root_with_an_overlay_keeps_it_on_the_end(self, qtbot):
        """Folding it into an assignment the caller is making anyway is
        what avoids a second setStyleSheet and its palette cascade."""
        root = QWidget()
        qtbot.addWidget(root)
        setattr(root, ATTR, "\nQFrame#Late { color: blue; }")
        folded = T.preserve_widget_qss_overlay(root, "QLabel { color: red; }")
        assert folded.startswith("QLabel { color: red; }")
        assert folded.endswith("QFrame#Late { color: blue; }")

    def test_a_non_string_stylesheet_is_coerced(self, qtbot):
        root = QWidget()
        qtbot.addWidget(root)
        assert T.preserve_widget_qss_overlay(root, None) == "None"


class TestClearingTheOverlays:

    def test_with_no_application_it_clears_nothing(self, monkeypatch):
        monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
        assert T.clear_widget_qss_overlays() == 0

    def test_a_widget_with_no_overlay_is_left_alone(self, qtbot):
        root = QWidget()
        qtbot.addWidget(root)
        root.setStyleSheet("QLabel { color: red; }")
        T.clear_widget_qss_overlays()
        assert root.styleSheet() == "QLabel { color: red; }"

    def test_the_owned_suffix_is_removed_and_the_rest_kept(self, qtbot):
        root = QWidget()
        qtbot.addWidget(root)
        suffix = "\nQFrame#Late { color: blue; }"
        root.setStyleSheet("QLabel { color: red; }" + suffix)
        setattr(root, ATTR, suffix)

        assert T.clear_widget_qss_overlays() >= 1
        assert root.styleSheet() == "QLabel { color: red; }"
        assert getattr(root, ATTR) == ""

    def test_a_stylesheet_that_no_longer_ends_in_the_suffix_is_not_cut(
            self, qtbot):
        """Something else re-styled the root after the overlay went on.

        Blindly trimming by length would eat the end of whatever it says
        now, so the suffix is removed only if it is still there -- the
        bookkeeping is cleared either way.
        """
        root = QWidget()
        qtbot.addWidget(root)
        setattr(root, ATTR, "\nQFrame#Late { color: blue; }")
        root.setStyleSheet("QLabel { color: green; }")

        T.clear_widget_qss_overlays()
        assert root.styleSheet() == "QLabel { color: green; }"
        assert getattr(root, ATTR) == ""

    def test_a_widget_deleted_mid_sweep_does_not_stop_the_sweep(self,
                                                                monkeypatch):
        """`except RuntimeError` -- Qt was draining its queue.

        The sweep runs over every live widget, and one of them going away
        underneath it is ordinary during teardown.
        """
        class _Gone:
            def styleSheet(self):        # noqa: N802 - Qt naming
                raise RuntimeError("Internal C++ object already deleted.")

        gone = _Gone()
        setattr(gone, ATTR, "\nQFrame#Late { color: blue; }")

        class _App:
            @staticmethod
            def allWidgets():            # noqa: N802 - Qt naming
                return [gone]

        assert T.clear_widget_qss_overlays(app=_App()) == 0


class TestInstallingLateBlocks:

    def test_with_no_root_it_does_nothing(self):
        assert T.ensure_widget_qss_applied("Anything", root=None) is False

    def test_with_no_application_it_does_nothing(self, qtbot, monkeypatch):
        root = QWidget()
        qtbot.addWidget(root)
        monkeypatch.setattr(QApplication, "instance",
                            staticmethod(lambda: None))
        assert T.ensure_widget_qss_applied("Anything", root=root) is False

    def test_an_application_with_no_stylesheet_yet_does_nothing(self, qtbot,
                                                                monkeypatch):
        """Nothing to append to: the startup pass has not run."""
        root = QWidget()
        qtbot.addWidget(root)

        real_instance = QApplication.instance()

        class _Bare:
            @staticmethod
            def styleSheet():            # noqa: N802 - Qt naming
                return ""

            def __getattr__(self, name):
                return getattr(real_instance, name)

        monkeypatch.setattr(QApplication, "instance",
                            staticmethod(lambda: _Bare()))
        assert T.ensure_widget_qss_applied("Anything", root=root) is False


class TestInstallingTheSameBlocksTwice:
    """The second call must be a no-op, and say so.

    `ensure_widget_qss_applied` exists to avoid a global restyle, so it
    would be self-defeating if calling it again re-assigned the same
    stylesheet: `setStyleSheet` re-polishes the root and everything under
    it, which on a settings screen is several thousand widgets.

    The `desired == current` check is what makes it free. It is also what
    lets callers call it defensively -- on every show, on every palette
    event -- without measuring first.
    """

    def test_the_second_call_changes_nothing_and_returns_false(
            self, qtbot, registry_sandbox):
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        previous = app.styleSheet()
        try:
            T.register_widget_qss(
                "R8Late", lambda palette, opacity:
                "QFrame#R8Late { color: #123456; }")
            # An application sheet that does NOT already carry the block,
            # so the installer has something to append.
            app.setStyleSheet("QWidget { color: black; }")

            root = QWidget()
            qtbot.addWidget(root)

            first = T.ensure_widget_qss_applied("R8Late", root=root)
            assert first is True, "the block was never installed"
            installed = root.styleSheet()
            assert "R8Late" in installed

            second = T.ensure_widget_qss_applied("R8Late", root=root)
            assert second is False, (
                "installing the same blocks again re-assigned the "
                "stylesheet, which re-polishes the whole subtree")
            assert root.styleSheet() == installed
        finally:
            T.unregister_widget_qss("R8Late")
            app.setStyleSheet(previous)

    def test_a_root_that_cannot_be_styled_is_survived(
            self, qtbot, registry_sandbox):
        """`except (AttributeError, RuntimeError)`.

        A root whose C++ half has gone, or an object that is not a widget
        at all, must not take the caller down -- this runs from show
        handlers and palette events.
        """
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        previous = app.styleSheet()
        try:
            T.register_widget_qss(
                "R8Dead", lambda palette, opacity:
                "QFrame#R8Dead { color: #654321; }")
            app.setStyleSheet("QWidget { color: black; }")

            class _Gone:
                def styleSheet(self):    # noqa: N802 - Qt naming
                    raise RuntimeError("Internal C++ object already deleted.")

            assert T.ensure_widget_qss_applied("R8Dead", root=_Gone()) is False
        finally:
            T.unregister_widget_qss("R8Dead")
            app.setStyleSheet(previous)
