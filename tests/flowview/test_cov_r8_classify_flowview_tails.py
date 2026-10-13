"""Classify's FlowView box: the paths taken when something is missing.

Almost every one of these is a failure branch on an OPTIONAL panel, and
the rule they all serve is stated in the source twice: an optional
visualisation must never cost Classify. The screen has to open, and keep
working, whether or not FlowView imports, whether or not a collector can
be read, and whether or not the settings column has the shape the
installer expects.

Branches that only run when something is broken are exactly the ones a
suite drifts away from, because nothing is broken in a healthy run.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt.screens import classify
from spacr.qt.screens.app_screen import AppScreen

pytestmark = pytest.mark.qt


def _screen(qtbot, app_key: str = classify.HOST_KEY) -> AppScreen:
    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    return screen


def _section(qtbot):
    screen = _screen(qtbot)
    section = classify.LazyFlowViewSection(screen)
    qtbot.addWidget(section)
    return screen, section


class TestTheCollectorItCollectsWith:

    def test_a_collector_that_cannot_be_read_is_not_reused(self, qtbot,
                                                           monkeypatch):
        """A broken visualisation never reaches Classify.

        `collector.snapshot()` is somebody else's object by the time this
        runs. If asking it for nodes raises, the section builds its own
        preview graph rather than propagating.
        """
        _screen_obj, section = _section(qtbot)

        class _Broken:
            def snapshot(self):
                raise RuntimeError("the collector is in a bad state")

        from spacr.flowview import trace

        monkeypatch.setattr(trace, "get_collector", lambda: _Broken())
        result = section._collector_for_open_panel()
        assert result is not None

    def test_a_live_graph_is_reused_rather_than_rebuilt(self, qtbot,
                                                        monkeypatch):
        """The other side: a collector with nodes is kept as it is.

        Rebuilding it would throw away the run the user is looking at and
        replace it with an empty preview.
        """
        _screen_obj, section = _section(qtbot)

        # imported inside the method, so patch it at its source
        from spacr.flowview import classify_blueprint

        built = []

        def must_not_be_called(*_a, **_k):
            built.append(1)
            raise AssertionError("a live graph was rebuilt")

        monkeypatch.setattr(classify_blueprint, "classify_graph",
                            must_not_be_called)

        class _Snapshot:
            nodes = {"a": object()}

        class _Live:
            def snapshot(self):
                return _Snapshot()

        from spacr.flowview import trace

        live = _Live()
        monkeypatch.setattr(trace, "get_collector", lambda: live)
        section._collector_for_open_panel()
        assert built == [], "a collector with nodes was thrown away"


class TestWhatItSaysWhenThePanelWillNotOpen:

    def test_an_import_error_names_the_missing_qt_extra(self, qtbot):
        section = _section(qtbot)[1]
        section._show_open_error(ImportError("no PySide6 here"))
        assert section._error_label.text()

    def test_a_flowview_panel_that_cannot_be_imported_still_reports(
            self, qtbot, monkeypatch):
        """THE UNCOVERED PAIR.

        The error message itself is looked up from
        `spacr.flowview.panel`. When that import is what failed, asking
        it for the message fails too -- so there is a fallback, and
        without it the error path would raise while reporting an error.
        """
        import builtins

        section = _section(qtbot)[1]
        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "spacr.flowview.panel":
                raise ImportError("flowview is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        section._show_open_error(ImportError("no panel"))
        assert section._error_label.text() == classify.FLOWVIEW_OPEN_ERROR


class TestShowingAndHidingWithNoPanel:

    def test_showing_an_expanded_section_whose_panel_will_not_build(
            self, qtbot, monkeypatch):
        """`showEvent` starts the panel only if there is one.

        `_ensure_panel` returns None when FlowView cannot open, and the
        show handler has to survive that -- a screen that raised out of
        showEvent would not paint at all.
        """
        _screen_obj, section = _section(qtbot)
        monkeypatch.setattr(section, "_ensure_panel", lambda: None)
        monkeypatch.setattr(type(section), "is_expanded", lambda self: True)
        section.show()
        qtbot.waitExposed(section)
        assert section.panel() is None


class TestInstallingTheSectionIntoAnUnexpectedScreen:

    def test_a_screen_that_is_not_classify_gets_nothing(self, qtbot):
        assert classify.install_flowview(_screen(qtbot, "mask")) is None

    def test_a_settings_column_with_no_layout_is_left_alone(self, qtbot,
                                                            monkeypatch):
        """`layout is None` -- there is nowhere to insert, so it stops."""
        screen = _screen(qtbot)
        bare = QWidget()
        qtbot.addWidget(bare)
        assert bare.layout() is None
        monkeypatch.setattr(screen, "_settings_content", bare,
                            raising=False)
        monkeypatch.setattr(screen, "_flowview_section", None, raising=False)
        assert classify.install_flowview(screen) is None

    def test_a_screen_with_no_settings_column_at_all_is_left_alone(
            self, qtbot, monkeypatch):
        screen = _screen(qtbot)
        monkeypatch.setattr(screen, "_settings_content", None, raising=False)
        monkeypatch.setattr(screen, "_flowview_section", None, raising=False)
        assert classify.install_flowview(screen) is None

    def test_an_installer_that_raises_costs_the_box_and_nothing_else(
            self, qtbot, monkeypatch):
        """The broad guard, and the reason the source gives for it.

        An optional UI must not cost Classify. If building the section
        raises for any reason, the screen keeps its settings column and
        simply has no FlowView box.
        """
        screen = _screen(qtbot)
        monkeypatch.setattr(screen, "_flowview_section", None, raising=False)

        # Patch the CONSTRUCTOR, not the name: `install_flowview` does an
        # isinstance() against the class first, and a function there fails
        # with a TypeError before reaching the branch under test.
        def explode(self, *_a, **_k):
            raise RuntimeError("the section will not build")

        monkeypatch.setattr(classify.LazyFlowViewSection, "__init__", explode)
        assert classify.install_flowview(screen) is None
        assert screen._settings_content is not None, (
            "a failed install damaged the settings column"
        )


def test_the_activation_screen_builder_returns_a_widget(qtbot):
    """`_build_activation` is one of the fold builders and had no test.

    It is reached through the fold strip in the running app, which is
    why it was never called directly -- and a builder that returns
    something other than a widget fails at layout time, far from here.
    """
    built = classify._build_activation(None)
    assert isinstance(built, QWidget)
    qtbot.addWidget(built)


def test_a_panel_still_opens_when_the_screen_it_came_from_has_gone(
        qtbot, monkeypatch):
    """The section outliving its screen must not stop the panel opening.

    `_screen_ref` is a weak reference. Once the screen is collected it
    answers None, and the only thing that costs is the QSS re-apply --
    which is decoration for a screen that no longer exists. The panel
    itself is built and kept either way.

    This is the last branch in the module, and it can only be reached by
    the screen dying between construction and first expansion.
    """
    _screen_obj, section = _section(qtbot)

    class _FakePanel(QWidget):
        def __init__(self, _collector, parent=None, auto_start=False,
                     embedded=False):
            super().__init__(parent)

    import spacr.flowview.panel as panel_module

    monkeypatch.setattr(panel_module, "FlowViewPanel", _FakePanel)
    monkeypatch.setattr(section, "_screen_ref", lambda: None)

    applied = []
    monkeypatch.setattr(classify, "ensure_widget_qss_applied",
                        lambda *a, **k: applied.append(a))

    panel = section._ensure_panel()
    assert isinstance(panel, _FakePanel), "the panel was not built"
    assert section.panel() is panel
    assert applied == [], (
        "it re-applied QSS against a screen that has been collected")
    label = section._error_label
    assert label is None or label.text() == "", (
        "a dead screen reference was reported to the user as an error")
