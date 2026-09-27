"""An attached preview card still lands above the console when the screen
refuses the optional parts, and a panel handed over directly is never built.

``_attach`` builds a registered preview's card, puts it in the runtime
splitter directly above the console, and asks the screen to adopt it as a
runtime pane; a screen whose adoption raises still gets the card and its
toggle. A splitter that cannot say where the console is falls back to the
Run row. A panel assigned to the host replaces the one still waiting to be
built, and a late panel whose translation hook raises is left as it is.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from spacr.qt import preview_registry as pr  # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part will not answer")


@pytest.fixture
def builders(monkeypatch):
    """A module of preview builders the registry can resolve by name."""
    module = types.ModuleType("probe_preview_builders")
    built = []

    def build(screen, panel_later=False):
        card = QWidget()
        QVBoxLayout(card)
        card.setObjectName("ProbePreviewCard")
        built.append(panel_later)
        return None, card

    def fill(host, card):
        panel = QLabel("filled", card)
        card.layout().addWidget(panel)
        return panel

    module.build = build
    module.fill = fill
    monkeypatch.setitem(sys.modules, "probe_preview_builders", module)
    module.built = built
    return module


def _screen(qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    splitter = QSplitter(Qt.Vertical, screen)
    figures = QWidget()
    console = QWidget()
    splitter.addWidget(figures)
    splitter.addWidget(console)
    screen._runtime_splitter = splitter
    screen._console_wrap = console
    return screen


def test_a_screen_that_will_not_adopt_the_card_still_shows_it_above_the_console(
        qapp, qtbot, builders):
    screen = _screen(qtbot)
    screen.adopt_runtime_pane = _boom
    spec = pr.PreviewSpec(builder="probe_preview_builders:build",
                          fill="probe_preview_builders:fill",
                          title="Probe preview")

    host = pr._attach(screen, "probe", spec)

    assert host is not None
    assert builders.built == [True]
    splitter = screen._runtime_splitter
    card = screen.findChild(QWidget, "ProbePreviewCard")
    assert splitter.indexOf(card) == splitter.indexOf(screen._console_wrap) - 1
    assert card.isHidden()
    assert host.toggle.text() == "Probe preview"
    assert not host.panel_is_built()


class _SplitterGone:
    def indexOf(self, _widget):
        raise RuntimeError("Internal C++ object already deleted.")


def test_a_splitter_that_cannot_find_the_console_falls_back_to_the_run_row(
        qapp):
    screen = types.SimpleNamespace(_runtime_splitter=_SplitterGone(),
                                   _console_wrap=object())

    assert pr._insert_above_console(screen, QWidget()) is False


def test_a_panel_handed_to_the_host_replaces_the_one_waiting(qapp, qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    filled = []
    host = pr._PreviewHost(screen, pr.PreviewSpec(builder="x:y"), None,
                           QWidget(screen),
                           fill=lambda h, c: filled.append(1))
    assert not host.panel_is_built()
    given = QLabel("given", screen)

    host.panel = given

    assert host.panel_is_built()
    assert host.panel is given
    assert filled == []


def test_a_late_panel_whose_translation_fails_is_left_as_it_is(qapp):
    panel = QLabel("Preview")
    translated = []
    pr._dress_a_late_panel(
        types.SimpleNamespace(_translate_a_late_part=translated.append), panel)
    assert translated == [panel]

    pr._dress_a_late_panel(
        types.SimpleNamespace(_translate_a_late_part=_boom), panel)
    assert panel.text() == "Preview"
