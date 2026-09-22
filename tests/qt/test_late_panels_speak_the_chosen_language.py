"""A panel that arrives after the language pass still speaks the language.

``MainWindow`` translates a module screen ONCE, when it builds it. Two
things are parented into the screen afterwards and never meet that pass:
the preview :mod:`spacr.qt.preview_registry` declares for a module, built
and inserted the first time the module is opened, and the toggle it puts on
the settings strip. Measured on a Swedish cold start before this was
watched for, the whole Plaque analysis preview and its toggle sat in
English -- 112 captions on one screen -- beside a settings form that had
been translated correctly.

The screen watches the two hosts a late panel lands in and runs the pass
over the new subtree as it arrives. What is pinned here is the user-facing
result: open a module in Swedish, and the preview that appears is Swedish.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QAbstractButton, QLabel

from spacr.qt.i18n import retranslate_widget_tree, tr
from spacr.qt.screens.app_screen import AppScreen


def _late_preview_app() -> str:
    """The module whose preview the registry still attaches after the pass.

    Plaque analysis was the first such module and the one this was measured
    on; its preview is now built by ``AppScreen`` itself (``owned_by_screen``)
    and so meets the screen's own pass. The late path is still used by every
    preview the registry attaches, so the test follows that path rather
    than a module name.
    """
    from spacr.qt import preview_registry

    late = sorted(key for key, spec in preview_registry.PREVIEWS.items()
                  if not spec.owned_by_screen)
    assert late, "no module attaches its preview late any more"
    return late[0]


PREVIEW_APP = _late_preview_app()


@pytest.fixture()
def swedish_screen(qtbot, monkeypatch):
    """A module screen built and translated exactly as MainWindow does it."""
    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    screen = AppScreen(PREVIEW_APP)
    qtbot.addWidget(screen)
    # `MainWindow._on_nav_selected` runs one pass over a screen it has just
    # built, and nothing runs another. Reproduce that, then attach the
    # preview the way the stack hook does when the module is first shown.
    retranslate_widget_tree(screen)
    return screen


def _install_preview(screen, qtbot):
    from spacr.qt import preview_registry

    host = preview_registry.install(screen)
    assert host is not None, f"{screen.app_key} declares no preview any more"
    # The pass is deferred one turn of the event loop, because a widget can
    # be parented before its own children exist.
    qtbot.wait(20)
    return host


def _captions(widget) -> list:
    texts = [w.text() for w in widget.findChildren(QLabel)]
    texts += [w.text() for w in widget.findChildren(QAbstractButton)]
    return [str(t) for t in texts if str(t).strip()]


def test_the_declared_preview_arrives_translated(swedish_screen, qtbot):
    """The panel installed after the pass is Swedish, not English."""
    host = _install_preview(swedish_screen, qtbot)
    captions = _captions(host.panel)
    assert captions, "the preview panel rendered no captions at all"

    english = [text for text in captions if tr(text, "sv") != text]
    assert not english, (
        f"{len(english)} caption(s) still English in the late preview: "
        f"{english[:5]}")


def test_the_toggle_that_opens_it_is_translated_too(swedish_screen, qtbot):
    """It goes on the settings strip, which is also built after the pass."""
    host = _install_preview(swedish_screen, qtbot)
    assert host.toggle is not None
    assert host.toggle.text() == tr("Live preview", "sv") != "Live preview"


def test_a_panel_added_to_the_runtime_column_later_is_translated(
    swedish_screen, qtbot,
):
    """Not only the registry's preview: anything parented in afterwards.

    The watch is on the host, not on the preview, so a card any other seam
    inserts above the actions row is covered by the same mechanism.
    """
    from PySide6.QtWidgets import QPushButton

    button = QPushButton("Run")
    wrap = swedish_screen._runtime_wrap
    wrap.layout().addWidget(button)
    qtbot.wait(20)
    assert button.text() == tr("Run", "sv") != "Run"


def test_the_pass_leaves_a_user_typed_value_alone(swedish_screen, qtbot):
    """Translating late arrivals must not rewrite data the user entered."""
    from PySide6.QtWidgets import QLineEdit

    field = QLineEdit("/data/my plate")
    swedish_screen._runtime_wrap.layout().addWidget(field)
    qtbot.wait(20)
    assert field.text() == "/data/my plate"
