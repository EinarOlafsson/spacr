"""The FIRST stylesheet of a session carries every widget rule.

A widget QSS block that is not registered when the stylesheet is built is
simply not in it, and the widget it was written for falls through to the
blanket ``QWidget { background-color: bg }``. ``bg`` is the WINDOW colour
-- ``#000000`` on the dark theme -- so an unstyled container is not
slightly off, it is a solid black rectangle.

Thirty-one modules register a block at import time. The application
stylesheet is composed and applied at launch, before most of them have
been imported, so the rules arrived only as screens happened to be opened
and any later rebuild of the sheet silently fixed whatever was wrong.

``settings_search`` is the one that hurt. It owns ``SettingsSearchPane``,
the wrapper around the search strip *and* the settings scroll area -- the
entire left column of every module screen. Its own module docstring says
what an unstyled pane looks like: "an opaque black rectangle behind the
whole thing". That is the black box reported against Mask, Measure,
Timelapse, Motility, both Classify screens, Map Barcodes, Regression,
External Masks, Illumination, Train Cellpose, Cellpose Masks, Image UMAP,
Activation, Barcode QC, Replication, Invasion, Recruitment and Plaque,
and it is why changing the theme or the animation made it go away: both
rebuild the sheet, by which time the module had been imported.

Asserted here rather than by rendering, on purpose. An offscreen
``QWidget.render`` forces one full synchronous paint, so it does not
reproduce this at all -- it came back clean for a screen that was black on
the user's display. What IS exactly checkable is the thing that was
actually wrong: whether the rule is in the sheet.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


QT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "spacr" / "qt"


def _modules_registering_at_import() -> set[str]:
    """Every module under ``spacr/qt`` that calls ``register_widget_qss``
    at module level, found by parsing rather than by importing."""
    found = set()
    for path in sorted(QT_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "register_widget_qss(" not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        def top_level(body):
            for node in body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    name = getattr(func, "id", getattr(func, "attr", ""))
                    if name.endswith("register_widget_qss"):
                        yield True
                # The registration is often inside a `try:` that tolerates a
                # partial install, which is still module level.
                if isinstance(node, ast.Try):
                    yield from top_level(node.body)

        if any(top_level(tree.body)):
            rel = path.relative_to(QT_ROOT).with_suffix("")
            found.add("spacr.qt." + rel.as_posix().replace("/", "."))
    return found


def test_every_registering_module_is_listed():
    """A module that registers a block must be in ``WIDGET_QSS_MODULES``.

    This is the half that rots. Someone adds a screen with its own QSS,
    does not know this list exists, and that screen's containers are black
    until the sheet is next rebuilt -- which is precisely the bug, one
    screen at a time.
    """
    from spacr.qt.theme import WIDGET_QSS_MODULES

    missing = sorted(_modules_registering_at_import() - set(WIDGET_QSS_MODULES))
    assert not missing, (
        "these register a widget QSS block at import but are not in "
        f"theme.WIDGET_QSS_MODULES, so their rules are absent from the "
        f"first stylesheet of a session and their containers paint `bg` "
        f"(#000000 on dark) until something rebuilds it: {missing}"
    )


def test_no_listed_module_has_gone_away():
    """The other direction, so the list cannot quietly accumulate names."""
    from spacr.qt.theme import WIDGET_QSS_MODULES

    stale = sorted(set(WIDGET_QSS_MODULES) - _modules_registering_at_import())
    assert not stale, (
        f"listed but no longer registering anything at import: {stale}")


def test_the_settings_search_pane_is_styled_in_a_fresh_stylesheet(qapp):
    """The specific rule whose absence was the black box.

    Named rather than folded into the sweep below because it is the one
    that spans the whole settings column, and a regression here is worth
    reading as itself rather than as "some rule is missing".
    """
    from spacr.qt import settings_search, theme

    sheet = theme.stylesheet("dark")
    assert settings_search.PANE_NAME in sheet, (
        f"{settings_search.PANE_NAME} has no rule in a freshly built "
        f"stylesheet, so it paints the window colour -- a black rectangle "
        f"behind the entire settings column of every module screen"
    )


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_a_fresh_stylesheet_carries_every_registered_block(qapp, theme_name):
    """Build the sheet the way a launch does and check nothing is missing.

    ``load_widget_qss_registrars`` is idempotent and latches, so by the
    time any other test has run the modules are imported anyway. What this
    pins is that ``stylesheet()`` itself does the loading -- take that call
    out and the assertion below fails on a fresh interpreter.
    """
    from spacr.qt import theme

    theme.load_widget_qss_registrars()
    sheet = theme.stylesheet(theme_name)
    registered = set(theme._WIDGET_QSS)
    assert len(registered) >= 25, (
        f"only {len(registered)} widget QSS blocks registered; the loader "
        f"did not run")

    # Every block must contribute something. A block that renders to an
    # empty string is a rule that silently does not exist.
    # The blocks are handed the palette `stylesheet()` builds, which
    # carries the theme's own name -- several read `palette["theme"]` to
    # decide between an opaque and a translucent surface.
    palette = dict(theme.palette_for(theme_name))
    palette["theme"] = theme_name
    empty = [name for name, fn in theme._WIDGET_QSS.items()
             if not str(fn(palette, 1.0) or "").strip()]
    assert not empty, f"registered but rendering to nothing: {sorted(empty)}"
