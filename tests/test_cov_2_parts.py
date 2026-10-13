"""Optional arguments of the Home-variant widget kit that no variant passes.

The kit under ``spacr/resources/home/versions/_generators/parts.py`` builds the
thirty review renders. Three of its options are declared and unused by the
current thirty: a primary action in the brand bar, a fixed-width search field,
and a fixed-width start-run panel. An option that is never built is an option
nobody knows is broken, and the next reviewer reaching for one would find out
during a render rather than here.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GENERATORS = os.path.join(REPO_ROOT, "spacr", "resources", "home", "versions",
                          "_generators")


def _load(name: str, module_name: str):
    """Import one generator module under an explicit module name."""
    path = os.path.join(GENERATORS, f"{name}.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kit(qapp):
    """``common`` and ``parts``, loaded under the plain names they expect.

    ``parts`` imports ``common`` by plain name, so both have to occupy those
    entries in :data:`sys.modules` while they load; the originals go back on
    teardown. Depending on ``qapp`` means a QApplication already exists when
    ``common.bootstrap()`` runs, which keeps it a guest -- it must not
    redirect the process-wide ``QSettings`` or restyle an application it does
    not own.
    """
    if not os.path.isdir(GENERATORS):
        pytest.skip("home-screen variant generators not present")
    names = ("common", "parts")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        common = _load("common", "common")
        common.bootstrap()
        parts = _load("parts", "parts")
        yield types.SimpleNamespace(common=common, parts=parts, app=qapp)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def ctx(kit):
    """A dark rendering context, unthemed.

    ``apply_theme()`` is not called: these are structural assertions about
    object names and fixed sizes, which the stylesheet does not decide.
    """
    return kit.common.Ctx(kit.app, "dark")


def _buttons(widget):
    from PySide6.QtWidgets import QPushButton
    return widget.findChildren(QPushButton)


def test_a_primary_action_in_the_brand_bar_is_named_for_the_stylesheet(ctx, kit):
    """The accent styling is applied by object name, so it has to be set.

    ``top_bar`` takes ``(label, primary)`` pairs, and the theme paints
    ``PrimaryButton`` differently from every other button. A primary action
    that did not get the name would render as an ordinary one -- the exact
    failure that is invisible in code review and visible only in the render.
    """
    bar = kit.parts.top_bar(ctx, title="spaCR",
                            actions=(("Run", True), ("Docs", False)))

    buttons = _buttons(bar)
    assert [b.text() for b in buttons] == ["Run", "Docs"]
    assert buttons[0].objectName() == "PrimaryButton"
    assert buttons[1].objectName() != "PrimaryButton"


def test_a_search_field_takes_a_fixed_width_only_when_it_is_asked_for(ctx, kit):
    """``width=0`` means "let the layout decide", not "zero pixels wide".

    The search-first variants put the field in a stretching row; a fixed width
    is for the ones that centre it. Treating the default as a real width would
    collapse the field to nothing in every variant that omits it.
    """
    default = kit.parts.search_box(ctx, "Search apps")
    assert default.minimumWidth() != default.maximumWidth(), \
        "no fixed width was requested, so none is imposed"

    sized = kit.parts.search_box(ctx, "Search apps", width=420)
    assert sized.minimumWidth() == 420
    assert sized.maximumWidth() == 420
    assert sized.placeholderText() == "Search apps"
    assert sized.objectName() == "Search"
    assert kit.parts.search_box(ctx, "x", big=True).objectName() == "SearchBig"


def test_the_start_run_panel_takes_a_fixed_width_only_when_it_is_asked_for(
        ctx, kit):
    """The panel's height is always fixed; its width is the caller's choice.

    It sits beside other cards in some variants and spans the column in
    others. Fixing the width unconditionally would break the spanning
    variants, and never fixing it would break the side-by-side ones.
    """
    default = kit.parts.start_run_panel(ctx)
    assert default.minimumHeight() == default.maximumHeight() == 210
    assert default.minimumWidth() != default.maximumWidth()

    sized = kit.parts.start_run_panel(ctx, width=560, height=240)
    assert sized.minimumWidth() == sized.maximumWidth() == 560
    assert sized.minimumHeight() == sized.maximumHeight() == 240
