"""A fifth Live Preview costs a declaration, not a change to the shared screen.

The four existing previews each cost an arm in
``AppScreen._build_runtime_panel``, two attribute names in a null-out block,
and a row in a toggle table two hundred lines below — which is why there was
never a fifth, and why the two modules that would most obviously benefit
(Cellpose Masks and Plaque Assay, whose entire job is "did the mask come out
right") had none.

The sampling contract is inherited rather than reimplemented: the panels
reached through this registry are the shipped ones, which group a plate from
file names alone and open a bounded reproducible sample of it. Nothing here
enumerates or opens a directory, and one of the tests below pins that.
"""
from __future__ import annotations

import pytest

from spacr.qt.preview_registry import (
    PREVIEWS,
    PreviewSpec,
    install,
    preview_app_keys,
    register_preview,
    unregister_preview,
)


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1280, 860)
    return win


def _screen(window, qtbot, app_key):
    """Open a module and attach whatever it declares.

    ``install`` is called here rather than relied on from
    ``shortcuts._install_window_hooks`` so this file tests the registry
    rather than the wiring, and passes at the commit that adds it. It is
    idempotent, so it is also the assertion that the launch path and this
    one cannot disagree.
    """
    from spacr.qt.settings_search import install as install_search
    window._on_nav_selected(app_key)
    qtbot.wait(50)
    screen = window._screens[app_key]
    install_search(screen)
    install(screen)
    return screen


# ---------------------------------------------------------------------------
# 1. The registry is the answer to "which modules have a preview"
# ---------------------------------------------------------------------------

def test_the_four_shipped_previews_are_declared():
    for key in ("mask", "measure", "timelapse", "motility"):
        assert key in PREVIEWS
        assert PREVIEWS[key].owned_by_screen, (
            f"{key}'s preview is built by AppScreen; declaring it otherwise "
            "would give it a second card")


def test_the_two_new_ones_are_attached_through_the_seam():
    for key in ("cellpose_masks", "analyze_plaques"):
        assert key in PREVIEWS
        assert not PREVIEWS[key].owned_by_screen


def test_declaring_the_same_module_twice_is_refused():
    spec = PreviewSpec(builder="a:b")
    register_preview("_test_preview_module", spec)
    try:
        with pytest.raises(ValueError, match="already registered"):
            register_preview("_test_preview_module", spec)
        register_preview("_test_preview_module", spec, replace=True)
        assert "_test_preview_module" in preview_app_keys()
    finally:
        assert unregister_preview("_test_preview_module")


# ---------------------------------------------------------------------------
# 2. Attaching one
# ---------------------------------------------------------------------------

def test_a_declared_module_gains_a_preview_card(window, qtbot):
    screen = _screen(window, qtbot, "cellpose_masks")
    host = getattr(screen, "_registry_preview", None)
    assert host is not None, "the declared preview was never attached"
    assert host.card.parentWidget() is screen._runtime_wrap


def test_the_card_sits_above_the_run_row(window, qtbot):
    """Above the actions row is the last thing the eye crosses on the way to
    Run; a panel the user has to go and find is a panel nobody opens."""
    screen = _screen(window, qtbot, "cellpose_masks")
    layout = screen._runtime_wrap.layout()
    host = screen._registry_preview
    assert layout.indexOf(host.card) < layout.indexOf(screen._actions_row)


def test_the_card_starts_hidden_behind_a_toggle(window, qtbot):
    screen = _screen(window, qtbot, "cellpose_masks")
    host = screen._registry_preview
    assert not host.card.isVisible()
    assert host.toggle.isCheckable()

    host.toggle.setChecked(True)
    assert not host.card.isHidden()
    host.toggle.setChecked(False)
    assert host.card.isHidden()


def test_the_toggle_lands_in_the_settings_strip(window, qtbot):
    screen = _screen(window, qtbot, "cellpose_masks")
    bar = screen._settings_search
    assert bar.isAncestorOf(screen._registry_preview.toggle)


def test_opening_it_pushes_the_modules_settings_in(window, qtbot):
    """The Mask panel reads `diameter`, `flow_threshold` and `CP_prob`
    straight out of the dict, which is what makes the reuse honest rather
    than approximate."""
    screen = _screen(window, qtbot, "cellpose_masks")
    screen._settings_model.set_value_for_key("diameter", 44)
    host = screen._registry_preview
    assert not host._primed
    host.toggle.setChecked(True)
    assert host._primed
    assert host.panel._diameter.value() == pytest.approx(44)


def test_propagation_is_translated_into_the_modules_own_names(window, qtbot):
    """The panel speaks Mask's per-compartment names. `cell_diameter` means
    nothing to a module that has one object type and calls it `diameter`."""
    screen = _screen(window, qtbot, "cellpose_masks")
    host = screen._registry_preview
    host.on_propagate({
        "cell_diameter": 33.0,
        "cell_FT": 0.6,
        # Whole numbers only: Cellpose Masks declares `CP_prob` with an int
        # default, so `convert_settings_dict_for_gui` gives it an integer
        # spin box and a fractional probability is truncated on the way in.
        # That is the module's own declaration, not something propagation
        # can fix, and the test asserts what the widget can actually hold.
        "cell_CP_prob": -2,
    })
    collected = screen._settings_model.collect()
    assert collected["diameter"] == pytest.approx(33.0)
    assert collected["flow_threshold"] == pytest.approx(0.6)
    assert collected["CP_prob"] == -2


def test_untranslatable_values_are_dropped_not_passed_through(window, qtbot):
    """`set_value_for_key` returns False in silence for a key the module
    does not have, which would leave "propagate" looking like it worked."""
    screen = _screen(window, qtbot, "cellpose_masks")
    before = screen._settings_model.collect()
    screen._registry_preview.on_propagate({"cell_channel": 9})
    assert screen._settings_model.collect() == before


def test_a_module_the_shared_screen_already_serves_gets_no_second_card(
        window, qtbot):
    screen = _screen(window, qtbot, "mask")
    assert getattr(screen, "_registry_preview", None) is None
    assert install(screen) is None
    assert screen._live_preview_card is not None


def test_a_module_with_no_declaration_is_left_alone(window, qtbot):
    screen = _screen(window, qtbot, "regression")
    assert install(screen) is None


def test_installing_twice_attaches_one_card(window, qtbot):
    screen = _screen(window, qtbot, "analyze_plaques")
    host = screen._registry_preview
    assert install(screen) is host
    from spacr.qt.widgets.card import Card
    titles = [c for c in screen._runtime_wrap.findChildren(Card)
              if c is host.card]
    assert len(titles) == 1


# ---------------------------------------------------------------------------
# 3. The sampling contract is inherited, not reimplemented
# ---------------------------------------------------------------------------

def test_the_registry_never_touches_the_filesystem(monkeypatch, window,
                                                     qtbot):
    """Attaching and opening a preview must not enumerate a plate.

    The panels sample a folder from file names alone and cache the result;
    a registry that listed a directory on the way in would put that cost
    back on the GUI thread it was taken off.
    """
    import os
    calls = []
    real_scandir = os.scandir
    real_listdir = os.listdir
    monkeypatch.setattr(
        os, "scandir",
        lambda *a, **k: (calls.append(("scandir", a)), real_scandir(*a, **k))[1])
    monkeypatch.setattr(
        os, "listdir",
        lambda *a, **k: (calls.append(("listdir", a)), real_listdir(*a, **k))[1])

    screen = _screen(window, qtbot, "cellpose_masks")
    screen._registry_preview.toggle.setChecked(True)
    qtbot.wait(20)
    assert not calls, f"the preview walked the filesystem: {calls[:3]}"


def test_the_reused_panel_still_owns_a_bounded_sampler(window, qtbot):
    from spacr.qt.widgets.preview_controls import (
        DEFAULT_MAX_SETS, ImageSetSampler,
    )
    screen = _screen(window, qtbot, "analyze_plaques")
    sampler = screen._registry_preview.panel._sampler
    assert isinstance(sampler, ImageSetSampler)
    assert sampler.sample() == []
    sampler.set_max(DEFAULT_MAX_SETS)


def test_every_declared_builder_resolves():
    """A named builder is only as good as the import behind it."""
    from spacr.qt.preview_registry import _resolve
    for key, spec in PREVIEWS.items():
        assert callable(_resolve(spec.builder)), (
            f"{key} names a preview builder that cannot be imported: "
            f"{spec.builder}")
