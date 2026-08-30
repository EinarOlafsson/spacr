"""The Regression screen does not build the sweep panel to keep it hidden.

The sweep card starts collapsed behind a toggle, and the panel inside it is a
whole second screen carrying its own results panel -- which builds ELEVEN
pyqtgraph plots. Measured on this screen those eleven were 0.32 s of a 0.88 s
construction: a third of the cost of opening the module, paid by every user,
for a card most runs never open.

Nothing is switched off. The card, the toggle and the panel are unchanged;
the panel is built the first time anything needs it. That is the optimisation
the laptop item asks for rather than the feature removal it calls the fallback.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _regression(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen


def _plot_count(monkeypatch):
    """Count FastPlot constructions while a screen is built."""
    import spacr.qt.widgets.fast_plots as fast_plots

    seen = []
    original = fast_plots.FastPlot.__init__

    def traced(self, *args, **kwargs):
        seen.append(type(self).__name__)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fast_plots.FastPlot, "__init__", traced)
    return seen


def test_the_card_exists_and_the_panel_does_not(qtbot):
    screen = _regression(qtbot)
    assert screen._sweep_card is not None, "the card must still be there"
    assert screen._sweep is not None, "the holder must still be there"
    assert screen._sweep.built() is False, "the panel was built up front again"


def test_opening_it_builds_it(qtbot):
    screen = _regression(qtbot)
    panel = screen._sweep.panel()
    assert panel is not None
    assert screen._sweep.built() is True


def test_it_is_built_only_once(qtbot):
    screen = _regression(qtbot)
    first = screen._sweep.panel()
    assert screen._sweep.panel() is first


def test_the_drop_handler_still_finds_it(qtbot):
    """It resolves the sweep by looking for the score/count pair, so asking
    for either has to build the panel -- a drop onto a card nobody opened
    must work exactly as it did when the panel was built up front."""
    from spacr.qt.dnd_handlers import _sweep_panel

    screen = _regression(qtbot)
    assert screen._sweep.built() is False
    found = _sweep_panel(screen)
    assert found is not None
    assert screen._sweep.built() is True


def test_flipping_the_toggle_builds_it(qtbot):
    """THE REAL PATH. `_on_sweep_switch` seeds the panel from the settings
    form, and asking for `apply_settings` is what builds it -- so a user who
    opens the card gets a working panel and never sees the deferral.

    Asserted through the handler rather than through `show()`: Qt does not
    deliver a show event to a widget whose ancestor is hidden, so a test that
    called `show()` would be asserting something Qt never promised.
    """
    screen = _regression(qtbot)
    assert screen._sweep.built() is False
    screen._on_sweep_switch(True)
    assert screen._sweep.built() is True, (
        "opening the card left the panel unbuilt, so the card is empty")


def test_the_card_becomes_visible_when_it_is_opened(qtbot):
    screen = _regression(qtbot)
    screen._on_sweep_switch(True)
    assert screen._sweep_card.isVisibleTo(screen) is True


def test_closing_it_again_keeps_the_panel(qtbot):
    """Built once, not once per toggle."""
    screen = _regression(qtbot)
    screen._on_sweep_switch(True)
    panel = screen._sweep.panel()
    screen._on_sweep_switch(False)
    screen._on_sweep_switch(True)
    assert screen._sweep.panel() is panel


def test_the_screen_builds_half_as_many_plots(qtbot, monkeypatch):
    """Eleven, not twenty-two: one results panel, not two."""
    seen = _plot_count(monkeypatch)
    _regression(qtbot)
    assert len(seen) > 0, "the screen drew no plots at all -- wrong probe"
    assert len(seen) <= 12, f"{len(seen)} plots built at construction: {seen}"


def test_forcing_the_sweep_builds_the_other_half(qtbot, monkeypatch):
    """Proof the plots were deferred and not deleted."""
    seen = _plot_count(monkeypatch)
    screen = _regression(qtbot)
    deferred = len(seen)
    screen._sweep.panel()
    assert len(seen) > deferred, "opening the sweep built no plots -- it is gone"


def test_a_private_name_does_not_build_it(qtbot):
    """The forwarding must not fire on every attribute Qt probes for."""
    screen = _regression(qtbot)
    with pytest.raises(AttributeError):
        screen._sweep._not_a_real_attribute
    assert screen._sweep.built() is False


def test_the_translation_walk_does_not_build_the_hidden_sweep(qtbot):
    """Optional-method probes are introspection, not a request for the panel.

    The application translates each new screen before it is shown.  Its
    walker asks every widget whether it has ``set_url`` and
    ``retranslate_dynamic_content``.  A catch-all proxy used to turn either
    harmless probe into :meth:`panel`, eagerly constructing the hidden second
    regression screen and all of its plots.
    """
    from spacr.qt.i18n import retranslate_widget_tree

    screen = _regression(qtbot)
    assert screen._sweep.built() is False
    retranslate_widget_tree(screen)
    assert screen._sweep.built() is False
