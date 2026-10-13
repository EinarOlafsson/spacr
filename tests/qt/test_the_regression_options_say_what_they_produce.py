"""The regression options name the analysis, not the mechanism.

Three analyses were reachable and none was findable: a reader had to know
that `inference`, `regression_type` and `level` interact, and in which
direction. The maintainer, repeatedly: "a problem i keep coming back to".

The stored VALUES do not change -- every settings file already written goes
on meaning what it meant. What changes is what the dropdown says.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def model(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    return screen._settings_model


def _options(widget):
    return {widget.itemData(i): widget.itemText(i)
            for i in range(widget.count())}


def test_the_stored_values_are_unchanged(model):
    """The whole point of labelling rather than renaming."""
    assert set(_options(model._widgets["level"])) == {"both", "grna", "gene"}
    assert set(_options(model._widgets["inference"])) == {
        "auto", "parametric", "nonparametric"}


def test_each_level_says_what_it_produces(model):
    labels = _options(model._widgets["level"])
    assert "one estimate per guide" in labels["grna"]
    assert "one estimate per gene" in labels["gene"]
    assert "own family" in labels["both"]


def test_each_inference_says_what_it_costs(model):
    """The limit is the reason to pick one: a simultaneous fit needs more
    wells than terms, and a permutation does not."""
    labels = _options(model._widgets["inference"])
    assert "more wells than terms" in labels["parametric"]
    assert "permutation" in labels["nonparametric"]
    assert "any width" in labels["nonparametric"]


def test_level_is_live_where_the_module_opens(model):
    """The default is nonparametric, and `level` was greyed there by a
    regression_type that path never reads -- so the control a reader needs
    to find genes was dead on arrival."""
    assert model._widgets["inference"].currentData() == "nonparametric"
    assert model._widgets["level"].isEnabled() is True


def test_the_section_names_the_third_analysis(model):
    """Gene effects with guide variability is fitted-side only, and a reader
    choosing permutation must be told rather than quietly given guides."""
    from spacr.qt.screens.settings_model import CATEGORY_TOOLTIPS

    help_text = " ".join(str(v) for k, v in CATEGORY_TOOLTIPS.items()
                         if "MODEL" in str(k).upper())
    assert "mixed" in help_text
    assert "variance component" in help_text
    assert "no permutation equivalent" in help_text.lower()


def test_the_cost_of_a_slow_fit_is_said_before_it_is_chosen(model):
    """Instruction 273 section 3.

    The measurement already existed and reached a user only AFTER they had
    started the run -- printed by the console banner, which is after the
    decision. It is on the control that makes the decision now.
    """
    box = model._widgets["regression_type"]
    label = getattr(box, "_spacr_setting_label", None)
    help_text = str((label.property("apiTooltipHtml") or label.toolTip() or "")
                    if label else "") + str(box.toolTip() or "")
    assert "tens of minutes" in help_text
    assert "REML" in help_text, "the shape of the cost is not named"
    assert "ols at level" in help_text, "no faster alternative is offered"


def test_the_box_and_the_banner_cannot_disagree():
    """One source for the measurement, so two hand-written copies cannot
    drift -- and the second one edited is the one nobody believes."""
    import inspect

    from spacr.qt.screens import settings_model

    source = inspect.getsource(settings_model.attach_api_tooltip)
    assert "mixed_cost_note()" in source
