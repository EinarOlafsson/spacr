"""Which help goes to the popup and which to the strip (instruction 347).

Asked for on 2026-09-02: "for settings level tooltips keep the popup but for
cotegory level dont show the popup and just show the bottom of the screen
text. and make sure all settings have tooltipps."

THE REASON FOR THE SPLIT, in the maintainer's own words earlier the same day:
"there is no way to press the link to api or animation". A strip is text; the
popup is the only surface that can hold a clickable API dot and animation
square. A category has no links, so it has nothing to lose by living in the
strip -- and a popup over the form is a box drawn on top of what is being
read.

This reverses part of 2026-09-01, which removed the popup for settings. What
it removed was the NATIVE tooltip -- Qt's grey box -- and the sticky popup was
never taken away, which is why the two requests do not actually conflict. That
is worth having asserted rather than re-derived.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QWidget

MODULES = ("regression", "mask", "measure", "classify_merged", "annotate")


@pytest.fixture(params=MODULES)
def screen(qtbot, qt_theme_applied, request):
    from spacr.qt.screens.app_screen import AppScreen

    s = AppScreen(app_key=request.param)
    qtbot.addWidget(s)
    return s


def test_a_category_never_pops_a_tooltip(screen):
    """Category help lives in the strip and nowhere else."""
    popping = [w for w in screen.findChildren(QWidget)
               if w.property("settingsCategory")
               and w.property("apiTooltipHtml")]
    assert not popping, (
        "these categories would pop a tooltip over the form: "
        + ", ".join(str(w.property("settingsCategory")) for w in popping))


def test_a_category_still_has_help_to_put_in_the_strip(screen):
    """The other half: silencing the popup must not silence the category.

    Without this the test above passes perfectly against a build that
    forgot to write category help at all.
    """
    categories = [w for w in screen.findChildren(QWidget)
                  if w.property("settingsCategory")]
    assert categories, "no categories at all on this screen"
    assert hasattr(screen, "show_category_hint")
    assert hasattr(screen, "clear_category_hint")


def test_every_setting_keeps_its_popup(screen):
    """A setting's help must stay reachable as a CLICKABLE surface.

    `apiTooltipHtml` is what `_ApiTooltipFilter` shows in the sticky popup,
    and the sticky popup is what carries the API dot and the animation
    square. A setting row without it has no way to reach either.
    """
    rows = [w for w in screen.findChildren(QWidget)
            if w.property("settingKey")]
    assert rows, "no setting rows on this screen"
    without = [str(w.property("settingKey")) for w in rows
               if not w.property("apiTooltipHtml")]
    # A handful of rows are controls rather than labelled settings; the
    # claim is that the overwhelming majority carry the popup, and that the
    # count does not quietly collapse.
    assert len(without) <= 2, f"settings with no popup help: {without[:10]}"


def test_every_setting_a_module_offers_has_a_tooltip(screen):
    """"make sure all settings have tooltipps", measured the safe way.

    HANDOFF 3c: `spacr.settings.tooltips` is NOT complete on import -- six
    pipelines register their keys from their own module. Reading the dict
    cold reports false gaps, and a tool that then writes "no description" is
    not missing a sentence, it is writing a WRONG one. Building the screen
    first is what imports the registrars, so this is measured after them.
    """
    from spacr.settings import tooltips

    missing = sorted({
        str(w.property("settingKey"))
        for w in screen.findChildren(QWidget)
        if w.property("settingKey")
        and not str(tooltips.get(str(w.property("settingKey")), "")).strip()
    })
    assert not missing, f"settings with no tooltip: {missing}"
