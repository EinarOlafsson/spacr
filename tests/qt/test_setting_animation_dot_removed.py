"""Neither dot is drawn beside a setting, and the popup one opened is gone.

Every setting label used to carry two coloured dots: a teal one linking to the
API page and a purple one that opened the setting's animation in a window of
its own. The hover tooltip shows that animation inline now, on request, so the
purple dot was 585 marks of clutter in front of a window nothing else needed.
The teal one followed it: a column of dots read as texture rather than as one
affordance per setting, and the API link is in the hover text either way.

Measured on the rendered label, not on a class name: the removal is only real
if no purple ink reaches the screen. The probe below is run against a purple
dot built on purpose first, so it is known to be able to fail.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QWidget

from spacr.qt.screens.settings_model import (
    install_api_tooltips,
    refresh_api_tooltips,
)
from spacr.qt.widgets import animation_zoom as az
from spacr.qt.widgets.dot_link import DotLink
from spacr.qt.widgets.info_link import InfoLink


#: A setting the packaged registry really does have an animation for — the
#: only kind that ever grew the second dot.
ANIMATED_KEY = "n_neighbors"

#: The colour the removed dot painted itself.
PURPLE = (0x9B, 0x00, 0x9B)


def _setting_row(qtbot, key: str):
    root = QWidget()
    qtbot.addWidget(root)
    form = QFormLayout(root)
    label = QLabel(key.replace("_", " ").title(), root)
    field = QSpinBox(root)
    form.addRow(label, field)
    install_api_tooltips(root, "umap", {field: key})
    root.resize(520, 120)
    root.show()
    qtbot.waitExposed(root)
    return root, label, field


def _purple_pixels(widget) -> int:
    """How many pixels of the rendered widget are the removed dot's purple."""
    pixels = az.from_qimage(widget.grab().toImage()).astype(int)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    want = np.array(PURPLE, dtype=int)
    # Deliberately looking for the dot's CORE, not its anti-aliased rim.
    # Subpixel text rendering puts dark violet fringes (~70, 24, 91) on every
    # glyph edge, which a loose tolerance counts as a purple dot; a real dot
    # is 7 pixels across and most of them land on the nominal colour.
    return int(np.count_nonzero(
        (red > 100) & (blue > 100) & (green < 80)
        & (np.abs(red - blue) < 40)
        & (np.abs(pixels - want).max(axis=2) < 60)))


# ---------------------------------------------------------------------------
# 1. No setting label carries the dot
# ---------------------------------------------------------------------------

def test_an_animated_setting_gets_no_dot_of_either_colour(qtbot):
    _root, label, _field = _setting_row(qtbot, ANIMATED_KEY)

    assert label.property("settingHelpLabel"), (
        "the label was never decorated, so counting its dots proves nothing")
    assert "href=" in str(label.property("apiTooltipHtml")), (
        "the API link left with the dot instead of staying in the hover text")
    assert label.parentWidget().findChildren(DotLink) == []
    assert label.parentWidget().findChildren(InfoLink) == []


def test_no_purple_ink_is_rendered_beside_an_animated_setting(qtbot):
    """The probe is proved against a real purple dot before it is trusted."""
    _root, label, _field = _setting_row(qtbot, ANIMATED_KEY)
    assert _purple_pixels(label.parentWidget()) == 0

    control = DotLink(
        tooltip="control",
        colours=("#9B009B", "#D14AD1", "#700070", "#765A76"),
        accessible_description="a purple dot, on purpose",
    )
    qtbot.addWidget(control)
    control.show()
    qtbot.waitExposed(control)
    assert _purple_pixels(control) > 5, (
        "the purple probe cannot see a purple dot it was pointed at")


def test_a_combined_control_row_carries_no_dot_either(qtbot):
    """The label-less branch — a checkbox that is its own row label."""
    from PySide6.QtWidgets import QCheckBox, QVBoxLayout

    root = QWidget()
    qtbot.addWidget(root)
    QVBoxLayout(root)
    box = QCheckBox("Verbose", root)
    root.layout().addWidget(box)
    install_api_tooltips(root, "umap", {box: ANIMATED_KEY})

    assert "href=" in str(box.property("apiTooltipHtml")), (
        "the checkbox was never decorated, so counting its dots proves "
        "nothing")
    assert root.findChildren(DotLink) == []


def test_no_widget_claims_either_dot_role_any_more(qtbot):
    """``refresh_api_tooltips`` used to retranslate both dots' captions."""
    root, label, field = _setting_row(qtbot, ANIMATED_KEY)
    refresh_api_tooltips(root, "sv")

    roles = {w.property("apiTooltipDisplayRole")
             for w in root.findChildren(QWidget)}
    assert "animation-link" not in roles
    assert "api-link" not in roles
    # The help itself was retranslated rather than lost with the dots.
    assert label.property("apiTooltipDisplayRole") == "tooltip"
    assert str(label.property("apiTooltipHtml"))
    assert field.toolTip() == ""


# ---------------------------------------------------------------------------
# 2. Nothing constructs the removed popup
# ---------------------------------------------------------------------------

def test_the_animation_link_module_is_gone():
    assert importlib.util.find_spec("spacr.qt.widgets.animation_link") is None


def test_the_widget_package_no_longer_exports_the_popup():
    from spacr.qt import widgets

    for name in ("AnimationLink", "AnimationPopup", "SettingLinkStack"):
        assert not hasattr(widgets, name), f"{name} is still exported"
        assert name not in widgets.__all__


def test_no_source_file_constructs_or_imports_the_removed_popup():
    """An orphaned constructor call would import-error at runtime, not here.

    ``AnimationLink`` alone is too coarse a needle — the tooltip's own footer
    word is styled through ``QLabel#HoverTooltipAnimationLink`` — so this
    looks for the three things that would actually break: the module import,
    the popup type, and the dot stack being constructed.
    """
    root = Path(__file__).resolve().parents[2] / "spacr"
    needles = (
        "animation_link import",
        "widgets.animation_link",
        "AnimationPopup",
        "SettingLinkStack(",
    )
    offenders = {}
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [needle for needle in needles if needle in text]
        if hits:
            offenders[str(path.relative_to(root))] = hits
    assert offenders == {}, f"still referenced: {offenders}"
