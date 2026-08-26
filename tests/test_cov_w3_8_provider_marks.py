"""The provider marks: the SVG path parser, and what each state paints.

The parser is driven with real ``d`` strings and the resulting geometry is
measured -- bounding boxes, arc lengths, end points -- rather than checked
for "did not raise". The widget half renders through ``QWidget.grab()`` and
counts pixels, so a state that quietly paints nothing cannot pass.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor, QHoverEvent, QImage, QPainterPath,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets.provider_marks import (  # noqa: E402
    BRAND, GITHUB_MARK, MARKS, ProviderMark, _parse_svg_path, _svg_arc,
    github_path, mark_for,
)

pytestmark = pytest.mark.qt


def _pixels(widget) -> np.ndarray:
    """The widget's own render as an (H, W, 3) RGB array."""
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
    raw = np.frombuffer(memoryview(image.constBits()), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine() // 4, 4)
    return rows[:, :image.width(), :3][:, :, ::-1]


# --------------------------------------------------------------------------
# The SVG path parser
# --------------------------------------------------------------------------

def test_absolute_lines_land_where_the_string_says():
    path = _parse_svg_path("M 10 10 L 30 10 L 30 40 Z")
    bounds = path.boundingRect()
    assert (bounds.left(), bounds.top()) == (10.0, 10.0)
    assert (bounds.width(), bounds.height()) == (20.0, 30.0)
    assert path.currentPosition() == QPointF(10.0, 10.0)


def test_a_relative_move_is_followed_by_relative_lines():
    """``m`` switches the implicit command to ``l``, not to ``L``."""
    path = _parse_svg_path("m 5 5 10 0 0 10")
    bounds = path.boundingRect()
    assert (bounds.left(), bounds.top()) == (5.0, 5.0)
    assert (bounds.right(), bounds.bottom()) == (15.0, 15.0)


def test_horizontal_and_vertical_commands_move_one_axis_only():
    absolute = _parse_svg_path("M 0 0 H 20 V 30")
    assert absolute.currentPosition() == QPointF(20.0, 30.0)
    relative = _parse_svg_path("M 4 4 h 6 v -3")
    assert relative.currentPosition() == QPointF(10.0, 1.0)


def test_relative_and_absolute_cubics_describe_the_same_curve():
    absolute = _parse_svg_path("M 0 0 C 0 10 10 10 10 0")
    relative = _parse_svg_path("M 0 0 c 0 10 10 10 10 0")
    assert absolute.boundingRect() == relative.boundingRect()
    assert absolute.currentPosition() == QPointF(10.0, 0.0)


def test_a_smooth_cubic_reflects_the_previous_control_point():
    """``S`` without a preceding curve falls back to the current point."""
    reflected = _parse_svg_path("M 0 0 C 0 10 10 10 10 0 S 20 -10 20 0")
    assert reflected.currentPosition() == QPointF(20.0, 0.0)
    assert reflected.boundingRect().top() < 0.0
    cold_start = _parse_svg_path("M 0 0 S 10 10 10 0")
    assert cold_start.currentPosition() == QPointF(10.0, 0.0)
    relative = _parse_svg_path("M 0 0 C 0 10 10 10 10 0 s 10 -10 10 0")
    assert relative.currentPosition() == QPointF(20.0, 0.0)


def test_a_degenerate_arc_is_drawn_as_the_straight_line_it_is():
    path = QPainterPath()
    path.moveTo(QPointF(0.0, 0.0))
    _svg_arc(path, QPointF(0.0, 0.0), QPointF(10.0, 0.0), 0.0, 5.0,
             0.0, 0, 1)
    assert path.currentPosition() == QPointF(10.0, 0.0)
    assert path.length() == pytest.approx(10.0, abs=1e-6)


def test_a_half_circle_arc_bulges_by_its_radius():
    path = _parse_svg_path("M 0 0 A 5 5 0 0 1 10 0")
    bounds = path.boundingRect()
    assert bounds.width() == pytest.approx(10.0, abs=0.05)
    assert bounds.height() == pytest.approx(5.0, abs=0.05)
    assert path.length() == pytest.approx(math.pi * 5.0, rel=0.01)


def test_the_sweep_flag_puts_the_arc_on_the_other_side():
    """Sweep 1 is the positive-angle direction, which in Qt's y-down box
    bulges above the chord; sweep 0 mirrors it below."""
    above = _parse_svg_path("M 0 0 A 5 5 0 0 1 10 0").boundingRect()
    below = _parse_svg_path("M 0 0 A 5 5 0 0 0 10 0").boundingRect()
    assert above.top() < 0.0 and above.bottom() == pytest.approx(0.0, abs=0.05)
    assert below.bottom() > 0.0 and below.top() == pytest.approx(0.0, abs=0.05)


def test_the_large_arc_flag_takes_the_long_way_round():
    short = _parse_svg_path("M 0 0 A 8 8 0 0 1 10 0").length()
    long = _parse_svg_path("M 0 0 A 8 8 0 1 1 10 0").length()
    assert long > short * 2.0


def test_radii_too_small_for_the_chord_are_grown_to_fit():
    """SVG says undersized radii are scaled up; the result is a half circle."""
    path = _parse_svg_path("M 0 0 A 1 1 0 0 1 10 0")
    bounds = path.boundingRect()
    assert bounds.width() == pytest.approx(10.0, abs=0.05)
    assert bounds.height() == pytest.approx(5.0, abs=0.05)


def test_a_rotated_arc_is_not_the_unrotated_one():
    plain = _parse_svg_path("M 0 0 A 8 4 0 0 1 10 0").boundingRect()
    turned = _parse_svg_path("M 0 0 A 8 4 40 0 1 10 0").boundingRect()
    assert turned.height() != pytest.approx(plain.height(), abs=0.2)


def test_a_relative_arc_ends_where_the_offset_says():
    path = _parse_svg_path("M 4 4 a 5 5 0 0 1 10 0")
    assert path.currentPosition().x() == pytest.approx(14.0, abs=1e-6)
    assert path.currentPosition().y() == pytest.approx(4.0, abs=1e-6)


def test_a_command_the_parser_cannot_draw_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unsupported command"):
        _parse_svg_path("M 0 0 T 10 10")


# --------------------------------------------------------------------------
# The marks themselves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", sorted(MARKS))
def test_every_mark_fills_the_box_it_is_given(code):
    box = QRectF(10.0, 20.0, 60.0, 60.0)
    path = mark_for(code, box)
    assert path is not None and not path.isEmpty()
    bounds = path.boundingRect()
    assert box.contains(bounds.adjusted(0.5, 0.5, -0.5, -0.5))
    assert bounds.width() > box.width() * 0.5
    assert bounds.height() > box.height() * 0.5


def test_an_unknown_provider_has_no_mark_rather_than_a_wrong_one():
    assert mark_for("copilot", QRectF(0, 0, 10, 10)) is None
    assert mark_for("", QRectF(0, 0, 10, 10)) is None
    assert mark_for(None, QRectF(0, 0, 10, 10)) is None


def test_the_octocat_keeps_its_proportions_in_a_wide_box():
    natural = _parse_svg_path(GITHUB_MARK).boundingRect()
    wide = github_path(QRectF(0.0, 0.0, 200.0, 50.0)).boundingRect()
    assert wide.height() == pytest.approx(50.0, abs=0.5)
    assert wide.width() / wide.height() == pytest.approx(
        natural.width() / natural.height(), rel=1e-6)
    assert wide.center().x() == pytest.approx(100.0, abs=0.5)
    assert wide.center().y() == pytest.approx(25.0, abs=0.5)


def test_an_empty_box_gets_the_mark_at_its_own_scale():
    unscaled = github_path(QRectF()).boundingRect()
    parsed = _parse_svg_path(GITHUB_MARK).boundingRect()
    assert unscaled.width() == pytest.approx(parsed.width(), abs=1e-6)
    assert unscaled.height() == pytest.approx(parsed.height(), abs=1e-6)


# --------------------------------------------------------------------------
# The widget
# --------------------------------------------------------------------------

def _mark(qtbot, code="gpt", **kwargs):
    widget = ProviderMark(code, code.upper(), **kwargs)
    qtbot.addWidget(widget)
    widget.resize(widget.sizeHint())
    return widget


def test_the_default_state_comes_from_availability(qtbot):
    ready = _mark(qtbot, "claude")
    missing = _mark(qtbot, "gpt", available=False)
    assert ready.status == ProviderMark.READY
    assert missing.status == ProviderMark.NOT_INSTALLED
    signed_out = _mark(qtbot, "gemini", status=ProviderMark.SIGNED_OUT)
    assert signed_out.status == ProviderMark.SIGNED_OUT


def test_a_left_click_chooses_the_provider_and_a_right_click_does_not(qtbot):
    widget = _mark(qtbot, "gemini")
    picked = []
    widget.chosen.connect(picked.append)
    qtbot.mouseClick(widget, Qt.LeftButton, pos=QPoint(20, 20))
    assert picked == ["gemini"]
    qtbot.mouseClick(widget, Qt.RightButton, pos=QPoint(20, 20))
    assert picked == ["gemini"]


def test_an_uninstalled_provider_is_still_choosable(qtbot):
    widget = _mark(qtbot, "gpt", available=False)
    picked = []
    widget.chosen.connect(picked.append)
    qtbot.mouseClick(widget, Qt.LeftButton, pos=QPoint(20, 20))
    assert picked == ["gpt"]


def test_hovering_lights_the_mark_and_leaving_puts_it_out(qtbot):
    widget = _mark(qtbot, "claude")
    assert widget._colours()[1].alpha() == 0
    QApplication.sendEvent(widget, QHoverEvent(
        QEvent.Type.HoverEnter, QPointF(5, 5), QPointF(-1, -1)))
    assert widget._hovered is True
    assert widget._colours()[1].alpha() == 22
    QApplication.sendEvent(widget, QHoverEvent(
        QEvent.Type.HoverLeave, QPointF(-1, -1), QPointF(5, 5)))
    assert widget._hovered is False
    assert widget._colours()[1].alpha() == 0


def test_the_chosen_mark_is_the_brightest_of_the_three_states(qtbot):
    widget = _mark(qtbot, "claude")
    resting = widget._colours()[1].alpha()
    widget._hovered = True
    hovered = widget._colours()[1].alpha()
    widget.set_chosen(True)
    chosen = widget._colours()[1].alpha()
    assert resting == 0 < hovered < chosen
    assert widget.is_chosen()
    widget.set_chosen(False)
    assert not widget.is_chosen()


def test_a_ready_assistant_takes_its_brand_colour(qtbot):
    for code in ("claude", "gpt", "gemini"):
        widget = _mark(qtbot, code)
        ink = widget._colours()[0]
        assert (ink.red(), ink.green(), ink.blue()) == (
            QColor(BRAND[code]).red(), QColor(BRAND[code]).green(),
            QColor(BRAND[code]).blue())


def test_a_signed_in_github_takes_the_themes_ink_not_its_own_near_black(qtbot):
    from spacr.qt.theme import active_palette

    widget = _mark(qtbot, "github")
    ink = widget._colours()[0]
    assert ink.name() == QColor(active_palette()["fg"]).name()


def test_an_unavailable_mark_gains_a_brand_halo_only_on_hover(qtbot):
    widget = _mark(qtbot, "gpt", available=False)
    ink, halo = widget._colours()
    assert ink.alpha() == 190
    assert halo.alpha() == 0
    widget._hovered = True
    _ink, halo = widget._colours()
    assert halo.alpha() == 30
    assert halo.name() == QColor(BRAND["gpt"]).name()


def test_an_unavailable_mark_is_drawn_in_the_themes_muted_ink(qtbot):
    """The state the mark is meant to show is 'can I use this one'.

    Signed out, GitHub must not be the colour it is signed in: its ready ink
    IS the palette foreground, so a muted state that fell back to the
    foreground would leave a ten-step alpha difference as the entire answer.
    The palette spells its secondary ink ``fg_muted``; ``muted`` is not a key
    any spaCR palette defines, and reading it took that fallback every time.
    """
    from spacr.qt.theme import active_palette

    palette = active_palette()
    missing = _mark(qtbot, "github", available=False)
    ready = _mark(qtbot, "github")
    assert missing._colours()[0].name() == QColor(palette["fg_muted"]).name()
    assert missing._colours()[0].name() != ready._colours()[0].name()


def test_the_mark_paints_ink_and_its_name(qtbot):
    widget = _mark(qtbot, "claude")
    widget.resize(120, 120)
    painted = _pixels(widget)
    assert len(np.unique(painted.reshape(-1, 3), axis=0)) > 3


def test_choosing_a_mark_paints_more_than_leaving_it_alone(qtbot):
    widget = _mark(qtbot, "gemini")
    widget.resize(120, 120)
    resting = _pixels(widget).copy()
    widget.set_chosen(True)
    chosen = _pixels(widget)
    assert not np.array_equal(resting, chosen)
    assert int((chosen != resting).any(axis=-1).sum()) > 50


def test_a_provider_that_needs_setting_up_says_so_under_its_name(qtbot):
    ready = _mark(qtbot, "gpt")
    ready.resize(120, 120)
    quiet = _pixels(ready)[80:, :].copy()
    needs_install = _mark(qtbot, "gpt", available=False)
    needs_install.resize(120, 120)
    loud = _pixels(needs_install)[80:, :]
    assert not np.array_equal(quiet, loud)

    sign_in = _mark(qtbot, "gpt", status=ProviderMark.SIGNED_OUT)
    sign_in.resize(120, 120)
    assert not np.array_equal(_pixels(sign_in)[80:, :], quiet)


def test_an_unknown_provider_still_paints_its_name(qtbot):
    """No mark is drawn, but the control is still labelled and clickable."""
    widget = _mark(qtbot, "copilot")
    widget.resize(120, 120)
    painted = _pixels(widget)
    assert len(np.unique(painted.reshape(-1, 3), axis=0)) > 1


def test_the_size_hint_leaves_room_for_the_mark_and_the_label(qtbot):
    widget = _mark(qtbot, "claude")
    assert widget.sizeHint().width() >= widget.minimumSize().width()
    assert widget.sizeHint().height() >= widget.minimumSize().height()
