"""The layout of a setting's hover tooltip: explanation left, animation right.

Four things were asked for and all four are geometry, so all four are
measured here rather than described:

* the animation is to the RIGHT of the text, not above or below it;
* the text starts level with the TOP of the animation, not centred on it;
* the text column is exactly as WIDE as the square animation;
* and a preference turns the animation off entirely.

The zoom that makes the animation worth showing at this size is measured in
``test_setting_animation_zoom.py``; the end-to-end check that the frames
which reach the screen are the zoomed ones is here, because it is the widget
that has to put them there.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel

from spacr.qt import preferences as prefs
from spacr.qt.widgets import animation_zoom as az
from spacr.qt.widgets.hover_tooltip import HoverTooltip


HTML = (
    "<b>Cell diameter</b> <i>(int)</i><br>"
    "Expected cell diameter in pixels. Cellpose rescales each image so that "
    "objects match the scale its model was trained at, so a wrong value here "
    "quietly degrades every downstream measurement.<br>"
    "<a href='https://example.invalid/'>Open spaCR API documentation</a>"
)

#: A setting the packaged registry really does have an animation for.
ANIMATED_KEY = "cell_diameter"


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)``, and that
    constructor resolves to the NATIVE location whatever ``setPath`` says —
    a fixture that assumed otherwise once silently erased the real store.
    Replacing the accessor is the only reliable isolation, and the assertion
    below refuses to run if it ever stops working.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def tooltip(qtbot):
    """A fresh popup.

    Deliberately not ``HoverTooltip.instance()``: the singleton outlives the
    test and every other Qt test in the session shares it.
    """
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    return popup


def _anchor(qtbot, key: str = ANIMATED_KEY) -> QLabel:
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    if key:
        label.setProperty("settingKey", key)
    return label


# ---------------------------------------------------------------------------
# The four layout requirements
# ---------------------------------------------------------------------------

def test_the_animation_sits_to_the_right_of_the_text(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)

    text = tooltip.text_label()
    view = tooltip.animation_view()
    assert view.isVisible()
    assert view.x() >= text.x() + text.width(), (
        f"animation at x={view.x()} is not right of text ending at "
        f"x={text.x() + text.width()}")
    # Right of, and on the same row — not wrapped underneath.
    assert view.y() < text.y() + text.height()


def test_the_text_box_starts_level_with_the_top_of_the_animation(
        tooltip, qtbot):
    """Both in the POPUP's coordinates.

    The prose lives inside a text column now, so ``text.y()`` is an offset
    within that column and comparing it with the animation's ``y`` would be
    comparing two different origins — a test that passes by accident.
    """
    from PySide6.QtCore import QPoint

    tooltip.show_for(_anchor(qtbot), HTML)

    text = tooltip.text_label()
    view = tooltip.animation_view()
    text_top = text.mapTo(tooltip, QPoint(0, 0)).y()
    view_top = view.mapTo(tooltip, QPoint(0, 0)).y()
    assert text_top == view_top, (
        "the text is not top-aligned with the animation "
        f"(text y={text_top}, animation y={view_top})")
    assert view.height() > text.height(), (
        "the prose happens to fill the whole square, so this test cannot "
        "tell top alignment from centring")


def test_the_first_line_of_prose_is_at_the_top_of_the_square(tooltip, qtbot):
    """Where the ink lands, not where the box does.

    Matching widget geometry is not enough on its own: drop the layout's
    ``AlignTop`` and the label stretches to the full height of the square
    with its own default ``AlignVCenter`` still in force — top-aligned by
    geometry, centred to the eye. So this measures the rendered pixels.
    """
    tooltip.show_for(_anchor(qtbot), HTML)
    view = tooltip.animation_view()
    painted = az.from_qimage(tooltip.grab().toImage())

    column = painted[:, :view.x()]            # the text column only
    background = np.median(column.reshape(-1, 3), axis=0)
    ink = (np.abs(column.astype(int) - background).max(axis=2) > 40)
    rows = np.nonzero(ink.any(axis=1))[0]
    assert rows.size, "no text was rendered at all"

    # One line of 12 px type, plus the frame's own top margin.
    assert rows[0] - view.y() < 24, (
        f"the first line of text starts {rows[0] - view.y()} px below the "
        f"top of the animation; it should be level with it")


def test_the_text_column_is_exactly_as_wide_as_the_animation(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)

    text = tooltip.text_label()
    view = tooltip.animation_view()
    assert text.width() == view.width() == HoverTooltip.ANIMATION_SIZE
    assert view.width() == view.height(), "the animation box is not square"


def test_the_preference_off_removes_the_animation(tooltip, qtbot):
    prefs.set_setting_animations_enabled(False)
    tooltip.show_for(_anchor(qtbot), HTML)

    view = tooltip.animation_view()
    assert tooltip.animation() is None
    assert not view.isVisible()
    assert view.frame_count() == 0, "frames were decoded for a hidden panel"
    assert not view.is_playing()
    # And the text is no longer squeezed into half a popup.
    assert tooltip.text_label().maximumWidth() == HoverTooltip.TEXT_WIDTH


# ---------------------------------------------------------------------------
# The preference, in use
# ---------------------------------------------------------------------------

def test_the_preference_defaults_to_on():
    assert prefs.DEFAULT_SETTING_ANIMATIONS is True
    assert prefs.get_setting_animations_enabled() is True


def test_the_preference_is_read_on_every_hover(tooltip, qtbot):
    """The popup is a singleton that outlives the Preferences dialog.

    Anything cached here would keep animating until the app was restarted,
    which is exactly the complaint the preference exists to answer.
    """
    anchor = _anchor(qtbot)
    tooltip.show_for(anchor, HTML)
    assert tooltip.animation() is not None

    prefs.set_setting_animations_enabled(False)
    tooltip.show_for(anchor, HTML)
    assert tooltip.animation() is None

    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(anchor, HTML)
    assert tooltip.animation() is not None


def test_the_preference_survives_a_round_trip():
    prefs.set_setting_animations_enabled(False)
    assert prefs.get_setting_animations_enabled() is False
    prefs.set_setting_animations_enabled(True)
    assert prefs.get_setting_animations_enabled() is True


def test_the_preferences_dialog_offers_the_toggle(qtbot, monkeypatch):
    """A preference nobody can reach is not a preference."""
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
    prefs.set_setting_animations_enabled(False)

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    toggle = dialog.findChild(QWidget, "SettingAnimationsEnabled")
    assert toggle is not None, "no control for the setting-animation switch"
    assert toggle.isChecked() is False, "the dialog ignored the stored value"

    toggle.setChecked(True)
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs.get_setting_animations_enabled() is True


# ---------------------------------------------------------------------------
# What actually reaches the screen
# ---------------------------------------------------------------------------

def test_the_frames_on_screen_are_the_zoomed_ones(tooltip, qtbot):
    """End to end: measure the pixmap the label is showing.

    The zoom is verified against the arrays in its own module's tests; this
    one closes the loop by measuring what the widget put on screen, so a
    later refactor cannot leave the zoom working and the widget bypassing it.
    """
    tooltip.show_for(_anchor(qtbot), HTML)
    view = tooltip.animation_view()
    pixmap = view.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() == pixmap.height() == HoverTooltip.ANIMATION_SIZE

    frame = az.from_qimage(pixmap.toImage())
    # One frame, not the union, so it can only be smaller than the target —
    # never larger. `cell_diameter` draws its whole shape in every frame.
    measured = az.content_extent([frame])
    assert az.MIN_FILL <= measured <= az.MAX_FILL, (
        f"the displayed frame covers {measured:.1%} of the square")


def test_the_animation_plays_and_stops_with_the_popup(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)
    view = tooltip.animation_view()
    assert view.frame_count() > 1
    assert view.is_playing()

    tooltip.hide()
    assert not view.is_playing(), (
        "the singleton kept swapping pixmaps into a hidden label")

    tooltip.show()
    assert view.is_playing()


def test_re_hovering_the_same_setting_does_not_restart_the_animation(
        tooltip, qtbot):
    anchor = _anchor(qtbot)
    tooltip.show_for(anchor, HTML)
    view = tooltip.animation_view()
    first = view.slug()
    frames = view.frame_count()

    tooltip.show_for(anchor, HTML)
    assert view.slug() == first
    assert view.frame_count() == frames


# ---------------------------------------------------------------------------
# Everything that is not a setting
# ---------------------------------------------------------------------------

def test_a_tooltip_without_a_setting_key_is_text_only(tooltip, qtbot):
    """Section headers and tiles share this popup and have no animation."""
    tooltip.show_for(_anchor(qtbot, key=""), "<b>Some section</b>")

    assert tooltip.animation() is None
    assert not tooltip.animation_view().isVisible()
    assert tooltip.text_label().maximumWidth() == HoverTooltip.TEXT_WIDTH


def test_a_setting_with_no_packaged_animation_is_text_only(tooltip, qtbot):
    from spacr.setting_animations import animation_for_setting

    key = "src"
    assert animation_for_setting(key) is None, (
        f"{key} gained an animation; pick another unmapped key")
    tooltip.show_for(_anchor(qtbot, key=key), HTML)
    assert tooltip.animation() is None
    assert not tooltip.animation_view().isVisible()


def test_a_real_decorated_settings_form_gets_the_animation(tooltip, qtbot):
    """The seam: nothing passes the animation in, so the anchor must carry it.

    ``install_api_tooltips`` is what actually puts hover help on a settings
    panel, and it hands the popup only ``(label, html)``. If it ever stopped
    tagging the label with ``settingKey`` the tooltips would still work and
    silently lose every animation, which is why this asserts through that
    function rather than through a hand-made anchor.
    """
    from PySide6.QtWidgets import QFormLayout, QSpinBox, QWidget

    from spacr.qt.screens.settings_model import install_api_tooltips

    owner = QWidget()
    qtbot.addWidget(owner)
    form = QFormLayout(owner)
    label = QLabel("Cell diameter")
    field = QSpinBox()
    field.setProperty("settingKey", ANIMATED_KEY)
    form.addRow(label, field)
    install_api_tooltips(owner, "mask")

    assert label.property("settingKey") == ANIMATED_KEY
    tooltip.show_for(label, str(label.property("apiTooltipHtml")))
    assert tooltip.animation() is not None
    assert tooltip.animation().slug == ANIMATED_KEY
    assert tooltip.animation_view().isVisible()


def test_an_explicit_animation_argument_wins_over_the_anchor(tooltip, qtbot):
    """Callers keep the option of saying which animation, or none."""
    tooltip.show_for(_anchor(qtbot), HTML, None)
    assert tooltip.animation() is None
    assert not tooltip.animation_view().isVisible()


def test_the_two_argument_call_still_works(tooltip, qtbot):
    """Both existing call sites pass exactly ``(anchor, html)``."""
    tooltip.show_for(_anchor(qtbot, key=""), "<b>Plain</b>")
    assert tooltip.isVisible()
    assert tooltip.text_label().text() == "<b>Plain</b>"


def test_an_empty_body_shows_nothing(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), "")
    assert not tooltip.isVisible()


def test_a_dead_anchor_does_not_take_the_event_loop_down(tooltip, qtbot):
    """The popup holds a plain reference to a widget it does not own.

    Surviving the dead anchor is half of it; the other half is that the popup
    then does what it was called to do — drops the stale reference and hides.
    A ``_maybe_hide`` that caught the RuntimeError and returned early would
    have left the tooltip on screen for the rest of the session and still
    passed the old "must not raise" version of this test.
    """
    import shiboken6

    anchor = _anchor(qtbot)
    tooltip.show_for(anchor, HTML)
    assert tooltip.isVisible()

    # `deleteLater()` + `del` did NOT reproduce the bug: the deferred delete
    # needs an event-loop turn, so the C++ object was still alive and
    # `underMouse()` answered normally. Destroy the C++ half outright and keep
    # the Python wrapper — which is exactly the state a module switch inside
    # the hide delay leaves the singleton holding.
    anchor.setParent(None)
    shiboken6.delete(anchor)
    assert not shiboken6.isValid(anchor), "the anchor is not actually dead"

    tooltip._maybe_hide()
    assert not tooltip.isVisible()
    assert tooltip._anchor is None, "the dead anchor was kept"


def test_the_text_wraps_to_the_narrow_column_without_clipping(tooltip, qtbot):
    """A 220-pixel column makes the popup tall; it must not truncate."""
    tooltip.show_for(_anchor(qtbot), HTML)
    text = tooltip.text_label()
    assert text.height() >= text.heightForWidth(HoverTooltip.ANIMATION_SIZE)
    assert tooltip.height() >= tooltip.animation_view().height()


def test_the_popup_is_wide_enough_for_both_columns(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.width() >= 2 * HoverTooltip.ANIMATION_SIZE


def test_frames_are_real_pictures_not_blank(tooltip, qtbot):
    """A black square would satisfy every geometric assertion above."""
    tooltip.show_for(_anchor(qtbot), HTML)
    frame = az.from_qimage(tooltip.animation_view().pixmap().toImage())
    assert np.count_nonzero(frame) > 0
