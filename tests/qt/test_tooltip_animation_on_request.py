"""What a hover costs now, and what the **Animation** word buys.

The tooltip used to decode a movie on every hover of an animated setting —
one GIF read, cropped, zoomed and cached, across the 143 settings the 94
packaged animations cover. It shows text only now, and reveals the animation
when the reader clicks **Animation** in the footer.

The saving is measured by counting the REAL decode calls, not by reading a
flag: a flag can say "off" while the loader runs anyway, which is exactly the
kind of test that passes over a bug.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QSettings, Qt

from PySide6.QtWidgets import QLabel

from spacr.qt import preferences as prefs
from spacr.qt.widgets import animation_zoom as az
from spacr.qt.widgets.hover_tooltip import HoverTooltip


#: Settings the packaged registry really does have animations for.
ANIMATED_KEYS = ("cell_diameter", "cell_CP_prob", "nucleus_diameter")

HTML = (
    "<b>Cell diameter</b> <i>(int)</i><br>"
    "Expected cell diameter in pixels. Cellpose rescales each image so that "
    "objects match the scale its model was trained at, so a wrong value here "
    "quietly degrades every downstream measurement.<br>"
    "<a href='https://example.invalid/'>Open spaCR API documentation</a>"
)


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never touch the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)`` and that
    constructor resolves to the NATIVE location whatever ``setPath`` says, so
    replacing the accessor is the only reliable isolation.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def tooltip(qtbot):
    """A fresh popup — never the singleton, which the whole session shares."""
    popup = HoverTooltip()
    qtbot.addWidget(popup)
    yield popup
    popup.hide()


@pytest.fixture
def decodes(monkeypatch):
    """Every real call to the GIF loader, in order.

    Wrapped rather than stubbed: the animation still has to decode when it IS
    asked for, and a stub would make the reveal tests measure nothing.
    ``_AnimationView.load`` imports this name from the module at call time, so
    patching the module attribute is what the widget actually reaches.
    """
    calls = []
    real = az.zoomed_animation

    def counting(path, size):
        calls.append(str(path))
        return real(path, size)

    monkeypatch.setattr(az, "zoomed_animation", counting)
    return calls


def _anchor(qtbot, key: str = ANIMATED_KEYS[0]) -> QLabel:
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    if key:
        label.setProperty("settingKey", key)
    return label


# ---------------------------------------------------------------------------
# 1. What a plain hover costs
# ---------------------------------------------------------------------------

def test_a_plain_hover_shows_text_and_decodes_nothing(tooltip, qtbot, decodes):
    """The measurement, on the loader itself."""
    for key in ANIMATED_KEYS:
        tooltip.show_for(_anchor(qtbot, key), HTML)
        assert tooltip.text_label().text(), f"{key}: no text either"
        assert not tooltip.animation_view().isVisible()
        assert tooltip.animation() is None
        assert tooltip.animation_view().frame_count() == 0
        assert not tooltip.animation_view().is_playing()

    assert decodes == [], (
        f"{len(decodes)} animation(s) were decoded for text-only hovers")


def test_the_counter_can_see_a_decode_when_there_is_one(
        tooltip, qtbot, decodes):
    """The control for the test above.

    Without this, "no decodes happened" is equally true of a probe wired to
    nothing at all.
    """
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot), HTML)
    assert len(decodes) == 1, (
        "the decode counter cannot see a decode it was pointed at")


def test_hovering_the_whole_registry_decodes_nothing(tooltip, qtbot, decodes):
    """Every animated setting, since a decode each was the cost removed."""
    from spacr.qt.screens.settings_model import format_tooltip, get_tooltips
    from spacr.setting_animations import animation_for_setting

    animated = 0
    for key, text in get_tooltips().items():
        if animation_for_setting(key) is None:
            continue
        animated += 1
        tooltip.show_for(_anchor(qtbot, key), format_tooltip(text, "mask", key))
    assert animated > 100, "the animation registry did not load"
    assert decodes == [], f"{len(decodes)} decodes across {animated} hovers"


def test_the_word_is_offered_on_a_setting_that_has_an_animation(
        tooltip, qtbot):
    """Text only, but not silent about what is available."""
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.animation_link().isVisible()
    assert tooltip.offered_animation() is not None
    assert tooltip.offered_animation().slug == ANIMATED_KEYS[0]


# ---------------------------------------------------------------------------
# 2. What the click buys
# ---------------------------------------------------------------------------

def test_the_click_decodes_exactly_once_and_reveals_the_square(
        tooltip, qtbot, decodes):
    tooltip.show_for(_anchor(qtbot), HTML)
    assert decodes == []

    tooltip.animation_link().clicked.emit()
    assert len(decodes) == 1
    view = tooltip.animation_view()
    assert view.isVisible()
    assert view.frame_count() > 1
    assert view.is_playing()


def test_the_revealed_square_lands_at_the_measured_geometry(tooltip, qtbot):
    """Right of the text, top-aligned, 220 px square, content at 70-80%.

    All four in the popup's own coordinates: the prose lives inside a text
    column, so its own ``y`` is an offset within that column and comparing it
    with the square's would be comparing two origins.

    The help text is deliberately long enough that the text column ends up
    TALLER than the square. Beside a column of exactly 220 px, top and bottom
    alignment put the square in the same place and the second assertion below
    cannot fail — which is the shape the first version of this test had.
    """
    long_html = HTML.replace(
        "quietly degrades every downstream measurement.",
        "quietly degrades every downstream measurement. " + "More help. " * 60)
    tooltip.show_for(_anchor(qtbot), long_html)
    tooltip.animation_link().clicked.emit()

    text = tooltip.text_label()
    view = tooltip.animation_view()
    text_origin = text.mapTo(tooltip, QPoint(0, 0))
    view_origin = view.mapTo(tooltip, QPoint(0, 0))

    assert view_origin.x() >= text_origin.x() + text.width(), (
        f"the square at x={view_origin.x()} is not right of text ending at "
        f"x={text_origin.x() + text.width()}")
    assert tooltip.text_column().height() > HoverTooltip.ANIMATION_SIZE, (
        "the text column is exactly as tall as the square, so this test "
        "cannot tell a top-aligned square from a bottom-aligned one")
    assert view_origin.y() == text_origin.y(), (
        f"text top {text_origin.y()} is not level with square top "
        f"{view_origin.y()}")
    assert view.width() == view.height() == HoverTooltip.ANIMATION_SIZE

    frame = az.from_qimage(view.pixmap().toImage())
    extent = az.content_extent([frame])
    assert az.MIN_FILL <= extent <= az.MAX_FILL, (
        f"the revealed frame covers {extent:.1%} of its square")


def test_every_packaged_animation_the_word_reveals_lands_at_220(
        tooltip, qtbot):
    """The reveal is one code path; measure it on every packaged animation."""
    from spacr.setting_animations import iter_setting_animations

    packaged = list(iter_setting_animations())
    sizes, extents = set(), []
    revealed = 0
    for animation in packaged:
        key = animation.settings[0]
        tooltip.show_for(_anchor(qtbot, key), HTML)
        if not tooltip.animation_link().isVisible():
            continue
        tooltip.animation_link().clicked.emit()
        view = tooltip.animation_view()
        if not view.isVisible():
            continue
        revealed += 1
        sizes.add((view.width(), view.height()))
        extents.append(az.content_extent(
            [az.from_qimage(view.pixmap().toImage())]))
        # Fold back, so the next iteration starts from a plain hover again.
        tooltip.animation_link().clicked.emit()

    assert len(packaged) > 50, "the animation registry did not load"
    assert revealed == len(packaged), (
        f"only {revealed} of {len(packaged)} animations could be revealed")
    assert sizes == {(HoverTooltip.ANIMATION_SIZE,
                      HoverTooltip.ANIMATION_SIZE)}, sizes
    median = float(np.median(extents))
    assert az.MIN_FILL <= median <= az.MAX_FILL, (
        f"median content extent is {median:.1%}")


def test_a_mouse_click_on_the_word_reveals_it_too(tooltip, qtbot):
    """Through the real event, not just the signal."""
    tooltip.show_for(_anchor(qtbot), HTML)
    qtbot.mouseClick(tooltip.animation_link(), Qt.LeftButton)
    assert tooltip.animation_view().isVisible()


# ---------------------------------------------------------------------------
# 3. Where the reveal lives
# ---------------------------------------------------------------------------

def test_one_press_never_puts_the_rest_of_the_run_back_on_the_decode_path(
        tooltip, qtbot, decodes):
    """A whole sweep of the registry after a press still decodes nothing.

    The reason the reveal is per setting: a session-wide one would make one
    click cost a decoded movie on every later hover for the rest of the run,
    which is exactly what a weak machine cannot afford.
    """
    from spacr.qt.screens.settings_model import format_tooltip, get_tooltips
    from spacr.setting_animations import animation_for_setting

    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    tooltip.animation_link().clicked.emit()
    assert len(decodes) == 1

    swept = 0
    for key, text in get_tooltips().items():
        if key == ANIMATED_KEYS[0] or animation_for_setting(key) is None:
            continue
        swept += 1
        tooltip.show_for(_anchor(qtbot, key), format_tooltip(text, "mask", key))
        assert not tooltip.animation_view().isVisible(), f"{key} was revealed"
    assert swept > 100, "the animation registry did not load"
    assert len(decodes) == 1, (
        f"{len(decodes) - 1} settings decoded off the back of one press")


def test_the_press_names_one_setting(tooltip, qtbot):
    """The state is a key, so it is inspectable and cannot be a global flag."""
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    assert tooltip.toggled_setting() is None
    tooltip.animation_link().clicked.emit()
    assert tooltip.toggled_setting() == ANIMATED_KEYS[0]

    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[1]), HTML)
    assert tooltip.animations_shown() is False
    tooltip.animation_link().clicked.emit()
    assert tooltip.toggled_setting() == ANIMATED_KEYS[1], (
        "the press did not move to the setting it was made on")


def test_the_reveal_is_not_per_hover_within_one_setting(tooltip, qtbot):
    """Re-hovering the SAME setting does not put it back to text only.

    Moving the pointer from a label into the popup below it and back fires a
    fresh ``Enter``; dropping the reveal there would fight the reader for no
    gain, and it cannot leak, because the key has not changed.
    """
    anchor = _anchor(qtbot)
    tooltip.show_for(anchor, HTML)
    tooltip.animation_link().clicked.emit()

    tooltip.show_for(anchor, HTML)
    assert tooltip.animation_view().isVisible()


def test_the_reveal_does_not_outlive_the_process(tooltip, qtbot):
    """It lives on the singleton; the next run starts text only again."""
    tooltip.show_for(_anchor(qtbot), HTML)
    tooltip.animation_link().clicked.emit()
    assert tooltip.animations_shown() is True

    fresh = HoverTooltip()
    qtbot.addWidget(fresh)
    fresh.show_for(_anchor(qtbot), HTML)
    assert fresh.animations_shown() is False


# ---------------------------------------------------------------------------
# 3b. Nothing is held before a press, and little after
# ---------------------------------------------------------------------------

def test_no_frames_are_held_until_a_press(tooltip, qtbot, decodes):
    """Lazy, measured on the pixmaps as well as on the loader."""
    view = tooltip.animation_view()
    for key in ANIMATED_KEYS:
        tooltip.show_for(_anchor(qtbot, key), HTML)
        assert view.frame_count() == 0, f"{key} held frames unasked"
    assert decodes == []


def test_moving_to_another_setting_drops_the_previous_frames(
        tooltip, qtbot, decodes):
    """The pixmap cache is one animation deep, and it is the visible one."""
    view = tooltip.animation_view()
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    tooltip.animation_link().clicked.emit()
    assert view.frame_count() > 1
    assert view.slug() == ANIMATED_KEYS[0]

    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[1]), HTML)
    assert view.frame_count() == 0, (
        "the previous setting's pixmaps are still resident")
    assert view.slug() == ""
    assert len(decodes) == 1


def test_folding_and_re_pressing_the_same_setting_does_not_decode_again(
        tooltip, qtbot, decodes):
    """Bounded at one animation, and it is what makes a repeat press free."""
    view = tooltip.animation_view()
    tooltip.show_for(_anchor(qtbot), HTML)
    tooltip.animation_link().clicked.emit()
    frames = view.frame_count()
    assert frames > 1 and len(decodes) == 1

    tooltip.animation_link().clicked.emit()          # fold away
    assert not view.isVisible()
    assert not view.is_playing(), "a hidden panel is still swapping pixmaps"
    assert view.frame_count() == frames, (
        "the frames of the setting still under the pointer were thrown away")

    tooltip.animation_link().clicked.emit()          # and back
    assert view.isVisible()
    assert view.is_playing()
    assert len(decodes) == 1, "re-pressing the same setting decoded again"


# ---------------------------------------------------------------------------
# 4. The preference and the default agree
# ---------------------------------------------------------------------------

def test_one_default_stated_in_both_places():
    assert prefs.DEFAULT_SETTING_ANIMATIONS is False
    assert prefs.get_setting_animations_enabled() is False
    assert HoverTooltip().animations_shown() is False


def test_the_preference_means_do_not_ask_me(tooltip, qtbot, decodes):
    """On, the animation appears with no click — the only difference."""
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.animation_view().isVisible()
    assert len(decodes) == 1
    assert tooltip.animation_link().isVisible(), (
        "there is no way left to fold this one away")


def test_a_press_cannot_stop_the_preference_taking_effect(tooltip, qtbot):
    """The press names one setting, so it can never speak for the others.

    This is why no baseline bookkeeping is needed: turning the preference on
    reaches every setting the reader did not press, and turning it off again
    reaches every setting including the pressed one on the next hover.
    """
    pressed, other = ANIMATED_KEYS[0], ANIMATED_KEYS[1]
    tooltip.show_for(_anchor(qtbot, pressed), HTML)
    tooltip.animation_link().clicked.emit()
    assert tooltip.animations_shown() is True

    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot, other), HTML)
    assert tooltip.animation_view().isVisible(), (
        "the preference did not reach a setting nobody pressed")

    prefs.set_setting_animations_enabled(False)
    tooltip.show_for(_anchor(qtbot, other), HTML)
    assert not tooltip.animation_view().isVisible()


def test_the_word_folds_away_an_animation_the_preference_turned_on(
        tooltip, qtbot):
    """The press works in both directions, whichever way it started."""
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    assert tooltip.animation_view().isVisible()

    tooltip.animation_link().clicked.emit()
    assert tooltip.animations_shown() is False
    assert not tooltip.animation_view().isVisible()
    assert prefs.get_setting_animations_enabled() is True, (
        "folding one popup away rewrote the global preference")

    # And it folded away THIS setting only: the next one obeys the preference.
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[1]), HTML)
    assert tooltip.animation_view().isVisible()


# ---------------------------------------------------------------------------
# 5. One press, one animation
# ---------------------------------------------------------------------------

def test_revealing_one_setting_leaves_the_next_tooltip_hidden(
        tooltip, qtbot, decodes):
    """The rule the user asked for, stated as plainly as it was asked.

    Pressing **Animation** on one setting shows that setting's animation and
    nothing else. The next tooltip starts hidden again and decodes nothing
    until its own press — which is the whole point on a weak machine, where a
    single click must not quietly put every later hover back on the decode
    path.
    """
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    tooltip.animation_link().clicked.emit()
    assert tooltip.animation_view().isVisible()
    assert len(decodes) == 1

    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[1]), HTML)
    assert not tooltip.animation_view().isVisible(), (
        "the second setting inherited the first one's reveal")
    assert tooltip.animation() is None
    assert len(decodes) == 1, (
        f"the second tooltip decoded without being asked: {decodes[1:]}")

    tooltip.animation_link().clicked.emit()
    assert tooltip.animation_view().isVisible()
    assert len(decodes) == 2
