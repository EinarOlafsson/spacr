"""What a hover costs now, and what the **Animation** word buys.

The tooltip used to decode a movie on every hover of an animated setting —
141 packaged animations, one GIF read, cropped, zoomed and cached per hover.
It shows text only now, and reveals the animation when the reader clicks
**Animation** in the footer.

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
    """All 141, since 141 decoded movies was the cost being removed."""
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

def test_the_reveal_is_session_scoped_not_per_setting(tooltip, qtbot, decodes):
    """One click, then every animated setting shows without another."""
    tooltip.show_for(_anchor(qtbot, ANIMATED_KEYS[0]), HTML)
    tooltip.animation_link().clicked.emit()
    assert len(decodes) == 1

    for key in ANIMATED_KEYS[1:]:
        tooltip.show_for(_anchor(qtbot, key), HTML)
        assert tooltip.animation_view().isVisible(), f"{key} stayed hidden"
    assert len(decodes) == len(ANIMATED_KEYS)


def test_the_reveal_is_not_per_hover(tooltip, qtbot):
    """Re-hovering the same setting does not put it back to text only."""
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
    assert fresh.animations_shown() is False


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


def test_turning_the_preference_off_takes_a_session_reveal_with_it(
        tooltip, qtbot):
    """Preferences keeps the last word, so the two switches cannot disagree.

    The session override is remembered against the preference value it was
    made under. Change that value and the override is dropped, which is what
    stops one click from making the dialog's switch useless for the rest of
    the run.
    """
    anchor = _anchor(qtbot)
    tooltip.show_for(anchor, HTML)
    tooltip.animation_link().clicked.emit()
    assert tooltip.animations_shown() is True

    prefs.set_setting_animations_enabled(True)
    assert tooltip.animations_shown() is True, "the two now agree"

    prefs.set_setting_animations_enabled(False)
    assert tooltip.animations_shown() is False, (
        "the click still overrides a preference the user has since changed")

    tooltip.show_for(anchor, HTML)
    assert not tooltip.animation_view().isVisible()


def test_the_word_folds_away_an_animation_the_preference_turned_on(
        tooltip, qtbot):
    """The reveal works in both directions, whichever way it started."""
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.animation_view().isVisible()

    tooltip.animation_link().clicked.emit()
    assert tooltip.animations_shown() is False
    assert not tooltip.animation_view().isVisible()
    assert tooltip.animation_view().frame_count() == 0
    assert prefs.get_setting_animations_enabled() is True, (
        "folding one popup away rewrote the global preference")
