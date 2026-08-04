"""What the user asked the setting tooltip to look like, measured.

Six complaints, six measurements — none of them taken on trust:

* one tooltip appears on hover, not two;
* a text-only popup is exactly as tall as its text;
* beside an animation, the text block is exactly as tall as the square;
* the square's corners really are round (its corner pixels are background);
* a word inside the popup reveals the animation and folds it away again, and
  the popup resizes with it;
* the last line is two words — **API** in the theme accent, **Animation** in
  teal, neither underlined — and each one does its job.

Every pixel assertion here carries a control that PROVES it can fail: the
underline probe is run against a real ``<a href>`` label, the colour probe is
scored against the *other* word's colour, and the one-tooltip count is taken
again on a widget the popup has not claimed.
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QHelpEvent
from PySide6.QtWidgets import QApplication, QLabel, QToolTip

from spacr.qt import preferences as prefs
from spacr.qt.theme import active_palette
from spacr.qt.widgets import animation_zoom as az
from spacr.qt.widgets.hover_tooltip import TEAL, HoverTooltip, split_api_link


#: A setting the packaged registry really does have an animation for.
ANIMATED_KEY = "cell_diameter"

API_URL = "https://spacr.readthedocs.io/en/latest/api/mask.html#cell_diameter"

HTML = (
    "<b>Cell diameter</b> <i>(int)</i><br>"
    "Expected cell diameter in pixels. Cellpose rescales each image so that "
    "objects match the scale its model was trained at, so a wrong value here "
    "quietly degrades every downstream measurement.<br>"
    f'<a href="{API_URL}">Open spaCR API documentation</a>'
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


def _anchor(qtbot, key: str = ANIMATED_KEY) -> QLabel:
    label = QLabel("Cell diameter")
    qtbot.addWidget(label)
    if key:
        label.setProperty("settingKey", key)
    return label


def _reveal(tooltip, anchor, html: str = HTML):
    """Hover, then click **Animation**.

    Animations are off until asked for, so a test that wants to measure one
    has to ask. The assertion in the middle is what stops this helper from
    quietly passing if the default ever flips back.
    """
    tooltip.show_for(anchor, html)
    assert not tooltip.animation_view().isVisible(), (
        "the animation was already showing; the reveal proves nothing")
    tooltip.animation_link().clicked.emit()
    return tooltip


def _grab(tooltip):
    return tooltip.grab().toImage()


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def _background(image) -> np.ndarray:
    """The popup's own surface colour, read a few pixels inside its border."""
    pixel = image.pixelColor(3, 3)
    return np.array([pixel.red(), pixel.green(), pixel.blue()], dtype=float)


def _word_patch(tooltip, word) -> np.ndarray:
    image = _grab(tooltip)
    origin = word.mapTo(tooltip, QPoint(0, 0))
    patch = image.copy(origin.x(), origin.y(), word.width(), word.height())
    return az.from_qimage(patch).astype(float)


def _ink_direction(patch: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Unit vector of the ink's colour, measured away from the background.

    Anti-aliased 12-pixel type never lands on its nominal RGB — every glyph
    pixel is ``background + alpha * (colour - background)``. Averaging the
    difference and normalising divides the unknown alpha out, which leaves a
    quantity that CAN be compared with a declared colour.
    """
    diff = patch.reshape(-1, 3) - background
    ink = diff[np.abs(diff).max(axis=1) > 8]
    assert len(ink) > 20, "the word did not render enough ink to measure"
    mean = ink.mean(axis=0)
    return mean / np.linalg.norm(mean)


def _longest_ink_runs(patch: np.ndarray, background: np.ndarray) -> list:
    """Longest horizontal run of ink on each row of a rendered word."""
    ink = np.abs(patch - background).max(axis=2) > 20
    runs = []
    for row in ink:
        best = current = 0
        for lit in row:
            current = current + 1 if lit else 0
            best = max(best, current)
        runs.append(best)
    return runs


# ---------------------------------------------------------------------------
# 1. One tooltip, not two
# ---------------------------------------------------------------------------

def _visible_tooltip_windows() -> list:
    """Every tooltip-class top-level window currently on screen."""
    found = []
    for widget in QApplication.topLevelWidgets():
        try:
            if widget.isVisible() and widget.windowType() == Qt.ToolTip:
                found.append(widget)
        except RuntimeError:
            continue
    return found


def _request_tooltip(widget) -> None:
    """Do what Qt's tooltip timer does ~700 ms after the pointer settles."""
    where = widget.mapToGlobal(QPoint(3, 3))
    QApplication.sendEvent(
        widget, QHelpEvent(QEvent.ToolTip, QPoint(3, 3), where))
    QApplication.processEvents()


def _clear_native_tooltip(qtbot) -> None:
    """Take any native tooltip off screen and wait until it is really gone."""
    QToolTip.hideText()
    QApplication.processEvents()
    qtbot.waitUntil(lambda: QToolTip.text() == ""
                    and not [w for w in _visible_tooltip_windows()
                             if w.objectName() != "HoverTooltip"],
                    timeout=2000)


def test_hovering_a_real_setting_shows_exactly_one_tooltip(qtbot):
    """The reported bug: the sticky popup, then Qt's own on top of it.

    ``AppScreen.eventFilter`` calls ``refresh_api_tooltips`` on every
    ``Enter``, which re-applies the label's native ``toolTip()`` (with
    ``setToolTipDuration(-1)``, so it never goes away) for the accessibility
    tree — and then shows the sticky popup. Unlike ``_ApiTooltipFilter``,
    that filter does not swallow ``QEvent.ToolTip``, so Qt popped the native
    one up a second later, over the top of the first.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import refresh_api_tooltips

    _clear_native_tooltip(qtbot)
    if HoverTooltip._INSTANCE is not None:
        HoverTooltip._INSTANCE.hide()
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    labels = list(screen._hint_map)
    hovered, unclaimed = labels[0], labels[1]
    assert _visible_tooltip_windows() == [], "a tooltip was already on screen"

    # The bug, reproduced. `refresh_api_tooltips` is exactly what
    # `AppScreen.eventFilter` runs on Enter, and it is what plants the native
    # tooltip; this label simply never gets the sticky popup on top of it.
    refresh_api_tooltips(unclaimed)
    assert unclaimed.toolTip(), "the screen stopped planting a native tooltip"
    _request_tooltip(unclaimed)
    assert QToolTip.text() == unclaimed.toolTip(), (
        "a native tooltip does not fire here at all, so the assertions below "
        "would pass against the very bug they exist to catch")
    assert len(_visible_tooltip_windows()) == 1, (
        "the window count cannot see a native tooltip appear")

    _clear_native_tooltip(qtbot)

    # The same label, hovered for real: the sticky popup and nothing else.
    QApplication.sendEvent(hovered, QEvent(QEvent.Type.Enter))
    QApplication.processEvents()
    popup = HoverTooltip._INSTANCE
    assert popup is not None and popup.isVisible(), "no tooltip at all"
    assert hovered.toolTip(), "the label lost the text screen readers use"

    _request_tooltip(hovered)
    assert QToolTip.text() == "", (
        "a second, native tooltip appeared for the setting label: "
        f"{QToolTip.text()[:60]!r}")
    assert _visible_tooltip_windows() == [popup], (
        "more than one tooltip is on screen: "
        f"{[w.objectName() for w in _visible_tooltip_windows()]}")

    popup._hide_timer.stop()
    popup._anchor = None
    popup.hide()


def test_the_anchor_keeps_its_tooltip_text_for_screen_readers(tooltip, qtbot):
    """Suppressed, not deleted.

    ``toolTip()`` is what the accessibility tree reads out, and the screens
    deliberately keep it. The fix eats the tooltip EVENT instead.
    """
    anchor = _anchor(qtbot)
    anchor.setToolTip("<b>Cell diameter</b> — accessible help")
    tooltip.show_for(anchor, HTML)
    assert anchor.toolTip(), "the anchor's accessible tooltip text was erased"


def test_claiming_the_same_anchor_twice_installs_one_suppressor(
        tooltip, qtbot):
    """Qt keeps a LIST of filters; re-hovering must not grow it.

    A doubled suppressor would still swallow the tooltip, so the symptom
    here is not a second popup — it is a singleton that adds a filter to
    every label it is ever shown for and never takes one away. What is
    measured is therefore the remove/install pairing itself.
    """
    anchor = _anchor(qtbot)
    calls = []
    real_install = anchor.installEventFilter
    real_remove = anchor.removeEventFilter

    anchor.installEventFilter = lambda f: (
        calls.append(("install", f)), real_install(f))[1]
    anchor.removeEventFilter = lambda f: (
        calls.append(("remove", f)), real_remove(f))[1]

    for _ in range(4):
        tooltip.show_for(anchor, HTML)

    suppressor = tooltip._tooltip_suppressor
    ours = [action for action, f in calls if f is suppressor]
    assert ours == ["remove", "install"] * 4, (
        f"the popup does not remove before it installs: {ours}")


# ---------------------------------------------------------------------------
# 2 & 3. Heights
# ---------------------------------------------------------------------------

def _inner_height(tooltip) -> int:
    """The popup's height inside its own 1-pixel border.

    ``contentsRect`` is what the layout is given, so comparing against it
    keeps the arithmetic about slack rather than about frame chrome.
    """
    return tooltip.contentsRect().height()


def _boxed(tooltip, content_height: int) -> int:
    margins = tooltip.layout().contentsMargins()
    return content_height + margins.top() + margins.bottom()


def test_a_text_only_popup_is_exactly_as_tall_as_its_text(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot, key=""), "<b>Some section</b><br>Short.")

    label = tooltip.text_label()
    assert not tooltip.animation_view().isVisible()
    assert _inner_height(tooltip) == _boxed(tooltip, label.height()), (
        f"a text-only popup wraps {label.height()} px of text in "
        f"{_inner_height(tooltip)} px")
    assert label.height() == label.heightForWidth(label.width()), (
        "the text box itself carries slack")
    assert not tooltip._links.isVisible(), (
        "an empty footer row still costs the layout its spacing")


def test_the_text_block_is_exactly_as_tall_as_the_animation(tooltip, qtbot):
    _reveal(tooltip, _anchor(qtbot))

    column = tooltip.text_column()
    view = tooltip.animation_view()
    assert view.isVisible()
    assert column.height() == view.height() == HoverTooltip.ANIMATION_SIZE, (
        f"text block is {column.height()} px beside a {view.height()} px "
        f"square")
    assert _inner_height(tooltip) == _boxed(tooltip, view.height()), (
        "the popup is taller than the square it wraps")


def test_every_packaged_animation_keeps_its_text_inside_the_square(
        tooltip, qtbot):
    """The width ladder, exercised on the real help text of every animation.

    At a fixed 220-pixel column all 141 of these wrapped to between 238 and
    323 pixels — taller than the square, so every popup was a tall ribbon.
    The column widens instead.
    """
    from spacr.qt.screens.settings_model import format_tooltip, get_tooltips
    from spacr.setting_animations import animation_for_setting

    prefs.set_setting_animations_enabled(True)   # measure them all, not one
    heights = []
    for key, text in get_tooltips().items():
        if animation_for_setting(key) is None:
            continue
        tooltip.show_for(_anchor(qtbot, key), format_tooltip(text, "mask", key))
        assert tooltip.animation_view().isVisible(), f"{key}: no animation"
        heights.append((_inner_height(tooltip), key))
        assert tooltip.text_column().height() == HoverTooltip.ANIMATION_SIZE, (
            f"{key}: text block {tooltip.text_column().height()} px")
    assert len(heights) > 100, "the animation registry did not load"
    square = _boxed(tooltip, HoverTooltip.ANIMATION_SIZE)
    assert {height for height, _ in heights} == {square}, (
        f"popups of mixed height: {sorted({h for h, _ in heights})}")


def test_the_column_only_widens_when_the_prose_needs_it(tooltip, qtbot):
    """Short help keeps the neat pair of equal columns."""
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot), "<b>Cell diameter</b><br>Short.")
    assert tooltip.text_column().width() == HoverTooltip.ANIMATION_SIZE

    long_text = "<b>Cell diameter</b><br>" + ("Long help. " * 40)
    tooltip.show_for(_anchor(qtbot), long_text)
    assert tooltip.text_column().width() > HoverTooltip.ANIMATION_SIZE
    assert tooltip.text_column().width() <= HoverTooltip.TEXT_WIDTH
    assert tooltip.text_column().height() == HoverTooltip.ANIMATION_SIZE


# ---------------------------------------------------------------------------
# 4. Rounded corners
# ---------------------------------------------------------------------------

def test_the_animation_square_has_rounded_corners(tooltip, qtbot):
    """Sampled on the rendered popup, not asserted from a stylesheet.

    A ``border-radius`` in the sheet rounds only the background painted
    UNDER the pixmap, and the pixmap is opaque to its own edges — which is
    why the corners stayed square while the rule said otherwise.
    """
    _reveal(tooltip, _anchor(qtbot))
    view = tooltip.animation_view()
    image = _grab(tooltip)
    background = _background(image)
    origin = view.mapTo(tooltip, QPoint(0, 0))

    def sample(dx, dy):
        pixel = image.pixelColor(origin.x() + dx, origin.y() + dy)
        return np.array([pixel.red(), pixel.green(), pixel.blue()], float)

    last = view.width() - 1
    for dx, dy, name in ((0, 0, "top-left"), (last, 0, "top-right"),
                         (0, last, "bottom-left"), (last, last, "bottom-right")):
        corner = sample(dx, dy)
        assert np.abs(corner - background).max() <= 4, (
            f"the {name} corner is {corner.tolist()}, not the popup's "
            f"{background.tolist()} — the square is not rounded")

    # ... and the square really is there between its corners, so the test
    # cannot be satisfied by simply not drawing the animation.
    edge = sample(view.width() // 2, 0)
    assert np.abs(edge - background).max() > 8, (
        "the top edge is background too; nothing was drawn at all")


# ---------------------------------------------------------------------------
# 5. The reveal
# ---------------------------------------------------------------------------

def test_the_animation_word_reveals_the_square_and_folds_it_back(
        tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)
    view = tooltip.animation_view()
    narrow, short = tooltip.width(), tooltip.height()
    assert not view.isVisible(), "a plain hover is text only"
    assert view.frame_count() == 0, "a plain hover decoded frames anyway"
    assert tooltip.offered_animation() is not None, (
        "the popup does not know which animation the word would show")

    tooltip.animation_link().clicked.emit()
    assert view.isVisible()
    assert tooltip.animation() is not None
    assert view.frame_count() > 1
    assert tooltip.width() > narrow and tooltip.height() > short, (
        f"the popup did not grow: {tooltip.size()} was {narrow}x{short}")

    tooltip.animation_link().clicked.emit()
    assert not view.isVisible()
    assert view.frame_count() == 0, "frames stayed decoded for a hidden panel"
    assert (tooltip.width(), tooltip.height()) == (narrow, short)


def test_the_reveal_survives_a_move_to_the_next_setting(tooltip, qtbot):
    """Revealed once, revealed for the settings hovered after it.

    Session scope, deliberately: per-hover would un-reveal the moment the
    pointer moved on, and per-setting would ask a reader who wants animations
    to click every one of them.
    """
    _reveal(tooltip, _anchor(qtbot))
    assert tooltip.animations_shown()

    tooltip.show_for(_anchor(qtbot, "cell_CP_prob"), HTML)
    assert tooltip.animation_view().isVisible()
    assert tooltip.animation() is not None
    assert tooltip.animation().slug != ANIMATED_KEY, (
        "the second setting is showing the first one's animation")


def test_the_reveal_is_this_popup_and_not_the_next_process(tooltip, qtbot):
    """It lives on the singleton, so a fresh popup starts unrevealed."""
    _reveal(tooltip, _anchor(qtbot))
    assert tooltip.animations_shown()

    fresh = HoverTooltip()
    qtbot.addWidget(fresh)
    assert fresh.animations_shown() is False
    fresh.show_for(_anchor(qtbot), HTML)
    assert not fresh.animation_view().isVisible()


def test_the_reveal_does_not_touch_the_global_preference(tooltip, qtbot):
    """Showing one animation now is not the same as never being asked."""
    assert prefs.get_setting_animations_enabled() is False
    _reveal(tooltip, _anchor(qtbot))
    assert prefs.get_setting_animations_enabled() is False


def test_the_preference_on_reveals_without_a_click(tooltip, qtbot):
    """The one place to say "always" — and it means "stop asking me"."""
    prefs.set_setting_animations_enabled(True)
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.animations_shown() is True
    assert tooltip.animation_view().isVisible()
    # And the word is still there, to fold this one away.
    assert tooltip.animation_link().isVisible()


def test_a_setting_without_an_animation_hides_the_word(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot, key="src"), HTML)
    assert not tooltip.animation_link().isVisible()
    assert tooltip.api_link().isVisible(), "the API word went with it"


# ---------------------------------------------------------------------------
# 6. Two words, two colours, no underline
# ---------------------------------------------------------------------------

def test_the_footer_is_two_words_and_not_a_sentence(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)
    assert tooltip.api_link().text() == "API"
    assert tooltip.animation_link().text() == "Animation"
    assert "Open spaCR API documentation" not in tooltip.text_label().text()
    assert "<a " not in tooltip.text_label().text().lower(), (
        "the old link is still in the prose")
    assert tooltip.api_url() == API_URL


def test_api_is_left_of_animation(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot), HTML)
    api = tooltip.api_link().mapTo(tooltip, QPoint(0, 0)).x()
    word = tooltip.animation_link().mapTo(tooltip, QPoint(0, 0)).x()
    assert api < word, "the words are not in the order 'API Animation'"


def test_the_two_words_render_in_the_declared_colours(tooltip, qtbot):
    """Measured on the rendered ink, and scored against the WRONG colour too.

    Without the second score this passes for any pair of colours that are
    merely both non-background.
    """
    tooltip.show_for(_anchor(qtbot), HTML)
    background = _background(_grab(tooltip))
    accent = active_palette()["accent"]

    for word, declared, other in ((tooltip.api_link(), accent, TEAL),
                                  (tooltip.animation_link(), TEAL, accent)):
        measured = _ink_direction(_word_patch(tooltip, word), background)
        want = _rgb(declared) - background
        want /= np.linalg.norm(want)
        wrong = _rgb(other) - background
        wrong /= np.linalg.norm(wrong)
        assert float(measured @ want) > 0.99, (
            f"{word.text()} renders as {measured.round(3).tolist()}, not "
            f"{declared} ({want.round(3).tolist()})")
        assert float(measured @ want) > float(measured @ wrong) + 0.02, (
            f"{word.text()} is as close to {other} as it is to {declared}")


def test_the_blue_is_the_theme_accent_and_the_teal_is_the_dna_rain_default():
    """Where the two colours come from, stated once."""
    from spacr.qt.widgets.dna_rain import DEFAULT_COLOR

    assert TEAL == DEFAULT_COLOR == "#009B9B"
    assert "accent" in active_palette()
    # The palette has no teal of its own: `info` is a second name for blue.
    assert active_palette()["info"] == active_palette()["accent"]


def test_neither_word_is_underlined(tooltip, qtbot):
    """A rendered underline is a near-full-width run of ink on one row.

    The control below is the same word inside a real ``<a href>``, which Qt
    underlines — so this probe is known to be able to fail.
    """
    tooltip.show_for(_anchor(qtbot), HTML)
    background = _background(_grab(tooltip))

    for word in (tooltip.api_link(), tooltip.animation_link()):
        runs = _longest_ink_runs(_word_patch(tooltip, word), background)
        assert max(runs) < 0.8 * word.width(), (
            f"{word.text()} has a {max(runs)} px run of ink across a "
            f"{word.width()} px word — that is an underline")
        assert not word.font().underline()

    control = QLabel()
    qtbot.addWidget(control)
    control.setTextFormat(Qt.RichText)
    control.setStyleSheet(
        "QLabel { background: #16171a; color: #4A9EFF; font-size: 12px; }")
    control.setText("<a href='x'>Animation</a>")
    control.adjustSize()
    control.show()
    patch = az.from_qimage(control.grab().toImage()).astype(float)
    runs = _longest_ink_runs(patch, patch[0, 0])
    assert max(runs) >= 0.8 * control.width(), (
        "the underline probe cannot see an underline it was pointed at")


def test_clicking_api_opens_the_documentation(tooltip, qtbot, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "spacr.qt.widgets.hover_tooltip.QDesktopServices.openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    tooltip.show_for(_anchor(qtbot), HTML)
    qtbot.mouseClick(tooltip.api_link(), Qt.LeftButton)
    assert opened == [API_URL]


def test_clicking_animation_toggles_instead_of_opening_a_browser(
        tooltip, qtbot, monkeypatch):
    opened = []
    monkeypatch.setattr(
        "spacr.qt.widgets.hover_tooltip.QDesktopServices.openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    tooltip.show_for(_anchor(qtbot), HTML)
    assert not tooltip.animation_view().isVisible()
    qtbot.mouseClick(tooltip.animation_link(), Qt.LeftButton)
    assert opened == []
    assert tooltip.animation_view().isVisible()


def test_a_body_with_no_link_hides_the_api_word(tooltip, qtbot):
    tooltip.show_for(_anchor(qtbot, key=""), "<b>Some section</b><br>Text.")
    assert not tooltip.api_link().isVisible()
    assert tooltip.api_url() == ""


# ---------------------------------------------------------------------------
# The split itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body,url", [
    ("<b>A</b><br>Text.<br><a href='https://x/'>Open spaCR API "
     "documentation</a>", "https://x/"),
    ('<b>A</b><br>Text.<br><a href="https://x/?a=1&amp;b=2">docs</a>',
     "https://x/?a=1&b=2"),
    ("<b>A</b><br>Text.", ""),
])
def test_split_api_link_takes_only_a_trailing_link(body, url):
    head, found = split_api_link(body)
    assert found == url
    assert "Open spaCR API documentation" not in head
    if url:
        assert not head.rstrip().endswith("<br>")


def test_a_link_in_the_middle_of_the_prose_is_left_alone():
    body = "See <a href='https://x/'>this</a> for details."
    head, url = split_api_link(body)
    assert (head, url) == (body, "")


def test_the_url_is_unescaped_so_the_browser_gets_the_real_query():
    _head, url = split_api_link('T<br><a href="https://x/?a=1&amp;b=2">d</a>')
    assert url == "https://x/?a=1&b=2"
