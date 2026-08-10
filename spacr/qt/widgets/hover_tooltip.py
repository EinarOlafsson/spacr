"""
HoverTooltip — a QFrame-based popup that stays visible when the mouse
enters it. Unlike QToolTip, users can move their cursor into the popup
to click links inside.

Usage::

    tip = HoverTooltip.instance()
    tip.show_for(some_widget, "some html")   # on hover-enter
    tip.start_hide()                          # on hover-leave

The popup cancels its own hide timer if the mouse enters it, and only
actually hides when neither the anchor nor the popup itself is under
the cursor.

A hover shows **text only**. No GIF is decoded, no frames are cached and no
timer runs until the reader asks for the animation: 141 settings have one, and
each is a ~73 ms decoded movie. A hover that only wanted the sentence should
not pay for one. Measured, a sweep of all 141: 0 decodes, 2.6 ms a hover.

Asked for, the animation appears to the RIGHT of the text. Both columns start
at the same top edge, so the first line of prose and the first frame are read
together rather than one being hunted for beside the other::

    +-----------------------------+-----------------------------+
    | Cell diameter (int)         |                             |
    | Expected cell diameter in   |         (animation)         |
    | pixels...                   |                             |
    | API  Animation              |                             |
    +-----------------------------+-----------------------------+

The text column is exactly as tall as the animation square and no taller:
its width is widened, one step at a time, until the prose fits inside the
square's height. With no animation beside it the popup shrinks to the
text — nothing is padded out to a shape it does not need.

The last line is two words, not a sentence: **API** in the theme accent opens
the same documentation page the old ``Open spaCR API documentation`` link did,
and **Animation** in teal reveals the square — or folds it away again.

That reveal is PER SETTING. Pressing **Animation** on ``cell_diameter`` shows
``cell_diameter``'s animation and nothing else; move to the next setting and it
is hidden again until its own **Animation** is pressed. A session-wide reveal
was the obvious alternative and is exactly wrong for the machines this was
asked for: one click would put every later hover back on the ~73 ms decode
path for the rest of the run, which is the cost the change exists to avoid.
Measured, the same sweep of 141 settings taken straight after a press: still
0 decodes.

Re-hovering the SAME setting keeps its reveal, so moving the pointer between a
label and the popup below it does not fight the reader. The state is one key
and one bool — see :meth:`HoverTooltip.animations_shown`.

Nothing is decoded before a press. The **Animation** word is offered from a
registry lookup, which reads no pixels; the GIF is read, measured, cropped,
zoomed and rounded only when the word is pressed. Two caches sit under that,
and neither is ever filled speculatively:

* :func:`spacr.qt.widgets.animation_zoom.zoomed_animation` already keeps the
  eight most recent zooms (~2.6 MB each), which is what turns a repeat press
  from ~73 ms into ~2.9 ms;
* this widget keeps the finished pixmaps of ONE animation — the one last
  revealed, ~3.5 MB — so folding a setting away and back is free. They are
  dropped the moment the pointer moves to a different setting.

The *Setting animations* preference is the escape hatch for a reader who wants
them always: on, every tooltip starts revealed and the word folds THIS one
away; off — the default — every tooltip starts hidden and the word reveals
THIS one. Because a press only ever names one setting, it can never leave the
preference unable to take effect.

Which animation is decided by the anchor's ``settingKey`` property, so no
caller has to pass one; the callers that put help on a label already set it.
Anything without that property — a section header, a home tile — gets a
text-only popup with no **Animation** word to click.

The popup is ONE surface. Its two layout containers paint nothing, so the
rounded grey frame is the only fill and the page-opacity preference moves it
as a single layer — see :meth:`HoverTooltip._apply_theme` for the black slab
that taught us.

Only one tooltip ever appears. The screens that anchor this popup also leave
a native Qt tooltip on the same label (``refresh_api_tooltips`` re-applies it
on every ``Enter``, for the accessibility tree), and Qt's own tooltip timer
would pop that up a second later, on top of this one — two tooltips, one
after the other. Claiming an anchor therefore installs
:class:`_NativeTooltipSuppressor` on it, which swallows ``QEvent.ToolTip``
while leaving ``toolTip()`` intact for screen readers.
"""
from __future__ import annotations

import logging
import re
from html import unescape
from typing import Optional, Tuple

from PySide6.QtCore import (QEvent, QObject, QPoint, QRectF, QTimer, QUrl, Qt,
                            Signal)
from PySide6.QtGui import (QDesktopServices, QGuiApplication, QPainter,
                           QPainterPath, QPixmap)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QToolTip,
                               QVBoxLayout, QWidget)

from ..theme import SPACING, active_palette, font_px


LOGGER = logging.getLogger(__name__)

#: Sentinel for "work the animation out from the anchor". ``None`` cannot do
#: that job: it is the perfectly good answer "this tooltip has no animation",
#: which callers need to be able to say.
_DERIVE = object()

#: Qt's own "no maximum", which PySide6 does not export. Needed to undo a
#: `setFixedWidth`/`setFixedHeight`, both of which pin the minimum as well.
_UNBOUNDED = 16777215

#: The teal half of the footer. The palette has no teal — `info` is a second
#: name for the blue accent — so this is the DNA-rain default, which is the
#: only teal spaCR already ships as a named constant.
TEAL = "#009B9B"

_ANCHOR_RE = re.compile(
    r"<a\b[^>]*?href\s*=\s*([\"'])(.*?)\1[^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_BREAKS_RE = re.compile(r"(?:<br\s*/?>|\s)+$", re.IGNORECASE)


def split_api_link(html: str) -> Tuple[str, str]:
    """Split a trailing documentation link off a tooltip body.

    ``settings_model.format_tooltip`` ends every setting's help with
    ``<a href="...">Open spaCR API documentation</a>``. The popup renders
    that destination as its own **API** word instead, so the anchor is taken
    out of the prose here rather than in the formatter — the same string is
    still used verbatim by the hint strip, the accessibility tree and every
    other consumer of ``format_tooltip``.

    Only a link that really is the last thing in the body is taken; a link
    inside a sentence stays where the author put it.

    :returns: ``(body_without_the_link, url)``; ``url`` is ``""`` when there
        was no trailing link.
    """
    last = None
    for match in _ANCHOR_RE.finditer(html):
        last = match
    if last is None or html[last.end():].strip():
        return html, ""
    body = _TRAILING_BREAKS_RE.sub("", html[:last.start()])
    return body, unescape(last.group(2))


def _anchor_setting_key(anchor: Optional[QWidget]) -> str:
    """The setting an anchor speaks for, or ``""``.

    Both the reveal and the animation lookup are keyed on this, so it is one
    function: two readings of the same property could drift apart and leave a
    press scoped to a setting other than the one on screen.
    """
    if anchor is None:
        return ""
    try:
        key = anchor.property("settingKey")
    except RuntimeError:
        # The anchor's C++ half is gone; there is nothing to read.
        return ""
    return str(key) if key else ""


class _NativeTooltipSuppressor(QObject):
    """Swallow ``QEvent.ToolTip`` on every widget the popup speaks for.

    The settings screens keep a native Qt tooltip on each setting label —
    ``refresh_api_tooltips`` re-applies it on every ``Enter``, with
    ``setToolTipDuration(-1)`` so it never times out — because that string is
    what the accessibility tree reads out. Qt's tooltip timer then shows it
    roughly 700 ms after the pointer settles, which is *after*
    :meth:`HoverTooltip.show_for` has already put the sticky popup on screen:
    two tooltips, one after the other, the second covering the first.

    Deleting the label's ``toolTip()`` would fix the picture and cost the
    screen reader its text. Eating the event keeps both.
    """

    def eventFilter(self, watched, event):  # noqa: N802 (Qt naming)
        """Return ``True`` for tooltip requests, so Qt shows nothing."""
        if event.type() == QEvent.ToolTip:
            return True
        return super().eventFilter(watched, event)


class _LinkWord(QLabel):
    """One coloured, underline-free word that behaves like a link.

    Not an ``<a>``: Qt's rich text underlines anchors, and the two words were
    asked for without one. Colour comes from the popup's own stylesheet (see
    :meth:`HoverTooltip._apply_theme`) so both words re-theme together.
    """

    clicked = Signal()

    def __init__(self, text: str, object_name: str,
                 parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setObjectName(object_name)
        self.setTextFormat(Qt.PlainText)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)

    def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
        """Emit :attr:`clicked` for a left button released over the word."""
        if (event.button() == Qt.LeftButton
                and self.rect().contains(event.position().toPoint())):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _AnimationView(QLabel):
    """Plays one pre-zoomed setting animation at a fixed square size.

    Not a ``QMovie``: the frames are cropped and rescaled from the packaged
    GIF before they are shown (see
    :mod:`spacr.qt.widgets.animation_zoom`), which ``QMovie`` cannot do. It
    plays finished frames on a timer instead, honouring each frame's own
    delay so the generated timing survives.

    The corners are rounded into the frames themselves. A stylesheet
    ``border-radius`` rounds only the background the label paints *under* the
    pixmap, and the pixmap is opaque to its own edges — so the square stayed
    square however the sheet was written.
    """

    #: Corner radius of the square, in pixels.
    CORNER_RADIUS = 10

    def __init__(self, size: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("SettingTooltipAnimation")
        self._size = int(size)
        self.setFixedSize(self._size, self._size)
        self.setAlignment(Qt.AlignCenter)
        self._frames: list = []
        self._delays: list = []
        self._index = 0
        self._slug = ""
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

    # ------------------------------------------------------------------
    def slug(self) -> str:
        """Slug of the animation currently loaded, or ``""``."""
        return self._slug

    def rounded(self, pixmap: QPixmap) -> QPixmap:
        """Return ``pixmap`` clipped to this view's rounded rectangle.

        The black backing is painted inside the same path, not left to the
        stylesheet: a square background behind a rounded pixmap would simply
        fill the corners back in.
        """
        radius = float(self.CORNER_RADIUS)
        out = QPixmap(pixmap.size())
        out.fill(Qt.transparent)
        painter = QPainter(out)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            path = QPainterPath()
            path.addRoundedRect(QRectF(pixmap.rect()), radius, radius)
            painter.setClipPath(path)
            painter.fillPath(path, Qt.black)
            painter.drawPixmap(0, 0, pixmap)
        finally:
            painter.end()
        return out

    def load(self, animation) -> bool:
        """Load and start ``animation``; return whether anything is showing.

        A failure to decode is not an error worth interrupting a hover for —
        it returns ``False`` and the tooltip falls back to text only.
        """
        from .animation_zoom import to_qimage, zoomed_animation

        if animation is None:
            self.clear_animation()
            return False
        if self._slug == animation.slug and self._frames:
            # Same setting hovered again: keep playing rather than restart.
            self.play()
            return True

        zoomed = zoomed_animation(str(animation.path), self._size)
        if zoomed is None or not zoomed.frames:
            LOGGER.warning(
                "Could not load setting animation %s from %s",
                animation.slug, animation.path,
            )
            self.clear_animation()
            return False

        self._frames = [
            self.rounded(QPixmap.fromImage(to_qimage(frame)))
            for frame in zoomed.frames
        ]
        self._delays = list(zoomed.delays)
        self._slug = animation.slug
        self._index = 0
        self.setPixmap(self._frames[0])
        self.setAccessibleName(animation.title)
        self.setAccessibleDescription(
            "Animated explanation of this spaCR setting."
        )
        self._schedule()
        return True

    def clear_animation(self) -> None:
        """Stop playing and drop the frames."""
        self._timer.stop()
        self._frames = []
        self._delays = []
        self._index = 0
        self._slug = ""
        self.setPixmap(QPixmap())

    def stop(self) -> None:
        """Pause playback without forgetting the loaded animation."""
        self._timer.stop()

    def play(self) -> None:
        """Resume playback of the frames already loaded."""
        if self._frames and not self._timer.isActive():
            self._schedule()

    def is_playing(self) -> bool:
        """Whether the frame timer is currently running."""
        return self._timer.isActive()

    def frame_count(self) -> int:
        """How many frames are loaded."""
        return len(self._frames)

    #: Floor on a frame delay. The packaged animations run at 80 ms, but a
    #: hand-made GIF claiming 10 ms would spin this timer against the paint
    #: loop for no visible gain.
    MIN_DELAY_MS = 20

    # ------------------------------------------------------------------
    def _schedule(self) -> None:
        # One delay per frame, guaranteed by `read_frames`; a still image has
        # nothing to schedule.
        if len(self._frames) < 2:
            return
        self._timer.start(
            max(self.MIN_DELAY_MS, int(self._delays[self._index])))

    def _advance(self) -> None:
        if not self._frames:
            return
        self._index = (self._index + 1) % len(self._frames)
        self.setPixmap(self._frames[self._index])
        self._schedule()


class HoverTooltip(QFrame):
    """Sticky QFrame popup that survives cursor entry so users can click links.

    Access via :meth:`instance` — the popup is a process-wide singleton.
    """

    _INSTANCE: Optional["HoverTooltip"] = None

    #: Side of the square animation box.
    ANIMATION_SIZE = 220

    #: Width the text uses when there is no animation to sit beside.
    TEXT_WIDTH = 380

    #: Widths tried, in order, for the text column when an animation IS
    #: beside it. The first one whose prose fits inside the square's height
    #: wins, so short help keeps the neat pair of equal columns and long help
    #: spreads sideways instead of growing a tall ribbon. Measured against
    #: every packaged animation's help text: 380 is enough for all of them.
    TEXT_WIDTH_STEPS = (ANIMATION_SIZE, 260, 300, 340, TEXT_WIDTH)

    def __init__(self):
        # Popup window with tool-tip semantics but our own paint control.
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setObjectName("HoverTooltip")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._apply_theme()

        self._text_column = QWidget(self)
        self._text_column.setObjectName("HoverTooltipTextColumn")
        column = QVBoxLayout(self._text_column)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SPACING["xs"])

        self._label = QLabel(self._text_column)
        self._label.setObjectName("HoverTooltipText")
        self._label.setTextFormat(Qt.RichText)
        self._label.setOpenExternalLinks(True)
        self._label.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.LinksAccessibleByMouse
        )
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(self.TEXT_WIDTH)
        # Belt and braces with the layout's AlignTop below. If the label is
        # ever stretched to the height of the animation, QLabel's default
        # AlignVCenter would float the prose down to the middle of the square
        # while the widget's top edge stayed put — top-aligned by geometry
        # and centred to the eye, which is not what was asked for.
        self._label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        column.addWidget(self._label)

        self._links = QWidget(self._text_column)
        self._links.setObjectName("HoverTooltipLinks")
        links_row = QHBoxLayout(self._links)
        links_row.setContentsMargins(0, 0, 0, 0)
        links_row.setSpacing(SPACING["sm"])
        self._api_link = _LinkWord("API", "HoverTooltipApiLink", self._links)
        self._api_link.setAccessibleName("API")
        self._api_link.setAccessibleDescription(
            "Open spaCR API documentation for this setting."
        )
        self._api_link.clicked.connect(self.open_api_documentation)
        self._animation_link = _LinkWord(
            "Animation", "HoverTooltipAnimationLink", self._links)
        self._animation_link.setAccessibleName("Animation")
        self._animation_link.setAccessibleDescription(
            "Show or hide this setting's animation."
        )
        self._animation_link.clicked.connect(self.toggle_animation)
        links_row.addWidget(self._api_link)
        links_row.addWidget(self._animation_link)
        links_row.addStretch(1)
        column.addWidget(self._links)
        # Holds the prose and the two words at the TOP of a column that is
        # stretched to the height of the square beside it.
        column.addStretch(1)

        self._animation_view = _AnimationView(self.ANIMATION_SIZE, self)
        self._animation_view.hide()
        self._animation = None
        self._offered_animation = None
        # The reveal, in two fields: which setting the reader pressed
        # **Animation** on, and what they pressed it to. It applies to that
        # setting and to nothing else, so hovering anything else falls back to
        # the preference. A session-wide flag here is precisely the behaviour
        # that was rejected — one press must not put every later hover back on
        # the decode path.
        self._setting_key = ""
        self._toggled_key: Optional[str] = None
        self._toggled_to = False
        self._api_url = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                SPACING["sm"], SPACING["xs"])
        lay.setSpacing(SPACING["sm"])
        # AlignTop on both, so the text starts level with the first frame
        # instead of floating to the middle of a 220-pixel square.
        lay.addWidget(self._text_column, 0, Qt.AlignTop)
        lay.addWidget(self._animation_view, 0, Qt.AlignTop)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._maybe_hide)
        self._anchor: Optional[QWidget] = None
        self._tooltip_suppressor = _NativeTooltipSuppressor(self)

    def _apply_theme(self) -> None:
        """Refresh the popup's inline style from the theme on screen."""
        # This widget is a separate top-level window, so app-level QSS does
        # not reliably reach it. It is also a singleton that survives a
        # Preferences theme switch, hence this must be refreshed on show.
        palette = active_palette()
        self.setStyleSheet(
            f"QFrame#HoverTooltip {{"
            f"  background-color: {palette['surface_alt']};"
            f"  border: 1px solid {palette['border']};"
            f"  border-radius: 6px;"
            f"}}"
            # The two layout containers paint NOTHING. Both are plain
            # `QWidget`s, so without this they inherit the application sheet's
            # blanket `QWidget { background-color: bg }` — and `bg` is the
            # WINDOW colour, #000000 in the dark theme, not a surface. The
            # result was a black slab covering all but a 6-pixel margin of the
            # popup's own rounded grey: 20669 black pixels inside a #161719
            # frame. `theme.clear_container_surfaces` exists for exactly this
            # and could not help — it only tags ANONYMOUS widgets as
            # scaffolding, and both of these are named.
            #
            # Transparent rather than re-filled with the frame's colour on
            # purpose: one surface, one alpha. Painting the same grey twice is
            # what left the System panel's meters unable to thin out when the
            # page-opacity slider moved, because the two translucent layers
            # composited into something darker than either.
            f"QWidget#HoverTooltipTextColumn,"
            f"QWidget#HoverTooltipLinks {{"
            f"  background: transparent;"
            f"}}"
            f"QLabel {{"
            f"  color: {palette['fg']};"
            f"  font-size: {font_px('small')}px;"
            f"  background: transparent;"
            f"}}"
            # Transparent, not black: the rounded corners are cut into the
            # frames themselves, and a background painted by the sheet would
            # square them off again.
            f"QLabel#SettingTooltipAnimation {{"
            f"  background: transparent;"
            f"}}"
            # Two words, two colours, no underline anywhere.
            f"QLabel#HoverTooltipApiLink {{"
            f"  color: {palette['accent']};"
            f"  text-decoration: none;"
            f"}}"
            f"QLabel#HoverTooltipAnimationLink {{"
            f"  color: {TEAL};"
            f"  text-decoration: none;"
            f"}}"
        )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "HoverTooltip":
        """Return the process-wide singleton, creating it on first access."""
        if cls._INSTANCE is None:
            cls._INSTANCE = HoverTooltip()
        return cls._INSTANCE

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def show_for(self, anchor: QWidget, html: str, animation=_DERIVE) -> None:
        """Show the tooltip beneath ``anchor`` with body ``html``.

        :param anchor: widget the popup docks to (clamped to its screen).
        :param html: rich-text body; empty strings are ignored. A trailing
            documentation link is moved out of the prose and into the **API**
            word at the foot of the popup.
        :param animation: a :class:`spacr.setting_animations.SettingAnimation`
            to play beside the text, or ``None`` for text only. Left out, it
            is derived from the anchor's ``settingKey`` property — every
            caller that attaches setting help already sets that, so none of
            them had to change.
        """
        if not html:
            return
        self._apply_theme()
        self._anchor = anchor
        self._claim_anchor(anchor)
        # Which setting this tooltip is for, which is what the reveal is
        # scoped to. Read before `_set_animation`, because that is what asks
        # `animations_shown()` whether this particular setting was pressed.
        self._setting_key = _anchor_setting_key(anchor)
        body, url = split_api_link(str(html))
        self._api_url = url
        self._api_link.setVisible(bool(url))
        self._label.setText(body)
        self._set_animation(self._resolve_animation(anchor, animation))
        self.adjustSize()
        self._position_under(anchor)
        self.show()

    def start_hide(self, delay_ms: int = 250) -> None:
        """Schedule a hide after ``delay_ms`` unless the cursor re-enters."""
        self._hide_timer.start(delay_ms)

    def cancel_hide(self) -> None:
        """Cancel any pending hide timer (called on cursor re-entry)."""
        self._hide_timer.stop()

    def animation(self):
        """The animation currently shown beside the text, or ``None``."""
        return self._animation

    def offered_animation(self):
        """The animation this anchor has, shown or collapsed by the toggle."""
        return self._offered_animation

    def animation_view(self) -> _AnimationView:
        """The square animation panel — exposed for layout tests."""
        return self._animation_view

    def text_label(self) -> QLabel:
        """The explanation panel — exposed for layout tests."""
        return self._label

    def text_column(self) -> QWidget:
        """The prose and the two link words, as one block."""
        return self._text_column

    def api_link(self) -> _LinkWord:
        """The blue **API** word."""
        return self._api_link

    def animation_link(self) -> _LinkWord:
        """The teal **Animation** word that toggles the square."""
        return self._animation_link

    def api_url(self) -> str:
        """Documentation URL taken out of the body, or ``""``."""
        return self._api_url

    def animations_shown(self) -> bool:
        """Whether the setting currently hovered shows its animation.

        Off unless this setting was asked for. A press on **Animation** names
        one setting; every other setting falls back to the *Setting
        animations* preference, which means "show animations without asking"
        and defaults, like this, to off.

        Scoped to a setting rather than to the session on purpose: a reveal
        that outlived the setting would put every later hover back on the
        ~73 ms decode path after a single press, which is the cost the reader
        was trying to avoid. Nothing here needs to guard the preference
        either — a press cannot reach past the setting it named.

        Read on every hover, never cached: the popup is a process-wide
        singleton that outlives the Preferences dialog.
        """
        from ..preferences import get_setting_animations_enabled

        if (self._toggled_key is not None
                and self._toggled_key == self._setting_key):
            return self._toggled_to
        return get_setting_animations_enabled()

    def toggled_setting(self) -> Optional[str]:
        """The one setting a press has spoken for, or ``None`` — for tests."""
        return self._toggled_key

    # ------------------------------------------------------------------
    # The two words
    # ------------------------------------------------------------------
    def open_api_documentation(self) -> None:
        """Open the documentation page the body's trailing link pointed at."""
        if not self._api_url:
            return
        QDesktopServices.openUrl(QUrl(self._api_url))

    def toggle_animation(self) -> None:
        """Reveal this setting's animation, or fold it away again.

        Deliberately not written to :mod:`spacr.qt.preferences`, and
        deliberately naming one setting. This is the reader asking to see
        *this* animation; the preference is the reader asking to stop being
        asked about any of them. Because the press names a setting, it cannot
        turn animations on for the next one, and it cannot leave the
        preference unable to take effect.
        """
        # Read BEFORE the key is claimed: afterwards `animations_shown` would
        # answer with the state being written here rather than the one being
        # inverted.
        wanted = not self.animations_shown()
        self._toggled_key = self._setting_key
        self._toggled_to = wanted
        self._set_animation(self._offered_animation)
        self.adjustSize()
        if self.isVisible() and self._anchor is not None:
            self._position_under(self._anchor)

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _resolve_animation(self, anchor: QWidget, animation):
        """Which animation this anchor HAS, shown or not.

        A registry lookup, not a decode — nothing here reads a GIF, so asking
        it on every hover costs nothing even when the answer stays folded
        away. Whether it is put on screen is :meth:`animations_shown`'s call,
        made in :meth:`_set_animation`; this one has to answer regardless,
        because a setting with no animation is the one case where there is no
        **Animation** word to click.
        """
        if animation is not _DERIVE:
            return animation

        key = _anchor_setting_key(anchor)
        if not key:
            return None
        try:
            from spacr.setting_animations import (
                SettingAnimationError, animation_for_setting,
            )
            return animation_for_setting(str(key))
        except SettingAnimationError:
            LOGGER.exception(
                "Setting animation registry is invalid; %s keeps text help "
                "only", key,
            )
            return None

    def _set_animation(self, animation) -> None:
        """Show ``animation`` beside the text, or fall back to text only."""
        self._offered_animation = animation
        revealed = self.animations_shown()
        if animation is not None and revealed:
            # The only line in this class that reads a GIF, and it is reached
            # only from a press or from the preference being on.
            showing = self._animation_view.load(animation)
        elif (animation is not None
                and self._animation_view.slug() == animation.slug):
            # Folded away while still on the same setting: pause, keep the
            # finished pixmaps (~3.5 MB for one animation), so pressing again
            # costs nothing. Bounded at one animation and never filled before
            # a press -- moving to any other setting hits the branch below.
            self._animation_view.stop()
            showing = False
        else:
            # Nothing to show, or not asked for: decode nothing and drop the
            # previous setting's frames. This is the default path, and it is
            # why a plain hover costs no decode and holds no pixmaps.
            self._animation_view.clear_animation()
            showing = False
        self._animation = animation if showing else None
        self._animation_view.setVisible(showing)
        # Offered but hidden -> the word is the invitation. Showing -> the
        # word folds it away again. Revealed but undecodable -> hide it: a
        # word that visibly does nothing is worse than no word. No animation
        # for this setting at all -> nothing to say.
        self._animation_link.setVisible(
            animation is not None and (showing or not revealed))
        # Hidden, not merely empty: a zero-height row still costs the layout
        # its spacing, which is exactly the slack a text-only popup was asked
        # to lose.
        self._links.setVisible(
            self._api_link.isVisibleTo(self._links)
            or self._animation_link.isVisibleTo(self._links))
        self._resize_text_column(showing)

    def _text_height_at(self, width: int) -> int:
        """Height the prose plus the two words need at ``width`` pixels."""
        prose = max(0, self._label.heightForWidth(width))
        return prose + SPACING["xs"] + self._links.sizeHint().height()

    def _fitting_text_width(self) -> int:
        """Narrowest column width whose text fits inside the square's height."""
        for width in self.TEXT_WIDTH_STEPS:
            if self._text_height_at(width) <= self.ANIMATION_SIZE:
                return width
        return self.TEXT_WIDTH

    def _unpin_text(self) -> None:
        """Drop every explicit size on the text block.

        Not housekeeping — a precondition of measuring it. ``QLabel``'s
        ``heightForWidth`` ends in ``expandedTo(minimumSize())``, so a label
        still pinned by the *previous* hover answers every width with the
        previous hover's height, and the width ladder below then reads the
        same number at 220 px and at 900 px and never finds a fit.
        """
        self._label.setMinimumSize(0, 0)
        self._label.setMaximumSize(_UNBOUNDED, _UNBOUNDED)
        self._links.setMinimumWidth(0)
        self._links.setMaximumWidth(_UNBOUNDED)
        self._text_column.setMinimumSize(0, 0)
        self._text_column.setMaximumSize(_UNBOUNDED, _UNBOUNDED)

    def _resize_text_column(self, with_animation: bool) -> None:
        """Size the text block: square-high beside an animation, tight alone.

        Polished first, and that is not a formality. ``_apply_theme`` sets the
        prose font through the popup's stylesheet, and an unpolished widget
        answers ``heightForWidth`` in the *application* font — 119 px where
        the real answer is 250. The first hover of the session therefore
        picked the narrowest column in the ladder and then pinned the label to
        a height barely half the text, clipping the help. Every later hover
        measured correctly, so the fault only ever showed on the first one.
        """
        self.ensurePolished()
        self._label.ensurePolished()
        self._links.ensurePolished()
        self._unpin_text()
        if not with_animation:
            # Nothing pinned: the layout takes the popup down to what the
            # prose and the two words actually occupy.
            self._label.setMaximumWidth(self.TEXT_WIDTH)
            return
        width = self._fitting_text_width()
        self._label.setFixedWidth(width)
        # QLabel's own size hint for wrapped rich text is derived from its
        # preferred, not its actual, width; without this the popup opens tall
        # enough for a couple of lines and clips the rest.
        prose = max(0, self._label.heightForWidth(width))
        needed = prose + SPACING["xs"] + self._links.sizeHint().height()
        self._label.setFixedHeight(prose)
        self._links.setFixedWidth(width)
        self._text_column.setFixedWidth(width)
        # Exactly the height of the square. `max` only matters for prose too
        # long to fit even at the widest step, where the alternative would be
        # truncating the user's help text.
        self._text_column.setFixedHeight(max(self.ANIMATION_SIZE, needed))

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def _position_under(self, anchor: Optional[QWidget]) -> None:
        """Dock the popup just below ``anchor``, clamped to its screen."""
        try:
            below_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
        except (AttributeError, RuntimeError):
            below_left = QPoint(0, 0)
        screen = QGuiApplication.screenAt(below_left) \
            or QGuiApplication.primaryScreen()
        if screen is None:
            self.move(below_left)
            return
        geo = screen.availableGeometry()
        x = min(max(geo.left(), below_left.x()), geo.right() - self.width())
        y = below_left.y() + 4
        if y + self.height() > geo.bottom():
            # Not enough space below — flip above
            try:
                top = anchor.mapToGlobal(anchor.rect().topLeft()).y()
            except (AttributeError, RuntimeError):
                top = below_left.y()
            y = top - self.height() - 4
        self.move(x, y)

    # ------------------------------------------------------------------
    # The one-tooltip rule
    # ------------------------------------------------------------------
    def _claim_anchor(self, anchor: Optional[QWidget]) -> None:
        """Take the anchor's tooltip duty away from Qt's own popup.

        Idempotent by construction: Qt keeps a *list* of event filters and
        calls each installation separately, so the remove-then-install pair
        is what stops a re-hovered label from stacking suppressors. It also
        keeps this filter LAST installed and therefore FIRST called, ahead of
        the screen's own filter.
        """
        if anchor is None:
            return
        try:
            anchor.removeEventFilter(self._tooltip_suppressor)
            anchor.installEventFilter(self._tooltip_suppressor)
        except RuntimeError:
            # The anchor's C++ half is gone; there is no tooltip to suppress.
            return
        # A native tooltip already on screen — from the widget the pointer
        # crossed on its way here — would otherwise sit over this one.
        QToolTip.hideText()

    # ------------------------------------------------------------------
    def _maybe_hide(self) -> None:
        if self.underMouse():
            return
        # `self._anchor is not None` is not enough. The tooltip is a
        # process-wide singleton holding a plain reference to a widget it does
        # not own, and the hide is deferred by a timer -- so hovering a
        # settings label and switching module inside the delay destroys the
        # anchor's C++ object while this timer is still pending. The Python
        # wrapper survives, so the None check passes, and underMouse() then
        # raises RuntimeError('Internal C++ object already deleted') inside
        # the Qt event loop, where there is nobody to catch it.
        anchor = self._anchor
        if anchor is not None:
            try:
                if anchor.underMouse():
                    return
            except RuntimeError:
                # The anchored widget is gone; nothing can be hovering it.
                self._anchor = None
        self.hide()

    def hideEvent(self, event):
        """Stop decoding frames the moment the popup leaves the screen.

        The popup is a singleton, so without this its timer would keep
        swapping pixmaps into an invisible label for the rest of the session
        after the last hover.
        """
        self._animation_view.stop()
        super().hideEvent(event)

    def showEvent(self, event):
        """Resume the loaded animation when the popup comes back."""
        super().showEvent(event)
        if self._animation is not None:
            self._animation_view.play()

    def enterEvent(self, event):
        """Cancel the hide timer when the cursor enters the popup."""
        self.cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Restart the hide timer with a short delay when the cursor leaves."""
        self.start_hide(delay_ms=100)
        super().leaveEvent(event)


__all__ = ["HoverTooltip", "TEAL", "split_api_link"]
