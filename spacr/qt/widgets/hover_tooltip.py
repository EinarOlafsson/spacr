"""
HoverTooltip — a QFrame-based popup that stays visible when the mouse
enters it. Unlike QToolTip, users can move their cursor into the popup
to click links inside.

Usage:
    tip = HoverTooltip.instance()
    tip.show_for(some_widget, "some html")   # on hover-enter
    tip.start_hide()                          # on hover-leave
The popup cancels its own hide timer if the mouse enters it, and only
actually hides when neither the anchor nor the popup itself is under
the cursor.

Layout: explanation on the left, the setting's animation on the right.
Two columns of the same width, both starting at the same top edge, so the
first line of prose and the first frame of the animation are read together
rather than one being hunted for beside the other::

    +-----------------------------+-----------------------------+
    | Cell diameter (int)         |                             |
    | Expected cell diameter in   |         (animation)         |
    | pixels...                   |                             |
    | Open spaCR API documenta... |                             |
    +-----------------------------+-----------------------------+

Which animation is decided by the anchor's ``settingKey`` property, so no
caller has to pass one; the callers that put help on a label already set it.
Anything without that property — a section header, a home tile — gets the
text-only popup it always got. So does everything, if the user turns
:func:`spacr.qt.preferences.get_setting_animations_enabled` off.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from ..theme import SPACING, active_palette


LOGGER = logging.getLogger(__name__)

#: Sentinel for "work the animation out from the anchor". ``None`` cannot do
#: that job: it is the perfectly good answer "this tooltip has no animation",
#: which callers need to be able to say.
_DERIVE = object()


class _AnimationView(QLabel):
    """Plays one pre-zoomed setting animation at a fixed square size.

    Not a ``QMovie``: the frames are cropped and rescaled from the packaged
    GIF before they are shown (see
    :mod:`spacr.qt.widgets.animation_zoom`), which ``QMovie`` cannot do. It
    plays finished frames on a timer instead, honouring each frame's own
    delay so the generated timing survives.
    """

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
            QPixmap.fromImage(to_qimage(frame)) for frame in zoomed.frames
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

    #: Side of the square animation box, and therefore the width of the text
    #: column beside it — the two are the same width by design, so the popup
    #: reads as two equal panels rather than a caption with a picture stuck
    #: on the end.
    ANIMATION_SIZE = 220

    #: Width the text uses when there is no animation to sit beside. The
    #: narrow column only makes sense as half of a pair.
    TEXT_WIDTH = 380

    def __init__(self):
        # Popup window with tool-tip semantics but our own paint control.
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint,
        )
        self.setObjectName("HoverTooltip")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._apply_theme()
        self._label = QLabel(self)
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
        self._animation_view = _AnimationView(self.ANIMATION_SIZE, self)
        self._animation_view.hide()
        self._animation = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(SPACING["sm"], SPACING["xs"],
                                SPACING["sm"], SPACING["xs"])
        lay.setSpacing(SPACING["sm"])
        # AlignTop on both, so the text starts level with the first frame
        # instead of floating to the middle of a 220-pixel square.
        lay.addWidget(self._label, 0, Qt.AlignTop)
        lay.addWidget(self._animation_view, 0, Qt.AlignTop)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._maybe_hide)
        self._anchor: Optional[QWidget] = None

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
            f"QLabel {{"
            f"  color: {palette['fg']};"
            f"  font-size: 12px;"
            f"  background: transparent;"
            f"}}"
            # The animations are drawn on black and only make sense on it;
            # over a light surface they would sit in a grey haze. No border:
            # the label is sized to the pixmap exactly, and a border would
            # eat into the content rect and clip a pixel off every edge.
            f"QLabel#SettingTooltipAnimation {{"
            f"  background: #000000;"
            f"  border-radius: 4px;"
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
        :param html: rich-text body; empty strings are ignored.
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
        self._label.setText(html)
        self._set_animation(self._resolve_animation(anchor, animation))
        self.adjustSize()
        # Position: just below the anchor, left-aligned to its left edge,
        # clamped to the screen so we never overflow.
        try:
            below_left = anchor.mapToGlobal(anchor.rect().bottomLeft())
        except Exception:
            below_left = QPoint(0, 0)
        screen = QGuiApplication.screenAt(below_left) \
                 or QGuiApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = min(max(geo.left(), below_left.x()),
                    geo.right() - self.width())
            y = below_left.y() + 4
            if y + self.height() > geo.bottom():
                # Not enough space below — flip above
                y = anchor.mapToGlobal(anchor.rect().topLeft()).y() \
                    - self.height() - 4
            self.move(x, y)
        else:
            self.move(below_left)
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

    def animation_view(self) -> _AnimationView:
        """The square animation panel — exposed for layout tests."""
        return self._animation_view

    def text_label(self) -> QLabel:
        """The explanation panel — exposed for layout tests."""
        return self._label

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    def _resolve_animation(self, anchor: QWidget, animation):
        """Decide which animation, if any, belongs beside this tooltip."""
        from ..preferences import get_setting_animations_enabled

        if not get_setting_animations_enabled():
            # Checked first and every time. The tooltip is a singleton that
            # outlives Preferences, so a cached answer would keep animating
            # for the rest of the session after the user cleared the box.
            return None
        if animation is not _DERIVE:
            return animation

        try:
            key = anchor.property("settingKey") if anchor is not None else None
        except RuntimeError:
            # The anchor's C++ half is gone; there is nothing to look up.
            return None
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
        showing = self._animation_view.load(animation)
        self._animation = animation if showing else None
        self._animation_view.setVisible(showing)
        if showing:
            # Equal columns: the text is exactly as wide as the square.
            width = self.ANIMATION_SIZE
            self._label.setFixedWidth(width)
        else:
            # Undo the fixed width — setFixedWidth pins the minimum too, and
            # a text-only tooltip pinned to 220 px would be a tall ribbon.
            width = self.TEXT_WIDTH
            self._label.setMinimumWidth(0)
            self._label.setMaximumWidth(width)
        # QLabel's own size hint for wrapped rich text is derived from its
        # preferred, not its actual, width; without this the popup opens tall
        # enough for a couple of lines and clips the rest.
        self._label.setMinimumHeight(max(0, self._label.heightForWidth(width)))

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
