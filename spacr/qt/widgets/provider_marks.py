"""Draw the assistant providers as their marks rather than as their names.

    "the claude gpt and gemeni buttons should be the logos for each not just
     buttons"

THREE WORDS IN THREE BOXES IS A LIST WEARING BUTTONS. A mark is recognised
before it is read, which is the whole reason a vendor has one -- and this is
the one question on the setup screen where the user already knows the answer
and only has to point at it.

DRAWN, NOT SHIPPED. Each mark is a `QPainterPath` built here: a bitmap would
be a vendor asset in the repository with its own licence, its own resolution
and its own two versions for light and dark. A path scales to any button
size, takes the brand colour when the provider is available and the theme's
muted ink when it is not, and adds nothing to the wheel.

The shapes are the recognisable geometry of each mark, not a facsimile:
Claude's radiating burst, OpenAI's six-fold knot, Gemini's four-pointed
spark. They identify whose command-line tool spaCR would drive, which is
what the question is asking.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import QSizePolicy, QWidget

#: Each provider's brand colour, used only when its CLI is actually here.
#:
#: AN UNAVAILABLE PROVIDER IS DRAWN IN THE THEME'S MUTED INK, not in a greyed
#: version of its own colour: the question is "which of these can I use", and
#: colour is the fastest answer to it.
BRAND: Dict[str, str] = {
    "claude": "#D97757",
    "gpt": "#10A37F",
    "gemini": "#4285F4",
    # GitHub's mark is monochrome, so "in colour" means in the theme's own
    # ink rather than in a brand hue -- "if the colour of the icon is black
    # and white go from grey to black and white". Signed out it takes the
    # muted ink like any other unavailable mark.
    "github": "#181717",
}

#: How much of the widget the mark occupies, leaving room for the label.
MARK_FRACTION = 0.52


def claude_path(box: QRectF) -> QPainterPath:
    """A radiating burst: petals narrow at the hub and wide at the tip.

    WIDE AT THE OUTSIDE, which is both the shape of the mark and what
    makes it carry. Drawn as tapered points -- narrow at the tip, all the
    area at the centre -- it measured a third of the ink of the knot beside
    it, and the lightest of three logos in a row reads as the disabled one.
    """
    path = QPainterPath()
    centre = box.center()
    outer = min(box.width(), box.height()) / 2.0
    hub = outer * 0.14
    spokes = 8
    for index in range(spokes):
        angle = 2.0 * math.pi * index / spokes - math.pi / 2.0
        near = math.pi / spokes * 0.20
        far = math.pi / spokes * 0.58
        points = (
            (hub, angle - near), (outer, angle - far),
            (outer, angle + far), (hub, angle + near),
        )
        for step, (radius, at) in enumerate(points):
            point = QPointF(centre.x() + radius * math.cos(at),
                            centre.y() + radius * math.sin(at))
            path.moveTo(point) if step == 0 else path.lineTo(point)
        path.closeSubpath()
    return path


def gpt_path(box: QRectF) -> QPainterPath:
    """A six-fold knot: six lobes on a common ring, drawn as one outline."""
    path = QPainterPath()
    centre = box.center()
    radius = min(box.width(), box.height()) / 2.0
    ring = radius * 0.56
    lobe = radius * 0.44
    for index in range(6):
        angle = 2.0 * math.pi * index / 6.0 - math.pi / 2.0
        at = QPointF(centre.x() + ring * math.cos(angle),
                     centre.y() + ring * math.sin(angle))
        path.addEllipse(at, lobe, lobe)
    return path


def gemini_path(box: QRectF) -> QPainterPath:
    """A four-pointed spark: a diamond with concave sides."""
    path = QPainterPath()
    centre = box.center()
    reach = min(box.width(), box.height()) / 2.0
    waist = reach * 0.30
    points = []
    for index in range(4):
        angle = 2.0 * math.pi * index / 4.0 - math.pi / 2.0
        points.append(QPointF(centre.x() + reach * math.cos(angle),
                              centre.y() + reach * math.sin(angle)))
    path.moveTo(points[0])
    for index in range(4):
        nxt = points[(index + 1) % 4]
        # The control point sits near the centre, which is what pulls each
        # side inward and makes the four points read as a spark rather than
        # as a plain diamond.
        between = 2.0 * math.pi * (index + 0.5) / 4.0 - math.pi / 2.0
        path.quadTo(QPointF(centre.x() + waist * math.cos(between),
                            centre.y() + waist * math.sin(between)), nxt)
    path.closeSubpath()
    return path


def github_path(box: QRectF) -> QPainterPath:
    """The Octocat silhouette, reduced to a circle with ears and a tail.

    Not a traced logo: a recognisable mark drawn from primitives, like every
    other mark here. The head is the circle, two arcs above it are the ears,
    and the tail is the stroke that makes it read as the GitHub mark rather
    than as a plain dot.
    """
    path = QPainterPath()
    centre = box.center()
    reach = min(box.width(), box.height()) / 2.0
    head = reach * 0.82
    path.addEllipse(centre, head, head)

    # The two ears, as small circles just inside the top of the head.
    ear = head * 0.30
    for side in (-1.0, 1.0):
        path.addEllipse(QPointF(centre.x() + side * head * 0.52,
                                centre.y() - head * 0.62), ear, ear)

    # The tail, curving down and out from the lower right of the head.
    tail = QPainterPath()
    start = QPointF(centre.x() + head * 0.10, centre.y() + head * 0.62)
    tail.moveTo(start)
    tail.quadTo(QPointF(centre.x() + head * 0.72, centre.y() + head * 0.88),
                QPointF(centre.x() + head * 0.86, centre.y() + reach * 0.98))
    tail.quadTo(QPointF(centre.x() + head * 0.62, centre.y() + head * 0.96),
                start)
    path.addPath(tail)
    return path


#: Provider code -> the function that draws its mark.
MARKS = {
    "claude": claude_path,
    "gpt": gpt_path,
    "gemini": gemini_path,
    "github": github_path,
}


def mark_for(code: str, box: QRectF) -> Optional[QPainterPath]:
    """The path for ``code`` inside ``box``, or None for an unknown one."""
    draw = MARKS.get(str(code or ""))
    return draw(box) if draw is not None else None


class ProviderMark(QWidget):
    """One provider, drawn as its mark with its name beneath.

    Selectable rather than clickable-once: three of these are a radio group,
    and :meth:`set_chosen` is how the group tells this one it lost.

    :param code: provider key -- ``claude``, ``gpt`` or ``gemini``.
    :param label: the name under the mark.
    :param available: whether the vendor CLI is on PATH. This is SHOWN, not
        enforced: an unavailable provider is drawn in the muted ink and says
        what would install it, and is still choosable. The setup screen
        writes a PREFERENCE and launches nothing, so choosing a provider
        before installing its CLI is an ordinary thing to want, and the
        console says so at the point it would actually be used.
    """

    # Gating the choice on availability was reported 2026-08-22 -- "for the
    # ai assistant i can only click claude" -- and the report is right: a
    # preference is not a launch.

    #: Emitted with the provider code when this mark is chosen.
    chosen = Signal(str)

    def __init__(self, code: str, label: str, available: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.code = str(code)
        self.label = str(label)
        self.available = bool(available)
        self._chosen = False
        self._hovered = False
        self.setMinimumSize(QSize(72, 82))
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_Hover, True)
        # EVERY MARK IS CLICKABLE, so every one gets the hand.
        self.setCursor(Qt.PointingHandCursor)

    # ------------------------------------------------------------- state

    def is_chosen(self) -> bool:
        """Whether this provider is the selected one."""
        return self._chosen

    def set_chosen(self, chosen: bool) -> None:
        """Select or deselect, and repaint."""
        chosen = bool(chosen)
        if chosen != self._chosen:
            self._chosen = chosen
            self.update()

    # ------------------------------------------------------------ events

    def mousePressEvent(self, event):           # noqa: N802 - Qt naming
        # AVAILABILITY DOES NOT GATE THE CHOICE. It is drawn -- brand colour
        # when the CLI is here, muted ink when it is not -- and said in the
        # tooltip, which is information rather than obstruction.
        if event.button() == Qt.LeftButton:
            self.chosen.emit(self.code)
        super().mousePressEvent(event)

    def event(self, event):
        from PySide6.QtCore import QEvent

        if event.type() == QEvent.Type.HoverEnter:
            self._hovered = True
            self.update()
        elif event.type() == QEvent.Type.HoverLeave:
            self._hovered = False
            self.update()
        return super().event(event)

    # ----------------------------------------------------------- painting

    def sizeHint(self) -> QSize:                # noqa: N802 - Qt naming
        return QSize(88, 92)

    def paintEvent(self, event):                # noqa: N802 - Qt naming
        try:
            self._paint()
        except Exception:                        # pragma: no cover
            # Decoration is never load-bearing: an unpainted mark is still a
            # control that answers the question when it is clicked.
            pass

    def _colours(self) -> Tuple[QColor, QColor]:
        """``(ink, halo)`` for the current state."""
        from ..theme import active_palette

        palette = active_palette()
        if not self.available:
            ink = QColor(palette.get("muted", palette["fg"]))
            ink.setAlpha(110)
            return ink, QColor(0, 0, 0, 0)
        ink = QColor(BRAND.get(self.code, palette["accent"]))
        halo = QColor(ink)
        if self._chosen:
            halo.setAlpha(46)
        elif self._hovered:
            halo.setAlpha(22)
        else:
            halo.setAlpha(0)
            ink.setAlpha(200)
        return ink, halo

    def _paint(self) -> None:
        from ..theme import active_palette

        palette = active_palette()
        ink, halo = self._colours()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            rect = QRectF(self.rect())

            if halo.alpha():
                painter.setPen(Qt.NoPen)
                painter.setBrush(halo)
                painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 12, 12)
            if self._chosen:
                pen = painter.pen()
                pen.setColor(QColor(ink))
                pen.setWidthF(1.6)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 12, 12)

            side = min(rect.width(), rect.height()) * MARK_FRACTION
            box = QRectF(rect.center().x() - side / 2.0,
                         rect.top() + rect.height() * 0.14,
                         side, side)
            path = mark_for(self.code, box)
            if path is not None:
                painter.setPen(Qt.NoPen)
                painter.setBrush(ink)
                painter.drawPath(path)

            text_ink = QColor(palette["fg"] if self.available
                              else palette.get("muted", palette["fg"]))
            if not self.available:
                text_ink.setAlpha(150)
            painter.setPen(text_ink)
            painter.drawText(
                QRectF(rect.left(), box.bottom() + 4.0,
                       rect.width(), rect.bottom() - box.bottom() - 4.0),
                Qt.AlignHCenter | Qt.AlignTop, self.label)
        finally:
            painter.end()
