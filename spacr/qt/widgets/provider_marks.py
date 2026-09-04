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

The three assistant shapes are the recognisable geometry of each mark, not
a facsimile: Claude's radiating burst, OpenAI's six-fold knot, Gemini's
four-pointed spark. They identify whose command-line tool spaCR would
drive, which is what the question is asking.

GITHUB IS THE EXCEPTION, and deliberately. GitHub publishes the Octocat's
path data for use as a link mark, so that one is the real thing rather
than a circle with ears -- still a path, still recoloured per theme,
still nothing added to the wheel.
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QColor, QPainter, QPainterPath,
                           QTransform)
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
    # muted ink like any other unavailable mark. The value here is the
    # LIGHT-theme ink; on a dark theme `_ready_ink` inverts it, because
    # GitHub's own near-black on a near-black card is an invisible mark.
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


#: GitHub's own mark: the path out of `Octicons-mark-github.svg`.
#:
#: TRACED FROM NOTHING AND INVENTED FROM NOTHING. The three assistant
#: marks are drawn from primitives because their logos are not ours to
#: reproduce; this one is the octicons mark itself, which GitHub ships
#: under MIT for exactly this use. So the Octocat is the Octocat rather
#: than a circle with ears, which is what stood here.
#:
#: Copied verbatim from the file, including its 16-unit coordinates --
#: the SVG carries a `scale(64)` to reach its 1024 viewBox, and this is
#: scaled into whatever box it is asked for instead.
GITHUB_MARK = (
    "M8 0C3.58 0 0 3.58 0 8C0 11.54 2.29 14.53 5.47 15.59C5.87 15.6"
    "6 6.02 15.42 6.02 15.21C6.02 15.02 6.01 14.39 6.01 13.72C4 14."
    "09 3.48 13.23 3.32 12.78C3.23 12.55 2.84 11.84 2.5 11.65C2.22 "
    "11.5 1.82 11.13 2.49 11.12C3.12 11.11 3.57 11.7 3.72 11.94C4.4"
    "4 13.15 5.59 12.81 6.05 12.6C6.12 12.08 6.33 11.73 6.56 11.53C"
    "4.78 11.33 2.92 10.64 2.92 7.58C2.92 6.71 3.23 5.99 3.74 5.43C"
    "3.66 5.23 3.38 4.41 3.82 3.31C3.82 3.31 4.49 3.1 6.02 4.13C6.6"
    "6 3.95 7.34 3.86 8.02 3.86C8.7 3.86 9.38 3.95 10.02 4.13C11.55"
    " 3.09 12.22 3.31 12.22 3.31C12.66 4.41 12.38 5.23 12.3 5.43C12"
    ".81 5.99 13.12 6.7 13.12 7.58C13.12 10.65 11.25 11.33 9.47 11."
    "53C9.76 11.78 10.01 12.26 10.01 13.01C10.01 14.08 10 14.94 10 "
    "15.21C10 15.42 10.15 15.67 10.55 15.59C13.71 14.53 16 11.53 16"
    " 8C16 3.58 12.42 0 8 0Z"
)

#: Every token an SVG path can hold: one command letter, or one number.
_SVG_TOKEN = re.compile(r"[MmZzLlHhVvCcSsQqTtAa]|-?\d*\.?\d+(?:e[-+]?\d+)?")

#: The parsed mark, kept because parsing it on every paint is waste.
_GITHUB_PATH: Optional[QPainterPath] = None


def _parse_svg_path(d: str) -> QPainterPath:
    """An SVG path's ``d`` attribute as a QPainterPath.

    Supports the commands the marks here actually use -- moves, lines,
    cubics, smooth cubics, arcs and close. Anything else raises rather
    than drawing something wrong quietly.
    """
    tokens = _SVG_TOKEN.findall(d)
    path = QPainterPath()
    i = 0
    cmd = None
    cur = QPointF(0, 0)
    start = QPointF(0, 0)
    last_c = None
    def num():
        """Read the next number from the path string, advancing the cursor."""
        nonlocal i
        v = float(tokens[i]); i += 1; return v
    while i < len(tokens):
        if re.match(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]; i += 1
            if cmd in "Zz":
                path.closeSubpath(); cur = QPointF(start); continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            x, y = num(), num()
            p = QPointF(cur.x()+x, cur.y()+y) if rel else QPointF(x, y)
            path.moveTo(p); cur = p; start = QPointF(p)
            cmd = "l" if rel else "L"
        elif c == "L":
            x, y = num(), num()
            p = QPointF(cur.x()+x, cur.y()+y) if rel else QPointF(x, y)
            path.lineTo(p); cur = p
        elif c == "H":
            x = num(); p = QPointF(cur.x()+x if rel else x, cur.y())
            path.lineTo(p); cur = p
        elif c == "V":
            y = num(); p = QPointF(cur.x(), cur.y()+y if rel else y)
            path.lineTo(p); cur = p
        elif c == "C":
            x1,y1,x2,y2,x,y = (num() for _ in range(6))
            if rel:
                c1 = QPointF(cur.x()+x1, cur.y()+y1); c2 = QPointF(cur.x()+x2, cur.y()+y2)
                p = QPointF(cur.x()+x, cur.y()+y)
            else:
                c1 = QPointF(x1,y1); c2 = QPointF(x2,y2); p = QPointF(x,y)
            path.cubicTo(c1, c2, p); last_c = c2; cur = p
        elif c == "S":
            x2,y2,x,y = (num() for _ in range(4))
            if rel:
                c2 = QPointF(cur.x()+x2, cur.y()+y2); p = QPointF(cur.x()+x, cur.y()+y)
            else:
                c2 = QPointF(x2,y2); p = QPointF(x,y)
            c1 = QPointF(2*cur.x()-last_c.x(), 2*cur.y()-last_c.y()) if last_c else QPointF(cur)
            path.cubicTo(c1, c2, p); last_c = c2; cur = p
        elif c == "A":
            rx, ry, rot, laf, sf, x, y = (num() for _ in range(7))
            p = QPointF(cur.x()+x, cur.y()+y) if rel else QPointF(x, y)
            _svg_arc(path, cur, p, rx, ry, rot, int(laf), int(sf))
            cur = p
        else:
            raise ValueError(f"unsupported command {cmd!r}")
    return path


def _svg_arc(path: QPainterPath, p0: QPointF, p1: QPointF,
             rx: float, ry: float, rot_deg: float,
             large: int, sweep: int) -> None:
    """SVG endpoint arc -> cubic segments (endpoint to centre conversion)."""
    if rx == 0 or ry == 0:
        path.lineTo(p1); return
    phi = math.radians(rot_deg)
    dx2, dy2 = (p0.x()-p1.x())/2.0, (p0.y()-p1.y())/2.0
    x1 = math.cos(phi)*dx2 + math.sin(phi)*dy2
    y1 = -math.sin(phi)*dx2 + math.cos(phi)*dy2
    rx, ry = abs(rx), abs(ry)
    lam = x1*x1/(rx*rx) + y1*y1/(ry*ry)
    if lam > 1:
        rx *= math.sqrt(lam); ry *= math.sqrt(lam)
    num_ = rx*rx*ry*ry - rx*rx*y1*y1 - ry*ry*x1*x1
    den = rx*rx*y1*y1 + ry*ry*x1*x1
    co = math.sqrt(max(0.0, num_/den)) * (-1 if large == sweep else 1)
    cx1 = co * rx * y1 / ry
    cy1 = -co * ry * x1 / rx
    cx = math.cos(phi)*cx1 - math.sin(phi)*cy1 + (p0.x()+p1.x())/2.0
    cy = math.sin(phi)*cx1 + math.cos(phi)*cy1 + (p0.y()+p1.y())/2.0
    def ang(ux, uy, vx, vy):
        """The signed angle between two vectors, as SVG's arc maths defines it."""
        dot = ux*vx + uy*vy
        n = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot/n)))
        return -a if ux*vy - uy*vx < 0 else a
    th1 = ang(1, 0, (x1-cx1)/rx, (y1-cy1)/ry)
    dth = ang((x1-cx1)/rx, (y1-cy1)/ry, (-x1-cx1)/rx, (-y1-cy1)/ry)
    if not sweep and dth > 0: dth -= 2*math.pi
    elif sweep and dth < 0: dth += 2*math.pi
    steps = max(1, int(abs(dth) / (math.pi/2) + 1))
    delta = dth/steps
    t = 4.0/3.0 * math.tan(delta/4.0)
    for s in range(steps):
        a0 = th1 + s*delta; a1 = a0 + delta
        c0, s0 = math.cos(a0), math.sin(a0)
        c1_, s1 = math.cos(a1), math.sin(a1)
        def pt(c_, s_):
            """One point on the arc, from its cosine and sine."""
            return QPointF(cx + rx*math.cos(phi)*c_ - ry*math.sin(phi)*s_,
                           cy + rx*math.sin(phi)*c_ + ry*math.cos(phi)*s_)
        p_start = pt(c0, s0); p_end = pt(c1_, s1)
        d1 = QPointF(-rx*math.cos(phi)*s0 - ry*math.sin(phi)*c0,
                     -rx*math.sin(phi)*s0 + ry*math.cos(phi)*c0)
        d2 = QPointF(-rx*math.cos(phi)*s1 - ry*math.sin(phi)*c1_,
                     -rx*math.sin(phi)*s1 + ry*math.cos(phi)*c1_)
        path.cubicTo(QPointF(p_start.x()+t*d1.x(), p_start.y()+t*d1.y()),
                     QPointF(p_end.x()-t*d2.x(), p_end.y()-t*d2.y()), p_end)


def github_path(box: QRectF) -> QPainterPath:
    """GitHub's Octocat mark, scaled to fill ``box``.

    Uniform scale on the larger of the two ratios and centred, so the cat
    keeps its proportions in a box of any shape.
    """
    global _GITHUB_PATH
    if _GITHUB_PATH is None:
        _GITHUB_PATH = _parse_svg_path(GITHUB_MARK)
    bounds = _GITHUB_PATH.boundingRect()
    if bounds.isEmpty() or box.isEmpty():
        return QPainterPath(_GITHUB_PATH)
    scale = min(box.width() / bounds.width(), box.height() / bounds.height())
    transform = QTransform()
    transform.translate(
        box.center().x() - bounds.center().x() * scale,
        box.center().y() - bounds.center().y() * scale)
    transform.scale(scale, scale)
    return transform.map(_GITHUB_PATH)


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
    :param parent: parent widget; ownership only.
    :param status: ``READY``, ``SIGNED_OUT`` or ``NOT_INSTALLED``. Empty
        DERIVES ONE FROM ``available``, which is what every caller written
        before the third state did; that derivation can only ever produce
        ready-or-not-installed, so a caller that knows the CLI is present
        but signed out has to say so here.
    """

    # Gating the choice on availability was reported 2026-08-22 -- "for the
    # ai assistant i can only click claude" -- and the report is right: a
    # preference is not a launch.

    #: Emitted with the provider code when this mark is chosen.
    chosen = Signal(str)

    #: What a provider can be, and what the user has to do about it.
    #:
    #: THREE STATES, NOT TWO. `available` was one boolean covering both "the
    #: CLI is missing" and "the CLI is there and signed out", and the two
    #: need different things from the user -- one an install, the other a
    #: sign-in. Drawn the same, they were also drawn as nearly nothing:
    #: muted ink at alpha 110 over a mark with no halo, which is what "GPT
    #: brings no text and no color just a rim" was.
    READY = "ready"
    SIGNED_OUT = "signed out"
    NOT_INSTALLED = "not installed"

    #: What the mark says under its name in each state. See the
    #: constructor's ``status`` for why it is not derived from
    #: ``available``.
    STATUS_TEXT = {
        READY: "",
        SIGNED_OUT: "sign in",
        NOT_INSTALLED: "install",
    }

    def __init__(self, code: str, label: str, available: bool = True,
                 parent: Optional[QWidget] = None, status: str = ""):
        super().__init__(parent)
        self.code = str(code)
        self.label = str(label)
        self.available = bool(available)
        #: One of READY / SIGNED_OUT / NOT_INSTALLED. Defaults from
        #: ``available`` so every existing caller keeps working.
        self.status = str(status) or (self.READY if available
                                      else self.NOT_INSTALLED)
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
        except Exception:
            # Decoration is never load-bearing: an unpainted mark is still a
            # control that answers the question when it is clicked.
            pass

    def _ready_ink(self, palette) -> QColor:
        """The mark's colour once the tool behind it is ready to use.

        A brand hue for the three assistants, and the READABLE end of
        monochrome for GitHub: white on a dark card, GitHub's own near-
        black on a light one. A mark drawn in #181717 on #0d0e10 is a
        signed-in state nobody can see.
        """
        if self.code != "github":
            return QColor(BRAND.get(self.code, palette["accent"]))
        # THE THEME'S OWN INK, which is the end of monochrome that can be
        # read against the card it sits on: white on a dark theme, near
        # black on a light one. GitHub's #181717 IS the light-theme value;
        # painted on a dark card it is a signed-in state nobody can see.
        return QColor(palette.get("fg", BRAND["github"]))

    def _colours(self) -> Tuple[QColor, QColor]:
        """``(ink, halo)`` for the current state."""
        from ..theme import active_palette

        palette = active_palette()
        if not self.available:
            # THE BRAND FILL IS WHAT "READY" LOOKS LIKE, so a provider that
            # is not installed does not get one: "GPT and Gemini should only
            # get their color fill when they are installed". The mark is
            # muted ink -- legible, at alpha 190 rather than the 110 that
            # made it a ghost -- and HOVER gives the brand background, so
            # the colour still tells you which provider you are pointing at.
            ink = QColor(palette.get("fg_muted", palette["fg"]))
            ink.setAlpha(190)
            halo = QColor(BRAND.get(self.code, palette["accent"]))
            halo.setAlpha(30 if self._hovered else 0)
            return ink, halo
        ink = self._ready_ink(palette)
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

            # THE NAME IS ALWAYS LEGIBLE. It used to fade with the mark, so
            # a provider that was not set up had no readable name either.
            painter.setPen(QColor(palette["fg"]))
            below = QRectF(rect.left(), box.bottom() + 4.0,
                           rect.width(), rect.bottom() - box.bottom() - 4.0)
            painter.drawText(below, Qt.AlignHCenter | Qt.AlignTop, self.label)

            # AND WHAT TO DO ABOUT IT, under the name, in the brand colour.
            # A greyed control that does not say why is the thing this
            # replaces.
            note = self.STATUS_TEXT.get(self.status, "")
            if note:
                from ..i18n import tr

                small = painter.font()
                small.setPointSizeF(max(6.5, small.pointSizeF() - 2.0))
                painter.setFont(small)
                painter.setPen(QColor(BRAND.get(self.code,
                                                palette["accent"])))
                painter.drawText(
                    below.adjusted(0, painter.fontMetrics().height() + 2,
                                   0, 0),
                    Qt.AlignHCenter | Qt.AlignTop, tr(note))
        finally:
            painter.end()
