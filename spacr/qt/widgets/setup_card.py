"""Render first-run questions in a rounded, pointer-responsive card.

The accent follows the corner nearest the pointer. All visual effects are
decorative: if a blur, background, or accent cannot be painted, the card
continues to display its questions and save their answers.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

#: Corner order: top-left, top-right, bottom-right, bottom-left.
CORNERS = ("topLeft", "topRight", "bottomRight", "bottomLeft")

#: The surface the arc preference is measured against, in pixels.
#:
#: The first-run window's own size. It is the card the look was tuned on,
#: so every other card lights the same fraction of its rim as this one
#: does -- see :meth:`SetupCard.accent_span`.
REFERENCE_CARD = (980.0, 700.0)


class SetupCard(QWidget):
    """A rounded, translucent card whose border lights at the near corner.

    :param radius: corner radius in pixels.
    :param arc: how far along each edge the accent runs, in pixels.
    """

    #: Fallback chase fraction, when the preference cannot be read.
    #:
    #: THE PREFERENCE IS THE REAL ONE -- see `spacr.qt.preferences.
    #: get_rim_lag`. This is what the card uses when there is no settings
    #: store to ask, which is the case in a bare widget test.
    EASE = 0.16

    #: Frames the tail is smeared over, as a fraction of the arc.
    #:
    #: THE ENDS FADE OUT rather than stopping. A run of rim at one alpha
    #: has two hard
    #: ends that arrive and leave abruptly; a run that fades has none, which
    #: is what makes it read as a highlight travelling rather than as a
    #: segment being switched on.
    FADE = 0.5

    #: Rim pixels per drawn segment.
    #:
    #: The stroke is many short lines because a QPen carries ONE colour and
    #: this run has to fade along its length. Below about six pixels the
    #: steps are past what the eye resolves against an anti-aliased edge;
    #: four is comfortably under it.
    STEP_PX = 4.0

    #: Never more segments than this, however large the card.
    #:
    #: A ceiling rather than a guess: at sixty frames a second the cost is
    #: paid every frame, and past this the extra lines are drawing detail
    #: finer than the anti-aliasing that is already smoothing them.
    MAX_STEPS = 320

    def __init__(self, parent: Optional[QWidget] = None, *,
                 radius: int = 18, arc: Optional[int] = None,
                 lag: Optional[float] = None, align: str = "",
                 mode: str = ""):
        super().__init__(parent)
        self._mode = str(mode or "").strip().lower()
        #: Radians of the animation cycle, advanced one frame at a time.
        #:
        #: COUNTED IN FRAMES, not read off a clock. The timer is the only
        #: thing driving this and it ticks at a known interval, so a frame
        #: count is a phase -- and unlike `time.time()` it stops when the
        #: card is hidden, which is what keeps a pulse from jumping when a
        #: dialog is reopened.
        self._phase = 0.0
        self._radius = int(radius)
        # THE THREE THAT ARE A MATTER OF TASTE come from the preference
        # store unless the caller names one. How long the run is, how hard
        # it chases, and whether it is centred on the pointer or trails
        # behind it are all things to look at and decide about, so they are
        # settings rather than constants -- and a caller that wants a
        # particular look for a particular card can still say so.
        self._arc = int(arc) if arc is not None else self._preferred_arc()
        self._lag = float(lag) if lag is not None else None
        self._align = str(align or "").strip().lower()
        self._corner = 0
        #: Accent position as a clockwise fraction of the card perimeter,
        #: measured from the top-left corner. A continuous position lets the
        #: highlight move smoothly between corners.
        self._at = 0.0
        #: Where it is heading. The gap between the two is what the easing
        #: closes, one tick at a time.
        self._towards = 0.0
        #: Laps still to run, signed: + is clockwise, - anticlockwise.
        #: WHILE THIS IS NON-ZERO THE POINTER DOES NOT STEER. A lap dragged
        #: off course by a mouse movement is not a lap, and the user cannot
        #: tell whether it went round -- which is the whole signal.
        self._laps = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)             # ~60fps
        self._timer.timeout.connect(self._tick)
        # MOUSE TRACKING, or `mouseMoveEvent` only fires while a button is
        # held -- which is never, on a card the user is only reading.
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover, True)

    # ------------------------------------------------------------------
    def nearest_corner(self, point: QPointF) -> int:
        """Index into :data:`CORNERS` of the corner nearest ``point``.

        Ties go to the earlier corner, which only happens at the exact
        centre and is therefore never seen.
        """
        rect = QRectF(self.rect())
        points = (rect.topLeft(), rect.topRight(),
                  rect.bottomRight(), rect.bottomLeft())
        best, chosen = None, 0
        for index, corner in enumerate(points):
            dx = point.x() - corner.x()
            dy = point.y() - corner.y()
            distance = dx * dx + dy * dy
            if best is None or distance < best:
                best, chosen = distance, index
        return chosen

    def corner(self) -> str:
        """The corner the accent is currently on."""
        return CORNERS[self._corner]

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event):            # noqa: N802 - Qt naming
        self._follow(event.position())
        super().mouseMoveEvent(event)

    def event(self, event):
        # A hover move arrives even when the widget has no mouse grab, which
        # is the ordinary case here.
        try:
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.HoverMove:
                self._follow(event.position())
        except Exception:                        # pragma: no cover
            pass
        return super().event(event)

    def _follow(self, position) -> None:
        """Aim the accent at a point in this widget's own coordinates.

        IT USED TO SET ONLY `_corner`, which `_tick` recomputes from `_at`
        on the very next frame -- so while the pointer was over the CARD,
        which is nearly the whole dialog, nothing steered the rim at all.
        Only the dialog's margin, which has its own handler, ever moved it.
        That is the "unsynced on the inside" of the 2026-08-22 report.
        """
        self.flow_towards(QPointF(position))

    # ------------------------------------------------------------------
    def paintEvent(self, event):                # noqa: N802 - Qt naming
        try:
            self._paint()
        except Exception:                        # pragma: no cover
            # Decoration is never load-bearing: an unpainted card is still a
            # card with working controls on it.
            pass
        super().paintEvent(event)

    # ---------------------------------------------------- where it flows to

    def perimeter_position(self, point: QPointF):
        """Where on the rim ``point`` is, as a fraction clockwise from 0.

        THE RAY FROM THE CENTRE, not the nearest edge. Projecting a point
        onto whichever edge happens to be closest is discontinuous along
        the diagonals: the pointer crosses one and the target jumps from
        the middle of the top edge to the middle of the left, which is what
        was reported as the rim being unsynced with the pointer inside the
        card. The point where the ray from the centre through the pointer
        leaves the rectangle moves continuously, and is always the place on
        the rim that is actually in the pointer's direction -- which is what
        "flows towards the mouse" has to mean if the mouse can be anywhere.

        Because it is a RAY it needs no clamping, and answers for a pointer
        outside the card as readily as for one inside it.

        THE RIM IS TREATED AS A RECTANGLE, not as the rounded path it is
        drawn on. The corner arcs are a few pixels of a perimeter hundreds
        long, and paying for exact arc length here would buy an accuracy no
        eye can see on a value that is chasing a mouse anyway.

        :returns: the fraction, or None when the pointer is exactly at the
            centre and no direction can be read from it.
        """
        rect = QRectF(self.rect())
        width = max(1.0, rect.width())
        height = max(1.0, rect.height())
        half_w, half_h = width / 2.0, height / 2.0
        dx = float(point.x()) - half_w
        dy = float(point.y()) - half_h
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return None

        # Scale the ray until it touches whichever pair of sides it reaches
        # first. The larger of the two normalised components decides.
        span = max(abs(dx) / half_w, abs(dy) / half_h)
        x = half_w + dx / span
        y = half_h + dy / span
        # Rounding can leave it a hair outside; the run below assumes it is
        # on the boundary.
        x = min(max(x, 0.0), width)
        y = min(max(y, 0.0), height)

        total = 2.0 * (width + height)
        if y <= 1e-6:                       # top edge, left to right
            run = x
        elif x >= width - 1e-6:             # right edge, down
            run = width + y
        elif y >= height - 1e-6:            # bottom edge, right to left
            run = width + height + (width - x)
        else:                               # left edge, up
            run = 2.0 * width + height + (height - y)
        return (run / total) % 1.0

    def flow_towards(self, point: QPointF) -> None:
        """Aim the accent at ``point``. Ignored while a circuit is running.

        A point at the exact centre names no direction and is ignored too,
        rather than being read as the top-left corner.
        """
        if self._laps:
            return
        target = self.perimeter_position(point)
        if target is None:
            return
        self._towards = target
        self._start()

    def circuit(self, clockwise: bool = True) -> None:
        """Send the accent once round: clockwise, or anti- for Previous.

        THE DIRECTION IS THE MESSAGE. It tells the user which way they went
        through the slides, which is worth more than the animation -- so a
        circuit is never merged with another or shortened to catch up.
        """
        self._laps += 1.0 if clockwise else -1.0
        self._start()

    @property
    def position(self) -> float:
        """The accent's place on the rim, 0..1 clockwise from the top-left."""
        return self._at % 1.0

    @property
    def spinning(self) -> bool:
        """Whether a circuit is running."""
        return bool(self._laps)

    def _aim_at_the_cursor(self) -> bool:
        """Point the accent at wherever the cursor is. True if it moved.

        READ, NOT RECEIVED. A widget is sent mouse events only while the
        pointer is over it, and a modal dialog gets none at all once the
        pointer leaves the window -- so an accent that tracks "towards the
        mouse" cannot be driven by events and still follow a mouse that is
        somewhere else. The cursor position is global and always readable,
        which is the whole of the fix for "doesn't track the mouse on the
        outside".
        """
        try:
            from PySide6.QtGui import QCursor

            here = self.mapFromGlobal(QCursor.pos())
        except Exception:                        # pragma: no cover
            return False
        target = self.perimeter_position(QPointF(here))
        if target is None:
            return False
        moved = abs(((target - self._towards + 0.5) % 1.0) - 0.5) > 1e-9
        self._towards = target
        return moved

    @staticmethod
    def _preferred_arc() -> int:
        """The stored rim length, or the shipped default."""
        try:
            from ..preferences import get_rim_length

            return int(get_rim_length())
        except Exception:                                    # noqa: BLE001
            return 280

    def ease(self) -> float:
        """How far the accent closes the gap to the pointer each frame."""
        if self._lag is not None:
            return float(self._lag)
        try:
            from ..preferences import get_rim_lag

            return float(get_rim_lag())
        except Exception:                                    # noqa: BLE001
            return float(self.EASE)

    def mode(self) -> str:
        """``'glow'``, ``'rainbow'`` or ``'beat'``."""
        if self._mode:
            return self._mode
        try:
            from ..preferences import get_rim_mode

            return str(get_rim_mode())
        except Exception:                                    # noqa: BLE001
            return "glow"

    def period(self) -> float:
        """Seconds for one pulse, or one turn of the hue."""
        try:
            from ..preferences import get_rim_period

            return float(get_rim_period())
        except Exception:                                    # noqa: BLE001
            return 2.4

    def animates(self) -> bool:
        """Whether the rim changes when nothing else does.

        A pulse and a moving spectrum have to repaint on a still card; a
        plain glow must NOT, or an arrived rim costs sixty composites a
        second over a live backdrop for no visible change.
        """
        return self.mode() in ("rainbow", "beat")

    def alignment(self) -> str:
        """``'centre'`` or ``'head'`` -- where the run sits on the pointer."""
        if self._align:
            return self._align
        try:
            from ..preferences import get_rim_alignment

            return str(get_rim_alignment())
        except Exception:                                    # noqa: BLE001
            return "centre"

    def reread_the_preferences(self) -> None:
        """Take the length, lag and alignment again, and redraw.

        Called when the settings that own them change, so the card the user
        is looking at while they drag a slider is the one that answers.
        """
        self._arc = self._preferred_arc()
        self.update()

    def _start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def showEvent(self, event):                 # noqa: N802 - Qt naming
        """Start following as soon as there is something to follow."""
        super().showEvent(event)
        self._start()

    def hideEvent(self, event):                 # noqa: N802 - Qt naming
        """A card nobody is looking at does not need sixty frames a second."""
        super().hideEvent(event)
        self._timer.stop()

    def _tick(self) -> None:
        """One frame: run the laps down, else ease towards the pointer."""
        # The animation clock advances whatever else happens, so a pulse
        # keeps its rhythm through a circuit and across a slide change.
        self._phase += self._timer.interval() / 1000.0
        if self._laps:
            step = 0.03 if self._laps > 0 else -0.03
            self._at += step
            self._laps -= step
            # A LAP ENDS EXACTLY, not approximately: floating error across
            # thirty-odd frames would otherwise leave the accent a little
            # further round after every circuit, and after ten slides it
            # would be somewhere the pointer never put it.
            if abs(self._laps) < 0.031:
                # THE REMAINDER IS TRAVELLED, not undone: `_laps` is what is
                # still owed, so the last partial step ADDS it. Subtracting
                # left the accent 2% short of home on every circuit, which
                # after ten slides is a fifth of the way round from where
                # the pointer last put it.
                self._at = round(self._at + self._laps, 6)
                self._laps = 0.0
        else:
            # WHERE THE POINTER IS NOW, read fresh every frame. See
            # `_aim_at_the_cursor`: events cannot carry a pointer that is
            # outside the window, and this accent is meant to follow one.
            self._aim_at_the_cursor()
            gap = ((self._towards - self._at + 0.5) % 1.0) - 0.5
            if abs(gap) < 0.002:
                if self._at == self._towards and not self.animates():
                    # ARRIVED AND NOTHING MOVED. The timer keeps running --
                    # it is what notices the cursor moving again -- but a
                    # repaint of a card that has not changed is sixty
                    # needless composites a second over a live backdrop.
                    # A pulsing or spectral rim DOES change, so it paints.
                    return
                self._at = self._towards
            else:
                self._at += gap * self.ease()  # ease, not jump: water
        self._corner = int((self.position + 0.125) % 1.0 * 4) % 4
        self.update()

    def _paint(self) -> None:
        from ..theme import active_palette

        palette = active_palette()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)

            body = QColor(palette.get("surface", palette["bg"]))
            body.setAlpha(216)          # translucent: the blobs show through
            painter.setPen(Qt.NoPen)
            painter.setBrush(body)
            # THE BODY COVERS THE WHOLE CARD, not the inset the rim is
            # stroked on. Inset by a pixel it left a gap to whatever sits
            # behind -- a hairline along the straight edges, but half again
            # as wide across the diagonal of each corner, where the drifting
            # backdrop showed through as a blue crescent on every rounded
            # part. The rim still strokes the inset rect, so its width has
            # somewhere to sit.
            painter.drawRoundedRect(QRectF(self.rect()),
                                    self._radius, self._radius)

            # THE RESTING RIM IS DARK GREY, not a faint white. It was the
            # foreground ink at alpha 38, which on a dark theme is a pale
            # line round the card and competes with the accent travelling
            # along it -- the lit part should be the only bright part.
            # `border` is the palette's own dark grey, and on a light
            # theme it is the grey that reads against white.
            edge = QColor(palette.get("border", palette["fg"]))
            edge.setAlpha(235)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(edge, 1.0))
            painter.drawRoundedRect(rect, self._radius, self._radius)

            # THE ACCENT, a run of rim centred on `position`, FADING AT
            # BOTH ENDS.
            #
            # DRAWN AS SEGMENTS OF THE ROUNDED PATH ITSELF rather than as
            # the four hand-built corner paths below. A continuous position
            # cannot be expressed as one of four corners, and
            # `QPainterPathStroker` would give the outline of the stroke
            # rather than a segment of it -- so the segments come from
            # `QPainterPath.pointAtPercent` along the whole rim, which is
            # the one thing Qt measures in arc length for us.
            #
            # SEGMENT BY SEGMENT, because a QPen carries ONE colour: a run
            # that fades has to be many short strokes, each with its own
            # alpha and its own width. Twenty-four of them is below the
            # threshold at which the joins are visible and well inside the
            # frame budget at 60 fps.
            self._paint_accent(painter, QColor(palette["accent"]), rect)
        finally:
            painter.end()

    def accent_span(self, rect: QRectF) -> float:
        """How much of the rim is lit, as a fraction of its length.

        THE SAME FRACTION ON EVERY CARD, so a settings popup and the
        first-run window wear one rim rather than two takes on it. The arc
        preference is a length in pixels and is read as the length it
        should look on the REFERENCE surface; measured against each card's
        own perimeter instead, the same 280 px covers a sixth of the setup
        window and two fifths of a small popup, and the short one reads as
        a thick bright band -- "the rim is to thick and bright. make the
        rim and window look exactly like the setup spacr window."
        """
        rim = QPainterPath()
        rim.addRoundedRect(QRectF(0.0, 0.0, *REFERENCE_CARD),
                           self._radius, self._radius)
        total = max(1.0, rim.length())
        return min(0.62, max(0.04, float(self._arc) * 2.0 / total))

    def accent_peak(self) -> float:
        """Where along the run the accent is brightest, 0..1.

        THE BRIGHT PART IS WHAT THE POINTER IS ON. With the run CENTRED on
        the pointer the peak belongs in the middle, or the brightest part
        sits to one side of the thing it is pointing at; with the run
        trailing from its head, it belongs near the front, where a wake is
        brightest.
        """
        return 0.5 if self.alignment() == "centre" else 0.72

    def accent_alpha(self, along: float) -> float:
        """Opacity at ``along`` (0 at one end, 1 at the peak), 0..1.

        Both ends fall to zero, so neither has an edge to arrive or leave
        by. Where the peak sits is :meth:`accent_peak`.
        """
        along = min(max(float(along), 0.0), 1.0)
        peak = self.accent_peak()
        if along <= peak:
            ramp = along / peak
        else:
            ramp = (1.0 - along) / max(1e-6, 1.0 - peak)
        # Squared, so the fall is gentle near the peak and quick at the ends
        # -- the shape a wake has.
        return max(0.0, min(1.0, ramp ** (1.0 + self.FADE)))

    def accent_start(self, span: float) -> float:
        """Where the lit run begins, as a fraction of the rim.

        With ``centre`` alignment the MIDDLE of the run lands on
        :attr:`position`, which is what puts the light on the pointer
        rather than beside it; with ``head`` the run ends there and trails
        backwards.
        """
        if self.alignment() == "centre":
            return self.position - span / 2.0
        return self.position - span

    def _accent_path(self, rect: QRectF) -> QPainterPath:
        """The lit run of rim, as one path.

        Kept because the SHAPE of the run is worth asserting on its own,
        separately from how it is shaded. THE WHOLE RIM IS BUILT ONCE and
        sampled along its length, so the run crosses a corner without the
        seam four separate corner paths have -- which is what makes it read
        as flowing rather than as switching.
        """
        rim = QPainterPath()
        rim.addRoundedRect(rect, self._radius, self._radius)
        span = self.accent_span(rect)
        start = self.accent_start(span)

        out = QPainterPath()
        steps = 48
        for index in range(steps + 1):
            at = (start + span * index / steps) % 1.0
            point = rim.pointAtPercent(at)
            if index == 0:
                out.moveTo(point)
            else:
                out.lineTo(point)
        return out

    def ink_at(self, along: float, accent: QColor) -> QColor:
        """The colour of the run at ``along`` (0 at the tail, 1 at the head).

        `glow` and `beat` are the theme's accent the whole way; `rainbow`
        walks the hue along the run and turns it over time, so the light
        carries a spectrum that moves rather than a band that sits still.
        """
        if self.mode() != "rainbow":
            return QColor(accent)
        turn = self._phase / max(0.1, self.period())
        hue = (float(along) + turn) % 1.0
        spectral = QColor()
        # SATURATION AND VALUE FROM THE ACCENT, so a rainbow on a pale
        # theme is not the same searing colour as one on a dark theme.
        spectral.setHsvF(hue, min(1.0, accent.saturationF() + 0.25),
                         max(accent.valueF(), 0.85))
        return spectral

    def beat(self) -> float:
        """A multiplier on the run's brightness, 0..1.

        One for every mode but `beat`, which breathes: the run brightens
        and dims on a steady cycle. It never reaches zero -- a rim that
        vanishes reads as a fault rather than as a pulse.
        """
        if self.mode() != "beat":
            return 1.0
        import math

        cycle = math.sin(2.0 * math.pi * self._phase / max(0.1, self.period()))
        return 0.45 + 0.55 * (0.5 + 0.5 * cycle)

    def _paint_accent(self, painter, colour: QColor, rect: QRectF) -> None:
        """Stroke the lit run as segments that fade towards both ends.

        WHERE THE RUN SITS ON THE POINTER is :meth:`accent_start`, and it
        is a setting: centred puts the middle of the light on the pointer,
        head puts its leading end there and trails the rest behind.
        """
        rim = QPainterPath()
        rim.addRoundedRect(rect, self._radius, self._radius)
        span = self.accent_span(rect)
        start = self.accent_start(span)
        # ONE STEP PER `STEP_PX` OF RIM, not a fixed count. At 24 segments a
        # run this long was 23 px a step: the alpha moved in visible jumps
        # and every corner was cut into four straight chords, which is the
        # "chunky" of the 2026-08-22 report. The count now follows the
        # length being drawn, so it stays smooth on a card of any size and
        # costs nothing on a small one.
        run_px = max(1.0, span * rim.length())
        steps = int(min(self.MAX_STEPS,
                        max(24.0, run_px / self.STEP_PX)))
        previous = rim.pointAtPercent(start % 1.0)
        previous_alpha = self.accent_alpha(0.0)
        previous_along = 0.0
        for index in range(1, steps + 1):
            along = index / steps
            point = rim.pointAtPercent((start + span * along) % 1.0)
            alpha = self.accent_alpha(along)
            if alpha > 0.004 or previous_alpha > 0.004:
                # THE MIDPOINT ALPHA, so a segment is the shade of the rim
                # it covers rather than of the end it stops at -- which is
                # what leaves a visible edge between one segment and the
                # next at the faint end.
                middle = (alpha + previous_alpha) / 2.0
                ink = self.ink_at((along + previous_along) / 2.0, colour)
                ink.setAlpha(int(round(235 * middle * self.beat())))
                # THE WIDTH TAPERS WITH THE ALPHA. A constant-width stroke
                # fading to nothing still shows its full thickness where it
                # is faint, which reads as a smear; a taper reads as a wake.
                #
                # ROUND CAPS AND JOINS: a round cap on a 5 px segment
                # overlaps its neighbour by half a width, so the joins fill
                # instead of leaving the pale notch a flat cap leaves.
                pen = QPen(ink, 1.2 + 2.2 * middle, Qt.SolidLine,
                           Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(previous, point)
            previous = point
            previous_alpha = alpha
            previous_along = along

    def _corner_path(self, rect: QRectF) -> QPainterPath:
        """The two edge runs meeting at the current corner, with its arc."""
        radius, arc = float(self._radius), float(self._arc)
        path = QPainterPath()
        name = CORNERS[self._corner]
        if name == "topLeft":
            box = QRectF(rect.left(), rect.top(), radius * 2, radius * 2)
            path.moveTo(rect.left(), rect.top() + radius + arc)
            path.lineTo(rect.left(), rect.top() + radius)
            path.arcTo(box, 180, -90)
            path.lineTo(rect.left() + radius + arc, rect.top())
        elif name == "topRight":
            box = QRectF(rect.right() - radius * 2, rect.top(),
                         radius * 2, radius * 2)
            path.moveTo(rect.right() - radius - arc, rect.top())
            path.lineTo(rect.right() - radius, rect.top())
            path.arcTo(box, 90, -90)
            path.lineTo(rect.right(), rect.top() + radius + arc)
        elif name == "bottomRight":
            box = QRectF(rect.right() - radius * 2, rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.right(), rect.bottom() - radius - arc)
            path.lineTo(rect.right(), rect.bottom() - radius)
            path.arcTo(box, 0, -90)
            path.lineTo(rect.right() - radius - arc, rect.bottom())
        else:
            box = QRectF(rect.left(), rect.bottom() - radius * 2,
                         radius * 2, radius * 2)
            path.moveTo(rect.left() + radius + arc, rect.bottom())
            path.lineTo(rect.left() + radius, rect.bottom())
            path.arcTo(box, 270, -90)
            path.lineTo(rect.left(), rect.bottom() - radius - arc)
        return path
