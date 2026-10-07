class AuroraEngine(_BufferedEngine):
    """Folded curtains of vertical rays, rippling along their own length.

    See the block comment above for the phenomenon, for why the colour ramp
    is anchored to the frame rather than to the curtain, and for why it is
    painted as layered brush fills per curtain rather than as hundreds of
    sprites. The frame raster is native at ordinary Detail, within the
    physical screen-pixel budget, and is returned with independent ownership.
    :data:`AURORA_BUFFER_EDGE` remains the legacy comparison edge; it no
    longer caps the active buffer.

    :meth:`geometry` yields ``(x, y_bottom, visible_height, brightness)`` per
    sampled column of every curtain, in pixels, ``AURORA_COLUMNS + 1`` of them
    per curtain in curtain order. The painter builds its paths from exactly
    those numbers, so a test that tracks a fold crest through ``geometry`` is
    tracking the crest that is on screen. ``brightness`` is the surge, and it
    is the *model's* value: what gets painted is the same function with its
    phase quantised into :data:`AURORA_PULSE_STEPS` so the surge texture can
    be cached (see :meth:`_surge`).
    """

    name = "aurora"
    base_edge = AURORA_BUFFER_EDGE

    def __init__(self, *args, **kwargs):
        """Roll the aurora's bands and their drift."""
        self._tiles: Dict[Tuple[int, int], QImage] = {}
        self._surges: Dict[int, QImage] = {}
        self._pulse_mask: Optional[QImage] = None
        super().__init__(*args, **kwargs)

    def buffer_size(self, width: int, height: int) -> Tuple[int, int]:
        """Sample native display pixels within the actual screen budget."""
        detail = min(1.0, self.resolution)
        bw, bh = max(1, int(width * detail)), max(1, int(height * detail))
        scale = min(1.0, math.sqrt(self.max_pixels / (bw * bh)))
        return max(1, int(bw * scale)), max(1, int(bh * scale))

    def buffer_scale(self, width: int, height: int) -> float:
        """Report the aurora sampling ratio for explicit detail controls."""
        bw, bh = self.buffer_size(width, height)
        return max(1.0, width / bw, height / bh)

    def _shade(self, width: int, height: int) -> QImage:
        """Clear the owned raster before painting the current curtains."""
        bw, bh = self.buffer_size(width, height)
        buf = QImage(bw, bh, QImage.Format_RGB32)
        buf.fill(self.identity)
        inner = QPainter(buf)
        try:
            inner.setCompositionMode(self.mode)
            inner.setPen(Qt.NoPen)
            self._paint_field(inner, buf.width(), buf.height())
        finally:
            inner.end()
        return self._soften(buf, width, height)

    def _configure(self, rng: random.Random) -> None:
        """Roll this theme's constants from the seed.

        ONCE, at construction: an engine is deterministic, so the same seed and
        the same call sequence always produce the same animation.
        """
        self.curtains: List[Curtain] = []
        for i in range(_pool_size(AURORA_CURTAINS)):
            base = (AURORA_BASE[i % len(AURORA_BASE)]
                    + (i // len(AURORA_BASE)) * AURORA_TIER_OFFSET)
            self.curtains.append(Curtain(
                y=base + rng.uniform(-AURORA_BASE_JITTER, AURORA_BASE_JITTER),
                height=rng.uniform(*AURORA_THICKNESS),
                tilt=rng.uniform(-AURORA_TILT, AURORA_TILT),
                drift=rng.uniform(*AURORA_DRIFT),
                rate=2 * math.pi / rng.uniform(*AURORA_DRIFT_PERIOD),
                phase=rng.uniform(0.0, 2 * math.pi),
                hue_rate=2 * math.pi / rng.uniform(*AURORA_HUE_PERIOD),
                hue_phase=rng.uniform(0.0, 2 * math.pi),
                fold_phase=tuple(rng.uniform(0.0, 2 * math.pi)
                                 for _ in AURORA_FOLDS),
                pulse_phase=rng.uniform(0.0, 2 * math.pi),
                depth=i,
                color=i,
            ))

    def _restyle(self) -> None:
        """Re-roll the colours after a palette change."""
        super()._restyle()
        self._tiles = {}
        self._surges = {}

    def _resize(self) -> None:
        """Re-lay the bands for a new widget size."""
        self._tiles = {}
        self._surges = {}

    def count(self) -> int:
        """How many curtains are painted right now."""
        return self.element_count(AURORA_CURTAINS, len(self.curtains))

    def _rate(self, curtain: Curtain) -> float:
        """How fast the bands drift, given the current speed setting.

        :returns: the drift rate.
        """
        return AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][0]

    def fold(self, curtain: Curtain, u: float, t: float) -> float:
        """Fold displacement at position ``u`` along the arc, as a fraction
        of the canvas height.

        ``u`` runs 0..1 from one end of the arc to the other. Each component
        is ``sin(2*pi*(u - v*t)/lambda)``: at a fixed time it is a shape in
        ``u``, and as ``t`` advances that shape *slides along u* at ``v``
        while the arc itself goes nowhere. That is the whole difference
        between an aurora and a curtain being dragged sideways, and it is the
        one property of this engine worth testing directly.

        Scaled by the size setting along with everything else: a curtain half
        the height with folds the same depth is a different phenomenon, not a
        smaller one.
        """
        rate = self._rate(curtain)
        total = 0.0
        for (amp, wavelength, speed), phase in zip(AURORA_FOLDS,
                                                   curtain.fold_phase):
            total += amp * math.sin(
                2 * math.pi * (u - speed * rate * t) / wavelength + phase)
        return total * self.size

    def pulse(self, curtain: Curtain, u: float, t: float) -> float:
        """The surge's brightness at ``u``, in 0..1. Another travelling wave,
        deliberately faster and shorter than every fold."""
        depth, wavelength, speed = AURORA_PULSE
        travelling = 0.5 + 0.5 * math.sin(
            2 * math.pi * (u - speed * self._rate(curtain) * t) / wavelength
            + curtain.pulse_phase)
        return 1.0 - depth + depth * travelling

    def anchor(self, curtain: Curtain, height: int) -> Tuple[float, float]:
        """``(ramp zero, ray length)`` for one curtain, in pixels.

        The ramp's zero is the altitude the emission stops at, so it sits
        below everything the fold and the tilt can do — that is what keeps
        every column of the sheet inside its own colour ramp.
        """
        base = curtain.y + curtain.drift * math.sin(
            curtain.rate * self.time + curtain.phase)
        reach = (AURORA_FOLD_REACH + abs(curtain.tilt) * 0.5) * self.size
        return ((base + reach) * height,
                max(1.0, curtain.height * self.size * height))

    def geometry(self, width: int, height: int) -> Tuple[tuple, ...]:
        """Every travelling wave, evaluated along every arc.

        The loop body is :meth:`fold` and :meth:`pulse` written out with
        their constant parts hoisted — ``sin(2*pi*(u - v*t)/lambda + phi)``
        is ``sin(k*u + (phi - k*v*t))``, and ``k`` and the bracket do not
        depend on the column. It is the same arithmetic; it is here rather
        than behind those two calls because this runs a hundred and twenty
        times a frame and a Python call is not free. ``test_aurora_geometry_
        is_the_model_it_documents`` holds the two forms together.
        """
        t = self.time
        out = []
        sin = math.sin
        two_pi = 2 * math.pi
        span = width * AURORA_OVERHANG
        left = (width - span) * 0.5
        columns = AURORA_COLUMNS
        p_depth, p_wavelength, p_speed = AURORA_PULSE
        for curtain in self.curtains[:self.count()]:
            rate = self._rate(curtain)
            depth_alpha = AURORA_DEPTHS[
                curtain.depth % len(AURORA_DEPTHS)][2]
            zero, ray = self.anchor(curtain, height)
            base = curtain.y + curtain.drift * math.sin(
                curtain.rate * t + curtain.phase)
            top = zero - ray
            tilt = curtain.tilt * self.size
            folds = [(amp * self.size, two_pi / wavelength,
                      phase - two_pi * speed * rate * t / wavelength)
                     for (amp, wavelength, speed), phase
                     in zip(AURORA_FOLDS, curtain.fold_phase)]
            p_k = two_pi / p_wavelength
            p_phase = (curtain.pulse_phase
                       - two_pi * p_speed * rate * t / p_wavelength)
            for i in range(columns + 1):
                u = i / columns
                displacement = base + tilt * (u - 0.5)
                for amp, k, phase in folds:
                    displacement += amp * sin(k * u + phase)
                y = displacement * height
                bright = depth_alpha * (
                    1.0 - p_depth + p_depth
                    * (0.5 + 0.5 * sin(p_k * u + p_phase)))
                out.append((left + u * span, y,
                            y - top if y > top else 0.0, bright))
        return tuple(out)

    def hue_phase(self, curtain: Curtain) -> float:
        """Where this curtain's slow colour shimmer stands, in 0..1."""
        return 0.5 + 0.5 * math.sin(
            curtain.hue_rate * self.time + curtain.hue_phase)

    def curtain_color(self, curtain: Curtain, quantised: bool = False
                      ) -> QColor:
        """The curtain's body colour right now.

        Always built from the palette's *first* colour, wandering up to
        :data:`AURORA_HUE_BLEND` of the way towards one of the others and
        back. Every curtain shares that body colour on purpose: the body of
        an aurora is a single emission line — 557.7 nm oxygen — and the
        palette's remaining entries are the top and the fringe, which the
        ramp puts above and below it. Giving curtain two a red body and
        curtain three a violet one, which is what indexing the palette by
        curtain would do, is the one thing that stops the whole theme reading
        as an aurora.

        :param curtain: simulated curtain whose colour index and shimmer phase
            choose the palette target and its current blend toward it.
        :param quantised: snap the shimmer to :data:`AURORA_HUE_STEPS` so the
            ray tile can be cached.
        """
        colors = self.paint_colors
        body = colors[0]
        wander = colors[1 + curtain.color % (len(colors) - 1)] \
            if len(colors) > 1 else body
        u = self.hue_phase(curtain)
        if quantised:
            u = round(u * (AURORA_HUE_STEPS - 1)) / (AURORA_HUE_STEPS - 1)
        return _mix(body, wander, AURORA_HUE_BLEND * u)

    def ramp_colors(self, curtain: Curtain, quantised: bool = False
                    ) -> Dict[str, QColor]:
        """The four palette roles for one curtain: the body, the high red,
        the low fringe, and the overlap between body and high.

        Fixed roles rather than a rotation, because the vertical order is
        physics. With ``borealis``, ``main`` is the 557.7 nm green, ``high``
        the 630.0 nm red, ``fringe`` the 427.8 nm violet and ``blend`` the
        pale yellow-green where the first two overlap. A palette with fewer
        than four colours reuses what it has.
        """
        colors = self.paint_colors
        n = len(colors)
        main = self.curtain_color(curtain, quantised=quantised)
        high = colors[1 % n]
        fringe = colors[2 % n]
        blend = colors[3] if n > 3 else _mix(main, high, 0.5)
        return {"main": main, "high": high, "fringe": fringe, "blend": blend}

    def ray_lengths(self, curtain: Curtain) -> Tuple[float, ...]:
        """Each ray's current length, as a fraction of the full one.

        One value per entry in :data:`AURORA_TILE_RAYS`, quantised into
        :data:`AURORA_LENGTH_STEPS` so the tile stays cacheable -- a length
        that followed the clock exactly would rebuild every tile every
        frame, which is the cost the tile cache exists to avoid.

        The periods in :data:`AURORA_RAY_LIFE` are deliberately not
        multiples of one another, and the curtain's own phase is added, so
        two curtains never breathe together either.
        """
        low, high = AURORA_RAY_LENGTH
        out = []
        for period, offset in AURORA_RAY_LIFE:
            angle = (2 * math.pi * (self.time / period + offset)
                     + curtain.pulse_phase)
            unit = 0.5 * (1.0 + math.sin(angle))
            stepped = round(unit * (AURORA_LENGTH_STEPS - 1)) \
                / (AURORA_LENGTH_STEPS - 1)
            out.append(low + (high - low) * stepped)
        return tuple(out)

    def _tile(self, curtain: Curtain, peak: float, width: int,
              height: int) -> QImage:
        """The ray comb crossed with the vertical colour ramp, as a tiling
        texture, built at the exact pixel size it will be painted at.

        Cached per (curtain, quantised shimmer, size). Nothing about it
        changes from frame to frame: the ray period and the ray length are
        fixed for a given canvas, and the curtain's slow colour shimmer is
        quantised into :data:`AURORA_HUE_STEPS`. Three dozen of these get
        built in the life of the widget, against three a frame if the tile
        followed the clock.
        """
        step = int(round(self.hue_phase(curtain) * (AURORA_HUE_STEPS - 1)))
        lengths = self.ray_lengths(curtain)
        key = (curtain.depth, step, width, height, lengths, peak)
        tile = self._tiles.get(key)
        if tile is not None:
            return tile
        if len(self._tiles) >= AURORA_TILE_CACHE:
            self._tiles = {}

        top_f, bottom_f = AURORA_TILE_RAMP
        ramp_top = int(round(top_f * height))
        ramp_bottom = max(ramp_top + 1, int(round(bottom_f * height)))
        tile = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        tile.fill(Qt.transparent)
        roles = self.ramp_colors(curtain, quantised=True)
        alpha = peak * AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][2]
        inner = QPainter(tile)
        inner.setPen(Qt.NoPen)
        gradient = QLinearGradient(0.0, float(ramp_bottom), 0.0,
                                   float(ramp_top))
        for stop, role, scale in AURORA_RAMP:
            gradient.setColorAt(stop, _with_alpha(roles[role], alpha * scale))
        inner.setBrush(gradient)
        inner.drawRect(0, ramp_top, width, ramp_bottom - ramp_top)
        inner.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        comb = QLinearGradient(0.0, 0.0, float(width), 0.0)
        floor = QColor(0, 0, 0, int(round(255 * AURORA_TILE_FLOOR)))
        comb.setColorAt(0.0, floor)
        for centre, half, strength in AURORA_TILE_RAYS:
            comb.setColorAt(max(0.0, centre - half), floor)
            comb.setColorAt(centre,
                            QColor(0, 0, 0, int(round(255 * strength))))
            comb.setColorAt(min(1.0, centre + half), floor)
        comb.setColorAt(1.0, floor)
        inner.setBrush(comb)
        inner.drawRect(0, 0, width, height)

        for (centre, half, _strength), length in zip(AURORA_TILE_RAYS,
                                                     lengths):
            if length >= 1.0:
                continue
            left = int(round(max(0.0, centre - half) * width))
            right = int(round(min(1.0, centre + half) * width))
            if right <= left:
                continue
            kept = int(round((ramp_bottom - ramp_top) * length))
            cut_bottom = ramp_bottom - kept
            feather = max(1, int(round(
                (ramp_bottom - ramp_top) * AURORA_RAY_FEATHER)))
            if cut_bottom <= 0:
                continue
            fade = QLinearGradient(0.0, float(max(0, cut_bottom - feather)),
                                   0.0, float(cut_bottom))
            fade.setColorAt(0.0, QColor(0, 0, 0, 0))
            fade.setColorAt(1.0, QColor(0, 0, 0, 255))
            inner.setBrush(fade)
            inner.drawRect(left, 0, right - left, cut_bottom)
        inner.end()
        self._tiles[key] = tile
        return tile

    def _mask(self) -> QImage:
        """The surge's vertical profile: solid along the lower edge, gone by
        :data:`AURORA_PULSE_HEIGHT` of the way up. Built once, then reused as
        the alpha of every per-frame surge image.

        Positioned in the *padded* band (see :data:`AURORA_PULSE_PAD`), which
        is why the stops are not at 0 and ``AURORA_PULSE_HEIGHT``: the
        curtain's lower edge sits a padding's worth up from the bottom of the
        image, and the ray length is a padded fraction of its height.
        """
        if self._pulse_mask is None:
            width, height = AURORA_PULSE_TEXTURE
            pad = AURORA_PULSE_PAD
            band = 1.0 + 2 * pad
            edge = pad / band
            reach = AURORA_PULSE_HEIGHT / band
            mask = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
            mask.fill(Qt.transparent)
            inner = QPainter(mask)
            inner.setPen(Qt.NoPen)
            fade = QLinearGradient(0.0, float(height), 0.0, 0.0)
            fade.setColorAt(0.0, QColor(0, 0, 0, 255))
            fade.setColorAt(edge, QColor(0, 0, 0, 255))
            fade.setColorAt(edge + reach * 0.45, QColor(0, 0, 0, 185))
            fade.setColorAt(min(1.0, edge + reach), QColor(0, 0, 0, 0))
            fade.setColorAt(1.0, QColor(0, 0, 0, 0))
            inner.setBrush(fade)
            inner.drawRect(0, 0, width, height)
            inner.end()
            self._pulse_mask = mask
        return self._pulse_mask

    def _surge(self, curtain: Curtain, peak: float) -> QImage:
        """The travelling surge for one curtain, as a small image.

        Horizontally it is the pulse; vertically it is the cached fade. It
        has to be a two-dimensional texture rather than a gradient brush: a
        horizontal gradient alone has no vertical falloff, so it would cut
        off in a hard line across the curtain, and putting the falloff in the
        path instead only moves the hard line somewhere else.

        Cached on the pulse's phase, quantised, plus the curtain's shimmer
        step — which is everything its content depends on, so the cache is a
        memo and not an approximation of the model. It still steps in time,
        and :data:`AURORA_PULSE_STEPS` is what decides how finely.
        """
        width, height = AURORA_PULSE_TEXTURE
        _depth, wavelength, speed = AURORA_PULSE
        phase = (curtain.pulse_phase
                 - 2 * math.pi * speed * self._rate(curtain) * self.time
                 / wavelength)
        step = int(round(phase % (2 * math.pi)
                         / (2 * math.pi) * AURORA_PULSE_STEPS))
        hue = int(round(self.hue_phase(curtain) * (AURORA_HUE_STEPS - 1)))
        key = (curtain.depth, step % AURORA_PULSE_STEPS, hue, peak)
        image = self._surges.get(key)
        if image is not None:
            return image
        if len(self._surges) >= AURORA_PULSE_CACHE:
            self._surges = {}

        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)
        inner = QPainter(image)
        inner.setPen(Qt.NoPen)
        color = self.curtain_color(curtain, quantised=True)
        gain = peak * AURORA_PULSE_GAIN
        depth_alpha = AURORA_DEPTHS[curtain.depth % len(AURORA_DEPTHS)][2]
        quantised = step % AURORA_PULSE_STEPS * 2 * math.pi \
            / AURORA_PULSE_STEPS
        gradient = QLinearGradient(0.0, 0.0, float(width), 0.0)
        stops = width
        for k in range(stops):
            u = k / (stops - 1)
            bright = depth_alpha * self._pulse_at(u, quantised)
            gradient.setColorAt(u, _with_alpha(color, gain * bright))
        inner.setBrush(gradient)
        inner.drawRect(0, 0, width, height)
        inner.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        inner.drawImage(0, 0, self._mask())
        inner.end()
        self._surges[key] = image
        return image

    @staticmethod
    def _pulse_at(u: float, phase: float) -> float:
        """The surge profile at ``u`` for a given travelling phase."""
        depth, wavelength, _speed = AURORA_PULSE
        return 1.0 - depth + depth * (
            0.5 + 0.5 * math.sin(2 * math.pi * u / wavelength + phase))

    def _paint_field(self, painter: QPainter, width: int, height: int) -> None:
        """Draw one frame's field of shapes.

        :param painter: the painter to draw with.
        :param width: the widget's width in pixels.
        :param height: its height in pixels.
        """
        peak = (AURORA_ALPHA_DARK if self.dark else AURORA_ALPHA_LIGHT) \
            * self._fractional_alpha_scale(AURORA_CURTAINS)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        samples = self.geometry(width, height)
        stride = AURORA_COLUMNS + 1
        top_f, bottom_f = AURORA_TILE_RAMP
        rays_per_tile = len(AURORA_TILE_RAYS)
        pulse_w, pulse_h = AURORA_PULSE_TEXTURE
        ray_px = max(AURORA_RAY_MIN_PX,
                     AURORA_RAY_SPACING * self.size * width)
        for index, curtain in enumerate(self.curtains[:self.count()]):
            columns = samples[index * stride:(index + 1) * stride]
            if len(columns) < 2:
                continue
            zero, ray = self.anchor(curtain, height)
            top = zero - ray
            sheet = self._sheet(columns, top)

            spacing = ray_px * AURORA_DEPTHS[
                curtain.depth % len(AURORA_DEPTHS)][1]
            tile = self._tile(
                curtain, peak,
                max(AURORA_TILE_MIN_PX,
                    int(round(spacing * rays_per_tile))),
                max(AURORA_TILE_MIN_PX,
                    int(round(ray / (bottom_f - top_f)))))
            brush = QBrush(tile)
            brush.setTransform(QTransform.fromTranslate(
                0.0, zero - bottom_f * tile.height()))
            painter.setBrush(brush)
            painter.drawPath(sheet)

            left, right = columns[0][0], columns[-1][0]
            band = ray * (1.0 + 2 * AURORA_PULSE_PAD)
            surge = QBrush(self._surge(curtain, peak))
            surge.setTransform(QTransform(
                (right - left) / pulse_w, 0.0, 0.0, band / pulse_h,
                left, top - ray * AURORA_PULSE_PAD))
            painter.setBrush(surge)
            painter.drawPath(self._sheet(
                columns, zero - ray * (AURORA_PULSE_HEIGHT
                                       + AURORA_PULSE_PAD)))
            roles = self.ramp_colors(curtain, quantised=True)
            for offset, weight, strength, role in (
                    (0.105, 1.8, 0.23, "main"),
                    (0.255, 1.1, 0.14, "blend"),
                    (0.43, 0.8, 0.13, "high")):
                contour = QPainterPath(QPointF(
                    columns[0][0], columns[0][1] - ray * offset))
                for x, y, _height, _bright in columns[1:]:
                    contour.lineTo(x, y - ray * offset)
                fade = QLinearGradient(left, 0.0, right, 0.0)
                glint = 0.78 + 0.22 * math.sin(
                    self.time * self._rate(curtain) * 0.29
                    + curtain.hue_phase + offset * 19.0)
                fade.setColorAt(0.0, _with_alpha(roles[role], 0.0))
                fade.setColorAt(0.18, _with_alpha(
                    roles[role], peak * strength * glint))
                fade.setColorAt(0.76, _with_alpha(
                    roles[role], peak * strength * 0.75 * glint))
                fade.setColorAt(1.0, _with_alpha(roles[role], 0.0))
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QBrush(fade), max(0.7, weight * self.size),
                                    Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawPath(contour)
                painter.setPen(Qt.NoPen)
        painter.setRenderHint(QPainter.Antialiasing, False)

    @staticmethod
    def _sheet(columns, top: float) -> QPainterPath:
        """The sheet as a closed path: along its folded lower edge, then
        straight back across a flat top.

        The top is flat, and that is not a shortcut. It sits exactly where
        the colour ramp has faded to nothing, so the polygon's upper boundary
        is invisible — which is the only way to get a *diffuse* top out of a
        hard-edged polygon. All the visible shape is in the lower edge, which
        is where a real curtain keeps it too.
        """
        path = QPainterPath()
        path.moveTo(columns[0][0], columns[0][1])
        for x, y, _h, _b in columns[1:]:
            path.lineTo(x, y)
        path.lineTo(columns[-1][0], top)
        path.lineTo(columns[0][0], top)
        path.closeSubpath()
        return path
