#!/usr/bin/env python3
"""Shared flat-vector drawing helpers for spaCR candidate icons.

House style (derived from spacr/resources/icons/plaque.png and measure.png):
  * pure white artwork on a fully transparent background (alpha carries the shape)
  * flat: no gradients, no shading, no colour
  * a mix of thin outlined strokes and solid white fills
  * square canvas, subject fills most of the frame with a modest margin
  * measured stroke widths, normalised to a 1024 canvas: ~9 px (plaque dish),
    ~20 px (measure), ~23 px (recruitment).  Defaults here: 22 primary,
    14 secondary, 10 fine.

Everything is authored in a normalised 0..1 coordinate space and scaled to the
canvas, so the same path renders correctly at any size.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QTransform,
)

# Ensure a QGuiApplication exists for font handling on the offscreen platform.
try:  # pragma: no cover - environment plumbing
    from PySide6.QtGui import QGuiApplication

    if QGuiApplication.instance() is None:
        _APP = QGuiApplication([])
except Exception:  # pragma: no cover
    _APP = None

N = 1024
WHITE = QColor(255, 255, 255)

W_MAIN = 22.0   # primary stroke, in 1024-canvas units
W_SEC = 14.0    # secondary stroke
W_FINE = 10.0   # fine detail


class Cv:
    """A square RGBA canvas with normalised (0..1) drawing primitives."""

    def __init__(self, n: int = N):
        self.n = n
        self.img = QImage(n, n, QImage.Format_ARGB32_Premultiplied)
        self.img.fill(QColor(0, 0, 0, 0))
        self.p = QPainter(self.img)
        self.p.setRenderHint(QPainter.Antialiasing, True)
        self.p.setRenderHint(QPainter.SmoothPixmapTransform, True)

    # ---------------------------------------------------------------- core
    def finish(self) -> QImage:
        if self.p.isActive():
            self.p.end()
        return self.img

    def save(self, path: str) -> None:
        self.finish().save(path)

    def pen(self, w: float, cap=Qt.RoundCap) -> QPen:
        pen = QPen(WHITE)
        pen.setWidthF(max(0.4, w * self.n / 1024.0))
        pen.setCapStyle(cap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def stroke(self, path: QPainterPath, w: float = W_MAIN, dash=None) -> None:
        pen = self.pen(w)
        if dash:
            pen.setCapStyle(Qt.FlatCap)
            pen.setDashPattern(list(dash))
        self.p.setPen(pen)
        self.p.setBrush(Qt.NoBrush)
        self.p.drawPath(path)

    def fill(self, path: QPainterPath) -> None:
        self.p.setPen(Qt.NoPen)
        self.p.setBrush(QBrush(WHITE))
        self.p.drawPath(path)

    def pt(self, x: float, y: float) -> QPointF:
        return QPointF(x * self.n, y * self.n)

    # ------------------------------------------------------------ shapes
    def line(self, x1, y1, x2, y2, w=W_MAIN, dash=None):
        pa = QPainterPath()
        pa.moveTo(self.pt(x1, y1))
        pa.lineTo(self.pt(x2, y2))
        self.stroke(pa, w, dash)

    def polyline(self, pts, w=W_MAIN, close=False, filled=False, dash=None):
        pa = QPainterPath()
        pa.moveTo(self.pt(*pts[0]))
        for q in pts[1:]:
            pa.lineTo(self.pt(*q))
        if close:
            pa.closeSubpath()
        if filled:
            self.fill(pa)
        else:
            self.stroke(pa, w, dash)

    def smooth(self, pts, w=W_MAIN, dash=None, closed=False, filled=False):
        """Catmull-Rom through pts, emitted as cubic beziers."""
        p = [tuple(q) for q in pts]
        if closed:
            ext = [p[-1]] + p + [p[0], p[1]]
        else:
            ext = [p[0]] + p + [p[-1]]
        pa = QPainterPath()
        pa.moveTo(self.pt(*p[0]))
        for i in range(1, len(ext) - 2):
            p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
            c1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
            pa.cubicTo(self.pt(*c1), self.pt(*c2), self.pt(*p2))
        if closed:
            pa.closeSubpath()
        if filled:
            self.fill(pa)
        else:
            self.stroke(pa, w, dash)

    def circ(self, cx, cy, r, w=W_MAIN, dash=None):
        pa = QPainterPath()
        pa.addEllipse(self.pt(cx, cy), r * self.n, r * self.n)
        self.stroke(pa, w, dash)

    def disc(self, cx, cy, r):
        pa = QPainterPath()
        pa.addEllipse(self.pt(cx, cy), r * self.n, r * self.n)
        self.fill(pa)

    def ring(self, cx, cy, r, thick):
        """Solid annulus (even-odd)."""
        pa = QPainterPath()
        pa.setFillRule(Qt.OddEvenFill)
        pa.addEllipse(self.pt(cx, cy), (r + thick / 2) * self.n, (r + thick / 2) * self.n)
        pa.addEllipse(self.pt(cx, cy), (r - thick / 2) * self.n, (r - thick / 2) * self.n)
        self.fill(pa)

    def ell(self, cx, cy, rx, ry, rot=0.0, w=W_MAIN, filled=False, dash=None):
        pa = QPainterPath()
        pa.addEllipse(QPointF(0.0, 0.0), rx * self.n, ry * self.n)
        t = QTransform()
        t.translate(cx * self.n, cy * self.n)
        t.rotate(rot)
        pa = t.map(pa)
        if filled:
            self.fill(pa)
        else:
            self.stroke(pa, w, dash)

    def arc(self, cx, cy, r, a0, span, w=W_MAIN, dash=None):
        rr = QRectF((cx - r) * self.n, (cy - r) * self.n, 2 * r * self.n, 2 * r * self.n)
        pa = QPainterPath()
        pa.arcMoveTo(rr, a0)
        pa.arcTo(rr, a0, span)
        self.stroke(pa, w, dash)

    def rect(self, x, y, w_, h, w=W_MAIN, filled=False, r=0.0):
        pa = QPainterPath()
        if r > 0:
            pa.addRoundedRect(QRectF(x * self.n, y * self.n, w_ * self.n, h * self.n),
                              r * self.n, r * self.n)
        else:
            pa.addRect(QRectF(x * self.n, y * self.n, w_ * self.n, h * self.n))
        if filled:
            self.fill(pa)
        else:
            self.stroke(pa, w, dash=None)

    def bar(self, x, y, w_, h, r=None, filled=True, w=W_SEC):
        """Rounded capsule bar."""
        rr = r if r is not None else min(w_, h) / 2.0
        self.rect(x, y, w_, h, w=w, filled=filled, r=rr)

    # ------------------------------------------------------- compound art
    def lens(self, cx, cy, rx, ry, rot=0.0, bulge=1.9):
        """The plaque 'lens' blob: two arcs meeting at pointed tips."""
        q = self.n
        pa = QPainterPath()
        pa.moveTo(-rx * q, 0.0)
        pa.quadTo(0.0, -ry * bulge * q, rx * q, 0.0)
        pa.quadTo(0.0, ry * bulge * q, -rx * q, 0.0)
        pa.closeSubpath()
        t = QTransform()
        t.translate(cx * q, cy * q)
        t.rotate(rot)
        self.fill(t.map(pa))

    def arrow(self, x1, y1, x2, y2, w=W_MAIN, head=0.055, ratio=0.52, tail=True):
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        if tail:
            self.line(x1, y1, bx, by, w)
        hw = head * ratio
        px, py = -math.sin(ang), math.cos(ang)
        self.polyline(
            [(x2, y2), (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)],
            close=True, filled=True,
        )

    def cell(self, cx, cy, r, ry=None, rot=0.0, w=W_SEC, nuc=0.40, nuc_off=(0.0, 0.0)):
        """Stylised cell: thin outline plus a solid nucleus."""
        self.ell(cx, cy, r, ry if ry is not None else r * 0.88, rot, w)
        if nuc > 0:
            self.disc(cx + nuc_off[0] * r, cy + nuc_off[1] * r, r * nuc)

    def parasite(self, cx, cy, L, rot=0.0, fat=0.30):
        """Crescent tachyzoite, pointed at both tips."""
        q = self.n
        h = L * fat
        pa = QPainterPath()
        pa.moveTo(-L / 2 * q, 0.0)
        pa.cubicTo(-L * 0.30 * q, -h * 1.85 * q, L * 0.30 * q, -h * 1.85 * q, L / 2 * q, 0.0)
        pa.cubicTo(L * 0.26 * q, -h * 0.72 * q, -L * 0.26 * q, -h * 0.72 * q, -L / 2 * q, 0.0)
        pa.closeSubpath()
        t = QTransform()
        t.translate(cx * q, cy * q)
        t.rotate(rot)
        self.fill(t.map(pa))

    def tick_ring(self, cx, cy, r, n_, inner, outer, w=W_SEC, a0=0.0):
        for i in range(n_):
            a = a0 + 2 * math.pi * i / n_
            self.line(cx + inner * math.cos(a), cy + inner * math.sin(a),
                      cx + outer * math.cos(a), cy + outer * math.sin(a), w)

    def dot_ring(self, cx, cy, r, n_, rd, a0=0.0, filled=True, w=W_FINE):
        for i in range(n_):
            a = a0 + 2 * math.pi * i / n_
            x, y = cx + r * math.cos(a), cy + r * math.sin(a)
            if filled:
                self.disc(x, y, rd)
            else:
                self.circ(x, y, rd, w)

    def axes(self, x0, y0, x1, y1, w=W_SEC, ticks=0, tick=0.022):
        """L-shaped axis frame with origin at (x0, y1)."""
        self.polyline([(x0, y0), (x0, y1), (x1, y1)], w=w)
        for i in range(1, ticks + 1):
            fx = x0 + (x1 - x0) * i / (ticks + 1)
            fy = y1 - (y1 - y0) * i / (ticks + 1)
            self.line(fx, y1, fx, y1 + tick, w * 0.8)
            self.line(x0, fy, x0 - tick, fy, w * 0.8)

    def magnifier(self, cx, cy, r, ang_deg=45.0, w=W_MAIN, handle=0.9):
        self.circ(cx, cy, r, w)
        a = math.radians(ang_deg)
        self.line(cx + r * math.cos(a), cy + r * math.sin(a),
                  cx + r * (1 + handle) * math.cos(a), cy + r * (1 + handle) * math.sin(a),
                  w * 1.35)

    def clip_circle(self, cx, cy, r):
        pa = QPainterPath()
        pa.addEllipse(self.pt(cx, cy), r * self.n, r * self.n)
        self.p.save()
        self.p.setClipPath(pa)

    def clip_rect(self, x, y, w_, h):
        pa = QPainterPath()
        pa.addRect(QRectF(x * self.n, y * self.n, w_ * self.n, h * self.n))
        self.p.save()
        self.p.setClipPath(pa)

    def unclip(self):
        self.p.restore()


# ---------------------------------------------------------------- output
def render(fn, path, n=N):
    c = Cv(n)
    fn(c)
    c.save(path)
    return path


def contact_sheet(images, out_path, bg, cols=5, cell=340, pad=26, label_h=42, note=None,
                  thumb=True):
    """Grid of numbered variants on a solid background.

    On a light background the white artwork would be invisible, so the alpha
    mask is re-tinted to dark ink; the shape being judged is identical.
    """
    rows = (len(images) + cols - 1) // cols
    head_h = 56 if note else 0
    w = cols * cell + (cols + 1) * pad
    h = head_h + rows * (cell + label_h) + (rows + 1) * pad
    sheet = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
    sheet.fill(QColor(bg))
    p = QPainter(sheet)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    dark = QColor(bg).lightnessF() < 0.5
    ink = QColor(255, 255, 255) if dark else QColor(20, 22, 26)
    faint = QColor(255, 255, 255, 46) if dark else QColor(20, 22, 26, 46)
    font = QFont()
    font.setPixelSize(int(label_h * 0.62))
    font.setBold(True)
    p.setFont(font)
    if note:
        nf = QFont()
        nf.setPixelSize(26)
        p.setFont(nf)
        p.setPen(QPen(ink))
        p.drawText(QRectF(pad, 8, w - 2 * pad, head_h), Qt.AlignLeft | Qt.AlignVCenter, note)
        p.setFont(font)
    for i, img in enumerate(images):
        r, cix = divmod(i, cols)
        x = pad + cix * (cell + pad)
        y = head_h + pad + r * (cell + label_h + pad)
        p.setPen(QPen(faint, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(x - 6, y - 6, cell + 12, cell + 12), 12, 12)
        scaled = img.scaled(cell, cell, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if not dark:
            # white artwork is invisible on light: recolour the mask to ink
            tinted = QImage(scaled.size(), QImage.Format_ARGB32_Premultiplied)
            tinted.fill(Qt.transparent)
            tp = QPainter(tinted)
            tp.drawImage(0, 0, scaled)
            tp.setCompositionMode(QPainter.CompositionMode_SourceIn)
            tp.fillRect(tinted.rect(), ink)
            tp.end()
            scaled = tinted
        p.drawImage(int(x), int(y), scaled)
        if thumb:
            t48 = img.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if not dark:
                tt = QImage(t48.size(), QImage.Format_ARGB32_Premultiplied)
                tt.fill(Qt.transparent)
                tp = QPainter(tt)
                tp.drawImage(0, 0, t48)
                tp.setCompositionMode(QPainter.CompositionMode_SourceIn)
                tp.fillRect(tt.rect(), ink)
                tp.end()
                t48 = tt
            p.setPen(QPen(faint, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRectF(x + cell - 54, y + cell - 54, 52, 52))
            p.drawImage(int(x + cell - 53), int(y + cell - 53), t48)
        p.setPen(QPen(ink))
        p.drawText(QRectF(x, y + cell + 2, cell, label_h), Qt.AlignHCenter | Qt.AlignVCenter,
                   "%02d" % (i + 1))
    p.end()
    sheet.save(out_path)
    return out_path
