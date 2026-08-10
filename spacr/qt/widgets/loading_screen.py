"""The screen that covers the preload.

spaCR imports ~3.1 s of heavy modules (torch, cellpose, pandas and the
pipeline that depends on them) before the first click on a module can be
instant. Measured on a real windowed launch: ``spacr.core`` alone is 1968 ms
and ``spacr.deep_spacr`` 711 ms.

Those imports have to run on the MAIN thread -- doing them on a worker races
Qt's own GPU initialisation and segfaults, which is recorded above
:class:`spacr.qt.app._PipelinePreloader`. So the event loop *will* be blocked
for seconds. The only question is whether the user is looking at a window
that appears interactive while it happens.

This is the answer: cover the whole window until the work is done. A freeze
behind a screen that says LOADING is not a freeze, and the same three seconds
stop being a defect.

THE PROGRESS IS THE PRODUCT'S OWN SENTENCE. Rather than a bar, the three
phases of :data:`STRAP_LINE` light up in turn as the modules land --
microscopy, then single-cell analysis, then genotype-to-phenotype. It reads
as the pipeline describing itself, and it is honest: the denominator is the
number of modules, which is known before the first one is imported.
"""
from __future__ import annotations

from typing import Optional, Sequence

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPixmap, QFont
from PySide6.QtWidgets import QWidget

from ..preferences import scaled_px
from ..iconset import RESOURCE_DIR

#: The one place this sentence lives. The home screen shows it beside the
#: logo too, so a change here changes both -- and it needs a translation row
#: like every other UI string (see tests/qt/test_i18n.py).
STRAP_PHASES = (
    "End-to-end microscopy",
    "single-cell image analysis",
    "genotype-to-phenotype mapping",
)

#: Rendered form, for callers that want the whole sentence.
STRAP_LINE = "  →  ".join(STRAP_PHASES)

#: Taken from the installer icon (spacr/resources/icons/app_icon.ico), whose
#: dominant colour is exactly this. Sampled rather than guessed so the
#: loading screen and the thing the user double-clicked are the same colour.
INSTALLER_GREEN = "#003737"

#: The mark, in white, as it appears on the installer icon.
LOGO_FILE = "logo_spacr.png"


class LoadingScreen(QWidget):
    """Full-window cover shown until the pipeline modules are imported.

    :param total: how many steps will be reported. Zero or negative means
        "unknown", and the phases simply stay dim rather than dividing by it.
    :param parent: the window this fills. It is sized to the parent and
        resizes with it, rather than being a separate top-level -- a second
        window would earn its own taskbar entry and could be dragged off the
        thing it is supposed to be covering.
    """

    def __init__(self, total: int = 0, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LoadingScreen")
        self._total = max(0, int(total))
        self._done = 0
        self._logo: Optional[QPixmap] = None
        # Opaque: this covers a partly-built window, and any transparency
        # would show the thing it exists to hide.
        self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        if parent is not None:
            self.setGeometry(parent.rect())
        self._load_logo()

    # -- state -------------------------------------------------------------
    def set_total(self, total: int) -> None:
        """Set the denominator; repaints if it changed."""
        total = max(0, int(total))
        if total != self._total:
            self._total = total
            self.update()

    def advance(self, done: Optional[int] = None) -> None:
        """Report progress. Without an argument, counts one more step."""
        self._done = self._done + 1 if done is None else max(0, int(done))
        self.update()

    def fraction(self) -> float:
        """Completed share, 0.0 to 1.0. Zero when the total is unknown."""
        if self._total <= 0:
            return 0.0
        return min(1.0, self._done / float(self._total))

    def lit_phases(self) -> int:
        """How many of the three phases are lit at the current fraction.

        The last phase lights only at completion, so a user never sees the
        sentence finished while work is still running.
        """
        f = self.fraction()
        if f >= 1.0:
            return len(STRAP_PHASES)
        return min(len(STRAP_PHASES) - 1, int(f * len(STRAP_PHASES)))

    # -- painting ----------------------------------------------------------
    def _load_logo(self) -> None:
        try:
            import os
            path = os.path.join(RESOURCE_DIR, LOGO_FILE)
            pix = QPixmap(path)
            if not pix.isNull():
                self._logo = pix
        except Exception:
            # A missing logo must not stop the app from starting. The screen
            # still covers the window and still reports progress.
            self._logo = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QColor(INSTALLER_GREEN))

            side = scaled_px(140)
            gap = scaled_px(28)
            font = QFont(self.font())
            font.setPixelSize(max(11, scaled_px(15)))
            painter.setFont(font)
            metrics = painter.fontMetrics()

            # Translated at PAINT time, not at import: the loading screen is
            # built before the user's language preference has necessarily
            # been read, and a phase cached in English would stay English.
            phases = [_translate(p) for p in STRAP_PHASES]
            widths = [metrics.horizontalAdvance(p) for p in phases]
            arrow_w = metrics.horizontalAdvance("  →  ")
            text_w = sum(widths) + arrow_w * (len(phases) - 1)

            block_w = side + gap + text_w
            x = (self.width() - block_w) / 2.0
            y = self.height() / 2.0

            if self._logo is not None:
                scaled = self._logo.scaled(
                    side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(
                    int(x), int(y - scaled.height() / 2.0), scaled)

            # The sentence, one phase at a time.
            lit = self.lit_phases()
            tx = x + side + gap
            baseline = y + metrics.ascent() / 2.0 - metrics.descent() / 2.0
            for index, phase in enumerate(phases):
                painter.setPen(QColor(255, 255, 255,
                                      255 if index < lit else 90))
                painter.drawText(int(tx), int(baseline), phase)
                tx += widths[index]
                if index < len(phases) - 1:
                    painter.setPen(QColor(255, 255, 255,
                                          255 if index + 1 < lit else 70))
                    painter.drawText(int(tx), int(baseline), "  →  ")
                    tx += arrow_w

            # A hairline under the sentence, filled to the same fraction.
            # Thin on purpose: the sentence is the progress indicator, and a
            # second loud one would compete with it.
            rule_y = baseline + metrics.descent() + scaled_px(12)
            rule_w = text_w
            rule_x = x + side + gap
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 45))
            painter.drawRect(QRectF(rule_x, rule_y, rule_w, 2.0))
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.drawRect(
                QRectF(rule_x, rule_y, rule_w * self.fraction(), 2.0))
        finally:
            painter.end()


def _translate(text: str) -> str:
    """Translate one phase, falling back to English.

    Imported lazily and defensively: this widget is on the launch path, and a
    loading screen that cannot render because the catalog failed to import
    would take the whole application with it.
    """
    try:
        from ..i18n import tr
        return tr(text)
    except Exception:
        return text


def strap_phrases() -> Sequence[str]:
    """The strap line's phases, translated, for the home screen to reuse.

    The home screen shows the same sentence beside the same logo, so it takes
    the words from here rather than repeating them -- one string, one place,
    one set of translation rows.
    """
    return tuple(_translate(p) for p in STRAP_PHASES)


def strap_line() -> str:
    """The whole strap line as one translated string."""
    return "  →  ".join(strap_phrases())
