"""Optional Qt graphics items used by the live FlowView panel.

The module is safe to import when PySide6 is unavailable.  Constructing a
live item in that environment raises an actionable error while the headless
model, tracing, and static exporters remain usable.
"""

from __future__ import annotations

import math

from .layout import NodeLayout
from .model import Edge, Node
from .theme import (
    CARD,
    CORNER_RADIUS,
    FONT_FAMILY,
    LABEL_SIZE,
    METRIC_SIZE,
    STATE_SIZE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THUMBNAIL_SIZE,
    node_accent,
    state_label,
)

QT_INSTALL_COMMAND = "pip install spacr[flowview]"
QT_MISSING_MESSAGE = (
    "The FlowView live panel requires PySide6. "
    f"Install it with `{QT_INSTALL_COMMAND}`."
)

try:
    from PySide6.QtCore import QRectF, Qt
    from PySide6.QtGui import (
        QBrush,
        QColor,
        QFont,
        QImage,
        QPainter,
        QPainterPath,
        QPen,
    )
    from PySide6.QtWidgets import QGraphicsItem
except ImportError as error:
    QT_AVAILABLE = False
    _QT_IMPORT_ERROR = error
else:
    QT_AVAILABLE = True
    _QT_IMPORT_ERROR = None


if not QT_AVAILABLE:

    class _MissingQtItem:
        """Placeholder that preserves an actionable optional-dependency error."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ImportError(QT_MISSING_MESSAGE) from _QT_IMPORT_ERROR

    class NodeItem(_MissingQtItem):
        """Placeholder used when PySide6 is unavailable.

        Construction raises :class:`ImportError` with the
        ``pip install spacr[flowview]`` remediation.
        """

    class EdgeItem(_MissingQtItem):
        """Placeholder used when PySide6 is unavailable.

        Construction raises :class:`ImportError` with the
        ``pip install spacr[flowview]`` remediation.
        """

else:

    def _colour(value: str, alpha: int = 255) -> QColor:
        colour = QColor(value)
        colour.setAlpha(alpha)
        return colour


    def edge_width(volume: int | None) -> float:
        """Map transfer volume to the same restrained logarithmic scale as SVG."""

        if volume is None or volume <= 0:
            return 1.0
        return min(6.0, 1.0 + 0.8 * math.log10(volume + 1.0))


    def _load_thumbnail(path: str | None) -> QImage:
        image = QImage()
        if path is not None:
            image.load(path)
        return image


    class NodeItem(QGraphicsItem):
        """With PySide6 available, ``NodeItem`` paints a selectable node card.

        :param node: run-graph node whose label, state, metrics, progress,
            and optional thumbnail the card displays.
        :param box: layout coordinates and dimensions used to position and
            bound the card.
        """

        def __init__(self, node: Node, box: NodeLayout) -> None:
            super().__init__()
            self.node = node
            self.box = box
            self._thumbnail = _load_thumbnail(node.thumbnail)
            self.setPos(box.x, box.y)
            self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
            self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
            self.setToolTip(f"{node.label} — {state_label(node.state)}")

        @property
        def node_id(self) -> str:
            """Stable model identifier used by the inspector."""

            return self.node.id

        def boundingRect(self) -> QRectF:  # noqa: N802 - Qt virtual name
            """Return the card-local paint bounds."""

            return QRectF(0.0, 0.0, self.box.width, self.box.height)

        def update_node(self, node: Node, box: NodeLayout) -> bool:
            """Replace displayed data, returning whether any paint data changed."""

            if node == self.node and box == self.box:
                return False
            self.prepareGeometryChange()
            self.node = node
            self.box = box
            self._thumbnail = _load_thumbnail(node.thumbnail)
            self.setPos(box.x, box.y)
            self.setToolTip(f"{node.label} — {state_label(node.state)}")
            self.update()
            return True

        def paint(self, painter: QPainter, option: object, widget: object = None) -> None:
            """Paint one restrained card using the shared FlowView visual tokens."""

            del option, widget
            rect = self.boundingRect()
            accent = _colour(node_accent(self.node.kind, self.node.state))
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setBrush(QBrush(_colour(CARD)))
            painter.setPen(QPen(_colour("#FFFFFF", 26), 1.0))
            painter.drawRoundedRect(rect, CORNER_RADIUS, CORNER_RADIUS)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(accent))
            painter.drawRoundedRect(
                QRectF(0.0, 0.0, 4.0, rect.height()),
                2.0,
                2.0,
            )

            state = state_label(self.node.state)
            state_font = QFont(FONT_FAMILY.split(",", maxsplit=1)[0])
            state_font.setPixelSize(STATE_SIZE)
            painter.setFont(state_font)
            state_width = max(45.0, painter.fontMetrics().horizontalAdvance(state) + 14.0)
            state_rect = QRectF(rect.width() - state_width - 12.0, 12.0, state_width, 20.0)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(accent, 1.0))
            painter.drawRoundedRect(state_rect, CORNER_RADIUS, CORNER_RADIUS)
            painter.setPen(QPen(_colour(TEXT_PRIMARY)))
            painter.drawText(state_rect, Qt.AlignmentFlag.AlignCenter, state)

            label_font = QFont(FONT_FAMILY.split(",", maxsplit=1)[0])
            label_font.setPixelSize(LABEL_SIZE)
            label_font.setWeight(QFont.Weight.DemiBold)
            painter.setFont(label_font)
            painter.setPen(QPen(_colour(TEXT_PRIMARY)))
            painter.drawText(
                QRectF(16.0, 10.0, max(1.0, rect.width() - state_width - 44.0), 40.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                self.node.label,
            )

            cursor_y = 58.0
            if not self._thumbnail.isNull():
                size = min(THUMBNAIL_SIZE, rect.width() - 32.0)
                painter.drawImage(QRectF(16.0, cursor_y, size, size), self._thumbnail)
                cursor_y += size + 16.0

            metric_font = QFont(FONT_FAMILY.split(",", maxsplit=1)[0])
            metric_font.setPixelSize(METRIC_SIZE)
            painter.setFont(metric_font)
            painter.setPen(QPen(_colour(TEXT_SECONDARY)))
            for name, value in sorted(self.node.metrics.items())[:3]:
                painter.drawText(
                    QRectF(16.0, cursor_y - 12.0, rect.width() - 32.0, 16.0),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    f"{name}: {value}",
                )
                cursor_y += 16.0

            if self.node.progress is not None and self.node.progress[1] > 0:
                current, total = self.node.progress
                fraction = max(0.0, min(1.0, current / total))
                painter.setPen(QPen(accent, 2.0))
                painter.drawLine(
                    4.0,
                    rect.height() - 2.0,
                    4.0 + (rect.width() - 8.0) * fraction,
                    rect.height() - 2.0,
                )
            painter.restore()


    class EdgeItem(QGraphicsItem):
        """With PySide6 available, ``EdgeItem`` paints a non-interactive edge.

        :param edge: transfer whose volume controls logarithmic stroke width
            and whose optional label is painted on the curve.
        :param source: layout geometry of the source node.
        :param target: layout geometry of the target node.
        :param source_running: whether to initially paint the dashed marker
            for an active source stage.
        """

        def __init__(
            self,
            edge: Edge,
            source: NodeLayout,
            target: NodeLayout,
            *,
            source_running: bool = False,
        ) -> None:
            super().__init__()
            self.edge = edge
            self._source_running = bool(source_running)
            self._path = self._make_path(source, target)
            self.setZValue(-1.0)
            self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        @staticmethod
        def _make_path(source: NodeLayout, target: NodeLayout) -> QPainterPath:
            start_x = source.x + source.width
            start_y = source.centre_y
            end_x = target.x
            end_y = target.centre_y
            bend = max(32.0, abs(end_x - start_x) * 0.45)
            path = QPainterPath()
            path.moveTo(start_x, start_y)
            path.cubicTo(
                start_x + bend,
                start_y,
                end_x - bend,
                end_y,
                end_x,
                end_y,
            )
            return path

        @property
        def source_running(self) -> bool:
            """Whether the source stage is currently active."""

            return self._source_running

        def boundingRect(self) -> QRectF:  # noqa: N802 - Qt virtual name
            """Include the complete antialiased stroke in scene updates."""

            padding = edge_width(self.edge.volume) + 2.0
            return self._path.boundingRect().adjusted(-padding, -padding, padding, padding)

        def set_source_running(self, running: bool) -> bool:
            """Update the non-colour running marker only when it changed."""

            normalised = bool(running)
            if normalised == self._source_running:
                return False
            self._source_running = normalised
            self.update()
            return True

        def paint(self, painter: QPainter, option: object, widget: object = None) -> None:
            """Paint a cubic edge, label, and optional running dash pattern."""

            del option, widget
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            pen = QPen(_colour(TEXT_SECONDARY, 184), edge_width(self.edge.volume))
            pen.setCosmetic(True)
            if self._source_running:
                pen.setDashPattern([7.0, 6.0])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(self._path)
            if self.edge.label:
                label_font = QFont(FONT_FAMILY.split(",", maxsplit=1)[0])
                label_font.setPixelSize(METRIC_SIZE)
                painter.setFont(label_font)
                painter.setPen(QPen(_colour(TEXT_SECONDARY)))
                middle = self._path.pointAtPercent(0.5)
                painter.drawText(
                    QRectF(middle.x() - 70.0, middle.y() - 22.0, 140.0, 18.0),
                    Qt.AlignmentFlag.AlignCenter,
                    self.edge.label,
                )
            painter.restore()


__all__ = [
    "EdgeItem",
    "NodeItem",
    "QT_AVAILABLE",
    "QT_INSTALL_COMMAND",
    "QT_MISSING_MESSAGE",
]

if QT_AVAILABLE:
    __all__.append("edge_width")
