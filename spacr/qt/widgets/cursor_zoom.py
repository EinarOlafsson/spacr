"""Explicit cursor anchoring for image views, independent of Qt mouse history."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsView


def zoom_at_pointer(view, factor, position=None):
    """Scale an image while retaining the scene point beneath a viewport point.

    :param view: QGraphicsView whose owner suppresses peer updates while zooming.
    :param factor: positive, finite magnification multiplier.
    :param position: viewport coordinates; None uses its center for buttons.
    :returns: whether a zoom was applied. Extra scene margins allow anchoring
        even when the image is smaller than the viewport. The scene itself
        remains unchanged so fit-to-image still uses the actual image bounds.
    """
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0 or view.scene() is None:
        return False
    position = QPointF(position if position is not None else view.viewport().rect().center())
    inverse, valid = view.viewportTransform().inverted()
    if not valid:
        return False
    target = inverse.map(position)
    anchor = view.transformationAnchor()
    view.setTransformationAnchor(QGraphicsView.NoAnchor)
    try:
        view.scale(factor, factor)
        scale = abs(view.transform().m11())
        if scale == 0:
            return False
        margin_x = view.viewport().width() / scale
        margin_y = view.viewport().height() / scale
        bounds = view.scene().sceneRect().united(
            view.scene().sceneRect().translated(target - view.scene().sceneRect().center()))
        view.setSceneRect(bounds.adjusted(-margin_x, -margin_y, margin_x, margin_y))
        inverse, _ = view.viewportTransform().inverted()
        center = inverse.map(QPointF(view.viewport().rect().center()))
        view.centerOn(target + center - inverse.map(position))
    finally:
        view.setTransformationAnchor(anchor)
    return True
