"""Lay several masks and images over one another, in a panel of their own.

Once a preview session has made more than one mask, the masks and the
field can be ticked, given an opacity and put in a stacking order, and the
composite is drawn in a panel beside the preview's other views.

TWO HALVES. :func:`composite` is the arithmetic, on plain arrays, so the
picture a user sees can be checked pixel by pixel against a hand
calculation. :class:`MaskComparisonDialog` is the popup that chooses what
goes into it. Neither knows about the Mask module; the live preview hands
in its session's masks and its field, and draws the answer.

THE ARITHMETIC IS "OVER", LAYER BY LAYER, FROM THE BOTTOM. The canvas
starts black. Each ticked layer, bottom first, is laid on with its opacity
``a``: where it covers, ``out = out * (1 - a) + colour * a``; where it does
not, ``out`` is left alone. An image covers every pixel; a mask covers its
objects only, so the background of a mask never dims what is under it.
The result is rounded to the nearest whole grey level.

THE STACKING ORDER READS LIKE A LAYER LIST. The popup lists the top of the
stack first, the way every drawing program does, and :meth:`chosen` hands
the layers back bottom first, the order :func:`composite` lays them on.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QGridLayout, QLabel, QSlider, QToolButton,
                               QVBoxLayout, QWidget)

from ..i18n import tr

MASK = "mask"
IMAGE = "image"


@dataclass
class Layer:
    """One thing the comparison can draw.

    :param name: what the popup calls it.
    :param kind: :data:`MASK` or :data:`IMAGE`.
    :param array: an ``H x W`` label mask, or an ``H x W`` or ``H x W x 3``
        ``uint8`` picture.
    :param colour: the colour a mask's objects are filled with.
    :param ticked: whether it is drawn.
    :param opacity: 0 to 1.
    """

    name: str
    kind: str
    array: np.ndarray
    colour: Tuple[int, int, int] = (255, 255, 255)
    ticked: bool = True
    opacity: float = 0.5


def layer_rgb(layer: Layer) -> Tuple[np.ndarray, np.ndarray]:
    """A layer's colour and the pixels it covers.

    :returns: ``(H x W x 3 float32 colour, H x W bool coverage)``.
    """
    array = np.asarray(layer.array)
    if layer.kind == MASK:
        cover = array > 0
        rgb = np.empty(array.shape[:2] + (3,), np.float32)
        rgb[...] = np.asarray(layer.colour, np.float32)
        return rgb, cover
    if array.ndim == 2:
        rgb = np.repeat(array[..., None], 3, axis=-1)
    else:
        rgb = array[..., :3]
    return rgb.astype(np.float32), np.ones(array.shape[:2], bool)


def composite(layers: Sequence[Layer],
              shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
    """Lay the ticked ``layers`` over a black canvas, bottom first.

    :param layers: bottom of the stack first. Unticked layers and layers
        of another size than the first are skipped.
    :param shape: the canvas size; the first ticked layer's when omitted.
    :returns: ``H x W x 3`` ``uint8``, or ``None`` when nothing is ticked.
    """
    drawn = [layer for layer in layers if layer.ticked]
    if not drawn:
        return None
    if shape is None:
        shape = tuple(np.asarray(drawn[0].array).shape[:2])
    out = np.zeros(tuple(shape) + (3,), np.float32)
    for layer in drawn:
        if tuple(np.asarray(layer.array).shape[:2]) != tuple(shape):
            continue
        alpha = float(min(1.0, max(0.0, layer.opacity)))
        rgb, cover = layer_rgb(layer)
        weight = (cover.astype(np.float32) * alpha)[..., None]
        out = out * (1.0 - weight) + rgb * weight
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


class MaskComparisonDialog(QDialog):
    """Tick, opacity and stacking order for each mask and image.

    Rows are listed top of the stack first. The dialog is a plain
    ``QDialog``, so spaCR's card and rim reach it the way they reach every
    other popup.

    :param layers: the choices, TOP OF THE STACK FIRST.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, layers: Sequence[Layer], parent=None) -> None:
        """Build one row per layer, and OK and Cancel."""
        super().__init__(parent)
        self.setObjectName("MaskComparisonDialog")
        self.setWindowTitle(tr("Compare masks"))
        self._layers: List[Layer] = [replace(layer) for layer in layers]
        column = QVBoxLayout(self)
        intro = QLabel(tr(
            "Tick the masks and images to lay over one another. The top "
            "row is drawn on top; the arrows move a row up or down the "
            "stack."), self)
        intro.setWordWrap(True)
        column.addWidget(intro)
        self._rows_host = QWidget(self)
        self._grid = QGridLayout(self._rows_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._rows_host)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        column.addWidget(buttons)
        self._rows: List[Tuple[QCheckBox, QSlider]] = []
        self._build_rows()

    def _build_rows(self) -> None:
        """Lay one row per layer, in the current order, from scratch."""
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._rows = []
        for header, text in enumerate((tr("Show"), tr("Opacity"),
                                       tr("Order"))):
            self._grid.addWidget(QLabel(text, self._rows_host), 0,
                                 (0, 1, 3)[header])
        for index, layer in enumerate(self._layers):
            row = index + 1
            tick = QCheckBox(layer.name, self._rows_host)
            tick.setChecked(layer.ticked)
            if layer.kind == MASK:
                tick.setStyleSheet("color: rgb(%d, %d, %d);" % layer.colour)
            slider = QSlider(Qt.Horizontal, self._rows_host)
            slider.setRange(0, 100)
            slider.setValue(int(round(layer.opacity * 100)))
            slider.setMinimumWidth(120)
            value = QLabel(f"{slider.value()} %", self._rows_host)
            slider.valueChanged.connect(
                lambda v, label=value: label.setText(f"{v} %"))
            up = QToolButton(self._rows_host)
            up.setText("▲")
            up.setToolTip(tr("Move up the stack"))
            up.setEnabled(index > 0)
            up.clicked.connect(lambda _=False, i=index: self.move(i, -1))
            down = QToolButton(self._rows_host)
            down.setText("▼")
            down.setToolTip(tr("Move down the stack"))
            down.setEnabled(index < len(self._layers) - 1)
            down.clicked.connect(lambda _=False, i=index: self.move(i, 1))
            self._grid.addWidget(tick, row, 0)
            self._grid.addWidget(slider, row, 1)
            self._grid.addWidget(value, row, 2)
            self._grid.addWidget(up, row, 3)
            self._grid.addWidget(down, row, 4)
            self._rows.append((tick, slider))

    def _read_rows(self) -> None:
        """Copy the ticks and sliders back into the layers."""
        for layer, (tick, slider) in zip(self._layers, self._rows):
            layer.ticked = tick.isChecked()
            layer.opacity = slider.value() / 100.0

    def rows(self) -> List[Tuple[QCheckBox, QSlider]]:
        """Each row's tick and opacity slider, top of the stack first."""
        return list(self._rows)

    def move(self, index: int, step: int) -> None:
        """Move row ``index`` by ``step`` places; ``-1`` is up the stack."""
        target = index + step
        if not (0 <= index < len(self._layers)
                and 0 <= target < len(self._layers)):
            return
        self._read_rows()
        layers = self._layers
        layers[index], layers[target] = layers[target], layers[index]
        self._build_rows()

    def listed(self) -> List[Layer]:
        """Every layer as the popup has it now, top of the stack first."""
        self._read_rows()
        return [replace(layer) for layer in self._layers]

    def chosen(self) -> List[Layer]:
        """The ticked layers, BOTTOM FIRST, ready for :func:`composite`."""
        return [layer for layer in reversed(self.listed()) if layer.ticked]
