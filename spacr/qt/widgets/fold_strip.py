"""A row of icon buttons, one per module folded into a host screen.

A module that has been folded into another one stops being a tile on Home
and becomes a button on its host's masthead. The button IS the module's
icon: no text beside it, the module's own one-line description as its
tooltip, and the same maturity colour on hover that the tile used to
light up in -- green-cyan for alpha, magenta for beta, blue for stable.

That last part is the whole point of doing it here rather than with a
plain ``QPushButton`` at each site. The hover colour is not decoration,
it is the maturity of the code behind the button, and it is read from
:data:`spacr.qt.theme.STAGE_HOVER` through :func:`spacr.qt.app.app_stage`
-- the same two tables the tiles read. A fold that hard-coded its colour
would drift the day a module was signed off, and the button would go on
promising alpha long after the tile had stopped.

Typical use, from a host screen's ``__init__``::

    from spacr.qt.widgets.fold_strip import FoldStrip

    self.folds = FoldStrip([
        ("train_cellpose", self._open_training),
        ("model_zoo",      self._open_model_zoo),
    ], parent=self)
    masthead_layout.addWidget(self.folds)

Each entry is ``(app_key, callback)``. The key is the registry key the
module had as a tile, which is what supplies the icon, the name and the
stage; the callback is what the button does when pressed.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .. import iconset

#: Edge of the square icon button, in logical pixels before font scaling.
BUTTON_PX = 34

#: Edge of the icon inside it. Smaller than the button so the hover fill
#: reads as a plate behind the mark rather than as a border touching it.
ICON_PX = 20

#: Space between two folded buttons.
GAP_PX = 6

#: The objectName every fold button carries, so one QSS rule in
#: :mod:`spacr.qt.theme` can style all of them at once.
BUTTON_NAME = "FoldButton"


def _describe(key: str) -> Tuple[str, str, str]:
    """Return ``(name, description, stage)`` for a registry key.

    Imported lazily and defensively: this widget is constructed while a
    screen is being built, and :mod:`spacr.qt.app` imports screens. A
    module-level import would close that circle.
    """
    try:
        from .. import app as app_module
    except Exception:                                   # pragma: no cover
        return key.replace("_", " ").title(), "", "stable"
    name, description = key.replace("_", " ").title(), ""
    for row in getattr(app_module, "APPS", ()):
        if row and row[0] == key:
            name = row[1] or name
            description = row[2] or ""
            break
    stage_of = getattr(app_module, "app_stage", None)
    stage = stage_of(key) if callable(stage_of) else "stable"
    return name, description, stage


class FoldButton(QPushButton):
    """One folded module, drawn as its own icon and nothing else."""

    def __init__(self, key: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.app_key = key
        name, description, stage = _describe(key)
        self.setObjectName(BUTTON_NAME)
        # The stage rides as a Qt property so the stylesheet can select on
        # it -- QPushButton#FoldButton[stage="alpha"]:hover -- exactly as
        # the tiles do. Setting it before the first polish means the first
        # paint already has the right colour.
        self.setProperty("stage", stage)
        self.setFlat(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(QSize(BUTTON_PX, BUTTON_PX))
        self.setIconSize(QSize(ICON_PX, ICON_PX))
        icon = None
        try:
            icon = iconset.app_icon(key)
        except Exception:                               # pragma: no cover
            icon = None
        if icon is not None and not icon.isNull():
            self.setIcon(icon)
        else:
            # No icon shipped for this key: fall back to the initial
            # rather than to an empty square the user cannot identify.
            self.setText(name[:1].upper())
        # The name leads the tooltip because the button has no label; the
        # description follows it as the sentence the tile carried.
        self.setToolTip(f"{name}\n{description}".strip())
        self.setAccessibleName(name)


class FoldStrip(QWidget):
    """The row of :class:`FoldButton` for one host screen."""

    def __init__(
        self,
        folds: Iterable[Tuple[str, Callable[[], None]]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.buttons: list[FoldButton] = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(GAP_PX)
        for key, callback in folds:
            button = FoldButton(key, self)
            if callable(callback):
                button.clicked.connect(lambda _checked=False, cb=callback: cb())
            row.addWidget(button)
            self.buttons.append(button)
        row.addStretch(1)

    def keys(self) -> Sequence[str]:
        """The registry keys this strip carries, in the order shown."""
        return [b.app_key for b in self.buttons]

    def button_for(self, key: str) -> Optional[FoldButton]:
        """The button for ``key``, or None if this strip has no such fold."""
        for button in self.buttons:
            if button.app_key == key:
                return button
        return None
