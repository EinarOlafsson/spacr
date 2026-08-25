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

Each entry is ``(app_key, callback)``, or ``(app_key, callback, checkable)``
when the button is a switch rather than a press. The key is the registry
key the module had as a tile, which is what supplies the icon, the name
and the stage; the callback is what the button does when pressed.

A CHECKABLE FOLD IS A MODULE THAT LIVES ON THE HOST. Where a folded
module is nothing but a few settings categories its host does not show --
Timelapse and Motility on Mask Generation are the case -- the button does
not open anything: it reveals those categories and turns the pipeline
gate they belong to on. That is a state, not an action, so the button
holds it: it stays lit while the module is part of the run, and the
callback is handed the new state rather than called bare.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

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

#: One entry in a strip.
#:
#: ``(key, callback)`` for a button that opens something; the callback
#: takes no arguments, because "pressed" carries no state worth passing
#: on. ``(key, callback, True)`` for a switch, whose callback is handed
#: the new state so that one function answers both directions.
FoldEntry = Union[Tuple[str, Callable[[], None]],
                  Tuple[str, Callable[[bool], None], bool]]

#: How strongly a checked fold button is filled with its stage colour.
#:
#: Between the hover fill (0.22) and the pressed one (0.40) in
#: :mod:`spacr.qt.theme`, so a switched-on module reads as more than
#: hovered and less than held down -- and hovering one that is already on
#: still changes it, which is what tells a user the button is live.
CHECKED_ALPHA = 0.30


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

    def __init__(self, key: str, parent: Optional[QWidget] = None,
                 checkable: bool = False) -> None:
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
        if checkable:
            self.setCheckable(True)
            self._install_checked_fill(stage)
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

    def _install_checked_fill(self, stage: str) -> None:
        """Give a switch-shaped fold button a lit "on" state.

        The application stylesheet dresses ``#FoldButton`` for hover and
        for pressed, both of which are momentary; a checkable one also
        has to say so while nobody is touching it, or the only way to
        learn that Timelapse is part of the run is to press it and watch
        what appears.

        Written as a widget-local rule for the ``:checked`` state alone,
        so it MERGES with the application's hover and pressed rules
        rather than replacing them -- and the colour comes out of
        :data:`spacr.qt.theme.STAGE_HOVER`, the table the tiles and the
        hover rule read, so there is still exactly one place a stage
        colour is written down.
        """
        from ..theme import STAGE_HOVER, css_color

        hue = STAGE_HOVER.get(stage)
        if hue is None:
            # A maturity the table has never heard of: leave the button
            # with the shipped hover and pressed rules rather than
            # inventing a colour that no tile lights up in.
            return
        self.setStyleSheet(
            f"QPushButton#{BUTTON_NAME}:checked {{\n"
            f"    background-color: {css_color(hue, CHECKED_ALPHA)};\n"
            f"    border: 1px solid {hue};\n"
            f"}}"
        )


class FoldStrip(QWidget):
    """The row of :class:`FoldButton` for one host screen."""

    def __init__(
        self,
        folds: Iterable[FoldEntry],
        parent: Optional[QWidget] = None,
    ) -> None:
        """Build one button per fold.

        :param folds: ``(key, callback)`` for a button that opens
            something, or ``(key, callback, True)`` for one that switches
            a module on and off. A switch is handed the new state, so the
            one callback answers both directions; a plain button is
            called with no arguments, because "pressed" carries no state
            worth passing on.
        :param parent: the masthead the strip is hung on.
        """
        super().__init__(parent)
        self.buttons: list[FoldButton] = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(GAP_PX)
        for entry in folds:
            key, callback = entry[0], entry[1]
            checkable = bool(entry[2]) if len(entry) > 2 else False
            button = FoldButton(key, self, checkable=checkable)
            if callable(callback):
                if checkable:
                    button.toggled.connect(
                        lambda on, cb=callback: cb(on))
                else:
                    button.clicked.connect(
                        lambda _checked=False, cb=callback: cb())
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
