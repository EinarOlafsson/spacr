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
BUTTON_PX = 51

#: Edge of the icon inside it. Smaller than the button so the hover fill
#: reads as a plate behind the mark rather than as a border touching it.
ICON_PX = 30

#: Space between two folded buttons.
#:
#: IT GREW WITH THEM. The complaint that sent the buttons up by half was
#: that they read as CROWDED, and buttons that grow while the gap stays
#: put are more crowded, not less -- the same gap between larger marks is
#: a smaller share of the strip.
GAP_PX = 9

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


#: The host screens that fold other modules into themselves.
#:
#: Membership comes from each host's ``FOLDED_APPS`` (or Make Masks'
#: ``FOLD_ORDER``), while the name, description, and maturity come from the
#: available ``FOLD_FALLBACK`` tables. Keeping those questions separate is
#: important: Map Barcodes carries shared fallback copy for several other
#: hosts, but it does not draw their buttons.
FOLD_HOST_MODULES = (
    "spacr.qt.screens.make_masks",
    "spacr.qt.screens.map_barcodes",
    "spacr.qt.screens.image_umap",
    "spacr.qt.screens.regression",
    "spacr.qt.screens.measure",
    "spacr.qt.screens.mask",
    "spacr.qt.screens.classify",
    "spacr.qt.screens.annotate",
)


def folded_modules() -> dict:
    """Every folded key, as ``key -> (name, description, stage, host)``.

    The host is determined by the button list it actually draws, not by the
    location of fallback text. The latter may be shared: Map Barcodes keeps
    descriptions for folds owned by Image UMAP, Regression, Mask, and other
    screens. Treating that shared catalog as membership sends documentation
    and tutorial links to the wrong screen.

    A host's own fallback record is preferred. If it does not keep one, the
    other host tables are searched in declaration order. The first host to
    list a duplicated key still wins, making an accidental double fold
    deterministic until its invariant test reports the duplication.

    Imported lazily, one host at a time and each guarded: this module is
    imported while a screen is being built, and the hosts import it back.

    :returns: a fresh dict; callers may keep or mutate it.
    """
    import importlib

    hosts = []
    for module_name in FOLD_HOST_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:                               # noqa: BLE001
            continue
        members = getattr(module, "FOLDED_APPS", None)
        if members is None:
            members = getattr(module, "FOLD_ORDER", ())
        try:
            members = tuple(str(key) for key in (members or ()))
        except TypeError:
            continue
        table = getattr(module, "FOLD_FALLBACK", None)
        if not isinstance(table, dict):
            table = {}
        hosts.append((module_name, members, table))

    fallback_tables = [table for _name, _members, table in hosts if table]
    found: dict = {}
    for module_name, members, own_table in hosts:
        for key in members:
            entry = own_table.get(key)
            if not entry:
                entry = next(
                    (table[key] for table in fallback_tables if table.get(key)),
                    None,
                )
            if key in found or not entry:
                continue
            name, description, stage = (tuple(entry) + ("", "", ""))[:3]
            found[key] = (name, description, stage or "stable", module_name)
    return found


def folded_fallback(key: str) -> Tuple[str, str, str]:
    """``(name, description, stage)`` a folded key kept, or three blanks.

    The answer for a module that is a button on some host's masthead rather
    than a tile on Home. Blank when the key is not folded anywhere, so a
    caller can tell "folded, and this is what it said" from "never heard of
    it" -- which the registry cannot, because it answers both the same way.
    """
    entry = folded_modules().get(str(key))
    return (entry[0], entry[1], entry[2]) if entry else ("", "", "")


def _describe(key: str) -> Tuple[str, str, str]:
    """Return ``(name, description, stage)`` for one folded module's key.

    THE REGISTRY ANSWERS FIRST, AND STOPS ANSWERING the day the key's row
    is dropped -- which is how folding a module ends. From then on the
    only record is the host's own ``FOLD_FALLBACK``, and it has to be
    consulted for all three fields rather than just the name:
    :func:`spacr.qt.app.app_stage` reports "stable" for a key it has never
    heard of, so a module somebody assessed as alpha goes on lighting up
    in the colour of finished code. The name matters as much -- without
    the fallback it comes back as the key title-cased, which turns Explain
    CV Model into "Explain Cv" and AnnData Export into "Anndata Export".

    Imported lazily and defensively: this widget is constructed while a
    screen is being built, and :mod:`spacr.qt.app` imports screens. A
    module-level import would close that circle.
    """
    default_name = key.replace("_", " ").title()
    try:
        from .. import app as app_module
    except Exception:                                   # pragma: no cover
        app_module = None
    name, description, registered = default_name, "", False
    for row in (getattr(app_module, "APPS", ()) if app_module else ()):
        if row and row[0] == key:
            name = row[1] or name
            description = row[2] or ""
            registered = True
            break
    if registered:
        stage_of = getattr(app_module, "app_stage", None)
        stage = stage_of(key) if callable(stage_of) else "stable"
        return name, description, stage
    kept_name, kept_description, kept_stage = folded_fallback(key)
    return (kept_name or name, kept_description or description,
            kept_stage or "stable")


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

    def set_stage(self, stage: str) -> None:
        """Re-state the maturity this button is drawn in.

        The stage is read from the app registry when the button is built,
        and the registry stops answering for a module the day its row is
        dropped -- which is what folding one ends in. Restating it has to
        move BOTH things the stage decides: the Qt property the shipped
        hover rule selects on, and the widget-local ``:checked`` fill,
        which is a stylesheet computed once from whatever the stage was
        at construction. Moving only the property left a switch that
        hovered in its own colour and lit stable-blue when it was on.

        :param stage: ``"alpha"``, ``"beta"`` or ``"stable"``.
        """
        stage = str(stage or "")
        if not stage or self.property("stage") == stage:
            return
        self.setProperty("stage", stage)
        if self.isCheckable():
            self._install_checked_fill(stage)
        # A property the stylesheet selects on is only read at polish, so
        # a button already on screen keeps the old colour until it is
        # polished again.
        self.style().unpolish(self)
        self.style().polish(self)

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


# ---------------------------------------------------------------------------
# The other half of the icon: the settings the module left behind
# ---------------------------------------------------------------------------
#
# A fold that becomes a BUTTON keeps its picture -- the button is the
# picture. A fold that becomes SETTINGS CATEGORIES has no button and so
# nowhere obvious to put it, and a group of settings that arrived from
# somewhere else says nothing about where. The mark goes on the heading:
# the same icon, beside the category name, on the host's own form.


def mark_folded_sections(key: str, sections: Iterable[QWidget]
                         ) -> Tuple[str, ...]:
    """Mark settings-section headings with a folded module's icon.

    :param key: Folded module registry key.
    :param sections: Section widgets containing that module's settings.
    :returns: Titles of sections marked successfully. Invalid sections and
        modules without artwork are skipped.
    """
    name = _describe(key)[0]
    marked = []
    for section in sections:
        setter = getattr(section, "set_source_app", None)
        if not callable(setter):
            continue
        try:
            if setter(key, name):
                marked.append(_category_title(section))
        except Exception:                               # noqa: BLE001
            continue
    return tuple(marked)


def mark_folded_categories(sections: Iterable[QWidget],
                           categories: dict) -> dict:
    """Mark host categories associated with folded modules.

    :param sections: Host settings sections.
    :param categories: Mapping from module key to category titles. Titles are
        matched case-insensitively.
    :returns: Mapping from module keys to titles marked successfully.
    """
    by_title = {}
    for section in sections:
        by_title.setdefault(_category_title(section).strip().upper(),
                            []).append(section)
    marked = {}
    for key, titles in (categories or {}).items():
        found = []
        for title in titles:
            for section in by_title.get(str(title).strip().upper(), ()):
                found.extend(mark_folded_sections(key, (section,)))
        if found:
            marked[key] = tuple(found)
    return marked


def _category_title(section) -> str:
    """What a settings category is called, as it was written down.

    ``settingsCategorySource`` is the mixed-case name every settings
    section carries; ``title()`` answers with the uppercased heading, so
    it is the fallback rather than the first question.
    """
    source = section.property("settingsCategorySource")
    if source:
        return str(source)
    title = getattr(section, "title", None)
    return str(title() if callable(title) else (title or ""))
