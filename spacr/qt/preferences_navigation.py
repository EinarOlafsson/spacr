"""Open the Preferences dialog on one tab, at one row.

Instruction 422. A preference result in the Help search field has to land on
the row the user asked for, and the dialog it lands in is built by seven
thousand lines of procedure with no schema to address. So this module
navigates the BUILT dialog, by the two handles the build leaves behind:

* every page sets an object name -- ``PreferencesTabGeneral``,
  ``PreferencesTabAnimation`` -- and those names are what
  ``tools/build_help_search_index.py`` reads out of the same source, so the
  index and the dialog agree by construction rather than by a second list;
* every row's caption goes through :func:`spacr.qt.i18n.tr`, so the English
  in the index is turned into the displayed caption the same way the dialog
  turned it, and a Korean interface is matched as well as an English one.

Matching the DISPLAYED caption against an English literal was the version
before this one, and it worked in exactly one language.

Kept out of ``preferences.py`` on purpose: that module is imported headless,
and everything here needs live widgets.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QWidget,
)

from .i18n import tr

LOG = logging.getLogger("spacr.qt.preferences_navigation")

#: The tab bar's object name, set where the dialog is built.
TABS_NAME = "PreferencesTabs"

#: How long a row reached from the Help search stays outlined, matching the
#: settings panel's mark so the two arrivals look like one feature.
MARK_MS = 4000


def tab_widget(dialog: QWidget) -> Optional[QTabWidget]:
    """The dialog's tab bar.

    :param dialog: the preferences dialog.
    :returns: the :class:`QTabWidget`, or ``None``.
    """
    found = dialog.findChild(QTabWidget, TABS_NAME)
    if found is not None:
        return found
    return dialog.findChild(QTabWidget)


def page_named(dialog: QWidget, object_name: str) -> Optional[QWidget]:
    """The page whose object name is ``object_name``.

    :param dialog: the preferences dialog.
    :param object_name: e.g. ``"PreferencesTabTheme"``.
    :returns: the page widget, or ``None``.
    """
    if not object_name:
        return None
    return dialog.findChild(QWidget, str(object_name))


def show_tab(dialog: QWidget, object_name: str, label: str = "") -> bool:
    """Bring the page named ``object_name`` to the front, at row ``label``.

    :param dialog: the preferences dialog, already built.
    :param object_name: the page's object name.
    :param label: the English caption of the row to mark; ``""`` marks none.
    :returns: True when the page was found and shown.
    """
    tabs = tab_widget(dialog)
    page = page_named(dialog, object_name)
    if tabs is None or page is None:
        return False
    for index in range(tabs.count()):
        holder = tabs.widget(index)
        if holder is page or (holder is not None
                              and page in holder.findChildren(QWidget)):
            tabs.setCurrentIndex(index)
            break
    else:
        return False
    if label:
        QTimer.singleShot(0, lambda: reveal_row(page, label))
    return True


def _captions(label: str) -> List[str]:
    """Every spelling of ``label`` a live dialog might be showing.

    :param label: the English caption from the generated index.
    :returns: the translated caption first, then the English one.
    """
    out = []
    try:
        translated = tr(label)
        if translated:
            out.append(str(translated))
    except Exception:
        LOG.debug("could not translate %r", label, exc_info=True)
    if label not in out:
        out.append(str(label))
    return out


def row_field(page: QWidget, label: str) -> Optional[QWidget]:
    """The field widget of the row captioned ``label``.

    :param page: the preferences page.
    :param label: the English caption.
    :returns: the field widget, or ``None`` when no row carries that caption.
    """
    wanted = {c.replace("&", "").strip() for c in _captions(label)}
    for form in page.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            item = form.itemAt(row, QFormLayout.LabelRole)
            caption = item.widget() if item is not None else None
            text = caption.text() if isinstance(caption, QLabel) else ""
            if text.replace("&", "").strip() not in wanted:
                continue
            field = form.itemAt(row, QFormLayout.FieldRole)
            widget = field.widget() if field is not None else None
            return widget if widget is not None else caption
    return None


def reveal_row(page: QWidget, label: str) -> bool:
    """Scroll to the row captioned ``label`` and outline it.

    A STATIC outline for a few seconds rather than anything that moves: a
    mark that pulses would owe an answer to the Animation preferences, and
    this one owes none and is not lost on a reader who turned motion off.

    :param page: the preferences page.
    :param label: the English caption.
    :returns: True when the row was found.
    """
    widget = row_field(page, label)
    if widget is None:
        return False
    scroll = page.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    try:
        if isinstance(scroll, QScrollArea):
            scroll.ensureWidgetVisible(widget, 0, 40)
        widget.setFocus(Qt.ShortcutFocusReason)
    except RuntimeError:
        return False
    previous = widget.styleSheet()
    widget.setProperty("spacrRevealed", True)
    widget.setStyleSheet(
        previous + "\nQWidget { border: 1px solid palette(highlight); }")

    def _unmark() -> None:
        """Put the row back the way it was found."""
        try:
            widget.setProperty("spacrRevealed", False)
            widget.setStyleSheet(previous)
        except RuntimeError:
            LOG.debug("the marked row went away before the mark did")

    QTimer.singleShot(MARK_MS, _unmark)
    return True
