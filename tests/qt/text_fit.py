"""Find user-facing text that its widget cuts off or elides (item 746).

``offenders(root)`` walks the visible widgets under ``root`` and returns one
record per piece of text that does not fit the room its widget gives it:

* a ``QLabel`` whose plain or rich text needs more width (single line) or
  more height (word wrapped) than its contents rectangle;
* a ``QAbstractButton`` smaller than the size its own style asks for its
  text;
* a ``QLineEdit`` placeholder wider than the editor's text area;
* a ``QComboBox`` whose current text is wider than its edit field;
* a ``QGroupBox`` title wider than the box;
* a ``QTabBar`` tab drawn narrower than its text needs while the bar elides;
* a header section narrower than its text.

Only what a person can see is measured: hidden widgets, widgets outside
their window and empty texts are skipped. Widgets marked with the dynamic
property ``spacrElidesByDesign`` (paths, file names, table cells) are
exempt. Each record is a stable key -- the window's class, the widget's
object name or class with its text -- so a ratchet can list them exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QFontMetrics, QTextDocument
from PySide6.QtGui import Qt as _GuiQt
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QGroupBox,
                               QHeaderView, QLabel, QLineEdit, QStyle,
                               QStyleOptionComboBox, QTabBar, QWidget)

#: Pixels of slack before a measurement counts as cut: font metrics and
#: layouts round independently.
SLACK = 2

ELIDES_BY_DESIGN = "spacrElidesByDesign"


@dataclass(frozen=True)
class Offender:
    """One piece of text that does not fit."""

    window: str
    kind: str
    name: str
    text: str
    needed: int
    room: int

    def key(self) -> str:
        """A stable identity for a ratchet list."""
        text = " ".join(self.text.split())[:60]
        return f"{self.window}|{self.kind}|{self.name}|{text}"


def _plain(text: str) -> str:
    if _GuiQt.mightBeRichText(text):
        document = QTextDocument()
        document.setHtml(text)
        return document.toPlainText()
    return text


def _name(widget: QWidget) -> str:
    return widget.objectName() or type(widget).__name__


def _shown(widget: QWidget) -> bool:
    if not widget.isVisible() or widget.width() <= 0 or widget.height() <= 0:
        return False
    if widget.property(ELIDES_BY_DESIGN):
        return False
    region = widget.visibleRegion()
    return not region.isEmpty()


def _label_need(label: QLabel):
    text = label.text()
    if not text or label.pixmap() is not None and not label.pixmap().isNull():
        return None
    rect = label.contentsRect()
    margin = label.margin()
    room = rect.adjusted(margin, margin, -margin, -margin)
    if room.width() <= 0:
        return None
    rich = (label.textFormat() == Qt.RichText
            or (label.textFormat() == Qt.AutoText and _GuiQt.mightBeRichText(text)))
    if label.wordWrap():
        needed = label.heightForWidth(label.width())
        if needed > label.height() + SLACK:
            return ("height", needed, label.height())
        return None
    if rich:
        document = QTextDocument()
        document.setDefaultFont(label.font())
        document.setDocumentMargin(0)
        document.setHtml(text)
        width = int(document.idealWidth() + 0.999)
        height = int(document.size().height() + 0.999)
    else:
        metrics = QFontMetrics(label.font())
        lines = text.split("\n")
        width = max(metrics.horizontalAdvance(line) for line in lines)
        height = metrics.height() * len(lines)
    if width > room.width() + SLACK:
        return ("width", width, room.width())
    if height > room.height() + SLACK and label.minimumHeight() < height:
        return ("height", height, room.height())
    return None


def _button_need(button: QAbstractButton):
    if not button.text():
        return None
    hint = button.sizeHint()
    if hint.width() > button.width() + SLACK:
        return ("width", hint.width(), button.width())
    if hint.height() > button.height() + SLACK + 2:
        metrics = QFontMetrics(button.font())
        if metrics.height() > button.height():
            return ("height", metrics.height(), button.height())
    return None


def _placeholder_need(edit: QLineEdit):
    if edit.text() or not edit.placeholderText():
        return None
    margins = edit.textMargins()
    room = (edit.contentsRect().width() - margins.left() - margins.right()
            - 4)
    needed = QFontMetrics(edit.font()).horizontalAdvance(edit.placeholderText())
    if needed > room + SLACK:
        return ("width", needed, room)
    return None


def _combo_need(combo: QComboBox):
    text = combo.currentText()
    if not text or combo.isEditable():
        return None
    option = QStyleOptionComboBox()
    option.initFrom(combo)
    option.currentText = text
    field = combo.style().subControlRect(
        QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, combo)
    needed = QFontMetrics(combo.font()).horizontalAdvance(text)
    if not combo.itemIcon(combo.currentIndex()).isNull():
        needed += combo.iconSize().width() + 4
    if needed > field.width() + SLACK:
        return ("width", needed, field.width())
    return None


def _group_need(box: QGroupBox):
    if not box.title():
        return None
    needed = QFontMetrics(box.font()).horizontalAdvance(box.title()) + 16
    if needed > box.width() + SLACK:
        return ("width", needed, box.width())
    return None


def _tab_needs(bar: QTabBar):
    if bar.elideMode() == Qt.ElideNone and bar.usesScrollButtons():
        return []
    found = []
    for index in range(bar.count()):
        if not bar.isTabVisible(index) or not bar.tabText(index):
            continue
        actual = bar.tabRect(index).width()
        wanted = bar.tabSizeHint(index).width()
        if wanted > actual + SLACK:
            found.append((bar.tabText(index), wanted, actual))
    return found


def _header_needs(header: QHeaderView):
    model = header.model()
    if model is None or header.property(ELIDES_BY_DESIGN):
        return []
    found = []
    metrics = QFontMetrics(header.font())
    orientation = header.orientation()
    count = (model.columnCount() if orientation == Qt.Horizontal
             else model.rowCount())
    if orientation != Qt.Horizontal:
        return []
    for logical in range(count):
        if header.isSectionHidden(logical):
            continue
        text = model.headerData(logical, orientation, Qt.DisplayRole)
        if text is None or not str(text):
            continue
        size = header.sectionSize(logical)
        if size <= 0:
            continue
        needed = metrics.horizontalAdvance(str(text)) + 12
        if needed > size + SLACK:
            found.append((str(text), needed, size))
    return found


def offenders(root: QWidget, window_name: str = "", *,
              with_widgets: bool = False) -> list:
    """Every piece of visible text under ``root`` that does not fit.

    :param with_widgets: return ``(offender, widget)`` pairs, for screenshots.
    """
    window = window_name or type(root).__name__
    found: List[Offender] = []
    widgets = [root] + root.findChildren(QWidget)
    for widget in widgets:
        if not _shown(widget):
            continue
        need = None
        kind = ""
        if isinstance(widget, QLabel):
            need, kind = _label_need(widget), "label"
            text = _plain(widget.text())
        elif isinstance(widget, QAbstractButton):
            need, kind = _button_need(widget), "button"
            text = widget.text()
        elif isinstance(widget, QLineEdit):
            need, kind = _placeholder_need(widget), "placeholder"
            text = widget.placeholderText()
        elif isinstance(widget, QComboBox):
            need, kind = _combo_need(widget), "combo"
            text = widget.currentText()
        elif isinstance(widget, QGroupBox):
            need, kind = _group_need(widget), "group"
            text = widget.title()
        if need is not None:
            found.append((Offender(window, f"{kind}-{need[0]}", _name(widget),
                                   text, need[1], need[2]), widget))
        if isinstance(widget, QTabBar):
            for text, needed, room in _tab_needs(widget):
                found.append((Offender(window, "tab", _name(widget), text,
                                       needed, room), widget))
        if isinstance(widget, QHeaderView):
            for text, needed, room in _header_needs(widget):
                found.append((Offender(window, "header", _name(widget), text,
                                       needed, room), widget))
    if with_widgets:
        return found
    return [record for record, _widget in found]


def bounding_rect(widget: QWidget) -> QRect:
    """The widget's rectangle in its window, for screenshots of offenders."""
    top_left = widget.mapTo(widget.window(), widget.rect().topLeft())
    return QRect(top_left, widget.size())


#: The configurations the ratchet sweeps: Zoom and interface language.
CONFIGURATIONS = (("1.0", "en"), ("2.0", "en"), ("1.0", "de"), ("2.0", "de"))


def sweep(scale: float, language: str, shots: str = "") -> list:
    """Open the main window and every module screen; list what is cut off.

    Run in a fresh process (see :func:`main`): preferences are written to a
    private store under a temporary ``SPACR_HOME``, the window is 1600 x
    1000 and each screen is measured after the event loop has settled.
    Visible dialogs (the first-run notices) are measured with the screen
    in front. The first-run tour, which otherwise appears after a delay
    that depends on load, is marked seen and measured once, on purpose,
    over the home screen, and so are the anonymised-report notice and each
    page of Preferences.

    :param scale: the Zoom factor.
    :param language: the interface language code.
    :param shots: a folder to save a picture of each offender into.
    :returns: sorted unique ``"config|kind|name|text"`` keys, each with
        the first screen it was seen on. Digits in the text become ``#``,
        so a count or a clock does not change the key.
    """
    import os
    import tempfile

    from PySide6.QtCore import QRect as _QRect
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    tmp = tempfile.mkdtemp(prefix="spacr-text-fit-")
    os.environ["SPACR_HOME"] = os.path.join(tmp, "home")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(tmp, "config")
    os.environ["SPACR_LANGUAGE"] = language
    os.environ["SPACR_NO_NETWORK"] = "1"
    app = QApplication.instance() or QApplication([])
    from spacr.qt import preferences

    store = os.path.join(tmp, "preferences.ini")
    preferences._settings = lambda: QSettings(store, QSettings.IniFormat)
    preferences.set_font_scale(float(scale))
    preferences.set_language(language)
    preferences.set_reduce_motion(True)
    preferences.set_easy_mode(False)
    from spacr.qt import app as app_mod
    from spacr.qt.first_run import mark_tour_seen, maybe_show_tour

    mark_tour_seen()
    preferences.apply_preferences_to_app(app)
    window = app_mod.MainWindow(initial_app="__home__")
    window.resize(1600, 1000)
    window.show()
    config = f"{language}-{scale}"
    found = {}

    def settled(screen):
        """Offenders once two looks a quarter second apart agree."""
        from PySide6.QtTest import QTest

        previous = None
        for _ in range(16):
            QTest.qWait(250)
            roots = [window] + [w for w in app.topLevelWidgets()
                                if w.isVisible() and w is not window]
            pairs = [pair for root in roots
                     for pair in offenders(root, screen, with_widgets=True)]
            current = sorted(record.key() for record, _widget in pairs)
            if current == previous:
                return pairs
            previous = current
        return pairs

    def measure(screen):
        for record, widget in settled(screen):
            text = re.sub(r"\d+", "#", " ".join(record.text.split()))[:60]
            key = f"{config}|{record.kind}|{record.name}|{text}"
            if key in found:
                continue
            found[key] = screen
            if shots:
                top = widget.window()
                corner = widget.mapTo(top, widget.rect().topLeft())
                area = _QRect(corner, widget.size()).adjusted(
                    -40, -20, 40, 20).intersected(top.rect())
                top.grab(area).save(os.path.join(
                    shots, f"{len(found):03d}-{screen}-{record.kind}.png"))

    measure("__home__")
    tour = maybe_show_tour(window, force=True)
    if tour is not None:
        measure("__tour__")
        tour.close()
        tour.deleteLater()
    from spacr.qt.install_consent import AnonymousReportNotice

    notice = AnonymousReportNotice(window)
    notice.show()
    measure("__report_notice__")
    notice.close()
    notice.deleteLater()

    from PySide6.QtWidgets import QTabWidget

    dialog = preferences.PreferencesDialog(window)
    dialog.show()
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    for index in range(tabs.count() if tabs is not None else 0):
        tabs.setCurrentIndex(index)
        measure(f"__preferences__:{index}")
    dialog.close()
    dialog.deleteLater()

    for key, *_rest in app_mod.APPS:
        window._on_nav_selected(key)
        measure(key)
    return sorted(f"{key}|@{screen}" for key, screen in found.items())


def main(argv=None) -> int:
    """``python -m tests.qt.text_fit SCALE LANGUAGE OUT [SHOTS]``."""
    import json
    import os
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    scale, language, out = args[0], args[1], args[2]
    shots = args[3] if len(args) > 3 else ""
    keys = sweep(float(scale), language, shots)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(keys, handle, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
