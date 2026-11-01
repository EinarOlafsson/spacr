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
    if label.wordWrap() or label.hasHeightForWidth():
        if not rich:
            metrics = QFontMetrics(label.font())
            widest = max((metrics.horizontalAdvance(word)
                          for word in text.split()), default=0)
            if widest > room.width() + SLACK:
                return ("width", widest, room.width())
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
    painted = getattr(combo, "_painted_text_layout", None)
    if callable(painted):
        font, rect, flags, text = painted()
        bounds = QFontMetrics(font).boundingRect(
            QRect(0, 0, rect.width(), 1 << 20), flags, text)
        if bounds.width() > rect.width() + SLACK:
            return ("width", bounds.width(), rect.width())
        if bounds.height() > rect.height() + SLACK:
            return ("height", bounds.height(), rect.height())
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


def _protocol_states():
    """Protocol choices the protocol sweep walks, as (family, format, choices).

    Each dependent field is varied once, so every long convention, organism
    and sequencer name is shown at least once.
    """
    from spacr.qt.organisms import ORGANISMS
    from spacr.regex_infer import _METADATA_CONVENTIONS
    from spacr.qt.widgets.sample_project import workflow_map

    wizard = workflow_map()["protocol_wizard"]
    states = [(family, "", {}) for family in wizard["families"]]
    for key in [row["key"] for row in _METADATA_CONVENTIONS]:
        states.append(("plate_images", "tiff_other",
                       {"metadata_type": key, "goal": "recommended"}))
    for key in wizard["image_formats"]:
        states.append(("small_images", "other_metadata",
                       {"image_format": key, "metadata_type": "custom"}))
    for key in wizard["sequencers"]:
        states.append(("sequencing", "fastq_gz", {"sequencer": key}))
    for key in wizard["read_modes"]:
        states.append(("sequencing", "fastq_gz",
                       {"sequencer": "illumina", "read_mode": key,
                        "single_direction": "R1"}))
    for key in wizard["barcodes"]:
        states.append(("features", "database_crops",
                       {"experiment": "screen", "barcodes": key}))
    for key in ["other", *ORGANISMS]:
        states.append(("features", "database",
                       {"experiment": "other", "organism": key}))
    return states


def protocol_sweep(scale: float, language: str) -> list:
    """Walk the Navigator protocol steps and list what is cut off in them.

    Every family, then each long dependent choice (all filename
    conventions, image formats, sequencers, read pairings, barcode
    states and organisms) is chosen through the real selectors, and the
    protocol controls and description box are measured after each choice.

    :param scale: the Zoom factor.
    :param language: the interface language code.
    :returns: sorted unique offender keys, each with its first state.
    """
    import os
    import tempfile

    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication

    tmp = tempfile.mkdtemp(prefix="spacr-text-fit-protocols-")
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
    preferences.set_ambient_enabled(False)
    preferences.set_easy_mode(True)
    from spacr.qt import app as app_mod
    from spacr.qt.first_run import mark_tour_seen

    mark_tour_seen()
    preferences.apply_preferences_to_app(app)
    window = app_mod.MainWindow(initial_app="__home__")
    window.resize(1600, 1000)
    window.show()
    page = window._startup
    config = f"{language}-{scale}"
    found = {}

    def settle(ms=150):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def pick(combo, key):
        index = combo.findData(key)
        if index < 0:
            return
        combo.setCurrentIndex(index)
        combo.activated.emit(index)

    settle(800)
    page.path_edit.protocols.setChecked(True)
    for family, format_, choices in _protocol_states():
        page._protocol_step = 0
        page._protocol_render()
        pick(page._protocol_select, family)
        if format_:
            page._protocol_next()
            pick(page._protocol_select, format_)
            for name, value in choices.items():
                combo = page._protocol_detail_combos.get(name)
                if combo is not None:
                    pick(combo, value)
        settle()
        state = "/".join([family, format_, *choices.values()]).strip("/")
        for record in offenders(page.field.protocol_controls, "NavigatorProtocols"):
            text = re.sub(r"\d+", "#", " ".join(record.text.split()))[:60]
            found.setdefault(f"{config}|{record.kind}|{record.name}|{text}", state)
    window.close()
    return sorted(f"{key}|@{state}" for key, state in found.items())


#: Complete protocols whose assumption lists the box check reads, with the
#: longest lists first: (family, format, choices).
PROTOCOL_BOX_PATHS = (
    ("sequencing", "fastq_gz", {"sequencer": "ont", "read_mode": "single",
                                "single_direction": "R2"}),
    ("features", "database_crops", {"experiment": "screen", "barcodes": "unmapped"}),
    ("features", "database", {"experiment": "drug"}),
    ("features", "database", {"experiment": "other", "organism": "mammalian"}),
    ("small_images", "other_metadata", {"image_format": "nd2",
                                        "metadata_type": "leica_matrix_screener",
                                        "goal": "recommended"}),
    ("plate_images", "tiff_other", {"metadata_type": "imagexpress"}),
    ("sequencing", "other", {}),
)


def protocol_box_sweep(scale: float, language: str, size: str) -> list:
    """Check that the description box shows a protocol's whole assumption list when it can.

    For each path the box is read on the last choice step and once the
    protocol is reached. Whenever the wrapped text fits in the free height
    below the composer, the box must show all of it without scrolling.

    :param scale: the Zoom factor.
    :param language: the interface language code.
    :param size: the window size as ``WIDTHxHEIGHT``.
    :returns: one ``config|state|needed>shown`` entry per box that scrolls
        although its text would fit.
    """
    import os
    import tempfile

    from PySide6.QtCore import QEventLoop, QSettings, QTimer
    from PySide6.QtWidgets import QApplication

    tmp = tempfile.mkdtemp(prefix="spacr-text-fit-box-")
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
    preferences.set_ambient_enabled(False)
    preferences.set_easy_mode(True)
    from spacr.qt import app as app_mod
    from spacr.qt.first_run import mark_tour_seen

    mark_tour_seen()
    preferences.apply_preferences_to_app(app)
    width, height = (int(part) for part in size.split("x"))
    window = app_mod.MainWindow(initial_app="__home__")
    window.resize(width, height)
    window.show()
    page = window._startup
    config = f"{language}-{scale}-{size}"
    failures = []

    def settle(ms=200):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec()

    def pick(combo, key):
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)
            combo.activated.emit(index)

    def check(state):
        settle()
        box = page.field.details
        margins = box.viewportMargins()
        frame = 2 * box.frameWidth()
        document = box.document().clone()
        document.setTextWidth(max(1, box.width() - margins.left() - margins.right() - frame))
        needed = (int(document.size().height() + 0.999) + margins.top()
                  + margins.bottom() + frame)
        free = page.field._details_grip._max
        if (box.isVisible() and needed <= free
                and box.verticalScrollBar().maximum() > 0):
            failures.append(f"{config}|{state}|{needed}>{box.height()}")

    settle(800)
    page.path_edit.protocols.setChecked(True)
    for family, format_, choices in PROTOCOL_BOX_PATHS:
        page._protocol_step = 0
        page._protocol_render()
        pick(page._protocol_select, family)
        page._protocol_next()
        pick(page._protocol_select, format_)
        for name, value in choices.items():
            combo = page._protocol_detail_combos.get(name)
            if combo is not None:
                pick(combo, value)
        state = "/".join([family, format_, *choices.values()])
        check(state + "@choices")
        page._protocol_next()
        dialog = getattr(window, "_protocol_browser_dialog", None)
        if dialog is not None:
            dialog.close()
        check(state + "@protocol")
        page._protocol_back()
    window.close()
    return failures


def main(argv=None) -> int:
    """``python -m tests.qt.text_fit SCALE LANGUAGE OUT [SHOTS] [--protocols | --protocol-box=WxH]``."""
    import json
    import os
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    protocols = "--protocols" in args
    box = next((arg.split("=", 1)[1] for arg in args
                if arg.startswith("--protocol-box=")), "")
    args = [arg for arg in args
            if arg != "--protocols" and not arg.startswith("--protocol-box=")]
    scale, language, out = args[0], args[1], args[2]
    shots = args[3] if len(args) > 3 else ""
    if box:
        keys = protocol_box_sweep(float(scale), language, box)
    elif protocols:
        keys = protocol_sweep(float(scale), language)
    else:
        keys = sweep(float(scale), language, shots)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(keys, handle, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
