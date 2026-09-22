"""Every piece of text reads: bright on the dark theme, dark on the light one.

Maintainer, 2026-09-21: "make sure that when in dark mode text in and outside
of fields is bright, and the oposite should be true for light mode."

MEASURED OFF RENDERED PIXELS, not off the palette. A palette can say white
while a native style, a stylesheet rule nobody reached, or a translucent
panel over the page paints something else, and only the pixels are what a
user reads. Each screen is rendered into an image pre-filled with the
theme's page colour, and for every visible text-bearing widget the
commonest colour in its box is taken as the background and the colour that
stands out most from it as the ink. The ratio is WCAG's.

Two floors: 4.5:1 for text, 3:1 for a placeholder or a disabled control,
which must stay readable while looking inactive. A disabled control must
also be DIMMER than an enabled one, or "disabled" cannot be seen at all.

The first measurement, before the fixes this file arrived with: disabled
QPlainTextEdit and QTextEdit were painted exactly like enabled ones on both
themes; the dark theme's dim ink (placeholders, disabled fields, captions)
was 3.5:1 against a field; the light theme's outlined action buttons drew
their captions in the constant #4A9EFF, 2.2:1 on the light page; and the
first setup slide drew its GPU verdict in the dark theme's green on the
light theme too.
"""
from __future__ import annotations

import collections
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtCore import QMargins, QPoint, QRect, QSettings     # noqa: E402
from PySide6.QtGui import QColor, QImage, QStandardItem, QStandardItemModel  # noqa: E402
from PySide6.QtWidgets import (QAbstractButton, QAbstractScrollArea,  # noqa: E402
                               QAbstractSpinBox, QApplication, QCheckBox,
                               QComboBox, QDoubleSpinBox, QFormLayout,
                               QGraphicsOpacityEffect,
                               QLabel, QLineEdit, QListWidget, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableView, QTabWidget,
                               QTextEdit, QWidget)

TEXT_FLOOR = 4.5
QUIET_FLOOR = 3.0
MODES = ("dark", "light")


def _luminance(rgb) -> float:
    def channel(value):
        value /= 255.0
        return value / 12.92 if value <= 0.04045 else (
            (value + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _ratio(a, b) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _sample(image: QImage, rect: QRect):
    """(background, ink, ratio) inside ``rect`` of ``image``."""
    rect = rect.intersected(image.rect())
    counts = collections.Counter()
    for y in range(rect.top(), rect.bottom() + 1):
        for x in range(rect.left(), rect.right() + 1):
            c = image.pixelColor(x, y)
            counts[(c.red(), c.green(), c.blue())] += 1
    if not counts:
        return None
    background = counts.most_common(1)[0][0]
    ink = max(counts, key=lambda colour: _ratio(colour, background))
    return background, ink, _ratio(background, ink)


def _render(widget, page: str) -> QImage:
    """``widget`` painted over the theme's page, as the window would be."""
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor(page))
    widget.render(image)
    return image


def _text_box(widget, root) -> QRect:
    """Where ``widget``'s text is, in ``root``'s coordinates.

    Spin boxes and combos lose their right-hand 40 %: the arrows are drawn
    in the muted ink by design and are not text.
    """
    rect = widget.rect().adjusted(3, 3, -3, -3)
    if isinstance(widget, (QComboBox, QAbstractSpinBox)):
        rect.setWidth(int(rect.width() * 0.6))
    if isinstance(widget, QAbstractScrollArea):
        viewport = widget.viewport()
        return QRect(viewport.mapTo(root, QPoint(2, 2)),
                     viewport.size().shrunkBy(QMargins(2, 2, 2, 2)))
    return QRect(widget.mapTo(root, rect.topLeft()), rect.size())


def _carries_text(widget) -> bool:
    if isinstance(widget, QLabel):
        return bool(widget.text().strip()) and widget.pixmap().isNull()
    if isinstance(widget, QLineEdit):
        return bool(widget.text() or widget.placeholderText())
    if isinstance(widget, (QPlainTextEdit, QTextEdit)):
        return bool(widget.toPlainText() or widget.placeholderText())
    if isinstance(widget, QComboBox):
        return bool(widget.currentText())
    if isinstance(widget, QAbstractSpinBox):
        return bool(widget.text())
    if isinstance(widget, QAbstractButton):
        return bool(widget.text().strip())
    return False


def _is_quiet(widget) -> bool:
    """A placeholder or a disabled control: held to the 3:1 floor."""
    if not widget.isEnabled():
        return True
    if isinstance(widget, QLineEdit) and not widget.text():
        return True
    if isinstance(widget, (QPlainTextEdit, QTextEdit)) and not (
            widget.toPlainText()):
        return True
    return False


def _fading(widget, root) -> bool:
    """Is ``widget`` part-way through a fade? The setup slide's greeting
    fades up and away on a timer; mid-fade it is not yet meant to be read,
    and a render can land on any frame of it."""
    while widget is not None:
        effect = widget.graphicsEffect()
        if (isinstance(effect, QGraphicsOpacityEffect)
                and effect.opacity() < 0.99):
            return True
        if widget is root:
            return False
        widget = widget.parentWidget()
    return False


def _audit(root, image: QImage, where: str):
    """Every visible text-bearing widget under ``root`` below its floor."""
    failures, seen = [], 0
    for widget in root.findChildren(QWidget):
        if (not widget.isVisible() or widget.visibleRegion().isEmpty()
                or widget.width() < 12 or widget.height() < 12
                or not _carries_text(widget) or _fading(widget, root)):
            continue
        measured = _sample(image, _text_box(widget, root))
        if measured is None:
            continue
        seen += 1
        background, ink, ratio = measured
        floor = QUIET_FLOOR if _is_quiet(widget) else TEXT_FLOOR
        if ratio < floor:
            failures.append(
                f"{where}: {type(widget).__name__}#{widget.objectName()} "
                f"{getattr(widget, 'text', lambda: '')()!r:.40} ink {ink} on "
                f"{background} is {ratio:.2f}:1 < {floor}:1")
    return seen, failures


@pytest.fixture
def mode_store(monkeypatch, tmp_path, qt_theme_applied):
    """A private preference store; the test sets the theme in it."""
    from spacr.qt import preferences as prefs

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    return store


@pytest.fixture
def in_mode(mode_store, request):
    """Palette of the requested mode on the application, dark put back."""
    from spacr.qt import preferences as prefs
    from spacr.qt import theme

    app = QApplication.instance()

    def switch(name: str):
        prefs.set_theme(name)
        theme.apply_qpalette(app, name)
        return theme.stylesheet(name), theme.palette_for(name)

    yield switch
    theme.apply_qpalette(app, "dark")


def _gallery():
    """One of every text-bearing control spaCR's screens use."""
    root = QWidget()
    root.setObjectName("ContrastGallery")
    form = QFormLayout(root)
    form.addRow(QLabel("A label outside any field"))
    form.addRow("Line edit", QLineEdit("Typed text"))
    placeholder = QLineEdit()
    placeholder.setPlaceholderText("Placeholder text")
    form.addRow("Placeholder", placeholder)
    disabled = QLineEdit("Disabled text")
    disabled.setEnabled(False)
    form.addRow("Disabled", disabled)
    spin = QSpinBox()
    spin.setRange(0, 99999)
    spin.setValue(88888)
    form.addRow("Spin box", spin)
    dspin = QDoubleSpinBox()
    dspin.setValue(88.88)
    form.addRow("Double spin box", dspin)
    spin_off = QSpinBox()
    spin_off.setRange(0, 99999)
    spin_off.setValue(88888)
    spin_off.setEnabled(False)
    form.addRow("Disabled spin box", spin_off)
    combo = QComboBox()
    combo.addItems(["Combo item", "Second"])
    form.addRow("Combo", combo)
    editable = QComboBox()
    editable.setEditable(True)
    editable.addItems(["Editable combo", "Second"])
    form.addRow("Editable combo", editable)
    combo_off = QComboBox()
    combo_off.addItems(["Disabled combo"])
    combo_off.setEnabled(False)
    form.addRow("Disabled combo", combo_off)
    rich = QTextEdit()
    rich.setPlainText("Rich text")
    rich.setFixedHeight(44)
    form.addRow("Text edit", rich)
    plain = QPlainTextEdit()
    plain.setPlainText("Plain text")
    plain.setFixedHeight(44)
    form.addRow("Plain text edit", plain)
    plain_ph = QPlainTextEdit()
    plain_ph.setPlaceholderText("Plain placeholder")
    plain_ph.setFixedHeight(44)
    form.addRow("Plain placeholder", plain_ph)
    plain_off = QPlainTextEdit()
    plain_off.setPlainText("Disabled plain text")
    plain_off.setFixedHeight(44)
    plain_off.setEnabled(False)
    form.addRow("Disabled plain text", plain_off)
    form.addRow(QCheckBox("A check box"))
    check_off = QCheckBox("A disabled check box")
    check_off.setEnabled(False)
    form.addRow(check_off)
    form.addRow(QPushButton("A button"))
    primary = QPushButton("A primary action")
    primary.setObjectName("PrimaryButton")
    form.addRow(primary)
    rows = QListWidget()
    rows.addItems(["A list row"])
    rows.setFixedHeight(44)
    form.addRow("List", rows)
    table = QTableView()
    model = QStandardItemModel(2, 2, table)
    model.setHorizontalHeaderLabels(["Header", "Second"])
    for r in range(2):
        for c in range(2):
            model.setItem(r, c, QStandardItem("Cell"))
    table.setModel(model)
    table.setFixedHeight(110)
    form.addRow("Table", table)
    root.resize(620, 980)
    return root, {"placeholder": placeholder, "disabled": disabled,
                  "line": form.itemAt(1, QFormLayout.FieldRole).widget(),
                  "plain": plain, "plain_off": plain_off,
                  "combo": combo, "table": table}


def _table_boxes(table, root):
    """The first cell and the first header section, in ``root``."""
    viewport = table.viewport()
    cell = table.visualRect(table.model().index(0, 0))
    header = table.horizontalHeader()
    section = QRect(
        header.mapTo(root, QPoint(header.sectionViewportPosition(0), 0)),
        header.rect().size())
    section.setWidth(header.sectionSize(0))
    return (QRect(viewport.mapTo(root, cell.topLeft()),
                  cell.size()).adjusted(2, 2, -2, -2),
            section.adjusted(2, 2, -4, -4))


@pytest.mark.parametrize("mode", MODES)
def test_every_control_in_the_gallery_reads(mode, in_mode, qtbot):
    sheet, palette = in_mode(mode)
    root, named = _gallery()
    qtbot.addWidget(root)
    root.setStyleSheet(sheet)
    root.show()
    qtbot.waitExposed(root)
    image = _render(root, palette["page"])
    seen, failures = _audit(root, image, mode)
    assert seen >= 18, f"only {seen} text widgets were measured"
    for box, what in zip(_table_boxes(named["table"], root),
                         ("table cell", "table header")):
        _bg, _ink, ratio = _sample(image, box)
        if ratio < TEXT_FLOOR:
            failures.append(f"{mode}: {what} is {ratio:.2f}:1")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mode", MODES)
def test_the_ink_goes_the_right_way(mode, in_mode, qtbot):
    """Bright text on dark, dark text on light -- in and outside fields."""
    sheet, palette = in_mode(mode)
    root, named = _gallery()
    qtbot.addWidget(root)
    root.setStyleSheet(sheet)
    root.show()
    qtbot.waitExposed(root)
    image = _render(root, palette["page"])
    label = root.findChild(QLabel)
    for widget in (label, named["line"], named["plain"], named["combo"]):
        background, ink, _ratio_ = _sample(image, _text_box(widget, root))
        if mode == "dark":
            assert _luminance(ink) > _luminance(background), widget
            assert _luminance(ink) > 0.6, (widget, ink)
        else:
            assert _luminance(ink) < _luminance(background), widget
            assert _luminance(ink) < 0.1, (widget, ink)


@pytest.mark.parametrize("mode", MODES)
def test_disabled_text_is_dimmer_than_enabled_text(mode, in_mode, qtbot):
    sheet, palette = in_mode(mode)
    root, named = _gallery()
    qtbot.addWidget(root)
    root.setStyleSheet(sheet)
    root.show()
    qtbot.waitExposed(root)
    image = _render(root, palette["page"])
    for on, off in (("line", "disabled"), ("plain", "plain_off")):
        _b1, _i1, enabled = _sample(image, _text_box(named[on], root))
        _b2, _i2, disabled = _sample(image, _text_box(named[off], root))
        assert disabled < enabled * 0.75, (
            f"{mode}: disabled {off} reads {disabled:.2f}:1 against "
            f"{enabled:.2f}:1 enabled -- it does not look disabled")


@pytest.mark.parametrize("mode", MODES)
def test_the_combo_popup_list_reads(mode, in_mode, qtbot):
    sheet, palette = in_mode(mode)
    root, named = _gallery()
    qtbot.addWidget(root)
    root.setStyleSheet(sheet)
    root.show()
    qtbot.waitExposed(root)
    combo = named["combo"]
    combo.showPopup()
    try:
        view = combo.view()
        popup = view.window()
        qtbot.waitUntil(popup.isVisible)
        image = _render(popup, palette["bg"])
        rect = view.visualRect(view.model().index(1, 0))
        box = QRect(view.viewport().mapTo(popup, rect.topLeft()),
                    rect.size()).adjusted(2, 2, -2, -2)
        _bg, _ink, ratio = _sample(image, box)
        assert ratio >= TEXT_FLOOR, f"{mode}: popup row {ratio:.2f}:1"
    finally:
        combo.hidePopup()


@pytest.mark.parametrize("mode", MODES)
def test_every_preferences_tab_reads(mode, in_mode, qtbot):
    """The dialog a user opens to change the theme, every tab of it."""
    from spacr.qt.preferences import PreferencesDialog

    sheet, palette = in_mode(mode)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.setStyleSheet(sheet)
    dialog.resize(1000, 760)
    dialog.show()
    qtbot.waitExposed(dialog)
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    failures, seen = [], 0
    for index in range(tabs.count()):
        tabs.setCurrentIndex(index)
        QApplication.processEvents()
        image = _render(dialog, palette["bg"])
        counted, failed = _audit(dialog, image,
                                 f"{mode}/{tabs.tabText(index)}")
        seen += counted
        failures += failed
    assert seen >= 80, f"only {seen} text widgets were measured"
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("mode", MODES)
def test_the_first_run_setup_reads(mode, in_mode, qtbot):
    """The startup window: the setup slides a fresh install opens on."""
    from spacr.qt.widgets.setup_slides import SetupSlides

    sheet, palette = in_mode(mode)
    slides = SetupSlides(None)
    qtbot.addWidget(slides)
    slides.setStyleSheet(sheet)
    slides.resize(1100, 760)
    slides.show()
    qtbot.waitExposed(slides)
    slides.next()
    QApplication.processEvents()
    image = _render(slides, palette["bg"])
    seen, failures = _audit(slides, image, f"{mode}/setup")
    assert seen >= 4, f"only {seen} text widgets were measured"
    assert not failures, "\n".join(failures)


def test_a_light_page_gets_the_light_themes_verdict_colours(in_mode):
    """The GPU verdict on the first slide used the dark theme's green on
    the light page as well, at under 2.5:1."""
    from spacr.qt import theme
    from spacr.qt.widgets import setup_slides

    in_mode("light")
    light = theme.palette_for("light")
    assert setup_slides.verdict_ink(True) == light["success"]
    assert setup_slides.verdict_ink(False) == light["error"]
    for ink in (setup_slides.verdict_ink(True),
                setup_slides.verdict_ink(False)):
        assert theme.contrast_ratio(ink, light["page"]) >= TEXT_FLOOR
    in_mode("dark")
    assert setup_slides.verdict_ink(True) == setup_slides.GPU_YES_INK


@pytest.mark.parametrize("name", ("dark", "light", "glass", "cell"))
def test_an_outlined_action_caption_reads_on_every_page_surface(name):
    """The constant button blue is kept for fills and outlines; its
    caption takes a colour that reads on the theme's own surfaces."""
    from spacr.qt import theme

    palette = theme.palette_for(name)
    ink = theme.button_accent_text(palette)
    for role in ("page", "surface", "surface_alt"):
        surface = theme.effective_surface(name, role)
        assert theme.contrast_ratio(ink, surface) >= TEXT_FLOOR, (
            name, role, ink, surface)


@pytest.mark.parametrize("name", ("dark", "light", "glass"))
def test_the_stylesheet_captions_outlined_actions_in_that_colour(name):
    """The stylesheet's palette carries rgba surfaces, which the contrast
    helper cannot read; the rule must still get the hex answer. The first
    version of this fix fell back to the constant blue there, and the light
    theme's Run button kept its 2.2:1 caption."""
    import re

    from spacr.qt import theme

    sheet = theme.stylesheet(name)
    rule = re.search(
        r'QPushButton#PrimaryButton,\s*QPushButton\[buttonActionRole='
        r'"positive"\]\s*\{([^}]*)\}', sheet)
    assert rule, "the outlined action rule is gone"
    caption = re.search(r'(?<![-\w])color:\s*([^;]+);', rule.group(1))
    assert caption.group(1).strip().lower() == theme.button_accent_text(
        theme.palette_for(name)).lower()
