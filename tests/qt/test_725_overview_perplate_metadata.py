"""Overview metadata step, per plate: each plate keeps its own filename settings,
which a plate list button applies to one plate or to all; the right-hand
container parses a sample of every plate in its own labelled section; channel
colours read on every theme and open a colour picker on a double-click.

Item 725, 1.5.1.6, from the maintainer's notes of 2026-10-26.
"""
import json
import threading

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QAbstractItemView, QApplication, QLabel, QPushButton

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
OTHER = r"(?P<plateID>q\d+)_(?P<wellID>[A-Z]\d+)_s(?P<fieldID>\d+)_w(?P<chanID>\d+)\.tif"


def _plate(folder, wells, name="p1-{well}-f1-c{channel}.tif"):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            pixels = np.full((24, 32), 100 * (index + 1) + channel, dtype=np.uint16)
            tifffile.imwrite(folder / name.format(well=well, channel=channel), pixels)
    return folder


@pytest.fixture
def plates(tmp_path):
    screen = tmp_path / "screen"
    return (_plate(screen / "plate1", ("A01", "A02")),
            _plate(screen / "plate2", ("B01", "B02", "B03"), name="q2_{well}_s1_w{channel}.tif"),
            _plate(screen / "plate3", ("C01",), name="q3_{well}_s1_w{channel}.tif"))


def _wizard(qtbot, source, **settings):
    wizard = browser._OverviewMetadataWizard(str(source), {"custom_regex": PATTERN, **settings})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    wizard.resize(1400, 900)
    wizard.scope.setCurrentIndex(wizard.scope.findData("folder"))
    wizard._scan()
    return wizard


def _read(wizard):
    return {field.plate for field in wizard._fields}


def _row(wizard, folder):
    return [wizard.plate_table.item(row, 2).text() for row in range(wizard.plate_table.rowCount())].index(str(folder))


def _choose(wizard, folder):
    wizard.plate_table.setCurrentCell(_row(wizard, folder), 0)


def _double_click(qtbot, table, item):
    point = table.visualItemRect(item).center()
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=point)
    qtbot.mouseDClick(table.viewport(), Qt.LeftButton, pos=point)


def _type(wizard, text):
    wizard.regex.setText(text)
    wizard.regex.textEdited.emit(text)


def _apply_button(wizard, folder):
    return wizard.plate_table.cellWidget(_row(wizard, folder), 5)


def _sections(wizard):
    return {section.property("platePath"): section for section in wizard._plate_sections}


def _texts(section, name):
    return [label.text() for label in section.findChildren(QLabel, name)]


def test_choosing_a_plate_shows_its_own_filename_settings_and_names_it(qtbot, plates):
    first, second = (str(path) for path in plates[:2])
    wizard = _wizard(qtbot, plates[0].parent, plate_regex={second: OTHER})
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    assert wizard.editing_plate.text() == "Editing plate: plate1"
    assert wizard.regex.text() == PATTERN
    _choose(wizard, second)
    assert wizard.regex.text() == OTHER
    assert wizard.preset.currentData() == "custom"
    assert wizard.editing_plate.text() == "Editing plate: plate2"
    assert wizard.plate_table.item(_row(wizard, second), 0).font().bold()
    assert not wizard.plate_table.item(_row(wizard, first), 0).font().bold()
    assert _sections(wizard)[second].property("editing") is True
    assert _sections(wizard)[first].property("editing") is False
    _choose(wizard, first)
    assert wizard.regex.text() == PATTERN
    assert wizard.editing_plate.text() == "Editing plate: plate1"
    assert wizard.configuration()["custom_regex"] == PATTERN
    assert wizard.configuration()["plate_regex"] == {second: OTHER}


def test_typing_a_regex_edits_only_the_chosen_plate(qtbot, plates):
    second = str(plates[1])
    wizard = _wizard(qtbot, plates[0].parent)
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1"} and wizard.plate_table.rowCount() == 3, timeout=15000)
    _choose(wizard, second)
    assert wizard.regex.text() == PATTERN
    _type(wizard, OTHER)
    assert wizard._plate_regex == {second: OTHER}
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    assert wizard.configuration()["custom_regex"] == PATTERN


def test_apply_settings_beside_a_plate_and_apply_to_all(qtbot, plates):
    first, second, third = (str(path) for path in plates)
    wizard = _wizard(qtbot, plates[0].parent)
    qtbot.waitUntil(lambda: wizard.plate_table.rowCount() == 3 and bool(wizard._fields), timeout=15000)
    header = wizard.plate_table.horizontalHeaderItem(5).text()
    assert header == "Settings" and wizard.plate_table.horizontalHeader().visualIndex(5) == 1
    buttons = {wizard.plate_table.cellWidget(row, 5).property("platePath"): wizard.plate_table.cellWidget(row, 5)
               for row in range(3)}
    assert set(buttons) == {first, second, third}
    assert all(isinstance(button, QPushButton) and button.text() == "Apply settings" for button in buttons.values())
    _choose(wizard, second)
    _type(wizard, OTHER)
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    _apply_button(wizard, third).click()
    assert wizard._plate_regex == {second: OTHER, third: OTHER}
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2", "plate3"}, timeout=15000)
    assert wizard.editing_plate.text() == "Editing plate: plate2"
    apply_all = wizard.findChild(QPushButton, "OverviewWizardApplyAll")
    assert apply_all.text() == "Apply to all plates"
    apply_all.click()
    assert wizard._plate_regex == {}
    qtbot.waitUntil(lambda: _read(wizard) == {"plate2", "plate3"}, timeout=15000)
    assert wizard.configuration()["custom_regex"] == OTHER


def test_applying_to_the_first_plate_leaves_the_other_plates_alone(qtbot, plates):
    first, second, third = (str(path) for path in plates)
    wizard = _wizard(qtbot, plates[0].parent, plate_regex={second: OTHER})
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    _choose(wizard, second)
    _apply_button(wizard, first).click()
    assert wizard._plate_regex == {third: PATTERN}
    assert wizard.regex.text() == OTHER and wizard.editing_plate.text() == "Editing plate: plate2"
    qtbot.waitUntil(lambda: _read(wizard) == {"plate2"} and wizard.next.isEnabled(), timeout=15000)
    assert wizard.configuration()["custom_regex"] == OTHER
    _choose(wizard, third)
    assert wizard.regex.text() == PATTERN


def _column(table, caption):
    return [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())].index(caption)


def _values(table, caption):
    column = _column(table, caption)
    return sorted({table.item(row, column).text() for row in range(table.rowCount())})


def test_the_right_container_parses_each_plate_in_its_own_section(qtbot, plates):
    first, second = (str(path) for path in plates[:2])
    wizard = _wizard(qtbot, plates[0].parent, plate_regex={second: OTHER})
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    sections = _sections(wizard)
    assert list(sections)[:2] == [first, second]
    assert _texts(sections[first], "OverviewWizardPlatePreviewTitle") == ["plate1"]
    assert _texts(sections[second], "OverviewWizardPlatePreviewTitle") == ["plate2"]
    assert _texts(sections[first], "OverviewWizardPlatePreviewRegex") == ["Filename regex: " + PATTERN]
    assert _texts(sections[second], "OverviewWizardPlatePreviewRegex") == ["Filename regex: " + OTHER]
    assert sections[first].property("plateTone") != sections[second].property("plateTone")
    assert sections[first].objectName() == sections[second].objectName() == "OverviewWizardPlatePreview"
    one = wizard.preview
    two = sections[second].findChild(browser.QWidget, "OverviewWizardPlateParsePreview")
    assert one.parent() is sections[first] and two.parent() is sections[second]
    assert _values(one, "Well") == ["A01", "A02"] and _values(one, "Plate") == ["p1"]
    assert _values(two, "Well") == ["B01", "B02", "B03"] and _values(two, "Plate") == ["q2"]
    assert _values(two, "Channel") == ["1", "2"] and _values(two, "Field") == ["1"]
    assert all(not label.isVisibleTo(section) for section in sections.values()
               for label in section.findChildren(QLabel, "OverviewWizardPlatePreviewError"))
    qss = browser._overview_regex_qss(browser._OverviewMetadataWizard and __import__(
        "spacr.qt.theme", fromlist=["palette_for"]).palette_for("dark"))
    assert 'QFrame#OverviewWizardPlatePreview[plateTone="1"]' in qss


def test_an_invalid_regex_shows_its_error_in_its_own_section(qtbot, plates):
    first, second = (str(path) for path in plates[:2])
    wizard = _wizard(qtbot, plates[0].parent, plate_regex={second: OTHER})
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2"}, timeout=15000)
    _choose(wizard, second)
    _type(wizard, "(unclosed")
    assert wizard._plate_regex == {second: OTHER}
    sections = _sections(wizard)
    errors = {path: section.findChild(QLabel, "OverviewWizardPlatePreviewError") for path, section in sections.items()}
    assert errors[second].isVisibleTo(sections[second])
    assert errors[second].text().startswith("Invalid filename regex: ")
    assert not errors[first].isVisibleTo(sections[first])
    assert _texts(sections[second], "OverviewWizardPlatePreviewRegex") == ["Filename regex: (unclosed"]
    assert not wizard.next.isEnabled()
    _choose(wizard, first)
    _choose(wizard, second)
    assert wizard.regex.text() == "(unclosed"
    _type(wizard, OTHER)
    assert not _sections(wizard)[second].findChild(QLabel, "OverviewWizardPlatePreviewError").isVisibleTo(
        _sections(wizard)[second])
    _choose(wizard, first)
    _type(wizard, "[broken")
    qtbot.waitUntil(lambda: _sections(wizard)[first].findChild(
        QLabel, "OverviewWizardPlatePreviewError").isVisibleTo(_sections(wizard)[first]), timeout=5000)
    assert "unterminated character set" in wizard.status.text()
    assert _sections(wizard)[first].findChild(QLabel, "OverviewWizardPlatePreviewError").text().startswith(
        "Invalid filename regex: unterminated character set")
    assert not _sections(wizard)[second].findChild(QLabel, "OverviewWizardPlatePreviewError").isVisibleTo(
        _sections(wizard)[second])


def _luminance(colour):
    def channel(value):
        value /= 255.
        return value / 12.92 if value <= .03928 else ((value + .055) / 1.055) ** 2.4
    return .2126 * channel(colour.red()) + .7152 * channel(colour.green()) + .0722 * channel(colour.blue())


def _contrast(one, two):
    high, low = sorted((_luminance(one), _luminance(two)), reverse=True)
    return (high + .05) / (low + .05)


@pytest.fixture
def themed(qt_theme_applied):
    from spacr.qt.theme import apply_qpalette, stylesheet

    def use(name):
        apply_qpalette(qt_theme_applied, name)
        qt_theme_applied.setStyleSheet(stylesheet(name))
    yield use
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


@pytest.mark.parametrize("theme", ("dark", "light", "cell", "glass", "high_contrast"))
def test_channel_colour_text_reads_on_every_theme(qtbot, plates, themed, theme):
    themed(theme)
    wizard = _wizard(qtbot, plates[0])
    wizard.show()
    qtbot.waitUntil(lambda: wizard.channel_table.rowCount() == 2, timeout=15000)
    table = wizard.channel_table
    table.clearSelection()
    table.setCurrentCell(-1, -1)
    QApplication.processEvents()
    for row in range(2):
        item = table.item(row, 3)
        assert item.data(Qt.ForegroundRole) is None and item.data(Qt.BackgroundRole) is None
        assert QColor(item.data(Qt.DecorationRole)).name() == QColor(item.text()).name()
        rect = table.visualItemRect(item)
        image = table.viewport().grab(rect).toImage()
        corner = image.pixelColor(image.width() - 2, image.height() // 2)
        far = max((image.pixelColor(x, y) for x in range(image.width() // 3, image.width() - 2)
                   for y in range(image.height())), key=lambda colour: _contrast(colour, corner))
        assert _contrast(far, corner) >= 4.5, (theme, item.text(), far.name(), corner.name())


def test_double_clicking_a_colour_opens_the_picker_and_writes_the_choice_back(qtbot, plates, monkeypatch):
    from spacr.qt.widgets import colour_picker

    asked = []
    monkeypatch.setattr(colour_picker, "pick_colour",
                        lambda parent, initial=None, title=None: asked.append((parent, initial, title))
                        or QColor("#123456"))
    wizard = _wizard(qtbot, plates[0])
    wizard.show()
    qtbot.waitUntil(lambda: wizard.channel_table.rowCount() == 2, timeout=15000)
    table = wizard.channel_table
    before = table.item(1, 3).text()
    _double_click(qtbot, table, table.item(1, 3))
    assert asked == [(wizard, before, "Channel 2 colour")]
    assert table.item(1, 3).text() == "#123456"
    assert QColor(table.item(1, 3).data(Qt.DecorationRole)).name() == "#123456"
    assert table.state() != QAbstractItemView.EditingState
    assert wizard._channel_display()[1][2] == "#123456"
    table.item(0, 3).setText("orange")
    assert wizard._channel_display()[1][1] == "#ffa500"
    assert table.editTriggers() & QAbstractItemView.AnyKeyPressed
    _double_click(qtbot, table, table.item(0, 1))
    assert table.state() == QAbstractItemView.EditingState
    table.closePersistentEditor(table.item(0, 1))
    table.setFocus()
    monkeypatch.setattr(colour_picker, "pick_colour", lambda *args, **kwargs: QColor())
    _double_click(qtbot, table, table.item(1, 3))
    assert table.item(1, 3).text() == "#123456"


def test_the_picker_is_the_spacr_colour_picker_with_rgb_and_hex(monkeypatch, qtbot, plates):
    from PySide6.QtWidgets import QColorDialog

    seen = []
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(
        lambda initial, parent, title, options: seen.append((initial.name(), title, options)) or QColor("#00ff7f")))
    wizard = _wizard(qtbot, plates[0])
    qtbot.waitUntil(lambda: wizard.channel_table.rowCount() == 2, timeout=15000)
    start = wizard.channel_table.item(0, 3).text()
    assert wizard._pick_channel_colour(0).name() == "#00ff7f"
    assert seen == [(QColor(start).name(), "Channel 1 colour", QColorDialog.ColorDialogOption.DontUseNativeDialog)]
    assert wizard.channel_table.item(0, 3).text() == "#00ff7f"


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


def test_per_plate_settings_are_saved_and_restored_on_reopen(qtbot, plates):
    first, second, third = (str(path) for path in plates)
    wizard = _wizard(qtbot, f"{first}; {second}; {third}")
    qtbot.waitUntil(lambda: wizard.plate_table.rowCount() == 3 and bool(wizard._fields), timeout=15000)
    _choose(wizard, second)
    _type(wizard, OTHER)
    _apply_button(wizard, third).click()
    qtbot.waitUntil(lambda: _read(wizard) == {"plate1", "plate2", "plate3"} and wizard.next.isEnabled(),
                    timeout=15000)
    settings = wizard.configuration()
    assert settings["plate_regex"] == {second: OTHER, third: OTHER}
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={**settings, "source_scope": "folder"})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and view._normalization_ready and not view._index_jobs.is_busy(),
                    timeout=15000)
    assert {field.plate for field in view._fields} == {"plate1", "plate2", "plate3"}
    view._set_line_collapsed("row", 0, True)
    path = plates[0] / "results" / "overview_view.json"
    qtbot.waitUntil(lambda: path.is_file() and "plate_regex" in path.read_text(), timeout=10000)
    assert json.loads(path.read_text())["options"]["plate_regex"] == {second: OTHER, third: OTHER}
    again = _wizard(qtbot, first)
    qtbot.waitUntil(lambda: _read(again) == {"plate1", "plate2", "plate3"}, timeout=15000)
    assert again.regex.text() == PATTERN
    _choose(again, third)
    assert again.regex.text() == OTHER and again.editing_plate.text() == "Editing plate: plate3"
    sections = _sections(again)
    assert _texts(sections[third], "OverviewWizardPlatePreviewRegex") == ["Filename regex: " + OTHER]
    fields, info = browser._overview_wizard_probe(first, PATTERN, "auto", threading.Event(), "folder")
    assert info["plate_regex"] == {second: OTHER, third: OTHER}
