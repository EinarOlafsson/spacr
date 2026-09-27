"""Revealing one setting from a lookup, and the strip's bookkeeping when a
part of the screen goes away under it.

``SettingsSearchBar.reveal`` is what a Help-search result lands on: it
must return ``False`` for a setting the module does not render (or whose
waiting heading will not open), show the row with every other category shut
even when one category cannot be shut, outline the row and then put the
field's own style back. Around it: a narrow strip puts a new trailing
control under the box, a strip with no model hides nothing, and a section,
table or row whose C++ half is gone is skipped rather than raised on.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from spacr.qt import settings_search as ss  # noqa: E402
from spacr.qt.settings_search import SettingsSearchBar  # noqa: E402

pytestmark = pytest.mark.qt

APP_KEY = "reveal_probe"


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part is gone")


class _Section(QWidget):
    def __init__(self, parent=None, *, refuses=False):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self.expanded = True
        self._refuses = refuses

    def is_expanded(self):
        return self.expanded

    def set_expanded(self, on):
        if self._refuses and not on:
            raise RuntimeError("this section will not collapse")
        self.expanded = bool(on)

    def add_row(self, label, field):
        self._form.addRow(QLabel(label), field)


class _Plain(QWidget):
    """A section with a form and no collapse protocol."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)


class _Model:
    def __init__(self, widgets, hidden_by_run=None):
        self._widgets = dict(widgets)
        self._keys = list(widgets)
        if hidden_by_run is not None:
            self.keys_hidden_by_the_run = hidden_by_run

    def keys_matching(self, query):
        return [k for k in self._keys if query in k]

    def modified_keys(self):
        return []

    def essential_keys(self):
        return list(self._keys)


@pytest.fixture(autouse=True)
def _forget_the_level():
    ss.forget_disclosure(APP_KEY)
    yield
    ss.forget_disclosure(APP_KEY)


@pytest.fixture
def screen(qapp, qtbot):
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.app_key = APP_KEY
    first = _Section(screen)
    second = _Section(screen)
    plain = _Plain(screen)
    fields = {}
    for section, key in ((first, "cell_diameter"), (second, "plot_dpi")):
        field = QLineEdit(section)
        field.setStyleSheet("QLineEdit { color: red; }")
        section.add_row(key, field)
        fields[key] = field
    screen.fields = fields
    screen.first, screen.second = first, second
    screen._settings_sections = [first, second, plain]
    screen._settings_model = _Model(fields)
    return screen


def test_a_setting_the_module_does_not_render_is_not_revealed(screen):
    bar = SettingsSearchBar(screen)

    assert bar.reveal("no_such_setting") is False
    assert bar.revealed_key() == ""


def test_a_setting_under_a_heading_that_will_not_open_is_not_revealed(
        screen):
    bar = SettingsSearchBar(screen)
    bar._index["waiting_key"] = (screen.first, None)

    assert bar.reveal("waiting_key") is False

    opened = []
    screen._open_the_heading_of = opened.append
    assert bar.reveal("waiting_key") is False
    assert opened == ["waiting_key"]


def test_a_heading_that_opens_on_request_reveals_its_row(screen):
    bar = SettingsSearchBar(screen)
    field = screen.fields["cell_diameter"]
    bar._index["cell_diameter"] = (screen.first, None)

    def _open(key):
        bar._index[key] = (screen.first, field)

    screen._open_the_heading_of = _open

    assert bar.reveal("cell_diameter") is True
    assert bar.revealed_key() == "cell_diameter"


def test_a_revealed_row_is_shown_even_when_a_category_will_not_shut(
        screen, qtbot, monkeypatch):
    monkeypatch.setattr(ss, "_MARK_MS", 0)
    screen.show()
    bar = SettingsSearchBar(screen)
    field = screen.fields["cell_diameter"]
    screen.second._refuses = True

    assert bar.reveal("cell_diameter") is True

    assert screen.first.expanded is True
    assert field.property("spacrRevealed") is True
    assert "palette(highlight)" in field.styleSheet()
    qtbot.waitUntil(lambda: field.property("spacrRevealed") is False)
    assert field.styleSheet() == "QLineEdit { color: red; }"
    qtbot.waitUntil(field.hasFocus)


def test_a_row_that_went_away_before_it_could_be_scrolled_to_is_let_go(
        screen):
    bar = SettingsSearchBar(screen)
    field = QLineEdit()
    shiboken6.delete(field)

    assert bar._scroll_to(field) is None


def test_a_narrow_strip_puts_a_new_control_under_the_box(screen):
    bar = SettingsSearchBar(screen)
    bar.resize(40, bar.height())
    bar._fit(40)
    extra = QLabel("Live preview")

    bar.add_trailing_widget(extra)

    assert bar._compact is True
    assert bar._wrap_row.indexOf(extra) >= 0


def test_a_strip_with_no_model_hides_nothing(qapp, qtbot):
    bare = QWidget()
    qtbot.addWidget(bare)
    bare.app_key = APP_KEY

    assert SettingsSearchBar(bare).keys_it_hides() == set()


def test_a_run_whose_hidden_list_cannot_be_read_hides_nothing(screen):
    screen._settings_model = _Model(screen.fields, hidden_by_run=_boom)
    bar = SettingsSearchBar(screen)
    bar._disclosure.setChecked(True)

    assert bar.keys_it_hides() == set()


def test_a_heading_whose_parent_is_gone_keeps_its_own_count(screen,
                                                            monkeypatch):
    bar = SettingsSearchBar(screen)
    counts = {id(screen.first): 2, id(screen.second): 1}

    monkeypatch.setattr(ss, "_logical_parent", _boom)
    assert bar._counting_the_sub_headings(counts) == counts

    calls = []

    def _once_then_gone(node):
        calls.append(node)
        if len(calls) % 2 == 1:
            return screen
        raise RuntimeError("the umbrella went away")

    monkeypatch.setattr(ss, "_logical_parent", _once_then_gone)
    assert bar._counting_the_sub_headings(counts) == counts


def test_a_table_whose_section_is_gone_answers_for_nothing(screen):
    screen._object_grid = type("_Grid", (), {"parentWidget": _boom})()
    screen._object_grid_binding = object()
    bar = SettingsSearchBar(screen)

    assert bar._grid_section() == (None, frozenset())
