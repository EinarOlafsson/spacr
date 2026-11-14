"""Filtering a settings form, and the four ways the form can refuse.

The strip's promise is that the query, the Modified filter and the
disclosure level are one path, so they can never disagree about what is on
screen. Everything below that promise is degradation: a model whose search
raises, a section with no form in it, a Qt too old for ``setRowVisible``, a
screen that is not a settings screen at all. Each of those has to leave a
usable panel rather than a blank one -- "the filter excluded that setting"
and "the panel will not draw" look identical to a user for about a second,
and then only one of them recovers.

The screens here are real widgets carrying the same four attributes a real
settings screen exposes, with real ``QFormLayout`` rows in them, so the
row-visibility calls are the Qt ones and the index is really built from a
rendered form.
"""

import pytest
from PySide6.QtWidgets import QFormLayout, QLabel, QLineEdit, QVBoxLayout, QWidget

from spacr.qt import settings_search as ss
from spacr.qt.settings_search import (ALL, ESSENTIALS, SettingsSearchBar,
                                      _StackWatcher, disclosure_for,
                                      forget_disclosure, remember_disclosure)


# ---------------------------------------------------------------------------
# stand-ins with the real screen's shape
# ---------------------------------------------------------------------------

class _Section(QWidget):
    """A settings section: a form, and the collapse protocol the strip uses."""

    def __init__(self, parent=None, *, with_form=True, collapsible=True):
        super().__init__(parent)
        self._expanded = True
        self._collapsible = collapsible
        if with_form:
            self._form = QFormLayout(self)
        else:
            QVBoxLayout(self)

    def is_expanded(self):
        return self._expanded

    def set_expanded(self, on):
        self._expanded = bool(on)

    def add_row(self, label, field):
        self._form.addRow(QLabel(label), field)


class _Model:
    """The four methods the strip asks a settings model for."""

    def __init__(self, widgets, *, matching=None, modified=(), essential=(),
                 raises=()):
        self._widgets = dict(widgets)
        self._matching = matching
        self._modified = list(modified)
        self._essential = list(essential)
        self._raises = set(raises)

    def keys_matching(self, query):
        if "keys_matching" in self._raises:
            raise RuntimeError("the search index is not built")
        if self._matching is not None:
            return list(self._matching)
        return [k for k in self._widgets if query.lower() in k.lower()]

    def modified_keys(self):
        if "modified_keys" in self._raises:
            raise RuntimeError("no defaults to compare against")
        return list(self._modified)

    def essential_keys(self):
        if "essential_keys" in self._raises:
            raise RuntimeError("this module declares no essentials")
        return list(self._essential)


@pytest.fixture
def form_screen(qapp):
    """A screen with two sections, four settings, and a model over them."""
    screen = QWidget()
    screen.app_key = "w2_2_probe"
    first = _Section(screen)
    second = _Section(screen)
    fields = {}
    for section, keys in ((first, ("cell_diameter", "cell_cellprob_threshold")),
                          (second, ("plot_dpi", "verbose"))):
        for key in keys:
            field = QLineEdit(section)
            section.add_row(key, field)
            fields[key] = field
    screen._settings_sections = [first, second]
    screen._settings_model = _Model(fields, modified=["plot_dpi"],
                                    essential=["cell_diameter", "plot_dpi"])
    screen._sections = (first, second)
    return screen


@pytest.fixture(autouse=True)
def forget_this_modules_choice():
    """The disclosure level is persisted, so clear it around each test."""
    forget_disclosure("w2_2_probe")
    yield
    forget_disclosure("w2_2_probe")


# ---------------------------------------------------------------------------
# the remembered disclosure level
# ---------------------------------------------------------------------------

def test_a_module_starts_on_essentials_and_remembers_being_told_otherwise(
        qapp):
    """A user meets a few settings at a time until they say otherwise."""
    assert disclosure_for("w2_2_probe") == ESSENTIALS

    remember_disclosure("w2_2_probe", ALL)
    assert disclosure_for("w2_2_probe") == ALL

    forget_disclosure("w2_2_probe")
    assert disclosure_for("w2_2_probe") == ESSENTIALS


# ---------------------------------------------------------------------------
# filtering a real form
# ---------------------------------------------------------------------------


def test_a_trailing_control_joins_the_strip_rather_than_a_second_one(
        qapp, form_screen):
    """The seam other modules use to put a settings-scoped control here."""
    bar = SettingsSearchBar(form_screen)
    extra = QLabel("Live preview")
    bar.add_trailing_widget(extra)
    assert extra.parent() is bar


# ---------------------------------------------------------------------------
# following the stack
# ---------------------------------------------------------------------------


def test_a_watcher_ignores_a_stack_it_cannot_read(qapp):
    """A window being torn down is not a screen to install into."""
    class _Hostile(QWidget):
        @property
        def _stack(self):
            raise RuntimeError("the window is going away")

    watcher = _StackWatcher(QWidget())
    watcher._window = _Hostile()
    assert watcher.install_current() is None


def test_an_empty_stack_installs_nothing(qapp):
    """A stack showing no widget has no settings form on it."""
    class _Stack(QWidget):
        def currentWidget(self):
            return None

    window = QWidget()
    window._stack = _Stack()
    watcher = _StackWatcher(window)
    assert watcher.install_current() is None
    watcher.on_current_changed(0)


def test_the_search_box_reports_what_was_typed_into_it(qapp, form_screen):
    """`query` is how a caller reads the box rather than reaching for the
    widget."""
    bar = SettingsSearchBar(form_screen)
    assert bar.query() == ""
    bar.set_query("  cell ")
    assert bar.query() == "  cell "


def test_a_partial_view_counts_what_it_hid(qapp, form_screen):
    """The line under the controls says how many of how many are showing."""
    bar = SettingsSearchBar(form_screen)
    bar.set_level(ALL)
    bar.set_modified_only(True)
    text = bar.count_text()
    assert "1" in text and "4" in text
    assert "modified only" in text


# ---------------------------------------------------------------------------
# the strip's own styling
# ---------------------------------------------------------------------------

def test_the_strip_paints_nothing_behind_itself(qapp):
    """It is type and controls sitting on the page, not a card.

    A painted strip composites the pane's own surface onto whatever is behind
    it, which is the "the container is not subject to the opacity setting"
    report.
    """
    from spacr.qt.theme import palette_for

    palette = dict(palette_for("dark"), theme="dark", font_scale=1.0)
    qss = ss._bar_qss(palette, 1.0)
    assert f"QWidget#{ss.PANE_NAME}" in qss
    assert f"QWidget#{ss.BAR_NAME}" in qss
    assert "background: transparent" in qss
    assert f"QLineEdit#{ss.INPUT_NAME}" in qss
