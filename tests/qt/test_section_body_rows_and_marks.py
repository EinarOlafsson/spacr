"""Settings categories: the body that detaches, the rows, and the module mark.

:class:`spacr.qt.widgets.section.Section` is one settings category.  Its body
detaches while hidden and must come back intact; its rows can wrap a label
in a right-aligning host; and a category folded in from another module shows
that module's mark (:func:`~spacr.qt.widgets.section.module_mark`).  Two
readers of a section are walked here too: the settings panel asking whether a
heading still shows anything, and the Mask Generation screen placing its
example button and fold marks on sections.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent                                     # noqa: E402
from PySide6.QtGui import QIcon                                       # noqa: E402
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QHBoxLayout,   # noqa: E402
                               QLabel, QLineEdit, QVBoxLayout, QWidget)

from spacr.qt.widgets.section import Section, module_mark             # noqa: E402

pytestmark = pytest.mark.qt


def test_a_detached_body_comes_back_with_its_live_fields(qtbot, monkeypatch,
                                                         subtests):
    """``_detach_body_while_hidden`` / ``_attach_body`` / ``_holds``: a cut
    destroy link, nested detached categories and a missing ambient module."""
    with subtests.test(msg="the destroy link is already gone"):
        page = QWidget()
        qtbot.addWidget(page)
        shut = Section("Shut")
        shut.add_row("Hidden", QLabel("field"))
        QVBoxLayout(page).addWidget(shut)
        assert shut._detach_body_while_hidden() > 0
        assert shut._body_is_detached()
        shut.destroyed.disconnect(shut._body.deleteLater)
        assert shut._attach_body() is True
        assert shut._body.parentWidget() is shut
        assert not shut._body_is_detached()

    with subtests.test(msg="nested detached categories"):
        outer, inner, leaf = Section("Outer"), Section("Inner"), Section("Leaf")
        qtbot.addWidget(outer)
        inner.add_row("", leaf)
        outer.add_row("", inner)
        field = QLabel("Original value")
        leaf.add_row("Value", field)
        inner._detach_body_while_hidden()
        outer._detach_body_while_hidden()
        found = outer._nested_sections()
        assert len(found) == 2 and set(found) == {inner, leaf}
        assert outer._holds(field)
        assert outer._attach_body()
        assert inner._attach_body()
        assert field.text() == "Original value"
        assert outer.eventFilter(outer._header,
                                 QEvent(QEvent.Type.Enter)) is False

    for expanded in (False, True):
        with subtests.test(msg=f"no ambient module, expanded={expanded}"):
            section = Section("Editable category")
            qtbot.addWidget(section)
            label = QLabel("Preserved field", section._body)
            section.add_row("Name", label)
            section._expanded = expanded
            monkeypatch.setitem(sys.modules, "spacr.qt.widgets.ambient",
                                SimpleNamespace())
            assert section._category_ripple() is None
            assert section._holds(label)
            assert label.text() == "Preserved field"
            monkeypatch.undo()


def test_a_wrapped_row_label_is_hosted_not_copied(qapp, subtests):
    """``add_row(..., wrap_label=True)`` puts a string in a new QLabel and a
    caller's QLabel in the host as itself."""
    for case, make_caption in [("a string", lambda holder: "Cell diameter"),
                               ("a live label",
                                lambda holder: QLabel("Cell diameter", holder))]:
        with subtests.test(msg=case):
            holder = QWidget()
            section = Section("Embedding & Clustering", holder)
            field = QCheckBox("on", holder)
            caption = make_caption(holder)
            section.add_row(caption, field, wrap_label=True)
            host = section._form.itemAt(0, QFormLayout.LabelRole).widget()
            assert host.objectName() == "SettingLabelWithInfo"
            labels = host.findChildren(QLabel)
            assert [w.text() for w in labels] == ["Cell diameter"]
            if isinstance(caption, QLabel):
                assert labels == [caption]
            assert section._row_widgets == [(caption, field)]


def test_a_category_shows_the_mark_of_its_source_module(qtbot, monkeypatch,
                                                        subtests):
    """``module_mark`` and ``set_source_app``: a real icon is drawn once in
    one header layout; no icon, a null icon or a raising iconset draw none."""
    from spacr.qt import iconset

    with subtests.test(msg="no mark, no header layout"):
        section = Section("A heading")
        qtbot.addWidget(section)
        assert section.source_app() == ""
        assert section.source_mark() is None
        assert section._header.layout() is None

    with subtests.test(msg="a module's mark, built once"):
        section = Section("Object measurements")
        qtbot.addWidget(section)
        assert section.set_source_app("measure", "Measure") is True
        assert section.source_app() == "measure"
        badge = section.source_mark()
        assert badge.accessibleName() == "Measure"
        assert not badge.pixmap().isNull()
        first = section._header.layout()
        assert first is not None
        assert section._source_row() is first
        assert section.source_mark() is badge

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("icon resources are not installed")

    for case, attribute, replacement, key in [
            ("an unregistered key", None, None, "not_a_registered_app_at_all"),
            ("an icon that decodes to nothing", "app_icon",
             lambda _key: QIcon(), "measure"),
            ("an iconset that raises", "bundled_icon_path", _refuse,
             "measure"),
    ]:
        with subtests.test(msg=case):
            if attribute:
                monkeypatch.setattr(iconset, attribute, replacement)
            assert module_mark(key) is None
            monkeypatch.undo()


def test_the_settings_panel_reads_what_a_heading_still_shows(qtbot, subtests):
    """``settings_model.section_shows_anything`` and ``_slot_headings`` walk
    a section's form: hidden rows, child headings, layout-only rows, and a
    body that is not a form."""
    from spacr.qt.screens import settings_model as sm

    with subtests.test(msg="hidden own rows, visible child heading"):
        parent = Section("Advanced settings")
        qtbot.addWidget(parent)
        parent.add_row("Diameter", QLineEdit())
        child = Section("Cell")
        child.add_row("Minimum size", QLineEdit())
        parent.add_prose(child)
        for row in range(parent._form.rowCount()):
            parent._form.setRowVisible(row, False)
        assert sm.section_shows_anything(child) is True
        assert sm.section_shows_anything(parent) is True

    for case, extra_layout_row in [("every row hidden, no children", False),
                                   ("a row holding only a layout", True)]:
        with subtests.test(msg=case):
            section = Section("Pathogen")
            qtbot.addWidget(section)
            section.add_row("Diameter", QLineEdit())
            if extra_layout_row:
                section._form.addRow(QLabel("composite"), QHBoxLayout())
            section._form.setRowVisible(0, False)
            assert sm.section_shows_anything(section) is False

    with subtests.test(msg="a heading whose body is not a form"):
        holder = QWidget()
        qtbot.addWidget(holder)
        QVBoxLayout(holder)
        formless = Section("No form", holder)
        formless._form = QVBoxLayout()
        widgets = sm.SettingsWidgets("measure", parent=holder)
        assert widgets._slot_headings() == {}


class _Screen:
    """The attributes the mask module asks a screen for, and nothing else."""

    def __init__(self, **attrs):
        self.applied = []
        self.__dict__.update(attrs)

    def apply_settings_dict(self, settings):
        """Record an applied settings dict."""
        self.applied.append(dict(settings))
        return len(settings)


class _Model:
    """A settings model holding only its widgets."""

    def __init__(self, widgets=None):
        self._widgets = dict(widgets or {})
        self._defaults = {}


def test_the_mask_screen_marks_and_places_onto_sections(qtbot, subtests):
    """``mask.mark_fold_sources`` marks only sections whose module has art;
    ``install_example_data_button`` lands in the section holding ``src``."""
    from spacr.qt.screens import mask as mask_module
    from spacr.qt.screens.map_barcodes import CategoryFoldSet

    with subtests.test(msg="only folds with artwork are marked"):
        titles = ("Time Axes & Tracking (Beta)", "Nothing In Particular")
        marked_section, anonymous_section = (Section(t) for t in titles)
        for section, title in zip((marked_section, anonymous_section), titles):
            section.setProperty("settingsCategorySource", title)
            qtbot.addWidget(section)
        screen = _Screen(_settings_sections=())
        folds = CategoryFoldSet(
            screen, {"timelapse": ("timelapse",), "not_a_module": ()})
        folds.folds["timelapse"].sections = [marked_section]
        folds.folds["not_a_module"].sections = [anonymous_section]
        screen._category_folds = folds
        assert mask_module.mark_fold_sources(screen) == {
            "timelapse": ("Time Axes & Tracking (Beta)",)}
        assert marked_section.source_app() == "timelapse"
        assert anonymous_section.source_mark() is None

    with subtests.test(msg="the example button follows src"):
        src = QLineEdit()
        holder = Section("Source")
        holder.add_prose(src)
        elsewhere = Section("Somewhere Else")
        plain = QWidget()
        for widget in (holder, elsewhere, plain):
            qtbot.addWidget(widget)
        orphan = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                         _settings_sections=(QWidget(), Section("Empty")))
        assert mask_module.install_example_data_button(orphan) is None
        placed = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                         _settings_sections=(plain, elsewhere, holder))
        button = mask_module.install_example_data_button(placed)
        assert holder.isAncestorOf(button)
        assert not elsewhere.isAncestorOf(button)
