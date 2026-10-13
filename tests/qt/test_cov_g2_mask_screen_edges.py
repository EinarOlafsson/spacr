"""Mask Generation's organize button and OPS switch at their edges."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QTabWidget, QWidget  # noqa: E402

from spacr.qt.screens import mask as ms  # noqa: E402


def _host(qtbot, **attrs):
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.app_key = ms.HOST_KEY
    for name, value in attrs.items():
        setattr(screen, name, value)
    return screen


def test_the_organize_button_needs_the_host_a_src_field_and_its_section(qtbot):
    other = QWidget()
    qtbot.addWidget(other)
    assert ms._install_organize_button(other) is None
    widget = QWidget()
    qtbot.addWidget(widget)
    screen = _host(qtbot, _settings_model=SimpleNamespace(_widgets={}))
    assert ms._install_organize_button(screen) is None
    screen._settings_model = SimpleNamespace(_widgets={"src": widget})
    screen._settings_sections = []
    assert ms._install_organize_button(screen) is None


def test_an_unreadable_form_opens_the_organizer_without_a_source(qtbot,
                                                                 monkeypatch):
    import spacr.qt.widgets.organize_for_measure as ofm

    opened = []
    monkeypatch.setattr(ofm, "_open_and_organize",
                        lambda screen, mode, source, done: opened.append(source))

    def broken():
        raise RuntimeError("form not built")

    screen = _host(qtbot, _settings_model=SimpleNamespace(collect=broken))
    ms._open_organize(screen)
    assert opened == [""]


def test_organizing_for_another_layout_leaves_the_form(qtbot):
    screen = _host(qtbot)
    assert ms._on_organized(screen, None, object(), None, "measure") is False


class _Page(QWidget):
    def __init__(self, broken=False):
        super().__init__()
        self.broken = broken
        self.hidden_by_us = False

    def hide(self):
        if self.broken:
            raise RuntimeError("deleted")
        self.hidden_by_us = True
        super().hide()


def _ops(qtbot, monkeypatch, hide_result=False):
    screen = _host(qtbot)
    switch = QCheckBox()
    qtbot.addWidget(switch)
    monkeypatch.setattr(ms, "hide_as_page", lambda page, screen: hide_result)
    return ms._OpsPage(screen, switch), screen, switch


def test_closing_with_no_page_does_nothing_and_a_loose_page_is_hidden(
        qtbot, monkeypatch):
    ops, _screen, _switch = _ops(qtbot, monkeypatch)
    ops.opener = SimpleNamespace(window=None)
    ops.set_shown(False)
    page = _Page()
    qtbot.addWidget(page)
    ops.page = page
    ops.set_shown(False)
    assert page.hidden_by_us and ops.page is None
    broken = _Page(broken=True)
    qtbot.addWidget(broken)
    ops.page = broken
    ops.set_shown(False)
    assert ops.page is None


def test_the_strip_is_followed_only_when_it_can_be(qtbot, monkeypatch):
    ops, screen, _switch = _ops(qtbot, monkeypatch)
    ops._watch_the_strip()
    assert ops._watching is False
    screen._fold_pages = SimpleNamespace(tabCloseRequested=SimpleNamespace(
        connect=lambda slot: (_ for _ in ()).throw(RuntimeError("gone"))))
    ops._watch_the_strip()
    assert ops._watching is False


def test_a_closed_tab_only_resets_the_switch_for_our_page(qtbot, monkeypatch):
    ops, screen, switch = _ops(qtbot, monkeypatch)
    ops._page_closed(0)
    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    mine, other = QWidget(), QWidget()
    tabs.addTab(other, "other")
    tabs.addTab(mine, "ops")
    screen._fold_pages = tabs
    ops.page = mine
    switch.setChecked(True)
    ops._page_closed(0)
    assert ops.page is mine and switch.isChecked()

    class _Gone:
        def indexOf(self, page):
            raise RuntimeError("deleted")

    screen._fold_pages = _Gone()
    ops._page_closed(5)
    assert ops.page is None and not switch.isChecked()


def test_restating_the_shown_state_leaves_the_switch_alone(qtbot, monkeypatch):
    ops, _screen, switch = _ops(qtbot, monkeypatch)
    ops._restate(False)
    ops.switch = None
    ops._restate(True)
    assert not switch.isChecked()


def test_the_ops_switch_installs_once_and_only_on_the_host(qtbot):
    other = QWidget()
    qtbot.addWidget(other)
    switch = QCheckBox()
    qtbot.addWidget(switch)
    assert ms.install_ops_switch(other, switch) is None
    screen = _host(qtbot)
    first = ms.install_ops_switch(screen, switch)
    assert ms.install_ops_switch(screen, switch) is first


def test_folds_survive_a_broken_organize_button_and_an_empty_strip(qtbot,
                                                                   monkeypatch):
    def broken(screen):
        raise RuntimeError("no form")

    class _Folds:
        def __init__(self, *a, **k):
            pass

        def mount(self):
            return True

        def build_strip(self, header):
            return None

    monkeypatch.setattr(ms, "install_example_data_button", lambda screen: None)
    monkeypatch.setattr(ms, "_install_organize_button", broken)
    monkeypatch.setattr(ms, "CategoryFoldSet", _Folds)
    screen = _host(qtbot, _header=SimpleNamespace(add_trailing=lambda w: None))
    assert ms.install_folds(screen) is None
