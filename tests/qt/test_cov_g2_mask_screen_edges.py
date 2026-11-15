"""Mask Generation's organize button and OPS switch at their edges."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QWidget  # noqa: E402

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


def test_the_ops_switch_installs_once_and_only_on_the_host(qtbot):
    other = QWidget()
    qtbot.addWidget(other)
    switch = QCheckBox()
    qtbot.addWidget(switch)
    assert ms.install_ops_switch(other, switch) is None
    screen = _host(qtbot)
    first = ms.install_ops_switch(screen, switch)
    assert ms.install_ops_switch(screen, switch) is first
