"""Mask Generation: the OPS page switch, the example-plate button and the folds.

The OPS switch must follow the page it opens -- a page closed by its own
cross puts the switch back, a page that is already gone costs nothing -- and
``install_folds`` must survive every part of itself that fails, each failure
costing only its own feature.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QCheckBox, QLineEdit, QPushButton,  # noqa: E402
                               QTabWidget, QWidget)

from spacr.qt.screens import mask as ms                          # noqa: E402
from spacr.qt.widgets.section import Section                      # noqa: E402

pytestmark = pytest.mark.qt

LOGGER = "spacr.qt.screens.mask"


class _Page(QWidget):
    """A page that records being hidden, or raises like a deleted one."""

    def __init__(self, broken=False):
        super().__init__()
        self.broken = broken
        self.hidden_by_us = False

    def hide(self):
        if self.broken:
            raise RuntimeError("deleted")
        self.hidden_by_us = True
        super().hide()


class _Screen:
    """The attributes the module asks a screen for, and nothing else."""

    def __init__(self, **attrs):
        self.applied = []
        self.__dict__.update(attrs)

    def apply_settings_dict(self, settings):
        self.applied.append(dict(settings))
        return len(settings)


class _Model:
    def __init__(self, widgets=None):
        self._widgets = dict(widgets or {})
        self._defaults = {}


class _Console:
    def __init__(self):
        self.text = []

    def append_stdout(self, text):
        self.text.append(text)


class _Header(QWidget):
    """A masthead that records what was hung on it."""

    def __init__(self):
        super().__init__()
        self.trailing = []

    def add_trailing(self, widget):
        self.trailing.append(widget)
        widget.setParent(self)


def _host(qtbot, **attrs):
    screen = QWidget()
    qtbot.addWidget(screen)
    screen.app_key = ms.HOST_KEY
    for name, value in attrs.items():
        setattr(screen, name, value)
    return screen


def _ops(qtbot):
    screen = _host(qtbot)
    switch = QCheckBox()
    qtbot.addWidget(switch)
    return ms._OpsPage(screen, switch), screen, switch


def _page(qtbot, broken=False):
    page = _Page(broken=broken)
    qtbot.addWidget(page)
    return page


def _src_form(qtbot):
    """A settings form whose third section is the one holding ``src``."""
    src = QLineEdit()
    holder = Section("Source")
    holder.add_prose(src)
    elsewhere = Section("Somewhere Else")
    plain = QWidget()
    for widget in (holder, elsewhere, plain):
        qtbot.addWidget(widget)
    return src, holder, (plain, elsewhere, holder)


def test_the_ops_switch_follows_the_page_it_opened(qtbot, subtests):
    """Closing, following the strip, a closed tab and restating the switch."""
    with subtests.test(msg="closing with no page and no window does nothing"):
        ops, _screen, switch = _ops(qtbot)
        ops.opener = SimpleNamespace(window=None)
        ops.set_shown(False)
        assert ops.page is None
        assert switch.isChecked() is False

    with subtests.test(msg="a loose page is hidden and forgotten"):
        page = _page(qtbot)
        ops.page = page
        ops.set_shown(False)
        assert (page.hidden_by_us, ops.page) == (True, None)

    with subtests.test(msg="the opener's window is closed when no page is"):
        window = _page(qtbot)
        ops.opener = SimpleNamespace(window=window)
        ops.set_shown(False)
        assert (window.hidden_by_us, ops.page) == (True, None)

    with subtests.test(msg="a page Qt already deleted is dropped quietly"):
        ops.opener = SimpleNamespace(window=None)
        ops.page = _page(qtbot, broken=True)
        ops.set_shown(False)
        assert ops.page is None

    with subtests.test(msg="no strip, or a strip that refuses, is not followed"):
        ops, screen, _switch = _ops(qtbot)
        ops._watch_the_strip()
        assert ops._watching is False
        screen._fold_pages = SimpleNamespace(tabCloseRequested=SimpleNamespace(
            connect=lambda slot: (_ for _ in ()).throw(RuntimeError("gone"))))
        ops._watch_the_strip()
        assert ops._watching is False

    with subtests.test(msg="a real strip is followed once"):
        tabs = QTabWidget()
        qtbot.addWidget(tabs)
        connected = []
        screen._fold_pages = SimpleNamespace(tabCloseRequested=SimpleNamespace(
            connect=connected.append))
        ops._watch_the_strip()
        ops._watch_the_strip()
        assert ops._watching is True
        assert connected == [ops._page_closed]

    ops, screen, switch = _ops(qtbot)
    with subtests.test(msg="a closed tab with no strip changes nothing"):
        ops.page = QWidget()
        switch.setChecked(True)
        ops._page_closed(0)
        assert switch.isChecked() is True

    tabs = QTabWidget()
    qtbot.addWidget(tabs)
    mine, other = QWidget(), QWidget()
    tabs.addTab(other, "other")
    tabs.addTab(mine, "ops")
    screen._fold_pages = tabs
    ops.page = mine
    with subtests.test(msg="another page's tab leaves our switch on"):
        ops._page_closed(0)
        assert ops.page is mine
        assert switch.isChecked() is True

    with subtests.test(msg="our own tab puts the switch back"):
        toggled = []
        switch.toggled.connect(toggled.append)
        ops._page_closed(1)
        assert ops.page is None
        assert switch.isChecked() is False
        assert toggled == []  # moved with its signals blocked

    with subtests.test(msg="a closed tab with no page of ours does nothing"):
        switch.setChecked(True)
        ops._page_closed(1)
        assert switch.isChecked() is True

    with subtests.test(msg="a deleted strip counts as our page closing"):
        class _Gone:
            def indexOf(self, page):
                raise RuntimeError("deleted")

        screen._fold_pages = _Gone()
        ops.page = mine
        ops._page_closed(5)
        assert ops.page is None
        assert switch.isChecked() is False

    with subtests.test(msg="restating the shown state, or with no switch"):
        ops._restate(False)
        assert switch.isChecked() is False
        ops.switch = None
        ops._restate(True)
        assert switch.isChecked() is False


def test_the_example_plate_button_is_installed_once_above_src(
        qtbot, monkeypatch, tmp_path, subtests):
    """Only Mask Generation with a ``src`` control gets it, and it loads."""
    from spacr.qt import synthetic

    src, holder, sections = _src_form(qtbot)
    for case_id, attrs in [
            ("another screen", {"app_key": "measure",
                                "_settings_model": _Model({"src": src})}),
            ("no src control", {"app_key": "mask",
                                "_settings_model": _Model()}),
            ("no settings model", {"app_key": "mask"})]:
        with subtests.test(msg=f"no button: {case_id}"):
            guest = _Screen(_settings_sections=sections, **attrs)
            assert ms.install_example_data_button(guest) is None
            assert getattr(guest, "_example_images_button", None) is None

    console = _Console()
    screen = _Screen(app_key="mask", _settings_model=_Model({"src": src}),
                     _settings_sections=sections, _console=console)
    button = ms.install_example_data_button(screen)
    with subtests.test(msg="installed in the src section, once"):
        assert isinstance(button, QPushButton)
        assert button.text() == ms.EXAMPLE_BUTTON_TEXT
        assert button.toolTip() == ms.EXAMPLE_BUTTON_TOOLTIP
        assert holder.isAncestorOf(button)
        assert screen._example_images_button is button
        assert ms.install_example_data_button(screen) is button

    with subtests.test(msg="pressing it writes a plate and fills the form"):
        dst = str(tmp_path / "plate")
        layout = SimpleNamespace(image_files=[str(tmp_path / "a.tif")],
                                 src=dst,
                                 notes={"channels": (0,), "n_fields": 1})
        monkeypatch.setattr(synthetic, "generate_mask_demo",
                            lambda folder: layout)
        monkeypatch.setattr(synthetic, "demo_settings",
                            lambda app, source: {"src": source})
        monkeypatch.setattr(ms, "example_plate_folder", lambda: dst)
        button.click()
        assert screen.applied == [{"src": dst}]
        assert console.text == [
            f"Example plate ready: 1 image(s) in {dst} (1 written now, 0 "
            f"already there) — 1 field(s), channels 0.\n"
            f"src is now {dst}, and 1 setting(s) on this form were filled "
            f"from the plate: src.\n"]


def test_install_folds_survives_each_part_that_fails(qtbot, qt_theme_applied,
                                                    monkeypatch, caplog,
                                                    subtests):
    """A broken button or an unmountable fold costs only itself."""
    from spacr.qt.screens.app_screen import AppScreen

    with subtests.test(msg="another screen is left alone"):
        guest = _Screen(app_key="measure")
        assert ms.install_folds(guest) is None
        assert ms.fold_set(guest) is None

    with subtests.test(msg="a screen with no settings form carries no strip"):
        header = _Header()
        qtbot.addWidget(header)
        bare = _Screen(app_key="mask", _header=header)
        assert ms.install_folds(bare) is None
        assert header.trailing == []
        assert ms.fold_set(bare) is None

    with subtests.test(msg="a real screen mounts its folds once"):
        real = AppScreen(app_key="mask")
        qtbot.addWidget(real)
        assert ms.install_folds(real) is None
        folds = ms.fold_set(real)
        assert folds is not None
        assert ms.install_folds(real) is None
        assert ms.fold_set(real) is folds

    def refuse(screen):
        raise RuntimeError("the src section moved")

    with subtests.test(msg="buttons that raise are logged, not raised"):
        monkeypatch.setattr(ms, "install_example_data_button", refuse)
        monkeypatch.setattr(ms, "_install_organize_button", refuse)
        caplog.clear()
        host = _host(qtbot,
                     _header=SimpleNamespace(add_trailing=lambda w: None))
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert ms.install_folds(host) is None
        messages = [r.getMessage() for r in caplog.records
                    if r.name == LOGGER]
        assert messages == [
            "Could not install the example-plate button on mask",
            "Could not install the organize button on mask",
            "Could not mount Time controls for mask"]
        assert ms.fold_set(host) is None

    with subtests.test(msg="a refused example button keeps the switches"):
        caplog.clear()
        screen = AppScreen(app_key="mask")
        qtbot.addWidget(screen)
        with caplog.at_level(logging.DEBUG, logger=LOGGER):
            assert ms.install_folds(screen) is None
        assert ms.fold_set(screen) is not None
        assert "Could not install the example-plate button on mask" in [
            r.getMessage() for r in caplog.records]
